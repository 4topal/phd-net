"""8x8 Blok DCT, Harmonik Enerji Oranı (HER) ve Öğrenilebilir Frekans Kapısı (LFG).

Bileşenler:
1. `BlockDCT8x8`: Ortogonal 8x8 2B DCT-II ve ters IDCT-II dönüşümü.
2. `build_dct_band_masks`: Parametrik bant sınırları (`low_max`, `mid_max`, `include_dc`) ile
   düşük (`Omega_low`), orta (`Omega_mid`) ve yüksek (`Omega_high`) frekans alt-bant maskeleri.
3. `HarmonicEnergyRatio`: Blok bazlı ve görüntü bazlı `HER = sum(|mid|) / (sum(|low|) + eps)`
   ile düşük/orta/yüksek bant enerji haritaları.
4. `LearnableFreqGate` (LFG): 8x8 DCT katsayılarını öğrenilebilir frekans kapıları ve HER
   dikkat mekanizmasıyla modüle ederek PHD-Net'in 2. dalı (Harmonik Kol) için `28x28` ve `14x14`
   frekans öznitelik haritaları üretir.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_dct_matrix(block_size: int = 8, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Ortonormal 1B DCT-II dönüşüm matrisi `D` `(block_size, block_size)` oluşturur."""
    u = torch.arange(block_size, dtype=dtype).unsqueeze(1)
    x = torch.arange(block_size, dtype=dtype).unsqueeze(0)
    dct = torch.cos(math.pi * (2.0 * x + 1.0) * u / (2.0 * block_size))
    dct[0, :] *= math.sqrt(1.0 / block_size)
    dct[1:, :] *= math.sqrt(2.0 / block_size)
    return dct


def build_dct_band_masks(
    low_max: int = 3,
    mid_max: int = 8,
    include_dc: bool = False,
    block_size: int = 8,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manhattan frekans indeksi `k = u + v` üzerinden düşük/orta/yüksek 8x8 bant maskeleri üretir.

    Args:
        low_max: Düşük frekans bandı üst sınırı (`u + v <= low_max`).
        mid_max: Orta frekans bandı üst sınırı (`low_max < u + v <= mid_max`).
        include_dc: True ise `(0, 0)` DC bileşeni düşük banda dahil edilir; False ise hariç tutulur.
        block_size: DCT blok boyutu (varsayılan 8 -> `u + v` aralığı `[0, 14]`).
    Returns:
        `(low_mask, mid_mask, high_mask)` — her biri `(block_size, block_size)` boyutlu `{0, 1}` tensör.
    """
    max_freq = 2 * (block_size - 1)
    if not (0 <= low_max < mid_max < max_freq):
        raise ValueError(
            f"Geçersiz bant sınırları: 0 <= low_max ({low_max}) < mid_max ({mid_max}) < {max_freq} olmalı."
        )

    u = torch.arange(block_size, device=device).unsqueeze(1)
    v = torch.arange(block_size, device=device).unsqueeze(0)
    freq_idx = u + v  # (8, 8), 0..14

    low_min = 0 if include_dc else 1
    low_mask = ((freq_idx >= low_min) & (freq_idx <= low_max)).to(dtype)
    mid_mask = ((freq_idx > low_max) & (freq_idx <= mid_max)).to(dtype)
    high_mask = (freq_idx > mid_max).to(dtype)
    return low_mask, mid_mask, high_mask


class BlockDCT8x8(nn.Module):
    """Görüntüyü örtüşmeyen 8x8 bloklara ayırıp ortonormal 2B DCT-II uygular."""

    def __init__(self, block_size: int = 8) -> None:
        super().__init__()
        self.block_size = block_size
        self.register_buffer("dct_mat", _create_dct_matrix(block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, C, H, W)` -> `(B, C, H_b, W_b, 8, 8)` blok DCT katsayıları."""
        bs = self.block_size
        _, _, h, w = x.shape
        pad_h = (bs - (h % bs)) % bs
        pad_w = (bs - (w % bs)) % bs
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        b, c, hp, wp = x.shape
        hb, wb = hp // bs, wp // bs
        blocks = x.view(b, c, hb, bs, wb, bs).permute(0, 1, 2, 4, 3, 5).contiguous()
        dct_mat = self.dct_mat.to(dtype=x.dtype, device=x.device)
        return torch.einsum("ux,bcijxy,vy->bcijuv", dct_mat, blocks, dct_mat)

    def inverse(self, dct_blocks: torch.Tensor) -> torch.Tensor:
        """`(B, C, H_b, W_b, 8, 8)` -> `(B, C, H_b*8, W_b*8)` ters 2B IDCT-II."""
        bs = self.block_size
        b, c, hb, wb, _, _ = dct_blocks.shape
        dct_mat = self.dct_mat.to(dtype=dct_blocks.dtype, device=dct_blocks.device)
        spatial_blocks = torch.einsum("ux,bcijuv,vy->bcijxy", dct_mat, dct_blocks, dct_mat)
        return spatial_blocks.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, hb * bs, wb * bs)


class HarmonicEnergyRatio(nn.Module):
    """8x8 DCT blokları üzerinden Harmonik Enerji Oranı (`HER`) ve alt-bant enerjilerini hesaplar.

    Tanım:
        `HER = sum_{(u,v) in Omega_mid} |DCT(u,v)| / (sum_{(u,v) in Omega_low} |DCT(u,v)| + eps)`
    """

    def __init__(
        self,
        low_max: int = 3,
        mid_max: int = 8,
        include_dc: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.low_max = low_max
        self.mid_max = mid_max
        self.include_dc = include_dc
        self.eps = eps
        self.block_dct = BlockDCT8x8(block_size=8)

        low_m, mid_m, high_m = build_dct_band_masks(
            low_max=low_max, mid_max=mid_max, include_dc=include_dc, block_size=8
        )
        self.register_buffer("low_mask", low_m, persistent=False)
        self.register_buffer("mid_mask", mid_m, persistent=False)
        self.register_buffer("high_mask", high_m, persistent=False)

    def compute_from_dct_blocks(self, dct_blocks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Önceden hesaplanmış `(B, C, H_b, W_b, 8, 8)` DCT bloklarından HER istatistiklerini üretir."""
        abs_dct = dct_blocks.abs()
        low_m = self.low_mask.to(dtype=abs_dct.dtype, device=abs_dct.device)
        mid_m = self.mid_mask.to(dtype=abs_dct.dtype, device=abs_dct.device)
        high_m = self.high_mask.to(dtype=abs_dct.dtype, device=abs_dct.device)

        low_energy = (abs_dct * low_m).sum(dim=(-2, -1))    # (B, C, H_b, W_b)
        mid_energy = (abs_dct * mid_m).sum(dim=(-2, -1))    # (B, C, H_b, W_b)
        high_energy = (abs_dct * high_m).sum(dim=(-2, -1))  # (B, C, H_b, W_b)

        her_map = mid_energy / (low_energy + self.eps)  # (B, C, H_b, W_b)
        her_channel = mid_energy.sum(dim=(-2, -1)) / (low_energy.sum(dim=(-2, -1)) + self.eps)  # (B, C)
        her_global = mid_energy.sum(dim=(1, 2, 3)) / (low_energy.sum(dim=(1, 2, 3)) + self.eps)  # (B,)
        mean_abs_spectrum = abs_dct.mean(dim=(2, 3))  # (B, C, 8, 8)

        return {
            "her_map": her_map,
            "her_channel": her_channel,
            "her_global": her_global,
            "low_energy": low_energy,
            "mid_energy": mid_energy,
            "high_energy": high_energy,
            "mean_abs_spectrum": mean_abs_spectrum,
        }

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """`(B, C, H, W)` uzamsal görüntüden DCT bloklarını çıkarıp HER istatistiklerini döndürür."""
        dct_blocks = self.block_dct(image)
        out = self.compute_from_dct_blocks(dct_blocks)
        out["dct_blocks"] = dct_blocks
        return out


class LearnableFreqGate(nn.Module):
    """Öğrenilebilir Frekans Kapılama (LFG) katmanı — PHD-Net 2. Dal (Harmonik Kol).

    8x8 blok DCT katsayılarını öğrenilebilir frekans-bant kapıları ve HER haritasıyla birleştirip
    `28x28` ve `14x14` çözünürlükte frekans öznitelikleri üretir.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels_28: int = 96,
        out_channels_14: int = 160,
        low_max: int = 3,
        mid_max: int = 8,
        include_dc: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.her_calc = HarmonicEnergyRatio(
            low_max=low_max, mid_max=mid_max, include_dc=include_dc, eps=eps
        )

        # Her kanal ve 8x8 frekans konumu için öğrenilebilir kapı parametresi (başlangıçta sigmoid(0)*2 = 1.0)
        self.freq_gate_logits = nn.Parameter(torch.zeros(in_channels, 8, 8))
        # Düşük / orta / yüksek bant ağırlık ölçekleri
        self.band_scales = nn.Parameter(torch.ones(in_channels, 3))

        # Girdi kanalları: in_channels*64 (kapılanmış log-DCT) + in_channels (HER) + in_channels*3 (bant enerjileri)
        raw_freq_dim = in_channels * 64 + in_channels * 4

        self.proj_28 = nn.Sequential(
            nn.Conv2d(raw_freq_dim, out_channels_28, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels_28),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels_28,
                out_channels_28,
                kernel_size=3,
                padding=1,
                groups=out_channels_28,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels_28),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels_28, out_channels_28, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels_28),
        )

        # HER tabanlı uzamsal dikkat kapısı
        self.her_spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels_28 // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels_28 // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels_28 // 2, out_channels_28, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # 28x28 -> 14x14 derin harmonik öznitelik bloğu
        self.down_14 = nn.Sequential(
            nn.Conv2d(
                out_channels_28,
                out_channels_28,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=out_channels_28,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels_28),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels_28, out_channels_14, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels_14),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        rgb: Optional[torch.Tensor] = None,
        dct_blocks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """`rgb` `(B, 3, 224, 224)` veya önceden hesaplanmış `dct_blocks` `(B, 3, 28, 28, 8, 8)` alır.

        Returns:
            `(feat_28, feat_14, her_stats)` üçlüsü.
        """
        if dct_blocks is None:
            if rgb is None:
                raise ValueError("`rgb` veya `dct_blocks` girdilerinden en az biri verilmelidir.")
            dct_blocks = self.her_calc.block_dct(rgb)

        her_stats = self.her_calc.compute_from_dct_blocks(dct_blocks)
        b, c, hb, wb, _, _ = dct_blocks.shape

        # Öğrenilebilir 8x8 frekans kapısı + bant ölçekleme
        freq_gate = 2.0 * torch.sigmoid(self.freq_gate_logits)  # (C, 8, 8)
        low_m = self.her_calc.low_mask.to(dct_blocks.dtype)
        mid_m = self.her_calc.mid_mask.to(dct_blocks.dtype)
        high_m = self.her_calc.high_mask.to(dct_blocks.dtype)
        dc_m = (1.0 - (low_m + mid_m + high_m)).clamp(min=0.0)

        scales = F.softplus(self.band_scales)  # (C, 3) pozitif ölçekler
        band_weight_map = (
            dc_m.unsqueeze(0)
            + scales[:, 0:1].unsqueeze(-1) * low_m.unsqueeze(0)
            + scales[:, 1:2].unsqueeze(-1) * mid_m.unsqueeze(0)
            + scales[:, 2:3].unsqueeze(-1) * high_m.unsqueeze(0)
        )  # (C, 8, 8)

        total_gate = (freq_gate * band_weight_map).view(1, c, 1, 1, 8, 8)
        gated_log_dct = torch.log1p(dct_blocks.abs() * total_gate)  # (B, C, Hb, Wb, 8, 8)
        flat_dct = gated_log_dct.permute(0, 1, 4, 5, 2, 3).reshape(b, c * 64, hb, wb)

        her_map = her_stats["her_map"]
        high_to_low = her_stats["high_energy"] / (her_stats["low_energy"] + self.her_calc.eps)
        band_stack = torch.cat(
            [
                her_map,
                torch.log1p(her_stats["low_energy"]),
                torch.log1p(her_stats["mid_energy"]),
                torch.log1p(her_stats["high_energy"]),
            ],
            dim=1,
        )  # (B, 4*C, Hb, Wb)

        raw_in = torch.cat([flat_dct, band_stack], dim=1)
        feat_28_base = self.proj_28(raw_in)
        her_attn = self.her_spatial_gate(torch.cat([her_map, high_to_low], dim=1))
        feat_28 = F.silu(feat_28_base * (1.0 + her_attn))
        feat_14 = self.down_14(feat_28)
        return feat_28, feat_14, her_stats


if __name__ == "__main__":
    print("=== learnable_freq_gate.py Kendi Kendini Test Modu ===")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {dev}")

    x = torch.rand(2, 3, 224, 224, device=dev)

    # 1. Ortonormal 8x8 DCT / IDCT geri-çatım (reconstruction) doğrulaması
    dct_op = BlockDCT8x8().to(dev)
    dct_blocks = dct_op(x)
    x_rec = dct_op.inverse(dct_blocks)
    rec_err = float((x - x_rec).abs().max().item())
    print(f"[OK] 8x8 DCT blok boyutu       : {tuple(dct_blocks.shape)} | IDCT maks. hata={rec_err:.2e}")
    assert rec_err < 1e-5, f"DCT geri-çatım hatası çok yüksek: {rec_err}"

    # 2. Parametrik bant sınırları doğrulaması
    low_m, mid_m, high_m = build_dct_band_masks(low_max=3, mid_max=8, include_dc=False, device=dev)
    print(
        f"[OK] Bant katsayı sayıları     : DC=1, Low={int(low_m.sum())}, "
        f"Mid={int(mid_m.sum())}, High={int(high_m.sum())} (Toplam={1 + int((low_m + mid_m + high_m).sum())})"
    )

    # 3. HER ve LearnableFreqGate ileri + geri yayılım testi
    lfg = LearnableFreqGate(in_channels=3, out_channels_28=96, out_channels_14=160, low_max=3, mid_max=8).to(dev)
    feat_28, feat_14, stats = lfg(rgb=x)
    loss = feat_28.mean() + feat_14.mean()
    loss.backward()

    print(
        f"[OK] HER küresel değerler      : {[round(v, 4) for v in stats['her_global'].tolist()]} | "
        f"HER harita boyutu={tuple(stats['her_map'].shape)}"
    )
    print(
        f"[OK] LFG çıktıları             : feat_28={tuple(feat_28.shape)}, feat_14={tuple(feat_14.shape)} | "
        f"grad_ok={lfg.freq_gate_logits.grad is not None}"
    )
