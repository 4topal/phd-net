"""Log-Gabor Tabanlı Monokrom (Kovesi) ve Kuaterniyon Faz Uyumu (QPC) + Faz Dikkat Kapısı.

Matematiksel Kaynaklar ve Formülasyon Şeffaflığı:
1. Monokrom Faz Uyumu (`MonochromePhaseCongruency`):
   - Kaynak: P. Kovesi (1999), "Image Features From Phase Congruency", Videre: Journal of
     Computer Vision Research, 1(3), 1-26; P. Kovesi (2003), `phasecong3` algoritması.
   - Çok ölçekli ve çok yönlü 2B Log-Gabor kuadratür filtre bankası, Rayleigh medyan gürültü
     kestirimi, frekans yayılım ağırlıklandırması ve faz sapma fonksiyonu:
     `DeltaPhi(x,y) = cos(phi - phi_mean) - |sin(phi - phi_mean)|`.

2. Kuaterniyon / Renk Faz Uyumu (`QuaternionPhaseCongruency` — QPC):
   - Temel renk temsili kaynağı: T. A. Ell & S. J. Sangwine (2007), "Hypercomplex Fourier
     Transforms of Color Images", IEEE Transactions on Image Processing, 16(1), 22-35.
     Saf kuaterniyon görüntü alanı `q(x,y) = R(x,y)i + G(x,y)j + B(x,y)k` ve akromatik gri
     eksen `mu = (i + j + k) / sqrt(3)` etrafında parlaklık-krominans dik ayrıştırması kullanılır.
   - ETİKET — [KENDİ YAKLAŞIMIMIZ / ÖZGÜN UYARLAMA]:
     Referans yol haritası (§2.2) `q = Ri + Gj + Bk` ve Kromatik Faz Dispersiyonu (`DeltaPhi_chrom`)
     kavramını tanımlamakta, ancak kapalı-form bir makale denklemi vermemektedir. Bu modülde
     Kovesi (1999) Log-Gabor kuadratür analizini Ell & Sangwine (2007) saf kuaterniyon uzayıyla
     birleştiren 3-kanallı QPC tensörü **kendi yaklaşımımız** olarak tasarlanmıştır:
       * Kanal 0 (`qpc_joint`): RGB kuaterniyon genliği üzerinden birleşik hiperkompleks faz uyumu,
       * Kanal 1 (`qpc_chrom`): `mu` eksenine dik krominans düzlemi `(C1, C2)` faz uyumu,
       * Kanal 2 (`delta_phi_chrom`): R, G, B kanallarının yerel birim fazörleri arasındaki
         genlik-ağırlıklı dairesel faz dağılımı (kromatik faz dispersiyonu). Sentetik yüz birleştirme
         ve kanal-bağımsız enterpolasyon sınırlarındaki faz uyumsuzluğunu yakalar.

3. `PhaseAttentionGate`:
   - PHD-Net Uzamsal-Faz kolunda (`Stem` çıktısı 112x112) QPC haritasından uzamsal-kanal dikkat
     maskesi üreten öğrenilebilir PyTorch katmanı.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_log_gabor_bank(
    height: int,
    width: int,
    n_scales: int = 4,
    n_orientations: int = 6,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    sigma_on_f: float = 0.55,
    d_theta_on_sigma: float = 1.2,
    lp_cutoff: float = 0.45,
    lp_order: int = 15,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """2B FFT uzayında (kaydırılmamış koordinatlarda) tek-taraflı analitik Log-Gabor bankası üretir.

    Returns:
        `(n_scales, n_orientations, height, width)` boyutlu gerçel frekans maskesi.
        Gerçel görüntünün 2B FFT'si ile çarpılıp ters FFT alındığında gerçel kısım çift (even),
        sanal kısım tek (odd / Hilbert kuadratür) tepkisini verir.
    """
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")

    radius = torch.sqrt(grid_x.square() + grid_y.square())
    radius_safe = radius.clone()
    radius_safe[0, 0] = 1.0

    theta = torch.atan2(-grid_y, grid_x)

    # Köşe örtüşmesini (aliasing) bastıran Kovesi keskin Butterworth alçak-geçiren filtresi
    lowpass = 1.0 / (1.0 + (radius / lp_cutoff).pow(2 * lp_order))

    log_sigma = math.log(sigma_on_f)
    radial_filters = []
    for s in range(n_scales):
        wavelength = min_wavelength * (mult ** s)
        f0 = 1.0 / wavelength
        log_rad = torch.log(radius_safe / f0)
        rad = torch.exp(-(log_rad.square()) / (2.0 * log_sigma * log_sigma)) * lowpass
        rad[0, 0] = 0.0
        radial_filters.append(rad)
    radial_stack = torch.stack(radial_filters, dim=0)  # (S, H, W)

    theta_sigma = math.pi / (n_orientations * d_theta_on_sigma)
    angular_filters = []
    for o in range(n_orientations):
        angle = o * math.pi / n_orientations
        ds = torch.sin(theta) * math.cos(angle) - torch.cos(theta) * math.sin(angle)
        dc = torch.cos(theta) * math.cos(angle) + torch.sin(theta) * math.sin(angle)
        dtheta = torch.atan2(ds, dc).abs()
        # Tek-yönlü açısal Gauss -> analitik sinyal için 2.0 çarpanı
        ang = 2.0 * torch.exp(-(dtheta.square()) / (2.0 * theta_sigma * theta_sigma))
        angular_filters.append(ang)
    angular_stack = torch.stack(angular_filters, dim=0)  # (O, H, W)

    bank = radial_stack.unsqueeze(1) * angular_stack.unsqueeze(0)  # (S, O, H, W)
    return bank


def _kovesi_pc_from_quadrature(
    even: torch.Tensor,
    odd: torch.Tensor,
    mult: float = 2.1,
    k_noise: float = 2.0,
    cut_off: float = 0.5,
    gain: float = 10.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Kuadratür `(even, odd)` tepkilerinden Kovesi (1999) faz uyumunu hesaplar.

    Args:
        even: `(B, S, O, H, W)` çift filtre tepkisi (veya çok kanallı birleşim öncesi).
        odd: `(B, S, O, H, W)` tek filtre tepkisi.
    Returns:
        `(B, 1, H, W)` aralığı `[0, 1]` olan faz uyumu haritası.
    """
    n_scales = even.shape[1]
    amp = torch.sqrt(even.square() + odd.square() + eps * eps)  # (B, S, O, H, W)

    sum_e = even.sum(dim=1)  # (B, O, H, W)
    sum_o = odd.sum(dim=1)   # (B, O, H, W)
    sum_a = amp.sum(dim=1)   # (B, O, H, W)
    max_a = amp.amax(dim=1)  # (B, O, H, W)

    mean_norm = torch.sqrt(sum_e.square() + sum_o.square() + eps * eps)
    mean_e = sum_e / mean_norm
    mean_o = sum_o / mean_norm

    # Faz sapma ağırlığı: cos(phi - phi_bar) - |sin(phi - phi_bar)|
    dot = even * mean_e.unsqueeze(1) + odd * mean_o.unsqueeze(1)
    cross = (even * mean_o.unsqueeze(1) - odd * mean_e.unsqueeze(1)).abs()
    energy_per_scale = dot - cross  # (B, S, O, H, W)
    energy_orient = energy_per_scale.sum(dim=1)  # (B, O, H, W)

    # En ince ölçekten (s=0) Rayleigh medyan gürültü eşiği T kestirimi
    amp_s0 = amp[:, 0].flatten(start_dim=2)  # (B, O, H*W)
    median_s0 = amp_s0.median(dim=-1).values  # (B, O)
    tau = median_s0 / math.sqrt(math.log(4.0))
    total_tau = tau * (1.0 - (1.0 / mult) ** n_scales) / (1.0 - (1.0 / mult))
    mean_noise = total_tau * math.sqrt(math.pi / 2.0)
    sigma_noise = total_tau * math.sqrt((4.0 - math.pi) / 2.0)
    threshold = (mean_noise + k_noise * sigma_noise).unsqueeze(-1).unsqueeze(-1)  # (B, O, 1, 1)

    # Frekans yayılım ağırlığı W(x, y)
    width_spread = (sum_a / (max_a + eps) - 1.0) / max(n_scales - 1, 1)
    weight = 1.0 / (1.0 + torch.exp(gain * (cut_off - width_spread)))

    weighted_energy = weight * F.relu(energy_orient - threshold)
    pc = weighted_energy.sum(dim=1, keepdim=True) / (sum_a.sum(dim=1, keepdim=True) + eps)
    return pc.clamp(0.0, 1.0)


class MonochromePhaseCongruency(nn.Module):
    """Kovesi (1999) referans monokrom Log-Gabor Faz Uyumu hesaplayıcısı."""

    def __init__(
        self,
        n_scales: int = 4,
        n_orientations: int = 6,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_on_f: float = 0.55,
        k_noise: float = 2.0,
        cut_off: float = 0.5,
        gain: float = 10.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.min_wavelength = min_wavelength
        self.mult = mult
        self.sigma_on_f = sigma_on_f
        self.k_noise = k_noise
        self.cut_off = cut_off
        self.gain = gain
        self.eps = eps
        self._cached_bank: Dict[Tuple[int, int, str, torch.dtype], torch.Tensor] = {}

    def _get_bank(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, str(device), dtype)
        if key not in self._cached_bank:
            self._cached_bank[key] = _build_log_gabor_bank(
                height=h,
                width=w,
                n_scales=self.n_scales,
                n_orientations=self.n_orientations,
                min_wavelength=self.min_wavelength,
                mult=self.mult,
                sigma_on_f=self.sigma_on_f,
                device=device,
                dtype=dtype,
            )
        return self._cached_bank[key]

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Girdi `(B, 1, H, W)` veya `(B, 3, H, W)` `[0, 1]` tensöründen `(B, 1, H, W)` monokrom PC üretir."""
        orig_dtype = image.dtype
        x = image.float()
        if x.shape[1] == 3:
            # ITU-R BT.601 parlaklık dönüşümü
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        elif x.shape[1] != 1:
            raise ValueError(f"Beklenen kanal sayısı 1 veya 3, alınan: {x.shape[1]}")

        _, _, h, w = x.shape
        bank = self._get_bank(h, w, x.device, x.dtype)  # (S, O, H, W)
        img_fft = torch.fft.fft2(x)  # (B, 1, H, W) complex
        resp = torch.fft.ifft2(img_fft.unsqueeze(2) * bank.unsqueeze(0))  # (B, S, O, H, W)
        pc = _kovesi_pc_from_quadrature(
            resp.real,
            resp.imag,
            mult=self.mult,
            k_noise=self.k_noise,
            cut_off=self.cut_off,
            gain=self.gain,
            eps=self.eps,
        )
        return pc.to(orig_dtype)


class QuaternionPhaseCongruency(nn.Module):
    """Kuaterniyon / Renk Faz Uyumu (QPC) hesaplayıcısı — [KENDİ YAKLAŞIMIMIZ].

    Saf kuaterniyon renk alanı `q(x,y) = R*i + G*j + B*k` (Ell & Sangwine, 2007) üzerinde
    Kovesi (1999) Log-Gabor kuadratür filtre bankasını çalıştırarak 3-kanallı çıktı üretir:
      - Kanal 0 (`qpc_joint`): Birleşik kuaterniyon enerjili faz uyumu `[0, 1]`
      - Kanal 1 (`qpc_chrom`): Akromatik gri eksene (`(i+j+k)/sqrt(3)`) dik krominans faz uyumu `[0, 1]`
      - Kanal 2 (`delta_phi_chrom`): Kromatik Faz Dispersiyonu (`DeltaPhi_chrom`) `[0, 1]`
    """

    def __init__(
        self,
        n_scales: int = 4,
        n_orientations: int = 6,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_on_f: float = 0.55,
        k_noise: float = 2.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.min_wavelength = min_wavelength
        self.mult = mult
        self.sigma_on_f = sigma_on_f
        self.k_noise = k_noise
        self.eps = eps
        self._cached_bank: Dict[Tuple[int, int, str, torch.dtype], torch.Tensor] = {}

    def _get_bank(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, str(device), dtype)
        if key not in self._cached_bank:
            self._cached_bank[key] = _build_log_gabor_bank(
                height=h,
                width=w,
                n_scales=self.n_scales,
                n_orientations=self.n_orientations,
                min_wavelength=self.min_wavelength,
                mult=self.mult,
                sigma_on_f=self.sigma_on_f,
                device=device,
                dtype=dtype,
            )
        return self._cached_bank[key]

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Girdi `(B, 3, H, W)` RGB `[0, 1]` -> Çıktı `(B, 3, H, W)` QPC tensörü."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Beklenen girdi boyutu (B, 3, H, W), alınan: {tuple(rgb.shape)}")

        orig_dtype = rgb.dtype
        x = rgb.float()
        _, _, h, w = x.shape
        bank = self._get_bank(h, w, x.device, x.dtype)  # (S, O, H, W)

        # RGB kanalları için tek seferde toplu 2B FFT ve kuadratür ters FFT
        # x_fft: (B, 3, H, W) -> resp: (B, 3, S, O, H, W)
        x_fft = torch.fft.fft2(x)
        resp = torch.fft.ifft2(
            x_fft.unsqueeze(2).unsqueeze(3) * bank.unsqueeze(0).unsqueeze(0)
        )
        e_rgb = resp.real  # (B, 3, S, O, H, W)
        o_rgb = resp.imag  # (B, 3, S, O, H, W)

        # --- Kanal 0: Birleşik Kuaterniyon Faz Uyumu (q = Ri + Gj + Bk) ---
        # Gri eksen yansıması + hiperkompleks enerji birleşimi
        # Akromatik eksen mu = (1, 1, 1)/sqrt(3) boyunca işaretli projeksiyon faz yönünü korur,
        # genlik ise tüm saf kuaterniyon bileşenlerinden gelir: ||q_{s,o}||_F
        e_lum = (e_rgb[:, 0] + e_rgb[:, 1] + e_rgb[:, 2]) / math.sqrt(3.0)
        o_lum = (o_rgb[:, 0] + o_rgb[:, 1] + o_rgb[:, 2]) / math.sqrt(3.0)
        q_amp = torch.sqrt(e_rgb.square().sum(dim=1) + o_rgb.square().sum(dim=1) + self.eps * self.eps)
        lum_amp = torch.sqrt(e_lum.square() + o_lum.square() + self.eps * self.eps)
        scale_factor = q_amp / lum_amp
        qpc_joint = _kovesi_pc_from_quadrature(
            e_lum * scale_factor,
            o_lum * scale_factor,
            mult=self.mult,
            k_noise=self.k_noise,
            eps=self.eps,
        )  # (B, 1, H, W)

        # --- Kanal 1: Dik Krominans Düzlemi Faz Uyumu (Ell & Sangwine, 2007 dik altuzayı) ---
        # C1 = (R - G)/sqrt(2), C2 = (R + G - 2B)/sqrt(6)
        e_c1 = (e_rgb[:, 0] - e_rgb[:, 1]) / math.sqrt(2.0)
        o_c1 = (o_rgb[:, 0] - o_rgb[:, 1]) / math.sqrt(2.0)
        e_c2 = (e_rgb[:, 0] + e_rgb[:, 1] - 2.0 * e_rgb[:, 2]) / math.sqrt(6.0)
        o_c2 = (o_rgb[:, 0] + o_rgb[:, 1] - 2.0 * o_rgb[:, 2]) / math.sqrt(6.0)

        chrom_amp = torch.sqrt(
            e_c1.square() + o_c1.square() + e_c2.square() + o_c2.square() + self.eps * self.eps
        )
        c_sum_e = e_c1 + e_c2
        c_sum_o = o_c1 + o_c2
        c_norm = torch.sqrt(c_sum_e.square() + c_sum_o.square() + self.eps * self.eps)
        qpc_chrom = _kovesi_pc_from_quadrature(
            (c_sum_e / c_norm) * chrom_amp,
            (c_sum_o / c_norm) * chrom_amp,
            mult=self.mult,
            k_noise=self.k_noise,
            eps=self.eps,
        )  # (B, 1, H, W)

        # --- Kanal 2: Kromatik Faz Dispersiyonu (DeltaPhi_chrom) [KENDİ YAKLAŞIMIMIZ] ---
        # Her kanalın birim fazörü u_c = (e_c, o_c) / |z_c|.
        # Kanallar hizalı fazdaysa ||mean_c(u_c)|| = 1 -> dispersiyon = 0.
        # Yüz yapıştırma/enterpolasyon sınırında kanal fazları ayrışırsa dispersiyon artar.
        ch_amp = torch.sqrt(e_rgb.square() + o_rgb.square() + self.eps * self.eps)  # (B, 3, S, O, H, W)
        unit_e = e_rgb / ch_amp
        unit_o = o_rgb / ch_amp
        mean_unit_len = torch.sqrt(
            unit_e.mean(dim=1).square() + unit_o.mean(dim=1).square() + self.eps * self.eps
        ).clamp(0.0, 1.0)
        phase_dispersion = 1.0 - mean_unit_len  # (B, S, O, H, W) in [0, 1]

        # Düşük enerjili düz arka planda gürültü fazörlerinin baskın olmaması için genlik kapılaması
        energy_gate = chrom_amp / (q_amp + 0.05)
        disp_num = (phase_dispersion * energy_gate * q_amp).sum(dim=(1, 2), keepdim=False)
        disp_den = q_amp.sum(dim=(1, 2), keepdim=False) + self.eps
        delta_phi_chrom = (disp_num / disp_den).unsqueeze(1).clamp(0.0, 1.0)  # (B, 1, H, W)

        out = torch.cat([qpc_joint, qpc_chrom, delta_phi_chrom], dim=1)
        return out.to(orig_dtype)

    @staticmethod
    def summary_map(qpc_tensor: torch.Tensor) -> torch.Tensor:
        """Görselleştirme için 3-kanallı QPC çıktısını tek kanallı `(B, 1, H, W)` haritaya indirger."""
        # Birleşik faz uyumu + kromatik faz dispersiyonu vurgusu
        combined = 0.50 * qpc_tensor[:, 0:1] + 0.25 * qpc_tensor[:, 1:2] + 0.25 * qpc_tensor[:, 2:3]
        return combined.clamp(0.0, 1.0)


class PhaseAttentionGate(nn.Module):
    """QPC haritasını uzamsal özellik haritalarına dikkat kapısı olarak uygulayan PyTorch katmanı.

    Formül:
        `F_out = F_spatial * (1 + alpha * sigmoid(ConvGate(QPC)))`
    """

    def __init__(self, spatial_channels: int, qpc_channels: int = 3, hidden_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(qpc_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, spatial_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, spatial_feat: torch.Tensor, qpc_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uzamsal özellikleri QPC dikkat maskesi ile modüle eder.

        Args:
            spatial_feat: `(B, C, H_s, W_s)` uzamsal stem özellikleri.
            qpc_map: `(B, 3, H, W)` QPC tensörü.
        Returns:
            `(gated_feat, attention_mask)` ikilisi; her ikisi de `(B, C, H_s, W_s)` boyutludur.
        """
        if qpc_map.shape[-2:] != spatial_feat.shape[-2:]:
            qpc_resized = F.interpolate(
                qpc_map, size=spatial_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            qpc_resized = qpc_map

        attn_mask = self.encoder(qpc_resized)
        gated_feat = spatial_feat * (1.0 + self.alpha * attn_mask)
        return gated_feat, attn_mask


def compute_qpc_numpy(rgb_uint8: np.ndarray, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """`(H, W, 3)` uint8 RGB görüntüden `(H, W)` monokrom PC ve `(H, W, 3)` QPC döndürür."""
    tensor = torch.from_numpy(rgb_uint8).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
    mono_mod = MonochromePhaseCongruency().to(device)
    qpc_mod = QuaternionPhaseCongruency().to(device)
    mono = mono_mod(tensor).squeeze(0).squeeze(0).cpu().numpy()
    qpc = qpc_mod(tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return mono, qpc


if __name__ == "__main__":
    print("=== phase_congruency.py Kendi Kendini Test Modu ===")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {dev}")

    # Sentetik test görüntüsü: Keskin kenar + yapay renk kaymalı (kromatik dispersiyonlu) sınır
    b, h, w = 2, 224, 224
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=dev),
        torch.linspace(-1.0, 1.0, w, device=dev),
        indexing="ij",
    )
    radius = torch.sqrt(xx.square() + yy.square())
    # Örnek 0: Hizalı akromatik daire sınırı | Örnek 1: R ve B kanalları kaydırılmış sentetik sınır
    aligned_step = (radius < 0.45).float()
    r_shift = (torch.sqrt((xx - 0.025).square() + yy.square()) < 0.45).float()
    b_shift = (torch.sqrt((xx + 0.025).square() + yy.square()) < 0.45).float()

    img0 = torch.stack([aligned_step, aligned_step, aligned_step], dim=0) * 0.8 + 0.1
    img1 = torch.stack([r_shift, aligned_step, b_shift], dim=0) * 0.8 + 0.1
    batch_rgb = torch.stack([img0, img1], dim=0)

    mono_pc = MonochromePhaseCongruency().to(dev)
    qpc_calc = QuaternionPhaseCongruency().to(dev)
    gate = PhaseAttentionGate(spatial_channels=48, qpc_channels=3).to(dev)

    if dev.type == "cuda":
        _ = qpc_calc(batch_rgb[:1])
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    mono_out = mono_pc(batch_rgb)
    qpc_out = qpc_calc(batch_rgb)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    dummy_stem = torch.randn(b, 48, 112, 112, device=dev, requires_grad=True)
    gated_out, attn_mask = gate(dummy_stem, qpc_out)
    loss = gated_out.mean()
    loss.backward()

    disp_aligned = float(qpc_out[0, 2].mean().item())
    disp_shifted = float(qpc_out[1, 2].mean().item())

    print(f"[OK] Monokrom Kovesi PC boyutu : {tuple(mono_out.shape)} | min={mono_out.min():.4f}, max={mono_out.max():.4f}")
    print(f"[OK] Kuaterniyon QPC boyutu    : {tuple(qpc_out.shape)} | süre (batch=2)={dt_ms:.2f} ms")
    print(
        f"[OK] Kromatik Faz Dispersiyonu : hizalı kenar ort.={disp_aligned:.5f} < "
        f"renk-kaymalı kenar ort.={disp_shifted:.5f}"
    )
    assert disp_shifted > disp_aligned, "Renk-kaymalı sınırda kromatik faz dispersiyonu daha yüksek olmalı!"
    print(
        f"[OK] PhaseAttentionGate çıktısı: {tuple(gated_out.shape)} | "
        f"grad_ok={dummy_stem.grad is not None and gate.alpha.grad is not None}"
    )
