"""PHD-Net (Phase-Harmonic Discrepancy Network) İki Dallı Mimari, Kayıplar ve GPU Benchmark.

Mimari Özeti:
1. Dal 1 — Uzamsal-Faz Kolu (`Spatial-Phase Branch`):
   - `Stem` (224x224 -> 112x112)
   - `PhaseAttentionGate` (QPC haritası ile uzamsal-faz dikkat kapılaması)
   - Hafif çok-ölçekli derinlemesine ayrılabilir evrişim omurgası (EdgeNeXt / RepVGG esinli bloklar:
     112x112 -> 56x56 -> 28x28 -> 14x14)
2. Dal 2 — Sıkıştırma-İnvaryant Harmonik Kolu (`Harmonic Branch`):
   - `BlockDCT8x8` -> `HarmonicEnergyRatio` (HER) -> `LearnableFreqGate` (LFG: 28x28 ve 14x14)
3. Dinamik Frekans-Uzamsal Kalibrasyon (`DFSC` Füzyon) & Hiyerarşik Köprü:
   - 28x28 ve 14x14 ölçeklerinde çift-yönlü çapraz kapılama (cross-gating) ile uzamsal ve
     harmonik özniteliklerin dinamik kalibrasyonu.
   - `bridge_28_to_14`: `fused_28` özniteliklerini `fused_14` üzerine bağlayan Stride-2 DWConv köprüsü.
4. Çok Görevli Çıktı Başlıkları:
   - Başlık 1: Ortak Boyun (`shared_neck: 352 -> 256`) üzerinden ikili sınıflandırma logiti (`logits: (B, 1)`)
     ve L2-normalize kontrastif gömme (`z_ccm: (B, 128)`).
   - Başlık 2: QPC sınır rehberlikli adli lokalizasyon maskesi logiti (`mask_logits: (B, 1, 224, 224)`),
     `compute_mask=False` seçeneğiyle hafifletilebilir.

Kontrastif Kayıp Tanımı (`CrossManipulationContrastiveLoss` — `L_CCM`):
   Birim hiperküre `S^{d-1}` üzerindeki L2-normalize gömmeler `{z_i}_{i=1}^B` ve etiketler
   `y_i in {0 (gerçek), 1 (sahte)}` için:
     - Gerçek yüz hiperküresel başlık terimi (`L_real`):
       `L_real = mean_{i != j, y_i = y_j = 0} relu(real_margin - <z_i, z_j>)^2`
       (Gerçek yüzleri tek bir 0D noktaya çökertmeden hiperküresel başlık içinde toplar.)
     - Gerçek-Sahte marjlı itme terimi (`L_sep`):
       Karesel menteşe (hinge) marjı + sıcaklık ölçekli Log-Sum-Exp hard-negative itme terimi.
     - Toplam: `L_CCM = L_real + L_sep`.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from frequency_extractor import ClassicalFrequencyExtractor
    from learnable_freq_gate import BlockDCT8x8, LearnableFreqGate
    from phase_congruency import PhaseAttentionGate, QuaternionPhaseCongruency
except ImportError:
    from src.frequency_extractor import ClassicalFrequencyExtractor
    from src.learnable_freq_gate import BlockDCT8x8, LearnableFreqGate
    from src.phase_congruency import PhaseAttentionGate, QuaternionPhaseCongruency

KOD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = KOD_ROOT / "outputs" / "logs" / "benchmark.json"


class MultiScaleDWBlock(nn.Module):
    """EdgeNeXt / RepVGG esinli hafif çok-ölçekli derinlemesine ayrılabilir evrişim bloğu."""

    def __init__(self, dim: int, expand_ratio: float = 3.0) -> None:
        super().__init__()
        hidden = int(round(dim * expand_ratio))
        half = dim // 2
        rem = dim - half

        # İki paralel derinlemesine çekirdek (3x3 yerel kenar + 5x5 geniş bağlam)
        self.dw_3x3 = nn.Conv2d(half, half, kernel_size=3, padding=1, groups=half, bias=False)
        self.dw_5x5 = nn.Conv2d(rem, rem, kernel_size=5, padding=2, groups=rem, bias=False)
        self.bn_dw = nn.BatchNorm2d(dim)

        self.pw_expand = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.bn_exp = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.pw_proj = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(dim)
        self._half = half

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, : self._half], x[:, self._half :]
        dw = torch.cat([self.dw_3x3(x1), self.dw_5x5(x2)], dim=1)
        dw = self.bn_dw(dw)
        out = self.pw_expand(dw)
        out = self.act(self.bn_exp(out))
        out = self.bn_proj(self.pw_proj(out))
        return x + out


class DownsampleStage(nn.Module):
    """Uzamsal çözünürlüğü yarıya indiren geçiş + `depth` adet `MultiScaleDWBlock`."""

    def __init__(self, in_dim: int, out_dim: int, depth: int, expand_ratio: float = 3.0) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[MultiScaleDWBlock(out_dim, expand_ratio=expand_ratio) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class DynamicFreqSpatialCalibration(nn.Module):
    """Dinamik Frekans-Uzamsal Kalibrasyon (DFSC) çift-yönlü füzyon modülü."""

    def __init__(self, spatial_dim: int, freq_dim: int, out_dim: int, use_cross_gating: bool = True) -> None:
        super().__init__()
        self.use_cross_gating = use_cross_gating
        self.freq_align = nn.Sequential(
            nn.Conv2d(freq_dim, spatial_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(spatial_dim),
            nn.SiLU(inplace=True),
        )
        if use_cross_gating:
            # Kanal-uzamsal karşılıklı kalibrasyon kapıları
            self.s_to_f_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(spatial_dim, spatial_dim // 4, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(spatial_dim // 4, spatial_dim, kernel_size=1),
                nn.Sigmoid(),
            )
            self.f_to_s_gate = nn.Sequential(
                nn.Conv2d(spatial_dim, 1, kernel_size=3, padding=1, bias=True),
                nn.Sigmoid(),
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(spatial_dim * 2, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(inplace=True),
            )
        else:
            # Ablasyon modu (no_dfsc): Doğrudan toplama sonrası projeksiyon
            self.out_proj = nn.Sequential(
                nn.Conv2d(spatial_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(inplace=True),
            )

    def forward(self, f_spatial: torch.Tensor, f_freq: torch.Tensor) -> torch.Tensor:
        freq_proj = self.freq_align(f_freq)
        if not self.use_cross_gating:
            return self.out_proj(f_spatial + freq_proj)
        cal_freq = freq_proj * (1.0 + self.s_to_f_gate(f_spatial))
        cal_spat = f_spatial * (1.0 + self.f_to_s_gate(freq_proj))
        return self.fuse(torch.cat([cal_spat, cal_freq], dim=1))


class LocalizationHead(nn.Module):
    """Çok ölçekli adli lokalizasyon ısı haritası çözücüsü (14x14 + 28x28 -> 224x224 maske logiti).

    Kusur 5 Düzeltmesi: QPC Sınır Rehberliği (`phase_attn`) ile 112x112 aşamasında kenar kapılaması.
    """

    def __init__(self, in_dim_14: int, skip_dim_28: int, mid_dim: int = 96) -> None:
        super().__init__()
        self.up_14_to_28 = nn.Sequential(
            nn.Conv2d(in_dim_14, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.SiLU(inplace=True),
        )
        self.fuse_28 = nn.Sequential(
            nn.Conv2d(mid_dim + skip_dim_28, mid_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.SiLU(inplace=True),
        )
        self.decode_56 = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim // 2),
            nn.SiLU(inplace=True),
        )
        self.decode_112 = nn.Sequential(
            nn.Conv2d(mid_dim // 2, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
        )
        self.edge_guide = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Sigmoid(),
        )
        self.final_conv = nn.Conv2d(24, 1, kernel_size=1, bias=True)

    def forward(
        self,
        fused_14: torch.Tensor,
        fused_28: torch.Tensor,
        out_hw: Tuple[int, int],
        phase_attn: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = F.interpolate(self.up_14_to_28(fused_14), size=fused_28.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse_28(torch.cat([x, fused_28], dim=1))
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.decode_56(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        feat_112 = self.decode_112(x)
        if phase_attn is not None:
            edge = phase_attn.mean(dim=1, keepdim=True)
            if edge.shape[-2:] != feat_112.shape[-2:]:
                edge = F.interpolate(edge, size=feat_112.shape[-2:], mode="bilinear", align_corners=False)
            feat_112 = feat_112 * (1.0 + self.edge_guide(edge))
        logits_112 = self.final_conv(feat_112)
        return F.interpolate(logits_112, size=out_hw, mode="bilinear", align_corners=False)


class PHDNetPreprocessor(nn.Module):
    """QPC ve 8x8 Blok DCT ön işleme hattı."""

    def __init__(self) -> None:
        super().__init__()
        self.qpc = QuaternionPhaseCongruency()
        self.block_dct = BlockDCT8x8(block_size=8)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qpc_map = self.qpc(rgb)
        dct_blocks = self.block_dct(rgb)
        return qpc_map, dct_blocks


class PHDNet(nn.Module):
    """PHD-Net (Phase-Harmonic Discrepancy Network) ana modeli.

    Ablasyon Desteği (`ablation_mode`):
      - "full": Varsayılan tam PHD-Net modeli
      - "no_qpc": QPC dikkat kapısını atlar (s_112_gated = s_112, phase_attn = None)
      - "no_lfg": Frekans özelliklerini sıfır tensörlerle değiştirir (yalnızca uzamsal kol)
      - "no_dfsc": DFSC çapraz kapılamasını kaldırıp doğrudan toplama (s + freq_proj) uygular
      - "classical_freq": Klasik FFT + DCT istatistiklerini (ClassicalFrequencyExtractor) kullanır
      - "no_ccm": Kontrastif kayıp ağırlığı olmadan eğitim için mimari mod
    """

    localization_head_status: str = "REQUIRES_GROUND_TRUTH_MASKS"

    def __init__(
        self,
        stem_dim: int = 48,
        stage_dims: Tuple[int, int, int] = (88, 176, 272),
        stage_depths: Tuple[int, int, int] = (2, 3, 4),
        expand_ratio: float = 2.5,
        embed_dim: int = 128,
        low_max: int = 3,
        mid_max: int = 8,
        ablation_mode: str = "full",
    ) -> None:
        super().__init__()
        valid_modes = {"full", "no_qpc", "no_lfg", "no_dfsc", "classical_freq", "no_ccm"}
        if ablation_mode not in valid_modes:
            raise ValueError(f"Geçersiz ablation_mode: {ablation_mode}. Desteklenenler: {valid_modes}")
        self.ablation_mode = ablation_mode
        self.preprocessor = PHDNetPreprocessor()

        # 1. Dal: Uzamsal-Faz Kolu
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.SiLU(inplace=True),
        )
        if self.ablation_mode != "no_qpc":
            self.phase_gate = PhaseAttentionGate(spatial_channels=stem_dim, qpc_channels=3, hidden_channels=32)
        else:
            self.phase_gate = None

        d1, d2, d3 = stage_dims
        n1, n2, n3 = stage_depths
        self.stage1_56 = DownsampleStage(stem_dim, d1, depth=n1, expand_ratio=expand_ratio)
        self.stage2_28 = DownsampleStage(d1, d2, depth=n2, expand_ratio=expand_ratio)
        self.stage3_14 = DownsampleStage(d2, d3, depth=n3, expand_ratio=expand_ratio)

        # 2. Dal: Sıkıştırma-İnvaryant Harmonik Kolu (DCT -> HER -> LFG) veya Klasik Frekans Ablasyonu
        if self.ablation_mode == "classical_freq":
            self.freq_branch = None
            self.classical_extractor = ClassicalFrequencyExtractor(low_max=low_max, mid_max=mid_max)
            self.classical_proj_28 = nn.Sequential(
                nn.Linear(91, 96, bias=False),
                nn.BatchNorm1d(96),
                nn.SiLU(inplace=True),
            )
            self.classical_proj_14 = nn.Sequential(
                nn.Linear(91, 160, bias=False),
                nn.BatchNorm1d(160),
                nn.SiLU(inplace=True),
            )
        elif self.ablation_mode == "no_lfg":
            self.freq_branch = None
            self.classical_extractor = None
        else:
            self.freq_branch = LearnableFreqGate(
                in_channels=3,
                out_channels_28=96,
                out_channels_14=160,
                low_max=low_max,
                mid_max=mid_max,
            )
            self.classical_extractor = None

        # Dinamik Frekans-Uzamsal Kalibrasyon (DFSC)
        use_cg = self.ablation_mode != "no_dfsc"
        self.dfsc_28 = DynamicFreqSpatialCalibration(spatial_dim=d2, freq_dim=96, out_dim=160, use_cross_gating=use_cg)
        self.dfsc_14 = DynamicFreqSpatialCalibration(spatial_dim=d3, freq_dim=160, out_dim=352, use_cross_gating=use_cg)

        # Kusur 4: 28->14 Köprü Bağlantısı (Stride-2 DWConv 160 -> 160 + PWConv 160 -> 352)
        self.bridge_28_to_14 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, stride=2, padding=1, groups=160, bias=False),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
            nn.Conv2d(160, 352, kernel_size=1, bias=False),
            nn.BatchNorm2d(352),
        )

        # Kusur 7: Ortak Kontrastif Boyun (`shared_neck: 352 -> 256`) + Sınıflandırma + Kontrastif Gömme
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_neck = nn.Sequential(
            nn.Linear(352, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
        )
        self.contrastive_head = nn.Linear(256, embed_dim, bias=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Başlık 2: Adli Lokalizasyon Isı Haritası
        self.localization_head = LocalizationHead(in_dim_14=352, skip_dim_28=160, mid_dim=96)

    def forward(
        self,
        rgb: torch.Tensor,
        qpc_map: Optional[torch.Tensor] = None,
        dct_blocks: Optional[torch.Tensor] = None,
        compute_mask: bool = True,
    ) -> Dict[str, Any]:
        """İleri yayılım.

        Ön işleme (`qpc_map`, `dct_blocks`) dışarıdan verildiğinde yalnızca sinir ağı katmanlarını çalıştırır;
        verilmediğinde `self.preprocessor` ile otomatik hesaplar.
        `compute_mask=False` seçildiğinde çıkarım hızlandırması için `localization_head` atlanır.
        """
        # Ön işleme (gerekiyorsa)
        if self.ablation_mode != "no_qpc" and qpc_map is None:
            qpc_map = self.preprocessor.qpc(rgb)

        if self.ablation_mode not in ("no_lfg", "classical_freq") and dct_blocks is None:
            dct_blocks = self.preprocessor.block_dct(rgb)

        # 1. Uzamsal-Faz Kolu
        s_112 = self.stem(rgb)
        if self.ablation_mode == "no_qpc" or self.phase_gate is None:
            s_112_gated = s_112
            phase_attn = None
        else:
            s_112_gated, phase_attn = self.phase_gate(s_112, qpc_map)

        s_56 = self.stage1_56(s_112_gated)
        s_28 = self.stage2_28(s_56)
        s_14 = self.stage3_14(s_28)

        # 2. Harmonik Kol (veya Frekans Ablasyonu)
        b = rgb.shape[0]
        if self.ablation_mode == "no_lfg":
            f_28 = s_28.new_zeros(b, 96, s_28.shape[2], s_28.shape[3])
            f_14 = s_14.new_zeros(b, 160, s_14.shape[2], s_14.shape[3])
            her_stats = {"her_global": s_28.new_zeros(b, 3)}
        elif self.ablation_mode == "classical_freq":
            class_feats = self.classical_extractor(rgb)["combined_features"]
            f_28_vec = self.classical_proj_28(class_feats)
            f_28 = f_28_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, s_28.shape[2], s_28.shape[3]).contiguous()
            f_14_vec = self.classical_proj_14(class_feats)
            f_14 = f_14_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, s_14.shape[2], s_14.shape[3]).contiguous()
            her_stats = {"her_global": s_28.new_zeros(b, 3)}
        else:
            f_28, f_14, her_stats = self.freq_branch(dct_blocks=dct_blocks)

        # 3. DFSC Füzyon + Kusur 4: bridge_28_to_14
        fused_28 = self.dfsc_28(s_28, f_28)
        fused_14 = self.dfsc_14(s_14, f_14) + self.bridge_28_to_14(fused_28)

        # 4. Başlık 1: Ortak Boyun -> Sınıflandırma + Kontrastif Gömme
        pooled = self.global_pool(fused_14).flatten(1)
        neck = self.shared_neck(pooled)
        logits = self.classifier(neck)
        z_ccm = F.normalize(self.contrastive_head(neck), p=2, dim=-1)

        # 5. Başlık 2: Lokalizasyon Maskesi (compute_mask=True ise)
        if compute_mask:
            mask_logits = self.localization_head(
                fused_14, fused_28, out_hw=rgb.shape[-2:], phase_attn=phase_attn
            )
        else:
            mask_logits = None

        return {
            "logits": logits,
            "embedding": z_ccm,
            "mask_logits": mask_logits,
            "her_global": her_stats["her_global"],
            "phase_attention": phase_attn,
        }


class DiceLoss(nn.Module):
    """Piksel düzeyinde manipülasyon maskesi için yumuşatılmış Dice kaybı."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, mask_logits: torch.Tensor, mask_targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(mask_logits).flatten(1)
        targets = mask_targets.float().flatten(1)
        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return (1.0 - dice).mean()


class CrossManipulationContrastiveLoss(nn.Module):
    """Çapraz Manipülasyon Kontrastif Kaybı (`L_CCM`).

    Kusur 7 Düzeltmesi:
        L2-normalize gömmeler `z_i in S^{d-1}` ve ikili etiketler `y_i in {0: gerçek, 1: sahte}` için:
        1) Gerçek yüz hiperküresel başlık terimi (`L_real`):
           `L_real = mean_{i != j, y_i = y_j = 0} relu(real_margin - z_i^T z_j)^2`
           Gerçek yüz temsillerini tek bir 0D noktaya çökertmeden hiperküresel başlık (`real_margin = 0.85`)
           içinde tutar.
        2) Gerçek-Sahte marjlı ayrıştırma terimi (`L_sep`):
           Karesel menteşe (hinge) marjı + sıcaklık ölçekli Log-Sum-Exp hard-negative itme terimi:
           `hinge_sep = mean_{i in I_real, k in I_fake} relu(z_i^T z_k - margin)^2`
           `hard_neg = temperature * log1p(mean(exp((z_i^T z_k - margin) / temperature)))`
           `L_sep = hinge_sep + 0.5 * hard_neg`
        `L_CCM = L_real + L_sep`.
    """

    def __init__(
        self,
        margin: float = 0.15,
        real_margin: float = 0.85,
        temperature: float = 0.10,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.real_margin = real_margin
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(embeddings, p=2, dim=-1)
        y = labels.view(-1)
        real_idx = torch.nonzero(y == 0, as_tuple=False).squeeze(-1)
        fake_idx = torch.nonzero(y == 1, as_tuple=False).squeeze(-1)

        loss_real = z.new_tensor(0.0)
        if real_idx.numel() >= 2:
            z_real = z[real_idx]
            sim_real = z_real @ z_real.t()
            n_r = z_real.shape[0]
            off_diag = ~torch.eye(n_r, dtype=torch.bool, device=z.device)
            loss_real = F.relu(self.real_margin - sim_real[off_diag]).square().mean()

        loss_sep = z.new_tensor(0.0)
        if real_idx.numel() >= 1 and fake_idx.numel() >= 1:
            sim_rf = z[real_idx] @ z[fake_idx].t()
            hinge_sep = F.relu(sim_rf - self.margin).square().mean()
            diff_scaled = torch.clamp((sim_rf - self.margin) / self.temperature, max=50.0)
            hard_neg = self.temperature * torch.log1p(torch.exp(diff_scaled).mean())
            loss_sep = hinge_sep + 0.5 * hard_neg

        return loss_real + loss_sep


class PHDNetMultiTaskLoss(nn.Module):
    """PHD-Net çok görevli kayıp sarmalayıcısı: `L_BCE + lambda_1 * L_Loc + lambda_2 * L_CCM`.

    `mask_targets=None` veya `outputs["mask_logits"] is None` verildiğinde yerelleştirme kaybı
    hesaplanmaz ve çıktıda `dice_evaluated=False` olarak raporlanır.
    `has_mask` tensörü sağlandığında yalnızca `has_mask == True` olan örnekler için kısmi yerelleştirme
    kaybı (`0.5 * BCEWithLogits + 0.5 * Dice`) hesaplanır.
    """

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_ccm: float = 0.5,
        ccm_margin: float = 0.15,
        ccm_real_margin: float = 0.85,
        ccm_temp: float = 0.10,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.ccm = CrossManipulationContrastiveLoss(
            margin=ccm_margin, real_margin=ccm_real_margin, temperature=ccm_temp
        )
        self.lambda_dice = lambda_dice
        self.lambda_ccm = lambda_ccm

    def forward(
        self,
        outputs: Dict[str, Any],
        labels: torch.Tensor,
        mask_targets: Optional[torch.Tensor] = None,
        has_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        y_float = labels.float().view(-1, 1)
        l_bce = self.bce(outputs["logits"], y_float)
        l_ccm = self.ccm(outputs["embedding"], labels.view(-1))
        total = l_bce + self.lambda_ccm * l_ccm

        mask_logits = outputs.get("mask_logits")
        dice_eval = False
        dice_val: Optional[float] = None

        if mask_targets is not None and mask_logits is not None:
            if has_mask is not None:
                valid_idx = torch.nonzero(has_mask.view(-1).bool(), as_tuple=False).squeeze(-1)
                if valid_idx.numel() > 0:
                    m_logits = mask_logits[valid_idx]
                    m_targets = mask_targets[valid_idx].float()
                    l_loc = 0.5 * F.binary_cross_entropy_with_logits(m_logits, m_targets) + 0.5 * self.dice(
                        m_logits, m_targets
                    )
                    total = total + self.lambda_dice * l_loc
                    dice_val = float(l_loc.detach().item())
                    dice_eval = True
            else:
                m_logits = mask_logits
                m_targets = mask_targets.float()
                l_loc = 0.5 * F.binary_cross_entropy_with_logits(m_logits, m_targets) + 0.5 * self.dice(
                    m_logits, m_targets
                )
                total = total + self.lambda_dice * l_loc
                dice_val = float(l_loc.detach().item())
                dice_eval = True

        return {
            "total_loss": total,
            "bce_loss": float(l_bce.detach().item()),
            "ccm_loss": float(l_ccm.detach().item()),
            "dice_loss": dice_val,
            "loc_loss": dice_val,
            "dice_evaluated": dice_eval,
        }


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """`(toplam_parametre, eğitilebilir_parametre)` sayılarını döndürür."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def collect_environment_info(
    device: torch.device,
    input_shape: Tuple[int, int, int, int],
    dtype: torch.dtype,
    warmup: int,
    runs: int,
) -> Dict[str, Any]:
    """Çalışma ortamı meta verilerini toplar."""
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    total_vram_mb = (
        round(torch.cuda.get_device_properties(device).total_memory / (1024**2), 2)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_type": device.type,
        "gpu_name": gpu_name,
        "gpu_total_vram_mb": total_vram_mb,
        "dtype": str(dtype).replace("torch.", ""),
        "input_shape": list(input_shape),
        "warmup_iterations": warmup,
        "timed_iterations": runs,
    }


def measure_callable_latency(
    fn: Callable[[], Any],
    device: torch.device,
    warmup: int = 50,
    runs: int = 250,
) -> Dict[str, float]:
    """GPU (CUDA Event) veya CPU üzerinde ısınma + `runs` tekrar ile gecikme ve tepe VRAM ölçer."""
    if runs < 200:
        raise ValueError(f"Benchmark protokolü gereği runs >= 200 olmalıdır (verilen={runs}).")

    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]

            for i in range(runs):
                start_events[i].record()
                _ = fn()
                end_events[i].record()

            torch.cuda.synchronize(device)
            times_ms = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=np.float64)
            peak_bytes = int(torch.cuda.max_memory_allocated(device))
        else:
            times_list: List[float] = []
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = fn()
                times_list.append((time.perf_counter() - t0) * 1000.0)
            times_ms = np.array(times_list, dtype=np.float64)
            peak_bytes = 0

    median_ms = float(np.median(times_ms))
    mean_ms = float(np.mean(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))
    min_ms = float(np.min(times_ms))
    max_ms = float(np.max(times_ms))
    fps_median = 1000.0 / median_ms if median_ms > 0 else 0.0

    return {
        "latency_median_ms": round(median_ms, 4),
        "latency_mean_ms": round(mean_ms, 4),
        "latency_p95_ms": round(p95_ms, 4),
        "latency_min_ms": round(min_ms, 4),
        "latency_max_ms": round(max_ms, 4),
        "fps_from_median": round(fps_median, 2),
        "max_memory_allocated_bytes": peak_bytes,
        "max_memory_allocated_mb": round(peak_bytes / (1024**2), 3),
    }


def update_benchmark_json(
    section_key: str,
    section_data: Dict[str, Any],
    env_info: Dict[str, Any],
    json_path: Path = DEFAULT_BENCHMARK_PATH,
) -> Dict[str, Any]:
    """`benchmark.json` dosyasını mevcut diğer model sonuçlarını silmeden günceller."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    if json_path.is_file():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}

    payload["environment"] = env_info
    payload.setdefault("models", {})[section_key] = section_data
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def run_phdnet_benchmark(
    warmup: int = 50,
    runs: int = 250,
    json_path: Path = DEFAULT_BENCHMARK_PATH,
) -> Dict[str, Any]:
    """PHD-Net için ön işleme (QPC/DCT), saf model ve uçtan uca GPU benchmark'ını çalıştırır."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    input_shape = (1, 3, 224, 224)

    model = PHDNet().to(device=device, dtype=dtype).eval()
    total_params, trainable_params = count_parameters(model)

    rgb = torch.rand(input_shape, device=device, dtype=dtype)
    with torch.no_grad():
        qpc_pre, dct_pre = model.preprocessor(rgb)

    # 1. Ön işleme bileşenlerini ayrı ölçümleme (QPC, DCT, QPC+DCT)
    qpc_metrics = measure_callable_latency(lambda: model.preprocessor.qpc(rgb), device, warmup=warmup, runs=runs)
    dct_metrics = measure_callable_latency(
        lambda: model.preprocessor.block_dct(rgb), device, warmup=warmup, runs=runs
    )
    prep_metrics = measure_callable_latency(lambda: model.preprocessor(rgb), device, warmup=warmup, runs=runs)

    # 2. Sinir ağı ileri yayılım ölçümü (önceden hesaplanmış QPC ve DCT ile)
    model_only_metrics = measure_callable_latency(
        lambda: model(rgb, qpc_map=qpc_pre, dct_blocks=dct_pre, compute_mask=True),
        device,
        warmup=warmup,
        runs=runs,
    )
    model_nomask_metrics = measure_callable_latency(
        lambda: model(rgb, qpc_map=qpc_pre, dct_blocks=dct_pre, compute_mask=False),
        device,
        warmup=warmup,
        runs=runs,
    )

    # 3. Uçtan uca (QPC + DCT + Model) ölçümü
    end_to_end_metrics = measure_callable_latency(
        lambda: model(rgb, compute_mask=True), device, warmup=warmup, runs=runs
    )
    end_to_end_nomask_metrics = measure_callable_latency(
        lambda: model(rgb, compute_mask=False), device, warmup=warmup, runs=runs
    )

    target_params = 3_800_000
    target_latency_ms = 8.5
    target_fps = 115.0
    target_vram_mb = 1228.8

    phd_report: Dict[str, Any] = {
        "model_name": "PHD-Net (Phase-Harmonic Discrepancy Network)",
        "training_status": "EĞİTİLMEMİŞ (Mimari ve çıkarım kıyaslama doğrulama aşaması)",
        "localization_head_status": (
            "Zemin gerçekliği yerelleştirme maskeleri sağlanmadığında Dice/BCE yerelleştirme kaybı devre dışı bırakılır."
        ),
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "total_millions": round(total_params / 1e6, 4),
            "target_total": target_params,
            "diff_from_target": total_params - target_params,
            "diff_percent": round((total_params - target_params) / target_params * 100.0, 2),
        },
        "preprocessing_separate": {
            "qpc_only": qpc_metrics,
            "block_dct_8x8_only": dct_metrics,
            "qpc_plus_dct_combined": prep_metrics,
        },
        "model_forward_only": model_only_metrics,
        "model_forward_nomask": model_nomask_metrics,
        "end_to_end_with_preprocessing": end_to_end_metrics,
        "end_to_end_nomask": end_to_end_nomask_metrics,
        "target_comparison": {
            "targets": {
                "params": target_params,
                "latency_ms": target_latency_ms,
                "fps": target_fps,
                "max_vram_mb": target_vram_mb,
            },
            "model_only_vs_target": {
                "latency_meets_target": model_only_metrics["latency_median_ms"] <= target_latency_ms,
                "latency_delta_ms": round(model_only_metrics["latency_median_ms"] - target_latency_ms, 4),
                "fps_meets_target": model_only_metrics["fps_from_median"] >= target_fps,
                "fps_delta": round(model_only_metrics["fps_from_median"] - target_fps, 2),
                "vram_meets_target": model_only_metrics["max_memory_allocated_mb"] < target_vram_mb,
            },
            "end_to_end_vs_target": {
                "latency_meets_target": end_to_end_metrics["latency_median_ms"] <= target_latency_ms,
                "latency_delta_ms": round(end_to_end_metrics["latency_median_ms"] - target_latency_ms, 4),
                "fps_meets_target": end_to_end_metrics["fps_from_median"] >= target_fps,
                "fps_delta": round(end_to_end_metrics["fps_from_median"] - target_fps, 2),
                "vram_meets_target": end_to_end_metrics["max_memory_allocated_mb"] < target_vram_mb,
            },
        },
    }

    env_info = collect_environment_info(device, input_shape, dtype, warmup=warmup, runs=runs)
    update_benchmark_json("phd_net", phd_report, env_info, json_path=json_path)
    return phd_report


if __name__ == "__main__":
    print("=== phasefreq_net.py (PHD-Net) Model Doğrulama ve Benchmark ===")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {dev}")

    # 1. Çok görevli kayıp ve geri yayılım doğrulaması (Flaw 4 test: mask_targets=None iken dfsc_28 gradyanı)
    net_test = PHDNet().to(dev)
    criterion = PHDNetMultiTaskLoss(lambda_dice=1.0, lambda_ccm=0.5)
    sample_rgb = torch.rand(4, 3, 224, 224, device=dev)
    sample_labels = torch.tensor([0, 0, 1, 1], device=dev)

    # compute_mask=False ile test
    out_dict_nomask = net_test(sample_rgb, compute_mask=False)
    assert out_dict_nomask["mask_logits"] is None, "compute_mask=False iken mask_logits None olmalıdır!"

    # mask_targets=None iken geri yayılım
    out_dict = net_test(sample_rgb, compute_mask=True)
    loss_dict = criterion(out_dict, sample_labels, mask_targets=None)
    loss_dict["total_loss"].backward()

    assert net_test.dfsc_28.fuse[0].weight.grad is not None, "Flaw 4 Hatası: dfsc_28 gradyan almadı!"
    grad_norm_dfsc28 = float(net_test.dfsc_28.fuse[0].weight.grad.abs().sum().item())
    assert grad_norm_dfsc28 > 0.0, "Flaw 4 Hatası: dfsc_28 gradyan normu 0!"
    print(
        f"[OK] İleri/Geri yayılım: logits={tuple(out_dict['logits'].shape)}, "
        f"embedding={tuple(out_dict['embedding'].shape)}, mask_logits={tuple(out_dict['mask_logits'].shape)}"
    )
    print(
        f"[OK] Kayıplar (maskesiz mod): BCE={loss_dict['bce_loss']:.4f}, "
        f"L_CCM={loss_dict['ccm_loss']:.4f}, Dice/Loc={loss_dict['dice_loss']} (dice_evaluated={loss_dict['dice_evaluated']})"
    )
    print(f"[OK] Flaw 4 Doğrulandı: mask_targets=None iken dfsc_28 gradyan normu: {grad_norm_dfsc28:.6f} > 0")

    # 2. Kısmi Maske Denetimi Doğrulaması (has_mask=[True, False, True, False])
    net_test.zero_grad()
    sample_masks = torch.zeros(4, 1, 224, 224, device=dev)
    sample_masks[0, :, 50:150, 50:150] = 1.0
    sample_masks[2, :, 60:160, 60:160] = 1.0
    has_mask_test = torch.tensor([True, False, True, False], device=dev)

    out_masked = net_test(sample_rgb, compute_mask=True)
    loss_masked = criterion(out_masked, sample_labels, mask_targets=sample_masks, has_mask=has_mask_test)
    loss_masked["total_loss"].backward()
    assert loss_masked["dice_evaluated"] is True, "has_mask True örnekler varken dice_evaluated True olmalı!"
    assert loss_masked["dice_loss"] is not None and loss_masked["dice_loss"] > 0, "Kısmi maske kaybı hesaplanmalı!"
    print(
        f"[OK] Kısmi Maske Denetimi: BCE={loss_masked['bce_loss']:.4f}, "
        f"CCM={loss_masked['ccm_loss']:.4f}, Loc={loss_masked['dice_loss']:.4f} "
        f"(dice_evaluated={loss_masked['dice_evaluated']})"
    )

    # has_mask tümü False testi
    has_mask_all_false = torch.tensor([False, False, False, False], device=dev)
    loss_all_false = criterion(out_masked, sample_labels, mask_targets=sample_masks, has_mask=has_mask_all_false)
    assert loss_all_false["dice_evaluated"] is False, "has_mask tümü False iken dice_evaluated False olmalı!"
    assert loss_all_false["dice_loss"] is None, "has_mask tümü False iken dice_loss None olmalı!"
    print("[OK] Kısmi Maske Tümü False Doğrulandı (Dice devre dışı bırakıldı).")

    # 3. Ablasyon Modları Doğrulaması
    print("\n--- Ablasyon Modları Doğrulaması ---")
    ablation_modes = ["full", "no_qpc", "no_lfg", "no_dfsc", "classical_freq"]
    for mode in ablation_modes:
        abl_net = PHDNet(ablation_mode=mode).to(dev)
        abl_out = abl_net(sample_rgb[:2])
        abl_loss = criterion(abl_out, sample_labels[:2])
        abl_loss["total_loss"].backward()
        tot_p, _ = count_parameters(abl_net)
        print(f"[OK] Mod: '{mode:14s}' | Param={tot_p:,} | logits={tuple(abl_out['logits'].shape)} | Kayıp={abl_loss['total_loss'].item():.4f}")
        del abl_net

    del net_test, sample_rgb, sample_masks, out_dict, out_masked
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # 4. GPU Benchmark (batch=1, warmup=50, runs=250)
    print("\n--- GPU Benchmark Başlatılıyor ---")
    report = run_phdnet_benchmark(warmup=50, runs=250, json_path=DEFAULT_BENCHMARK_PATH)
    p = report["parameters"]
    prep = report["preprocessing_separate"]
    m_only = report["model_forward_only"]
    e2e = report["end_to_end_with_preprocessing"]

    print("\n--- PHD-Net Benchmark Özeti (Batch=1, 224x224, FP32) ---")
    print(
        f"Parametre Sayısı        : {p['total']:,} ({p['total_millions']:.3f}M) | "
        f"Hedef: 3.800M | Fark: {p['diff_from_target']:+,} ({p['diff_percent']:+.2f}%)"
    )
    print(
        f"Ön İşleme (QPC ayrı)    : medyan={prep['qpc_only']['latency_median_ms']:.3f} ms, "
        f"p95={prep['qpc_only']['latency_p95_ms']:.3f} ms"
    )
    print(
        f"Ön İşleme (8x8 DCT ayrı): medyan={prep['block_dct_8x8_only']['latency_median_ms']:.3f} ms, "
        f"p95={prep['block_dct_8x8_only']['latency_p95_ms']:.3f} ms"
    )
    print(
        f"Ön İşleme (QPC+DCT top.): medyan={prep['qpc_plus_dct_combined']['latency_median_ms']:.3f} ms, "
        f"p95={prep['qpc_plus_dct_combined']['latency_p95_ms']:.3f} ms"
    )
    print(
        f"Yalnızca Model İleri    : medyan={m_only['latency_median_ms']:.3f} ms, "
        f"p95={m_only['latency_p95_ms']:.3f} ms | {m_only['fps_from_median']:.1f} FPS | "
        f"Tepe VRAM={m_only['max_memory_allocated_mb']:.2f} MB"
    )
    print(
        f"Uçtan Uca (Önİşl.+Model): medyan={e2e['latency_median_ms']:.3f} ms, "
        f"p95={e2e['latency_p95_ms']:.3f} ms | {e2e['fps_from_median']:.1f} FPS | "
        f"Tepe VRAM={e2e['max_memory_allocated_mb']:.2f} MB"
    )
    print(f"JSON kaydedildi         : {DEFAULT_BENCHMARK_PATH}")
