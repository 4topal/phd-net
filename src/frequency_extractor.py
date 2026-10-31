"""Klasik (Öğrenilmeyen) 2B FFT ve Blok DCT Frekans Öznitelik Çıkarıcı (Ablasyon Referansı).

Amaç:
    PHD-Net içindeki öğrenilebilir `LearnableFreqGate` (LFG) ve `HER` modüllerinin katkısını
    ablasyon deneylerinde sınamak için öğrenilebilir parametre içermeyen klasik frekans
    özniteliklerini çıkarır.

Referans Yöntemler:
    1. Azimutal Ortalamalı 2B FFT Güç Spektrumu (1B Radyal Profil):
       R. Durall, M. Keuper, J. Keuper (2020), "Watch your Up-Convolution: CNN Based Generative
       Deep Neural Networks are Failing to Reproduce Spectral Distributions", CVPR 2020.
    2. Öğrenilmeyen 8x8 Blok DCT Alt-Bant İstatistikleri:
       J. Frank et al. (2020), "Leveraging Frequency Analysis for Deep Fake Image Recognition",
       ICML 2020.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from learnable_freq_gate import HarmonicEnergyRatio


class ClassicalFrequencyExtractor(nn.Module):
    """Öğrenilebilir parametre içermeyen klasik FFT + 8x8 DCT öznitelik çıkarıcı."""

    def __init__(
        self,
        n_radial_bins: int = 64,
        low_max: int = 3,
        mid_max: int = 8,
        include_dc: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_radial_bins = n_radial_bins
        self.eps = eps
        self.her_calc = HarmonicEnergyRatio(
            low_max=low_max, mid_max=mid_max, include_dc=include_dc, eps=eps
        )
        self._cached_bin_idx: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_radial_bins(self, h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = f"{h}x{w}_{device}"
        if key not in self._cached_bin_idx:
            cy, cx = h // 2, w // 2
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32) - cy,
                torch.arange(w, device=device, dtype=torch.float32) - cx,
                indexing="ij",
            )
            max_r = float(np.hypot(cy, cx))
            norm_r = torch.sqrt(xx.square() + yy.square()) / max(max_r, 1.0)
            bin_idx = torch.clamp((norm_r * self.n_radial_bins).long(), 0, self.n_radial_bins - 1).view(-1)
            counts = torch.bincount(bin_idx, minlength=self.n_radial_bins).float().clamp(min=1.0)
            self._cached_bin_idx[key] = (bin_idx, counts)
        return self._cached_bin_idx[key]

    @torch.no_grad()
    def extract_fft_azimuthal_profile(self, rgb: torch.Tensor) -> torch.Tensor:
        """Girdi `(B, 3, H, W)` -> Azimutal ortalamalı normalize log-FFT güç profili `(B, n_radial_bins)`."""
        gray = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]  # (B, H, W)
        b, h, w = gray.shape
        fft2 = torch.fft.fftshift(torch.fft.fft2(gray.float()), dim=(-2, -1))
        log_mag = torch.log1p(fft2.abs()).view(b, -1)  # (B, H*W)

        bin_idx, counts = self._get_radial_bins(h, w, rgb.device)
        profile = torch.zeros(b, self.n_radial_bins, device=rgb.device, dtype=torch.float32)
        profile.scatter_add_(1, bin_idx.unsqueeze(0).expand(b, -1), log_mag)
        profile = profile / counts.unsqueeze(0)

        # DC bin'e göre bağıl normalizasyon (aydınlatma ölçeğinden bağımsızlık)
        profile = profile / (profile[:, 0:1] + self.eps)
        return profile

    @torch.no_grad()
    def extract_dct_band_statistics(self, rgb: torch.Tensor) -> torch.Tensor:
        """Girdi `(B, 3, H, W)` -> Öğrenilmeyen 8x8 DCT alt-bant özet vektörü `(B, 27)`.

        Kanal başına (R, G, B -> 3 x 9 = 27 boyut):
          - log1p(low_energy) ort. ve std (2)
          - log1p(mid_energy) ort. ve std (2)
          - log1p(high_energy) ort. ve std (2)
          - blok HER ort. ve std (2)
          - küresel yüksek/düşük enerji oranı HFER (1)
        """
        stats = self.her_calc(rgb.float())
        low_l = torch.log1p(stats["low_energy"])
        mid_l = torch.log1p(stats["mid_energy"])
        high_l = torch.log1p(stats["high_energy"])
        her_m = stats["her_map"]
        hfer = stats["high_energy"].sum(dim=(-2, -1)) / (stats["low_energy"].sum(dim=(-2, -1)) + self.eps)

        feats = torch.cat(
            [
                low_l.mean(dim=(-2, -1)),
                low_l.std(dim=(-2, -1)),
                mid_l.mean(dim=(-2, -1)),
                mid_l.std(dim=(-2, -1)),
                high_l.mean(dim=(-2, -1)),
                high_l.std(dim=(-2, -1)),
                her_m.mean(dim=(-2, -1)),
                her_m.std(dim=(-2, -1)),
                hfer,
            ],
            dim=1,
        )  # (B, 27)
        return feats

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Hem ayrı alt vektörleri hem de birleştirilmiş `(B, n_radial_bins + 27)` ablasyon vektörünü döndürür."""
        fft_prof = self.extract_fft_azimuthal_profile(rgb)
        dct_stat = self.extract_dct_band_statistics(rgb)
        combined = torch.cat([fft_prof, dct_stat], dim=1)
        return {
            "fft_azimuthal_profile": fft_prof,
            "dct_band_stats": dct_stat,
            "combined_features": combined,
        }


if __name__ == "__main__":
    print("=== frequency_extractor.py Kendi Kendini Test Modu ===")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ClassicalFrequencyExtractor(n_radial_bins=64).to(dev)

    kod_root = Path(__file__).resolve().parents[1]
    real_crops = sorted((kod_root / "data" / "processed_faces" / "celeb_real").rglob("*.png"))
    fake_crops = sorted((kod_root / "data" / "processed_faces" / "celeb_synthesis").rglob("*.png"))

    if real_crops and fake_crops:
        imgs = []
        for p in [real_crops[0], fake_crops[0]]:
            bgr = cv2.imread(str(p))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            imgs.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
        batch = torch.stack(imgs, dim=0).to(dev)
        src_desc = f"gerçek ({real_crops[0].name}) ve sahte ({fake_crops[0].name}) kırpılmış yüzler"
    else:
        batch = torch.rand(2, 3, 224, 224, device=dev)
        src_desc = "sentetik 224x224 test tensörü"

    t0 = time.perf_counter()
    out = extractor(batch)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print(f"[OK] Girdi kaynağı             : {src_desc}")
    print(f"[OK] FFT azimutal profil boyutu: {tuple(out['fft_azimuthal_profile'].shape)}")
    print(f"[OK] Klasik DCT istatistikleri : {tuple(out['dct_band_stats'].shape)}")
    print(f"[OK] Birleşik ablasyon vektörü : {tuple(out['combined_features'].shape)} | süre={dt_ms:.2f} ms")
    print(
        f"[OK] Yüksek frekans FFT kuyruk ort. (son 16 bin): "
        f"örnek_0={out['fft_azimuthal_profile'][0, -16:].mean():.4f}, "
        f"örnek_1={out['fft_azimuthal_profile'][1, -16:].mean():.4f}"
    )
