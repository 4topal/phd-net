"""Log-Gabor Tabanlı Monokrom (Kovesi) ve Kuaterniyon Faz Uyumu (QPC) + Faz Dikkat Kapısı.

Matematiksel Kaynaklar ve Formülasyon:
1. Monokrom Faz Uyumu (`MonochromePhaseCongruency`):
   - Kaynak: P. Kovesi (1999), "Image Features From Phase Congruency", Videre: Journal of
     Computer Vision Research, 1(3), 1-26; P. Kovesi (2003), `phasecong3` algoritması.
   - Çok ölçekli ve çok yönlü 2B Log-Gabor kuadratür filtre bankası, hızlı alt-örneklemeli
     Rayleigh medyan gürültü kestirimi, frekans yayılım ağırlıklandırması ve faz sapma fonksiyonu:
     `DeltaPhi(x,y) = cos(phi - phi_mean) - |sin(phi - phi_mean)|`.

2. Kuaterniyon / Renk Faz Uyumu (`QuaternionPhaseCongruency` — QPC):
   - Temel renk temsili: T. A. Ell & S. J. Sangwine (2007), "Hypercomplex Fourier
     Transforms of Color Images", IEEE Transactions on Image Processing, 16(1), 22-35.
     Saf kuaterniyon görüntü alanı `q(x,y) = R(x,y)i + G(x,y)j + B(x,y)k` ve akromatik gri
     eksen `mu = (i + j + k) / sqrt(3)` etrafında dik (orthonormal) alt-uzaylar:
     `L = (R + G + B) / sqrt(3)`, `C1 = (R - G) / sqrt(2)`, `C2 = (R + G - 2*B) / sqrt(6)`.
   - Kovesi (1999) Log-Gabor kuadratür analizini Ell & Sangwine (2007) dik kuaterniyon uzayıyla
     birleştiren 3-kanallı QPC tensörü:
       * Kanal 0 (`qpc_joint`): Dik {L, C1, C2} uzayının Öklid kuadratik normu üzerinden birleşik
         faz uyumu. Paydada A_lum bölmesi olmadığı için 0/0 izolüminant tekilliği tamamen kalkar.
       * Kanal 1 (`qpc_chrom`): Dik {C1, C2} 2B krominans düzlemi dönel-değişmez Öklid normu faz uyumu.
         C1 + C2 skaler toplam körlüğü giderilmiştir.
       * Kanal 2 (`delta_phi_chrom`): R, G, B kanallarının cebirsel çift-açı (cos 2phi, sin 2phi)
         genlik-ağırlıklı eksenel faz dispersiyonu. Doğal 180° zıt-polariteli kenarlarda yanlış
         alarm üretmez; sentetik yüz birleştirme ve kaymalı sınırları hassas yakalar.

3. `PhaseAttentionGate`:
   - PHD-Net Uzamsal-Faz kolunda (`Stem` çıktısı 112x112) QPC haritasından tepe-koruyucu
     (`MaxPool2d`) alt-örnekleme ile uzamsal-kanal dikkat maskesi üreten öğrenilebilir PyTorch katmanı.
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
    radial_stack = torch.stack(radial_filters, dim=0)

    theta_sigma = math.pi / (n_orientations * d_theta_on_sigma)
    angular_filters = []
    for o in range(n_orientations):
        angle = o * math.pi / n_orientations
        ds = torch.sin(theta) * math.cos(angle) - torch.cos(theta) * math.sin(angle)
        dc = torch.cos(theta) * math.cos(angle) + torch.sin(theta) * math.sin(angle)
        dtheta = torch.atan2(ds, dc).abs()
        ang = 2.0 * torch.exp(-(dtheta.square()) / (2.0 * theta_sigma * theta_sigma))
        angular_filters.append(ang)
    angular_stack = torch.stack(angular_filters, dim=0)

    bank = radial_stack.unsqueeze(1) * angular_stack.unsqueeze(0)
    return bank


def _compute_frequency_spread_weight(
    amp_scale: torch.Tensor,
    cut_off: float = 0.5,
    gain: float = 10.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Ölçek bazlı genlikten Kovesi frekans yayılım ağırlığı W_o hesaplar.

    Args:
        amp_scale: `(B, n_scales, n_orientations, H, W)` veya `(B, C, n_scales, n_orientations, H, W)`
    Returns:
        weight: `(B, n_orientations, H, W)` veya `(B, C, n_orientations, H, W)`
    """
    scale_dim = -4
    n_scales = amp_scale.shape[scale_dim]
    sum_a = amp_scale.sum(dim=scale_dim)
    max_a = amp_scale.amax(dim=scale_dim)
    width_spread = (sum_a / (max_a + eps) - 1.0) / max(n_scales - 1, 1)
    return 1.0 / (1.0 + torch.exp(gain * (cut_off - width_spread)))


def _kovesi_channel_stats(
    even: torch.Tensor,
    odd: torch.Tensor,
    mult: float = 2.1,
    k_noise: float = 2.0,
    cut_off: float = 0.5,
    gain: float = 10.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kuadratür (even, odd) tepkilerinden Kovesi (1999) faz istatistiklerini hesaplar.

    Kusur 8 Çözümü: En ince ölçekteki (s=0) Rayleigh gürültü medyan kestiriminde 2x uzamsal
    alt-örnekleme (stride=2) kullanılarak GPU sıralama yükü 4 kat azaltılır.

    Args:
        even: `(B, C, n_scales, n_orientations, H, W)` veya `(B, n_scales, n_orientations, H, W)`
        odd: `even` ile aynı boyutta tek (odd / Hilbert) kuadratür tepkisi.
    Returns:
        (energy_thresh, weight, sum_a, amp, energy_orient, threshold) tensörleri:
        - energy_thresh: `relu(E_{c,o} - T_{c,o})` (B, C, n_orientations, H, W)
        - weight: `W_{c,o}` (B, C, n_orientations, H, W)
        - sum_a: `A_{c,o}` (B, C, n_orientations, H, W)
        - amp: `A_{c,s,o}` (B, C, n_scales, n_orientations, H, W)
        - energy_orient: `E_{c,o}` (B, C, n_orientations, H, W)
        - threshold: `T_{c,o}` (B, C, n_orientations, 1, 1)
    """
    is_5d = even.ndim == 5
    if is_5d:
        even = even.unsqueeze(1)
        odd = odd.unsqueeze(1)

    _, _, n_scales, _, _, _ = even.shape
    amp = torch.sqrt(even.square() + odd.square() + eps * eps)

    sum_e = even.sum(dim=2)
    sum_o = odd.sum(dim=2)
    sum_a = amp.sum(dim=2)

    mean_norm = torch.sqrt(sum_e.square() + sum_o.square() + eps * eps)
    mean_e = sum_e / mean_norm
    mean_o = sum_o / mean_norm

    dot = even * mean_e.unsqueeze(2) + odd * mean_o.unsqueeze(2)
    cross = (even * mean_o.unsqueeze(2) - odd * mean_e.unsqueeze(2)).abs()
    energy_per_scale = dot - cross
    energy_orient = energy_per_scale.sum(dim=2)

    # Kusur 8: s=0 ölçeğinde stride=2 alt-örneklemeli Rayleigh medyan kestirimi
    amp_s0 = amp[:, :, 0, :, ::2, ::2].flatten(start_dim=3)
    median_s0 = amp_s0.median(dim=-1).values  # (B, C, n_orientations)
    tau = median_s0 / math.sqrt(math.log(4.0))
    total_tau = tau * (1.0 - (1.0 / mult) ** n_scales) / (1.0 - (1.0 / mult))
    mean_noise = total_tau * math.sqrt(math.pi / 2.0)
    sigma_noise = total_tau * math.sqrt((4.0 - math.pi) / 2.0)
    threshold = (mean_noise + k_noise * sigma_noise).unsqueeze(-1).unsqueeze(-1)  # (B, C, n_orientations, 1, 1)

    energy_thresh = F.relu(energy_orient - threshold)
    weight = _compute_frequency_spread_weight(amp, cut_off=cut_off, gain=gain, eps=eps)

    if is_5d:
        return (
            energy_thresh.squeeze(1),
            weight.squeeze(1),
            sum_a.squeeze(1),
            amp.squeeze(1),
            energy_orient.squeeze(1),
            threshold.squeeze(1),
        )

    return energy_thresh, weight, sum_a, amp, energy_orient, threshold


def _kovesi_pc_from_quadrature(
    even: torch.Tensor,
    odd: torch.Tensor,
    mult: float = 2.1,
    k_noise: float = 2.0,
    cut_off: float = 0.5,
    gain: float = 10.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Kuadratür `(even, odd)` tepkilerinden Kovesi (1999) faz uyumunu hesaplar."""
    energy_thresh, weight, sum_a, _, _, _ = _kovesi_channel_stats(
        even, odd, mult=mult, k_noise=k_noise, cut_off=cut_off, gain=gain, eps=eps
    )
    orient_dim = -3
    weighted_energy = weight * energy_thresh
    pc = weighted_energy.sum(dim=orient_dim, keepdim=True) / (sum_a.sum(dim=orient_dim, keepdim=True) + eps)
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
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        elif x.shape[1] != 1:
            raise ValueError(f"Beklenen kanal sayısı 1 veya 3, alınan: {x.shape[1]}")

        _, _, h, w = x.shape
        bank = self._get_bank(h, w, x.device, x.dtype)
        img_fft = torch.fft.fft2(x)
        resp = torch.fft.ifft2(img_fft.unsqueeze(2) * bank.unsqueeze(0))
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
    """Kuaterniyon / Renk Faz Uyumu (QPC) hesaplayıcısı.

    Saf kuaterniyon renk alanı `q(x,y) = R*i + G*j + B*k` (Ell & Sangwine, 2007) üzerinde
    Kovesi (1999) Log-Gabor kuadratür filtre bankasını dik (orthonormal) (L, C1, C2) uzayında
    çalıştırarak 3-kanallı çıktı üretir:
      - Kanal 0 (`qpc_joint`): Dik {L, C1, C2} uzayının Öklid kuadratik normu üzerinden
        birleşik kuaterniyon faz uyumu `[0, 1]` (Kusur 1: A_lum tekilliği giderilmiştir).
      - Kanal 1 (`qpc_chrom`): Dik {C1, C2} 2B krominans düzlemi dönel-değişmez Öklid normu
        faz uyumu `[0, 1]` (Kusur 2: C1 + C2 skaler toplam körlüğü giderilmiştir).
      - Kanal 2 (`delta_phi_chrom`): Cebirsel çift-açı (cos 2phi, sin 2phi) genlik-ağırlıklı
        eksenel kromatik faz dispersiyonu `[0, 1]` (Kusur 3: 180° zıt-faz kenar hatası giderilmiştir).
    """

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
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Girdi `(B, 3, H, W)` RGB `[0, 1]` -> Çıktı `(B, 3, H, W)` QPC tensörü."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Beklenen girdi boyutu (B, 3, H, W), alınan: {tuple(rgb.shape)}")

        orig_dtype = rgb.dtype
        x = rgb.float()
        b, _, h, w = x.shape
        bank = self._get_bank(h, w, x.device, x.dtype)

        # Kusur 1 & 2: 2B FFT öncesinde RGB tensörünü dik (L, C1, C2) uzayına dönüştür
        # L  = (R + G + B) / sqrt(3)      (Parseval enerjisi korunumlu akromatik eksen)
        # C1 = (R - G) / sqrt(2)          (Kırmızı-Yeşil dik krominans ekseni)
        # C2 = (R + G - 2*B) / sqrt(6)    (Sarı-Mavi dik krominans ekseni)
        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        inv_sqrt6 = 1.0 / math.sqrt(6.0)

        l_ch = (x[:, 0] + x[:, 1] + x[:, 2]) * inv_sqrt3
        c1_ch = (x[:, 0] - x[:, 1]) * inv_sqrt2
        c2_ch = (x[:, 0] + x[:, 1] - 2.0 * x[:, 2]) * inv_sqrt6
        lc1c2 = torch.stack([l_ch, c1_ch, c2_ch], dim=1)  # (B, 3, H, W)

        # Tek bir 2B FFT ve IFFT ile (E_L, E_C1, E_C2) ve (O_L, O_C1, O_C2) kuadratür tepkileri
        lc1c2_fft = torch.fft.fft2(lc1c2)
        resp = torch.fft.ifft2(
            lc1c2_fft.unsqueeze(2).unsqueeze(3) * bank.unsqueeze(0).unsqueeze(0)
        )
        e_lc1c2 = resp.real  # (B, 3, n_scales, n_orientations, H, W)
        o_lc1c2 = resp.imag  # (B, 3, n_scales, n_orientations, H, W)

        # Her dik kanal {L, C1, C2} için Kovesi faz istatistikleri
        (
            energy_thresh,  # (B, 3, n_orientations, H, W)
            _,
            _,
            amp_lc1c2,      # (B, 3, n_scales, n_orientations, H, W)
            _,
            _,
        ) = _kovesi_channel_stats(
            e_lc1c2,
            o_lc1c2,
            mult=self.mult,
            k_noise=self.k_noise,
            cut_off=self.cut_off,
            gain=self.gain,
            eps=self.eps,
        )

        thresh_l = energy_thresh[:, 0]    # relu(E_L - T_L)
        thresh_c1 = energy_thresh[:, 1]   # relu(E_C1 - T_C1)
        thresh_c2 = energy_thresh[:, 2]   # relu(E_C2 - T_C2)

        # Kuaterniyon ve krominans yerel genlikleri (ölçek bazlı):
        # amp_q_scale = sqrt(A_L^2 + A_C1^2 + A_C2^2)
        amp_q_scale = torch.sqrt(
            amp_lc1c2[:, 0].square() + amp_lc1c2[:, 1].square() + amp_lc1c2[:, 2].square()
        )  # (B, n_scales, n_orientations, H, W)
        # amp_chrom_scale = sqrt(A_C1^2 + A_C2^2)
        amp_chrom_scale = torch.sqrt(
            amp_lc1c2[:, 1].square() + amp_lc1c2[:, 2].square()
        )  # (B, n_scales, n_orientations, H, W)

        w_joint_o = _compute_frequency_spread_weight(
            amp_q_scale, cut_off=self.cut_off, gain=self.gain, eps=self.eps
        )  # (B, n_orientations, H, W)
        w_chrom_o = _compute_frequency_spread_weight(
            amp_chrom_scale, cut_off=self.cut_off, gain=self.gain, eps=self.eps
        )  # (B, n_orientations, H, W)

        # Kanal 0: Birleşik Kuaterniyon Faz Uyumu (qpc_joint) — Kusur 1 Çözümü
        # num_joint_o = W_joint_o * sqrt(relu(E_L - T_L)^2 + relu(E_C1 - T_C1)^2 + relu(E_C2 - T_C2)^2)
        # Paydada A_lum bölmesi olmadığı için 0/0 izolüminant tekilliği tamamen kalkar.
        joint_energy_o = torch.sqrt(
            thresh_l.square() + thresh_c1.square() + thresh_c2.square()
        )
        num_joint = (w_joint_o * joint_energy_o).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        den_joint = amp_q_scale.sum(dim=(1, 2)).unsqueeze(1) + self.eps    # (B, 1, H, W)
        qpc_joint = (num_joint / den_joint).clamp(0.0, 1.0)

        # Kanal 1: 2B Krominans Düzlemi Faz Uyumu (qpc_chrom) — Kusur 2 Çözümü
        # num_chrom_o = W_chrom_o * sqrt(relu(E_C1 - T_C1)^2 + relu(E_C2 - T_C2)^2)
        # 2B krominans Öklid normu ile dönel-değişmez (tüm renk tonlarında tam faz)
        chrom_energy_o = torch.sqrt(thresh_c1.square() + thresh_c2.square())
        num_chrom = (w_chrom_o * chrom_energy_o).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        den_chrom = amp_chrom_scale.sum(dim=(1, 2)).unsqueeze(1) + self.eps  # (B, 1, H, W)
        qpc_chrom = (num_chrom / den_chrom).clamp(0.0, 1.0)

        # Kanal 2: Genlik-Ağırlıklı Çift-Açı Eksenel Kromatik Faz Dispersiyonu (delta_phi_chrom) — Kusur 3 Çözümü
        # Dik dönüşümün transpozuyla (sıfır ek FFT maliyeti):
        # R = L / sqrt(3) + C1 / sqrt(2) + C2 / sqrt(6)
        # G = L / sqrt(3) - C1 / sqrt(2) + C2 / sqrt(6)
        # B = L / sqrt(3) - 2 * C2 / sqrt(6)
        e_l = e_lc1c2[:, 0]
        e_c1 = e_lc1c2[:, 1]
        e_c2 = e_lc1c2[:, 2]

        e_r = e_l * inv_sqrt3 + e_c1 * inv_sqrt2 + e_c2 * inv_sqrt6
        e_g = e_l * inv_sqrt3 - e_c1 * inv_sqrt2 + e_c2 * inv_sqrt6
        e_b = e_l * inv_sqrt3 - 2.0 * e_c2 * inv_sqrt6
        e_rgb = torch.stack([e_r, e_g, e_b], dim=1)  # (B, 3, n_scales, n_orientations, H, W)

        o_l = o_lc1c2[:, 0]
        o_c1 = o_lc1c2[:, 1]
        o_c2 = o_lc1c2[:, 2]

        o_r = o_l * inv_sqrt3 + o_c1 * inv_sqrt2 + o_c2 * inv_sqrt6
        o_g = o_l * inv_sqrt3 - o_c1 * inv_sqrt2 + o_c2 * inv_sqrt6
        o_b = o_l * inv_sqrt3 - 2.0 * o_c2 * inv_sqrt6
        o_rgb = torch.stack([o_r, o_g, o_b], dim=1)  # (B, 3, n_scales, n_orientations, H, W)

        ch_energy = e_rgb.square() + o_rgb.square() + self.eps * self.eps  # (B, 3, n_scales, n_orientations, H, W)
        ch_amp = torch.sqrt(ch_energy)  # (B, 3, n_scales, n_orientations, H, W)

        # Cebirsel çift-açı (axial) birim fazörler: phi -> phi + pi altında tam değişmez
        cos2 = (e_rgb.square() - o_rgb.square()) / ch_energy
        sin2 = (2.0 * e_rgb * o_rgb) / ch_energy

        # Kanal genlik ağırlıkları: gürültülü/düşük genlikli kanalların fazı bozmasını önler
        w_c = ch_amp / (ch_amp.sum(dim=1, keepdim=True) + self.eps)
        mean_cos2 = (w_c * cos2).sum(dim=1)  # (B, n_scales, n_orientations, H, W)
        mean_sin2 = (w_c * sin2).sum(dim=1)  # (B, n_scales, n_orientations, H, W)

        mean_vec_len = torch.sqrt(mean_cos2.square() + mean_sin2.square() + self.eps * self.eps).clamp(0.0, 1.0)
        axial_dispersion = 1.0 - mean_vec_len  # (B, n_scales, n_orientations, H, W)

        # Krominans enerji kapısı ve genlik-ağırlıklı ölçek/yönelim ortalaması
        energy_gate = amp_chrom_scale / (amp_q_scale + 0.05)  # (B, n_scales, n_orientations, H, W)
        weighted_disp = axial_dispersion * energy_gate * amp_q_scale
        disp_num = weighted_disp.sum(dim=(1, 2))  # (B, H, W)
        disp_den = amp_q_scale.sum(dim=(1, 2)) + self.eps  # (B, H, W)
        delta_phi_chrom = (disp_num / disp_den).unsqueeze(1).clamp(0.0, 1.0)  # (B, 1, H, W)

        out = torch.cat([qpc_joint, qpc_chrom, delta_phi_chrom], dim=1)
        return out.to(orig_dtype)

    @staticmethod
    def summary_map(qpc_tensor: torch.Tensor) -> torch.Tensor:
        """Görselleştirme için 3-kanallı QPC çıktısını tek kanallı `(B, 1, H, W)` haritaya indirger."""
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

        Kusur 5 Çözümü: Bilinear enterpolasyon yerine tepe-koruyucu (peak-preserving)
        `F.max_pool2d` (veya `F.adaptive_max_pool2d`) kullanılarak keskin 1-piksel
        faz sırtları ve dispersiyon tepeleri sönümlenmeden korunur.

        Args:
            spatial_feat: `(B, C, H_s, W_s)` uzamsal stem özellikleri.
            qpc_map: `(B, 3, H, W)` QPC tensörü.
        Returns:
            `(gated_feat, attention_mask)` ikilisi; her ikisi de `(B, C, H_s, W_s)` boyutludur.
        """
        if qpc_map.shape[-2:] != spatial_feat.shape[-2:]:
            h_q, w_q = qpc_map.shape[-2:]
            h_s, w_s = spatial_feat.shape[-2:]
            if h_q == 2 * h_s and w_q == 2 * w_s:
                qpc_resized = F.max_pool2d(qpc_map, kernel_size=2, stride=2)
            else:
                qpc_resized = F.adaptive_max_pool2d(qpc_map, (h_s, w_s))
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
    print("=== phase_congruency.py Doğrulama ve Test Modu ===")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {dev}")

    h, w = 224, 224
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=dev),
        torch.linspace(-1.0, 1.0, w, device=dev),
        indexing="ij",
    )

    # --- Test (a): İzolüminant Renk Adım Sınırı (Kusur 1 & 2 Doğrulaması) ---
    # Sabit parlaklık: L = (R + G + B) / sqrt(3) her yerde sabit (0.2 + 0.8 + 0.5 = 1.5)
    # Sol yarı: R=0.2, G=0.8, B=0.5; Sağ yarı: R=0.8, G=0.2, B=0.5
    step_x = (xx >= 0.0).float()
    r_iso = 0.2 + 0.6 * step_x
    g_iso = 0.8 - 0.6 * step_x
    b_iso = torch.full_like(step_x, 0.5)
    iso_img = torch.stack([r_iso, g_iso, b_iso], dim=0).unsqueeze(0)  # (1, 3, 224, 224)

    qpc_calc = QuaternionPhaseCongruency().to(dev)
    qpc_iso = qpc_calc(iso_img)

    assert not torch.isnan(qpc_iso).any(), "HATA: İzolüminant sınırda QPC NaN üretti!"
    assert not torch.isinf(qpc_iso).any(), "HATA: İzolüminant sınırda QPC Inf üretti!"
    assert qpc_iso[0, 0].max().item() > 0.05, f"HATA: qpc_joint pozitif olmalı, alınan: {qpc_iso[0, 0].max().item()}"
    assert qpc_iso[0, 1].max().item() > 0.05, f"HATA: qpc_chrom pozitif olmalı, alınan: {qpc_iso[0, 1].max().item()}"
    print(
        f"[OK] Test (a) İzolüminant Sınır: NaN/Inf yok | "
        f"qpc_joint maks={qpc_iso[0, 0].max().item():.4f}, "
        f"qpc_chrom maks={qpc_iso[0, 1].max().item():.4f}"
    )

    # --- Test (b): Zıt Polariteli Hizalı Kenar vs. Kaymalı Kenar (Kusur 3 Doğrulaması) ---
    # Zıt-polariteli ancak uzamsal olarak tam hizalı sınır:
    # R: 0.2 -> 0.8 (artış), G: 0.8 -> 0.2 (azalış / 180° zıt faz), B: 0.2 -> 0.8
    step_center = (xx >= 0.0).float()
    opp_r = 0.2 + 0.6 * step_center
    opp_g = 0.8 - 0.6 * step_center
    opp_b = 0.2 + 0.6 * step_center
    img_opp_aligned = torch.stack([opp_r, opp_g, opp_b], dim=0)

    # Renk-kaymalı (misaligned) sınır: R 5 piksel sola, B 5 piksel sağa kaymış
    step_r_shifted = (xx >= -0.05).float()
    step_b_shifted = (xx >= 0.05).float()
    shift_r = 0.2 + 0.6 * step_r_shifted
    shift_g = 0.2 + 0.6 * step_center
    shift_b = 0.2 + 0.6 * step_b_shifted
    img_shifted = torch.stack([shift_r, shift_g, shift_b], dim=0)

    test_b_batch = torch.stack([img_opp_aligned, img_shifted], dim=0)  # (2, 3, 224, 224)
    qpc_test_b = qpc_calc(test_b_batch)

    disp_opp = float(qpc_test_b[0, 2].mean().item())
    disp_shifted = float(qpc_test_b[1, 2].mean().item())
    print(
        f"[OK] Test (b) Eksenel Faz Dispersiyonu: zıt-polariteli hizalı={disp_opp:.5f} < "
        f"kaymalı sınır={disp_shifted:.5f}"
    )
    assert disp_shifted > disp_opp, (
        f"HATA: Kaymalı sınırda dispersiyon ({disp_shifted:.5f}) zıt-polariteli hizalı kenardan "
        f"({disp_opp:.5f}) yüksek olmalıdır!"
    )

    # --- Test (c): PhaseAttentionGate Tepe Koruyucu ve İleri/Geri Yayılım (Kusur 5 Doğrulaması) ---
    gate = PhaseAttentionGate(spatial_channels=48, qpc_channels=3).to(dev)
    dummy_stem = torch.randn(2, 48, 112, 112, device=dev, requires_grad=True)
    gated_out, attn_mask = gate(dummy_stem, qpc_test_b)

    loss = gated_out.mean()
    loss.backward()

    assert dummy_stem.grad is not None and gate.alpha.grad is not None, "HATA: Gradyan akışı başarısız!"
    assert tuple(gated_out.shape) == (2, 48, 112, 112), f"HATA: Beklenmeyen gated boyutu: {gated_out.shape}"
    assert tuple(attn_mask.shape) == (2, 48, 112, 112), f"HATA: Beklenmeyen attn_mask boyutu: {attn_mask.shape}"
    print(
        f"[OK] Test (c) PhaseAttentionGate: Çıktı boyutu {tuple(gated_out.shape)} | "
        f"stem_grad_norm={dummy_stem.grad.norm().item():.4f}, "
        f"alpha_grad={gate.alpha.grad.item():.4f}"
    )

    # --- Test (d): Monokrom Kovesi PC Doğrulaması ---
    mono_pc = MonochromePhaseCongruency().to(dev)
    mono_out = mono_pc(test_b_batch)
    assert not torch.isnan(mono_out).any(), "HATA: Monokrom PC NaN üretti!"
    print(
        f"[OK] Test (d) Monokrom Kovesi PC: Çıktı boyutu {tuple(mono_out.shape)} | "
        f"min={mono_out.min():.4f}, max={mono_out.max():.4f}"
    )

    print("=== Tüm testler başarıyla geçti! ===")
