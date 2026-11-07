"""PHD-Net (Phase-Harmonic Discrepancy Network) İki Dallı Mimari, Kayıplar ve GPU Benchmark.

Mimari Özeti:
1. Dal 1 — Uzamsal-Faz Kolu (`Spatial-Phase Branch`):
   - `Stem` (224x224 -> 112x112)
   - `PhaseAttentionGate` (QPC haritası ile uzamsal-faz dikkat kapılaması)
   - Hafif çok-ölçekli derinlemesine ayrılabilir evrişim omurgası (EdgeNeXt / RepVGG esinli bloklar:
     112x112 -> 56x56 -> 28x28 -> 14x14)
2. Dal 2 — Sıkıştırma-İnvaryant Harmonik Kolu (`Harmonic Branch`):
   - `BlockDCT8x8` -> `HarmonicEnergyRatio` (HER) -> `LearnableFreqGate` (LFG: 28x28 ve 14x14)
3. Dinamik Frekans-Uzamsal Kalibrasyon (`DFSC` Füzyon):
   - 28x28 ve 14x14 ölçeklerinde çift-yönlü çapraz kapılama (cross-gating) ile uzamsal ve
     harmonik özniteliklerin dinamik kalibrasyonu.
4. Çok Görevli Çıktı Başlıkları:
   - Başlık 1: İkili sınıflandırma logiti (`logits`: `(B, 1)`) + L2-normalize kontrastif gömme (`z_ccm`: `(B, 128)`)
   - Başlık 2: Piksel düzeyinde lokalizasyon maskesi logiti (`mask_logits`: `(B, 1, 224, 224)`)
     Not: Veri kümesinde yerelleştirme maskeleri bulunmadığında model ikili sınıflandırma ve
     kontrastif başlıklar üzerinden çalıştırılabilir; piksel düzeyinde zemin gerçekliği maskeleri
     sağlandığında yerelleştirme başlığı ve Dice kaybı devreye alınır.

Kontrastif Kayıp Tanımı (`CrossManipulationContrastiveLoss` — `L_CCM`):
   Birim hiperküre `S^{d-1}` üzerindeki L2-normalize gömmeler `{z_i}_{i=1}^B` ve etiketler
   `y_i in {0 (gerçek), 1 (sahte)}` için:
     - Gerçek yüz kümeleme terimi (`L_real`):
       `L_real = (1 / |P_real|) * sum_{(i,j) in P_real} (1 - <z_i, z_j>)`
       (Gerçek yüzleri gizil uzayda tek bir yoğun hiperküresel kümede toplar.)
     - Gerçek-Sahte marjlı itme terimi (`L_sep`):
       `L_sep = (1 / (|I_real| * |I_fake|)) * sum_{i in I_real, k in I_fake} max(0, <z_i, z_k> - m)^2`
       (Manipüle yüzleri gerçek kümeden `m` kosinüs marjının dışına iter; farklı sahte üretim
       yöntemlerini tek kümede olmaya zorlamaz.)
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

from learnable_freq_gate import BlockDCT8x8, LearnableFreqGate
from phase_congruency import PhaseAttentionGate, QuaternionPhaseCongruency

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

    def __init__(self, spatial_dim: int, freq_dim: int, out_dim: int) -> None:
        super().__init__()
        self.freq_align = nn.Sequential(
            nn.Conv2d(freq_dim, spatial_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(spatial_dim),
            nn.SiLU(inplace=True),
        )
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

    def forward(self, f_spatial: torch.Tensor, f_freq: torch.Tensor) -> torch.Tensor:
        freq_proj = self.freq_align(f_freq)
        cal_freq = freq_proj * (1.0 + self.s_to_f_gate(f_spatial))
        cal_spat = f_spatial * (1.0 + self.f_to_s_gate(freq_proj))
        return self.fuse(torch.cat([cal_spat, cal_freq], dim=1))


class LocalizationHead(nn.Module):
    """Çok ölçekli adli lokalizasyon ısı haritası çözücüsü (14x14 + 28x28 -> 224x224 maske logiti)."""

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
            nn.Conv2d(24, 1, kernel_size=1, bias=True),
        )

    def forward(self, fused_14: torch.Tensor, fused_28: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(self.up_14_to_28(fused_14), size=fused_28.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse_28(torch.cat([x, fused_28], dim=1))
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.decode_56(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        logits_112 = self.decode_112(x)
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
    """PHD-Net (Phase-Harmonic Discrepancy Network) ana modeli."""

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
    ) -> None:
        super().__init__()
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
        self.phase_gate = PhaseAttentionGate(spatial_channels=stem_dim, qpc_channels=3, hidden_channels=32)

        d1, d2, d3 = stage_dims
        n1, n2, n3 = stage_depths
        self.stage1_56 = DownsampleStage(stem_dim, d1, depth=n1, expand_ratio=expand_ratio)
        self.stage2_28 = DownsampleStage(d1, d2, depth=n2, expand_ratio=expand_ratio)
        self.stage3_14 = DownsampleStage(d2, d3, depth=n3, expand_ratio=expand_ratio)

        # 2. Dal: Sıkıştırma-İnvaryant Harmonik Kolu (DCT -> HER -> LFG)
        self.freq_branch = LearnableFreqGate(
            in_channels=3,
            out_channels_28=96,
            out_channels_14=160,
            low_max=low_max,
            mid_max=mid_max,
        )

        # Dinamik Frekans-Uzamsal Kalibrasyon (DFSC)
        self.dfsc_28 = DynamicFreqSpatialCalibration(spatial_dim=d2, freq_dim=96, out_dim=160)
        self.dfsc_14 = DynamicFreqSpatialCalibration(spatial_dim=d3, freq_dim=160, out_dim=352)

        # Başlık 1: Sınıflandırma + Kontrastif Gömme (L_CCM)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(352, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, embed_dim, bias=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(352, 128),
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
    ) -> Dict[str, torch.Tensor]:
        """İleri yayılım.

        Ön işleme (`qpc_map`, `dct_blocks`) dışarıdan verildiğinde yalnızca sinir ağı katmanlarını çalıştırır;
        verilmediğinde `self.preprocessor` ile otomatik hesaplar.
        """
        if qpc_map is None or dct_blocks is None:
            auto_qpc, auto_dct = self.preprocessor(rgb)
            if qpc_map is None:
                qpc_map = auto_qpc
            if dct_blocks is None:
                dct_blocks = auto_dct

        # Uzamsal-Faz Kolu
        s_112 = self.stem(rgb)
        s_112_gated, phase_attn = self.phase_gate(s_112, qpc_map)
        s_56 = self.stage1_56(s_112_gated)
        s_28 = self.stage2_28(s_56)
        s_14 = self.stage3_14(s_28)

        # Harmonik Kol
        f_28, f_14, her_stats = self.freq_branch(dct_blocks=dct_blocks)

        # DFSC Füzyon
        fused_28 = self.dfsc_28(s_28, f_28)
        fused_14 = self.dfsc_14(s_14, f_14)

        # Başlık 1: Sınıflandırma + Kontrastif Gömme
        pooled = self.global_pool(fused_14).flatten(1)
        logits = self.classifier(pooled)
        z_ccm = F.normalize(self.contrastive_proj(pooled), p=2, dim=-1)

        # Başlık 2: Lokalizasyon Maskesi Logiti (B, 1, H, W)
        mask_logits = self.localization_head(fused_14, fused_28, out_hw=rgb.shape[-2:])

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

    Tanım:
        L2-normalize gömmeler `z_i in S^{d-1}` ve ikili etiketler `y_i in {0: gerçek, 1: sahte}` için:
        1) Gerçek yüz sıkılık terimi (`L_real`):
           `L_real = mean_{i != j, y_i = y_j = 0} (1 - z_i^T z_j)`
           Gerçek yüz temsillerini gizil uzayda tek bir yoğun kümede toplar.
        2) Gerçek-Sahte marjlı ayrıştırma terimi (`L_sep`):
           `L_sep = mean_{y_i = 0, y_k = 1} max(0, z_i^T z_k - margin)^2`
           Farklı manipülasyon yöntemlerinden gelen sahte yüzleri gerçek kümeden `margin` kosinüs
           sınırının dışına (dış hiperküreye) iter.
        `L_CCM = L_real + L_sep`.
    """

    def __init__(self, margin: float = 0.20) -> None:
        super().__init__()
        self.margin = margin

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
            loss_real = (1.0 - sim_real[off_diag]).mean()

        loss_sep = z.new_tensor(0.0)
        if real_idx.numel() >= 1 and fake_idx.numel() >= 1:
            sim_rf = z[real_idx] @ z[fake_idx].t()
            loss_sep = F.relu(sim_rf - self.margin).square().mean()

        return loss_real + loss_sep


class PHDNetMultiTaskLoss(nn.Module):
    """PHD-Net çok görevli kayıp sarmalayıcısı: `L_BCE + lambda_1 * L_Dice + lambda_2 * L_CCM`.

    `mask_targets=None` verildiğinde `L_Dice` hesaplanmaz ve çıktıda `dice_evaluated=False` olarak raporlanır.
    """

    def __init__(self, lambda_dice: float = 1.0, lambda_ccm: float = 0.5, ccm_margin: float = 0.20) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.ccm = CrossManipulationContrastiveLoss(margin=ccm_margin)
        self.lambda_dice = lambda_dice
        self.lambda_ccm = lambda_ccm

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        y_float = labels.float().view(-1, 1)
        l_bce = self.bce(outputs["logits"], y_float)
        l_ccm = self.ccm(outputs["embedding"], labels.view(-1))
        total = l_bce + self.lambda_ccm * l_ccm

        if mask_targets is not None:
            l_dice = self.dice(outputs["mask_logits"], mask_targets)
            total = total + self.lambda_dice * l_dice
            dice_val: Optional[float] = float(l_dice.detach().item())
            dice_eval = True
        else:
            l_dice = None
            dice_val = None
            dice_eval = False

        return {
            "total_loss": total,
            "bce_loss": float(l_bce.detach().item()),
            "ccm_loss": float(l_ccm.detach().item()),
            "dice_loss": dice_val,
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
        lambda: model(rgb, qpc_map=qpc_pre, dct_blocks=dct_pre),
        device,
        warmup=warmup,
        runs=runs,
    )

    # 3. Uçtan uca (QPC + DCT + Model) ölçümü
    end_to_end_metrics = measure_callable_latency(lambda: model(rgb), device, warmup=warmup, runs=runs)

    target_params = 3_800_000
    target_latency_ms = 8.5
    target_fps = 115.0
    target_vram_mb = 1228.8

    phd_report: Dict[str, Any] = {
        "model_name": "PHD-Net (Phase-Harmonic Discrepancy Network)",
        "training_status": "EĞİTİLMEMİŞ (Mimari ve çıkarım kıyaslama doğrulama aşaması)",
        "localization_head_status": (
            "Zemin gerçekliği yerelleştirme maskeleri sağlanmadığında Dice kaybı devre dışı bırakılır."
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
        "end_to_end_with_preprocessing": end_to_end_metrics,
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

    # 1. Çok görevli kayıp ve geri yayılım doğrulaması
    net_test = PHDNet().to(dev)
    criterion = PHDNetMultiTaskLoss(lambda_dice=1.0, lambda_ccm=0.5)
    sample_rgb = torch.rand(4, 3, 224, 224, device=dev)
    sample_labels = torch.tensor([0, 0, 1, 1], device=dev)
    out_dict = net_test(sample_rgb)
    loss_dict = criterion(out_dict, sample_labels, mask_targets=None)
    loss_dict["total_loss"].backward()
    print(
        f"[OK] İleri/Geri yayılım: logits={tuple(out_dict['logits'].shape)}, "
        f"embedding={tuple(out_dict['embedding'].shape)}, mask_logits={tuple(out_dict['mask_logits'].shape)}"
    )
    print(
        f"[OK] Kayıplar (maskesiz mod): BCE={loss_dict['bce_loss']:.4f}, "
        f"L_CCM={loss_dict['ccm_loss']:.4f}, Dice={loss_dict['dice_loss']} (dice_evaluated={loss_dict['dice_evaluated']})"
    )
    del net_test, sample_rgb, out_dict
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # 2. GPU Benchmark (batch=1, warmup=50, runs=250)
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
