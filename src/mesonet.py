"""Meso-4 (Afchar et al., 2018) PyTorch Uygulaması ve Benchmark Modülü.

Kaynak:
    D. Afchar, V. Nozick, J. Yamagishi, I. Echizen (2018),
    "MesoNet: a Compact Facial Video Forgery Detection Network",
    IEEE International Workshop on Information Forensics and Security (WIFS), s. 1-7.

Notlar:
    - Orijinal makale 256x256 girdi kullanır (`flat_dim = 16 * 8 * 8 = 1024`, 27,977 parametre).
    - PHD-Net ön işleme hattıyla (224x224 kırpılmış yüzler) birebir adil karşılaştırma için
      varsayılan `image_size=224` (`flat_dim = 16 * 7 * 7 = 784`, 24,137 parametre) desteklenir;
      `image_size=256` da parametrik olarak seçilebilir.
    - Model mimarisi baseline kıyaslama ve çıkarım performansı (gecikme / bellek) analizi için eklenmiştir;
      ağırlık eğitimi sonrasında sınıflandırma performansı ölçülür.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from phasefreq_net import (
    DEFAULT_BENCHMARK_PATH,
    collect_environment_info,
    count_parameters,
    measure_callable_latency,
    update_benchmark_json,
)


class Meso4(nn.Module):
    """Afchar et al. (2018) Meso-4 mesoskopik CNN mimarisi."""

    def __init__(self, image_size: int = 224, num_classes: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        if image_size % 32 != 0:
            raise ValueError(f"image_size 32'nin tam katı olmalıdır (verilen={image_size}).")
        self.image_size = image_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

        feat_hw = image_size // 32
        flat_dim = 16 * feat_hw * feat_hw
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(flat_dim, 16),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.classifier(x)


def run_meso4_benchmark(
    image_size: int = 224,
    warmup: int = 50,
    runs: int = 250,
    json_path: Path = DEFAULT_BENCHMARK_PATH,
) -> Dict[str, Any]:
    """Meso-4 için PHD-Net ile aynı GPU benchmark protokolünü çalıştırır ve JSON'a kaydeder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    input_shape = (1, 3, image_size, image_size)

    model = Meso4(image_size=image_size).to(device=device, dtype=dtype).eval()
    total_params, trainable_params = count_parameters(model)

    orig_256_params, _ = count_parameters(Meso4(image_size=256))

    x = torch.rand(input_shape, device=device, dtype=dtype)
    metrics = measure_callable_latency(lambda: model(x), device, warmup=warmup, runs=runs)

    meso_report: Dict[str, Any] = {
        "model_name": "Meso-4 (Afchar et al., WIFS 2018)",
        "training_status": "EĞİTİLMEMİŞ (Baseline mimari hız ve VRAM analizi doğrulaması)",
        "input_resolution": f"{image_size}x{image_size}",
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "paper_256x256_reference_total": orig_256_params,
        },
        "model_forward_only": metrics,
    }

    env_info = collect_environment_info(device, input_shape, dtype, warmup=warmup, runs=runs)
    update_benchmark_json("meso4", meso_report, env_info, json_path=json_path)
    return meso_report


if __name__ == "__main__":
    print("=== mesonet.py (Meso-4 Baseline) Model Doğrulama ve Benchmark ===")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. İleri / geri yayılım doğrulaması
    m_test = Meso4(image_size=224).to(dev)
    x_test = torch.rand(4, 3, 224, 224, device=dev)
    y_test = torch.tensor([[0.0], [1.0], [0.0], [1.0]], device=dev)
    logits = m_test(x_test)
    loss = nn.BCEWithLogitsLoss()(logits, y_test)
    loss.backward()
    print(f"[OK] Meso-4 ileri/geri yayılım: çıktı={tuple(logits.shape)} | BCE={loss.item():.4f}")
    del m_test, x_test, logits
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # 2. GPU Benchmark
    rep = run_meso4_benchmark(image_size=224, warmup=50, runs=250, json_path=DEFAULT_BENCHMARK_PATH)
    m = rep["model_forward_only"]
    p = rep["parameters"]
    print("\n--- Meso-4 Benchmark Özeti (Batch=1, 224x224, FP32) ---")
    print(
        f"Parametre Sayısı        : {p['total']:,} (224x224) | "
        f"Orijinal 256x256 referans: {p['paper_256x256_reference_total']:,}"
    )
    print(
        f"Model İleri Gecikme     : medyan={m['latency_median_ms']:.3f} ms, "
        f"p95={m['latency_p95_ms']:.3f} ms | {m['fps_from_median']:.1f} FPS | "
        f"Tepe VRAM={m['max_memory_allocated_mb']:.2f} MB"
    )
    print(f"JSON güncellendi        : {DEFAULT_BENCHMARK_PATH}")
