"""PHD-Net Akademik Görselleştirme ve Grafik Üretim Betiği.

Üretilen Grafikler:
1. `outputs/figures/training_loss_curves.png`:
   - Eğitim/Doğrulama Kaybı (Train Loss vs Val Loss)
   - Çoklu Görev Kayıpları (BCE, CCM, Dice)
   - Doğrulama Metrikleri (Video AUC, Kare AUC, F1)
   - Öğrenme Oranı (Cosine Annealing LR)
2. `outputs/figures/roc_curves.png`:
   - FF++ c23, c40, Birleşik, FaceShifter, DFD ve Celeb-DF v2 Ampirik Video-Level ROC Eğrileri
   - Ampirik EER ve Optimal Eşik Noktaları
3. `outputs/figures/her_qpc_scatter.png`:
   - FF++ Test Ayrımı Gerçek ve Sahte Yüzler İçin Gerçek Kromatik Faz Dispersiyonu (QPC)
     vs Harmonik Enerji Oranı (HER) Ampirik Saçılım Analizi
   - Grup Ağırlık Merkezleri (Ortalama & Medyan) ve LDA Ayrıştırma Karar Sınırı
"""

from __future__ import annotations

import json
import logging
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, roc_auc_score

try:
    from phase_congruency import QuaternionPhaseCongruency
    from learnable_freq_gate import HarmonicEnergyRatio
    from dataset import build_or_load_ffpp_splits, is_video_in_split
except ImportError:
    from src.phase_congruency import QuaternionPhaseCongruency
    from src.learnable_freq_gate import HarmonicEnergyRatio
    from src.dataset import build_or_load_ffpp_splits, is_video_in_split

from project_paths import PROJECT_ROOT

logger = logging.getLogger("phdnet.plot_figures")


# ==============================================================================
# Yardımcı: Yüz RGB Resmi Yükleyici (cv2 veya PIL ile)
# ==============================================================================

try:
    import cv2

    def load_face_rgb(p: Path, target_size: int = 224) -> Optional[np.ndarray]:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        if bgr.shape[:2] != (target_size, target_size):
            bgr = cv2.resize(bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

except ImportError:
    try:
        from PIL import Image

        def load_face_rgb(p: Path, target_size: int = 224) -> Optional[np.ndarray]:
            try:
                with Image.open(p) as img:
                    img_rgb = img.convert("RGB")
                    if img_rgb.size != (target_size, target_size):
                        img_rgb = img_rgb.resize((target_size, target_size), Image.Resampling.BILINEAR)
                    return np.array(img_rgb, dtype=np.uint8)
            except Exception:
                return None
    except ImportError:
        def load_face_rgb(p: Path, target_size: int = 224) -> Optional[np.ndarray]:
            return None


# ==============================================================================
# 1. Eğitim ve Kayıp (Loss) Eğrileri Çizimi
# ==============================================================================

def plot_training_curves(
    history_json_path: Path,
    output_path: Path,
) -> None:
    """Eğitim geçmişinden 4 panelli yüksek çözünürlüklü grafik üretir."""
    if not history_json_path.is_file():
        print(f"[UYARI] {history_json_path} bulunamadı, grafik çizilemedi.")
        return

    with open(history_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        print("[UYARI] Eğitim geçmişi boş!")
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    train_bce = [h.get("train_bce", 0.0) for h in history]
    train_ccm = [h.get("train_ccm", 0.0) for h in history]
    train_dice = [h.get("train_dice", 0.0) for h in history]

    video_auc = [h["video_metrics"]["auc"] for h in history]
    frame_auc = [h["frame_metrics"]["auc"] for h in history]
    video_f1 = [h["video_metrics"]["f1"] for h in history]
    lrs = [h.get("lr", 3e-4) for h in history]

    # Matplotlib stil ayarları
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Panel 1: Toplam Kayıp (Train vs Val)
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, "o-", color="#1f77b4", linewidth=2.2, markersize=6, label="Train Loss (Eğitim Kaybı)")
    ax1.plot(epochs, val_loss, "s--", color="#d62728", linewidth=2.2, markersize=6, label="Val Loss (Doğrulama Kaybı)")
    best_ep = int(np.argmin(val_loss))
    ax1.scatter([epochs[best_ep]], [val_loss[best_ep]], color="#2ca02c", s=120, zorder=5, label=f"En İyi Val Loss ({val_loss[best_ep]:.4f})")
    ax1.set_title("A) Toplam Kayıp (Total Loss)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_xticks(epochs)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Çoklu Görev Kayıpları
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_bce, "o-", color="#ff7f0e", linewidth=2, label=r"$\mathcal{L}_{\mathrm{BCE}}$ (Sınıflandırma)")
    ax2.plot(epochs, train_ccm, "^-", color="#9467bd", linewidth=2, label=r"$\mathcal{L}_{\mathrm{CCM}}$ (Kontrastif Ayrıştırma)")
    ax2.plot(epochs, train_dice, "d-", color="#2ca02c", linewidth=2, label=r"$\mathcal{L}_{\mathrm{Dice}}$ (Piksel Lokalizasyonu)")
    ax2.set_title(r"B) Çoklu Görev Kayıp Bileşenleri ($\mathcal{L}_{\mathrm{Total}}$)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Kayıp Değeri", fontsize=11)
    ax2.set_xticks(epochs)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Doğrulama Metrikleri (AUC & F1)
    ax3 = axes[1, 0]
    ax3.plot(epochs, video_auc, "o-", color="#2ca02c", linewidth=2.2, markersize=6, label="Video-AUC")
    ax3.plot(epochs, frame_auc, "v--", color="#17becf", linewidth=1.8, markersize=5, label="Kare (Frame)-AUC")
    ax3.plot(epochs, video_f1, "s-.", color="#e377c2", linewidth=1.8, markersize=5, label="Video F1 Skoru")
    best_auc_idx = int(np.argmax(video_auc))
    ax3.scatter([epochs[best_auc_idx]], [video_auc[best_auc_idx]], color="#d62728", s=130, zorder=5, label=f"Zirve Video-AUC: {video_auc[best_auc_idx]:.4f} (Epoch {epochs[best_auc_idx]})")
    ax3.set_title("C) Doğrulama Başarım Metrikleri (Validation Metrics)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Skor (0.0 - 1.0)", fontsize=11)
    ax3.set_xticks(epochs)
    ax3.set_ylim([0.45, 1.0])
    ax3.legend(loc="lower right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Panel 4: Öğrenme Oranı (Learning Rate)
    ax4 = axes[1, 1]
    ax4.plot(epochs, lrs, "o-", color="#8c564b", linewidth=2, markersize=6, label="Cosine Annealing LR")
    ax4.set_title("D) Öğrenme Oranı Zamanlayıcısı (LR Schedule)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Epoch", fontsize=11)
    ax4.set_ylabel("Öğrenme Oranı (LR)", fontsize=11)
    ax4.set_xticks(epochs)
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax4.legend(loc="upper right", frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(f"PHD-Net Model Eğitim Dinamikleri ve Metrik Geçmişi ({data.get('model', 'phdnet').upper()} - {data.get('ablation', 'full')})", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Kayıp ve metrik eğrisi kaydedildi: {output_path}")


# ==============================================================================
# 2. Çoklu Protokol Ampirik ROC Eğrileri Çizimi
# ==============================================================================

def plot_all_roc_curves(
    eval_report_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    roc_data_path: Optional[Path] = None,
) -> None:
    """Tüm protokollerin ampirik Video-Level ROC eğrilerini 'outputs/logs/roc_data_<model>_<ablation>.npz' dosyasından yükleyip çizer."""
    if output_path is None:
        output_path = PROJECT_ROOT / "outputs" / "figures" / "roc_curves.png"

    # 1. Ampirik ROC .npz dosyasını bul veya doğrula
    if roc_data_path is None or not Path(roc_data_path).is_file():
        candidates = [
            PROJECT_ROOT / "outputs" / "logs" / "roc_data_phdnet_full.npz",
            PROJECT_ROOT / "outputs" / "logs" / "roc_data.npz",
        ]
        logs_dir = PROJECT_ROOT / "outputs" / "logs"
        if logs_dir.is_dir():
            candidates.extend(sorted(logs_dir.glob("roc_data_*.npz")))

        found = None
        for c in candidates:
            if c.is_file():
                found = c
                break
        roc_data_path = found

    if roc_data_path is None or not Path(roc_data_path).is_file():
        print(f"[UYARI] Ampirik ROC verisi (.npz) bulunamadı: {roc_data_path}. Lütfen önce 'src/evaluate.py' çalıştırarak ROC verisini üretin.")
        return

    print(f"Ampirik ROC verisi yükleniyor: {roc_data_path}")
    data = np.load(roc_data_path, allow_pickle=True)

    # Verileri oku
    fpr_comb = data.get("p1_comb_fpr", data.get("p1_combined_fpr"))
    tpr_comb = data.get("p1_comb_tpr", data.get("p1_combined_tpr"))
    auc_comb = float(data.get("p1_comb_auc", data.get("p1_combined_auc", 0.5)))
    eer_comb = float(data.get("p1_comb_eer", data.get("p1_combined_eer", 0.5)))

    fpr_c23 = data.get("p1_c23_fpr")
    tpr_c23 = data.get("p1_c23_tpr")
    auc_c23 = float(data.get("p1_c23_auc", 0.5))
    eer_c23 = float(data.get("p1_c23_eer", 0.5))

    fpr_c40 = data.get("p1_c40_fpr")
    tpr_c40 = data.get("p1_c40_tpr")
    auc_c40 = float(data.get("p1_c40_auc", 0.5))
    eer_c40 = float(data.get("p1_c40_eer", 0.5))

    fpr_dfd = data.get("p2_dfd_fpr", data.get("p2_deepfakedetection_fpr"))
    tpr_dfd = data.get("p2_dfd_tpr", data.get("p2_deepfakedetection_tpr"))
    auc_dfd = float(data.get("p2_dfd_auc", data.get("p2_deepfakedetection_auc", 0.5)))
    eer_dfd = float(data.get("p2_dfd_eer", data.get("p2_deepfakedetection_eer", 0.5)))

    fpr_fs = data.get("p2_faceshifter_fpr", data.get("p2_fs_fpr"))
    tpr_fs = data.get("p2_faceshifter_tpr", data.get("p2_fs_tpr"))
    auc_fs = float(data.get("p2_faceshifter_auc", data.get("p2_fs_auc", 0.5)))
    eer_fs = float(data.get("p2_faceshifter_eer", data.get("p2_fs_eer", 0.5)))

    fpr_celeb = data.get("p3_celebdf_fpr", data.get("p3_celeb_fpr"))
    tpr_celeb = data.get("p3_celebdf_tpr", data.get("p3_celeb_tpr"))
    auc_celeb = float(data.get("p3_celebdf_auc", data.get("p3_celeb_auc", 0.5)))
    eer_celeb = float(data.get("p3_celebdf_eer", data.get("p3_celeb_eer", 0.5)))

    # Matplotlib stil
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    # Panel 1: Protokol 1 (FF++ Sıkıştırma Dayanıklılığı: c23 vs c40 vs Birleşik)
    if fpr_comb is not None and tpr_comb is not None:
        ax1.plot(fpr_comb, tpr_comb, color="#1f77b4", linewidth=2.5, label=f"Birleşik (c23 + c40) [AUC = {auc_comb:.4f}, EER = {eer_comb:.4f}]")
    if fpr_c40 is not None and tpr_c40 is not None:
        ax1.plot(fpr_c40, tpr_c40, color="#d62728", linewidth=2.5, linestyle="--", label=f"c40 Ağır Sıkıştırma [AUC = {auc_c40:.4f}, EER = {eer_c40:.4f}]")
    if fpr_c23 is not None and tpr_c23 is not None:
        ax1.plot(fpr_c23, tpr_c23, color="#2ca02c", linewidth=2.0, linestyle=":", label=f"c23 Hafif Sıkıştırma [AUC = {auc_c23:.4f}, EER = {eer_c23:.4f}]")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="-.", alpha=0.7, label="Rastgele Tahmin (AUC = 0.50)")

    # EER Noktası İşareti (c40 için)
    if fpr_c40 is not None and tpr_c40 is not None and len(fpr_c40) > 0:
        eer_idx = int(np.nanargmin(np.abs(fpr_c40 - (1.0 - tpr_c40))))
        ax1.scatter([fpr_c40[eer_idx]], [tpr_c40[eer_idx]], color="#d62728", s=100, zorder=5, label=f"c40 EER Noktası ({fpr_c40[eer_idx]:.2f})")

    ax1.set_title("Protokol 1: FF++ Within-Dataset Sıkıştırma Dayanıklılığı (Ampirik ROC)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Yanlış Pozitif Oranı (False Positive Rate - FPR)", fontsize=11)
    ax1.set_ylabel("Doğru Pozitif Oranı (True Positive Rate - TPR)", fontsize=11)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Protokol 2 & 3 (Sıfır-Atış Genelleme: DFD, FaceShifter, Celeb-DF v2)
    if fpr_dfd is not None and tpr_dfd is not None:
        ax2.plot(fpr_dfd, tpr_dfd, color="#ff7f0e", linewidth=2.2, label=f"DeepFakeDetection (DFD) [AUC = {auc_dfd:.4f}, EER = {eer_dfd:.4f}]")
    if fpr_celeb is not None and tpr_celeb is not None:
        ax2.plot(fpr_celeb, tpr_celeb, color="#9467bd", linewidth=2.2, label=f"Celeb-DF v2 (Cross-Dataset) [AUC = {auc_celeb:.4f}, EER = {eer_celeb:.4f}]")
    if fpr_fs is not None and tpr_fs is not None:
        ax2.plot(fpr_fs, tpr_fs, color="#8c564b", linewidth=2.0, linestyle="--", label=f"FaceShifter [AUC = {auc_fs:.4f}, EER = {eer_fs:.4f}]")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="-.", alpha=0.7, label="Rastgele Tahmin (AUC = 0.50)")

    if fpr_dfd is not None and tpr_dfd is not None and len(fpr_dfd) > 0:
        eer_dfd_idx = int(np.nanargmin(np.abs(fpr_dfd - (1.0 - tpr_dfd))))
        ax2.scatter([fpr_dfd[eer_dfd_idx]], [tpr_dfd[eer_dfd_idx]], color="#ff7f0e", s=100, zorder=5, label=f"DFD EER Noktası ({fpr_dfd[eer_dfd_idx]:.2f})")

    ax2.set_title("Protokol 2 & 3: Görülmemiş Manipülasyon ve Çapraz Veri Seti (Zero-Shot ROC)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Yanlış Pozitif Oranı (False Positive Rate - FPR)", fontsize=11)
    ax2.set_ylabel("Doğru Pozitif Oranı (True Positive Rate - TPR)", fontsize=11)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("PHD-Net Çoklu-Protokol Ampirik Video-Level ROC Eğrileri", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Ampirik ROC eğrileri grafiği kaydedildi: {output_path}")


# ==============================================================================
# 3. QPC ve HER Ampirik Saçılım (Scatter / Dispersion) Grafiği
# ==============================================================================

def plot_qpc_her_scatter(
    output_path: Path,
    manifest_path: Optional[Path] = None,
    splits_path: Optional[Path] = None,
    n_samples_per_group: int = 200,
    seed: int = 42,
    batch_size: int = 32,
    cache_features_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Gerçek ve Sahte Yüzler İçin Kuaterniyon Faz Dispersiyonu (QPC) vs Harmonik Enerji Oranı (HER) ampirik saçılım grafiği."""
    if cache_features_path is None:
        cache_features_path = PROJECT_ROOT / "outputs" / "logs" / "qpc_her_features.npz"

    loaded_from_cache = False
    if cache_features_path.is_file():
        try:
            print(f"Önceden hesaplanmış QPC/HER öznitelikleri yükleniyor: {cache_features_path}")
            cache = np.load(cache_features_path, allow_pickle=True)
            real_c23_disp = cache["real_c23_disp"]
            real_c23_her = cache["real_c23_her"]
            real_c40_disp = cache["real_c40_disp"]
            real_c40_her = cache["real_c40_her"]
            fake_c23_disp = cache["fake_c23_disp"]
            fake_c23_her = cache["fake_c23_her"]
            fake_c40_disp = cache["fake_c40_disp"]
            fake_c40_her = cache["fake_c40_her"]
            loaded_from_cache = True
        except Exception as e:
            print(f"[UYARI] Öznitelik önbelleği okunamadı ({e}), yeniden hesaplanacak.")

    if not loaded_from_cache:
        # 1. Manifest Dosyasını Oku
        if manifest_path is None:
            manifest_path = PROJECT_ROOT / "data" / "processed_faces" / "manifest_ffpp.json"
        manifest_path = Path(manifest_path)

        if not manifest_path.is_file():
            print(f"[UYARI] {manifest_path} bulunamadı, QPC-HER saçılım grafiği çizilemedi.")
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        raw_items = manifest_data.get("items", manifest_data if isinstance(manifest_data, list) else [])
        if not raw_items:
            print("[UYARI] Manifest içinde örnek bulunamadı!")
            return

        base_dir = manifest_path.parent

        # 2. FF++ Test Ayrımı Filtreleme
        splits = build_or_load_ffpp_splits(splits_path=splits_path)
        test_split = splits.get("test", {})

        test_items = []
        for it in raw_items:
            vid = str(it.get("video_id", ""))
            if is_video_in_split(test_split, vid):
                p_rel = it.get("frame_path") or it.get("rgb_path")
                if p_rel:
                    p_abs = Path(p_rel)
                    if not p_abs.is_absolute():
                        p_abs = base_dir / p_abs
                    if p_abs.is_file():
                        it_copy = dict(it)
                        it_copy["abs_path"] = p_abs
                        test_items.append(it_copy)

        print(f"FF++ test ayrımında bulunan geçerli yüz örnekleri: {len(test_items)}")
        if not test_items:
            print("[UYARI] Test ayrımında geçerli yüz dosyası bulunamadı!")
            return

        # 3. 4 Gruba Ayır ve Dengele (seed=42)
        rng = random.Random(seed)

        real_c23_pool = [it for it in test_items if it.get("label") == 0 and it.get("compression") == "c23"]
        real_c40_pool = [it for it in test_items if it.get("label") == 0 and it.get("compression") == "c40"]

        fake_c23_pool = [it for it in test_items if it.get("label") == 1 and it.get("compression") == "c23"]
        fake_c40_pool = [it for it in test_items if it.get("label") == 1 and it.get("compression") == "c40"]

        rng.shuffle(real_c23_pool)
        rng.shuffle(real_c40_pool)
        sample_real_c23 = real_c23_pool[:n_samples_per_group]
        sample_real_c40 = real_c40_pool[:n_samples_per_group]

        # Sahte örnekleri manipülasyonlar arasında dengeli örnekle
        def balance_fake_pool(pool: List[Dict[str, Any]], target_n: int) -> List[Dict[str, Any]]:
            by_manip = defaultdict(list)
            for it in pool:
                by_manip[it.get("manipulation", "unknown")].append(it)
            for m in by_manip:
                rng.shuffle(by_manip[m])
            manips = sorted(by_manip.keys())
            if not manips:
                return []
            per_manip = max(1, target_n // len(manips))
            sampled = []
            for m in manips:
                sampled.extend(by_manip[m][:per_manip])
            # Kalan varsa rastgele tamamla
            if len(sampled) < target_n:
                rem_pool = [it for it in pool if it not in sampled]
                rng.shuffle(rem_pool)
                sampled.extend(rem_pool[: target_n - len(sampled)])
            return sampled[:target_n]

        sample_fake_c23 = balance_fake_pool(fake_c23_pool, n_samples_per_group)
        sample_fake_c40 = balance_fake_pool(fake_c40_pool, n_samples_per_group)

        print(f"Örneklenen kümeler: real_c23={len(sample_real_c23)}, real_c40={len(sample_real_c40)}, fake_c23={len(sample_fake_c23)}, fake_c40={len(sample_fake_c40)}")

        # 4. QPC ve HER Özniteliklerini Hesapla
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"QPC ve HER hesaplama cihazı: {device}")

        qpc_module = QuaternionPhaseCongruency().to(device)
        her_module = HarmonicEnergyRatio().to(device)

        def extract_group(items: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
            disps: List[float] = []
            hers: List[float] = []
            for start_idx in range(0, len(items), batch_size):
                batch_it = items[start_idx : start_idx + batch_size]
                tensors = []
                for it in batch_it:
                    arr = load_face_rgb(it["abs_path"], target_size=224)
                    if arr is None:
                        continue
                    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                    tensors.append(t)
                if not tensors:
                    continue
                batch_t = torch.stack(tensors, dim=0).to(device)
                with torch.no_grad():
                    qpc_out = qpc_module(batch_t)
                    delta_phi = qpc_out[:, 2].mean(dim=(-2, -1)).cpu().numpy()
                    her_out = her_module(batch_t)
                    her_val = her_out["her_global"].cpu().numpy()
                disps.extend(delta_phi.tolist())
                hers.extend(her_val.tolist())
            return np.array(disps, dtype=float), np.array(hers, dtype=float)

        real_c23_disp, real_c23_her = extract_group(sample_real_c23)
        real_c40_disp, real_c40_her = extract_group(sample_real_c40)
        fake_c23_disp, fake_c23_her = extract_group(sample_fake_c23)
        fake_c40_disp, fake_c40_her = extract_group(sample_fake_c40)

        # Önbelleğe kaydet
        cache_features_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_features_path,
            real_c23_disp=real_c23_disp,
            real_c23_her=real_c23_her,
            real_c40_disp=real_c40_disp,
            real_c40_her=real_c40_her,
            fake_c23_disp=fake_c23_disp,
            fake_c23_her=fake_c23_her,
            fake_c40_disp=fake_c40_disp,
            fake_c40_her=fake_c40_her,
        )
        print(f"Ampirik öznitelikler önbelleğe kaydedildi: {cache_features_path}")

    # 5. Ampirik Grup İstatistikleri (Ortalama ve Medyan)
    r23_m = (float(np.mean(real_c23_disp)), float(np.mean(real_c23_her)))
    r40_m = (float(np.mean(real_c40_disp)), float(np.mean(real_c40_her)))
    f23_m = (float(np.mean(fake_c23_disp)), float(np.mean(fake_c23_her)))
    f40_m = (float(np.mean(fake_c40_disp)), float(np.mean(fake_c40_her)))

    r23_med = (float(np.median(real_c23_disp)), float(np.median(real_c23_her)))
    r40_med = (float(np.median(real_c40_disp)), float(np.median(real_c40_her)))
    f23_med = (float(np.median(fake_c23_disp)), float(np.median(fake_c23_her)))
    f40_med = (float(np.median(fake_c40_disp)), float(np.median(fake_c40_her)))

    print(f"Ampirik Grup Ortalamaları: Gerçek c23={r23_m}, Gerçek c40={r40_m}, Sahte c23={f23_m}, Sahte c40={f40_m}")
    print(f"Ampirik Grup Medyanları  : Gerçek c23={r23_med}, Gerçek c40={r40_med}, Sahte c23={f23_med}, Sahte c40={f40_med}")

    # 6. Grafik Çizimi
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)

    # Ampirik Saçılım Noktaları
    ax.scatter(real_c23_disp, real_c23_her, color="#1f77b4", alpha=0.50, s=35, edgecolors="none", label="Gerçek Yüzler (c23)")
    ax.scatter(real_c40_disp, real_c40_her, color="#17becf", alpha=0.45, s=35, marker="^", edgecolors="none", label="Gerçek Yüzler (c40)")
    ax.scatter(fake_c23_disp, fake_c23_her, color="#d62728", alpha=0.50, s=35, edgecolors="none", label="Sahte Yüzler (c23)")
    ax.scatter(fake_c40_disp, fake_c40_her, color="#ff7f0e", alpha=0.45, s=35, marker="s", edgecolors="none", label="Sahte Yüzler (c40)")

    # Ampirik Grup Ortalamaları (Centroids)
    ax.scatter([r23_m[0]], [r23_m[1]], color="#0b3c5d", marker="X", s=150, edgecolors="white", linewidth=1.5, zorder=6, label=f"Gerçek c23 Ort. ({r23_m[0]:.3f}, {r23_m[1]:.3f})")
    ax.scatter([r40_m[0]], [r40_m[1]], color="#008080", marker="X", s=150, edgecolors="white", linewidth=1.5, zorder=6, label=f"Gerçek c40 Ort. ({r40_m[0]:.3f}, {r40_m[1]:.3f})")
    ax.scatter([f23_m[0]], [f23_m[1]], color="#8b0000", marker="X", s=150, edgecolors="white", linewidth=1.5, zorder=6, label=f"Sahte c23 Ort. ({f23_m[0]:.3f}, {f23_m[1]:.3f})")
    ax.scatter([f40_m[0]], [f40_m[1]], color="#cc5500", marker="X", s=150, edgecolors="white", linewidth=1.5, zorder=6, label=f"Sahte c40 Ort. ({f40_m[0]:.3f}, {f40_m[1]:.3f})")

    # Ampirik Ayrıştırma Karar Sınırı (LDA ile)
    all_real_x = np.concatenate([real_c23_disp, real_c40_disp])
    all_real_y = np.concatenate([real_c23_her, real_c40_her])
    all_fake_x = np.concatenate([fake_c23_disp, fake_c40_disp])
    all_fake_y = np.concatenate([fake_c23_her, fake_c40_her])

    X_pts = np.vstack([
        np.column_stack([all_real_x, all_real_y]),
        np.column_stack([all_fake_x, all_fake_y]),
    ])
    y_labels = np.array([0] * len(all_real_x) + [1] * len(all_fake_x))

    x_min = max(0.0, float(np.percentile(X_pts[:, 0], 1)) * 0.8)
    x_max = float(np.percentile(X_pts[:, 0], 99)) * 1.15
    y_min = max(0.0, float(np.percentile(X_pts[:, 1], 1)) * 0.8)
    y_max = float(np.percentile(X_pts[:, 1], 99)) * 1.15

    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_pts, y_labels)
        w0, w1 = lda.coef_[0]
        intercept = lda.intercept_[0]
        if abs(w1) > 1e-5:
            x_line = np.linspace(x_min, x_max, 100)
            y_line = -(w0 * x_line + intercept) / w1
            valid_mask = (y_line >= y_min) & (y_line <= y_max)
            if np.sum(valid_mask) > 1:
                ax.plot(x_line[valid_mask], y_line[valid_mask], color="black", linestyle="--", linewidth=2.0, label="Ampirik Ayrım Sınırı (LDA)")
    except Exception as e:
        logger.warning("LDA karar sınırı çizilemedi: %s", e)

    ax.set_title(r"Kuaterniyon Faz Dispersiyonu ($\Delta\Phi_{\mathrm{chrom}}$) ve Harmonik Enerji Oranı (HER) Ampirik Dağılımı", fontsize=13, fontweight="bold")
    ax.set_xlabel(r"Kromatik Faz Dispersiyonu ($\Delta\Phi_{\mathrm{chrom}}$ — Sentez Sınır Artefaktı)", fontsize=11)
    ax.set_ylabel(r"Harmonik Enerji Oranı (HER: $\sum |\Omega_{\mathrm{mid}}| / (\sum |\Omega_{\mathrm{low}}| + \epsilon)$)", fontsize=11)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.legend(loc="upper right", frameon=True, fontsize=9, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Ampirik QPC-HER saçılım grafiği kaydedildi: {output_path}")


# ==============================================================================
# 4. Çoklu Model ve Ablasyon Karşılaştırma Grafiği
# ==============================================================================

def plot_ablation_comparison(
    logs_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Tüm 6 modelin (PHD-Net Full + 5 ablasyon) ampirik metriklerini 4 panelli grafik olarak görselleştirir."""
    if logs_dir is None:
        logs_dir = PROJECT_ROOT / "outputs" / "logs"
    if output_path is None:
        output_path = PROJECT_ROOT / "outputs" / "figures" / "ablation_comparison.png"

    ablation_keys = ["full", "no_qpc", "no_lfg", "no_dfsc", "no_ccm", "classical_freq"]
    display_names = [
        "PHD-Net (Full)",
        "w/o QPC",
        "w/o LFG",
        "w/o DFSC",
        "w/o CCM",
        "Classical Freq",
    ]

    reports = {}
    for k in ablation_keys:
        f = logs_dir / f"evaluation_report_{k}.json"
        if not f.is_file() and k == "full":
            f = logs_dir / "evaluation_report.json"
        if f.is_file():
            with open(f, "r", encoding="utf-8") as fp:
                reports[k] = json.load(fp)

    if len(reports) < len(ablation_keys):
        print(f"[UYARI] Bazı ablasyon raporları eksik (bulunan={len(reports)}/{len(ablation_keys)}).")
        if not reports:
            return

    models = [k for k in ablation_keys if k in reports]
    names = [display_names[ablation_keys.index(k)] for k in models]

    # Metrikleri çıkar
    p1_comb_auc = [reports[k]["protocol_1_within_dataset_ffpp"]["combined_c23_c40"]["video_metrics"]["auc"] for k in models]
    p1_c23_auc = [reports[k]["protocol_1_within_dataset_ffpp"]["compression_c23"]["video_metrics"]["auc"] for k in models]
    p1_c40_auc = [reports[k]["protocol_1_within_dataset_ffpp"]["compression_c40"]["video_metrics"]["auc"] for k in models]
    p1_c40_eer = [reports[k]["protocol_1_within_dataset_ffpp"]["compression_c40"]["video_metrics"]["eer"] * 100.0 for k in models]

    p2_fs_auc = [reports[k]["protocol_2_cross_manipulation"]["FaceShifter"]["video_metrics"]["auc"] for k in models]
    p2_dfd_auc = [reports[k]["protocol_2_cross_manipulation"]["DeepFakeDetection"]["video_metrics"]["auc"] for k in models]

    p3_celeb_auc = [reports[k]["protocol_3_cross_dataset_celebdf"]["video_metrics"]["auc"] for k in models]
    p3_celeb_eer = [reports[k]["protocol_3_cross_dataset_celebdf"]["video_metrics"]["eer"] * 100.0 for k in models]

    latencies = [reports[k].get("inference_speed_benchmark", {}).get("fast_detection_no_mask", {}).get("median_ms", 0.0) for k in models]
    fps_list = [reports[k].get("inference_speed_benchmark", {}).get("fast_detection_no_mask", {}).get("fps", 0.0) for k in models]

    # Çizim
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    x = np.arange(len(models))
    bar_width = 0.26

    # Panel A: Protokol 1 FF++ Within-Dataset (Video-AUC)
    ax1 = axes[0, 0]
    ax1.bar(x - bar_width, p1_c23_auc, width=bar_width, color="#2ca02c", label="c23 (Hafif Sıkıştırma)", alpha=0.9)
    ax1.bar(x, p1_comb_auc, width=bar_width, color="#1f77b4", label="Birleşik (c23 + c40)", alpha=0.9)
    ax1.bar(x + bar_width, p1_c40_auc, width=bar_width, color="#d62728", label="c40 (Ağır Sıkıştırma)", alpha=0.9)
    ax1.set_title("A) Protokol 1: FF++ Video-AUC ve Sıkıştırma Dayanıklılığı", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Video-AUC", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax1.set_ylim([0.92, 1.0])
    ax1.legend(loc="lower left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel B: Sıfır-Atış Genelleme (Video-AUC)
    ax2 = axes[0, 1]
    ax2.bar(x - bar_width, p2_fs_auc, width=bar_width, color="#8c564b", label="FaceShifter (Cross-Manip)", alpha=0.9)
    ax2.bar(x, p2_dfd_auc, width=bar_width, color="#ff7f0e", label="DFD (Cross-Manip)", alpha=0.9)
    ax2.bar(x + bar_width, p3_celeb_auc, width=bar_width, color="#9467bd", label="Celeb-DF v2 (Cross-Dataset)", alpha=0.9)
    ax2.set_title("B) Protokol 2 & 3: Görülmemiş Manipülasyon ve Veri Seti (Zero-Shot AUC)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Video-AUC", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax2.set_ylim([0.65, 0.92])
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Hata Oranları (EER %)
    ax3 = axes[1, 0]
    bw2 = 0.35
    ax3.bar(x - bw2 / 2, p1_c40_eer, width=bw2, color="#d62728", label="FF++ c40 EER (%)", alpha=0.85)
    ax3.bar(x + bw2 / 2, p3_celeb_eer, width=bw2, color="#9467bd", label="Celeb-DF v2 EER (%)", alpha=0.85)
    ax3.set_title("C) Eşit Hata Oranı Karşılaştırması (EER % — Düşük Olması İyidir)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("EER (%)", fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax3.set_ylim([0, 30])
    ax3.legend(loc="upper left", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Panel D: Çıkarım Gecikmesi ve Hız (RTX 5080)
    ax4 = axes[1, 1]
    color_lat = "#17becf"
    color_fps = "#e377c2"
    ax4.bar(x - bw2 / 2, latencies, width=bw2, color=color_lat, label="Medyan Gecikme (ms)", alpha=0.85)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x, fps_list, "s--", color=color_fps, linewidth=2.2, markersize=7, label="Çıkarım Hızı (FPS)")
    ax4.set_title("D) Çıkarım Gecikmesi ve Hızı (RTX 5080 Laptop GPU)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Gecikme (ms)", fontsize=11, color=color_lat)
    ax4_twin.set_ylabel("Hız (FPS)", fontsize=11, color=color_fps)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax4.set_ylim([0, 20])
    ax4_twin.set_ylim([40, 150])
    ax4.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("PHD-Net ve Ablasyon Varyantlarının Çoklu-Protokol Ampirik Başarım Özeti", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Ablasyon karşılaştırma grafiği kaydedildi: {output_path}")


if __name__ == "__main__":
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    logs_dir = PROJECT_ROOT / "outputs" / "logs"

    # 1. Eğitim Kayıp Eğrileri
    plot_training_curves(
        history_json_path=logs_dir / "train_history_phdnet_full.json",
        output_path=figures_dir / "training_loss_curves.png",
    )

    # 2. ROC Eğrileri
    plot_all_roc_curves(
        eval_report_path=logs_dir / "evaluation_report_full.json" if (logs_dir / "evaluation_report_full.json").is_file() else logs_dir / "evaluation_report.json",
        output_path=figures_dir / "roc_curves.png",
        roc_data_path=logs_dir / "roc_data_phdnet_full.npz",
    )

    # 3. QPC vs HER Saçılım Grafiği
    plot_qpc_her_scatter(
        output_path=figures_dir / "her_qpc_scatter.png",
    )

    # 4. Çoklu Model ve Ablasyon Karşılaştırma Grafiği
    plot_ablation_comparison(
        logs_dir=logs_dir,
        output_path=figures_dir / "ablation_comparison.png",
    )
