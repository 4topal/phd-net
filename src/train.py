"""PHD-Net ve Meso-4 Eğitim Hattı (Training Pipeline).

Özellikler:
- Modeller: `PHDNet` (`--ablation full|no_qpc|no_lfg|no_dfsc|no_ccm|classical_freq`) ve `Meso4` baseline
- Pretrained EdgeNeXt-XS (`edgenext_xs`) uzamsal omurga için diferansiyel öğrenme oranı (0.25x backbone, 1.0x diğerleri)
- Karma Hassasiyet Eğitimi (AMP): CUDA üzerinde `bfloat16` (donanım destekliyorsa) veya `float16`
- Optimizasyon: `AdamW`, doğrusal ısınma (2 epoch) + Kosinüs Azalma (`CosineAnnealingLR`)
- Gradyan Kırpma (`max_norm=2.0`)
- Kare ve Video Düzeyi Doğrulama Metrikleri: AUC-ROC, EER, AP, F1, Accuracy
- Doğrulamada tam adli lokalizasyon maskesi denetimi ile train_loss ve val_loss ölçek eşitlemesi
- Sıkıştırma ayrıştırmalı video kümeleme (`{video_id}_{comp}`)
- Video-AUC skoruna dayalı erken durdurma (`--early_stop_patience`, varsayılan: 5 epoch)
- Hızlı Duman Testi (`--smoke_test`): `SyntheticSmokeDataset` ile 2 epoch hızlı çalıştırma
- En iyi model kontrol noktası (`best_<model>_<ablation>.pt`), en son model (`latest_<model>_<ablation>.pt`)
  ve kesintiden devam etme (`--resume auto`) desteği
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, roc_curve
except ImportError:
    # sklearn yoksa temel metrik fonksiyonları yedeklenir
    def roc_curve(y_true: np.ndarray, y_score: np.ndarray, pos_label: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        thresholds = np.sort(np.unique(y_score))[::-1]
        tpr, fpr = [0.0], [0.0]
        n_pos = np.sum(y_true == pos_label)
        n_neg = np.sum(y_true != pos_label)
        if n_pos == 0 or n_neg == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
        for t in thresholds:
            preds = (y_score >= t).astype(int)
            tp = np.sum((preds == 1) & (y_true == pos_label))
            fp = np.sum((preds == 1) & (y_true != pos_label))
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)
        return np.array(fpr), np.array(tpr), np.array([thresholds[0] + 1.0] + list(thresholds))

    def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(np.trapz(tpr, fpr))

    def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
        return roc_auc_score(y_true, y_score)

    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: int = 0) -> float:
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        denom = 2 * tp + fp + fn
        return float(2 * tp / denom) if denom > 0 else float(zero_division)

    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

# Proje içi içe aktarmalar
try:
    from dataset import (
        BalancedBatchSampler,
        DeepfakeFaceDataset,
        SyntheticSmokeDataset,
        build_or_load_ffpp_splits,
    )
    from mesonet import Meso4
    from phasefreq_net import PHDNet, PHDNetMultiTaskLoss
    from project_paths import PROJECT_ROOT, get_ffpp_root
except ImportError:
    from src.dataset import (
        BalancedBatchSampler,
        DeepfakeFaceDataset,
        SyntheticSmokeDataset,
        build_or_load_ffpp_splits,
    )
    from src.mesonet import Meso4
    from src.phasefreq_net import PHDNet, PHDNetMultiTaskLoss
    from src.project_paths import PROJECT_ROOT, get_ffpp_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phdnet.train")


# ==============================================================================
# Metrik ve EER Hesaplama Yardımcıları
# ==============================================================================

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Eşit Hata Oranı (Equal Error Rate — EER) hesaplar."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1.0 - tpr
        idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        return float(np.clip(eer, 0.0, 1.0))
    except Exception:
        return 0.5


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray, method: str = "youden") -> float:
    """ROC eğrisi üzerinden optimal sınıflandırma eşiğini hesaplar (Youden's J veya EER)."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_probs = np.asarray(y_probs, dtype=float).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
        if method == "youden":
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
        elif method == "eer":
            fnr = 1.0 - tpr
            best_idx = int(np.nanargmin(np.abs(fpr - fnr)))
        else:
            best_idx = int(np.argmax(tpr - fpr))
        return float(np.clip(thresholds[best_idx], 0.05, 0.95))
    except Exception:
        return 0.5


def compute_classification_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """İkili sınıflandırma metriklerini (hem sabit 0.5 hem de optimal eşik ile) güvenli şekilde hesaplar."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_probs = np.asarray(y_probs, dtype=float).reshape(-1)

    # Tek sınıf durumu kontrolü (ör. küçük test kümeleri)
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        acc = float(accuracy_score(y_true, (y_probs >= 0.5).astype(int)))
        return {
            "auc": 0.5,
            "eer": 0.5,
            "ap": 0.5,
            "f1": 0.0,
            "accuracy": acc,
            "optimal_threshold": 0.5,
            "opt_f1": 0.0,
            "opt_accuracy": acc,
        }

    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        auc = 0.5

    try:
        ap = float(average_precision_score(y_true, y_probs))
    except Exception:
        ap = 0.5

    eer = compute_eer(y_true, y_probs)

    # 1. Varsayılan (0.5) Eşik Değerleri
    fixed_thresh = 0.5 if threshold is None else threshold
    preds = (y_probs >= fixed_thresh).astype(int)
    f1 = float(f1_score(y_true, preds, zero_division=0))
    acc = float(accuracy_score(y_true, preds))

    # 2. Optimal Eşik (Youden's J İstatistiği: max(TPR - FPR))
    opt_thresh = find_optimal_threshold(y_true, y_probs, method="youden")
    opt_preds = (y_probs >= opt_thresh).astype(int)
    opt_f1 = float(f1_score(y_true, opt_preds, zero_division=0))
    opt_acc = float(accuracy_score(y_true, opt_preds))

    return {
        "auc": round(auc, 4),
        "eer": round(eer, 4),
        "ap": round(ap, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "optimal_threshold": round(opt_thresh, 4),
        "opt_f1": round(opt_f1, 4),
        "opt_accuracy": round(opt_acc, 4),
    }


def aggregate_video_metrics(
    video_ids: Sequence[str],
    y_true: Sequence[int],
    y_probs: Sequence[float],
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Kare tahminlerini `video_id` (veya `video_id_comp`) bazında ortalayarak video düzeyi metrikleri hesaplar."""
    v_probs_dict: Dict[str, List[float]] = defaultdict(list)
    v_labels_dict: Dict[str, int] = {}

    for vid, label, prob in zip(video_ids, y_true, y_probs):
        v_probs_dict[vid].append(float(prob))
        v_labels_dict[vid] = int(label)

    v_ids = sorted(v_probs_dict.keys())
    v_mean_probs = np.array([float(np.mean(v_probs_dict[v])) for v in v_ids], dtype=float)
    v_true_labels = np.array([v_labels_dict[v] for v in v_ids], dtype=int)

    return compute_classification_metrics(v_true_labels, v_mean_probs, threshold=threshold)


# ==============================================================================
# Öğrenme Oranı Zamanlayıcısı (Warmup + Cosine Annealing)
# ==============================================================================

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    last_step: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Adım bazlı doğrusal ısınma + kosinüs azalma öğrenme oranı zamanlayıcısı."""
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_step)


# ==============================================================================
# Model Fabrikası
# ==============================================================================

def create_model_and_loss(
    model_name: str,
    ablation_mode: str = "full",
    lambda_dice: float = 1.0,
    lambda_ccm: float = 0.5,
    device: torch.device = torch.device("cpu"),
    use_pretrained_backbone: bool = True,
    label_smoothing: float = 0.05,
) -> Tuple[nn.Module, nn.Module]:
    """Model ve kayıp fonksiyonunu parametrelere göre örneklendirir."""
    if model_name.lower() == "phdnet":
        model = PHDNet(
            ablation_mode=ablation_mode,
            use_pretrained_backbone=use_pretrained_backbone,
        ).to(device)
        actual_ccm = 0.0 if ablation_mode == "no_ccm" else lambda_ccm
        criterion = PHDNetMultiTaskLoss(
            lambda_dice=lambda_dice,
            lambda_ccm=actual_ccm,
            label_smoothing=label_smoothing,
        ).to(device)
    elif model_name.lower() == "meso4":
        model = Meso4(image_size=224).to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}. Seçenekler: 'phdnet', 'meso4'")

    return model, criterion


# ==============================================================================
# Tek Epoch Eğitim Fonksiyonu
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    is_phdnet: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_grad_norm: float = 2.0,
) -> Dict[str, float]:
    """Bir tam eğitim epoch'u çalıştırır."""
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_ccm = 0.0
    running_dice = 0.0
    total_steps = 0

    for batch in dataloader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        has_mask = batch["has_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if is_phdnet:
                outputs = model(rgb, compute_mask=True)
                loss_dict = criterion(outputs, labels, mask_targets=mask, has_mask=has_mask)
                loss = loss_dict["total_loss"]
                bce_val = loss_dict["bce_loss"]
                ccm_val = loss_dict["ccm_loss"]
                dice_val = loss_dict["dice_loss"] or 0.0
            else:
                logits = model(rgb)
                loss = criterion(logits, labels.float().view(-1, 1))
                bce_val = float(loss.detach().item())
                ccm_val = 0.0
                dice_val = 0.0

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += float(loss.detach().item())
        running_bce += bce_val
        running_ccm += ccm_val
        running_dice += dice_val
        total_steps += 1

        if total_steps % 200 == 0:
            logger.info(
                "  -> Adım [%d/%d] | Anlık Kayıp: Toplam=%.4f, BCE=%.4f, CCM=%.4f, Dice=%.4f",
                total_steps, len(dataloader), float(loss.detach().item()), bce_val, ccm_val, dice_val
            )

    steps = max(1, total_steps)
    return {
        "loss": round(running_loss / steps, 4),
        "bce": round(running_bce / steps, 4),
        "ccm": round(running_ccm / steps, 4),
        "dice": round(running_dice / steps, 4),
    }


# ==============================================================================
# Doğrulama Döngüsü Fonksiyonu
# ==============================================================================

@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_phdnet: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Doğrulama kümesi üzerinde kare ve video düzeyinde metrikleri hesaplar.

    Kayıp Ölçeği Eşitlemesi:
      `model(rgb, compute_mask=True)` ve `criterion(outputs, labels, mask_targets=mask, has_mask=has_mask)`
      ile `val_loss` tam olarak `train_loss` ile aynı ölçek ve bileşenlerle hesaplanır.
    Sıkıştırma Ayrıştırması:
      `video_id` ile `compression` bilgisi birleştirilerek (`f"{vid}_{comp}"`) c23 ve c40
      videolarının metriklerinin birbirine karışması önlenir.
    """
    model.eval()
    all_probs: List[float] = []
    all_labels: List[int] = []
    all_video_ids: List[str] = []
    running_loss = 0.0
    total_steps = 0

    for batch in dataloader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        has_mask = batch["has_mask"].to(device, non_blocking=True)
        v_ids = batch["video_id"]
        comps = batch.get("compression")

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if is_phdnet:
                outputs = model(rgb, compute_mask=True)
                loss_dict = criterion(outputs, labels, mask_targets=mask, has_mask=has_mask)
                loss = loss_dict["total_loss"]
                logits = outputs["logits"]
            else:
                logits = model(rgb)
                loss = criterion(logits, labels.float().view(-1, 1))

        probs = torch.sigmoid(logits.float()).view(-1).cpu().numpy()
        lbls = labels.view(-1).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(lbls.tolist())

        if comps is not None:
            grouped_ids = [f"{v}_{c}" for v, c in zip(v_ids, comps)]
        else:
            grouped_ids = list(v_ids)
        all_video_ids.extend(grouped_ids)

        running_loss += float(loss.detach().item())
        total_steps += 1

    mean_loss = round(running_loss / max(1, total_steps), 4)
    frame_metrics = compute_classification_metrics(np.array(all_labels), np.array(all_probs))
    video_metrics = aggregate_video_metrics(all_video_ids, all_labels, all_probs)

    return frame_metrics, video_metrics, mean_loss


# ==============================================================================
# Ana Eğitim Fonksiyonu
# ==============================================================================

def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    """Argümanlara göre tam eğitim döngüsünü koşturur."""
    # 1. Determinizm
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 2. Cihaz ve AMP Yapılandırması
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = args.amp and (device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            cuda_idx = device.index if device.index is not None else 0
            torch.cuda.set_per_process_memory_fraction(0.92, device=cuda_idx)
        except Exception as ex:
            logger.warning("CUDA bellek sınırı ayarlanamadı: %s", ex)
        logger.info("cuDNN benchmark etkinleştirildi ve CUDA VRAM sınırı %%92 olarak ayarlandı.")

    if use_amp and torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = None  # bfloat16 ölçekleyici gerektirmez
        logger.info("AMP etkin: bfloat16 hassasiyeti seçildi (RTX 5080 uyumlu).")
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        logger.info("AMP etkin: float16 hassasiyeti ve GradScaler seçildi.")
    else:
        amp_dtype = torch.float32
        scaler = None
        logger.info("AMP devre dışı: float32 kullanılıyor.")

    # 3. Dizinler
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 4. Veri Seti Oluşturma
    if args.smoke_test:
        logger.info("DUMAN TESTİ (Smoke Test): Sentetik veri kümeleri oluşturuluyor...")
        train_ds = SyntheticSmokeDataset(num_samples=128, target_size=224, seed=args.seed)
        val_ds = SyntheticSmokeDataset(num_samples=64, target_size=224, seed=args.seed + 1)
        batch_size = min(args.batch_size, 32)
        val_batch_size = min(batch_size * 2, 64)
        epochs = min(args.epochs, 2)
        num_workers = 0
    else:
        manifest_path = args.manifest_path
        if manifest_path is None and args.data_root is None:
            default_manifest = PROJECT_ROOT / "data" / "processed_faces" / "manifest_ffpp.json"
            if default_manifest.is_file():
                manifest_path = str(default_manifest)
                logger.info("Varsayılan manifest bulundu ve kullanılıyor: %s", manifest_path)

        logger.info("Gerçek veri yükleniyor (data_root=%s, manifest_path=%s)...", args.data_root, manifest_path)
        splits = build_or_load_ffpp_splits()
        train_ds = DeepfakeFaceDataset(
            data_root=args.data_root,
            manifest_path=manifest_path,
            split="train",
            compressions=args.compressions,
            manipulations=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
            is_train=True,
            target_size=224,
        )
        val_ds = DeepfakeFaceDataset(
            data_root=args.data_root,
            manifest_path=manifest_path,
            split="val",
            compressions=args.compressions,
            manipulations=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
            is_train=False,
            target_size=224,
        )
        batch_size = args.batch_size
        val_batch_size = min(batch_size, 128)
        epochs = args.epochs
        num_workers = args.num_workers

    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    # Dengeleyici Örnekleyici (BalancedBatchSampler)
    try:
        train_sampler = BalancedBatchSampler(train_ds, batch_size=batch_size, seed=args.seed)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            **loader_kwargs,
        )
    except Exception as e:
        logger.warning("BalancedBatchSampler oluşturulamadı (%s), standart DataLoader kullanılıyor.", e)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    logger.info("Eğitim örnek sayısı: %d | Doğrulama örnek sayısı: %d | Batch boyutu: Train=%d, Val=%d | Workers=%d",
                len(train_ds), len(val_ds), batch_size, val_batch_size, num_workers)

    # 5. Model ve Kayıp
    is_phdnet = (args.model.lower() == "phdnet")
    model, criterion = create_model_and_loss(
        model_name=args.model,
        ablation_mode=args.ablation,
        lambda_dice=args.lambda_dice,
        lambda_ccm=args.lambda_ccm,
        device=device,
        use_pretrained_backbone=not getattr(args, "scratch_backbone", False),
        label_smoothing=0.05,
    )

    # 6. Optimizer (Diferansiyel Öğrenme Oranı) ve Scheduler
    if is_phdnet:
        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "spatial_backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": 0.25 * args.lr, "name": "spatial_backbone"},
            {"params": other_params, "lr": 1.0 * args.lr, "name": "other_params"},
        ]
        logger.info(
            "Diferansiyel Öğrenme Oranı: spatial_backbone (lr=%.2e, %d param), diğer katmanlar (lr=%.2e, %d param)",
            0.25 * args.lr, sum(p.numel() for p in backbone_params),
            1.0 * args.lr, sum(p.numel() for p in other_params),
        )
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    best_ckpt_path = ckpt_dir / f"best_{args.model}_{args.ablation}.pt"
    latest_ckpt_path = ckpt_dir / f"latest_{args.model}_{args.ablation}.pt"
    history_path = log_dir / f"train_history_{args.model}_{args.ablation}.json"

    start_epoch = 1
    best_vid_auc = -1.0
    history: List[Dict[str, Any]] = []
    patience_counter = 0

    # Resume kontrolü
    resume_target = None
    if getattr(args, "resume", None):
        if args.resume == "auto":
            if latest_ckpt_path.is_file():
                resume_target = latest_ckpt_path
            elif best_ckpt_path.is_file():
                resume_target = best_ckpt_path
        else:
            cand = Path(args.resume)
            if cand.is_file():
                resume_target = cand
            elif (ckpt_dir / args.resume).is_file():
                resume_target = ckpt_dir / args.resume

    if resume_target is not None and resume_target.is_file():
        logger.info("Kontrol noktası yükleniyor (resume): %s", resume_target)
        ckpt_data = torch.load(resume_target, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"])
        if "optimizer_state_dict" in ckpt_data and optimizer is not None:
            try:
                optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            except Exception as e:
                logger.warning("Optimizer durumu yüklenemedi (%s), sıfırdan devam ediliyor.", e)

        start_epoch = ckpt_data.get("epoch", 0) + 1
        best_vid_auc = float(ckpt_data.get("best_video_auc", -1.0))
        logger.info("--> Kaldığı yer: Epoch %d | Önceki En İyi Video-AUC: %.4f", start_epoch, best_vid_auc)

        if history_path.is_file():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    old_hist_data = json.load(f)
                if isinstance(old_hist_data, dict) and "history" in old_hist_data:
                    history = [h for h in old_hist_data["history"] if h.get("epoch", 0) < start_epoch]
                    if best_vid_auc <= 0 and "best_video_auc" in old_hist_data:
                        best_vid_auc = float(old_hist_data["best_video_auc"])
            except Exception as ex:
                logger.warning("Geçmiş log dosyası okunamadı: %s", ex)

    steps_per_epoch = max(1, len(train_loader))
    start_step = (start_epoch - 1) * steps_per_epoch
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        warmup_epochs=min(2, epochs),
        total_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        last_step=start_step - 1 if start_step > 0 else -1,
    )

    # 7. Eğitim Döngüsü
    logger.info("Eğitim başlıyor: %s (ablasyon=%s, başlangıç_epoch=%d, toplam_epoch=%d, lr=%.1e, sabır=%d)",
                args.model, args.ablation, start_epoch, epochs, args.lr, args.early_stop_patience)

    t_start = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        train_res = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            is_phdnet=is_phdnet,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        val_frame, val_video, val_loss = evaluate_validation(
            model=model,
            criterion=criterion,
            dataloader=val_loader,
            device=device,
            is_phdnet=is_phdnet,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        current_lr = float(optimizer.param_groups[-1]["lr"])
        dt = time.perf_counter() - t0

        opt_th = val_video.get("optimal_threshold", 0.5)
        opt_f1 = val_video.get("opt_f1", val_video["f1"])
        opt_acc = val_video.get("opt_accuracy", val_video["accuracy"])

        logger.info(
            "Epoch [%d/%d] (%.1fs) | Kayıp: Train=%.4f (BCE=%.4f, CCM=%.4f, Dice=%.4f), Val=%.4f | "
            "Kare AUC=%.4f | Video AUC=%.4f, EER=%.4f | "
            "F1(0.5)=%.4f | Opt-F1(τ=%.2f)=%.4f, Opt-Acc=%.4f | LR=%.2e",
            epoch, epochs, dt,
            train_res["loss"], train_res["bce"], train_res["ccm"], train_res["dice"], val_loss,
            val_frame["auc"],
            val_video["auc"], val_video["eer"],
            val_video["f1"], opt_th, opt_f1, opt_acc,
            current_lr,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_res["loss"],
            "train_bce": train_res["bce"],
            "train_ccm": train_res["ccm"],
            "train_dice": train_res["dice"],
            "val_loss": val_loss,
            "frame_metrics": val_frame,
            "video_metrics": val_video,
            "lr": current_lr,
            "duration_sec": round(dt, 2),
        }
        history.append(epoch_record)

        # En son kontrol noktası (latest checkpoint) kaydı
        torch.save({
            "epoch": epoch,
            "model_name": args.model,
            "ablation_mode": args.ablation,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_video_auc": best_vid_auc,
            "frame_metrics": val_frame,
            "video_metrics": val_video,
            "args": vars(args),
        }, latest_ckpt_path)

        # En iyi kontrol noktası kaydı (Video-AUC bazlı) ve Erken Durdurma
        vid_auc = val_video["auc"]
        if vid_auc > best_vid_auc:
            best_vid_auc = vid_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "ablation_mode": args.ablation,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_video_auc": best_vid_auc,
                "frame_metrics": val_frame,
                "video_metrics": val_video,
                "args": vars(args),
            }, best_ckpt_path)
            logger.info("  --> Yeni en iyi model kaydedildi! Video-AUC: %.4f -> %s", best_vid_auc, best_ckpt_path)
        else:
            patience_counter += 1
            if args.early_stop_patience > 0:
                logger.info(
                    "  --> Video-AUC gelişmedi (sabır: %d/%d, en iyi: %.4f).",
                    patience_counter, args.early_stop_patience, best_vid_auc
                )
                if patience_counter >= args.early_stop_patience:
                    logger.info(
                        "Erken durdurma (Early Stopping) tetiklendi: Video-AUC %d epoch boyunca gelişmedi.",
                        patience_counter
                    )
                    break

        # Geçmiş JSON kaydı
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "ablation": args.ablation,
                "best_video_auc": best_vid_auc,
                "history": history,
            }, f, indent=2)

        # Süre / Saat Sınırı Kontrolü (Graceful Stop)
        should_stop = False
        if getattr(args, "max_duration_minutes", None) is not None:
            elapsed_min = (time.perf_counter() - t_start) / 60.0
            if elapsed_min >= args.max_duration_minutes:
                logger.info("Zaman limiti aşıldı (%.1f / %.1f dk). Eğitim güvenli şekilde durduruluyor.",
                            elapsed_min, args.max_duration_minutes)
                should_stop = True

        if getattr(args, "stop_at_local_time", None):
            try:
                import datetime
                now = datetime.datetime.now()
                target_parts = args.stop_at_local_time.strip().split(":")
                target_h, target_m = int(target_parts[0]), int(target_parts[1])
                target_dt = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
                if now >= target_dt:
                    logger.info("Hedef durdurma saatine (%s) ulaşıldı. Eğitim güvenli şekilde durduruluyor.",
                                args.stop_at_local_time)
                    should_stop = True
            except Exception as ex:
                logger.warning("stop_at_local_time kontrol hatası: %s", ex)

        if should_stop:
            logger.info("Eğitim kullanıcı tarafından belirlenen zaman sınırında tamamlandı.")
            break

    total_time = time.perf_counter() - t_start
    logger.info("Eğitim tamamlandı! Toplam süre: %.1f sn | En iyi Video-AUC: %.4f", total_time, best_vid_auc)

    return {
        "best_video_auc": best_vid_auc,
        "best_checkpoint": str(best_ckpt_path),
        "history_file": str(history_path),
        "total_time_sec": total_time,
    }


# ==============================================================================
# CLI ve Giriş Noktası
# ==============================================================================

def parse_args(raw_args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHD-Net / Meso-4 Eğitim Betiği")
    parser.add_argument("--model", type=str, default="phdnet", choices=["phdnet", "meso4"],
                        help="Eğitilecek model mimarisi")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_qpc", "no_lfg", "no_dfsc", "no_ccm", "classical_freq"],
                        help="Ablasyon varyantı")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Kırpılmış yüz veri kümesi kök dizini (varsayılan: data/processed_faces)")
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Önceden üretilmiş manifest.json yolu")
    parser.add_argument("--compressions", nargs="+", default=["c23", "c40"],
                        help="Eğitime dahil edilecek sıkıştırma seviyeleri (c23 c40)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Toplam eğitim epoch sayısı")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch boyutu")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Başlangıç öğrenme oranı (AdamW)")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Ağırlık azalma katsayısı")
    parser.add_argument("--lambda_dice", type=float, default=1.0,
                        help="Dice lokalizasyon kaybı ağırlığı")
    parser.add_argument("--lambda_ccm", type=float, default=0.5,
                        help="Kontrastif kayıp (L_CCM) ağırlığı")
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="Video-AUC skoruna dayalı erken durdurma sabrı (epoch sayısı)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader çalışan iş parçacığı sayısı")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Her worker için önceden yüklenecek batch sayısı")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="CUDA üzerinde Otomatik Karma Hassasiyet (AMP) kullan")
    parser.add_argument("--no_amp", action="store_false", dest="amp",
                        help="AMP kullanımını kapat")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Hızlı 2 epoch sentetik duman testi çalıştır")
    parser.add_argument("--scratch_backbone", action="store_true", default=False,
                        help="Pretrained edgenext_xs yerine rastgele ilklendirilmiş omurga kullan")
    parser.add_argument("--checkpoint_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "checkpoints"),
                        help="Model kontrol noktalarının kaydedileceği dizin")
    parser.add_argument("--log_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "logs"),
                        help="Eğitim geçmişinin kaydedileceği log dizini")
    parser.add_argument("--device", type=str, default=None,
                        help="Cihaz (cuda, cpu vb.)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Rastgelelik tohumu")
    parser.add_argument("--resume", type=str, default=None, nargs="?", const="auto",
                        help="Önceki kontrol noktasından eğitime devam et ('auto' veya dosya yolu)")
    parser.add_argument("--max_duration_minutes", type=float, default=None,
                        help="Maksimum eğitim süresi (dakika). Süre dolduğunda epoch bitiminde güvenli durur.")
    parser.add_argument("--stop_at_local_time", type=str, default=None,
                        help="Eğitimin durdurulacağı yerel saat (ör. '13:15' veya '13:30').")

    return parser.parse_args(raw_args)


if __name__ == "__main__":
    cli_args = parse_args()
    run_training(cli_args)
