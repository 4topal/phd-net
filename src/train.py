"""PHD-Net ve Meso-4 Eğitim Hattı (Training Pipeline).

Özellikler:
- Modeller: `PHDNet` (`--ablation full|no_qpc|no_lfg|no_dfsc|no_ccm|classical_freq`) ve `Meso4` baseline
- Karma Hassasiyet Eğitimi (AMP): CUDA üzerinde `bfloat16` (donanım destekliyorsa) veya `float16`
- Optimizasyon: `AdamW`, doğrusal ısınma (2 epoch) + Kosinüs Azalma (`CosineAnnealingLR`)
- Gradyan Kırpma (`max_norm=2.0`)
- Kare ve Video Düzeyi Doğrulama Metrikleri: AUC-ROC, EER, AP, F1, Accuracy
- Hızlı Duman Testi (`--smoke_test`): `SyntheticSmokeDataset` ile 2 epoch hızlı çalıştırma
- En iyi model kontrol noktası (`best_<model>_<ablation>.pt`) ve geçmiş log kaydı
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from dataset import (
    BalancedBatchSampler,
    DeepfakeFaceDataset,
    SyntheticSmokeDataset,
    build_or_load_ffpp_splits,
)
from mesonet import Meso4
from phasefreq_net import PHDNet, PHDNetMultiTaskLoss
from project_paths import PROJECT_ROOT, get_ffpp_root

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


def compute_classification_metrics(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    """İkili sınıflandırma metriklerini güvenli şekilde hesaplar."""
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
    preds = (y_probs >= 0.5).astype(int)
    f1 = float(f1_score(y_true, preds, zero_division=0))
    acc = float(accuracy_score(y_true, preds))

    return {
        "auc": round(auc, 4),
        "eer": round(eer, 4),
        "ap": round(ap, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
    }


def aggregate_video_metrics(
    video_ids: Sequence[str], y_true: Sequence[int], y_probs: Sequence[float]
) -> Dict[str, float]:
    """Kare tahminlerini `video_id` bazında ortalayarak video düzeyi metrikleri hesaplar."""
    v_probs_dict: Dict[str, List[float]] = defaultdict(list)
    v_labels_dict: Dict[str, int] = {}

    for vid, label, prob in zip(video_ids, y_true, y_probs):
        v_probs_dict[vid].append(float(prob))
        v_labels_dict[vid] = int(label)

    v_ids = sorted(v_probs_dict.keys())
    v_mean_probs = np.array([float(np.mean(v_probs_dict[v])) for v in v_ids], dtype=float)
    v_true_labels = np.array([v_labels_dict[v] for v in v_ids], dtype=int)

    return compute_classification_metrics(v_true_labels, v_mean_probs)


# ==============================================================================
# Öğrenme Oranı Zamanlayıcısı (Warmup + Cosine Annealing)
# ==============================================================================

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Adım bazlı doğrusal ısınma + kosinüs azalma öğrenme oranı zamanlayıcısı."""
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==============================================================================
# Model Fabrikası
# ==============================================================================

def create_model_and_loss(
    model_name: str,
    ablation_mode: str = "full",
    lambda_dice: float = 1.0,
    lambda_ccm: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, nn.Module]:
    """Model ve kayıp fonksiyonunu parametrelere göre örneklendirir."""
    if model_name.lower() == "phdnet":
        model = PHDNet(ablation_mode=ablation_mode).to(device)
        actual_ccm = 0.0 if ablation_mode == "no_ccm" else lambda_ccm
        criterion = PHDNetMultiTaskLoss(
            lambda_dice=lambda_dice,
            lambda_ccm=actual_ccm,
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
    """Doğrulama kümesi üzerinde kare ve video düzeyinde metrikleri hesaplar."""
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

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if is_phdnet:
                outputs = model(rgb, compute_mask=False)
                loss_dict = criterion(outputs, labels, mask_targets=None, has_mask=None)
                loss = loss_dict["total_loss"]
                logits = outputs["logits"]
            else:
                logits = model(rgb)
                loss = criterion(logits, labels.float().view(-1, 1))

        probs = torch.sigmoid(logits.float()).view(-1).cpu().numpy()
        lbls = labels.view(-1).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(lbls.tolist())
        all_video_ids.extend(list(v_ids))

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
        epochs = min(args.epochs, 2)
        num_workers = 0
    else:
        logger.info("Gerçek veri yükleniyor (data_root=%s)...", args.data_root)
        splits = build_or_load_ffpp_splits()
        train_ds = DeepfakeFaceDataset(
            data_root=args.data_root,
            manifest_path=args.manifest_path,
            split="train",
            compressions=args.compressions,
            manipulations=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
            is_train=True,
            target_size=224,
        )
        val_ds = DeepfakeFaceDataset(
            data_root=args.data_root,
            manifest_path=args.manifest_path,
            split="val",
            compressions=args.compressions,
            manipulations=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
            is_train=False,
            target_size=224,
        )
        batch_size = args.batch_size
        epochs = args.epochs
        num_workers = args.num_workers

    # Dengeleyici Örnekleyici (BalancedBatchSampler)
    try:
        train_sampler = BalancedBatchSampler(train_ds, batch_size=batch_size, seed=args.seed)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
    except Exception as e:
        logger.warning("BalancedBatchSampler oluşturulamadı (%s), standart DataLoader kullanılıyor.", e)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info("Eğitim örnek sayısı: %d | Doğrulama örnek sayısı: %d | Batch boyutu: %d",
                len(train_ds), len(val_ds), batch_size)

    # 5. Model ve Kayıp
    is_phdnet = (args.model.lower() == "phdnet")
    model, criterion = create_model_and_loss(
        model_name=args.model,
        ablation_mode=args.ablation,
        lambda_dice=args.lambda_dice,
        lambda_ccm=args.lambda_ccm,
        device=device,
    )

    # 6. Optimizer ve Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        warmup_epochs=min(2, epochs),
        total_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    # 7. Eğitim Döngüsü
    logger.info("Eğitim başlıyor: %s (ablasyon=%s, epoch=%d, lr=%.1e)",
                args.model, args.ablation, epochs, args.lr)

    best_vid_auc = -1.0
    history: List[Dict[str, Any]] = []
    best_ckpt_path = ckpt_dir / f"best_{args.model}_{args.ablation}.pt"
    history_path = log_dir / f"train_history_{args.model}_{args.ablation}.json"

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
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

        val_frame, val_video, val_loss = evaluate_validation(
            model=model,
            criterion=criterion,
            dataloader=val_loader,
            device=device,
            is_phdnet=is_phdnet,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        dt = time.perf_counter() - t0

        logger.info(
            "Epoch [%d/%d] (%.1fs) | Kayıp: Train=%.4f (BCE=%.4f, CCM=%.4f, Dice=%.4f), Val=%.4f | "
            "Kare AUC=%.4f, EER=%.4f | Video AUC=%.4f, EER=%.4f, F1=%.4f | LR=%.2e",
            epoch, epochs, dt,
            train_res["loss"], train_res["bce"], train_res["ccm"], train_res["dice"], val_loss,
            val_frame["auc"], val_frame["eer"],
            val_video["auc"], val_video["eer"], val_video["f1"],
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

        # En iyi kontrol noktası kaydı (Video-AUC bazlı)
        vid_auc = val_video["auc"]
        if vid_auc > best_vid_auc:
            best_vid_auc = vid_auc
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

        # Geçmiş JSON kaydı
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "ablation": args.ablation,
                "best_video_auc": best_vid_auc,
                "history": history,
            }, f, indent=2)

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
    parser.add_argument("--epochs", type=int, default=30,
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
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader çalışan iş parçacığı sayısı")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="CUDA üzerinde Otomatik Karma Hassasiyet (AMP) kullan")
    parser.add_argument("--no_amp", action="store_false", dest="amp",
                        help="AMP kullanımını kapat")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Hızlı 2 epoch sentetik duman testi çalıştır")
    parser.add_argument("--checkpoint_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "checkpoints"),
                        help="Model kontrol noktalarının kaydedileceği dizin")
    parser.add_argument("--log_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "logs"),
                        help="Eğitim geçmişinin kaydedileceği log dizini")
    parser.add_argument("--device", type=str, default=None,
                        help="Cihaz (cuda, cpu vb.)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Rastgelelik tohumu")

    return parser.parse_args(raw_args)


if __name__ == "__main__":
    cli_args = parse_args()
    run_training(cli_args)
