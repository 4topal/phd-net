"""PHD-Net ve Meso-4 Çoklu-Protokol Değerlendirme Hattı (Evaluation Pipeline).

Protokoller:
1. Protokol 1 — Within-Dataset (FF++ Test Ayrımı):
   - FF++ 140 test videosu (veya türetilmiş çiftleri) üzerinde:
     * `c23` sıkıştırma seviyesinde Kare ve Video AUC, EER, AP, F1, Accuracy
     * `c40` sıkıştırma seviyesinde Kare ve Video AUC, EER, AP, F1, Accuracy
     * Birleşik (`c23 + c40`) Kare ve Video metrikleri
     * Resmi maskeli alt kümelerde (`Deepfakes`, `Face2Face`) Piksel Düzeyi Dice ve IoU
2. Protokol 2 — Cross-Manipulation (Görülmemiş Manipülasyonlar):
   - Eğitimde yer almayan `FaceShifter` ve `DeepFakeDetection` (`c23` ve `c40`) üzerinde
     sıfır-atış (zero-shot) Kare ve Video AUC, EER, AP, F1
3. Protokol 3 — Cross-Dataset (Celeb-DF v2):
   - Resmi 518 test videosu (`List_of_testing_videos.txt`) üzerinde sıfır-atış
     Kare ve Video AUC, EER, AP, F1
4. Çıkarım Gecikmesi ve Hız Analizi:
   - `compute_mask=False` (hızlı tespit) ve `compute_mask=True` (adli yerelleştirme)
     modlarında medyan gecikme (ms) ve FPS ölçümü.

Tüm sonuçlar konsola özetlenir ve `outputs/logs/evaluation_report.json` dosyasına kaydedilir.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# sklearn / metrik yardımcıları
from train import (
    aggregate_video_metrics,
    compute_classification_metrics,
    compute_eer,
    create_model_and_loss,
)

# Veri seti ve yollar
from dataset import (
    DeepfakeFaceDataset,
    SyntheticSmokeDataset,
    build_or_load_ffpp_splits,
    parse_celebdf_test_list,
)
from mesonet import Meso4
from phasefreq_net import PHDNet
from project_paths import PROJECT_ROOT, get_celebdf_root, get_ffpp_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phdnet.evaluate")


# ==============================================================================
# Piksel Düzeyi Dice ve IoU Hesaplayıcı
# ==============================================================================

def compute_pixel_metrics(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Piksel düzeyinde adli yerelleştirme Dice ve IoU skorlarını hesaplar."""
    if pred_masks.numel() == 0 or target_masks.numel() == 0:
        return {"dice": 0.0, "iou": 0.0}

    probs = torch.sigmoid(pred_masks)
    binary_pred = (probs >= threshold).float()
    targets = target_masks.float()

    intersection = (binary_pred * targets).sum(dim=(-2, -1))
    pred_sum = binary_pred.sum(dim=(-2, -1))
    target_sum = targets.sum(dim=(-2, -1))

    dice_per_sample = (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
    union_per_sample = pred_sum + target_sum - intersection
    iou_per_sample = (intersection + 1e-6) / (union_per_sample + 1e-6)

    return {
        "dice": round(float(dice_per_sample.mean().item()), 4),
        "iou": round(float(iou_per_sample.mean().item()), 4),
    }


# ==============================================================================
# Veri Kümesi Üzerinde İleri Çıkarım Döngüsü
# ==============================================================================

@torch.no_grad()
def run_dataset_inference(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device = torch.device("cpu"),
    is_phdnet: bool = True,
    collect_masks: bool = False,
) -> Dict[str, Any]:
    """Verilen veri seti üzerinde model tahminlerini toplar."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_probs: List[float] = []
    all_labels: List[int] = []
    all_vids: List[str] = []
    all_manips: List[str] = []
    all_comps: List[str] = []

    pred_masks_list: List[torch.Tensor] = []
    target_masks_list: List[torch.Tensor] = []

    model.eval()

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        vids = batch["video_id"]
        manips = batch["manipulation"]
        comps = batch["compression"]

        if is_phdnet:
            out = model(rgb, compute_mask=collect_masks)
            logits = out["logits"]
            if collect_masks and out.get("mask_logits") is not None:
                has_m = batch["has_mask"]
                # Sadece resmi maskesi olan sahte örnekler için piksel skoru topla
                fake_masked_idx = torch.nonzero((batch["label"] == 1) & has_m, as_tuple=False).squeeze(-1)
                if fake_masked_idx.numel() > 0:
                    pred_masks_list.append(out["mask_logits"][fake_masked_idx].detach().cpu())
                    target_masks_list.append(batch["mask"][fake_masked_idx].cpu())
        else:
            logits = model(rgb)

        probs = torch.sigmoid(logits.float()).view(-1).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
        all_vids.extend(list(vids))
        all_manips.extend(list(manips))
        all_comps.extend(list(comps))

    res: Dict[str, Any] = {
        "probs": np.array(all_probs, dtype=float),
        "labels": np.array(all_labels, dtype=int),
        "video_ids": all_vids,
        "manipulations": all_manips,
        "compressions": all_comps,
    }

    if collect_masks and pred_masks_list:
        p_masks = torch.cat(pred_masks_list, dim=0)
        t_masks = torch.cat(target_masks_list, dim=0)
        res["pixel_metrics"] = compute_pixel_metrics(p_masks, t_masks)
    else:
        res["pixel_metrics"] = {"dice": None, "iou": None}

    return res


# ==============================================================================
# Çıkarım Gecikmesi ve Hız Benchmark'ı
# ==============================================================================

def benchmark_inference_latency(
    model: nn.Module,
    is_phdnet: bool,
    device: torch.device,
    image_size: int = 224,
    warmup: int = 15,
    runs: int = 50,
) -> Dict[str, Any]:
    """compute_mask=False ve compute_mask=True için çıkarım gecikmelerini ölçer."""
    dummy_input = torch.rand(1, 3, image_size, image_size, device=device)
    model.eval()

    def measure_fn(forward_fn) -> Dict[str, float]:
        with torch.no_grad():
            for _ in range(warmup):
                forward_fn()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            latencies = []
            for _ in range(runs):
                t0 = time.perf_counter()
                forward_fn()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latencies.append((time.perf_counter() - t0) * 1000.0)

        lat_arr = np.array(latencies)
        median_ms = float(np.median(lat_arr))
        p95_ms = float(np.percentile(lat_arr, 95))
        fps = round(1000.0 / median_ms, 1) if median_ms > 0 else 0.0
        return {
            "median_ms": round(median_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "fps": fps,
        }

    results: Dict[str, Any] = {}
    if is_phdnet:
        results["fast_detection_no_mask"] = measure_fn(lambda: model(dummy_input, compute_mask=False))
        results["forensic_localization_with_mask"] = measure_fn(lambda: model(dummy_input, compute_mask=True))
    else:
        results["baseline_meso4"] = measure_fn(lambda: model(dummy_input))

    return results


# ==============================================================================
# Protokol 1: Within-Dataset (FF++ Test Seti) Değerlendirmesi
# ==============================================================================

def evaluate_protocol1_ffpp(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    is_phdnet: bool,
) -> Dict[str, Any]:
    """Protokol 1: FF++ Test spliti üzerinde c23, c40 ve birleşik değerlendirme."""
    logger.info(">>> PROTOKOL 1: Within-Dataset (FF++ Test Ayrımı) Değerlendiriliyor...")

    if args.smoke_test:
        test_ds = SyntheticSmokeDataset(num_samples=96, target_size=224, seed=101)
    else:
        test_ds = DeepfakeFaceDataset(
            data_root=args.data_root,
            manifest_path=args.manifest_path,
            split="test",
            compressions=["c23", "c40"],
            manipulations=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
            is_train=False,
            target_size=224,
        )

    inf = run_dataset_inference(
        model=model,
        dataset=test_ds,
        batch_size=args.batch_size,
        num_workers=0 if args.smoke_test else args.num_workers,
        device=device,
        is_phdnet=is_phdnet,
        collect_masks=is_phdnet,
    )

    probs = inf["probs"]
    labels = inf["labels"]
    vids = inf["video_ids"]
    comps = np.array(inf["compressions"])
    manips = np.array(inf["manipulations"])

    # 1. Genel Birleşik (c23 + c40)
    comb_frame = compute_classification_metrics(labels, probs)
    comb_video = aggregate_video_metrics(vids, labels, probs)

    # 2. c23 Ayrımı
    idx_c23 = (comps == "c23") | (labels == 0)
    if np.sum(idx_c23) > 0:
        c23_frame = compute_classification_metrics(labels[idx_c23], probs[idx_c23])
        c23_video = aggregate_video_metrics(
            [vids[i] for i, b in enumerate(idx_c23) if b],
            labels[idx_c23],
            probs[idx_c23],
        )
    else:
        c23_frame, c23_video = comb_frame, comb_video

    # 3. c40 Ayrımı
    idx_c40 = (comps == "c40") | (labels == 0)
    if np.sum(idx_c40) > 0:
        c40_frame = compute_classification_metrics(labels[idx_c40], probs[idx_c40])
        c40_video = aggregate_video_metrics(
            [vids[i] for i, b in enumerate(idx_c40) if b],
            labels[idx_c40],
            probs[idx_c40],
        )
    else:
        c40_frame, c40_video = comb_frame, comb_video

    return {
        "samples_evaluated": len(labels),
        "combined_c23_c40": {
            "frame_metrics": comb_frame,
            "video_metrics": comb_video,
            "pixel_localization": inf["pixel_metrics"],
        },
        "compression_c23": {
            "frame_metrics": c23_frame,
            "video_metrics": c23_video,
        },
        "compression_c40": {
            "frame_metrics": c40_frame,
            "video_metrics": c40_video,
        },
    }


# ==============================================================================
# Protokol 2: Cross-Manipulation (Görülmemiş FF++ Manipülasyonları)
# ==============================================================================

def evaluate_protocol2_cross_manipulation(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    is_phdnet: bool,
) -> Dict[str, Any]:
    """Protokol 2: FaceShifter ve DeepFakeDetection üzerinde sıfır-atış değerlendirme."""
    logger.info(">>> PROTOKOL 2: Cross-Manipulation (FaceShifter & DFD) Değerlendiriliyor...")

    results: Dict[str, Any] = {}

    for unseen_manip in ["FaceShifter", "DeepFakeDetection"]:
        if args.smoke_test:
            ds = SyntheticSmokeDataset(num_samples=48, target_size=224, seed=202)
        else:
            ds = DeepfakeFaceDataset(
                data_root=args.data_root,
                manifest_path=args.manifest_path,
                split="test",
                compressions=["c23", "c40"],
                manipulations=[unseen_manip],
                is_train=False,
                target_size=224,
            )

        inf = run_dataset_inference(
            model=model,
            dataset=ds,
            batch_size=args.batch_size,
            num_workers=0 if args.smoke_test else args.num_workers,
            device=device,
            is_phdnet=is_phdnet,
            collect_masks=False,
        )

        frame_m = compute_classification_metrics(inf["labels"], inf["probs"])
        video_m = aggregate_video_metrics(inf["video_ids"], inf["labels"], inf["probs"])

        results[unseen_manip] = {
            "samples": len(inf["labels"]),
            "frame_metrics": frame_m,
            "video_metrics": video_m,
        }

    return results


# ==============================================================================
# Protokol 3: Cross-Dataset (Celeb-DF v2)
# ==============================================================================

def evaluate_protocol3_celebdf(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    is_phdnet: bool,
) -> Dict[str, Any]:
    """Protokol 3: Celeb-DF v2 resmi 518 test videosu üzerinde sıfır-atış değerlendirme."""
    logger.info(">>> PROTOKOL 3: Cross-Dataset (Celeb-DF v2) Değerlendiriliyor...")

    if args.smoke_test:
        celeb_ds = SyntheticSmokeDataset(num_samples=64, target_size=224, seed=303)
        sample_count = len(celeb_ds)
    else:
        test_entries = parse_celebdf_test_list(
            celebdf_root=args.celebdf_root,
            test_list_path=args.celebdf_test_list,
        )
        sample_count = len(test_entries)
        test_vid_ids = {str(e.get("video_id", "")) for e in test_entries}
        data_root_path = Path(args.data_root) if args.data_root else PROJECT_ROOT / "data" / "processed_faces"
        celeb_manifest = data_root_path / "manifest_celebdf.json"
        full_celeb_ds = DeepfakeFaceDataset(
            manifest_path=celeb_manifest if celeb_manifest.is_file() else None,
            data_root=data_root_path,
            manipulations=["Celeb-real", "YouTube-real", "Celeb-synthesis", "celebdf_real", "celebdf_synthesis"],
            is_train=False,
            target_size=224,
        )
        if len(full_celeb_ds.samples) > 0:
            filtered_samples = [
                s for s in full_celeb_ds.samples
                if not test_vid_ids or str(s.get("video_id", "")) in test_vid_ids
            ]
            if filtered_samples:
                full_celeb_ds.samples = filtered_samples
            celeb_ds = full_celeb_ds
        else:
            celeb_ds = DeepfakeFaceDataset(
                samples=test_entries,
                data_root=args.data_root,
                is_train=False,
                target_size=224,
            )

    inf = run_dataset_inference(
        model=model,
        dataset=celeb_ds,
        batch_size=args.batch_size,
        num_workers=0 if args.smoke_test else args.num_workers,
        device=device,
        is_phdnet=is_phdnet,
        collect_masks=False,
    )

    frame_m = compute_classification_metrics(inf["labels"], inf["probs"])
    video_m = aggregate_video_metrics(inf["video_ids"], inf["labels"], inf["probs"])

    return {
        "videos_in_test_list": sample_count,
        "samples_evaluated": len(inf["labels"]),
        "frame_metrics": frame_m,
        "video_metrics": video_m,
    }


# ==============================================================================
# Ana Değerlendirme Fonksiyonu
# ==============================================================================

def run_full_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Tüm değerlendirme protokollerini koşturur ve JSON raporunu kaydeder."""
    # 1. Cihaz
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Model ve Kontrol Noktası Yükleme
    model_name = args.model
    ablation_mode = args.ablation

    checkpoint_info: Dict[str, Any] = {}
    if args.checkpoint and Path(args.checkpoint).is_file():
        logger.info("Kontrol noktası yükleniyor: %s", args.checkpoint)
        ckpt_data = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt_data, dict):
            model_name = ckpt_data.get("model_name", model_name)
            ablation_mode = ckpt_data.get("ablation_mode", ablation_mode)
            state_dict = ckpt_data.get("model_state_dict", ckpt_data)
            checkpoint_info = {
                "checkpoint_path": str(args.checkpoint),
                "trained_epoch": ckpt_data.get("epoch"),
                "best_train_video_auc": ckpt_data.get("best_video_auc"),
            }
        else:
            state_dict = ckpt_data

        model, _ = create_model_and_loss(model_name=model_name, ablation_mode=ablation_mode, device=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.info("Kontrol noktası verilmedi veya bulunamadı, model rastgele ağırlıklarla başlatılıyor (smoke_test=%s)", args.smoke_test)
        model, _ = create_model_and_loss(model_name=model_name, ablation_mode=ablation_mode, device=device)

    is_phdnet = (model_name.lower() == "phdnet")
    model.eval()

    # 3. Protokol 1: Within-Dataset (FF++)
    proto1 = evaluate_protocol1_ffpp(model, args, device, is_phdnet)

    # 4. Protokol 2: Cross-Manipulation
    proto2 = evaluate_protocol2_cross_manipulation(model, args, device, is_phdnet)

    # 5. Protokol 3: Cross-Dataset (Celeb-DF v2)
    proto3 = evaluate_protocol3_celebdf(model, args, device, is_phdnet)

    # 6. Çıkarım Gecikmesi ve Hız Analizi
    logger.info(">>> Çıkarım Gecikmesi ve Hız Benchmark'ı Yapılıyor...")
    speed_rep = benchmark_inference_latency(model, is_phdnet=is_phdnet, device=device)

    # 7. Tam Raporun Derlenmesi
    full_report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "ablation": ablation_mode,
        "smoke_test": args.smoke_test,
        "device": str(device),
        "checkpoint_info": checkpoint_info,
        "protocol_1_within_dataset_ffpp": proto1,
        "protocol_2_cross_manipulation": proto2,
        "protocol_3_cross_dataset_celebdf": proto3,
        "inference_speed_benchmark": speed_rep,
    }

    # 8. Raporu Diske Kaydet
    rep_path = Path(args.output_report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Değerlendirme tamamlandı! Tam rapor kaydedildi: %s", rep_path)

    # 9. Konsol Özet Tablosu
    print("\n" + "=" * 80)
    print(f"  PHD-Net DEĞERLENDİRME RAPORU — Model: {model_name.upper()} (Ablasyon: {ablation_mode})")
    print("=" * 80)
    p1_comb = proto1["combined_c23_c40"]["video_metrics"]
    p1_c23 = proto1["compression_c23"]["video_metrics"]
    p1_c40 = proto1["compression_c40"]["video_metrics"]
    p1_pix = proto1["combined_c23_c40"]["pixel_localization"]

    print("\n--- PROTOKOL 1: FF++ Within-Dataset (Video-Level Metrikler) ---")
    print(f"  Birleşik (c23 + c40) : Video-AUC: {p1_comb['auc']:.4f} | EER: {p1_comb['eer']:.4f} | AP: {p1_comb['ap']:.4f} | F1: {p1_comb['f1']:.4f}")
    print(f"  c23 (Yüksek Kalite)  : Video-AUC: {p1_c23['auc']:.4f} | EER: {p1_c23['eer']:.4f} | AP: {p1_c23['ap']:.4f}")
    print(f"  c40 (Ağır Sıkıştırma): Video-AUC: {p1_c40['auc']:.4f} | EER: {p1_c40['eer']:.4f} | AP: {p1_c40['ap']:.4f}")
    if p1_pix.get("dice") is not None:
        print(f"  Piksel Yerelleştirme : Dice Skoru: {p1_pix['dice']:.4f} | IoU: {p1_pix['iou']:.4f}")

    print("\n--- PROTOKOL 2: Cross-Manipulation (Sıfır-Atış Video-Level) ---")
    for k, v in proto2.items():
        vm = v["video_metrics"]
        print(f"  {k:<18} : Video-AUC: {vm['auc']:.4f} | EER: {vm['eer']:.4f} | AP: {vm['ap']:.4f}")

    print("\n--- PROTOKOL 3: Celeb-DF v2 Cross-Dataset (Sıfır-Atış) ---")
    p3_vid = proto3["video_metrics"]
    p3_frame = proto3["frame_metrics"]
    print(f"  Celeb-DF Video-Level : Video-AUC: {p3_vid['auc']:.4f} | EER: {p3_vid['eer']:.4f} | AP: {p3_vid['ap']:.4f}")
    print(f"  Celeb-DF Frame-Level : Kare-AUC : {p3_frame['auc']:.4f} | EER: {p3_frame['eer']:.4f}")

    print("\n--- ÇIKARIM HIZI VE GECİKME ---")
    for mode_name, m_res in speed_rep.items():
        print(f"  {mode_name:<30}: Medyan={m_res['median_ms']} ms | p95={m_res['p95_ms']} ms | {m_res['fps']} FPS")
    print("=" * 80 + "\n")

    return full_report


# ==============================================================================
# CLI ve Giriş Noktası
# ==============================================================================

def parse_args(raw_args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHD-Net / Meso-4 Çoklu-Protokol Değerlendirme Betiği")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Değerlendirilecek model kontrol noktası (.pt dosyası)")
    parser.add_argument("--model", type=str, default="phdnet", choices=["phdnet", "meso4"],
                        help="Model mimarisi")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_qpc", "no_lfg", "no_dfsc", "no_ccm", "classical_freq"],
                        help="Ablasyon varyantı")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Sentetik duman testi ile tüm değerlendirme hatlarını koştur")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Kırpılmış yüzler veri kök dizini")
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Önceden üretilmiş manifest.json yolu")
    parser.add_argument("--ffpp_root", type=str, default=None,
                        help="FF++ veri seti kök dizini")
    parser.add_argument("--celebdf_root", type=str, default=None,
                        help="Celeb-DF v2 veri seti kök dizini")
    parser.add_argument("--celebdf_test_list", type=str, default=None,
                        help="Celeb-DF resmi test listesi dosyası (List_of_testing_videos.txt)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Değerlendirme batch boyutu")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader iş parçacığı sayısı")
    parser.add_argument("--device", type=str, default=None,
                        help="Değerlendirme cihazı (cuda, cpu vb.)")
    parser.add_argument("--output_report", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "logs" / "evaluation_report.json"),
                        help="Çıktı JSON raporunun kaydedileceği dosya yolu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Rastgelelik tohumu")

    return parser.parse_args(raw_args)


if __name__ == "__main__":
    cli_args = parse_args()
    run_full_evaluation(cli_args)
