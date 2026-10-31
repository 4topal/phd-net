"""Faz 2 Zorunlu Doğrulama Deneyleri: D2 (HER Sıkıştırma Testi) ve D3 (Görsel Panel).

Deneyler:
- D2 (`outputs/logs/d2_her_compression.json` + `outputs/figures/d2_her_crf_analysis.png`):
  FF++ C23 `original/` (5 video), `Deepfakes/` (5 eşleşen video) ve `Face2Face/` (5 eşleşen video)
  üzerinde `libx264` CRF {0, 18, 23, 32, 40} yeniden kodlamasıyla aynı kare ve sabit yüz kutularında
  HER (`sum|mid| / (sum|low| + eps)`) değişimini, video başına değişim katsayısını (`CV = std/mean`)
  ve gerçek/sahte ayrışmasını ölçer.
  METODOLOJİK NOT: Bu test mevcut C23 kaynaklarının `libx264` ile yeniden kodlanmasına dayalı bir CRF
  yaklaşımıdır; sıkıştırılmamış ham kaynaktan üretilmiş resmi FF++ C0 / C40 değildir.

- D3 (`outputs/figures/qpc_her_panel.png` + `outputs/logs/d3_panel_metadata.json`):
  Deterministik kare seçimiyle (alfabetik 1. Celeb-real videosu `id0_0000.mp4` kare #0 ve aynı kaynak
  sahnenin 1. sentetik karşılığı `id0_id16_0000.mp4` kare #0 — elle "en belirgin kare" seçilmemiştir)
  RGB | QPC haritası | 8x8 DCT bant enerjisi | CRF'ye göre HER grafiği panelini üretir.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from face_extractor import DEFAULT_MODEL_PATH, YuNetFaceExtractor, crop_face_with_margin
from learnable_freq_gate import HarmonicEnergyRatio
from phase_congruency import MonochromePhaseCongruency, QuaternionPhaseCongruency

KOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = KOD_ROOT.parent
TMP_CRF_DIR = KOD_ROOT / "outputs" / "tmp_crf"
D2_LOG_PATH = KOD_ROOT / "outputs" / "logs" / "d2_her_compression.json"
D3_LOG_PATH = KOD_ROOT / "outputs" / "logs" / "d3_panel_metadata.json"
D2_FIG_PATH = KOD_ROOT / "outputs" / "figures" / "d2_her_crf_analysis.png"
D3_FIG_PATH = KOD_ROOT / "outputs" / "figures" / "qpc_her_panel.png"

CRFS = [0, 18, 23, 32, 40]
CORE_CRFS = [0, 23, 40]


def read_specific_frames(video_path: Path, frame_indices: List[int]) -> Dict[int, np.ndarray]:
    """Videodan belirtilen kare indekslerini BGR olarak okur."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    target_set = set(frame_indices)
    max_idx = max(frame_indices) if frame_indices else 0
    frames: Dict[int, np.ndarray] = {}
    idx = 0
    while idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in target_set:
            frames[idx] = frame
        idx += 1
    cap.release()
    return frames


def bgr_crops_to_rgb_tensor(crops_bgr: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """224x224 BGR yüz listesini `(N, 3, 224, 224)` `[0, 1]` RGB tensörüne çevirir."""
    tensors = []
    for bgr in crops_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensors.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    return torch.stack(tensors, dim=0).to(device)


def compute_cv_percent(values: List[float]) -> float:
    """Değişim katsayısı `CV (%) = (std / mean) * 100` (örneklem std değil, CRF seviyeleri arası std/mean)."""
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return 0.0
    std = float(np.std(arr, ddof=0))
    return (std / mean) * 100.0


def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """İki grup arasındaki Cohen's d etki büyüklüğü `(mean_b - mean_a) / s_pooled`."""
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / max(len(a) + len(b) - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled)


def analyze_video_across_crfs(
    source_video: Path,
    crf_dir: Path,
    stem: str,
    extractor: YuNetFaceExtractor,
    her_calc: HarmonicEnergyRatio,
    device: torch.device,
    sample_indices: List[int],
) -> Dict[str, Any]:
    """Kaynak videoda yüz kutularını sabitleyip tüm CRF seviyelerinde aynı karelerin HER değerini ölçer."""
    src_frames = read_specific_frames(source_video, sample_indices)
    valid_bboxes: Dict[int, Tuple[float, float, float, float]] = {}
    src_crops: List[np.ndarray] = []
    valid_indices: List[int] = []

    for idx in sample_indices:
        if idx not in src_frames:
            continue
        det = extractor.detect_primary_face(src_frames[idx])
        if det is None:
            continue
        bbox, _ = det
        valid_bboxes[idx] = bbox
        valid_indices.append(idx)
        src_crops.append(crop_face_with_margin(src_frames[idx], bbox, margin_ratio=0.20, target_size=224))

    if not valid_indices:
        raise RuntimeError(f"Hiçbir örnek karede yüz bulunamadı: {source_video}")

    def _eval_crops(crops: List[np.ndarray]) -> Dict[str, Any]:
        batch = bgr_crops_to_rgb_tensor(crops, device)
        with torch.no_grad():
            st = her_calc(batch)
            her_vals = st["her_global"].cpu().numpy()
            low_vals = st["low_energy"].sum(dim=(1, 2, 3)).cpu().numpy()
            mid_vals = st["mid_energy"].sum(dim=(1, 2, 3)).cpu().numpy()
            high_vals = st["high_energy"].sum(dim=(1, 2, 3)).cpu().numpy()
            hfer_vals = high_vals / (low_vals + 1e-6)
        return {
            "her_mean": round(float(np.mean(her_vals)), 6),
            "her_std_across_frames": round(float(np.std(her_vals, ddof=0)), 6),
            "her_per_frame": [round(float(x), 6) for x in her_vals],
            "low_energy_mean": round(float(np.mean(low_vals)), 4),
            "mid_energy_mean": round(float(np.mean(mid_vals)), 4),
            "high_energy_mean": round(float(np.mean(high_vals)), 4),
            "hfer_mean": round(float(np.mean(hfer_vals)), 6),
        }

    results_by_level: Dict[str, Any] = {"source_c23": _eval_crops(src_crops)}

    for crf in CRFS:
        crf_video = crf_dir / f"{stem}_crf{crf}.mp4"
        crf_frames = read_specific_frames(crf_video, valid_indices)
        crops_crf = [
            crop_face_with_margin(crf_frames[i], valid_bboxes[i], margin_ratio=0.20, target_size=224)
            for i in valid_indices
            if i in crf_frames
        ]
        results_by_level[f"crf_{crf}"] = _eval_crops(crops_crf)

    her_core = [results_by_level[f"crf_{c}"]["her_mean"] for c in CORE_CRFS]
    her_all = [results_by_level[f"crf_{c}"]["her_mean"] for c in CRFS]
    mid_core = [results_by_level[f"crf_{c}"]["mid_energy_mean"] for c in CORE_CRFS]
    high_core = [results_by_level[f"crf_{c}"]["high_energy_mean"] for c in CORE_CRFS]
    hfer_core = [results_by_level[f"crf_{c}"]["hfer_mean"] for c in CORE_CRFS]

    her_0 = results_by_level["crf_0"]["her_mean"]
    her_23 = results_by_level["crf_23"]["her_mean"]
    her_40 = results_by_level["crf_40"]["her_mean"]
    rel_change_0_to_40 = ((her_40 - her_0) / her_0 * 100.0) if her_0 > 0 else 0.0
    rel_change_0_to_23 = ((her_23 - her_0) / her_0 * 100.0) if her_0 > 0 else 0.0

    return {
        "video": stem,
        "source_path": str(source_video),
        "matched_frame_indices": valid_indices,
        "num_frames": len(valid_indices),
        "levels": results_by_level,
        "variation_summary": {
            "her_cv_percent_crf_0_23_40": round(compute_cv_percent(her_core), 3),
            "her_cv_percent_all_5_crfs": round(compute_cv_percent(her_all), 3),
            "mid_energy_cv_percent_crf_0_23_40": round(compute_cv_percent(mid_core), 3),
            "high_energy_cv_percent_crf_0_23_40": round(compute_cv_percent(high_core), 3),
            "hfer_high_over_low_cv_percent_crf_0_23_40": round(compute_cv_percent(hfer_core), 3),
            "her_rel_change_crf0_to_crf23_percent": round(rel_change_0_to_23, 3),
            "her_rel_change_crf0_to_crf40_percent": round(rel_change_0_to_40, 3),
        },
    }


def run_d2_experiment(
    extractor: YuNetFaceExtractor,
    her_calc: HarmonicEnergyRatio,
    device: torch.device,
) -> Dict[str, Any]:
    """D2 FF++ CRF yeniden kodlama deneyini yürütür ve sonuçları JSON + grafik olarak kaydeder."""
    ff_base = REPO_ROOT / "datasets" / "ffc32" / "FaceForensics++_C23"
    sample_indices = [0, 15, 30, 45, 60, 75, 90, 105]

    orig_stems = ["000", "001", "002", "003", "004"]
    fake_stems = ["000_003", "001_870", "002_006", "003_000", "004_982"]

    groups: Dict[str, List[Dict[str, Any]]] = {
        "original": [],
        "Deepfakes": [],
        "Face2Face": [],
    }

    for st in orig_stems:
        groups["original"].append(
            analyze_video_across_crfs(
                ff_base / "original" / f"{st}.mp4",
                TMP_CRF_DIR / "ffpp_original",
                st,
                extractor,
                her_calc,
                device,
                sample_indices,
            )
        )
    for st in fake_stems:
        groups["Deepfakes"].append(
            analyze_video_across_crfs(
                ff_base / "Deepfakes" / f"{st}.mp4",
                TMP_CRF_DIR / "ffpp_deepfakes",
                st,
                extractor,
                her_calc,
                device,
                sample_indices,
            )
        )
        groups["Face2Face"].append(
            analyze_video_across_crfs(
                ff_base / "Face2Face" / f"{st}.mp4",
                TMP_CRF_DIR / "ffpp_face2face",
                st,
                extractor,
                her_calc,
                device,
                sample_indices,
            )
        )

    # Grup özetleri ve Real/Fake ayrışma analizi
    group_summaries: Dict[str, Any] = {}
    for gname, vids in groups.items():
        cvs_core = [v["variation_summary"]["her_cv_percent_crf_0_23_40"] for v in vids]
        cvs_mid = [v["variation_summary"]["mid_energy_cv_percent_crf_0_23_40"] for v in vids]
        cvs_high = [v["variation_summary"]["high_energy_cv_percent_crf_0_23_40"] for v in vids]
        cvs_hfer = [v["variation_summary"]["hfer_high_over_low_cv_percent_crf_0_23_40"] for v in vids]
        drops_40 = [v["variation_summary"]["her_rel_change_crf0_to_crf40_percent"] for v in vids]
        drops_23 = [v["variation_summary"]["her_rel_change_crf0_to_crf23_percent"] for v in vids]

        level_means = {}
        for lvl in ["source_c23"] + [f"crf_{c}" for c in CRFS]:
            vals = [v["levels"][lvl]["her_mean"] for v in vids]
            level_means[lvl] = {
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals, ddof=1)), 6),
            }

        group_summaries[gname] = {
            "num_videos": len(vids),
            "mean_her_cv_percent_crf_0_23_40": round(float(np.mean(cvs_core)), 3),
            "min_max_her_cv_percent_crf_0_23_40": [round(float(np.min(cvs_core)), 3), round(float(np.max(cvs_core)), 3)],
            "mean_raw_mid_energy_cv_percent": round(float(np.mean(cvs_mid)), 3),
            "mean_raw_high_energy_cv_percent": round(float(np.mean(cvs_high)), 3),
            "mean_hfer_cv_percent": round(float(np.mean(cvs_hfer)), 3),
            "mean_her_rel_change_crf0_to_crf23_percent": round(float(np.mean(drops_23)), 3),
            "mean_her_rel_change_crf0_to_crf40_percent": round(float(np.mean(drops_40)), 3),
            "her_by_level": level_means,
        }

    separation_by_level: Dict[str, Any] = {}
    for lvl in ["source_c23"] + [f"crf_{c}" for c in CRFS]:
        real_v = [v["levels"][lvl]["her_mean"] for v in groups["original"]]
        df_v = [v["levels"][lvl]["her_mean"] for v in groups["Deepfakes"]]
        f2f_v = [v["levels"][lvl]["her_mean"] for v in groups["Face2Face"]]
        all_fake_v = df_v + f2f_v

        separation_by_level[lvl] = {
            "real_mean": round(float(np.mean(real_v)), 6),
            "real_std": round(float(np.std(real_v, ddof=1)), 6),
            "deepfakes_mean": round(float(np.mean(df_v)), 6),
            "face2face_mean": round(float(np.mean(f2f_v)), 6),
            "all_fake_mean": round(float(np.mean(all_fake_v)), 6),
            "all_fake_std": round(float(np.std(all_fake_v, ddof=1)), 6),
            "diff_fake_minus_real": round(float(np.mean(all_fake_v) - np.mean(real_v)), 6),
            "cohens_d_real_vs_all_fake": round(cohens_d(real_v, all_fake_v), 4),
            "cohens_d_real_vs_deepfakes": round(cohens_d(real_v, df_v), 4),
            "cohens_d_real_vs_face2face": round(cohens_d(real_v, f2f_v), 4),
        }

    # D2 Grafiği
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    styles = {
        "original": ("#1b9e77", "o-", "FF++ Original (Gerçek, n=5)"),
        "Deepfakes": ("#d95f02", "s--", "FF++ Deepfakes (Sahte, n=5)"),
        "Face2Face": ("#7570b3", "^-.", "FF++ Face2Face (Sahte, n=5)"),
    }
    for gname, (color, marker_style, label) in styles.items():
        means = [group_summaries[gname]["her_by_level"][f"crf_{c}"]["mean"] for c in CRFS]
        stds = [group_summaries[gname]["her_by_level"][f"crf_{c}"]["std"] for c in CRFS]
        axes[0].plot(CRFS, means, marker_style, color=color, linewidth=2.0, markersize=6, label=label)
        axes[0].fill_between(
            CRFS,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=color,
            alpha=0.14,
        )
    axes[0].set_title("FF++ Yeniden Kodlama (CRF) Altında HER Değişimi", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("libx264 CRF Değeri (Kaynak: FF++ C23)")
    axes[0].set_ylabel("HER = Σ|Mid| / (Σ|Low| + ε)")
    axes[0].set_xticks(CRFS)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)

    # Sağ grafik: Değişim Katsayısı (CV%) karşılaştırması — HER vs Ham Mid vs Ham High vs HFER
    metrics_labels = ["HER\n(Mid/Low)", "Ham Mid\nEnerji", "Ham High\nEnerji", "HFER\n(High/Low)"]
    x_pos = np.arange(len(metrics_labels))
    width = 0.25
    for idx, (gname, (color, _, _)) in enumerate(styles.items()):
        gs = group_summaries[gname]
        vals = [
            gs["mean_her_cv_percent_crf_0_23_40"],
            gs["mean_raw_mid_energy_cv_percent"],
            gs["mean_raw_high_energy_cv_percent"],
            gs["mean_hfer_cv_percent"],
        ]
        bars = axes[1].bar(x_pos + (idx - 1) * width, vals, width=width, color=color, alpha=0.88, label=gname)
        for bar in bars:
            h_val = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                h_val + 0.6,
                f"{h_val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    max_bar = max(
        max(
            group_summaries[g]["mean_her_cv_percent_crf_0_23_40"],
            group_summaries[g]["mean_raw_mid_energy_cv_percent"],
            group_summaries[g]["mean_raw_high_energy_cv_percent"],
            group_summaries[g]["mean_hfer_cv_percent"],
        )
        for g in styles
    )
    axes[1].set_ylim(0.0, max_bar * 1.18)
    axes[1].set_title("CRF {0, 23, 40} Arası Değişim Katsayısı (CV%)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(metrics_labels)
    axes[1].set_ylabel("Video Başına Ortalama CV (%) — Düşük Daha Kararlı")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9)

    fig.suptitle(
        "D2 Deneyi: H.264 Kuantizasyonu Altında HER İnvaryans İddiasının Ampirik Testi (CRF Yaklaşımı)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    D2_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(D2_FIG_PATH, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "experiment": "D2 — HER Sıkıştırma Dayanıklılığı ve Gerçek/Sahte Ayrışma Testi",
        "methodological_note": (
            "Bu deney, yerel diskte yalnızca FaceForensics++ C23 bulunduğu için C23 videolarının "
            "ffmpeg libx264 ile CRF {0, 18, 23, 32, 40} seviyelerinde yeniden kodlanmasına dayalı bir "
            "CRF yaklaşımıdır (proxy). Sıkıştırılmamış ham kaynaktan üretilmiş resmi FF++ C0 (RAW) ve "
            "C40 veri setleri değildir."
        ),
        "dct_band_config": {"low_band": "1 <= u+v <= 3 (DC hariç)", "mid_band": "4 <= u+v <= 8", "high_band": "9 <= u+v <= 14"},
        "crfs_tested": CRFS,
        "core_crfs_for_cv": CORE_CRFS,
        "group_summaries": group_summaries,
        "real_vs_fake_separation": separation_by_level,
        "per_video_details": groups,
        "figure_path": str(D2_FIG_PATH),
    }
    D2_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    D2_LOG_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def run_d3_visual_panel(
    extractor: YuNetFaceExtractor,
    her_calc: HarmonicEnergyRatio,
    device: torch.device,
) -> Dict[str, Any]:
    """D3 görsel panelini (`outputs/figures/qpc_her_panel.png`) üretir ve meta verisini loglar."""
    celeb_base = REPO_ROOT / "datasets" / "celebdf"
    real_vid = celeb_base / "Celeb-real" / "id0_0000.mp4"
    fake_vid = celeb_base / "Celeb-synthesis" / "id0_id16_0000.mp4"
    selected_frame_idx = 0

    # 1. CRF eğrileri için her iki Celeb-DF videosunu analiz et (kare #0 dahil 8 kare)
    sample_indices = [0, 15, 30, 45, 60, 75, 90, 105]
    real_crf_stats = analyze_video_across_crfs(
        real_vid, TMP_CRF_DIR / "celebdf", "id0_0000", extractor, her_calc, device, sample_indices
    )
    fake_crf_stats = analyze_video_across_crfs(
        fake_vid, TMP_CRF_DIR / "celebdf", "id0_id16_0000", extractor, her_calc, device, sample_indices
    )

    # 2. Kare #0 için RGB, QPC ve 8x8 DCT alt-bant spektrumunu çıkar
    real_f0_bgr = read_specific_frames(real_vid, [selected_frame_idx])[selected_frame_idx]
    fake_f0_bgr = read_specific_frames(fake_vid, [selected_frame_idx])[selected_frame_idx]

    real_crop_bgr = extractor.extract_face_crop(real_f0_bgr)
    fake_crop_bgr = extractor.extract_face_crop(fake_f0_bgr)
    assert real_crop_bgr is not None and fake_crop_bgr is not None, "Kare #0 yüz tespiti başarısız!"

    real_rgb_u8 = cv2.cvtColor(real_crop_bgr, cv2.COLOR_BGR2RGB)
    fake_rgb_u8 = cv2.cvtColor(fake_crop_bgr, cv2.COLOR_BGR2RGB)

    pair_tensor = bgr_crops_to_rgb_tensor([real_crop_bgr, fake_crop_bgr], device)
    qpc_mod = QuaternionPhaseCongruency().to(device)
    mono_mod = MonochromePhaseCongruency().to(device)

    with torch.no_grad():
        qpc_pair = qpc_mod(pair_tensor)  # (2, 3, 224, 224)
        qpc_vis = QuaternionPhaseCongruency.summary_map(qpc_pair).squeeze(1).cpu().numpy()  # (2, 224, 224)
        mono_vis = mono_mod(pair_tensor).squeeze(1).cpu().numpy()
        chrom_disp = qpc_pair[:, 2].cpu().numpy()  # (2, 224, 224)

        her_out = her_calc(pair_tensor)
        # Kanal ortalamalı 8x8 mutlak DCT spektrumu -> log10(1 + |DCT|*100) görselleştirme için
        spec_8x8 = her_out["mean_abs_spectrum"].mean(dim=1).cpu().numpy()  # (2, 8, 8)
        log_spec = np.log10(1.0 + spec_8x8 * 100.0)
        # DC (0,0) görsel kontrastı ezmesin diye ayrı etiketlenecek, renk ölçeği AC katsayılarına göre ayarlanacak
        her_vals_f0 = her_out["her_global"].cpu().numpy()
        low_e_f0 = her_out["low_energy"].sum(dim=(1, 2, 3)).cpu().numpy()
        mid_e_f0 = her_out["mid_energy"].sum(dim=(1, 2, 3)).cpu().numpy()
        high_e_f0 = her_out["high_energy"].sum(dim=(1, 2, 3)).cpu().numpy()

    qpc_vmax = float(max(np.percentile(qpc_vis, 99.5), 0.25))
    ac_mask = np.ones((8, 8), dtype=bool)
    ac_mask[0, 0] = False
    dct_vmax = float(max(log_spec[0][ac_mask].max(), log_spec[1][ac_mask].max()))

    fig, axes = plt.subplots(2, 4, figsize=(18.5, 8.8), dpi=180)

    rows_meta = [
        ("Celeb-real (Gerçek)\nid0_0000.mp4 — Kare #0", real_rgb_u8, 0),
        ("Celeb-synthesis (Sahte)\nid0_id16_0000.mp4 — Kare #0", fake_rgb_u8, 1),
    ]

    def _draw_band_boundaries(ax: Any) -> None:
        """8x8 matris üzerinde u+v=3.5 (Low/Mid) ve u+v=8.5 (Mid/High) merdiven sınırlarını çizer."""
        # DC kutusu (0,0)
        ax.plot([-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], color="#00ffff", linewidth=1.8, linestyle="-")
        # Low/Mid sınırı: u + v <= 3
        xs_lm, ys_lm = [], []
        for u in range(0, 5):
            v = 4 - u
            xs_lm.extend([v - 0.5, v - 0.5])
            ys_lm.extend([u - 0.5, u + 0.5])
        ax.plot(xs_lm[:-1], ys_lm[:-1], color="#00ff88", linewidth=2.0, linestyle="--")
        # Mid/High sınırı: u + v <= 8
        xs_mh, ys_mh = [], []
        for u in range(1, 8):
            v = 9 - u
            if 0 <= v <= 8:
                xs_mh.extend([v - 0.5, v - 0.5])
                ys_mh.extend([u - 0.5, u + 0.5])
        ax.plot(xs_mh[:-1], ys_mh[:-1], color="#ff5555", linewidth=2.0, linestyle="--")

    for r_idx, (title_prefix, rgb_img, idx) in enumerate(rows_meta):
        # Sütun 0: RGB
        axes[r_idx, 0].imshow(rgb_img)
        axes[r_idx, 0].set_title(f"{title_prefix}\n224×224 RGB (%20 Marj)", fontsize=10, fontweight="bold")
        axes[r_idx, 0].axis("off")

        # Sütun 1: QPC Haritası
        im_qpc = axes[r_idx, 1].imshow(qpc_vis[idx], cmap="magma", vmin=0.0, vmax=qpc_vmax)
        axes[r_idx, 1].set_title(
            f"QPC Haritası (Log-Gabor Kuaterniyon)\n"
            f"QPC ort.={qpc_vis[idx].mean():.4f} | ΔΦ_chrom ort.={chrom_disp[idx].mean():.5f}",
            fontsize=9.5,
            fontweight="bold",
        )
        axes[r_idx, 1].axis("off")
        plt.colorbar(im_qpc, ax=axes[r_idx, 1], fraction=0.046, pad=0.03)

        # Sütun 2: 8x8 DCT Bant Enerjisi
        disp_mat = log_spec[idx].copy()
        dc_orig = disp_mat[0, 0]
        disp_mat[0, 0] = dct_vmax  # DC'yi AC skalasında doygun göster, gerçek değerini hücreye yaz
        im_dct = axes[r_idx, 2].imshow(disp_mat, cmap="viridis", vmin=0.0, vmax=dct_vmax)
        _draw_band_boundaries(axes[r_idx, 2])
        axes[r_idx, 2].text(0, 0, f"DC\n{dc_orig:.1f}", ha="center", va="center", color="black", fontsize=7, fontweight="bold")
        axes[r_idx, 2].text(1.2, 1.0, "Low", color="white", fontsize=8, fontweight="bold")
        axes[r_idx, 2].text(3.5, 3.2, "Mid", color="white", fontsize=8, fontweight="bold")
        axes[r_idx, 2].text(6.0, 6.2, "High", color="white", fontsize=8, fontweight="bold")
        axes[r_idx, 2].set_xticks(range(8))
        axes[r_idx, 2].set_yticks(range(8))
        axes[r_idx, 2].set_xlabel("Yatay Frekans v")
        axes[r_idx, 2].set_ylabel("Düşey Frekans u")
        axes[r_idx, 2].set_title(
            f"8×8 DCT Bant Enerjisi log10(1+100|DCT|)\n"
            f"HER={her_vals_f0[idx]:.3f} (Low={low_e_f0[idx]:.1f}, Mid={mid_e_f0[idx]:.1f}, High={high_e_f0[idx]:.1f})",
            fontsize=9.2,
            fontweight="bold",
        )
        plt.colorbar(im_dct, ax=axes[r_idx, 2], fraction=0.046, pad=0.03)

    # Sütun 3 (Üst): Kare #0 için CRF -> HER Grafiği (Celeb-real vs Celeb-synthesis)
    real_f0_crf = [real_crf_stats["levels"][f"crf_{c}"]["her_per_frame"][0] for c in CRFS]
    fake_f0_crf = [fake_crf_stats["levels"][f"crf_{c}"]["her_per_frame"][0] for c in CRFS]
    axes[0, 3].plot(CRFS, real_f0_crf, "o-", color="#1b9e77", linewidth=2.2, markersize=6, label=f"Celeb-real Kare #0 (CV={compute_cv_percent([real_f0_crf[0], real_f0_crf[2], real_f0_crf[4]]):.1f}%)")
    axes[0, 3].plot(CRFS, fake_f0_crf, "s--", color="#d95f02", linewidth=2.2, markersize=6, label=f"Celeb-synth Kare #0 (CV={compute_cv_percent([fake_f0_crf[0], fake_f0_crf[2], fake_f0_crf[4]]):.1f}%)")
    axes[0, 3].axhline(real_crf_stats["levels"]["source_c23"]["her_per_frame"][0], color="#1b9e77", linestyle=":", alpha=0.6, label="Kaynak Video (Real)")
    axes[0, 3].axhline(fake_crf_stats["levels"]["source_c23"]["her_per_frame"][0], color="#d95f02", linestyle=":", alpha=0.6, label="Kaynak Video (Synth)")
    axes[0, 3].set_title("Seçilen Kare (#0) — CRF'ye Göre HER", fontsize=10, fontweight="bold")
    axes[0, 3].set_xlabel("libx264 CRF Değeri")
    axes[0, 3].set_ylabel("HER = Σ|Mid| / (Σ|Low| + ε)")
    axes[0, 3].set_xticks(CRFS)
    axes[0, 3].grid(True, linestyle=":", alpha=0.6)
    axes[0, 3].legend(fontsize=8, loc="best")

    # Sütun 3 (Alt): Video Geneli (8 Kare Ort. ± Std) CRF -> HER Grafiği
    real_m_crf = [real_crf_stats["levels"][f"crf_{c}"]["her_mean"] for c in CRFS]
    real_s_crf = [real_crf_stats["levels"][f"crf_{c}"]["her_std_across_frames"] for c in CRFS]
    fake_m_crf = [fake_crf_stats["levels"][f"crf_{c}"]["her_mean"] for c in CRFS]
    fake_s_crf = [fake_crf_stats["levels"][f"crf_{c}"]["her_std_across_frames"] for c in CRFS]

    axes[1, 3].plot(
        CRFS,
        real_m_crf,
        "o-",
        color="#1b9e77",
        linewidth=2.2,
        markersize=6,
        label=f"Celeb-real Ort. (CV={real_crf_stats['variation_summary']['her_cv_percent_crf_0_23_40']:.1f}%)",
    )
    axes[1, 3].fill_between(CRFS, np.array(real_m_crf) - np.array(real_s_crf), np.array(real_m_crf) + np.array(real_s_crf), color="#1b9e77", alpha=0.15)
    axes[1, 3].plot(
        CRFS,
        fake_m_crf,
        "s--",
        color="#d95f02",
        linewidth=2.2,
        markersize=6,
        label=f"Celeb-synth Ort. (CV={fake_crf_stats['variation_summary']['her_cv_percent_crf_0_23_40']:.1f}%)",
    )
    axes[1, 3].fill_between(CRFS, np.array(fake_m_crf) - np.array(fake_s_crf), np.array(fake_m_crf) + np.array(fake_s_crf), color="#d95f02", alpha=0.15)
    axes[1, 3].set_title("Video Geneli (8 Kare Ort. ± Std) — CRF'ye Göre HER", fontsize=10, fontweight="bold")
    axes[1, 3].set_xlabel("libx264 CRF Değeri")
    axes[1, 3].set_ylabel("HER = Σ|Mid| / (Σ|Low| + ε)")
    axes[1, 3].set_xticks(CRFS)
    axes[1, 3].grid(True, linestyle=":", alpha=0.6)
    axes[1, 3].legend(fontsize=8, loc="best")

    fig.suptitle(
        "D3 Görsel Doğrulama Paneli — Deterministik Seçim: Alfabetik 1. Video (id0_0000 vs id0_id16_0000), Kare #0 (Elle Seçilmemiştir)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    D3_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(D3_FIG_PATH, bbox_inches="tight")
    plt.close(fig)

    d3_meta = {
        "experiment": "D3 — QPC ve HER Görsel Doğrulama Paneli",
        "selection_method": (
            "Deterministik / Algoritmik Seçim (Elle 'en belirgin' kare seçilmemiştir): "
            "Celeb-real klasörünün alfabetik ilk videosu (`id0_0000.mp4`) ve aynı kaynak aktör/sahnenin "
            "Celeb-synthesis klasöründeki alfabetik ilk karşılığı (`id0_id16_0000.mp4`) alınmış, "
            "her iki videonun `0.` karesi (`frame_idx=0`) doğrudan kullanılmıştır."
        ),
        "cherry_picked": False,
        "real_sample": {
            "video": str(real_vid),
            "frame_index": selected_frame_idx,
            "qpc_summary_mean": round(float(qpc_vis[0].mean()), 6),
            "monochrome_pc_mean": round(float(mono_vis[0].mean()), 6),
            "chromatic_phase_dispersion_mean": round(float(chrom_disp[0].mean()), 6),
            "her_frame0_source": round(float(her_vals_f0[0]), 6),
            "low_energy_frame0": round(float(low_e_f0[0]), 4),
            "mid_energy_frame0": round(float(mid_e_f0[0]), 4),
            "high_energy_frame0": round(float(high_e_f0[0]), 4),
            "crf_curve_stats": real_crf_stats["variation_summary"],
        },
        "synthesis_sample": {
            "video": str(fake_vid),
            "frame_index": selected_frame_idx,
            "qpc_summary_mean": round(float(qpc_vis[1].mean()), 6),
            "monochrome_pc_mean": round(float(mono_vis[1].mean()), 6),
            "chromatic_phase_dispersion_mean": round(float(chrom_disp[1].mean()), 6),
            "her_frame0_source": round(float(her_vals_f0[1]), 6),
            "low_energy_frame0": round(float(low_e_f0[1]), 4),
            "mid_energy_frame0": round(float(mid_e_f0[1]), 4),
            "high_energy_frame0": round(float(high_e_f0[1]), 4),
            "crf_curve_stats": fake_crf_stats["variation_summary"],
        },
        "figure_path": str(D3_FIG_PATH),
    }
    D3_LOG_PATH.write_text(json.dumps(d3_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return d3_meta


if __name__ == "__main__":
    print("=== D2 (HER Sıkıştırma Testi) ve D3 (Görsel Panel) Çalıştırılıyor ===")
    if not (TMP_CRF_DIR / "ffpp_original").is_dir():
        raise FileNotFoundError(
            f"Geçici CRF klipleri bulunamadı ({TMP_CRF_DIR}). Önce `./reencode_crf_clips.sh` çalıştırın."
        )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = YuNetFaceExtractor(model_path=DEFAULT_MODEL_PATH)
    her_calc = HarmonicEnergyRatio(low_max=3, mid_max=8, include_dc=False).to(dev)

    d2_res = run_d2_experiment(extractor, her_calc, dev)
    print(f"[OK] D2 tamamlandı -> {D2_LOG_PATH}")
    for gname, gs in d2_res["group_summaries"].items():
        print(
            f"  - {gname:10s}: HER CV(0/23/40)={gs['mean_her_cv_percent_crf_0_23_40']:.2f}% | "
            f"ΔHER(0->40)={gs['mean_her_rel_change_crf0_to_crf40_percent']:+.2f}% | "
            f"Ham High CV={gs['mean_raw_high_energy_cv_percent']:.2f}%"
        )

    d3_res = run_d3_visual_panel(extractor, her_calc, dev)
    print(f"[OK] D3 tamamlandı -> {D3_FIG_PATH}")
    print(
        f"  - Celeb-real  (id0_0000 #0)     : HER={d3_res['real_sample']['her_frame0_source']:.4f}, "
        f"QPC={d3_res['real_sample']['qpc_summary_mean']:.4f}, "
        f"ΔΦ_chrom={d3_res['real_sample']['chromatic_phase_dispersion_mean']:.5f}"
    )
    print(
        f"  - Celeb-synth (id0_id16_0000 #0): HER={d3_res['synthesis_sample']['her_frame0_source']:.4f}, "
        f"QPC={d3_res['synthesis_sample']['qpc_summary_mean']:.4f}, "
        f"ΔΦ_chrom={d3_res['synthesis_sample']['chromatic_phase_dispersion_mean']:.5f}"
    )
