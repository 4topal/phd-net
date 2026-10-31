"""Videolardan YuNet ONNX ile yüz tespiti ve %20 marjlı 224x224 kırpma modülü.

Özellikler:
- OpenCV `FaceDetectorYN` (YuNet 2023mar ONNX) ile hızlı yüz tespiti
- Model dosyası yüklenmeden önce SHA256 bütünlük doğrulaması
- N karede bir örnekleme (`--frame_interval`), en yüksek güven skorlu yüzü seçme
- Yüz kutusu çevresine %20 marj ekleme ve 224x224 boyutuna yeniden örnekleme
- Yüz bulunamayan kareleri video bazında ve toplamda sayıp JSON + konsol olarak loglama
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

YUNET_EXPECTED_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

KOD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = KOD_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_LOG_PATH = KOD_ROOT / "outputs" / "logs" / "face_extractor_log.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("face_extractor")


@dataclass
class VideoExtractionStats:
    video_path: str
    total_video_frames: int
    sampled_frames: int
    saved_faces: int
    missed_frames: int
    missed_frame_indices: List[int] = field(default_factory=list)


@dataclass
class ExtractionSummary:
    data_dir: str
    output_dir: str
    detector: str
    model_sha256: str
    frame_interval: int
    margin_ratio: float
    target_size: int
    videos_processed: int
    total_sampled_frames: int
    total_saved_faces: int
    total_missed_frames: int
    miss_rate: float
    elapsed_sec: float
    per_video: List[VideoExtractionStats] = field(default_factory=list)


def verify_sha256(file_path: Path, expected_sha256: str = YUNET_EXPECTED_SHA256) -> str:
    """Model dosyasının SHA256 özetini hesaplar ve beklenen değerle doğrular."""
    if not file_path.is_file():
        raise FileNotFoundError(f"YuNet model dosyası bulunamadı: {file_path}")
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA256 uyuşmazlığı ({file_path}): beklenen={expected_sha256}, ölçülen={digest}"
        )
    return digest


def crop_face_with_margin(
    frame_bgr: np.ndarray,
    bbox_xywh: Tuple[float, float, float, float],
    margin_ratio: float = 0.20,
    target_size: int = 224,
    square_box: bool = True,
) -> np.ndarray:
    """Tespit edilen yüz kutusuna %20 marj ekleyip `target_size x target_size` kırpar.

    Args:
        frame_bgr: `(H, W, 3)` BGR kare.
        bbox_xywh: YuNet çıktısı `(x, y, w, h)`.
        margin_ratio: Her kenara eklenecek oransal marj (varsayılan 0.20 = %20).
        target_size: Çıktı çözünürlüğü (varsayılan 224).
        square_box: True ise en-boy oranını bozmamak için merkez etrafında kare kutu alır.
    """
    img_h, img_w = frame_bgr.shape[:2]
    x, y, w, h = bbox_xywh
    cx = x + 0.5 * w
    cy = y + 0.5 * h

    if square_box:
        side = max(w, h) * (1.0 + 2.0 * margin_ratio)
        half_w = 0.5 * side
        half_h = 0.5 * side
    else:
        half_w = 0.5 * w * (1.0 + 2.0 * margin_ratio)
        half_h = 0.5 * h * (1.0 + 2.0 * margin_ratio)

    x1 = max(0, int(round(cx - half_w)))
    y1 = max(0, int(round(cy - half_h)))
    x2 = min(img_w, int(round(cx + half_w)))
    y2 = min(img_h, int(round(cy + half_h)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Geçersiz kırpma koordinatları: {(x1, y1, x2, y2)}")

    crop = frame_bgr[y1:y2, x1:x2]
    interp = cv2.INTER_AREA if (crop.shape[0] > target_size or crop.shape[1] > target_size) else cv2.INTER_LINEAR
    return cv2.resize(crop, (target_size, target_size), interpolation=interp)


class YuNetFaceExtractor:
    """OpenCV YuNet tabanlı yüz tespit ve ön işleme sınıfı."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        score_threshold: float = 0.60,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
        margin_ratio: float = 0.20,
        target_size: int = 224,
        verify_hash: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_sha256 = verify_sha256(
            self.model_path,
            YUNET_EXPECTED_SHA256 if verify_hash else "",
        )
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.margin_ratio = margin_ratio
        self.target_size = target_size

        self._detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        )
        self._current_input_size: Tuple[int, int] = (320, 320)

    def detect_primary_face(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[Tuple[float, float, float, float], float]]:
        """Karedeki en yüksek güven skorlu yüzü `(bbox_xywh, score)` olarak döndürür."""
        h, w = frame_bgr.shape[:2]
        if (w, h) != self._current_input_size:
            self._detector.setInputSize((w, h))
            self._current_input_size = (w, h)

        _, faces = self._detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None

        # YuNet satır formatı: [x, y, w, h, x_re, y_re, x_le, y_le, ..., score]
        best_idx = int(np.argmax(faces[:, -1]))
        best = faces[best_idx]
        bbox = (float(best[0]), float(best[1]), float(best[2]), float(best[3]))
        score = float(best[-1])
        if bbox[2] <= 2.0 or bbox[3] <= 2.0:
            return None
        return bbox, score

    def extract_face_crop(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Tek bir BGR kareden 224x224 BGR yüz kırpması döndürür (yüz yoksa None)."""
        det = self.detect_primary_face(frame_bgr)
        if det is None:
            return None
        bbox, _ = det
        return crop_face_with_margin(
            frame_bgr,
            bbox,
            margin_ratio=self.margin_ratio,
            target_size=self.target_size,
            square_box=True,
        )

    def process_video(
        self,
        video_path: Path,
        video_output_dir: Path,
        frame_interval: int = 30,
        max_frames_per_video: Optional[int] = None,
    ) -> VideoExtractionStats:
        """Tek bir videoyu işler, N karede bir yüz kırpıp PNG olarak kaydeder."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Video açılamadı: %s", video_path)
            return VideoExtractionStats(
                video_path=str(video_path),
                total_video_frames=0,
                sampled_frames=0,
                saved_faces=0,
                missed_frames=0,
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_output_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        sampled = 0
        saved = 0
        missed = 0
        missed_indices: List[int] = []

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                sampled += 1
                crop = self.extract_face_crop(frame_bgr)
                if crop is not None:
                    out_file = video_output_dir / f"frame_{frame_idx:05d}.png"
                    cv2.imwrite(str(out_file), crop)
                    saved += 1
                else:
                    missed += 1
                    missed_indices.append(frame_idx)

                if max_frames_per_video is not None and sampled >= max_frames_per_video:
                    break

            frame_idx += 1

        cap.release()
        return VideoExtractionStats(
            video_path=str(video_path),
            total_video_frames=max(total_frames, frame_idx),
            sampled_frames=sampled,
            saved_faces=saved,
            missed_frames=missed,
            missed_frame_indices=missed_indices,
        )


def collect_videos(data_dir: Path, max_videos: Optional[int] = None) -> List[Path]:
    """Dizindeki video dosyalarını deterministik sırada listeler."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Veri dizini bulunamadı: {data_dir}")
    if data_dir.is_file() and data_dir.suffix.lower() in VIDEO_EXTENSIONS:
        return [data_dir]

    videos = sorted(
        p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if max_videos is not None and max_videos > 0:
        videos = videos[:max_videos]
    return videos


def run_extraction(
    data_dir: Path,
    output_dir: Path,
    frame_interval: int = 30,
    max_videos: Optional[int] = None,
    margin_ratio: float = 0.20,
    target_size: int = 224,
    score_threshold: float = 0.60,
    model_path: Path = DEFAULT_MODEL_PATH,
    log_path: Optional[Path] = DEFAULT_LOG_PATH,
) -> ExtractionSummary:
    """Verilen dizindeki videoları işler ve özet raporu döndürür / kaydeder."""
    if frame_interval < 1:
        raise ValueError("frame_interval >= 1 olmalıdır.")

    t0 = time.perf_counter()
    extractor = YuNetFaceExtractor(
        model_path=model_path,
        score_threshold=score_threshold,
        margin_ratio=margin_ratio,
        target_size=target_size,
        verify_hash=True,
    )

    videos = collect_videos(data_dir, max_videos=max_videos)
    logger.info(
        "Yüz çıkarımı başlıyor: %d video | aralık=%d kare | marj=%%%.0f | boyut=%dx%d",
        len(videos),
        frame_interval,
        margin_ratio * 100.0,
        target_size,
        target_size,
    )

    per_video_stats: List[VideoExtractionStats] = []
    for vid in videos:
        rel_parent = vid.parent.relative_to(data_dir) if data_dir.is_dir() else Path(".")
        vid_out_dir = output_dir / rel_parent / vid.stem
        stats = extractor.process_video(
            video_path=vid,
            video_output_dir=vid_out_dir,
            frame_interval=frame_interval,
        )
        per_video_stats.append(stats)
        if stats.missed_frames > 0:
            logger.info(
                "  [%s] örneklenen=%d | kaydedilen=%d | yüz bulunamayan=%d (kareler: %s)",
                vid.name,
                stats.sampled_frames,
                stats.saved_faces,
                stats.missed_frames,
                stats.missed_frame_indices[:10],
            )
        else:
            logger.info(
                "  [%s] örneklenen=%d | kaydedilen=%d | yüz bulunamayan=0",
                vid.name,
                stats.sampled_frames,
                stats.saved_faces,
            )

    elapsed = time.perf_counter() - t0
    total_sampled = sum(s.sampled_frames for s in per_video_stats)
    total_saved = sum(s.saved_faces for s in per_video_stats)
    total_missed = sum(s.missed_frames for s in per_video_stats)
    miss_rate = (total_missed / total_sampled) if total_sampled > 0 else 0.0

    summary = ExtractionSummary(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        detector=f"OpenCV FaceDetectorYN ({model_path.name})",
        model_sha256=extractor.model_sha256,
        frame_interval=frame_interval,
        margin_ratio=margin_ratio,
        target_size=target_size,
        videos_processed=len(per_video_stats),
        total_sampled_frames=total_sampled,
        total_saved_faces=total_saved,
        total_missed_frames=total_missed,
        miss_rate=round(miss_rate, 6),
        elapsed_sec=round(elapsed, 3),
        per_video=per_video_stats,
    )

    logger.info(
        "Tamamlandı (%.2f sn): video=%d | örneklenen=%d | kaydedilen=%d | yüz bulunamayan=%d (%%%.2f)",
        elapsed,
        summary.videos_processed,
        summary.total_sampled_frames,
        summary.total_saved_faces,
        summary.total_missed_frames,
        summary.miss_rate * 100.0,
    )

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Log kaydedildi: %s", log_path)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Videolardan N karede bir YuNet yüz tespiti ve %20 marjlı 224x224 kırpma aracı."
    )
    parser.add_argument("--data_dir", type=Path, default=None, help="Girdi video klasörü veya video dosyası.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=KOD_ROOT / "data" / "processed_faces",
        help="Kırpılan yüz karelerinin kaydedileceği dizin.",
    )
    parser.add_argument("--frame_interval", type=int, default=30, help="Kaç karede bir örnekleneceği (N).")
    parser.add_argument("--max_videos", type=int, default=None, help="İşlenecek azami video sayısı.")
    parser.add_argument("--margin_ratio", type=float, default=0.20, help="Yüz kutusu marj oranı (0.20 = %%20).")
    parser.add_argument("--target_size", type=int, default=224, help="Çıktı yüz boyutu (224 -> 224x224).")
    parser.add_argument("--score_threshold", type=float, default=0.60, help="YuNet güven eşiği.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH, help="YuNet ONNX dosya yolu.")
    parser.add_argument("--log_path", type=Path, default=DEFAULT_LOG_PATH, help="JSON log dosya yolu.")
    return parser


if __name__ == "__main__":
    import sys

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.data_dir is not None:
        run_extraction(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            frame_interval=args.frame_interval,
            max_videos=args.max_videos,
            margin_ratio=args.margin_ratio,
            target_size=args.target_size,
            score_threshold=args.score_threshold,
            model_path=args.model_path,
            log_path=args.log_path,
        )
    else:
        print("=== face_extractor.py Kendi Kendini Test Modu ===")
        sha = verify_sha256(DEFAULT_MODEL_PATH)
        print(f"[OK] YuNet ONNX SHA256 doğrulandı: {sha}")

        default_celeb_real = KOD_ROOT.parent / "datasets" / "celebdf" / "Celeb-real"
        default_celeb_synth = KOD_ROOT.parent / "datasets" / "celebdf" / "Celeb-synthesis"

        if default_celeb_real.is_dir() and default_celeb_synth.is_dir():
            s_real = run_extraction(
                data_dir=default_celeb_real,
                output_dir=KOD_ROOT / "data" / "processed_faces" / "celeb_real",
                frame_interval=30,
                max_videos=2,
                log_path=KOD_ROOT / "outputs" / "logs" / "face_extractor_selftest_real.json",
            )
            s_fake = run_extraction(
                data_dir=default_celeb_synth,
                output_dir=KOD_ROOT / "data" / "processed_faces" / "celeb_synthesis",
                frame_interval=30,
                max_videos=2,
                log_path=KOD_ROOT / "outputs" / "logs" / "face_extractor_selftest_fake.json",
            )
            print(
                f"[ÖZET] Celeb-real: {s_real.total_saved_faces}/{s_real.total_sampled_frames} yüz "
                f"(bulunamayan={s_real.total_missed_frames}) | "
                f"Celeb-synthesis: {s_fake.total_saved_faces}/{s_fake.total_sampled_frames} yüz "
                f"(bulunamayan={s_fake.total_missed_frames})"
            )
        else:
            print("[UYARI] Veri seti dizini bulunamadı; boş kare üzerinde negatif yüz testi çalıştırılıyor.")
            extractor = YuNetFaceExtractor(model_path=DEFAULT_MODEL_PATH)
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            det = extractor.detect_primary_face(blank)
            assert det is None, "Boş karede yüz tespit edilmemeliydi!"
            print("[OK] Boş kare negatif yüz testi başarılı (missed=1).")
            sys.exit(0)
