"""Videolardan SCRFD-10G / YuNet ile yuz tespiti, %20 marjli 224x224 kirpma ve maske cikarim modulu.

Ozellikler:
- SCRFD-10G (5-nokta landmarkli, multi-stride {8, 16, 32} capraz olcekli) ONNXRuntime dedektoru
- OpenCV FaceDetectorYN (YuNet 2023mar ONNX) geriye donuk uyumlu dedektor ve otomatik fallback
- Model dosyasi yuklenmeden once SHA256 butunluk dogrulamasi ve ensure_scrfd_model yardimcisi
- 5-nokta landmark agirlik merkezi ile bbox merkezini harmanlayan eksenel (axis-aligned) %20 marjli kare kirpma
- Hibrit Birincil Yuz Secimi (Hybrid Primary Face Selection):
    * Mode A (Mask-Guided): Resmi maske mevcutsa maske ile kesisen yuzu secme ve eszamanli 224x224 maske kirpimi
    * Mode B (Continuity + Area * Score): Maske yoksa alan x (guven^2) x zamansal sureklilik faktorune gore secim
- FF++ ve Celeb-DF v2 yapilandirilmis veri cikarimi (1:4 gercek/sahte dengeleme, c23/c40, manifest.json)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from project_paths import get_celebdf_root, get_datasets_root, get_ffpp_root

KOD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YUNET_PATH = KOD_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_MODEL_PATH = DEFAULT_YUNET_PATH  # Geriye donuk uyumluluk
DEFAULT_SCRFD_MODEL_PATH = KOD_ROOT / "models" / "scrfd_10g_bnkps.onnx"
DEFAULT_LOG_PATH = KOD_ROOT / "outputs" / "logs" / "face_extractor_log.json"

YUNET_EXPECTED_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SCRFD_DOWNLOAD_URL = "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

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
    """Model dosyasinin SHA256 ozetini hesaplar ve beklenen degerle dogrular."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Model dosyasi bulunamadi: {file_path}")
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA256 uyusmazligi ({file_path}): beklenen={expected_sha256}, olculen={digest}"
        )
    return digest


def ensure_scrfd_model(
    model_path: Optional[Path] = None,
    download_if_missing: bool = True,
) -> Path:
    """SCRFD-10G ONNX model dosyasini kontrol eder, yoksa indirmeyi dener."""
    target = Path(model_path) if model_path is not None else DEFAULT_SCRFD_MODEL_PATH
    if target.is_file() and target.stat().st_size > 1024 * 1024:
        return target

    if download_if_missing:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("SCRFD-10G modeli indiriliyor: %s -> %s", SCRFD_DOWNLOAD_URL, target)
            import urllib.request
            urllib.request.urlretrieve(SCRFD_DOWNLOAD_URL, str(target))
            if target.is_file() and target.stat().st_size > 1024 * 1024:
                logger.info("SCRFD-10G modeli basariyla indirildi.")
                return target
        except Exception as exc:
            logger.warning(
                "SCRFD-10G modeli otomatik indirilemedi (%s). YuNet fallback kullanilacaktir.",
                exc,
            )
    return target


def distance2bbox(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """A merkezli mesafe tahminlerini [x1, y1, x2, y2] kutularina donusturur."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """A merkezli mesafe tahminlerini 5-nokta yuz landmarklarina donusturur."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def compute_crop_box(
    frame_shape: Tuple[int, int],
    bbox_xywh: Tuple[float, float, float, float],
    margin_ratio: float = 0.20,
    square_box: bool = True,
    landmarks_5pt: Optional[np.ndarray] = None,
) -> Tuple[int, int, int, int]:
    """Yuz kutusu ve opsiyonel 5 landmark merkezine gore (x1, y1, x2, y2) piksel kutusunu hesaplar.

    Landmarklar verildiginde 5-nokta agirlik merkezi ile bbox merkezi harmanlanir:
      cx = 0.5 * bbox_cx + 0.5 * kps_cx
      cy = 0.5 * bbox_cy + 0.5 * kps_cy
    Dondurme yapilmaz (strictly axis-aligned).
    """
    img_h, img_w = frame_shape[:2]
    x, y, w, h = bbox_xywh
    bbox_cx = x + 0.5 * w
    bbox_cy = y + 0.5 * h

    if landmarks_5pt is not None:
        kps_arr = np.asarray(landmarks_5pt, dtype=np.float32)
        if kps_arr.shape == (5, 2):
            kps_cx = float(np.mean(kps_arr[:, 0]))
            kps_cy = float(np.mean(kps_arr[:, 1]))
            cx = 0.5 * bbox_cx + 0.5 * kps_cx
            cy = 0.5 * bbox_cy + 0.5 * kps_cy
        else:
            cx, cy = bbox_cx, bbox_cy
    else:
        cx, cy = bbox_cx, bbox_cy

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
        x1 = max(0, min(int(round(x)), img_w - 1))
        y1 = max(0, min(int(round(y)), img_h - 1))
        x2 = max(x1 + 1, min(int(round(x + w)), img_w))
        y2 = max(y1 + 1, min(int(round(y + h)), img_h))

    return x1, y1, x2, y2


def crop_face_with_margin(
    frame_bgr: np.ndarray,
    bbox_xywh: Tuple[float, float, float, float],
    margin_ratio: float = 0.20,
    target_size: int = 224,
    square_box: bool = True,
    landmarks_5pt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Tespit edilen yuz kutusuna %20 marj ekleyip `target_size x target_size` kirpar."""
    x1, y1, x2, y2 = compute_crop_box(
        frame_bgr.shape[:2],
        bbox_xywh,
        margin_ratio=margin_ratio,
        square_box=square_box,
        landmarks_5pt=landmarks_5pt,
    )
    crop = frame_bgr[y1:y2, x1:x2]
    interp = (
        cv2.INTER_AREA
        if (crop.shape[0] > target_size or crop.shape[1] > target_size)
        else cv2.INTER_LINEAR
    )
    return cv2.resize(crop, (target_size, target_size), interpolation=interp)


def crop_mask_with_box(
    mask_frame_gray: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    target_size: int = 224,
    threshold: int = 128,
) -> np.ndarray:
    """Ayni koordinat kutusuyla maskeyi keser, 224x224 boyutuna INTER_NEAREST ile ornekler ve esikler."""
    x1, y1, x2, y2 = crop_box
    crop = mask_frame_gray[y1:y2, x1:x2]
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return np.where(resized >= threshold, 255, 0).astype(np.uint8)


def select_primary_face(
    detected_faces: List[Tuple[Tuple[float, float, float, float], float, Optional[np.ndarray]]],
    mask_frame_gray: Optional[np.ndarray] = None,
    prev_face_center: Optional[Tuple[float, float]] = None,
    frame_shape: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[Tuple[float, float, float, float], float, Optional[np.ndarray]]]:
    """Hibrit Birincil Yuz Secimi:
    - Mode A (Mask-Guided): Eger mask_frame_gray varsa ve non-zero (>30) pikseli varsa,
      maske bolgesiyle en cok kesisen / en yakin yuzu secer.
    - Mode B (Continuity + Area * Score): Maske yoksa veya bossa,
      area * (score ** 2) * exp(-dist / diag) kriterine gore siralama yapar.
    """
    if not detected_faces:
        return None
    if len(detected_faces) == 1:
        return detected_faces[0]

    img_h, img_w = frame_shape if frame_shape is not None else (1080, 1920)
    frame_diag = math.hypot(img_w, img_h)

    # Mode A: Mask-Guided Selection
    if mask_frame_gray is not None:
        nonzero_count = int(np.count_nonzero(mask_frame_gray > 30))
        if nonzero_count > 30:
            ys, xs = np.where(mask_frame_gray > 30)
            mx1, my1, mx2, my2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            mcx, mcy = 0.5 * (mx1 + mx2), 0.5 * (my1 + my2)
            mask_area = max(1.0, (mx2 - mx1) * (my2 - my1))

            best_overlap = -1.0
            best_face = None

            for face in detected_faces:
                (fx, fy, fw, fh), score, _ = face
                fx2, fy2 = fx + fw, fy + fh

                ix1 = max(fx, mx1)
                iy1 = max(fy, my1)
                ix2 = min(fx2, mx2)
                iy2 = min(fy2, my2)
                inter_w = max(0.0, ix2 - ix1)
                inter_h = max(0.0, iy2 - iy1)
                inter_area = inter_w * inter_h
                face_area = max(1.0, fw * fh)
                iou = inter_area / (face_area + mask_area - inter_area + 1e-6)

                px1 = max(0, min(int(round(fx)), img_w - 1))
                py1 = max(0, min(int(round(fy)), img_h - 1))
                px2 = max(px1 + 1, min(int(round(fx2)), img_w))
                py2 = max(py1 + 1, min(int(round(fy2)), img_h))
                mask_pixels_in_face = int(np.count_nonzero(mask_frame_gray[py1:py2, px1:px2] > 30))

                overlap_metric = mask_pixels_in_face + iou * 1000.0
                if overlap_metric > best_overlap:
                    best_overlap = overlap_metric
                    best_face = face

            if best_face is not None and best_overlap > 10.0:
                return best_face

            # Kesisim azsa maske merkezine en yakin yuz
            min_dist = float("inf")
            best_face = None
            for face in detected_faces:
                (fx, fy, fw, fh), _, _ = face
                fcx, fcy = fx + 0.5 * fw, fy + 0.5 * fh
                d = math.hypot(fcx - mcx, fcy - mcy)
                if d < min_dist:
                    min_dist = d
                    best_face = face
            if best_face is not None:
                return best_face

    # Mode B: Continuity + Area * Score
    best_score = -1.0
    best_face = None
    for face in detected_faces:
        (fx, fy, fw, fh), score, _ = face
        area = fw * fh
        fcx, fcy = fx + 0.5 * fw, fy + 0.5 * fh
        if prev_face_center is not None:
            dist = math.hypot(fcx - prev_face_center[0], fcy - prev_face_center[1])
            continuity_factor = math.exp(-dist / max(frame_diag, 1.0))
        else:
            continuity_factor = 1.0

        rank_score = area * (score ** 2) * continuity_factor
        if rank_score > best_score:
            best_score = rank_score
            best_face = face

    return best_face if best_face is not None else detected_faces[0]


class YuNetFaceExtractor:
    """OpenCV YuNet tabanli yuz tespit ve on isleme sinifi."""

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

    def detect_all_faces(
        self, frame_bgr: np.ndarray
    ) -> List[Tuple[Tuple[float, float, float, float], float, Optional[np.ndarray]]]:
        """Karedeki tum yuzleri [(bbox_xywh, score, kps_5pt), ...] olarak dondurur."""
        h, w = frame_bgr.shape[:2]
        if (w, h) != self._current_input_size:
            self._detector.setInputSize((w, h))
            self._current_input_size = (w, h)

        _, faces = self._detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return []

        results: List[Tuple[Tuple[float, float, float, float], float, Optional[np.ndarray]]] = []
        for face in faces:
            x, y, bw, bh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
            score = float(face[-1])
            if bw <= 2.0 or bh <= 2.0:
                continue

            # YuNet 5-nokta landmark: re, le, nt, rc, lc
            if len(face) >= 15:
                kps = np.array(
                    [
                        [face[4], face[5]],
                        [face[6], face[7]],
                        [face[8], face[9]],
                        [face[10], face[11]],
                        [face[12], face[13]],
                    ],
                    dtype=np.float32,
                )
            else:
                kps = None
            results.append(((x, y, bw, bh), score, kps))
        return results

    def detect_primary_face(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[Tuple[float, float, float, float], float]]:
        """Karedeki en yuksek guven skorlu yuzu `(bbox_xywh, score)` olarak dondurur (geriye donuk uyumlu)."""
        faces = self.detect_all_faces(frame_bgr)
        if not faces:
            return None
        # En yuksek skorlu yuzu sec
        best = max(faces, key=lambda f: f[1])
        return best[0], best[1]

    def extract_face_crop(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Tek bir BGR kareden 224x224 BGR yuz kirpmasi dondurur (yuz yoksa None)."""
        faces = self.detect_all_faces(frame_bgr)
        if not faces:
            return None
        selected = select_primary_face(faces, frame_shape=frame_bgr.shape[:2])
        if selected is None:
            return None
        bbox, _, kps = selected
        return crop_face_with_margin(
            frame_bgr,
            bbox,
            margin_ratio=self.margin_ratio,
            target_size=self.target_size,
            square_box=True,
            landmarks_5pt=kps,
        )

    def process_video(
        self,
        video_path: Path,
        video_output_dir: Path,
        frame_interval: int = 30,
        max_frames_per_video: Optional[int] = None,
    ) -> VideoExtractionStats:
        """Tek bir videoyu isler, N karede bir yuz kirpip PNG olarak kaydeder."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Video acilamadi: %s", video_path)
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
        prev_center: Optional[Tuple[float, float]] = None

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                sampled += 1
                faces = self.detect_all_faces(frame_bgr)
                selected = select_primary_face(
                    faces,
                    prev_face_center=prev_center,
                    frame_shape=frame_bgr.shape[:2],
                )
                if selected is not None:
                    bbox, score, kps = selected
                    crop = crop_face_with_margin(
                        frame_bgr,
                        bbox,
                        margin_ratio=self.margin_ratio,
                        target_size=self.target_size,
                        square_box=True,
                        landmarks_5pt=kps,
                    )
                    out_file = video_output_dir / f"frame_{frame_idx:05d}.png"
                    cv2.imwrite(str(out_file), crop)
                    saved += 1
                    prev_center = (bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3])
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


class SCRFDFaceExtractor:
    """SCRFD-10G (10GF, 5-point landmark) ONNX tabanli yuksek dogruluklu yuz tespit sinifi."""

    def __init__(
        self,
        model_path: Path = DEFAULT_SCRFD_MODEL_PATH,
        score_threshold: float = 0.50,
        nms_threshold: float = 0.40,
        input_size: Tuple[int, int] = (640, 640),
        margin_ratio: float = 0.20,
        target_size: int = 224,
    ) -> None:
        self.model_path = Path(model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.margin_ratio = margin_ratio
        self.target_size = target_size

        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        self._center_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

        self._session = None
        self._cv_net = None
        self._fallback_yunet: Optional[YuNetFaceExtractor] = None

        if not self.model_path.is_file():
            self.model_path = ensure_scrfd_model(self.model_path, download_if_missing=True)

        if self.model_path.is_file():
            try:
                import onnxruntime as ort
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 1
                sess_opts.inter_op_num_threads = 1
                sess_opts.log_severity_level = 3
                available_providers = ort.get_available_providers()
                providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available_providers]
                self._session = ort.InferenceSession(str(self.model_path), sess_options=sess_opts, providers=providers)
                logger.info("SCRFD ONNXRuntime baslatildi. Saglayicilar: %s", self._session.get_providers())
            except Exception as exc:
                logger.warning("SCRFD ONNXRuntime ile acilamadi (%s), cv2.dnn deneniyor...", exc)
                try:
                    self._cv_net = cv2.dnn.readNetFromONNX(str(self.model_path))
                    logger.info("SCRFD cv2.dnn ile basariyla acildi.")
                except Exception as exc2:
                    logger.warning("cv2.dnn ile de acilamadi (%s). YuNet fallback devrede.", exc2)
                    self._fallback_yunet = YuNetFaceExtractor(
                        target_size=self.target_size,
                        margin_ratio=self.margin_ratio,
                        score_threshold=self.score_threshold,
                    )
        else:
            logger.info("SCRFD modeli bulunamadi (%s). YuNet fallback aktif.", self.model_path)
            self._fallback_yunet = YuNetFaceExtractor(
                target_size=self.target_size,
                margin_ratio=self.margin_ratio,
                score_threshold=self.score_threshold,
            )

    def _get_anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        if key in self._center_cache:
            return self._center_cache[key]

        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if self._num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

        if len(self._center_cache) < 64:
            self._center_cache[key] = anchor_centers
        return anchor_centers

    def detect_all_faces(
        self, frame_bgr: np.ndarray
    ) -> List[Tuple[Tuple[float, float, float, float], float, Optional[np.ndarray]]]:
        """Karedeki tum yuzleri [(bbox_xywh, score, kps_5pt), ...] listesi olarak tespit eder."""
        if self._session is None and self._cv_net is None:
            if self._fallback_yunet is not None:
                return self._fallback_yunet.detect_all_faces(frame_bgr)
            return []

        orig_h, orig_w = frame_bgr.shape[:2]
        input_w, input_h = self.input_size
        scale_x = orig_w / float(input_w)
        scale_y = orig_h / float(input_h)

        img_resized = cv2.resize(frame_bgr, (input_w, input_h))
        blob = cv2.dnn.blobFromImage(
            img_resized,
            1.0 / 128.0,
            (input_w, input_h),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )

        if self._session is not None:
            input_name = self._session.get_inputs()[0].name
            net_outs = self._session.run(None, {input_name: blob})
        else:
            self._cv_net.setInput(blob)
            out_names = self._cv_net.getUnconnectedOutLayersNames()
            net_outs = self._cv_net.forward(out_names)

        all_boxes: List[Tuple[float, float, float, float]] = []
        all_scores: List[float] = []
        all_kpss: List[Optional[np.ndarray]] = []

        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            if scores.ndim > 1:
                scores = scores.flatten()

            bbox_preds = net_outs[idx + fmc]
            if bbox_preds.ndim == 3 and bbox_preds.shape[0] == 1:
                bbox_preds = bbox_preds[0]
            bbox_preds = bbox_preds * stride

            if self.use_kps and len(net_outs) > idx + fmc * 2:
                kps_preds = net_outs[idx + fmc * 2]
                if kps_preds.ndim == 3 and kps_preds.shape[0] == 1:
                    kps_preds = kps_preds[0]
                kps_preds = kps_preds * stride
            else:
                kps_preds = None

            feat_h = input_h // stride
            feat_w = input_w // stride
            anchor_centers = self._get_anchor_centers(feat_h, feat_w, stride)

            pos_inds = np.where(scores >= self.score_threshold)[0]
            if len(pos_inds) == 0:
                continue

            pos_scores = scores[pos_inds]
            pos_bboxes = bbox_preds[pos_inds]
            pos_anchors = anchor_centers[pos_inds]

            decoded_bboxes = distance2bbox(pos_anchors, pos_bboxes, max_shape=(input_h, input_w))

            if kps_preds is not None:
                pos_kpss = kps_preds[pos_inds]
                decoded_kps = distance2kps(pos_anchors, pos_kpss, max_shape=(input_h, input_w))
            else:
                decoded_kps = None

            for i in range(len(pos_inds)):
                x1 = float(decoded_bboxes[i, 0] * scale_x)
                y1 = float(decoded_bboxes[i, 1] * scale_y)
                x2 = float(decoded_bboxes[i, 2] * scale_x)
                y2 = float(decoded_bboxes[i, 3] * scale_y)

                x1 = max(0.0, min(x1, float(orig_w - 1)))
                y1 = max(0.0, min(y1, float(orig_h - 1)))
                x2 = max(x1 + 1.0, min(x2, float(orig_w)))
                y2 = max(y1 + 1.0, min(y2, float(orig_h)))

                bw = x2 - x1
                bh = y2 - y1
                if bw < 4.0 or bh < 4.0:
                    continue

                all_boxes.append((x1, y1, bw, bh))
                all_scores.append(float(pos_scores[i]))

                if decoded_kps is not None:
                    kps_5 = decoded_kps[i].reshape(5, 2).copy()
                    kps_5[:, 0] *= scale_x
                    kps_5[:, 1] *= scale_y
                    all_kpss.append(kps_5)
                else:
                    all_kpss.append(None)

        if not all_boxes:
            return []

        boxes_for_nms = [[b[0], b[1], b[2], b[3]] for b in all_boxes]
        keep_indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            all_scores,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )

        results = []
        if len(keep_indices) > 0:
            for k in keep_indices:
                idx_val = int(k[0]) if hasattr(k, "__len__") else int(k)
                results.append((all_boxes[idx_val], all_scores[idx_val], all_kpss[idx_val]))

        return results

    def detect_primary_face(
        self,
        frame_bgr: np.ndarray,
        mask_frame_gray: Optional[np.ndarray] = None,
        prev_face_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[Tuple[float, float, float, float], float, Optional[np.ndarray]]]:
        """Karedeki en uygun birincil yuzu Hibrit Secim ile dondurur."""
        faces = self.detect_all_faces(frame_bgr)
        return select_primary_face(
            faces,
            mask_frame_gray=mask_frame_gray,
            prev_face_center=prev_face_center,
            frame_shape=frame_bgr.shape[:2],
        )

    def extract_face_crop(
        self,
        frame_bgr: np.ndarray,
        mask_frame_gray: Optional[np.ndarray] = None,
        prev_face_center: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[float, float]]]:
        """Yuz (224x224), eslesmis maske (224x224 veya None) ve yuz merkezini dondurur."""
        selected = self.detect_primary_face(
            frame_bgr,
            mask_frame_gray=mask_frame_gray,
            prev_face_center=prev_face_center,
        )
        if selected is None:
            return None, None, prev_face_center

        bbox, score, kps = selected
        crop_box = compute_crop_box(
            frame_bgr.shape[:2],
            bbox,
            margin_ratio=self.margin_ratio,
            square_box=True,
            landmarks_5pt=kps,
        )
        x1, y1, x2, y2 = crop_box
        crop = frame_bgr[y1:y2, x1:x2]
        interp = (
            cv2.INTER_AREA
            if (crop.shape[0] > self.target_size or crop.shape[1] > self.target_size)
            else cv2.INTER_LINEAR
        )
        face_img = cv2.resize(crop, (self.target_size, self.target_size), interpolation=interp)

        mask_img = None
        if mask_frame_gray is not None:
            mask_img = crop_mask_with_box(mask_frame_gray, crop_box, target_size=self.target_size)

        new_center = (bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3])
        return face_img, mask_img, new_center


def collect_videos(data_dir: Path, max_videos: Optional[int] = None) -> List[Path]:
    """Dizindeki video dosyalarini deterministik sirada hizli (os.scandir) listeler."""
    import os
    if not data_dir.exists():
        raise FileNotFoundError(f"Veri dizini bulunamadi: {data_dir}")
    if data_dir.is_file() and data_dir.suffix.lower() in VIDEO_EXTENSIONS:
        return [data_dir]

    direct_videos: List[Path] = []
    has_subdirs = False
    with os.scandir(str(data_dir)) as it:
        for entry in it:
            if entry.is_file():
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in VIDEO_EXTENSIONS:
                    direct_videos.append(Path(entry.path))
            elif entry.is_dir():
                has_subdirs = True

    if direct_videos or not has_subdirs:
        videos = sorted(direct_videos, key=lambda p: p.name)
    else:
        videos = sorted(
            p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
    if max_videos is not None and max_videos > 0:
        videos = videos[:max_videos]
    return videos


def get_evenly_spaced_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    """Videodan esit aralikli kare indeksleri uretir."""
    if total_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))
    indices = np.round(np.linspace(0, total_frames - 1, num_frames)).astype(int)
    return sorted(list(set(indices.tolist())))


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
    detector_name: str = "yunet",
) -> ExtractionSummary:
    """Verilen dizindeki videolari isler ve ozet raporu dondurur / kaydeder (geriye donuk uyumlu)."""
    if frame_interval < 1:
        raise ValueError("frame_interval >= 1 olmalidir.")

    t0 = time.perf_counter()
    if detector_name.lower() == "scrfd":
        extractor = SCRFDFaceExtractor(
            model_path=model_path,
            score_threshold=score_threshold,
            margin_ratio=margin_ratio,
            target_size=target_size,
        )
        detector_label = f"SCRFD-10G ({extractor.model_path.name})"
        model_sha = ""
    else:
        extractor = YuNetFaceExtractor(
            model_path=model_path,
            score_threshold=score_threshold,
            margin_ratio=margin_ratio,
            target_size=target_size,
            verify_hash=True,
        )
        detector_label = f"OpenCV FaceDetectorYN ({model_path.name})"
        model_sha = extractor.model_sha256

    videos = collect_videos(data_dir, max_videos=max_videos)
    logger.info(
        "Yuz cikarimi basliyor: %d video | aralik=%d kare | marj=%%%.0f | boyut=%dx%d",
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

    elapsed = time.perf_counter() - t0
    total_sampled = sum(s.sampled_frames for s in per_video_stats)
    total_saved = sum(s.saved_faces for s in per_video_stats)
    total_missed = sum(s.missed_frames for s in per_video_stats)
    miss_rate = (total_missed / total_sampled) if total_sampled > 0 else 0.0

    summary = ExtractionSummary(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        detector=detector_label,
        model_sha256=model_sha,
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

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Log kaydedildi: %s", log_path)

    return summary


def _process_single_video_frames(
    extractor: Any,
    video_path: Path,
    mask_video_path: Optional[Path],
    target_frame_indices: Optional[List[int]],
    face_out_dir: Path,
    mask_out_dir: Optional[Path],
    num_frames: int = 8,
) -> List[Dict[str, Any]]:
    """Tek bir videodan belirtilen kare indekslerini okur, yuz + maske kirpar ve kaydeder."""
    expected_n = len(target_frame_indices) if target_frame_indices is not None else num_frames
    if face_out_dir.is_dir():
        existing_faces = sorted(face_out_dir.glob("frame_*.png"))
        if len(existing_faces) >= max(1, expected_n // 2):
            reused = []
            for fp in existing_faces:
                try:
                    f_idx = int(fp.stem.split("_")[-1])
                except ValueError:
                    f_idx = 0
                mp = (mask_out_dir / fp.name) if mask_out_dir is not None else None
                has_m = bool(mp is not None and mp.is_file())
                reused.append({
                    "frame_index": f_idx,
                    "face_path": str(fp),
                    "mask_path": str(mp) if has_m else None,
                    "has_mask": has_m,
                    "score": 1.0,
                })
            return reused

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Video acilamadi: %s", video_path)
        return []

    if target_frame_indices is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame_indices = get_evenly_spaced_frame_indices(total_frames, num_frames=num_frames)
    if not target_frame_indices:
        cap.release()
        return []

    mask_cap = None
    if mask_video_path is not None and mask_video_path.is_file():
        mask_cap = cv2.VideoCapture(str(mask_video_path))
        if not mask_cap.isOpened():
            mask_cap = None

    face_out_dir.mkdir(parents=True, exist_ok=True)
    if mask_out_dir is not None:
        mask_out_dir.mkdir(parents=True, exist_ok=True)

    extracted_items = []
    target_set = set(target_frame_indices)
    max_idx = max(target_frame_indices)

    frame_idx = 0
    prev_center: Optional[Tuple[float, float]] = None

    while frame_idx <= max_idx:
        ret = cap.grab()
        if not ret:
            break

        ret_m = mask_cap.grab() if mask_cap is not None else False

        if frame_idx in target_set:
            ok_f, frame_bgr = cap.retrieve()
            if not ok_f or frame_bgr is None:
                break

            mask_gray = None
            if mask_cap is not None and ret_m:
                ok_m, mask_bgr = mask_cap.retrieve()
                if ok_m and mask_bgr is not None:
                    mask_gray = (
                        cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
                        if len(mask_bgr.shape) == 3
                        else mask_bgr
                    )

            faces = extractor.detect_all_faces(frame_bgr)
            selected = select_primary_face(
                faces,
                mask_frame_gray=mask_gray,
                prev_face_center=prev_center,
                frame_shape=frame_bgr.shape[:2],
            )
            if selected is not None:
                bbox, score, kps = selected
                crop_box = compute_crop_box(
                    frame_bgr.shape[:2],
                    bbox,
                    margin_ratio=extractor.margin_ratio,
                    square_box=True,
                    landmarks_5pt=kps,
                )
                x1, y1, x2, y2 = crop_box
                crop = frame_bgr[y1:y2, x1:x2]
                interp = (
                    cv2.INTER_AREA
                    if (crop.shape[0] > extractor.target_size or crop.shape[1] > extractor.target_size)
                    else cv2.INTER_LINEAR
                )
                face_crop = cv2.resize(crop, (extractor.target_size, extractor.target_size), interpolation=interp)

                face_filename = f"frame_{frame_idx:05d}.png"
                face_file_path = face_out_dir / face_filename
                cv2.imwrite(str(face_file_path), face_crop, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])

                mask_file_path = None
                has_mask = False
                if mask_out_dir is not None and mask_gray is not None:
                    mask_crop = crop_mask_with_box(mask_gray, crop_box, target_size=extractor.target_size)
                    mask_filename = f"frame_{frame_idx:05d}.png"
                    mask_file_path = mask_out_dir / mask_filename
                    cv2.imwrite(str(mask_file_path), mask_crop, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                    has_mask = True

                prev_center = (bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3])
                extracted_items.append({
                    "frame_index": frame_idx,
                    "face_path": str(face_file_path),
                    "mask_path": str(mask_file_path) if mask_file_path is not None else None,
                    "has_mask": has_mask,
                    "score": round(float(score), 4),
                })

        frame_idx += 1

    cap.release()
    if mask_cap is not None:
        mask_cap.release()

    return extracted_items


def run_dataset_extraction(
    dataset_mode: str = "ffpp",
    detector_name: str = "scrfd",
    output_dir: Path = KOD_ROOT / "data" / "processed_faces",
    max_videos_per_category: Optional[int] = None,
    compressions: Sequence[str] = ("c23", "c40"),
    only_test_list: bool = True,
    manifest_path: Optional[Path] = None,
    num_workers: int = 16,
) -> Dict[str, Any]:
    """FF++ ve Celeb-DF v2 veri setleri icin yapilandirilmis toplu yuz/maske cikarim fonksiyonu."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    from dataset import build_or_load_ffpp_splits, is_video_in_split, parse_celebdf_test_list

    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = output_dir / f"manifest_{dataset_mode}.json"

    if detector_name.lower() == "scrfd":
        extractor: Any = SCRFDFaceExtractor()
    else:
        extractor = YuNetFaceExtractor()

    manifest_entries: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []

    if dataset_mode.lower() == "ffpp":
        ffpp_root = get_ffpp_root()
        splits = build_or_load_ffpp_splits()
        test_split_dict = splits.get("test", {})
        logger.info("FF++ Veri Seti Cikarimi: root=%s | kompresyon=%s | workers=%d", ffpp_root, compressions, num_workers)

        # 1. Gercek Videolar (original_sequences/youtube): 32 kare/video
        for comp in compressions:
            yt_dir = ffpp_root / "original_sequences" / "youtube" / comp / "videos"
            if not yt_dir.is_dir():
                yt_dir = ffpp_root / "original_sequences" / "youtube" / comp
            videos = collect_videos(yt_dir, max_videos=max_videos_per_category) if yt_dir.is_dir() else []
            logger.info("  [original/youtube/%s]: %d video siraya eklendi (32 kare/video)", comp, len(videos))
            for vid in videos:
                tasks.append({
                    "video_path": vid,
                    "mask_video_path": None,
                    "num_frames": 32,
                    "face_out_dir": output_dir / "faces" / "original_sequences" / "youtube" / comp / vid.stem,
                    "mask_out_dir": None,
                    "label": 0,
                    "manipulation": "youtube",
                    "compression": comp,
                    "dataset": "ffpp",
                    "video_id": vid.stem,
                })

        # 2. Egitim Manipulasyonlari: Deepfakes, Face2Face, FaceSwap, NeuralTextures (8 kare/video)
        train_manipulations = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        for manip in train_manipulations:
            for comp in compressions:
                manip_dir = ffpp_root / "manipulated_sequences" / manip / comp / "videos"
                if not manip_dir.is_dir():
                    manip_dir = ffpp_root / "manipulated_sequences" / manip / comp
                videos = collect_videos(manip_dir, max_videos=max_videos_per_category) if manip_dir.is_dir() else []
                logger.info("  [%s/%s]: %d video siraya eklendi (8 kare/video)", manip, comp, len(videos))

                masks_root = ffpp_root / "manipulated_sequences" / manip / "masks" / "videos"
                has_official_masks = masks_root.is_dir()

                for vid in videos:
                    mask_vid_path = masks_root / f"{vid.stem}.mp4" if has_official_masks else None
                    if mask_vid_path is not None and not mask_vid_path.is_file():
                        mask_vid_path = None
                    tasks.append({
                        "video_path": vid,
                        "mask_video_path": mask_vid_path,
                        "num_frames": 8,
                        "face_out_dir": output_dir / "faces" / "manipulated_sequences" / manip / comp / vid.stem,
                        "mask_out_dir": (
                            output_dir / "masks" / "manipulated_sequences" / manip / comp / vid.stem
                            if mask_vid_path is not None
                            else None
                        ),
                        "label": 1,
                        "manipulation": manip,
                        "compression": comp,
                        "dataset": "ffpp",
                        "video_id": vid.stem,
                    })

        # 3. Protokol 2 Test Manipulasyonlari: FaceShifter ve DeepFakeDetection (Test split: 140 video/kompresyon, 8 kare/video)
        test_sets = [
            ("manipulated_sequences", "FaceShifter", 1),
            ("manipulated_sequences", "DeepFakeDetection", 1),
        ]
        for category, sub_name, label in test_sets:
            for comp in compressions:
                sub_dir = ffpp_root / category / sub_name / comp / "videos"
                if not sub_dir.is_dir():
                    sub_dir = ffpp_root / category / sub_name / comp
                all_videos = collect_videos(sub_dir) if sub_dir.is_dir() else []
                if not all_videos:
                    continue
                if sub_name == "FaceShifter":
                    videos = [v for v in all_videos if is_video_in_split(test_split_dict, v.stem)]
                else:
                    import random as _rnd
                    rng = _rnd.Random(42)
                    shuffled = list(all_videos)
                    rng.shuffle(shuffled)
                    videos = sorted(shuffled[:140])
                if max_videos_per_category is not None:
                    videos = videos[:max_videos_per_category]

                logger.info("  [Protokol-2 Test: %s/%s/%s]: %d test videosu siraya eklendi", category, sub_name, comp, len(videos))
                masks_root = ffpp_root / category / sub_name / "masks" / "videos"
                has_official_masks = masks_root.is_dir()

                for vid in videos:
                    mask_vid_path = masks_root / f"{vid.stem}.mp4" if has_official_masks else None
                    if mask_vid_path is not None and not mask_vid_path.is_file():
                        mask_vid_path = None
                    tasks.append({
                        "video_path": vid,
                        "mask_video_path": mask_vid_path,
                        "num_frames": 8,
                        "face_out_dir": output_dir / "faces" / category / sub_name / comp / vid.stem,
                        "mask_out_dir": (
                            output_dir / "masks" / category / sub_name / comp / vid.stem
                            if mask_vid_path is not None
                            else None
                        ),
                        "label": label,
                        "manipulation": sub_name,
                        "compression": comp,
                        "dataset": "ffpp",
                        "video_id": vid.stem,
                    })

    elif dataset_mode.lower() == "celebdf":
        celebdf_root = get_celebdf_root()
        logger.info("Celeb-DF v2 Cikarimi: root=%s (only_test_list=%s)", celebdf_root, only_test_list)
        test_entries = parse_celebdf_test_list(celebdf_root=celebdf_root)
        if max_videos_per_category is not None:
            test_entries = test_entries[:max_videos_per_category]
        logger.info("  [Celeb-DF v2 Test Seti]: %d video siraya eklendi (16 kare/video)", len(test_entries))
        for entry in test_entries:
            vid = Path(entry["video_path"])
            if not vid.is_file():
                continue
            label = int(entry["label"])
            manip = str(entry["manipulation"])
            folder_name = vid.parent.name
            tasks.append({
                "video_path": vid,
                "mask_video_path": None,
                "num_frames": 16,
                "face_out_dir": output_dir / "faces" / "celebdf" / folder_name / vid.stem,
                "mask_out_dir": None,
                "label": label,
                "manipulation": manip,
                "compression": "original",
                "dataset": "celebdf",
                "video_id": vid.stem,
            })

    def _worker(t_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = _process_single_video_frames(
            extractor=extractor,
            video_path=t_info["video_path"],
            mask_video_path=t_info["mask_video_path"],
            target_frame_indices=None,
            face_out_dir=t_info["face_out_dir"],
            mask_out_dir=t_info["mask_out_dir"],
            num_frames=t_info["num_frames"],
        )
        entries = []
        for item in items:
            rel_mask = (
                str(Path(item["mask_path"]).relative_to(output_dir))
                if item["mask_path"] is not None
                else None
            )
            entries.append({
                "frame_path": str(Path(item["face_path"]).relative_to(output_dir)),
                "mask_path": rel_mask,
                "has_mask": item["has_mask"] if t_info["label"] == 1 else True,
                "label": t_info["label"],
                "manipulation": t_info["manipulation"],
                "compression": t_info["compression"],
                "dataset": t_info["dataset"],
                "video_id": t_info["video_id"],
                "frame_index": item["frame_index"],
            })
        return entries

    logger.info("Toplam %d video %d is parcacigi ile isleniyor...", len(tasks), num_workers)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting ({dataset_mode})"):
            try:
                manifest_entries.extend(fut.result())
            except Exception as exc:
                logger.warning("Video isleme hatasi: %s", exc)

    elapsed = time.perf_counter() - t0
    manifest_data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_mode": dataset_mode,
        "detector": detector_name,
        "total_frames": len(manifest_entries),
        "elapsed_sec": round(elapsed, 2),
        "items": manifest_entries,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifest kaydedildi: %s (%d kare, %.2f sn)", manifest_path, len(manifest_entries), elapsed)
    return manifest_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Videolardan SCRFD-10G / YuNet ile yuz ve maske cikarim araci."
    )
    parser.add_argument("--data_dir", type=Path, default=None, help="Girdi video klasoru veya video dosyasi (dir modu).")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=KOD_ROOT / "data" / "processed_faces",
        help="Kirpilan karelerin kaydedilecegi ana dizin.",
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="dir",
        choices=["dir", "ffpp", "celebdf"],
        help="Veri cikarim modu: generic dizin, ffpp veya celebdf.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="scrfd",
        choices=["scrfd", "yunet"],
        help="Kullanilacak yuz dedektoru: scrfd veya yunet.",
    )
    parser.add_argument("--frame_interval", type=int, default=30, help="Kaç karede bir orneklenecegi (dir modu).")
    parser.add_argument("--max_videos", type=int, default=None, help="Islenecek azami video sayisi.")
    parser.add_argument("--margin_ratio", type=float, default=0.20, help="Yuz kutusu marj orani (0.20 = %%20).")
    parser.add_argument("--target_size", type=int, default=224, help="Cikti yuz boyutu (224 -> 224x224).")
    parser.add_argument("--score_threshold", type=float, default=0.50, help="Guven esigi.")
    parser.add_argument("--model_path", type=Path, default=None, help="Ozel model dosya yolu.")
    parser.add_argument("--log_path", type=Path, default=DEFAULT_LOG_PATH, help="JSON log dosya yolu.")
    parser.add_argument("--only_test_list", action="store_true", default=True, help="Celeb-DF icin sadece test listesindekileri cikar.")
    parser.add_argument("--compressions", nargs="+", default=["c23", "c40"], help="FF++ sikistirma seviyeleri.")
    parser.add_argument("--num_workers", type=int, default=16, help="Paralel video cikarim is parcacigi sayisi.")
    return parser


if __name__ == "__main__":
    import sys

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.dataset_mode in ["ffpp", "celebdf"]:
        run_dataset_extraction(
            dataset_mode=args.dataset_mode,
            detector_name=args.detector,
            output_dir=args.output_dir,
            max_videos_per_category=args.max_videos,
            compressions=args.compressions,
            only_test_list=args.only_test_list,
            num_workers=args.num_workers,
        )
    elif args.data_dir is not None:
        run_extraction(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            frame_interval=args.frame_interval,
            max_videos=args.max_videos,
            margin_ratio=args.margin_ratio,
            target_size=args.target_size,
            score_threshold=args.score_threshold,
            model_path=args.model_path if args.model_path else (DEFAULT_SCRFD_MODEL_PATH if args.detector == "scrfd" else DEFAULT_YUNET_PATH),
            log_path=args.log_path,
            detector_name=args.detector,
        )
    else:
        print("=== face_extractor.py Kendi Kendini Test Modu ===")
        sha = verify_sha256(DEFAULT_YUNET_PATH)
        print(f"[OK] YuNet ONNX SHA256 dogrulandi: {sha}")

        # SCRFD test
        scrfd_det = SCRFDFaceExtractor(score_threshold=0.5)
        print(f"[OK] SCRFDFaceExtractor olusturuldu (Session: {scrfd_det._session is not None}, Fallback: {scrfd_det._fallback_yunet is not None})")

        # Sentetik kare uzerinde test
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        faces_blank = scrfd_det.detect_all_faces(blank)
        assert len(faces_blank) == 0, "Bos karede yuz bulunmamaliydi!"
        print("[OK] Bos kare negatif yuz testi basarili.")

        # crop_face_with_margin 5-point landmark testi
        test_bbox = (100.0, 100.0, 80.0, 80.0)
        test_kps = np.array([[120.0, 130.0], [160.0, 130.0], [140.0, 150.0], [125.0, 165.0], [155.0, 165.0]])
        crop_res = crop_face_with_margin(blank, test_bbox, margin_ratio=0.20, target_size=224, landmarks_5pt=test_kps)
        assert crop_res.shape == (224, 224, 3), f"Beklenen (224, 224, 3), alinan {crop_res.shape}"
        print("[OK] 5-point landmark ile crop_face_with_margin testi basarili.")

        # Hybrid Primary Face Selection testi
        # Mode A: Maske ile secim
        test_faces = [
            ((50.0, 50.0, 100.0, 100.0), 0.90, test_kps),
            ((300.0, 300.0, 100.0, 100.0), 0.85, test_kps),
        ]
        synthetic_mask = np.zeros((480, 640), dtype=np.uint8)
        synthetic_mask[310:380, 310:380] = 255  # 2. yuzun uzerinde maske
        selected_a = select_primary_face(test_faces, mask_frame_gray=synthetic_mask, frame_shape=(480, 640))
        assert selected_a is not None and selected_a[0][0] == 300.0, "Mode A maske ile kesisen yuzu secmeliydi!"
        print("[OK] Hybrid Selection Mode A (Mask-Guided) basarili.")

        # Mode B: Zamansal sureklilik testi
        prev_c = (55.0, 55.0)  # 1. yuze yakin
        selected_b = select_primary_face(test_faces, mask_frame_gray=None, prev_face_center=prev_c, frame_shape=(480, 640))
        assert selected_b is not None and selected_b[0][0] == 50.0, "Mode B onceki merkeze yakin yuzu secmeliydi!"
        print("[OK] Hybrid Selection Mode B (Continuity) basarili.")
        print("=== Tum testler basariyla gecti ===")
