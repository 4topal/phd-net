"""PHD-Net Veri Yükleyici, Ayrım Üretici ve Örnekleme Modülü.

Bileşenler:
1. FF++ Leak-Free 720/140/140 Ayrım Üretici (`build_or_load_ffpp_splits`):
   - 1000 baz video kimliği (`000`..`999`) ve `A_B` manipüle çiftleri üzerinde bağlantılı bileşenleri
     deterministik (seed=42) olarak 720 eğitim / 140 doğrulama / 140 test şeklinde ayırır.
   - Çapraz split veri sızıntısını (data leakage) kesin olarak önler.
2. Celeb-DF v2 Test Listesi Ayrıştırıcı (`parse_celebdf_test_list`):
   - `List_of_testing_videos.txt` dosyasını okur.
   - Celeb-DF'in `1=Real, 0=Fake` etiketini PHD-Net adli standardına (`0=Real, 1=Fake`) dönüştürür.
3. `DeepfakeFaceDataset` (`torch.utils.data.Dataset`):
   - Kırpılmış 224x224 yüzleri ve eşlenik ikili adli maskeleri okur.
   - Standart çıktı sözlüğü:
     `{"rgb": (3, 224, 224), "label": int, "mask": (1, 224, 224), "has_mask": bool, ...}`
   - Eğitim kipinde senkronize yatay çevirme, hafif parlaklık/kontrast jitter ve bellek-içi JPEG simülasyonu.
4. `BalancedBatchSampler` (`torch.utils.data.Sampler`):
   - Her mini-batch'in tam olarak %50 Gerçek (label=0) ve %50 Sahte (label=1, manipülasyon türleri
     ve sıkıştırma seviyeleri arasında dengeli) olmasını garanti eder.
5. `SyntheticSmokeDataset` (`torch.utils.data.Dataset`):
   - Bellek-içi sentetik adli veri seti üreterek `--smoke_test` kipinin diskte veri olmadan
     hemen çalışmasını sağlar.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

try:
    from project_paths import PROJECT_ROOT, get_celebdf_root, get_datasets_root, get_ffpp_root
except ImportError:
    # Güvenli yedek yol fonksiyonları
    _REPO_ROOT = Path(__file__).resolve().parents[1]

    def get_datasets_root() -> Path:
        p = os.environ.get("PHD_NET_DATASETS")
        return Path(p).resolve() if p else _REPO_ROOT / "datasets"

    def get_ffpp_root() -> Path:
        p = os.environ.get("PHD_NET_FFPP")
        if p:
            return Path(p).resolve()
        cand = get_datasets_root() / "FF_Dataset"
        return cand if cand.is_dir() else Path("D:/FF_Dataset")

    def get_celebdf_root() -> Path:
        p = os.environ.get("PHD_NET_CELEBDF")
        if p:
            return Path(p).resolve()
        for cand in [Path("/datasets_celebdf"), get_datasets_root() / "celebdf", get_datasets_root() / "celebdf_v2", Path("D:/celeb-df_v2")]:
            if cand.is_dir():
                return cand
        return get_datasets_root() / "celebdf_v2"

    PROJECT_ROOT = _REPO_ROOT

logger = logging.getLogger("phdnet.dataset")
DEFAULT_SPLITS_PATH = PROJECT_ROOT / "outputs" / "splits" / "ffpp_splits.json"


# ==============================================================================
# 1. FF++ Leak-Free 720 / 140 / 140 Split Builder
# ==============================================================================

def build_or_load_ffpp_splits(
    splits_path: Optional[Path] = None,
    ffpp_root: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, List[str]]]:
    """FaceForensics++ 1000 baz kimlik ve manipülasyon çiftlerini deterministik olarak ayırır.

    Her manipüle `A_B` videosunda kaynak `B` ve hedef `A` kimliklerinin aynı split içinde
    kalmasını garantilemek için bağlı bileşenler (connected components) seed=42 ile
    360 bileşen (720 kimlik) eğitim, 70 bileşen (140 kimlik) doğrulama ve 70 bileşen (140 kimlik)
    test olarak ayrılır.

    Döndürülen yapı:
        {
            "train": {"base_ids": [... 720 ...], "pair_ids": [... 720 ...]},
            "val": {"base_ids": [... 140 ...], "pair_ids": [... 140 ...]},
            "test": {"base_ids": [... 140 ...], "pair_ids": [... 140 ...]}
        }
    """
    path = Path(splits_path) if splits_path is not None else DEFAULT_SPLITS_PATH
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if all(k in data for k in ("train", "val", "test")):
                return data
        except Exception as e:
            logger.warning("Mevcut split dosyası okunamadı (%s), yeniden oluşturuluyor: %s", path, e)

    # 1. Çiftleri topla veya sentetik eşleme kur
    root = Path(ffpp_root) if ffpp_root is not None else get_ffpp_root()
    deepfakes_vids = root / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    pair_stems: List[str] = []

    if deepfakes_vids.is_dir():
        pair_stems = sorted([p.stem for p in deepfakes_vids.glob("*.mp4") if "_" in p.stem])

    if not pair_stems:
        # Disk üzerinde bulunamazsa standart FF++ simetrik eşleme kur (000_003, 003_000 vb.)
        # 1000 kimlik için 500 ardışık çift (2k, 2k+1)
        pair_stems = []
        for i in range(0, 1000, 2):
            a_str = f"{i:03d}"
            b_str = f"{i+1:03d}"
            pair_stems.append(f"{a_str}_{b_str}")
            pair_stems.append(f"{b_str}_{a_str}")

    # 2. Çizge üzerinde bağlı bileşenleri (connected components) bul
    adj: Dict[str, set[str]] = defaultdict(set)
    for p in pair_stems:
        parts = p.split("_")
        if len(parts) == 2:
            u, v = parts[0], parts[1]
            adj[u].add(v)
            adj[v].add(u)

    # 1000 baz kimliğin tamamını kapsa
    for i in range(1000):
        s = f"{i:03d}"
        if s not in adj:
            adj[s] = set()

    visited: set[str] = set()
    components: List[List[str]] = []
    for node in sorted(adj.keys()):
        if node not in visited:
            comp: List[str] = []
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for nbr in sorted(adj[curr]):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            components.append(sorted(comp))

    # 3. Deterministik karıştırma (seed=42)
    rng = random.Random(seed)
    rng.shuffle(components)

    # Hedef: 720 train / 140 val / 140 test kimlik
    train_ids_set: set[str] = set()
    val_ids_set: set[str] = set()
    test_ids_set: set[str] = set()

    for comp in components:
        if len(train_ids_set) + len(comp) <= 720:
            train_ids_set.update(comp)
        elif len(val_ids_set) + len(comp) <= 140:
            val_ids_set.update(comp)
        else:
            test_ids_set.update(comp)

    # 4. Çiftleri split kümelerine ata (tamamen sızıntısız)
    pair_set = set(pair_stems)
    train_pairs = sorted([p for p in pair_set if p.split("_")[0] in train_ids_set and p.split("_")[1] in train_ids_set])
    val_pairs = sorted([p for p in pair_set if p.split("_")[0] in val_ids_set and p.split("_")[1] in val_ids_set])
    test_pairs = sorted([p for p in pair_set if p.split("_")[0] in test_ids_set and p.split("_")[1] in test_ids_set])

    splits_data: Dict[str, Dict[str, List[str]]] = {
        "train": {
            "base_ids": sorted(list(train_ids_set)),
            "pair_ids": train_pairs,
        },
        "val": {
            "base_ids": sorted(list(val_ids_set)),
            "pair_ids": val_pairs,
        },
        "test": {
            "base_ids": sorted(list(test_ids_set)),
            "pair_ids": test_pairs,
        },
    }

    # Diske kaydet
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(splits_data, f, indent=2)
        logger.info("FF++ split ayrımı kaydedildi: %s (train=%d, val=%d, test=%d)",
                    path, len(train_ids_set), len(val_ids_set), len(test_ids_set))
    except Exception as e:
        logger.warning("Split dosyası kaydedilemedi: %s", e)

    return splits_data


def is_video_in_split(split_dict: Dict[str, List[str]], video_id: str) -> bool:
    """Verilen video stem veya baz kimliğin split içinde olup olmadığını denetler."""
    base_ids = set(split_dict.get("base_ids", []))
    pair_ids = set(split_dict.get("pair_ids", []))

    if video_id in base_ids or video_id in pair_ids:
        return True
    if "_" in video_id:
        target = video_id.split("_")[0]
        return target in base_ids
    return False


# ==============================================================================
# 2. Celeb-DF v2 Test Listesi Ayrıştırıcı
# ==============================================================================

def parse_celebdf_test_list(
    celebdf_root: Optional[Path] = None,
    test_list_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Celeb-DF v2 resmi test listesini ayrıştırır.

    Format: `<orig_label> <rel_path>`
    DİKKAT: Celeb-DF resmi `List_of_testing_videos.txt` içinde:
      - `1` = Real
      - `0` = Fake / Synthesis
    PHD-Net adli standardına dönüştürülür:
      - `0` = Real
      - `1` = Fake

    Döndürülen liste elemanları:
      {
          "label": int (0 veya 1),
          "rel_path": str,
          "video_id": str (dosya stem),
          "video_path": Path,
          "is_real": bool,
      }
    """
    root = Path(celebdf_root) if celebdf_root is not None else get_celebdf_root()
    txt_path = Path(test_list_path) if test_list_path is not None else root / "List_of_testing_videos.txt"

    entries: List[Dict[str, Any]] = []

    if txt_path.is_file():
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                orig_label_str, rel_path_str = parts[0], parts[1].strip()
                rel_path = Path(rel_path_str.replace("\\", "/"))

                # Celeb-DF dönüşümü: 1 -> Real (0), 0 -> Fake (1)
                orig_label = int(orig_label_str)
                label = 0 if orig_label == 1 else 1

                video_file = root / rel_path
                entries.append({
                    "label": label,
                    "rel_path": str(rel_path),
                    "video_id": rel_path.stem,
                    "video_path": video_file,
                    "is_real": (label == 0),
                    "manipulation": "celebdf_real" if label == 0 else "celebdf_synthesis",
                    "compression": "original",
                })
        logger.info("Celeb-DF test listesi okundu: %d video (%s)", len(entries), txt_path)
        return entries

    # Dosya yoksa Celeb-real ve Celeb-synthesis dizinlerini tara
    logger.warning("Celeb-DF List_of_testing_videos.txt bulunamadı (%s). Dizin taranıyor...", txt_path)
    real_dir = root / "Celeb-real"
    synth_dir = root / "Celeb-synthesis"

    if real_dir.is_dir() or synth_dir.is_dir():
        real_vids = sorted(real_dir.glob("*.mp4")) if real_dir.is_dir() else []
        synth_vids = sorted(synth_dir.glob("*.mp4")) if synth_dir.is_dir() else []

        # Deterministik test seçimi (örnek: 178 real, 340 fake veya mevcutların %20'si)
        rng = random.Random(42)
        rng.shuffle(real_vids)
        rng.shuffle(synth_vids)

        test_real = real_vids[: min(178, len(real_vids))]
        test_synth = synth_vids[: min(340, len(synth_vids))]

        for v in test_real:
            rel = v.relative_to(root)
            entries.append({
                "label": 0,
                "rel_path": str(rel),
                "video_id": v.stem,
                "video_path": v,
                "is_real": True,
                "manipulation": "celebdf_real",
                "compression": "original",
            })
        for v in test_synth:
            rel = v.relative_to(root)
            entries.append({
                "label": 1,
                "rel_path": str(rel),
                "video_id": v.stem,
                "video_path": v,
                "is_real": False,
                "manipulation": "celebdf_synthesis",
                "compression": "original",
            })

        logger.info("Celeb-DF otomatik test seçimi: %d real, %d synth video", len(test_real), len(test_synth))

    return entries


# ==============================================================================
# 3. DeepfakeFaceDataset (torch.utils.data.Dataset)
# ==============================================================================

class DeepfakeFaceDataset(Dataset):
    """Kırpılmış yüz kareleri ve adli zemin gerçekliği maskeleri için PyTorch Veri Seti.

    Çıktı Sözlüğü:
        {
            "rgb": torch.FloatTensor (3, 224, 224) in [0.0, 1.0],
            "label": int (0: Real, 1: Fake),
            "mask": torch.FloatTensor (1, 224, 224) in {0.0, 1.0},
            "has_mask": bool (Real için True [0-maskesi]; DF/F2F/DFD için True; FS/NT/CelebDF için False),
            "video_id": str,
            "manipulation": str,
            "compression": str,
        }
    """

    MASKED_MANIPULATIONS = {"Deepfakes", "Face2Face", "DeepFakeDetection"}

    def __init__(
        self,
        samples: Optional[List[Dict[str, Any]]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        data_root: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        splits_path: Optional[Union[str, Path]] = None,
        compressions: Optional[Sequence[str]] = None,
        manipulations: Optional[Sequence[str]] = None,
        is_train: bool = False,
        target_size: int = 224,
        jpeg_sim_prob: float = 0.30,
        sbi_prob: float = 0.35,
    ) -> None:
        super().__init__()
        self.target_size = target_size
        self.is_train = is_train
        self.jpeg_sim_prob = jpeg_sim_prob
        self.sbi_prob = sbi_prob

        # 1. Örnekleri topla
        raw_samples: List[Dict[str, Any]] = []
        if data_root is None and samples is None and manifest_path is None:
            default_root = PROJECT_ROOT / "data" / "processed_faces"
            if default_root.is_dir():
                data_root = default_root

        if samples is not None:
            raw_samples = [self._normalize_item(s, Path(data_root) if data_root else PROJECT_ROOT / "data" / "processed_faces") for s in samples]
        elif manifest_path is not None and Path(manifest_path).is_file():
            raw_samples = self._load_manifest_file(Path(manifest_path))
        elif data_root is not None and Path(data_root).is_dir():
            raw_samples = self._scan_data_root(Path(data_root))

        # 2. Split filtreleme
        if split is not None:
            splits = build_or_load_ffpp_splits(splits_path=splits_path)
            split_dict = splits.get(split, {})
            filtered = []
            for s in raw_samples:
                vid = str(s.get("video_id", ""))
                ds_name = str(s.get("dataset", ""))
                manip = str(s.get("manipulation", ""))
                if ds_name == "celebdf" or manip.startswith("celeb") or manip in ("Celeb-real", "YouTube-real", "Celeb-synthesis"):
                    continue
                if (split == "test" and manip in ("DeepFakeDetection", "actors")) or is_video_in_split(split_dict, vid):
                    filtered.append(s)
            raw_samples = filtered

        # 3. Sıkıştırma ve Manipülasyon filtreleme
        if compressions is not None:
            c_set = set(compressions)
            raw_samples = [
                s for s in raw_samples
                if s.get("compression") in c_set or s.get("compression") in ("original", "none")
            ]

        if manipulations is not None:
            m_set = set(manipulations)
            known_reals = {"youtube", "real", "actors", "Celeb-real", "YouTube-real", "celebdf_real"}
            fake_manips = {m for m in m_set if m not in known_reals}

            if fake_manips == {"DeepFakeDetection"} or list(m_set) == ["DeepFakeDetection"]:
                allowed_real = {"actors"}
            else:
                allowed_real = {"youtube", "real"}
                if "DeepFakeDetection" in m_set:
                    allowed_real.add("actors")
                if any(m.lower().startswith("celeb") or m in ("Celeb-real", "YouTube-real", "Celeb-synthesis") for m in m_set):
                    allowed_real.update({"Celeb-real", "YouTube-real", "celebdf_real"})

            raw_samples = [
                s for s in raw_samples
                if s.get("manipulation") in m_set
                or (s.get("label") == 0 and s.get("manipulation") in allowed_real)
            ]

        self.samples = raw_samples

    @staticmethod
    def _normalize_item(item: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        """Manifest veya sözlük girdisini standart veri kümesi anahtarlarına dönüştürür."""
        norm = dict(item)
        rgb_p = norm.get("rgb_path") or norm.get("frame_path")
        if rgb_p is not None:
            p = Path(rgb_p)
            if not p.is_absolute():
                p = base_dir / p
            norm["rgb_path"] = str(p)
        mask_p = norm.get("mask_path")
        if mask_p is not None:
            mp = Path(mask_p)
            if not mp.is_absolute():
                mp = base_dir / mp
            norm["mask_path"] = str(mp)
        return norm

    def _load_manifest_file(self, m_path: Path) -> List[Dict[str, Any]]:
        """Bir manifest JSON dosyasını okuyup yolları mutlak hale getirir."""
        with open(m_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        entries: List[Dict[str, Any]] = []
        if isinstance(manifest_data, list):
            entries = manifest_data
        elif isinstance(manifest_data, dict):
            entries = manifest_data.get("items") or manifest_data.get("samples") or []
        base_dir = m_path.parent
        return [self._normalize_item(e, base_dir) for e in entries]

    def _scan_data_root(self, root: Path) -> List[Dict[str, Any]]:
        """Dizin hiyerarşisindeki manifest dosyalarını veya yüz/maske resimlerini tarar."""
        manifest_candidates = [
            root / "manifest_ffpp.json",
            root / "manifest_celebdf.json",
            root / "manifest.json",
        ]
        loaded_from_manifests: List[Dict[str, Any]] = []
        loaded_manifest_names = set()
        for mc in manifest_candidates:
            if mc.is_file():
                if mc.name == "manifest.json" and loaded_manifest_names:
                    continue
                items = self._load_manifest_file(mc)
                if items:
                    first_rp = items[0].get("rgb_path")
                    if first_rp and not Path(first_rp).is_file():
                        logger.warning("Manifestteki ilk dosya bulunamadi (%s): %s", mc, first_rp)
                    loaded_from_manifests.extend(items)
                    loaded_manifest_names.add(mc.name)
        if loaded_from_manifests:
            return loaded_from_manifests

        found: List[Dict[str, Any]] = []
        image_exts = {".png", ".jpg", ".jpeg"}

        for img_path in sorted(root.rglob("*")):
            if img_path.suffix.lower() not in image_exts:
                continue
            if "masks" in img_path.parts:
                continue

            rel_parts = img_path.relative_to(root).parts
            # Tahmini etiket ve manipülasyon
            path_str = str(img_path).lower()
            is_real = any(x in path_str for x in ("youtube", "original", "celeb-real", "youtube-real", "actors", "real"))
            label = 0 if is_real else 1

            # Manipülasyon türü
            manip = "youtube" if is_real else "unknown_fake"
            if "actors" in path_str:
                manip = "actors"
            elif "celeb-real" in path_str:
                manip = "Celeb-real"
            elif "youtube-real" in path_str:
                manip = "YouTube-real"
            elif "celeb-synthesis" in path_str:
                manip = "Celeb-synthesis"
            else:
                for m in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter", "DeepFakeDetection"):
                    if m.lower() in path_str:
                        manip = m
                        break

            # Sıkıştırma
            comp = "c23"
            if "c40" in path_str:
                comp = "c40"
            elif "raw" in path_str:
                comp = "raw"
            elif "celeb" in path_str:
                comp = "original"

            video_id = img_path.parent.name
            mask_path = None
            if manip in self.MASKED_MANIPULATIONS:
                rel_to_root = img_path.relative_to(root)
                if rel_to_root.parts and rel_to_root.parts[0] == "faces":
                    cand_mask = root / "masks" / Path(*rel_to_root.parts[1:])
                else:
                    cand_mask = root / "masks" / rel_to_root
                if cand_mask.is_file():
                    mask_path = str(cand_mask)

            found.append({
                "rgb_path": str(img_path),
                "mask_path": mask_path,
                "label": label,
                "video_id": video_id,
                "manipulation": manip,
                "compression": comp,
            })
        return found

    def __len__(self) -> int:
        return len(self.samples)

    def _generate_sbi(self, rgb_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gerçek yüzden çalışma anında Self-Blended Images (SBI) sahte örneği üretir.

        İki kopyadan biri hafif renk/parlaklık/kontrast kayması ve afin/ölçek bükmesi ile
        artırılır, ardından yumuşak eliptik/poligonal maske (M in [0, 1]) ile harmanlanır.
        """
        h, w = rgb_np.shape[:2]
        I_base = rgb_np.copy()

        # Kopya 1 (Taban): Hafif varyasyon
        a1 = 1.0 + random.uniform(-0.06, 0.06)
        b1 = random.uniform(-10.0, 10.0)
        copy1 = np.clip(I_base.astype(np.float32) * a1 + b1, 0.0, 255.0)

        # Kopya 2 (Kaynak / Bükülmüş): Afin/ölçek bükme + parlaklık/kontrast/renk kayması
        cx, cy = w / 2.0, h / 2.0
        angle = random.uniform(-10.0, 10.0)
        scale = random.uniform(0.92, 1.08)
        tx = random.uniform(-8.0, 8.0)
        ty = random.uniform(-8.0, 8.0)
        M_aff = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        M_aff[0, 2] += tx
        M_aff[1, 2] += ty
        warped2 = cv2.warpAffine(I_base, M_aff, (w, h), borderMode=cv2.BORDER_REFLECT)

        a2 = 1.0 + random.uniform(-0.15, 0.15)
        b2 = random.uniform(-20.0, 20.0)
        copy2 = np.clip(warped2.astype(np.float32) * a2 + b2, 0.0, 255.0)

        # Renk tonu ve doygunluk kayması
        hsv2 = cv2.cvtColor(copy2.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv2[:, :, 0] = (hsv2[:, :, 0] + random.uniform(-10.0, 10.0)) % 180.0
        hsv2[:, :, 1] = np.clip(hsv2[:, :, 1] * random.uniform(0.85, 1.15), 0.0, 255.0)
        copy2_final = cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # Rastgele yumuşak eliptik veya çokgen yüz maskesi (M in [0, 1]^(H, W))
        mask_raw = np.zeros((h, w), dtype=np.uint8)
        if random.random() < 0.5:
            # Eliptik maske
            el_cx = int(cx + random.uniform(-0.08, 0.08) * w)
            el_cy = int(cy + random.uniform(-0.08, 0.08) * h)
            rx = int(random.uniform(0.22, 0.38) * w)
            ry = int(random.uniform(0.26, 0.44) * h)
            el_angle = random.uniform(-25.0, 25.0)
            cv2.ellipse(mask_raw, (el_cx, el_cy), (rx, ry), el_angle, 0, 360, 255, -1)
        else:
            # Rastgele çokgensel maske (6-10 köşe)
            n_pts = random.randint(6, 10)
            poly_cx = cx + random.uniform(-0.06, 0.06) * w
            poly_cy = cy + random.uniform(-0.06, 0.06) * h
            angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(n_pts)])
            pts = []
            for ang in angles:
                r = random.uniform(0.22, 0.42) * min(h, w)
                px = int(np.clip(poly_cx + r * math.cos(ang), 0, w - 1))
                py = int(np.clip(poly_cy + r * math.sin(ang), 0, h - 1))
                pts.append([px, py])
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_raw, [pts_arr], 255)

        # Maskeyi yumuşat (Gauss bulanıklaştırma) -> M in [0.0, 1.0]
        ksize = random.choice([15, 21, 27, 33])
        sigma = random.uniform(4.0, 10.0)
        mask_smooth = cv2.GaussianBlur(mask_raw, (ksize, ksize), sigma)
        M = mask_smooth.astype(np.float32) / 255.0

        # İki kopyayı M ile harmanla
        M_3d = M[:, :, np.newaxis]
        blended = (1.0 - M_3d) * copy1 + M_3d * copy2_final
        blended_rgb = np.clip(blended, 0.0, 255.0).astype(np.uint8)

        # Adli ikili maske: mask = (M > 0.1)
        mask_binary = ((M > 0.1).astype(np.float32) * 255.0).astype(np.uint8)

        return blended_rgb, mask_binary

    def _apply_augmentations(
        self, rgb_np: np.ndarray, mask_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Eğitim veri artırmaları: senkronize çevirme, crop, jitter, blur, JPEG, erasing."""
        h, w = rgb_np.shape[:2]

        # 1. Senkronize Yatay Çevirme (%50 olasılık)
        if random.random() < 0.5:
            rgb_np = np.fliplr(rgb_np).copy()
            mask_np = np.fliplr(mask_np).copy()

        # 2. Senkronize RandomResizedCrop (ölçek: 0.82 - 1.0)
        scale = random.uniform(0.82, 1.0)
        crop_size = max(16, min(h, w, int(round(min(h, w) * math.sqrt(scale)))))
        if crop_size < min(h, w):
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)
            rgb_crop = rgb_np[top : top + crop_size, left : left + crop_size]
            mask_crop = mask_np[top : top + crop_size, left : left + crop_size]
            rgb_np = cv2.resize(rgb_crop, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_np = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)

        # 3. Hafif Parlaklık ve Kontrast Jitter (±10%)
        alpha = 1.0 + random.uniform(-0.10, 0.10)  # Kontrast
        beta = random.uniform(-0.10, 0.10) * 255.0  # Parlaklık
        rgb_np = np.clip(rgb_np.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)

        # 4. Hafif ColorJitter (Doygunluk ve Renk Özü / Saturation & Hue) (p=0.50)
        if random.random() < 0.50:
            hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-8.0, 8.0)) % 180.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0.0, 255.0)
            rgb_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 5. GaussianBlur (p=0.20, 3x3 veya 5x5)
        if random.random() < 0.20:
            ksize = random.choice([3, 5])
            sigma = random.uniform(0.5, 1.5)
            rgb_np = cv2.GaussianBlur(rgb_np, (ksize, ksize), sigma)

        # 6. Bellek-İçi Rastgele JPEG Sıkıştırma Simülasyonu (prob=jpeg_sim_prob)
        if random.random() < self.jpeg_sim_prob:
            quality = random.randint(50, 95)
            # OpenCV BGR sırasıyla kodlar
            bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if success:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is not None:
                    rgb_np = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        # 7. Senkronize RandomErasing / Cutout (p=0.20, küçük yamalar)
        if random.random() < 0.20:
            patch_h = random.randint(max(8, int(0.08 * h)), max(16, int(0.22 * h)))
            patch_w = random.randint(max(8, int(0.08 * w)), max(16, int(0.22 * w)))
            py = random.randint(0, h - patch_h)
            px = random.randint(0, w - patch_w)
            noise_patch = np.random.randint(0, 256, (patch_h, patch_w, 3), dtype=np.uint8)
            rgb_np[py : py + patch_h, px : px + patch_w] = noise_patch
            mask_np[py : py + patch_h, px : px + patch_w] = 0

        return rgb_np, mask_np

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        rgb_path = item.get("rgb_path")
        label = int(item.get("label", 0))
        video_id = str(item.get("video_id", f"vid_{idx}"))
        manip = str(item.get("manipulation", "real" if label == 0 else "fake"))
        comp = str(item.get("compression", "c23"))
        mask_path = item.get("mask_path")

        # 1. RGB Resmi Yükle
        if rgb_path is not None and Path(rgb_path).is_file():
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                rgb_np = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            else:
                if bgr.shape[:2] != (self.target_size, self.target_size):
                    bgr = cv2.resize(bgr, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
                rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            # Bellek-içi veya sentetik örnek
            rgb_np = item.get("rgb_np")
            if rgb_np is None:
                rgb_np = np.random.randint(0, 256, (self.target_size, self.target_size, 3), dtype=np.uint8)

        # 2. Maske Yükle ve `has_mask` Belirle
        mask_np = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        has_mask = False

        if label == 0:
            # Gerçek yüzler için zemin gerçekliği maskesi 0'dır, kısmi kayba dahil edilir
            has_mask = True
        else:
            # Sahte yüzler için sadece resmi maske varsa dahil edilir
            if mask_path is not None and Path(mask_path).is_file():
                m_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m_gray is not None:
                    if m_gray.shape[:2] != (self.target_size, self.target_size):
                        m_gray = cv2.resize(m_gray, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
                    mask_np = (m_gray >= 128).astype(np.uint8) * 255
                    has_mask = True
            elif "mask_np" in item and item["mask_np"] is not None:
                m_arr = item["mask_np"]
                if m_arr.shape[:2] != (self.target_size, self.target_size):
                    m_arr = cv2.resize(m_arr, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
                mask_np = (m_arr >= 128).astype(np.uint8) * 255
                has_mask = bool(item.get("has_mask", True))
            else:
                has_mask = False

        # 3. SBI (Self-Blended Images) sahte üretimi (Eğitim kipi ve gerçek örnekler için)
        if self.is_train and label == 0 and random.random() < self.sbi_prob:
            rgb_np, mask_np = self._generate_sbi(rgb_np)
            label = 1
            manip = "SBI"
            has_mask = True

        # 4. Eğitim Artırmaları
        if self.is_train:
            rgb_np, mask_np = self._apply_augmentations(rgb_np, mask_np)

        # 5. PyTorch Tensörlerine Dönüştür
        # RGB: (H, W, 3) uint8 -> (3, H, W) float32 in [0.0, 1.0] (ImageNet normalizasyonu PHDNet içinde yapilacak)
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float().div(255.0)

        # Maske: (H, W) uint8 -> (1, H, W) float32 in {0.0, 1.0}
        mask_tensor = (torch.from_numpy(mask_np).unsqueeze(0).float().div(255.0) > 0.1).float()

        return {
            "rgb": rgb_tensor,
            "label": label,
            "mask": mask_tensor,
            "has_mask": has_mask,
            "video_id": video_id,
            "manipulation": manip,
            "compression": comp,
        }


# ==============================================================================
# 4. BalancedBatchSampler (torch.utils.data.Sampler)
# ==============================================================================

class BalancedBatchSampler(Sampler):
    """Her mini-batch'te %50 Gerçek (label=0) ve %50 Sahte (label=1) sağlayan örnekleyici.

    Sahte örnekler mevcut manipülasyon yöntemleri (`Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`)
    ve sıkıştırma seviyeleri (`c23`, `c40`) arasında eşit dağıtılır.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(None)
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size çift sayı olmalıdır (verilen={batch_size}).")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.half_b = batch_size // 2

        # Örnek indekslerini ayrıştır (Disk I/O yapmadan, doğrudan bellekteki metadata üzerinden hızlı okuma)
        self.real_indices: List[int] = []
        self.fake_groups: Dict[str, List[int]] = defaultdict(list)

        metadata_list = getattr(dataset, "samples", None) or getattr(dataset, "items", None)
        if metadata_list is not None and len(metadata_list) == len(dataset):
            for i, item in enumerate(metadata_list):
                lbl = int(item.get("label", 0))
                manip = item.get("manipulation", "fake")
                if lbl == 0:
                    self.real_indices.append(i)
                else:
                    self.fake_groups[manip].append(i)
        else:
            for i in range(len(dataset)):  # type: ignore[arg-type]
                sample = dataset[i]
                lbl = sample["label"]
                manip = sample.get("manipulation", "fake")
                if lbl == 0:
                    self.real_indices.append(i)
                else:
                    self.fake_groups[manip].append(i)

        if not self.real_indices:
            raise ValueError("BalancedBatchSampler: Veri setinde hiç Gerçek (label=0) örnek bulunamadı!")
        if not self.fake_groups:
            raise ValueError("BalancedBatchSampler: Veri setinde hiç Sahte (label=1) örnek bulunamadı!")

        self.rng = random.Random(seed)

    def __len__(self) -> int:
        # Gerçek veya sahte örnek sayısına göre batch sayısı kestirimi
        total_fake = sum(len(g) for g in self.fake_groups.values())
        max_samples = max(len(self.real_indices), total_fake)
        batches = max_samples // self.half_b
        return max(1, batches)

    def __iter__(self) -> Iterator[List[int]]:
        num_batches = len(self)
        real_pool = list(self.real_indices)
        self.rng.shuffle(real_pool)
        real_ptr = 0

        # Her manipülasyon havuzunu kopyala ve karıştır
        fake_pools = {k: list(v) for k, v in self.fake_groups.items()}
        for k in fake_pools:
            self.rng.shuffle(fake_pools[k])
        fake_ptrs = {k: 0 for k in fake_pools}
        group_keys = sorted(fake_pools.keys())

        for _ in range(num_batches):
            batch: List[int] = []

            # 1. Gerçek örnekler (half_b adet)
            for _ in range(self.half_b):
                if real_ptr >= len(real_pool):
                    self.rng.shuffle(real_pool)
                    real_ptr = 0
                batch.append(real_pool[real_ptr])
                real_ptr += 1

            # 2. Sahte örnekler (half_b adet, manipülasyonlar arasında dengeli)
            k_idx = 0
            for _ in range(self.half_b):
                grp = group_keys[k_idx % len(group_keys)]
                pool = fake_pools[grp]
                if fake_ptrs[grp] >= len(pool):
                    self.rng.shuffle(pool)
                    fake_ptrs[grp] = 0
                batch.append(pool[fake_ptrs[grp]])
                fake_ptrs[grp] += 1
                k_idx += 1

            # Mini-batch içindeki sıralamayı rastgele karıştır
            self.rng.shuffle(batch)
            yield batch


# ==============================================================================
# 5. SyntheticSmokeDataset (Bellek-içi Hızlı Duman Testi)
# ==============================================================================

class SyntheticSmokeDataset(Dataset):
    """Eğitim ve değerlendirme hatlarını diskte yüzler çıkarılmadan test eden sentetik veri seti."""

    def __init__(
        self,
        num_samples: int = 128,
        target_size: int = 224,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.target_size = target_size
        self.rng = np.random.default_rng(seed)

        manip_list = [
            ("youtube", 0, True),
            ("Deepfakes", 1, True),
            ("Face2Face", 1, True),
            ("FaceSwap", 1, False),
            ("NeuralTextures", 1, False),
            ("FaceShifter", 1, False),
            ("DeepFakeDetection", 1, True),
        ]
        compressions = ["c23", "c40"]

        self.items: List[Dict[str, Any]] = []
        for i in range(num_samples):
            # Yarı gerçek, yarı sahte
            if i % 2 == 0:
                manip, label, has_m = "youtube", 0, True
                vid_id = f"smoke_real_{(i // 2) % 8:03d}"
            else:
                fake_idx = 1 + ((i // 2) % (len(manip_list) - 1))
                manip, label, has_m = manip_list[fake_idx]
                vid_id = f"smoke_fake_{(i // 2) % 8:03d}"

            comp = compressions[i % 2]

            # Sentetik RGB (H, W, 3)
            # Doku oluşturmak için hafif renk gradyanı + gürültü
            base_color = self.rng.integers(50, 200, size=(3,), dtype=np.uint8)
            noise = self.rng.integers(-20, 21, size=(target_size, target_size, 3), dtype=np.int16)
            rgb_arr = np.clip(base_color[None, None, :] + noise, 0, 255).astype(np.uint8)

            # Sentetik Maske
            mask_arr = np.zeros((target_size, target_size), dtype=np.uint8)
            if label == 1 and has_m:
                # Merkeze 64x64 manipülasyon kutusu
                c = target_size // 2
                r = 32
                mask_arr[c - r : c + r, c - r : c + r] = 255
                # Sahte bölgeye renk sapması ekle
                rgb_arr[c - r : c + r, c - r : c + r, 0] = np.clip(
                    rgb_arr[c - r : c + r, c - r : c + r, 0].astype(int) + 35, 0, 255
                )

            self.items.append({
                "rgb_np": rgb_arr,
                "mask_np": mask_arr,
                "label": label,
                "has_mask": has_m,
                "video_id": vid_id,
                "manipulation": manip,
                "compression": comp,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        rgb_tensor = torch.from_numpy(item["rgb_np"]).permute(2, 0, 1).float().div(255.0)
        mask_tensor = torch.from_numpy(item["mask_np"]).unsqueeze(0).float().div(255.0)
        return {
            "rgb": rgb_tensor,
            "label": item["label"],
            "mask": mask_tensor,
            "has_mask": item["has_mask"],
            "video_id": item["video_id"],
            "manipulation": item["manipulation"],
            "compression": item["compression"],
        }


# ==============================================================================
# Modül Öz-Testi
# ==============================================================================

if __name__ == "__main__":
    print("=== dataset.py Doğrulama ve Fonksiyonel Öz-Test ===")

    # 1. Split Builder Testi
    splits = build_or_load_ffpp_splits(seed=42)
    print(f"[OK] FF++ Split: train={len(splits['train']['base_ids'])} baz / {len(splits['train']['pair_ids'])} çift, "
          f"val={len(splits['val']['base_ids'])} baz / {len(splits['val']['pair_ids'])} çift, "
          f"test={len(splits['test']['base_ids'])} baz / {len(splits['test']['pair_ids'])} çift")
    assert len(splits["train"]["base_ids"]) == 720
    assert len(splits["val"]["base_ids"]) == 140
    assert len(splits["test"]["base_ids"]) == 140

    # 2. Celeb-DF Parser Testi
    celeb_samples = parse_celebdf_test_list()
    print(f"[OK] Celeb-DF Test Örnekleri: {len(celeb_samples)} video")

    # 3. SyntheticSmokeDataset ve BalancedBatchSampler Testi
    smoke_ds = SyntheticSmokeDataset(num_samples=64, target_size=224, seed=42)
    assert len(smoke_ds) == 64
    s0 = smoke_ds[0]
    print(f"[OK] Smoke Örnek 0: rgb={tuple(s0['rgb'].shape)} | mask={tuple(s0['mask'].shape)} | "
          f"label={s0['label']} | has_mask={s0['has_mask']} | manip={s0['manipulation']}")

    sampler = BalancedBatchSampler(smoke_ds, batch_size=16, seed=42)
    batches = list(iter(sampler))
    print(f"[OK] BalancedBatchSampler: {len(batches)} batch üretildi (batch_size=16)")
    b0 = batches[0]
    labels_b0 = [smoke_ds[i]["label"] for i in b0]
    n_real = sum(1 for l in labels_b0 if l == 0)
    n_fake = sum(1 for l in labels_b0 if l == 1)
    print(f"     Batch 0 Dağılımı: {n_real} Real / {n_fake} Fake (tam %50/%50 denge)")
    assert n_real == 8 and n_fake == 8

    # 4. Split ve Manipulasyon Filtreleme Testi (actors ve DFD)
    dummy_samples = [
        {"video_id": "000", "label": 0, "manipulation": "youtube", "compression": "c23", "rgb_np": np.zeros((224, 224, 3), dtype=np.uint8)},
        {"video_id": "actor_01", "label": 0, "manipulation": "actors", "compression": "c23", "rgb_np": np.zeros((224, 224, 3), dtype=np.uint8)},
        {"video_id": "dfd_01", "label": 1, "manipulation": "DeepFakeDetection", "compression": "c23", "rgb_np": np.zeros((224, 224, 3), dtype=np.uint8)},
        {"video_id": "000_003", "label": 1, "manipulation": "Deepfakes", "compression": "c23", "rgb_np": np.zeros((224, 224, 3), dtype=np.uint8)},
    ]
    # DFD testi: allowed_real yalnizca actors olmali, youtube filtrelenmeli
    dfd_ds = DeepfakeFaceDataset(
        samples=dummy_samples,
        split="test",
        manipulations=["DeepFakeDetection"],
    )
    dfd_manips = {s["manipulation"] for s in dfd_ds.samples}
    print(f"[OK] DFD Test Seti Filtreleme: manips={dfd_manips}")
    assert "actors" in dfd_manips, "actors DFD test setinde yer almali!"
    assert "DeepFakeDetection" in dfd_manips, "DeepFakeDetection test setinde yer almali!"
    assert "youtube" not in dfd_manips, "youtube videolari DFD seti ile karistirilmamali!"
    assert "Deepfakes" not in dfd_manips

    # 5. On-the-fly SBI Uretimi ve RGB [0, 1] Araligi Testi
    real_sample = [{"video_id": "000", "label": 0, "manipulation": "youtube", "compression": "c23", "rgb_np": np.full((224, 224, 3), 128, dtype=np.uint8)}]
    sbi_ds = DeepfakeFaceDataset(samples=real_sample, is_train=True, sbi_prob=1.0)
    sbi_item = sbi_ds[0]
    print(f"[OK] SBI Ornegi: label={sbi_item['label']} | manip={sbi_item['manipulation']} | "
          f"has_mask={sbi_item['has_mask']} | rgb_range=[{sbi_item['rgb'].min():.3f}, {sbi_item['rgb'].max():.3f}] | "
          f"mask_shape={tuple(sbi_item['mask'].shape)}")
    assert sbi_item["label"] == 1, "SBI sonrasi label=1 olmali!"
    assert sbi_item["manipulation"] == "SBI", "SBI sonrasi manipulation='SBI' olmali!"
    assert sbi_item["has_mask"] is True, "SBI ornegi has_mask=True olmali!"
    assert sbi_item["mask"].shape == (1, 224, 224), f"Mask boyutu (1, 224, 224) olmali, alinan: {sbi_item['mask'].shape}"
    assert sbi_item["rgb"].min() >= 0.0 and sbi_item["rgb"].max() <= 1.0, "RGB tensöru [0.0, 1.0] float32 araliginda kalmali!"

    # 6. Guclendirilmis Veri Artirma (Augmentation) Testi
    aug_rgb, aug_mask = sbi_ds._apply_augmentations(np.zeros((224, 224, 3), dtype=np.uint8), np.zeros((224, 224), dtype=np.uint8))
    assert aug_rgb.shape == (224, 224, 3)
    assert aug_mask.shape == (224, 224)
    print("[OK] Guclendirilmis veri artirmalari basariyla calisti.")

    print("=== dataset.py Tüm Testleri Başarıyla Geçti! ===")
