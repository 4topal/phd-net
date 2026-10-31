# PHD-Net Kod Altyapısı (`kod/`) — Faz 2 (Altyapı ve Doğrulama)

**Ders:** Kocaeli Üniversitesi BLM403 Araştırma Problemleri  
**Öğrenci:** Ahmet Said TOPAL (230202053)  
**Kapsam:** Faz 2 mimari kurulum, ön işleme ve deneysel doğrulama altyapısı (**Eğitim Faz 3 / Hafta 7–8 kapsamındadır**).

---

## 1. Ortam ve Çalıştırma Mimarisi

Host sistemde **Python 3.14** bulunduğundan PyPI üzerinde resmi PyTorch tekerleği (wheel) mevcut değildir. Tüm kodlar yerel Podman imajı (`docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`) içinde NVIDIA CDI (`--device nvidia.com/gpu=all`) ve SELinux Enforcing uyumlu bağlamlarla (`kod/` için `:z`, `datasets/` için salt okunur `:ro`) çalıştırılır.

- **Ağ veya ek `pip install` gerekmez:** `torch`, `torchvision`, `opencv-python-headless`, `scipy`, `matplotlib`, `numpy`, `pillow`, `tqdm` baz imajda hazır kuruludur (`requirements.txt` sabit sürümleri referans olarak belgeler).
- **YuNet Yüz Dedektörü:** `models/face_detection_yunet_2023mar.onnx` dosyası çalışma anında SHA256 (`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`) doğrulamasından geçirilerek yüklenir.

---

## 2. Dizin Yapısı

```text
kod/
├── run.sh                      # Podman GPU sarmalayıcısı: ./run.sh src/<modul>.py [args]
├── reencode_crf_clips.sh       # D2/D3 için geçici libx264 CRF {0,18,23,32,40} klip üretici
├── requirements.txt            # Konteyner içi sabit paket sürümleri
├── README.md                   # Kurulum, kullanım ve modül durumları
├── FAZ2_RAPORU.md              # Faz 2 doğrulama sonuçları ve mühendislik raporu
├── models/                     # YuNet ONNX modeli (.gitignore kapsamında)
├── src/
│   ├── face_extractor.py       # YuNet yüz tespiti, %20 marj, 224x224 kırpma + kayıp kare logu
│   ├── phase_congruency.py     # Kovesi monokrom PC + Kuaterniyon Faz Uyumu (QPC) + PhaseAttentionGate
│   ├── learnable_freq_gate.py  # 8x8 Blok DCT + parametrik bantlar + HER + LearnableFreqGate (LFG)
│   ├── phasefreq_net.py        # İki dallı PHD-Net + BCE/Dice/L_CCM + ayrı ön işleme & GPU benchmark
│   ├── mesonet.py              # Meso-4 (Afchar et al., 2018) baseline + GPU benchmark
│   ├── frequency_extractor.py  # Klasik azimutal 2B FFT + 8x8 DCT ablasyon öznitelik çıkarıcı
│   └── run_d2_d3_experiments.py# Zorunlu D2 (HER sıkıştırma) ve D3 (görsel panel) deney betiği
├── data/processed_faces/       # Kırpılan yüz kareleri (.gitignore kapsamında)
└── outputs/
    ├── logs/                   # JSON benchmark ve deney kayıtları
    └── figures/                # Üretilen doğrulama grafikleri ve görsel paneller
```

---

## 3. Hızlı Çalıştırma Komutları

`kod/` dizini içindeyken tüm modüller kendi kendini test eden (`__main__`) bloklarıyla çalıştırılabilir:

```bash
cd /home/a/arastirma_problemleri/kod

# 1. Yüz çıkarıcı kendi kendini test (2 Celeb-real + 2 Celeb-synthesis videosu)
./run.sh src/face_extractor.py

# Özel CLI parametreleriyle yüz çıkarımı:
./run.sh src/face_extractor.py \
    --data_dir ../datasets/celebdf/Celeb-real \
    --output_dir data/processed_faces/celeb_real \
    --frame_interval 30 \
    --max_videos 5

# 2. Monokrom PC (Kovesi) + Kuaterniyon Faz Uyumu (QPC) + PhaseAttentionGate testi
./run.sh src/phase_congruency.py

# 3. 8x8 Blok DCT + HER + Öğrenilebilir Frekans Kapısı (LFG) testi
./run.sh src/learnable_freq_gate.py

# 4. PHD-Net mimari doğrulaması ve D1 GPU Benchmark (outputs/logs/benchmark.json)
./run.sh src/phasefreq_net.py

# 5. Meso-4 baseline doğrulaması ve D1 GPU Benchmark (outputs/logs/benchmark.json)
./run.sh src/mesonet.py

# 6. Klasik FFT + DCT ablasyon öznitelik çıkarıcı testi
./run.sh src/frequency_extractor.py

# 7. D2 (HER CRF sıkıştırma testi) ve D3 (Görsel panel) deneyleri
./reencode_crf_clips.sh
./run.sh src/run_d2_d3_experiments.py
```

---

## 4. Modül Durum Özeti

| Modül | Durum | Açıklama ve Önemli Notlar |
|---|---|---|
| `src/face_extractor.py` | **Çalışıyor** | YuNet ONNX (SHA256 doğrulamalı), %20 marj, 224×224 PNG kırpma, yüz bulunamayan kare sayımı/logu. |
| `src/phase_congruency.py` | **Çalışıyor** | Kovesi (1999) monokrom PC referansı + Ell & Sangwine (2007) tabanlı özgün 3-kanallı QPC + `PhaseAttentionGate`. |
| `src/learnable_freq_gate.py` | **Çalışıyor** | Ortonormal 8×8 DCT/IDCT (`<2e-6` geri-çatım hatası), parametrik alt-bantlar, HER ve `LearnableFreqGate`. |
| `src/phasefreq_net.py` | **Kısmen (Eğitimsiz / Maskesiz)** | İki dallı PHD-Net (3.76M param), BCE + `L_CCM` + `DiceLoss` ve D1 benchmark tam çalışıyor; **eğitim yapılmadı** ve veri setinde manipülasyon maskesi olmadığından **`localization_head` / `DiceLoss` eğitilmemiş ve değerlendirilmemiştir**. |
| `src/mesonet.py` | **Çalışıyor (Eğitimsiz)** | Afchar et al. (2018) Meso-4 uygulaması ve D1 GPU benchmark tam çalışıyor (Faz 2 gereği eğitilmemiştir). |
| `src/frequency_extractor.py` | **Çalışıyor** | Durall et al. (2020) azimutal 2B FFT profili + Frank et al. (2020) blok DCT istatistikleri. |
