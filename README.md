# PHD-Net (Phase-Harmonic Discrepancy Network)

PHD-Net, video yüz manipülasyonu tespiti için faz uyumu (Phase Congruency) ve harmonik frekans özelliklerini inceleyen derin öğrenme araştırma prototipidir. Depo; yüz tespiti ve hizalama, kuaterniyon faz uyumu, öğrenilebilir frekans kapılama mekanizması, PHD-Net iki dallı mimarisi, Meso-4 kıyaslama modeli ve sıkıştırma dayanıklılığı deneylerini içerir.

Proje, Linux ve Windows üzerinde aynı Podman konteyner imajıyla çalışacak şekilde platformlar arası mimaride tasarlanmıştır.

---

## Mimari Genel Bakış

PHD-Net, iki tamamlayıcı bilgi akışını birleştiren hafif (*lightweight*) bir mimariye sahiptir:

1. **Uzamsal-Faz Kolu (Spatial-Phase Branch):**
   - RGB yüz kırpmaları üzerinden çok ölçekli Log-Gabor kuadratür filtre bankası ile **Kuaterniyon Faz Uyumu (QPC)** haritaları üretilir.
   - Akromatik eksen ve krominans alt-uzayları analiz edilerek **Kromatik Faz Dispersiyonu ($\Delta\Phi_{\text{chrom}}$)** hesaplanır.
   - `PhaseAttentionGate` katmanı, uzamsal özellikleri faz uyumu haritaları ile dinamik olarak modüle eder.
   - EdgeNeXt ve RepVGG esinli hafif çok-ölçekli derinlemesine ayrılabilir evrişim blokları ile öznitelikler çıkarılır.

2. **Harmonik Frekans Kolu (Harmonic Branch):**
   - 8×8 blok 2B DCT-II dönüşümü uygulanır.
   - Düşük, orta ve yüksek frekans alt-bantları üzerinden **Harmonik Enerji Oranı (HER)** hesaplanır.
   - `LearnableFreqGate` (LFG), frekans katsayılarını öğrenilebilir kapılar ve HER haritası ile ağırlıklandırır.

3. **Dinamik Frekans-Uzamsal Kalibrasyon (DFSC):**
   - Uzamsal ve harmonik kollar arasında çift yönlü çapraz kapılama (*cross-gating*) uygulanarak öznitelikler kalibre edilir.

4. **Çok Görevli Başlıklar:**
   - İkili sınıflandırma başlığı ($B \times 1$) ve birim hiperküre üzerinde L2-normalize kontrastif gömme ($B \times 128, \mathcal{L}_{\text{CCM}}$).
   - Piksel düzeyinde adli lokalizasyon ısı haritası başlığı ($B \times 1 \times 224 \times 224$).

---

## Gereksinimler

- [Podman](https://podman.io/) (veya Windows için [Podman Desktop ve WSL 2](https://podman-desktop.io/docs/installation/windows-install))
- İlk konteyner imajı kurulumu sırasında internet erişimi

PyTorch ve gerekli Python kütüphaneleri `Containerfile` üzerinden otomatik olarak kurulur. Ana makinede Python bağımlılıkları kurmaya gerek yoktur.

---

## Kurulum ve Çalıştırma

Depoyu klonlayıp proje dizinine geçin:

```sh
git clone https://github.com/4topal/phd-net.git
cd phd-net
```

Konteyner imajı ilk çalıştırmada otomatik derlenir. İsteğe bağlı olarak önceden de oluşturulabilir:

```sh
podman build -t localhost/phd-net-dev:latest -f Containerfile .
```

### Windows (PowerShell)

Podman Desktop ve Podman makinesinin çalıştığını doğruladıktan sonra:

```powershell
& "$PWD/run.ps1" src/phase_congruency.py
```

GPU erişimi ile çalıştırmak için:

```powershell
& "$PWD/run.ps1" -Gpu src/phasefreq_net.py
```

İmajı yeniden derlemek için:

```powershell
& "$PWD/run.ps1" -Build src/phase_congruency.py
```

PowerShell betik çalıştırma engeliyle karşılaşılırsa:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Linux (Bash)

```bash
./run.sh src/phase_congruency.py
```

GPU erişimi ile çalıştırmak için:

```bash
./run.sh --gpu src/phasefreq_net.py
```

İmajı yeniden derlemek için:

```bash
./run.sh --build src/phase_congruency.py
```

---

## Modüller

Projedeki her modül kendi doğrulama ve test bloğunu içerir:

| Modül | İşlev |
|---|---|
| `src/face_extractor.py` | YuNet ONNX ile yüz tespiti ve %20 marjlı 224×224 kırpma |
| `src/phase_congruency.py` | Monokrom ve kuaterniyon faz uyumu (QPC), `PhaseAttentionGate` |
| `src/learnable_freq_gate.py` | 8×8 blok DCT, Harmonik Enerji Oranı (HER) ve `LearnableFreqGate` |
| `src/phasefreq_net.py` | PHD-Net iki dallı mimarisi, DFSC füzyonu ve benchmark modülü |
| `src/mesonet.py` | Meso-4 baseline karşılaştırma modeli ve benchmark modülü |
| `src/frequency_extractor.py` | Klasik 2B FFT (azimutal profil) ve DCT alt-bant öznitelik çıkarımı |
| `src/run_compression_experiments.py` | H.264 CRF sıkıştırma dayanıklılığı analizi ve görsel doğrulama paneli |

### Örnek Çalıştırma Komutları

```sh
# Faz uyumu bileşenlerini doğrula
./run.sh src/phase_congruency.py

# Öğrenilebilir frekans kapısını doğrula
./run.sh src/learnable_freq_gate.py

# PHD-Net model mimarisi ve çıkarım gecikmesini ölç
./run.sh --gpu src/phasefreq_net.py

# Video dizininden yüz kareleri çıkar
./run.sh src/face_extractor.py \
  --data_dir /datasets/celebdf/Celeb-real \
  --output_dir data/processed_faces/celeb_real \
  --frame_interval 30 \
  --max_videos 5
```

---

## Sıkıştırma Dayanıklılığı ve Görsel Doğrulama Deneyleri

Sıkıştırma dayanıklılığı testleri, FaceForensics++ C23 videolarının `libx264` ile farklı CRF seviyelerinde ({0, 18, 23, 32, 40}) yeniden kodlanarak HER metriklerinin incelenmesini sağlar:

```bash
# Geçici CRF kliplerini üret
./reencode_crf_clips.sh

# Sıkıştırma dayanıklılığı ve görsel doğrulama panelini çalıştır
./run.sh src/run_compression_experiments.py
```

Sonuçlar ve grafikler otomatik olarak `outputs/logs/` ve `outputs/figures/` dizinlerine kaydedilir.

---

## Veri Setleri Dizin Yapısı

Veri setleri depoya dahil edilmez. Varsayılan konum `datasets/` dizinidir; alternatif olarak `PHD_NET_DATASETS` ortam değişkeni kullanılabilir:

```text
datasets/
├── celebdf/
│   ├── Celeb-real/
│   └── Celeb-synthesis/
└── ffc32/
    └── FaceForensics++_C23/
        ├── original/
        ├── Deepfakes/
        └── Face2Face/
```

Konteyner içinde yapılandırılan veri dizini `/datasets` yoluna salt-okunur (`:ro`) bağlanır.

---

## Depo Yapısı

```text
.
├── Containerfile                         # Konteyner imaj tanımı (PyTorch + CUDA)
├── README.md                             # Proje dokümantasyonu
├── requirements.txt                      # Python bağımlılıkları
├── run.sh                                # Linux başlatıcı betik
├── run.ps1                               # Windows PowerShell başlatıcı betik
├── reencode_crf_clips.sh                 # CRF yeniden kodlama betiği
├── models/
│   └── face_detection_yunet_2023mar.onnx # YuNet yüz tespit modeli
├── src/
│   ├── face_extractor.py                 # Yüz tespit ve kırpma hattı
│   ├── frequency_extractor.py            # Klasik frekans öznitelik çıkarımı
│   ├── learnable_freq_gate.py            # Blok DCT, HER ve frekans kapısı
│   ├── mesonet.py                        # Meso-4 baseline mimarisi
│   ├── phase_congruency.py               # Kuaterniyon faz uyumu (QPC)
│   ├── phasefreq_net.py                  # PHD-Net ana mimarisi ve kayıplar
│   ├── project_paths.py                  # Ortak yol yapılandırması
│   └── run_compression_experiments.py    # Sıkıştırma ve görsel analizler
└── outputs/
    ├── figures/                          # Üretilen grafik ve paneller
    └── logs/                             # Kıyaslama ve deney kayıtları
```
