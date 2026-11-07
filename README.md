# PHD-Net

PHD-Net, video manipülasyonu tespiti için faz uyumu ve frekans özelliklerini inceleyen bir araştırma prototipidir. Depo; yüz karelerinin çıkarılması, faz/frekans özelliklerinin hesaplanması, PHD-Net mimarisi ve Meso-4 karşılaştırma modelini içerir.

Proje, Linux ve Windows üzerinde aynı Podman konteyner imajıyla çalışacak şekilde düzenlenmiştir. Ham veri setleri ve üretilen ara dosyalar depoya eklenmez.

## Gereksinimler

- Podman
- Windows için [Podman Desktop ve WSL 2](https://podman-desktop.io/docs/installation/windows-install)
- İlk imaj oluşturma sırasında internet erişimi

PyTorch ve diğer Python bağımlılıkları `Containerfile` ile oluşturulan imajda kurulur. Geliştirme bilgisayarına ayrıca Python paketleri kurmak gerekmez. Komut dosyaları varsayılan olarak CPU modunda çalışır.

## Kurulum

Depoyu klonlayın ve proje dizinine geçin. İmaj ilk çalıştırmada otomatik oluşturulur. İsterseniz önceden de oluşturabilirsiniz:

```sh
podman build -t localhost/phd-net-dev:latest -f Containerfile .
```

### Windows (PowerShell)

Podman Desktop'ı WSL 2 sağlayıcısıyla kurup Podman makinesinin çalıştığını doğrulayın. Ardından proje dizininde:

```powershell
& "$PWD/run.ps1" src/phase_congruency.py
```

PowerShell betiklerinin çalıştırılması engelleniyorsa yalnızca geçerli terminal oturumu için izin verin:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Bağımlılıkları veya `Containerfile`'ı değiştirdikten sonra imajı yeniden oluşturmak için:

```powershell
& "$PWD/run.ps1" -Build src/phase_congruency.py
```

### Linux (Bash)

```bash
./run.sh src/phase_congruency.py
```

İmajı yeniden oluşturmak için:

```bash
./run.sh --build src/phase_congruency.py
```

Her iki komut da aynı imajı kullanır. `run.sh` Linux'ta SELinux etiketlemesini uygular; PowerShell başlatıcısı Windows ana makine yollarını konteynere bağlar.

## Veri setleri

Veri setleri lisans ve boyut nedenleriyle depoda bulunmaz. Varsayılan konum depo kökündeki `datasets/` dizinidir. Başka bir konum kullanmak için `PHD_NET_DATASETS` değişkenini ayarlayın; her bilgisayarda bu değişken yerel veri dizinine işaret edebilir.

PowerShell:

```powershell
$env:PHD_NET_DATASETS = Join-Path $HOME 'datasets'
& "$PWD/run.ps1" src/face_extractor.py
```

Bash:

```bash
export PHD_NET_DATASETS="$HOME/datasets"
./run.sh src/face_extractor.py
```

Konteyner içinde yapılandırılan veri dizini `/datasets` olarak görünür. Yüz çıkarıcıya girdi ve çıktı konumlarını şu şekilde verebilirsiniz:

```text
src/face_extractor.py --data_dir /datasets/celebdf/Celeb-real --output_dir data/processed_faces/celeb_real
```

Ham veri, kırpılmış kareler ve ara çıktılar `.gitignore` kapsamındadır.

## Kullanım

Modüller kendi doğrulama/örnek çalıştırma kodlarını içerir. Başlatıcıya modül yolunu ve gerekiyorsa modül argümanlarını verin.

| Modül | İşlev |
|---|---|
| `src/face_extractor.py` | YuNet ile yüz tespiti ve kırpılmış kare çıkarımı |
| `src/phase_congruency.py` | Monokrom ve kuaterniyon faz uyumu, `PhaseAttentionGate` |
| `src/learnable_freq_gate.py` | Blok DCT, harmonik enerji oranı ve öğrenilebilir frekans kapısı |
| `src/phasefreq_net.py` | PHD-Net mimarisi ve benchmark doğrulaması |
| `src/mesonet.py` | Meso-4 karşılaştırma modeli |
| `src/frequency_extractor.py` | FFT ve DCT tabanlı frekans özellikleri |

Örnekler:

```sh
# Faz uyumu bileşenlerini doğrula
./run.sh src/phase_congruency.py

# Frekans kapısını doğrula
./run.sh src/learnable_freq_gate.py

# Belirli bir video dizininden yüz kareleri çıkar
./run.sh src/face_extractor.py \
  --data_dir /datasets/celebdf/Celeb-real \
  --output_dir data/processed_faces/celeb_real \
  --frame_interval 30 \
  --max_videos 5
```

PowerShell'de aynı modülleri `run.ps1` ile çalıştırın; örneğin `& "$PWD/run.ps1" src/phase_congruency.py`.

Modül listesinde yer almayan `src/run_d2_d3_experiments.py`, aşağıdaki CRF kliplerini ve ilgili veri setlerini gerektiren D2/D3 deneylerini çalıştırır.

## D2/D3 CRF deneyleri

CRF klip hazırlama akışı bu revizyonda taşınabilir hâle getirilmemiştir. `reencode_crf_clips.sh`, Bash ve `localhost/repo-lab/ister3-preprocess:v1` adlı önceden oluşturulmuş yerel Podman imajını kullanır. Bu adım için Linux veya uygun Bash/Podman ortamı gerekir. Klipler hazırlandıktan sonra D2/D3 Python modülü ortak proje imajında çalıştırılabilir:

```bash
./reencode_crf_clips.sh
./run.sh src/run_d2_d3_experiments.py
```

## Model ve deney kapsamı

- YuNet modeli `models/face_detection_yunet_2023mar.onnx` dosyasından yüklenir ve SHA256 özeti doğrulanır.
- PHD-Net ve Meso-4 mimari/benchmark doğrulamaları içerir; bu depodaki sonuçlar modellerin eğitildiği anlamına gelmez.
- Veri setinde manipülasyon maskeleri bulunmadığından PHD-Net'in yerelleştirme başlığı ve Dice kaybı eğitilip değerlendirilmemiştir.
- D2/D3 deneyleri gereken veri setlerini ve CRF kliplerini önceden bekler.

## Dizin yapısı

### Veri seti yerleşimi

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

Bu dizin yapısı veri setlerinin beklenen yerleşimini gösterir; veri dosyaları depoya eklenmez.

### Depo

```text
.
├── Containerfile
├── README.md
├── requirements.txt
├── run.sh
├── run.ps1
├── reencode_crf_clips.sh
├── models/
├── src/
├── datasets/                 # Yerel veri setleri; Git'e eklenmez
├── data/                     # İşlenmiş kareler; Git'e eklenmez
└── outputs/
    ├── logs/
    └── figures/
```
