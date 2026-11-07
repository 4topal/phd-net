#!/usr/bin/env bash
# Sıkıştırma ve görsel doğrulama analizleri için salt-okunur datasets/ videolarından geçici
# libx264 CRF {0, 18, 23, 32, 40} kopyalarını outputs/tmp_crf/ altına üretir.
# NOT: datasets/ dizinine kesinlikle yazılmaz (:ro mount).
set -euo pipefail

KOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${KOD_DIR}/.." && pwd)"
if [[ -n "${PHD_NET_DATASETS:-}" ]]; then
    DATASETS_DIR="${PHD_NET_DATASETS}"
elif [[ -d "${KOD_DIR}/datasets" ]]; then
    DATASETS_DIR="${KOD_DIR}/datasets"
else
    DATASETS_DIR="${REPO_DIR}/datasets"
fi
OUT_TMP="${KOD_DIR}/outputs/tmp_crf"
FFMPEG_IMAGE="localhost/repo-lab/ister3-preprocess:v1"

if [[ ! -d "${DATASETS_DIR}" ]]; then
    echo "Veri seti dizini bulunamadı: ${DATASETS_DIR}. PHD_NET_DATASETS ayarlayın." >&2
    exit 2
fi

mkdir -p "${OUT_TMP}"

podman run --rm \
    --user 0:0 \
    --entrypoint bash \
    -v "${DATASETS_DIR}:/datasets:ro" \
    -v "${OUT_TMP}:/out:z" \
    "${FFMPEG_IMAGE}" \
    -c '
set -euo pipefail
CRFS=(0 18 23 32 40)

FF_ORIG=("000.mp4" "001.mp4" "002.mp4" "003.mp4" "004.mp4")
FF_FAKE=("000_003.mp4" "001_870.mp4" "002_006.mp4" "003_000.mp4" "004_982.mp4")

mkdir -p /out/ffpp_original /out/ffpp_deepfakes /out/ffpp_face2face /out/celebdf

for v in "${FF_ORIG[@]}"; do
    stem="${v%.mp4}"
    for crf in "${CRFS[@]}"; do
        ffmpeg -nostdin -hide_banner -loglevel error -y \
            -i "/datasets/ffc32/FaceForensics++_C23/original/${v}" \
            -t 4 -an -c:v libx264 -preset fast -crf "${crf}" -pix_fmt yuv420p \
            "/out/ffpp_original/${stem}_crf${crf}.mp4"
    done
done

for v in "${FF_FAKE[@]}"; do
    stem="${v%.mp4}"
    for crf in "${CRFS[@]}"; do
        ffmpeg -nostdin -hide_banner -loglevel error -y \
            -i "/datasets/ffc32/FaceForensics++_C23/Deepfakes/${v}" \
            -t 4 -an -c:v libx264 -preset fast -crf "${crf}" -pix_fmt yuv420p \
            "/out/ffpp_deepfakes/${stem}_crf${crf}.mp4"
        ffmpeg -nostdin -hide_banner -loglevel error -y \
            -i "/datasets/ffc32/FaceForensics++_C23/Face2Face/${v}" \
            -t 4 -an -c:v libx264 -preset fast -crf "${crf}" -pix_fmt yuv420p \
            "/out/ffpp_face2face/${stem}_crf${crf}.mp4"
    done
done

for crf in "${CRFS[@]}"; do
    ffmpeg -nostdin -hide_banner -loglevel error -y \
        -i "/datasets/celebdf/Celeb-real/id0_0000.mp4" \
        -t 4 -an -c:v libx264 -preset fast -crf "${crf}" -pix_fmt yuv420p \
        "/out/celebdf/id0_0000_crf${crf}.mp4"
    ffmpeg -nostdin -hide_banner -loglevel error -y \
        -i "/datasets/celebdf/Celeb-synthesis/id0_id16_0000.mp4" \
        -t 4 -an -c:v libx264 -preset fast -crf "${crf}" -pix_fmt yuv420p \
        "/out/celebdf/id0_id16_0000_crf${crf}.mp4"
done
echo "[OK] Tüm CRF {0,18,23,32,40} geçici test klipleri üretildi."
'
