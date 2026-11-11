#!/usr/bin/env bash
# Ortak Podman imajında CPU (varsayılan) veya isteğe bağlı NVIDIA GPU çalıştırıcısı.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${PHD_NET_IMAGE:-localhost/phd-net-dev:latest}"
USE_GPU=0
BUILD_IMAGE=0

while (($#)); do
    case "$1" in
        --gpu)
            USE_GPU=1
            shift
            ;;
        --build)
            BUILD_IMAGE=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

if (($# < 1)); then
    echo "Kullanım: ./run.sh [--build] [--gpu] src/<modul>.py [argümanlar...]" >&2
    exit 2
fi

if ! command -v podman >/dev/null 2>&1; then
    echo "Podman bulunamadı. Önce Podman'ı kurup çalışır duruma getirin." >&2
    exit 127
fi

if [[ -n "${PHD_NET_DATASETS:-}" ]]; then
    DATASETS_HOST="${PHD_NET_DATASETS}"
    if [[ "${DATASETS_HOST}" == "~/"* ]]; then
        DATASETS_HOST="${HOME}/${DATASETS_HOST#\~/}"
    fi
elif [[ -d "${PROJECT_DIR}/datasets" ]]; then
    DATASETS_HOST="${PROJECT_DIR}/datasets"
elif [[ -d "${PROJECT_DIR}/../datasets" ]]; then
    DATASETS_HOST="${PROJECT_DIR}/../datasets"
else
    DATASETS_HOST="${PROJECT_DIR}/datasets"
fi

if [[ -n "${PHD_NET_DATASETS:-}" && ! -d "${DATASETS_HOST}" ]]; then
    echo "PHD_NET_DATASETS dizini bulunamadı: ${DATASETS_HOST}" >&2
    exit 2
fi
if [[ -d "${DATASETS_HOST}" ]]; then
    DATASETS_HOST="$(cd "${DATASETS_HOST}" && pwd)"
fi

if ((BUILD_IMAGE)) || ! podman image exists "${IMAGE}"; then
    podman build --file "${PROJECT_DIR}/Containerfile" --tag "${IMAGE}" "${PROJECT_DIR}"
fi

MOUNT_ARGS=(--volume "${PROJECT_DIR}:/workspace:z")
DATASETS_CONTAINER="/workspace/datasets"
if [[ -d "${DATASETS_HOST}" ]]; then
    MOUNT_ARGS+=(--volume "${DATASETS_HOST}:/datasets:ro,z")
    DATASETS_CONTAINER="/datasets"
fi

CELEBDF_HOST=""
if [[ -n "${PHD_NET_CELEBDF:-}" ]]; then
    CELEBDF_HOST="${PHD_NET_CELEBDF}"
elif [[ ! -d "${DATASETS_HOST}/celebdf_v2/Celeb-real" ]]; then
    if [[ -d "D:/celeb-df_v2/Celeb-real" ]]; then
        CELEBDF_HOST="D:/celeb-df_v2"
    elif [[ -d "/d/celeb-df_v2/Celeb-real" ]]; then
        CELEBDF_HOST="/d/celeb-df_v2"
    fi
fi

EXTRA_ENV_ARGS=()
if [[ -n "${CELEBDF_HOST}" && -d "${CELEBDF_HOST}" ]]; then
    CELEBDF_HOST="$(cd "${CELEBDF_HOST}" && pwd)"
    MOUNT_ARGS+=(--volume "${CELEBDF_HOST}:/datasets_celebdf:ro,z")
    EXTRA_ENV_ARGS+=(--env "PHD_NET_CELEBDF=/datasets_celebdf")
fi

GPU_ARGS=()
if ((USE_GPU)); then
    GPU_ARGS+=(--device nvidia.com/gpu=all)
fi

exec podman run --rm \
    --shm-size 4g \
    "${GPU_ARGS[@]}" \
    "${MOUNT_ARGS[@]}" \
    --workdir /workspace \
    --env PYTHONPATH=/workspace/src:/workspace \
    --env "PHD_NET_DATASETS=${DATASETS_CONTAINER}" \
    "${EXTRA_ENV_ARGS[@]}" \
    "${IMAGE}" \
    python3 "$@"
