#!/usr/bin/env bash
# Podman GPU çalıştırma sarmalayıcısı (Fedora + SELinux Enforcing + NVIDIA CDI)
# Kullanım: ./run.sh src/<modul>.py [argümanlar...]
set -euo pipefail

KOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${KOD_DIR}/.." && pwd)"
DATASETS_DIR="${REPO_DIR}/datasets"
IMAGE="docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

if [[ $# -lt 1 ]]; then
    echo "Kullanım: ./run.sh src/<modul>.py [argümanlar...]" >&2
    exit 1
fi

MOUNT_ARGS=(
    -v "${KOD_DIR}:${KOD_DIR}:z"
)

if [[ -d "${DATASETS_DIR}" ]]; then
    MOUNT_ARGS+=(-v "${DATASETS_DIR}:${DATASETS_DIR}:ro")
fi

exec podman run --rm \
    --device nvidia.com/gpu=all \
    "${MOUNT_ARGS[@]}" \
    -w "${KOD_DIR}" \
    -e "PYTHONPATH=${KOD_DIR}/src:${KOD_DIR}" \
    -e "MPLCONFIGDIR=/tmp/matplotlib" \
    "${IMAGE}" \
    python3 "$@"
