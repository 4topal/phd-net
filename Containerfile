FROM docker.io/pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/conda/lib

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["python3", "--version"]
