FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    grpcio==1.59.3 \
    grpcio-tools==1.59.3 \
    protobuf==4.25.1 \
    numpy==1.24.3 \
    onnx==1.15.0 \
    onnx2torch \
    torchinfo \
    einops

RUN pip3 install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl && \
    pip3 install "numpy<2"

WORKDIR /app
VOLUME ["/app/models"]
COPY requirements.txt ./