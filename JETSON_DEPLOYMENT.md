# Jetson Orin Nano Deployment Guide

This guide explains how to deploy the single server on a Jetson Orin Nano device using Docker.

## Prerequisites

### Hardware Requirements
- NVIDIA Jetson Orin Nano (4GB or 8GB)
- At least 8GB of free storage space
- Network connectivity

### Software Requirements
- JetPack 5.1 or later (includes Docker)
- Docker and docker-compose installed
- ONNX model files for your specific model

## Quick Start

1. **Clone or copy your project to the Jetson device**

2. **Add your ONNX model files to the `models/` directory:**
   ```bash
   mkdir -p models
   # Copy your ONNX files here, for example:
   # - EENetB0_1_FC_encoder1.onnx
   # - EENetB0_1_FC_classifier1.onnx
   # - EENetB0_1_FC_single.onnx
   ```

3. **Run the deployment script:**
   ```bash
   ./scripts/deploy_jetson.sh
   ```

4. **Verify the server is running:**
   ```bash
   docker ps
   docker-compose -f docker-compose.jetson.yml logs -f
   ```

## Manual Deployment

If you prefer to deploy manually:

### 1. Build the Docker Image
```bash
docker-compose -f docker-compose.jetson.yml build
```

### 2. Run the Container
```bash
docker-compose -f docker-compose.jetson.yml up -d
```

### 3. Check Status
```bash
docker-compose -f docker-compose.jetson.yml ps
```

## Configuration

### Environment Variables
You can modify the following environment variables in `docker-compose.jetson.yml`:

- `CUDA_VISIBLE_DEVICES`: Set to `0` for the first GPU, `1` for second, etc.
- `PYTHONUNBUFFERED`: Set to `1` for immediate log output

### Server Parameters
The server accepts the following command-line arguments:

- `-m, --model-name`: Model name (default: "EENetB0_1_FC")
- `-n, --encoder_num`: Encoder number (default: 1)
- `-p, --port`: Port number (default: 8180)
- `-s, --head-server`: Head server URL (default: "localhost:8180")
- `--split`: Split configuration (default: "1-5")

### Example Custom Configuration
```bash
docker-compose -f docker-compose.jetson.yml run --rm single-server \
  -m "MyModel" -n 2 -p 9000 -s "head-server:8180"
```

## Model Files

The server expects ONNX model files in the `models/{model_name}/` directory with the following naming convention:

- `encoder{encoder_num}.onnx` - Encoder model
- `classifier{classifier_num}.onnx` - Classifier model
- `single.onnx` - Full single model
- `original.onnx` - Original model (optional)
- `split_{split}.onnx` - Split model (optional)

## Monitoring and Logs

### View Real-time Logs
```bash
docker-compose -f docker-compose.jetson.yml logs -f
```

### Check Container Status
```bash
docker ps | grep jetson-single-server
```

### Monitor GPU Usage
```bash
nvidia-smi
```

### Check ONNX Runtime Providers
```bash
docker exec jetson-single-server python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## Troubleshooting

### Common Issues

1. **CUDA not available**
   - Ensure JetPack is properly installed
   - Check if CUDA drivers are loaded: `nvidia-smi`
   - Verify ONNX Runtime CUDA support is available

2. **Port already in use**
   - Change the port in docker-compose.jetson.yml
   - Or stop the existing service: `docker-compose -f docker-compose.jetson.yml down`

3. **Model files not found**
   - Ensure ONNX files are in the `models/` directory
   - Check file permissions
   - Verify the model naming convention

4. **Memory issues**
   - Jetson Orin Nano has limited RAM (4GB or 8GB)
   - Consider using smaller models or batch sizes
   - Monitor memory usage: `free -h`

5. **Performance issues**
   - Ensure CUDA execution provider is being used
   - Check GPU utilization: `nvidia-smi`
   - Consider model optimization for Jetson

### Debug Commands

```bash
# Enter the container for debugging
docker exec -it jetson-single-server bash

# Check available execution providers
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Test CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Check model files
ls -la /app/models/
```

## Performance Optimization

### For Jetson Orin Nano

1. **Use TensorRT optimization** (if available)
2. **Enable mixed precision** for better performance
3. **Optimize batch sizes** for your specific model
4. **Monitor thermal throttling** - Jetson devices can throttle under heavy load

### Memory Optimization

1. **Use smaller model variants** when possible
2. **Enable model quantization** (INT8/FP16)
3. **Limit concurrent requests** based on available memory

## Stopping the Service

```bash
# Stop the container
docker-compose -f docker-compose.jetson.yml down

# Remove the container and image (optional)
docker-compose -f docker-compose.jetson.yml down --rmi all
```

## Support

For issues specific to Jetson deployment:
1. Check the NVIDIA Jetson forums
2. Verify your JetPack version compatibility
3. Ensure all CUDA dependencies are properly installed

## Notes

- The Dockerfile uses NVIDIA L4T base image (r36.2.0) which is optimized for Jetson
- ONNX Runtime version 1.16.3 is used for compatibility with Jetson
- The container runs with GPU access enabled via NVIDIA runtime
- All logs are unbuffered for immediate visibility 