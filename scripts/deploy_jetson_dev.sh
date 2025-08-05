#!/bin/bash

# Jetson Orin Nano Development Deployment Script
# This script builds and runs the single server container on Jetson for development

set -e

echo "=== Jetson Orin Nano Development Deployment ==="

# Check if we're on a Jetson device
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. This script is designed for Jetson devices."
    echo "Continuing anyway..."
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p models logs

# Check if models directory has ONNX files
if [ ! "$(ls -A models 2>/dev/null)" ]; then
    echo "Warning: models directory is empty. Please add your ONNX model files:"
    echo "  - EENetB0_1_FC_encoder1.onnx"
    echo "  - EENetB0_1_FC_classifier1.onnx"
    echo "  - EENetB0_1_FC_single.onnx"
    echo "  - etc."
    echo ""
    echo "You can mount them later or add them now and re-run this script."
fi

# Build the Docker image
echo "Building Docker image for development..."
docker-compose -f docker-compose.jetson.yml build

# Run the container
echo "Starting the development server container..."
docker-compose -f docker-compose.jetson.yml up -d

echo ""
echo "=== Development Deployment Complete ==="
echo "Container is running with name: jetson-single-server"
echo "Server is accessible on port 8180"
echo ""
echo "IMPORTANT: Your Python code is now mounted as a volume!"
echo "Changes to ./system/ files will be reflected immediately without rebuilding."
echo ""
echo "To view logs:"
echo "  docker-compose -f docker-compose.jetson.yml logs -f"
echo ""
echo "To stop the server:"
echo "  docker-compose -f docker-compose.jetson.yml down"
echo ""
echo "To restart the server:"
echo "  docker-compose -f docker-compose.jetson.yml restart"
echo ""
echo "To rebuild the image (only needed for dependency changes):"
echo "  docker-compose -f docker-compose.jetson.yml build --no-cache" 