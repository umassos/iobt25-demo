# Environment Variables Setup for IOBT25 Demo

This document explains how to use environment variables with Docker Compose for flexible configuration.

## Quick Start

1. **Copy the example file:**
   ```bash
   cp env.example .env
   ```

2. **Modify `.env` for your environment:**
   ```bash
   # Edit .env file with your specific values
   nano .env
   ```

3. **Use with docker-compose:**
   ```bash
   docker-compose -f docker-compose.s1.yml up
   ```

## Environment Variables Reference

### Server Configuration
- `HEAD_SERVER_PORT` - Head server port (default: 8185)
- `S1_SERVER_PORT` - S1 server port (default: 8180)
- `S2_SERVER_PORT` - S2 server port (default: 8181)
- `S1_SERVER_UPSTREAM_HOST` - Upstream host for S1 (default: 192.168.79.12:8185)

### Docker Configuration
- `DOCKERFILE_S1` - Dockerfile for S1 server (default: Dockerfile.s1)
- `DOCKERFILE_S2` - Dockerfile for S2 server (default: Dockerfile.s2)
- `DOCKERFILE_HEAD` - Dockerfile for head server (default: Dockerfile.head)

### GPU Configuration
- `CUDA_VISIBLE_DEVICES` - GPU device ID (default: 0)
- `GPU_COUNT` - Number of GPUs (default: 1)
- `DOCKER_RUNTIME` - Docker runtime (default: nvidia)

### Paths
- `SOURCE_CODE_PATH` - Source code directory (default: ./system)
- `MODELS_PATH` - Models directory (default: ./models)
- `LOGS_PATH` - Logs directory (default: ./logs)

## Environment-Specific Configurations

### For Jetson Devices
```bash
# .env
DOCKERFILE_S1=Dockerfile.s1
DOCKERFILE_S2=Dockerfile.s2
DOCKERFILE_HEAD=Dockerfile.head
DOCKER_RUNTIME=nvidia
```

### For A2 Datacenter GPU
```bash
# .env
DOCKERFILE_S1=Dockerfile.a2
DOCKERFILE_S2=Dockerfile.a2
DOCKERFILE_HEAD=Dockerfile.a2
DOCKER_RUNTIME=nvidia
```

### For Development
```bash
# .env
DEV_MODE=true
MOUNT_SOURCE_CODE=true
SOURCE_CODE_PATH=./system
```

### For Production
```bash
# .env
DEV_MODE=false
MOUNT_SOURCE_CODE=false
```

## Multiple Environment Files

You can create multiple environment files for different scenarios:

```bash
# Development
cp env.example .env.dev
docker-compose --env-file .env.dev -f docker-compose.s1.yml up

# Production
cp env.example .env.prod
docker-compose --env-file .env.prod -f docker-compose.s1.yml up

# A2 specific
cp env.example .env.a2
docker-compose --env-file .env.a2 -f docker-compose.s1.yml up
```

## Example Usage

### Basic Usage
```bash
# Uses default values from .env
docker-compose -f docker-compose.s1.yml up
```

### Override Specific Variables
```bash
# Override port for this run
S1_SERVER_PORT=9000 docker-compose -f docker-compose.s1.yml up
```

### Use Different Environment File
```bash
docker-compose --env-file .env.a2 -f docker-compose.s1.yml up
```

## Benefits

1. **Flexibility**: Easy to switch between different configurations
2. **Environment Isolation**: Different settings for dev/staging/prod
3. **Hardware Adaptation**: Easy to switch between Jetson and A2 configurations
4. **Team Collaboration**: Everyone can use their own .env file
5. **Version Control**: .env files can be excluded from git while keeping examples

## Security Notes

- **Never commit `.env` files** to version control
- **Use `.env.example`** for documentation
- **Set sensitive values** via environment variables or secrets
- **Validate** environment variables in your application

## Troubleshooting

### Variables Not Working?
1. Check if `.env` file exists in the same directory as docker-compose.yml
2. Verify variable names match exactly (case-sensitive)
3. Restart docker-compose after changing .env

### Default Values Not Applied?
1. Ensure your docker-compose.yml uses `${VAR:-default}` syntax
2. Check for syntax errors in .env file
3. Verify docker-compose version supports variable substitution

### Port Conflicts?
1. Check if ports are already in use
2. Modify port variables in .env
3. Use `netstat -tulpn | grep :PORT` to check usage
