# CLAUDE.md — IoBT 25 MEL Demo

## Project Overview

**Multi-level Ensemble Learning (MEL)** fault-tolerance demo for IoBT 2025. The system runs distributed ML inference across Jetson Orin edge nodes, demonstrating graceful failover from a single model → MEL ensemble → solo encoder as servers fail.

All commands must be run from the **project root** (`iobt25-demo/`). Scripts assume this as the working directory.

---

## Architecture

### Three-Tier Failover Chain

```
[Original Server (8183)]  -- fails -->  [MEL Ensemble: S1 (8180) + S2 (8181) → Head (8185)]  -- fails -->  [S1 solo (8180)]
```

1. **Original**: single monolithic model (`original.onnx`), runs on one node.
2. **MEL Ensemble**: two encoder servers (S1, S2) run independently, forward intermediate features to a shared head server that combines them.
3. **S1 Solo**: if both Original and MEL are down, S1 falls back to local classifier inference.

### gRPC Service Flow (MEL Ensemble)

```
Client
  └─ PredictForward → S1 (single_server.py, port 8180)
       └─ encoder1.onnx → enc1_output → HeadService.Predict (head_server.py, port 8185)
  └─ PredictForward → S2 (single_server.py, port 8181)
       └─ encoder2.onnx → enc2_output → HeadService.Predict (head_server.py, port 8185)

Head waits for both enc1_output and enc2_output (matched by request_id), then runs head.onnx.
```

---

## `system/` Folder — Primary Focus

### Servers

**[system/single_server.py](system/single_server.py)**
- gRPC server implementing `EncoderService`
- Modes (controlled by CLI flags):
  - **Encoder mode** (default): loads `encoder{N}.onnx` + `classifier{N}.onnx`, forwards to head via `PredictForward`
  - **Original mode** (`--original`): loads `models/original.onnx`, runs standalone inference via `PredictOriginal`
- Falls back to local classifier if head server call fails (`PredictForward` catches exceptions)
- Writes exit timestamp to `system/results/fail_original.log` on SIGINT
- Key methods: `Predict` (local encode+classify), `PredictForward` (encode → head → fallback), `PredictOriginal` (monolithic), `Heartbeat`

**[system/head_server.py](system/head_server.py)**
- gRPC server implementing `HeadService`
- Waits for **two requests with the same `request_id`** before running inference
- Stores first arrival in `self.requests` dict; on second arrival, runs `head.onnx` with both encoder outputs
- Tracks per-request timing: `enc_service_time`, `enc_network_time` for both S1 and S2
- Returns combined `service_time = enc1_service_time + enc2_service_time + head_service_time`

### Clients

**[system/metrics_client.py](system/metrics_client.py)** ← main experiment client
- Drives the full failover experiment loop (not Streamlit; CLI + asyncio)
- Failover state machine using two counters: `orig_fail` and `mel_fail`
  - `orig_fail < 1` → use Original server
  - `mel_fail < 1` → use MEL ensemble (concurrent `PredictForward` on S1 + S2)
  - else → S1 solo `Predict`
- Heartbeat threads per server (`original_heartbeat`, `s1_heartbeat`, `s2_heartbeat`, `s12_heartbeat`) run at `config.heartbeat_interval` ms
- Failure events written to log files: `system/results/fail_original.log`, `system/results/fail_s2.log`
- Results saved to `system/results/{experiment_id}/response_times.csv`
- CLI args:
  - `-d` / `--duration`: experiment duration in seconds
  - `-f` / `--failover`: enable heartbeat threads
  - `-w` / `--write-log`: write per-request timing logs
  - `-i` / `--experiment-id`: label for output directory
  - `-t2` / `--task2`: use `config.server_task2` instead of original address

**[system/streamlit_app.py](system/streamlit_app.py)** ← GUI monitoring client
- Streamlit UI showing live server status and inference results for all 4 servers
- Uses MQTT (`mqtt_source.py`) to receive real camera frames from ZED sensor
- Mirrors `metrics_client.py` failover logic but with visual display
- Run with: `streamlit run system/streamlit_app.py`

**[system/run_client.py](system/run_client.py)** ← benchmarking utility
- Simple async gRPC client for measuring round-trip time
- Supports all RPC methods: `Predict`, `PredictForward`, `PredictOriginal`, `PredictFull`, `PredictSplit`
- For `PredictForward`: sends concurrent requests to `--server1` and `--server2`
- Saves results to `system/rpc_results/{hostname}_client_{function}_{model}_results.csv`

### Model Loading

**[system/run_onnx_utils.py](system/run_onnx_utils.py)**
- All model loading functions, always using `CUDAExecutionProvider`
- Model paths follow pattern: `models/{model_name}/{component}.onnx`
- Functions: `load_encoder`, `load_classifier`, `load_combined_head`, `load_single`, `load_original`, `load_split`
- `load_torch_*` functions are stubs (no-ops; torch conversion path is commented out)

### Configuration

**[system/config.py](system/config.py)**
- **Edit this when changing deployment targets**
- Server addresses (three environments commented in):
  - Obelix cluster (current active): `obelix192-195`
  - Jetson Orin Node 10 DVPG: `192.168.76/75/79.12`
  - LASS testbed: `192.168.0.175`
  - Local: `localhost`
- `heartbeat_interval`: ms between heartbeat probes (default: 10ms)
- `requests`: default request count (default: 1000)
- `topic`: MQTT camera topic

### Protobuf / gRPC

**[system/inference.proto](system/inference.proto)**
- Two services: `EncoderService` (S1/S2/Original) and `HeadService` (head)
- Key fields in `PredictRequest`: `request_id`, `input` (raw float32 bytes), `shape`, `enc_service_time`, `enc_send_time`
- Key fields in `PredictResponse`: `output`, `shape`, `full_model`, `has_result`, `service_time`, `network_time`
- Regenerate with: `python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. system/inference.proto`

**[system/inference_pb2.py](system/inference_pb2.py)** and **[system/inference_pb2_grpc.py](system/inference_pb2_grpc.py)** — generated, do not edit manually.

### Utilities

**[system/labels.py](system/labels.py)** — ImageNet class labels dict  
**[system/tieredimgnet_labels.json](system/tieredimgnet_labels.json)** — TinyImageNet labels (used for `tin` dataset experiments)  
**[system/mqtt_source.py](system/mqtt_source.py)** — MQTT client wrapper for receiving compressed images from ZED cameras  
**[system/convert_onnx.py](system/convert_onnx.py)** — utilities for exporting PyTorch models to ONNX  
**[system/run_onnx_single.py](system/run_onnx_single.py)** / **[run_onnx_single_all.sh](system/run_onnx_single_all.sh)** — single-shot ONNX inference runner for benchmarking  
**[system/test_jetson_server.py](system/test_jetson_server.py)** — test script for validating server behavior on Jetson  
**[system/plotting.py](system/plotting.py)** / **[system/plotting-downsampling.py](system/plotting-downsampling.py)** — result analysis and plot generation from CSV outputs

### Output Directories

- `system/results/` — experiment output CSVs and failure logs
- `system/rpc_results/` — per-host gRPC benchmark CSVs

---

## Model Naming Convention

```
ensemble-{arch}-c{cut_point}-lr-{lr}-{dataset}
```

- `arch`: `effnet` (EfficientNet-B0), `vit` (ViT-B/16), `resnet`
- `cut_point`: encoder split layer (C1–C6); higher = more computation at encoder
- `lr`: learning rate used during training (e.g. `0.005`, `0.0001`)
- `dataset`: `cifar` (CIFAR-100), `tin` (TinyImageNet, 608 classes)

**Model files per directory** (under `models/{model_name}/`):
- `encoder1.onnx`, `encoder2.onnx` — encoder halves for S1 and S2
- `classifier1.onnx` — local classifier (used for fallback `Predict`)
- `head.onnx` — combined head (`enc1_output` + `enc2_output` → logits)
- `single.onnx` — optional, full single-model inference

---

## Port Assignments

| Port | Role |
|------|------|
| 8180 | S1 encoder server |
| 8181 | S2 encoder server |
| 8185 | Head server (S12) |
| 8183 | Original standalone server |

---

## Docker

**Dockerfiles:**
- `Dockerfile` / `Dockerfile.head` / `Dockerfile.s1` / `Dockerfile.s2` — build images per role
- Base image: `nvcr.io/nvidia/l4t-jetpack:r36.4.0` (Jetson L4T with JetPack)
- ONNX Runtime GPU wheel installed from Ultralytics assets (aarch64, Python 3.10)

**Docker Compose files:**
- `docker-compose.multi.yml` — all four servers on one node (for local testing)
- `docker-compose.s1.yml` / `docker-compose.s2.yml` — individual server deployments
- `docker-compose.original.yml` — original server only
- Uses `network_mode: host` and `runtime: nvidia`

**Build and run all:**
```bash
docker compose -f docker-compose.multi.yml up --build
```

---

## Common Commands

**Run servers without Docker:**
```bash
# Original model
python3 system/single_server.py -m ensemble-effnet-c5-lr-0.005-tin --original -p 8183

# Ensemble encoders (S1 and S2)
python3 system/single_server.py -m ensemble-effnet-c5-lr-0.005-tin -n 1 -p 8180 -s obelix195:8185
python3 system/single_server.py -m ensemble-effnet-c5-lr-0.005-tin -n 2 -p 8181 -s obelix195:8185
python3 system/head_server.py   -m ensemble-effnet-c5-lr-0.005-tin -p 8185
```

**Run the metrics client:**
```bash
python3 system/metrics_client.py -d 0.05 -f -w -i my_experiment
```

**Run the Streamlit UI:**
```bash
streamlit run system/streamlit_app.py
```

**Inject failures:**
```bash
./fail_original.sh <start_delay_sec> <num_iterations> iobt25-original-server
./fail_s12.sh      <start_delay_sec> <num_iterations> iobt25-head-server
./fail_s2.sh       <start_delay_sec> <num_iterations> iobt25-s2-server
```

**Quick gRPC benchmark:**
```bash
cd system
python3 run_client.py --server1 localhost:8183 -f PredictOriginal -i 100 -m ensemble-effnet-c5-lr-0.005-tin
```

---

## Key Design Notes

- **Request matching in head server**: S1 and S2 send encoder outputs to the head server with the same `request_id`. The head stores the first arrival in a dict and runs inference only when the second arrives. This means requests must be sent concurrently (via `asyncio.gather`).
- **Timing propagation**: `enc_service_time` and `enc_send_time` are embedded in the `PredictRequest` so the head can compute end-to-end service time and network overhead.
- **Fallback in `PredictForward`**: if the head server call throws, the encoder server falls back to local classification using `classifier1.onnx` — this is the S1-solo fallback path.
- **Failure detection**: `metrics_client.py` uses two counters (`orig_fail`, `mel_fail`) written by heartbeat threads. The inference loop reads these to decide which path to use.
- **Log timestamps are nanoseconds** (`time.time_ns()`); response/service times elsewhere are seconds or converted to ms for display.
