# IoBT 2025 Experiments

All commands must be run from the project root (`iobt25-demo/`).

---

## I. EfficientNet — Failure Detection & Recovery

### 1. Start containers

Run each command on its respective node:

**Head node:**
```bash
MODEL=ensemble-effnet-c5-lr-0.005-tin docker compose -f docker-compose.head.yml up --build
```

**S1 node:**
```bash
MODEL=ensemble-effnet-c5-lr-0.005-tin HEAD=<head-node-ip>:8180 docker compose -f docker-compose.s1.yml up --build
```

**S2 node:**
```bash
MODEL=ensemble-effnet-c5-lr-0.005-tin HEAD=<head-node-ip>:8180 docker compose -f docker-compose.s2.yml up --build
```

**Original node:**
```bash
MODEL=effnet-b0 docker compose -f docker-compose.original.yml up --build
```

### 2. Gather initial metrics (baseline — no failures)

Run the client for 1 minute across each failover stage to gather baseline response and service times:

**Original only:**
```bash
python3 system/metrics_client.py -d 60 -w -i effnet_baseline_original
```

**MEL ensemble only (kill original first, then restart and skip to MEL):**
```bash
python3 system/metrics_client.py -d 60 -f -w -i effnet_baseline_mel
```

**S1 solo only (kill original + S2, then restart and skip to S1):**
```bash
python3 system/metrics_client.py -d 60 -f -w -i effnet_baseline_s1
```

### 3. Experiment A — Failure of original server

Start the client with failover enabled:
```bash
python3 system/metrics_client.py -d 120 -f -w -i effnet_fail_original
```

In a separate terminal, inject the failure after a delay (e.g. 10 seconds):
```bash
./fail_original.sh 10 1 effnet_fail_original effnet-b0
```

**Note the timestamps:**
- `system/results/fail_original.log` — `Killing_container` and `Container_killed` from the bash script
- `system/results/fail_original.log` — `Orig_down` from the heartbeat thread
- `system/results/fail_original.log` — `Mel_ready` from the client inference loop (first MEL request)

Failure detection time = `Orig_down` timestamp − `Container_killed` timestamp

### 4. Experiment B — Failure of S2 server

Start the client with failover enabled:
```bash
python3 system/metrics_client.py -d 120 -f -w -i effnet_fail_s2
```

In a separate terminal, inject the failure after a delay (e.g. 10 seconds):
```bash
./fail_s2.sh 10 1 effnet_fail_s2 ensemble-effnet-c5-lr-0.005-tin <head-node-ip>:8180
```

**Note the timestamps:**
- `system/results/fail_s2.log` — `Killing_container` and `Container_killed` from the bash script
- `system/results/fail_s2.log` — `S2_down` from the heartbeat thread
- `system/results/fail_s2.log` — `S2_ready` from the client inference loop (first S1-solo request)

Failure detection time = `S2_down` timestamp − `Container_killed` timestamp

---

## II. DeepSpeech2 — Failure Detection & Recovery

### 1. Start containers

Run each command on its respective node:

**Head node:**
```bash
MODEL=ensemble-deepspeech2-c2-librispeech docker compose -f docker-compose.head.yml up --build
```

**S1 node:**
```bash
MODEL=ensemble-deepspeech2-c2-librispeech HEAD=<head-node-ip>:8180 docker compose -f docker-compose.s1.yml up --build
```

**S2 node:**
```bash
MODEL=ensemble-deepspeech2-c2-librispeech HEAD=<head-node-ip>:8180 docker compose -f docker-compose.s2.yml up --build
```

**Original node:**
```bash
MODEL=deepspeech2 docker compose -f docker-compose.original.yml up --build
```

### 2. Gather initial metrics (baseline — no failures)

**Original only:**
```bash
python3 system/metrics_client.py -d 60 -w -i deepspeech_baseline_original --task deepspeech
```

**MEL ensemble only:**
```bash
python3 system/metrics_client.py -d 60 -f -w -i deepspeech_baseline_mel --task deepspeech
```

**S1 solo only:**
```bash
python3 system/metrics_client.py -d 60 -f -w -i deepspeech_baseline_s1 --task deepspeech
```

### 3. Experiment A — Failure of original server

```bash
python3 system/metrics_client.py -d 120 -f -w -i deepspeech_fail_original --task deepspeech
```

In a separate terminal:
```bash
./fail_original.sh 10 1 deepspeech_fail_original deepspeech2
```

**Note the timestamps** in `system/results/fail_original.log` (same fields as Experiment I.3).

### 4. Experiment B — Failure of S2 server

```bash
python3 system/metrics_client.py -d 120 -f -w -i deepspeech_fail_s2 --task deepspeech
```

In a separate terminal:
```bash
./fail_s2.sh 10 1 deepspeech_fail_s2 ensemble-deepspeech2-c2-librispeech <head-node-ip>:8180
```

**Note the timestamps** in `system/results/fail_s2.log` (same fields as Experiment I.4).

---

## III. Colocated Inference — Interference Effects

### 1. Start EfficientNet containers (all nodes)

Run each command on its respective node:

**Head node:**
```bash
MODEL=ensemble-effnet-c5-lr-0.005-tin docker compose -f docker-compose.head.yml up --build
```

**S1 node:**
```bash
MODEL=ensemble-effnet-c5-lr-0.005-tin HEAD=<head-node-ip>:8180 docker compose -f docker-compose.s1.yml up --build
```

**S2 node:**
```bash
MODEL=ensemble-effnet-c5-lr-0.005-tin HEAD=<head-node-ip>:8180 docker compose -f docker-compose.s2.yml up --build
```

**Original node:**
```bash
MODEL=effnet-b0 docker compose -f docker-compose.original.yml up --build
```

---

### III-A. Colocation: DeepSpeech2 original on S1 node

#### 2. Start original DeepSpeech2 server on the S1 node

On the S1 node, run alongside the existing S1 encoder container:
```bash
python3 system/single_server.py -m deepspeech2 --original -p 8190
```

Or via Docker:
```bash
ORIGINAL=deepspeech2 docker compose -f docker-compose.original.yml up
```

#### 3a. Sub-experiment: failure of original EfficientNet

```bash
python3 system/metrics_client.py -d 120 -f -w -i colocated_s1_fail_original
```

In a separate terminal:
```bash
./fail_original.sh 10 1 colocated_s1_fail_original effnet-b0
```

**Note timestamps** in `system/results/colocated_s1_fail_original/fail_original.log`.

#### 3b. Sub-experiment: failure of S2

```bash
python3 system/metrics_client.py -d 120 -f -w -i colocated_s1_fail_s2
```

In a separate terminal:
```bash
./fail_s2.sh 10 1 colocated_s1_fail_s2 ensemble-effnet-c5-lr-0.005-tin <head-node-ip>:8180
```

**Note timestamps** in `system/results/fail_s2.log`.

---

### III-B. Colocation: DeepSpeech2 original on head node

#### 2. Start original DeepSpeech2 server on the head node

On the head node, run alongside the existing head container:
```bash
python3 system/single_server.py -m deepspeech2 --original -p 8190
```

Or via Docker:
```bash
ORIGINAL=deepspeech2 docker compose -f docker-compose.original.yml up
```

#### 3. Sub-experiment: failure of original EfficientNet

```bash
python3 system/metrics_client.py -d 120 -f -w -i colocated_head_fail_original
```

In a separate terminal:
```bash
./fail_original.sh 10 1 colocated_head_fail_original effnet-b0
```

**Note timestamps** in `system/results/colocated_head_fail_original/fail_original.log`.
