#!/bin/bash

START_DELAY=$1
LOOP_COUNT=$2
EXPERIMENT_ID=$3
MODEL=$4   # e.g. ensemble-effnet-c5-lr-0.005-tin or ensemble-deepspeech2-c2-librispeech
HEAD=$5    # e.g. 192.168.79.12:8180

LOG_DIR="${EXPERIMENT_ID:+./system/results/$EXPERIMENT_ID}"
LOG_DIR="${LOG_DIR:-./system/results}"
LOG_FILE="$LOG_DIR/fail_s2.log"
mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

log_event() {
    echo "$(date +%s%N) - $1" >> "$LOG_FILE"
}

counter=0
while [[ $LOOP_COUNT -lt 0 || $counter -lt $LOOP_COUNT ]]; do

    echo "---------------- Iteration $counter ----------------"
    echo "Killing S2 container"
    log_event "Killing_container"
    docker compose -f docker-compose.s2.yml kill > /dev/null
    log_event "Container_killed"

    echo "Starting S2 container"
    log_event "Starting_container"
    MODEL=$MODEL HEAD=$HEAD docker compose -f docker-compose.s2.yml up -d > /dev/null
    log_event "Container_started"

    sleep $START_DELAY

    counter=$((counter+1))

done
