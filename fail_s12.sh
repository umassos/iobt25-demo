#!/bin/bash

START_DELAY=$1
LOOP_COUNT=$2
EXPERIMENT_ID=$3
MODEL=$4   # e.g. ensemble-effnet-c5-lr-0.005-tin or ensemble-deepspeech2-c2-librispeech

LOG_DIR="${EXPERIMENT_ID:+./system/results/$EXPERIMENT_ID}"
LOG_DIR="${LOG_DIR:-./system/results}"
LOG_FILE="$LOG_DIR/fail_s12.log"
mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

log_event() {
    echo "$(date +%s%N) - $1" >> "$LOG_FILE"
}

counter=0
while [[ $LOOP_COUNT -lt 0 || $counter -lt $LOOP_COUNT ]]; do

    echo "---------------- Iteration $counter ----------------"
    echo "Killing Head container"
    log_event "Killing_container"
    docker compose -f docker-compose.head.yml kill > /dev/null
    log_event "Container_killed"

    echo "Starting Head container"
    log_event "Starting_container"
    MODEL=$MODEL docker compose -f docker-compose.head.yml up -d > /dev/null
    log_event "Container_started"

    sleep $START_DELAY

    counter=$((counter+1))
done
