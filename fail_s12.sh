#!/bin/bash

LOG_FILE="./system/results/fail_s12.log"
START_DELAY=3
LOOP_COUNT=100
# Ensure log file exists
touch "$LOG_FILE"

log_event() {
    echo "$(date +%s%N) - $1" >> "$LOG_FILE"
}

counter=0
while [[ $LOOP_COUNT -lt 0 || $counter -lt $LOOP_COUNT ]]; do

    echo "---------------- Iteration $counter ----------------"
    # Start container
    log_event "Starting_container"
    docker-compose -f docker-compose.s12.yml up -d > /dev/null
    log_event "Container_started"

    sleep $START_DELAY

    log_event "Stopping_container"
    docker-compose -f docker-compose.s12.yml down > /dev/null
    log_event "Container_stopped"

    counter=$((counter+1))
done
