#!/bin/bash

LOG_FILE="./system/results/fail_s12.log"
START_DELAY=$1
LOOP_COUNT=$2
# Ensure log file exists
touch "$LOG_FILE"

log_event() {
    echo "$(date +%s%N) - $1" >> "$LOG_FILE"
}

counter=0
while [[ $LOOP_COUNT -lt 0 || $counter -lt $LOOP_COUNT ]]; do

    echo "---------------- Iteration $counter ----------------"
    # Start container
    log_event "Stopping_container"
    docker stop $3 > /dev/null
    log_event "Container_stopped"

    sleep $START_DELAY

    log_event "Starting_container"
    docker start $3 > /dev/null
    log_event "Container_started"

    counter=$((counter+1))
done
