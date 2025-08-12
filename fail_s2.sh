#!/bin/bash

LOG_FILE="./system/results/fail_s2.log"
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
    echo "Killing S2 container"
    # Start container
    log_event "Killing_container"
    docker-compose -f docker-compose.s2.yml kill > /dev/null
    log_event "Container_killed"

    echo "Starting S2 container"
    log_event "Starting_container"
    docker-compose -f docker-compose.s2.yml up -d > /dev/null
    log_event "Container_started"

    sleep $START_DELAY

    counter=$((counter+1))
    
done
