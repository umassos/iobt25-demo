# Multi-level Ensemble Learning (MEL) - IoBT 25 Experimentation Demo 
To set up the code, pull from this repository 
```
git@github.com:umassos/iobt25-demo.git
```

Download the zip file for the models from [here](https://drive.google.com/file/d/1i_83ZMIvsk6BaLOSRKpJRdpdlzz84pbt/view?usp=sharing). Extract and place the `models` into the parent folder.

Run everything from the parent project directory e.g. in this case from inside the `iobt25-demo` folder. THe scripts and code assume they are being run from this directory and not from any other sub-directories. 

Make sure the `system/config.py` is updated to reflect the necessary server address for the client and server to run. 

## Running the server containers
The servers are packaged into docker containers, you can use `docker compose` to build and run the containers.

### All containers on one node
To build and start all four containers (s1, s2, head, original) at once:
```
docker compose -f docker-compose.multi.yml up --build
```

To specify models — `ORIGINAL` for the original server, `FAIL` for head/s1/s2:
```
ORIGINAL=effnet-b0 FAIL=ensemble-effnet-c5-lr-0.005-tin docker compose -f docker-compose.multi.yml up --build
```

For example, to run with DeepSpeech2 models:
```
ORIGINAL=deepspeech2 FAIL=ensemble-deepspeech2-c2-librispeech docker compose -f docker-compose.multi.yml up --build
```

### Individual containers (for multi-node deployments)
Each server has its own compose file. The `MODEL` variable sets the model subdirectory under `models/`, and `HEAD` sets the head server address for s1/s2.

**Head server** (default port 8180):
```
MODEL=ensemble-effnet-c5-lr-0.005-tin docker compose -f docker-compose.head.yml up --build
```

**S1 encoder server** (default port 8181):
```
MODEL=ensemble-effnet-c5-lr-0.005-tin HEAD=192.168.79.12:8180 docker compose -f docker-compose.s1.yml up --build
```

**S2 encoder server** (default port 8182):
```
MODEL=ensemble-effnet-c5-lr-0.005-tin HEAD=192.168.79.12:8180 docker compose -f docker-compose.s2.yml up --build
```

**Original server** (default port 8190):
```
MODEL=effnet-b0 docker compose -f docker-compose.original.yml up --build
```

> All individual compose files reuse the same `iobt26-server:latest` image. If it is already built, omit `--build` to skip rebuilding.


## Running the client
The Streamlit client is designed to run on your local machine, connecting to the remote servers. Server addresses are configured in `system/config.py`.

```
streamlit run system/streamlit_app.py
```

To run the metrics client as a Docker container:
```
docker compose -f docker-compose.client.yml up
```

Override parameters via environment variables:
```
TASK=deepspeech DURATION=0.1 EXPERIMENT_ID=my_run docker compose -f docker-compose.client.yml up
```

## Failure Injection 
On the respective servers, the doker container can be killed by run the fail script with the following arguments. 
```
./fail_{server}.sh start_delay num_iterations docker_container
```

- `start_delay`: Time in seconds the script waits for before injecting a failure
- `num_iterations`: Number of times to inject failure
- `docker_container`: The docker container e.g. `iobt25-original-server`

For example, in order to inject failures for `original` model with a start delay of 2 seconds, 100 iterations 

```
./fail_original.sh 2 100 iobt25-original-server
```  

## Running the Servers without Docker
Running the original model server
```
python3 system/single_server.py -m effnet-b0 --original -p 8190
```

Running the ensemble model servers
```
python3 system/single_server.py -m ensemble-effnet-c5-lr-0.005-tin -n 1 -p 8181 -s obelix192:8180
python3 system/single_server.py -m ensemble-effnet-c5-lr-0.005-tin -n 2 -p 8182 -s obelix192:8180
python3 system/head_server.py -m ensemble-effnet-c5-lr-0.005-tin -p 8180
```

## Running the client
`-f`: For running heartbeats closed loop
```
python3 system/metrics_client.py -d 0.05 -f
```