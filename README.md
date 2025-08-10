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

For e.g. starting all the containers - s1, s2, s12 and original at once on the same node, 
```
docker compose -f docker-compose.multi.yml up --build
```

Individually, containers can also be started using the respective  `docker-compose.{server}.yml` files


## Running the client 
In your terminal, run 
```
streamlit run system/streamlit_app.py
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
