# Ensemble Fault Tolerant


## Running a standalone service 
### Server:
```
whanafy@obelix191:~/ensemble-inference$ python3 system/single_server.py -p 8180 -m Ens_Effnet_CIFAR100_C5 -n 1
```
### Client (Small Model)
```
whanafy@obelix190:~/ensemble-inference$ python3 system/run_client.py --server1 obelix192:8180 -f Predict -i 1000
```

### Client (Standalone Model)
```
whanafy@obelix190:~/ensemble-inference$ python3 system/run_client.py --server1 obelix192:8180 -f PredictFull -i 1000
```

### Client (Original Model)
```
whanafy@obelix190:~/ensemble-inference$ python3 system/run_client.py --server1 obelix192:8180 -f PredictOriginal -i 1000
```

## Running ensemble  
### Servers (Three):
```
whanafy@obelix192:~/ensemble-inference$ python3 system/single_server.py -p 8180 -m Ens_Effnet_CIFAR100_C5 -n 1 -s obelix194:8180
whanafy@obelix193:~/ensemble-inference$ python3 system/single_server.py -p 8180 -m Ens_Effnet_CIFAR100_C5 -n 2 -s obelix194:8180
whanafy@obelix194:~/ensemble-inference$ python3 system/head_server.py -p 8180 -m Ens_Effnet_CIFAR100_C5

```

### Client (Ensemble Model)
```
whanafy@obelix190:~/ensemble-inference$ python3 system/run_client.py --server1 obelix192:8180 --server2 obelix193:8180  -f PredictForward -i 1000
```


## Running model split
### Server (Three)
```
whanafy@obelix194:~/ensemble-inference$ python3 system/single_server.py -p 8180 --split 7-C
whanafy@obelix193:~/ensemble-inference$ python3 system/single_server.py -p 8180 -s obelix194:8180 --split 6
whanafy@obelix192:~/ensemble-inference$ python3 system/single_server.py -p 8180 -s obelix193:8180 --split 1-5
```

## Client
