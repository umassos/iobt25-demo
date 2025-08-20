import streamlit as st
import pandas as pd
import time
import numpy as np
from PIL import Image
import threading
import asyncio
import socket

#import mqtt_source
import grpc
import argparse
import logging
import timeit
import socket
import os
import io
import base64
from enum import Enum
from labels import imgnet_labels
import config
import json 
import argparse
from inference_pb2_grpc import EncoderServiceStub, HeadServiceStub
from inference_pb2 import PredictRequest, HeartbeatRequest, HeartbeatResponse

parent_dir = os.path.dirname(__file__)
print(parent_dir)
# Access a file in parent directory
tin_labels = {}
with open(os.path.join(parent_dir, 'tieredimgnet_labels.json'), 'r') as f:
    tin_labels = json.load(f)
    tin_labels = {int(k): v for k, v in tin_labels.items()}

# Import your existing functions
from run_client import (
    remote_request, EncoderServiceStub, HeadServiceStub, HeartbeatRequest, HeartbeatResponse
)

def msg_to_np(image_bytes):
    decoded_bytes = base64.b64decode(image_bytes)
    img = Image.open(io.BytesIO(decoded_bytes))
    mat = np.array(img)
    return mat

def msg_to_bytes(msg):
    png = base64.b64decode(msg.payload)
    return png

def encode_image(image: np.array):
    """ Encode image to bytes """
    image_bytes_buffer = io.BytesIO()
    result_image = Image.fromarray(image)
    result_image.save(image_bytes_buffer, "JPEG")
    image_bytes = base64.b64encode(image_bytes_buffer.getvalue()).decode("ascii")

    return image_bytes


def bytes_to_np(image_bytes):
    img = np.array(Image.open(io.BytesIO(image_bytes)))
    img = np.ascontiguousarray(img[...,[0, 1, 2]])

    return img

def convert_to_model_input(image):
    """
    Convert image from [H, W, C] format to [1, C, H, W] format for model input.
    Resizes image to 224x224 and normalizes if needed.
    """
    # Resize image to 224x224
    img_resized = Image.fromarray(image).resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    
    # Convert from HWC to CHW format
    img_chw = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension to make it [1, C, H, W]
    img_batch = np.expand_dims(img_chw, axis=0)
    
    # Ensure contiguous memory layout
    img_batch = np.ascontiguousarray(img_batch)
    
    return img_batch

class Status(Enum):
    ACTIVE = "🟢 Active"
    READY = " 🟡 Ready"
    DOWN = "🔴 Down"
    INITIALIZING = "⏳ Initializing"


source_num = 0
source_message = None
input_image = np.zeros([1, 3, 224, 224])
predicted_label = "Unknown"
confidence = 0.0


# Server results
image1 = np.zeros([640, 640, 3])
image2 = np.zeros([640, 640, 3])
image12 = np.zeros([640, 640, 3])

# Raw image for display
raw_image = np.zeros([224, 224, 3])

# Status tracking
app1_status = [{"node": config.server1_addr, "status": Status.INITIALIZING.value, "response_time": None, "service_time": None, "request_count": 0, "last_heartbeat": None}]
app2_status = [{"node": config.server2_addr, "status": Status.INITIALIZING.value, "response_time": None, "service_time": None, "request_count": 0, "last_heartbeat": None}]
app12_status = [{"node": config.server12_addr, "status": Status.INITIALIZING.value, "last_heartbeat": None}]
orig_status = [{"node": config.server_orig_addr, "status": Status.INITIALIZING.value, "response_time": None, "service_time": None, "request_count": 0, "last_heartbeat": None}]

orig_failover_time = 0
app1_failover_time = 0
app2_failover_time = 0
app12_failover_time = 0

orig_failover_status = {"last_heartbeat": None, "last_close_loop": None, "last_online_from_heartbeat": None, "last_online_from_close_loop": None, "last_failure_from_heartbeat": None, "last_failure_from_close_loop": None, "failure_detection_time": None}
app1_failover_status = {"last_heartbeat": None, "last_close_loop": None, "last_online_from_heartbeat": None, "last_online_from_close_loop": None, "last_failure_from_heartbeat": None, "last_failure_from_close_loop": None, "failure_detection_time": None}
app2_failover_status = {"last_heartbeat": None, "last_close_loop": None, "last_online_from_heartbeat": None, "last_online_from_close_loop": None, "last_failure_from_heartbeat": None, "last_failure_from_close_loop": None, "failure_detection_time": None}
app12_failover_status = {"last_heartbeat": None, "last_close_loop": None, "last_online_from_heartbeat": None, "last_online_from_close_loop": None, "last_failure_from_heartbeat": None, "last_failure_from_close_loop": None, "failure_detection_time": None}

heartbeat_times = {"orig": [0], "s1": [0], "s2": [0], "s12": [0]}

orig_heartbeat_failover_time = 0
app1_heartbeat_failover_time = 0
app2_heartbeat_failover_time = 0
app12_heartbeat_failover_time = 0

# Inference metrics for GUI
inference_metrics = {
    "current_request": 0,
    "total_requests": 0,
    "function": "",
    "model_name": "",
    "server1": "",
    "server2": "",
    "gather_time": 0.0,
    "last_times": [0.0],
    "average_times": [0.0],
    "progress": 0.0,
    "completed": False,
    "final_average_times": [0.0],
    "csv_path": "",
    "total_results": 0
}
inference_results = []

def get_mqtt_message():
    global source_num, source_message, input_image, raw_image
    while True:
        try:
            source_num, source_message = mqtt_client.get_message(config.topic)
            raw_image = bytes_to_np(msg_to_bytes(source_message))
            input_image = convert_to_model_input(raw_image)
            time.sleep(0.1)
        except Exception as e:
            print(f"Error getting MQTT message: {e}, trying again in 1 second...")
            time.sleep(1)

def original_close_loop():
    global orig_status, orig_failover_status
    # Setup gRPC connection for original server
    channel_original = grpc.insecure_channel(config.server_orig_addr)
    stub_original = EncoderServiceStub(channel_original)
    
    while True:
        init_time = time.time()
        orig_failover_status["last_close_loop"] = init_time
        try:
            heartbeat_request = HeartbeatRequest()
            heartbeat_response = stub_original.Heartbeat(heartbeat_request)
            recv_time = time.time()
            
            orig_failover_status["last_online_from_close_loop"] = recv_time
            orig_failover_status["last_failure_from_close_loop"] = None

        except Exception as close_loop_error:
            failure_time = time.time()
            if not orig_failover_status["last_failure_from_close_loop"]:
                orig_failover_status["last_failure_from_close_loop"] = failure_time

def s1_close_loop():
    global app1_status, app1_failover_status
    # Setup gRPC connection
    channel1 = grpc.insecure_channel(config.server1_addr)
    stub1 = EncoderServiceStub(channel1)
    
    while True:
        init_time = time.time()
        app1_failover_status["last_close_loop"] = init_time
        try:
            heartbeat_request = HeartbeatRequest()
            heartbeat_response = stub1.Heartbeat(heartbeat_request)
            recv_time = time.time()

            app1_failover_status["last_online_from_close_loop"] = recv_time
            app1_failover_status["last_failure_from_close_loop"] = None

        except Exception as close_loop_error:
            failure_time = time.time()
            if not app1_failover_status["last_failure_from_close_loop"]:
                app1_failover_status["last_failure_from_close_loop"] = failure_time

def s2_close_loop():
    global app2_status, app2_failover_status
    # Setup gRPC connection
    channel2 = grpc.insecure_channel(config.server2_addr)
    stub2 = EncoderServiceStub(channel2)
    
    while True:
        init_time = time.time()
        app2_failover_status["last_close_loop"] = init_time
        try:
            heartbeat_request = HeartbeatRequest()
            heartbeat_response = stub2.Heartbeat(heartbeat_request)
            recv_time = time.time()

            app2_failover_status["last_online_from_close_loop"] = recv_time
            app2_failover_status["last_failure_from_close_loop"] = None

        except Exception as close_loop_error:
            failure_time = time.time()
            if not app2_failover_status["last_failure_from_close_loop"]:
                app2_failover_status["last_failure_from_close_loop"] = failure_time

def s12_close_loop():
    global app12_status, app12_failover_status
    # Setup gRPC connection
    channel12 = grpc.insecure_channel(config.server12_addr)
    stub12 = HeadServiceStub(channel12)
    
    while True:
        init_time = time.time()
        app12_failover_status["last_close_loop"] = init_time
        try:
            heartbeat_request = HeartbeatRequest()
            heartbeat_response = stub12.Heartbeat(heartbeat_request)
            recv_time = time.time()
            
            app12_failover_status["last_online_from_close_loop"] = recv_time
            app12_failover_status["last_failure_from_close_loop"] = None
            
        except Exception as close_loop_error:
            failure_time = time.time()
            if not app12_failover_status["last_failure_from_close_loop"]:
                app12_failover_status["last_failure_from_close_loop"] = failure_time


def original_heartbeat():
    global orig_status, orig_failover_status, heartbeat_times
    # Setup gRPC connection for original server
    channel_original = grpc.insecure_channel(config.server_orig_addr)
    stub_original = EncoderServiceStub(channel_original)
    alive = True
    while True:
        # Check Original server heartbeat first
        init_time = time.time()
        orig_failover_status["last_heartbeat"] = init_time
        try:
            heartbeat_request = HeartbeatRequest()
            heartbeat_response = stub_original.Heartbeat(heartbeat_request)
            recv_time = time.time()
            heartbeat_times["orig"].append(recv_time - init_time)

            orig_status[0]["last_heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(orig_failover_status["last_heartbeat"]))
            orig_status[0]["status"] = Status.ACTIVE.value

            orig_failover_status["last_online_from_heartbeat"] = recv_time
            orig_failover_status["last_failure_from_heartbeat"] = None
            alive = True
        except Exception as heartbeat_error:
            log_time = time.time_ns()
            failure_time = time.time()
            if alive:
                alive = False
                if args.write_log:
                    with open(f"{args.experiment_dir}/actual_fail_orig.txt", "a") as f:
                        f.write(f"{log_time} - Orig_down\n")
            orig_status[0]["status"] = Status.DOWN.value
            if not orig_failover_status["last_failure_from_heartbeat"]:
                orig_failover_status["last_failure_from_heartbeat"] = failure_time

        time.sleep(config.heartbeat_interval / 1000)

def s1_heartbeat():
    global app1_status, app1_failover_status, heartbeat_times
    # Setup gRPC connection
    channel1 = grpc.insecure_channel(config.server1_addr)
    stub1 = EncoderServiceStub(channel1)

    print(config.server1_addr)
    
    while True:
        # Check S1 server heartbeat first
        init_time = time.time()
        app1_failover_status["last_heartbeat"] = init_time
        try:
            heartbeat_request = HeartbeatRequest()
            heartbeat_response = stub1.Heartbeat(heartbeat_request)
            recv_time = time.time()
            heartbeat_times["s1"].append(recv_time - init_time)
            
            app1_status[0]["last_heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(app1_failover_status["last_heartbeat"]))
            app1_status[0]["status"] = Status.READY.value if orig_status[0]["status"] == Status.ACTIVE.value else Status.ACTIVE.value
            
            app1_failover_status["last_online_from_heartbeat"] = recv_time
            app1_failover_status["last_failure_from_heartbeat"] = None

        except Exception as heartbeat_error:
            print("here")
            failure_time = time.time()
            app1_status[0]["status"] = Status.DOWN.value
            if not app1_failover_status["last_failure_from_heartbeat"]:
                app1_failover_status["last_failure_from_heartbeat"] = failure_time

        time.sleep(config.heartbeat_interval / 1000)

def s2_heartbeat():
    global app2_status, app2_failover_status, heartbeat_times
    # Setup gRPC connection
    channel2 = grpc.insecure_channel(config.server2_addr)
    stub2 = EncoderServiceStub(channel2)
    alive = True
    while True:
        # Check S2 server heartbeat first
        init_time = time.time()
        app2_failover_status["last_heartbeat"] = init_time
        try:
            heartbeat_request = HeartbeatRequest()
            heartbeat_response = stub2.Heartbeat(heartbeat_request)
            recv_time = time.time()
            heartbeat_times["s2"].append(recv_time - init_time)
            app2_status[0]["last_heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(app2_failover_status["last_heartbeat"]))
            app2_status[0]["status"] = Status.READY.value if orig_status[0]["status"] == Status.ACTIVE.value else Status.ACTIVE.value

            app2_failover_status["last_online_from_heartbeat"] = recv_time
            app2_failover_status["last_failure_from_heartbeat"] = None

        except Exception as heartbeat_error:
            log_time = time.time_ns()
            failure_time = time.time()
            if alive:
                alive = False
                if args.write_log:
                    with open(f"{args.experiment_dir}/actual_fail_s2.txt", "a") as f:
                        f.write(f"{log_time} - S2_down\n")
            app2_status[0]["status"] = Status.DOWN.value
            if not app2_failover_status["last_failure_from_heartbeat"]:
                app2_failover_status["last_failure_from_heartbeat"] = failure_time
        
        time.sleep(config.heartbeat_interval / 1000)

def s12_heartbeat():
    global app12_status, app12_failover_status, heartbeat_times
    
    # Setup gRPC connection for head server heartbeat
    channel12 = grpc.insecure_channel(config.server12_addr)
    stub12 = HeadServiceStub(channel12)
    alive = True
    while True:
        # Check head server heartbeat first
        init_time = time.time()
        app12_failover_status["last_heartbeat"] = init_time
        try:
            heartbeat_request = HeartbeatRequest()
            heartbeat_response = stub12.Heartbeat(heartbeat_request)
            recv_time = time.time()
            heartbeat_times["s12"].append(recv_time - init_time)
            app12_status[0]["last_heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(app12_failover_status["last_heartbeat"]))
            app12_status[0]["status"] = Status.READY.value if orig_status[0]["status"] == Status.ACTIVE.value else Status.ACTIVE.value

            app12_failover_status["last_online_from_heartbeat"] = recv_time
            app12_failover_status["last_failure_from_heartbeat"] = None
            alive = True
        except Exception as heartbeat_error:
            log_time = time.time_ns()
            failure_time = time.time()
            if alive:
                alive = False
                if args.write_log:
                    with open(f"{args.experiment_dir}/actual_fail_s12.txt", "a") as f:
                        f.write(f"{log_time} - S12_down\n") 
            app12_status[0]["status"] = Status.DOWN.value
            if not app12_failover_status["last_failure_from_heartbeat"]:
                app12_failover_status["last_failure_from_heartbeat"] = failure_time

        time.sleep(config.heartbeat_interval / 1000)


def get_inference_class(resp, data='imgnet'):
    global predicted_label
    try:     
        # (Note: Theses are logits and not probabilities)
        output_array = np.frombuffer(resp.output, dtype=np.float32)
        predicted_class = np.argmax(output_array)
        
        if data == 'imgnet':
            predicted_label = imgnet_labels.get(predicted_class, f"Unknown class {predicted_class}")
        elif data == 'tin':
            predicted_label = tin_labels.get(predicted_class, f"Unknown class {predicted_class}")
        else:
            predicted_label = f"Unknown class idx: {predicted_class}"
        return predicted_label
    except Exception as e:
        print(f"Error processing inference response: {e}")
        # predicted_label = "Unknown due to error in get_inference_class"
        return predicted_label 


def run_inference(server1, server2, server_original, requests, function, model_name, duration):
    global image1, image2, image12, app1_status, app2_status, app12_status, app1_failover_time, app2_failover_time, app12_failover_time, orig_status, orig_failover_time, predicted_label, input_image, experiment_id
    # Setup gRPC connection
    channel1 = grpc.insecure_channel(server1)
    channel2 = grpc.insecure_channel(server2)
    channel_original = grpc.insecure_channel(server_original)
    
    stub1 = EncoderServiceStub(channel1)
    stub2 = EncoderServiceStub(channel2)
    stub_original = EncoderServiceStub(channel_original)

    results = []

    response_times = []
    timestamps = []
    service_times = []
    sleep_interval = 1 / duration
    # print("sleep_interval", sleep_interval)
    orig_fail = 0
    mel_fail = 0

    experiment_start_time = time.time()
    i = 0
    input = input_image.astype(np.float32)    
    while time.time() - experiment_start_time < duration * 1000:
        i+=1
        curr_time = time.time()
        start_time = timeit.default_timer()
        if orig_fail < 1:
            try: 
                response, resp_time = asyncio.run(remote_request(input=input, request_id=i, function='PredictOriginal', stub=stub_original))
                response_times.append(resp_time)
                service_times.append(response.service_time)
                # resp, time = asyncio.run(remote_request(input=input, request_id=i, function='PredictOriginal', stub=stub_original))
                orig_status[0]["request_count"] += 1
                orig_status[0]["response_time"] = f'{resp_time * 1000:.4f}'
                orig_failover_time = 0
                results.append([resp_time])
                timestamps.append(curr_time)
                print("resp_time", resp_time)
                orig_status[0]["service_time"] = f'{response.service_time * 1000:.4f}'
                if args.write_log:
                    with open(f"{args.experiment_dir}/original_response_time.txt", "a") as f:
                        f.write(f"{orig_status[0]['response_time']}\n")
                    with open(f"{args.experiment_dir}/original_service_time.txt", "a") as f:
                        f.write(f"{orig_status[0]['service_time']}\n")
                
                # predicted_label = get_inference_class(resp, "imgnet")
                continue

            except Exception as e:
                print("here2")
                print(e)
                orig_fail += 1
                end_time = time.time()
                if orig_failover_time == 0:
                    orig_failover_time = end_time - start_time    
                orig_status[0]["response_time"] = None
        
        elif mel_fail < 1:
            mel_start_time = timeit.default_timer()
            try: 
                resp = [None, None]
                thread1 = threading.Thread(target=lambda: resp.__setitem__(0, asyncio.run(remote_request(input=input, request_id=i, function='PredictForward', stub=stub1))))
                thread2 = threading.Thread(target=lambda: resp.__setitem__(1, asyncio.run(remote_request(input=input, request_id=i, function='PredictForward', stub=stub2))))
                
                thread1.start()
                thread2.start()
                thread1.join()
                thread2.join()
                print("ny")
                times = [resp[0][1], resp[1][1]]
                print('Times:', times)
                results.append(times)
                app1_status[0]["response_time"] = f'{resp[0][1] * 1000:.4f}'
                app2_status[0]["response_time"] = f'{resp[1][1] * 1000:.4f}'
                app1_status[0]["request_count"] += 1
                app2_status[0]["request_count"] += 1
                app1_status[0]["service_time"] = f'{resp[0][0].service_time * 1000:.4f}'
                app2_status[0]["service_time"] = f'{resp[1][0].service_time * 1000:.4f}'


                if resp[0][0].has_result:
                    response_times.append(resp[0][1])
                    service_times.append(resp[0][0].service_time)
                    print("resp_time", resp[0][1])
                else:
                    response_times.append(resp[1][1])
                    service_times.append(resp[1][0].service_time)
                    print("resp_time", resp[0][1])
                timestamps.append(curr_time)

                if args.write_log:
                    with open(f"{args.experiment_dir}/s1_ensemble_response_time.txt", "a") as f:
                        f.write(f"{app1_status[0]['response_time']}\n")
                    with open(f"{args.experiment_dir}/s2_ensemble_response_time.txt", "a") as f:
                        f.write(f"{app2_status[0]['response_time']}\n")
                    with open(f"{args.experiment_dir}/s1_ensemble_service_time.txt", "a") as f:
                        f.write(f"{app1_status[0]['service_time']}\n")
                    with open(f"{args.experiment_dir}/s2_ensemble_service_time.txt", "a") as f:
                        f.write(f"{app2_status[0]['service_time']}\n")

                    with open(f"{args.experiment_dir}/s1_ensemble_network_time.txt", "a") as f:
                        f.write(f'{resp[0][0].network_time * 1000:.4f}\n')
                    with open(f"{args.experiment_dir}/s2_ensemble_network_time.txt", "a") as f:
                        f.write(f'{resp[1][0].network_time * 1000:.4f}\n')

                app1_failover_time = 0
                app2_failover_time = 0
                app12_failover_time = 0
                # predicted_label = get_inference_class(resp[0][0], "tin")
                continue

            except Exception as e:
                print("here3")
                end_time = timeit.default_timer()
                mel_fail += 1
                if app1_failover_time == 0:
                    app1_failover_time = end_time - mel_start_time
                if app2_failover_time == 0:
                    app2_failover_time = end_time - mel_start_time
                # app1_status[0]["status"] = Status.INACTIVE.value
                app1_status[0]["response_time"] = None
                app2_status[0]["response_time"] = None
                print(f"Inference ensemble model error: {e}")

        else: 
            try: 
                # if app1_status[0]["status"] == Status.READY.value or app1_status[0]["status"] == Status.ACTIVE.value:
                resp, resp_time = asyncio.run(remote_request(input=input, request_id=i, function='Predict', stub=stub1))
                app1_status[0]["response_time"] = f'{resp_time * 1000:.4f}'
                app1_status[0]["request_count"] += 1
                app1_failover_time = 0
                results.append(resp_time)
                response_times.append(resp_time)
                service_times.append(resp.service_time)
                timestamps.append(curr_time)
                print("resp_time", resp_time)
                # predicted_label = get_inference_class(resp, "tin")
                app1_status[0]["service_time"] = f'{resp.service_time * 1000:.4f}'
                if args.write_log:
                    with open(f"{args.experiment_dir}/s1_solo_response_time.txt", "a") as f:
                        f.write(f"{app1_status[0]['response_time']}\n")
                    with open(f"{args.experiment_dir}/s1_solo_service_time.txt", "a") as f:
                        f.write(f"{app1_status[0]['service_time']}\n")

                # elif app2_status[0]["status"] == Status.READY.value or app2_status[0]["status"] == Status.ACTIVE.value:
                # resp, resp_time = asyncio.run(remote_request(input=input, request_id=i, function='Predict', stub=stub2))
                # app2_status[0]["response_time"] = f'{resp_time * 1000:.4f}'
                # app2_status[0]["request_count"] += 1
                # app2_failover_time = 0
                # results.append(resp_time)
                # response_times.append(resp_time)
                # service_times.append(resp.service_time)
                # timestamps.append(curr_time)
                # print("resp_time", resp_time)
                # # predicted_label = get_inference_class(resp, "tin")
                # app2_status[0]["service_time"] = f'{resp.service_time * 1000:.4f}'
                # if args.write_log:
                #     with open(f"{args.experiment_dir}/s2_solo_response_time.txt", "a") as f:
                #         f.write(f"{app2_status[0]['response_time']}\n")
                #     with open(f"{args.experiment_dir}/s2_solo_service_time.txt", "a") as f:
                #         f.write(f"{app2_status[0]['service_time']}\n")
                continue 
            except Exception as e:
                print("here4")
                end_time = timeit.default_timer()
                if app1_failover_time == 0:
                    app1_failover_time = end_time - mel_start_time
                if app2_failover_time == 0:
                    app2_failover_time = end_time - mel_start_time
                app1_status[0]["response_time"] = None
                app2_status[0]["response_time"] = None
                print(f"Inference backup model error: {e}")

    print(i)

    # print(len(times), len(response_times), len(service_times))
    df = pd.DataFrame({
        "timestamps": timestamps,
        "response_time": response_times,
        "service_time": service_times,
    })
    print(df.tail(10))

    os.makedirs(args.experiment_dir, exist_ok=True)
    df.to_csv("{}/response_times.csv".format(args.experiment_dir), index=False)
    # df.to_csv("system/results/response_times.csv", index=False)

    # if len(results[0]) == 1:
    #     df = pd.DataFrame(results, columns=["Time"])
    # else:
    #     df = pd.DataFrame(results, columns=["Time1", "Time2"])
    # print("Average Time taken for both requests:", np.mean(results, axis=0))
    # hostname = socket.gethostname()
    # df.to_csv(
    #     f"system/rpc_results/{hostname.split('.')[0]}_client_{function}_{model_name}_results.csv",
    #     index=False,
    # )


def _parse_args():
    parser = argparse.ArgumentParser(description="Multi-level Ensemble Learning (MEL) Inference Client")
    parser.add_argument("-hb", "--hb-resp-time", action="store_true", help="Show heartbeat response times for each servers")
    parser.add_argument("-rt", "--refresh-time", type=float, default=1, help="Refresh time for the GUI in seconds")
    parser.add_argument("-st", "--service-time", action="store_true", help="Show the service inference time for the servers")
    parser.add_argument("-w", "--write-log", action="store_true", help="Start writing the metrics to file")
    parser.add_argument("-i", "--experiment-id", type=str, default=None, help="Experiment ID")
    parser.add_argument("-r", "--requests", type=int, default=1000, help="Number of requests to send")
    parser.add_argument("-d", "--duration", type=float, default=0.01, help="Duration of the experiment in seconds")
    parser.add_argument("-t2", "--task2", action="store_true", help="Run task 2")
    return parser.parse_args()


args = None
def main():
    global args
    args = _parse_args()

    if not args.experiment_id:
        experiment_id = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
    else:
        experiment_id = args.experiment_id
    
    args.experiment_dir = f"./system/results/{experiment_id}"
    
    
    

    orig_server_addr = config.server_orig_addr 
    if args.task2:
        orig_server_addr = config.server_task2
    run_inference(config.server1_addr, config.server2_addr, orig_server_addr, args.requests, "PredictForward", "model_name", args.duration)
    # s1_heartbeat()

if __name__ == "__main__":
    main() 
