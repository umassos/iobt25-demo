import streamlit as st
import pandas as pd
import time
import numpy as np
from PIL import Image
import threading
import asyncio
import socket

import mqtt_source
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
from streamlit.runtime.scriptrunner import add_script_run_ctx

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



mqtt_client = mqtt_source.MQTTSource()
mqtt_client.start()

source_num = 0
source_message = None
input_image = np.zeros([1, 3, 224, 224])
predicted_label = "Unknown"
confidence = 0.0
experiment_id = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
experiment_dir = f"./system/results/{experiment_id}"
os.makedirs(experiment_dir, exist_ok=True)

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
                with open(f"{experiment_dir}/actual_fail_orig.txt", "a") as f:
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
                with open(f"{experiment_dir}/actual_fail_s2.txt", "a") as f:
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
                with open(f"{experiment_dir}/actual_fail_s12.txt", "a") as f:
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


def run_inference(server1, server2, server_original, requests, function, model_name):
    global image1, image2, image12, app1_status, app2_status, app12_status, app1_failover_time, app2_failover_time, app12_failover_time, orig_status, orig_failover_time, predicted_label, input_image, experiment_id
    # Setup gRPC connection
    channel1 = grpc.insecure_channel(server1)
    channel2 = grpc.insecure_channel(server2)
    channel_original = grpc.insecure_channel(server_original)
    
    stub1 = EncoderServiceStub(channel1)
    stub2 = EncoderServiceStub(channel2)
    stub_original = EncoderServiceStub(channel_original)

    results = []

    
    for i in range(requests):
        input = input_image.astype(np.float32)    
        # Original model inference
        start_time = timeit.default_timer()
        try: 
            resp, time = asyncio.run(remote_request(input=input, request_id=i, function='PredictOriginal', stub=stub_original))
            orig_status[0]["request_count"] += 1
            orig_status[0]["response_time"] = f'{time * 1000:.2f}ms'
            orig_failover_time = 0
            results.append(time)
            orig_status[0]["service_time"] = f'{resp.service_time * 1000:.2f}ms'
            with open(f"{experiment_dir}/original_response_time.txt", "a") as f:
                f.write(f"{orig_status[0]['response_time']}\n")
            with open(f"{experiment_dir}/original_service_time.txt", "a") as f:
                f.write(f"{orig_status[0]['service_time']}\n")
            
            predicted_label = get_inference_class(resp, "imgnet")
            continue

        except Exception as e:
            end_time = timeit.default_timer()
            if orig_failover_time == 0:
                orig_failover_time = end_time - start_time
            # orig_status[0]["status"] = Status.INACTIVE.value
            orig_status[0]["response_time"] = None
            print(f"Inference original model error {e}")

        mel_start_time = timeit.default_timer()
        try: 
            resp = [None, None]
            
            thread1 = threading.Thread(target=lambda: resp.__setitem__(0, asyncio.run(remote_request(input=input, request_id=i, function='PredictForward', stub=stub1))))
            thread2 = threading.Thread(target=lambda: resp.__setitem__(1, asyncio.run(remote_request(input=input, request_id=i, function='PredictForward', stub=stub2))))
            
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()
            
            times = [resp[0][1], resp[1][1]]
            # print('Times:', times)
            results.append(times)
            app1_status[0]["response_time"] = f'{resp[0][1] * 1000:.2f}ms'
            app2_status[0]["response_time"] = f'{resp[1][1] * 1000:.2f}ms'
            app1_status[0]["request_count"] += 1
            app2_status[0]["request_count"] += 1
            app1_status[0]["service_time"] = f'{resp[0][0].service_time * 1000:.2f}ms'
            app2_status[0]["service_time"] = f'{resp[1][0].service_time * 1000:.2f}ms'

            with open(f"{experiment_dir}/s1_ensemble_response_time.txt", "a") as f:
                f.write(f"{app1_status[0]['response_time']}\n")
            with open(f"{experiment_dir}/s2_ensemble_response_time.txt", "a") as f:
                f.write(f"{app2_status[0]['response_time']}\n")
            with open(f"{experiment_dir}/s1_ensemble_service_time.txt", "a") as f:
                f.write(f"{app1_status[0]['service_time']}\n")
            with open(f"{experiment_dir}/s2_ensemble_service_time.txt", "a") as f:
                f.write(f"{app2_status[0]['service_time']}\n")

            app1_failover_time = 0
            app2_failover_time = 0
            app12_failover_time = 0
            predicted_label = get_inference_class(resp[0][0], "tin")
            continue

        except Exception as e:
            end_time = timeit.default_timer()
            if app1_failover_time == 0:
                app1_failover_time = end_time - mel_start_time
            if app2_failover_time == 0:
                app2_failover_time = end_time - mel_start_time
            # app1_status[0]["status"] = Status.INACTIVE.value
            app1_status[0]["response_time"] = None
            app2_status[0]["response_time"] = None
            print(f"Inference ensemble model error: {e}")


        try: 
            if app1_status[0]["status"] == Status.READY.value or app1_status[0]["status"] == Status.ACTIVE.value:
                resp, time = asyncio.run(remote_request(input=input, request_id=i, function='Predict', stub=stub1))
                app1_status[0]["response_time"] = f'{time * 1000:.2f}ms'
                app1_status[0]["request_count"] += 1
                app1_failover_time = 0
                results.append(time)
                predicted_label = get_inference_class(resp, "tin")
                app1_status[0]["service_time"] = f'{resp.service_time * 1000:.2f}ms'
                with open(f"{experiment_dir}/s1_solo_response_time.txt", "a") as f:
                    f.write(f"{app1_status[0]['response_time']}\n")
                with open(f"{experiment_dir}/s1_solo_service_time.txt", "a") as f:
                    f.write(f"{app1_status[0]['service_time']}\n")

            elif app2_status[0]["status"] == Status.READY.value or app2_status[0]["status"] == Status.ACTIVE.value:
                resp, time = asyncio.run(remote_request(input=input, request_id=i, function='Predict', stub=stub2))
                app2_status[0]["response_time"] = f'{time * 1000:.2f}ms'
                app2_status[0]["request_count"] += 1
                app2_failover_time = 0
                results.append(time)
                predicted_label = get_inference_class(resp, "tin")
                app2_status[0]["service_time"] = f'{resp.service_time * 1000:.2f}ms'
                with open(f"{experiment_dir}/s2_solo_response_time.txt", "a") as f:
                    f.write(f"{app2_status[0]['response_time']}\n")
                with open(f"{experiment_dir}/s2_solo_service_time.txt", "a") as f:
                    f.write(f"{app2_status[0]['service_time']}\n")
            continue 
        except Exception as e:
            end_time = timeit.default_timer()
            if app1_failover_time == 0:
                app1_failover_time = end_time - mel_start_time
            if app2_failover_time == 0:
                app2_failover_time = end_time - mel_start_time
            app1_status[0]["response_time"] = None
            app2_status[0]["response_time"] = None
            print(f"Inference backup model error: {e}")
            
        
    if len(results[0]) == 1:
        df = pd.DataFrame(results, columns=["Time"])
    else:
        df = pd.DataFrame(results, columns=["Time1", "Time2"])
    print("Average Time taken for both requests:", np.mean(results, axis=0))
    hostname = socket.gethostname()
    df.to_csv(
        f"system/rpc_results/{hostname.split('.')[0]}_client_{function}_{model_name}_results.csv",
        index=False,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Multi-level Ensemble Learning (MEL) Inference Client")
    parser.add_argument("-hb", "--hb-resp-time", action="store_true", help="Show heartbeat response times for each servers")
    parser.add_argument("-rt", "--refresh-time", type=float, default=1, help="Refresh time for the GUI in seconds")
    parser.add_argument("-st", "--service-time", action="store_true", help="Show the service inference time for the servers")
    return parser.parse_args()

def main():
    args = _parse_args()

    # Start background threads
    mqtt_thread = threading.Thread(target=get_mqtt_message)
    s1_thread = threading.Thread(target=s1_heartbeat)
    s2_thread = threading.Thread(target=s2_heartbeat)
    s12_thread = threading.Thread(target=s12_heartbeat)
    original_thread = threading.Thread(target=original_heartbeat)
    inference_thread = threading.Thread(target=run_inference, args=(config.server1_addr, config.server2_addr, config.server_orig_addr, config.requests, "PredictForward", "model_name"))
    
    # # Busy checking if the servers are alive
    # s1_close_loop_thread = threading.Thread(target=s1_close_loop)
    # s2_close_loop_thread = threading.Thread(target=s2_close_loop)
    # s12_close_loop_thread = threading.Thread(target=s12_close_loop)
    # original_close_loop_thread = threading.Thread(target=original_close_loop)

    add_script_run_ctx(mqtt_thread)
    add_script_run_ctx(s1_thread)
    add_script_run_ctx(s2_thread)
    add_script_run_ctx(s12_thread)
    add_script_run_ctx(original_thread)
    add_script_run_ctx(inference_thread)
    
    # add_script_run_ctx(s1_close_loop_thread)
    # add_script_run_ctx(s2_close_loop_thread)
    # add_script_run_ctx(s12_close_loop_thread)
    # add_script_run_ctx(original_close_loop_thread)

    mqtt_thread.start()
    s1_thread.start()
    s2_thread.start()
    s12_thread.start()
    original_thread.start()
    inference_thread.start()

    # s1_close_loop_thread.start()
    # s2_close_loop_thread.start()
    # s12_close_loop_thread.start()
    # original_close_loop_thread.start()

    st.set_page_config(layout="wide", page_title="MQTT Client Monitor")
    
    # Section 1: Input image and original server side by side
    # st.markdown("## Input Image & Original Server")
    input_col, orig_col = st.columns(2)
    with input_col:
        input_title = st.markdown("### Input")
        input_desc_display = st.markdown(f"**MQTT Topic**: {config.topic}")
        predicted_label_display = st.empty()
        # input_image_display = st.empty()
        # input_status = st.empty()
        raw_image_display = st.empty()
        raw_image_status = st.empty()
        orig_hb_time_display = st.empty()
        s1_hb_time_display = st.empty()
        s2_hb_time_display = st.empty()
        s12_hb_time_display = st.empty()
        # confidence_display = st.empty()
    
    with orig_col:
        orig_title = st.markdown("### Original Server")
        orig_desc_display = st.markdown(f"EfficientNet-B0")
        orig_monitor = st.empty()
        orig_heartbeat_switch_time = st.empty()

    # Section 2: Three server columns for s1, s2, s12
    st.markdown("## MEL Ensemble Servers")
    s1_col, s2_col, s12_col = st.columns(3)
    
    with s1_col:
        s1_title = st.markdown("### S1 Server")
        s1_monitor = st.empty()
        s1_heartbeat_switch_time = st.empty()

    with s2_col:
        s2_title = st.markdown("### S2 Server")
        s2_monitor = st.empty()
        s2_heartbeat_switch_time = st.empty()

    with s12_col:
        s12_title = st.markdown("### S12 Server")
        s12_monitor = st.empty()
        s12_heartbeat_switch_time = st.empty()


    try:
        while True:
            input_desc_display.markdown(f"**MQTT Topic**: {config.topic}")
            predicted_label_display.markdown(f"**Predicted Label**: {predicted_label}")

            if args.hb_resp_time:
                orig_hb_time_display.markdown(f"**Orig heartbeat**: {np.max(heartbeat_times['orig'])}ms")
                s1_hb_time_display.markdown(f"**S1 heartbeat**: {np.max(heartbeat_times['s1'])}ms")
                s2_hb_time_display.markdown(f"**S2 heartbeat**: {np.max(heartbeat_times['s2'])}ms")
                s12_hb_time_display.markdown(f"**S12 heartbeat**: {np.max(heartbeat_times['s12'])}ms")
                
            # Update raw image section
            if raw_image is not None and raw_image.size > 0:
                raw_image_display.image(raw_image, caption="Stream")
            else:
                raw_image_display.markdown("**No stream available**")
                raw_image_status.markdown("**Status**: Waiting for MQTT data...")

            # Update S1 server section
            s1_failure_heartbeat = (
                (app1_failover_status["last_failure_from_heartbeat"] - app1_failover_status["last_online_from_heartbeat"]) * 1000 
                if (app1_failover_status["last_failure_from_heartbeat"] and app1_failover_status["last_online_from_heartbeat"]) 
                else 0
            )
            s1_monitor.dataframe(pd.DataFrame(app1_status), hide_index=True)
            s1_heartbeat_switch_time.markdown(f"""**Failure detection time (heartbeat)**: {s1_failure_heartbeat:.2f}ms""")

            # Update S2 server section
            s2_failure_heartbeat = (
                (app2_failover_status["last_failure_from_heartbeat"] - app2_failover_status["last_online_from_heartbeat"]) * 1000 
                if (app2_failover_status["last_failure_from_heartbeat"] and app2_failover_status["last_online_from_heartbeat"]) 
                else 0
            )
            s2_monitor.dataframe(pd.DataFrame(app2_status), hide_index=True)
            s2_heartbeat_switch_time.markdown(
                f"**Failure detection time (heartbeat)**: {s2_failure_heartbeat:.2f}ms"
            )

            # Update S12 server section
            s12_failure_heartbeat = (
                (app12_failover_status["last_failure_from_heartbeat"] - app12_failover_status["last_online_from_heartbeat"]) * 1000 
                if (app12_failover_status["last_failure_from_heartbeat"] and app12_failover_status["last_online_from_heartbeat"]) 
                else 0
            )
            s12_monitor.dataframe(pd.DataFrame(app12_status), hide_index=True)
            s12_heartbeat_switch_time.markdown(
                f"**Failure detection time (heartbeat)**: {s12_failure_heartbeat:.2f}ms"
            )

            # Update Original server section
            orig_failure_heartbeat = (
                (orig_failover_status["last_failure_from_heartbeat"] - orig_failover_status["last_online_from_heartbeat"]) * 1000 
                if (orig_failover_status["last_failure_from_heartbeat"] and orig_failover_status["last_online_from_heartbeat"]) 
                else 0
            )
            
            orig_monitor.dataframe(pd.DataFrame(orig_status), hide_index=True)
            orig_heartbeat_switch_time.markdown(
                f"**Failure detection time (heartbeat)**: {orig_failure_heartbeat:.2f}ms"
            )
                
            time.sleep(args.refresh_time)  # Update every 2 seconds
            
    except Exception as e:
        st.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    main() 