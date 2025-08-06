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
    img = np.ascontiguousarray(img[...,[2, 1, 0]])

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

# Global variables (similar to iobt23 approach)
mqtt_client = mqtt_source.MQTTSource()
mqtt_client.start()

topic = "/dvpg_gq_orin_1/zed/rgb_left/compressed"

source_num = 0
source_message = None
input_image = np.zeros([1, 3, 224, 224])

# Server results
image1 = np.zeros([640, 640, 3])
image2 = np.zeros([640, 640, 3])
image3 = np.zeros([640, 640, 3])

# Status tracking
app1_status = [{"service_time": None, "count": 0, "flag": "PRIMARY", "node": "S1", "status": "Initializing...", "last_heartbeat": 0, "failure_time": 0}]
app2_status = [{"service_time": None, "count": 0, "flag": "PRIMARY", "node": "S2", "status": "Initializing...", "last_heartbeat": 0, "failure_time": 0}]
app3_status = [{"service_time": None, "count": 0, "flag": "PRIMARY", "node": "S12", "status": "Initializing...", "last_heartbeat": 0, "failure_time": 0}]
orig_status = [{"service_time": None, "count": 0, "flag": "PRIMARY", "node": "Original", "status": "Initializing...", "last_heartbeat": 0, "failure_time": 0}]

orig_failover_time = 0
app1_failover_time = 0
app2_failover_time = 0
app3_failover_time = 0

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
    global source_num, source_message, input_image
    while True:
        try:
            source_num, source_message = mqtt_client.get_message(topic)
            raw_image = bytes_to_np(msg_to_bytes(source_message))
            input_image = convert_to_model_input(raw_image)
            time.sleep(0.1)
        except Exception as e:
            print(f"Error getting MQTT message: {e}, trying again in 1 second...")
            time.sleep(1)

def update_s1_results():
    global image1, app1_status, app1_failover_time
    # Setup gRPC connection
    channel1 = grpc.insecure_channel("localhost:8180")
    stub1 = EncoderServiceStub(channel1)
    
    while True:
        try:
            # Check S1 server heartbeat first
            init_time = time.time()
            try:
                heartbeat_request = HeartbeatRequest()
                heartbeat_response = stub1.Heartbeat(heartbeat_request)
                app1_status[0]["status"] = "✅ Alive"
                heartbeat_time = time.time() - init_time
                app1_status[0]["last_heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + f" ({heartbeat_time:.2f}s)"
            except Exception as heartbeat_error:
                app1_status[0]["status"] = "❌ Dead"
                # app1_status[0]["last_heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                app2_failover_time = time.time() - init_time
                print(f"S1 Heartbeat error: {heartbeat_error}")
                time.sleep(1)
                continue
            
            # Use the current input_image for inference
            input_data = input_image.astype(np.float32)
            # Run async function in sync context
            result = asyncio.run(remote_request(input=input_data, request_id=source_num, function="PredictForward", stub=stub1))
            
            # Update status
            app1_status[0]["service_time"] = result
            app1_status[0]["count"] += 1
            
            # For now, just use a placeholder image
            image1 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
        except Exception as e:
            print(f"Error updating S1: {e}")
        time.sleep(1)

def update_s2_results():
    global image2, app2_status, app2_failover_time
    # Setup gRPC connection
    channel2 = grpc.insecure_channel("localhost:8181")
    stub2 = EncoderServiceStub(channel2)
    
    while True:
        try:
            # Check S2 server heartbeat first
            init_time = time.time()
            try:
                heartbeat_request = HeartbeatRequest()
                heartbeat_response = stub2.Heartbeat(heartbeat_request)
                app2_status[0]["status"] = "✅ Alive"
                heartbeat_time = time.time() - init_time
                app2_status[0]["last_heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + f" ({heartbeat_time:.2f}s)"
            except Exception as heartbeat_error:
                app2_status[0]["status"] = "❌ Dead"
                # app2_status[0]["last_heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + f" ({heartbeat_time:.2f}s)"
                app2_failover_time = time.time() - init_time
                print(f"S2 Heartbeat error: {heartbeat_error}")
                time.sleep(1)
                continue
            
            # Use the current input_image for inference
            input_data = input_image.astype(np.float32)
            # Run async function in sync context
            result = asyncio.run(remote_request(input=input_data, request_id=source_num, function="PredictForward", stub=stub2))
            
            # Update status
            app2_status[0]["service_time"] = result
            app2_status[0]["count"] += 1
            
            # For now, just use a placeholder image
            image2 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
        except Exception as e:
            print(f"Error updating S2: {e}")
        time.sleep(1)

def update_original_results():
    global orig_status, orig_failover_time
    # Setup gRPC connection for original server
    channel_original = grpc.insecure_channel("localhost:8183")
    stub_original = EncoderServiceStub(channel_original)
    
    while True:
        try:
            # Check Original server heartbeat first
            init_time = time.time()
            try:
                heartbeat_request = HeartbeatRequest()
                heartbeat_response = stub_original.Heartbeat(heartbeat_request)
                orig_status[0]["status"] = "✅ Alive"
                heartbeat_time = time.time() - init_time
                orig_status[0]["last_heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + f" ({heartbeat_time:.2f}s)"
            except Exception as heartbeat_error:
                orig_status[0]["status"] = "❌ Dead"
                # orig_status[0]["last_heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                orig_failover_time = time.time() - init_time
                print(f"Original Heartbeat error: {heartbeat_error}")
                time.sleep(1)
                continue
            
            # Use the current input_image for inference
            input_data = input_image.astype(np.float32)
            # Run async function in sync context
            result = asyncio.run(remote_request(input=input_data, request_id=source_num, function="PredictOriginal", stub=stub_original))
            
            # Update status
            orig_status[0]["service_time"] = result
            orig_status[0]["count"] += 1
            
            # For now, just use a placeholder image (could be the actual inference result)
            # image_original = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
        except Exception as e:
            print(f"Error updating Original: {e}")
        time.sleep(1)


def run_inference(server1, server2, server_original, requests, function, model_name):
    global image1, image2, image3, app1_status, app2_status, app3_status, app1_failover_time, app2_failover_time, app3_failover_time
    # Setup gRPC connection
    channel1 = grpc.insecure_channel(server1)
    channel2 = grpc.insecure_channel(server2)
    channel_original = grpc.insecure_channel(server_original)
    
    stub1 = EncoderServiceStub(channel1)
    stub2 = EncoderServiceStub(channel2)
    stub_original = EncoderServiceStub(channel_original)

    results = []

    input = input_image.astype(np.float32)
    for i in range(requests):
        gather_time = timeit.default_timer()
        try: 
            time = asyncio.run(remote_request(input=input, request_id=i, function='PredictOriginal', stub=stub_original))
            results.append(time)
        except Exception as e:
            print(f"Original model error: {e}")
            
        # if function == "PredictForward":
            times = asyncio.gather(
                remote_request(
                    input=input, request_id=i, function=function, stub=stub1
                ),
                remote_request(
                    input=input, request_id=i, function=function, stub=stub2
                ),
            )
        # else:
        #     time = remote_request(
        #             input=input, request_id=i, function=function, stub=stub1
        #         )
            # times = [time]
            gather_time = timeit.default_timer() - gather_time
            # logger.info(f"All Gather time: {gather_time:.4f} seconds")
            results.append(times)
        
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

def update_s12_results():
    global image3, app3_status, app3_failover_time
    # Setup gRPC connection for ensemble
    channel12 = grpc.insecure_channel("localhost:8180")  # Adjust as needed
    stub12 = EncoderServiceStub(channel12)
    
    # Setup gRPC connection for head server heartbeat
    head_channel = grpc.insecure_channel("localhost:8185")
    head_stub = HeadServiceStub(head_channel)
    
    while True:
        try:
            # Check head server heartbeat first
            init_time = time.time()
            try:
                heartbeat_request = HeartbeatRequest()
                heartbeat_response = head_stub.Heartbeat(heartbeat_request)
                app3_status[0]["status"] = "✅ Alive"
                heartbeat_time = time.time() - init_time
                app3_status[0]["last_heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + f" ({heartbeat_time:.2f}s)"
            except Exception as heartbeat_error:
                app3_status[0]["status"] = "❌ Dead"
                # app3_status[0]["last_heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                app3_failover_time = time.time() - init_time
                print(f"Heartbeat error: {heartbeat_error}")
                time.sleep(1)
                continue
            
            # Use the current input_image for inference
            input_data = input_image.astype(np.float32)
            # Run async function in sync context
            result = asyncio.run(remote_request(input=input_data, request_id=source_num, function="PredictForward", stub=stub12))
            
            # Update status
            app3_status[0]["service_time"] = result
            app3_status[0]["count"] += 1
            
            # For now, just use a placeholder image
            image3 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
        except Exception as e:
            print(f"Error updating S12: {e}")
        time.sleep(1)

def main():
    # Start background threads (like iobt23 approach)
    mqtt_thread = threading.Thread(target=get_mqtt_message)
    s1_thread = threading.Thread(target=update_s1_results)
    s2_thread = threading.Thread(target=update_s2_results)
    s12_thread = threading.Thread(target=update_s12_results)
    original_thread = threading.Thread(target=update_original_results)
    inference_thread = threading.Thread(target=run_inference, args=("localhost:8180", "localhost:8181", "localhost:8183", 100, "PredictForward", "model_name"))

    add_script_run_ctx(mqtt_thread)
    add_script_run_ctx(s1_thread)
    add_script_run_ctx(s2_thread)
    add_script_run_ctx(s12_thread)
    add_script_run_ctx(original_thread)
    add_script_run_ctx(inference_thread)

    mqtt_thread.start()
    s1_thread.start()
    s2_thread.start()
    s12_thread.start()
    original_thread.start()
    inference_thread.start()

    st.set_page_config(layout="wide", page_title="MQTT Client Monitor")
    
    # Section 1: Input image and original server side by side
    st.markdown("## Input Image & Original Server")
    input_col, orig_col_top = st.columns(2)
    with input_col:
        input_title = st.markdown("### Original Input")
        input_image_display = st.empty()
        input_status = st.empty()
    
    with orig_col_top:
        orig_title_top = st.markdown("### Original Server")
        orig_monitor_top = st.empty()
        orig_switch_time_top = st.empty()
    
    # Section 2: Three server columns for s1, s2, s12
    st.markdown("## MEL Servers (Backup)")
    s1_col, s2_col, s12_col = st.columns(3)
    
    with s1_col:
        s1_title = st.markdown("### S1 Server")
        # s1_image = st.empty()  # Commented out - no image display
        s1_monitor = st.empty()
        s1_switch_time = st.empty()

    with s2_col:
        s2_title = st.markdown("### S2 Server")
        # s2_image = st.empty()  # Commented out - no image display
        s2_monitor = st.empty()
        s2_switch_time = st.empty()

    with s12_col:
        s12_title = st.markdown("### S12 Server")
        # s12_image = st.empty()  # Commented out - no image display
        s12_monitor = st.empty()
        s12_switch_time = st.empty()



    try:
        while True:
            # Update input image section
            if input_image is not None and input_image.size > 0:
                # Convert from [1, 3, 224, 224] to [224, 224, 3] for display
                if len(input_image.shape) == 4:
                    display_img = np.transpose(input_image[0], (1, 2, 0))
                    # Ensure proper range and data type for display
                    if display_img.max() <= 1.0:
                        # If normalized (0-1), convert to 0-255
                        display_img = (display_img * 255).astype(np.uint8)
                    else:
                        # If already in 0-255 range, just convert to uint8
                        display_img = display_img.astype(np.uint8)
                    
                    input_image_display.image(display_img, caption="Input Image")
                input_status.markdown(f"**Input shape**: {input_image.shape}")
            else:
                input_image_display.markdown("**No input image available**")
                input_status.markdown("**Status**: Waiting for MQTT data...")

            # Update S1 server section
            # if image1 is not None and image1.size > 0:
            #     s1_image.image(image1, caption="S1 Result")
            s1_monitor.table(pd.DataFrame(app1_status))
            s1_switch_time.markdown(
                f"**Fail detection time**: {app1_failover_time:.6f}ms"
            )
            # else:
            #     s1_image.markdown("**No S1 result available**")

            # Update S2 server section
            # if image2 is not None and image2.size > 0:
            #     s2_image.image(image2, caption="S2 Result")
            s2_monitor.table(pd.DataFrame(app2_status))
            s2_switch_time.markdown(
                f"**Fail detection time**: {app2_failover_time:.6f}ms"
            )
            # else:
            #     s2_image.markdown("**No S2 result available**")

            # Update S12 server section
            # if image3 is not None and image3.size > 0:
            #     s12_image.image(image3, caption="S12 Result")
            s12_monitor.table(pd.DataFrame(app3_status))
            s12_switch_time.markdown(
                f"**Fail detection time**: {app3_failover_time:.6f}ms"
            )
            # else:
            #     s12_image.markdown("**No S12 result available**")

            # Update Original server section (top right)
            orig_monitor_top.table(pd.DataFrame(orig_status))
            orig_switch_time_top.markdown(
                f"**Fail detection time**: {orig_failover_time:.6f}ms"
            )
                
            time.sleep(2)  # Update every 2 seconds
            
    except Exception as e:
        st.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    main() 