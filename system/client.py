#!/usr/bin/env python3
"""
    Created date: 9/12/23
"""

import threading
import base64
import io
import os
import requests
import streamlit as st
import numpy as np
import pandas as pd
import json

import time

from PIL import Image
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Convert image bytes to numpy array
def msg_to_np(image_bytes):
    decoded_bytes = base64.b64decode(image_bytes)
    img = Image.open(io.BytesIO(decoded_bytes))
    mat = np.array(img)
    return mat

# Address of server nodes
dvgs_addresses = {
    "Node-1": "http://0.0.0.0:7777",
    "Node-2": "http://0.0.0.0:7777",
    "Node-3": "http://0.0.0.0:7777",
}

addresses = dvgs_addresses

results1 = []
results2 = []
results3 = []

# Permutation of server nodes
app1_addresses = ["Node-1", "Node-2", "Node-3"]
app2_addresses = ["Node-2", "Node-1", "Node-3"]
app3_addresses = ["Node-3", "Node-1", "Node-2"]

# app1_status = []
# for i, address in enumerate(app1_addresses):
#     flag = "PRIMARY" if i == 0 else "BACKUP"
#     app1_status.append(
#         {"address": address, "service_time": None, "count": 0, "flag": flag, "node": address}
#     )

# app2_status = []
# for i, address in enumerate(app2_addresses):
#     flag = "PRIMARY" if i == 0 else "BACKUP"
#     app2_status.append(
#         {"address": address, "service_time": None, "count": 0, "flag": flag, "node": address}
#     )

# app3_status = []
# for i, address in enumerate(app3_addresses):
#     flag = "PRIMARY" if i == 0 else "BACKUP"
#     app3_status.append(
#         {"address": address, "service_time": None, "count": 0, "flag": flag, "node": address}
#     )

# Just black image dummy
image1 = np.zeros([640, 640, 3])
image2 = np.zeros([640, 640, 3])
image3 = np.zeros([640, 640, 3])


# Failoveer metrics
service_time1: float = -1
service_time2: float = -1
service_time3: float = -1

flag1: str
flag2: str
flag3: str

app1_failover_time = 0
app2_failover_time = 0
app3_failover_time = 0

app1_address_idx = 0
app2_address_idx = 0
app3_address_idx = 0

# Create directory for the run results
os.makedirs("exps", exist_ok=True)
run_idx = 0
output_dir = f"exps/run{run_idx}"

while os.path.exists(output_dir):
    run_idx += 1
    output_dir = f"exps/run{run_idx}"

app1_result_dir = os.path.join(output_dir, "app1")
app2_result_dir = os.path.join(output_dir, "app2")
app3_result_dir = os.path.join(output_dir, "app3")

os.makedirs(output_dir)
os.makedirs(app1_result_dir)
os.makedirs(app2_result_dir)
os.makedirs(app3_result_dir)


def detect_failure():
    global app1_failover_time, app2_failover_time, app3_failover_time
    global app1_address_idx, app2_address_idx, app3_address_idx

    while True:
        start_t = time.time()
        for i, host_address in enumerate(app1_addresses):
            try:
                res = requests.get(addresses[host_address] + "/heartbeat")

                # Primary works
                if i == 0:
                    app1_failover_time = 0
                app1_address_idx = i
                break

            except:
                if app1_failover_time == 0:
                    app1_failover_time = time.time() - start_t
                continue

        for i, host_address in enumerate(app2_addresses):
            try:
                res = requests.get(addresses[host_address] + "/heartbeat")

                # Primary works
                if i == 0:
                    app2_failover_time = 0
                app2_address_idx = i
                break
            except:
                if app2_failover_time == 0:
                    app2_failover_time = time.time() - start_t
                continue

        for i, host_address in enumerate(app3_addresses):
            try:
                res = requests.get(addresses[host_address] + "/heartbeat")

                # Primary works
                if i == 0:
                    app3_failover_time = 0
                app3_address_idx = i
                break
            except:
                if app3_failover_time == 0:
                    app3_failover_time = time.time() - start_t
                continue

        time.sleep(1)


def update_image1():
    global image1, app1_status, app1_failover_time, app1_address_idx, results1, app1_result_dir

    img_dir = os.path.join(app1_result_dir, "imgs")
    os.makedirs(img_dir)

    while True:
        log_time = time.time()
        url = addresses[app1_addresses[app1_address_idx]] + "/dvpg1"

        try:
            res = requests.get(url)
        except:
            continue

        if res.status_code == 200:
            res_json = res.json()
            image1 = msg_to_np(res_json["image_bytes"])
            app1_status[app1_address_idx]["service_time"] = res_json["service_time"]
            app1_status[app1_address_idx]["count"] += 1

            save_img_path = os.path.join(img_dir, f"{app1_status[app1_address_idx]['count']:06d}.jpg")
            save_img = Image.fromarray(image1)
            save_img.save(save_img_path, "JPEG")

            results1.append(
                {
                    "time"            : log_time,
                    "app"             : "app1",
                    "flag"            : app1_status[app1_address_idx]["flag"],
                    "service_time"    : res_json["service_time"],
                    "node"            : app1_status[app1_address_idx]["node"],
                    "bboxes"          : res_json["bboxes"],
                    "conf"            : res_json["conf"],
                    "predicted_labels": res_json["predicted_labels"],
                    "image_path"      : save_img_path
                }
            )


def update_image2():
    global image2, app2_status, app2_failover_time, app2_address_idx, results2, app2_result_dir

    img_dir = os.path.join(app2_result_dir, "imgs")
    os.makedirs(img_dir)

    while True:
        log_time = time.time()
        url = addresses[app2_addresses[app2_address_idx]] + "/dvpg2"

        try:
            res = requests.get(url)
        except:
            continue

        if res.status_code == 200:
            res_json = res.json()
            image2 = msg_to_np(res_json["image_bytes"])
            app2_status[app2_address_idx]["service_time"] = res_json["service_time"]
            app2_status[app2_address_idx]["count"] += 1

            save_img_path = os.path.join(img_dir, f"{app2_status[app2_address_idx]['count']:06d}.jpg")
            save_img = Image.fromarray(image2)
            save_img.save(save_img_path, "JPEG")

            results2.append(
                {
                    "time"            : log_time,
                    "app"             : "app2",
                    "flag"            : app2_status[app2_address_idx]["flag"],
                    "service_time"    : res_json["service_time"],
                    "node"            : app2_status[app2_address_idx]["node"],
                    "bboxes"          : res_json["bboxes"],
                    "conf"            : res_json["conf"],
                    "predicted_labels": res_json["predicted_labels"],
                    "image_path"      : save_img_path
                }
            )


def update_image3():
    global image3, app3_status, app3_failover_time, app3_address_idx, results3, app3_result_dir
    img_dir = os.path.join(app3_result_dir, "imgs")
    os.makedirs(img_dir)

    while True:
        url = addresses[app3_addresses[app3_address_idx]] + "/dvpg3"

        try:
            res = requests.get(url)
        except:
            continue

        if res.status_code == 200:
            res_json = res.json()
            image3 = msg_to_np(res_json["image_bytes"])
            app3_status[app3_address_idx]["service_time"] = res_json["service_time"]
            app3_status[app3_address_idx]["count"] += 1

            save_img_path = os.path.join(img_dir, f"{app3_status[app3_address_idx]['count']:06d}.jpg")
            save_img = Image.fromarray(image3)
            save_img.save(save_img_path, "JPEG")

            results3.append(
                {
                    "time"            : log_time,
                    "app"             : "app3",
                    "flag"            : app3_status[app3_address_idx]["flag"],
                    "service_time"    : res_json["service_time"],
                    "node"            : app3_status[app3_address_idx]["node"],
                    "bboxes"          : res_json["bboxes"],
                    "conf"            : res_json["conf"],
                    "predicted_labels": res_json["predicted_labels"],
                    "image_path"      : save_img_path
                }
            )


t1 = threading.Thread(target=update_image1)
t2 = threading.Thread(target=update_image2)
t3 = threading.Thread(target=update_image3)
failure_detection_thread = threading.Thread(target=detect_failure)

add_script_run_ctx(t1)
add_script_run_ctx(t2)
add_script_run_ctx(t3)
add_script_run_ctx(failure_detection_thread)

t1.start()
t2.start()
t3.start()
failure_detection_thread.start()

st.set_page_config(layout="wide")
col1, col2, col3 = st.columns(3)
with col1:
    col1_title = st.markdown("### Sensor 1")
    image_col1 = st.empty()
    app1_monitor = st.empty()
    app1_switch_time = st.empty()

with col2:
    col2_title = st.markdown("### Sensor 2")
    image_col2 = st.empty()
    app2_monitor = st.empty()
    app2_switch_time = st.empty()

with col3:
    col3_title = st.markdown("### Sensor 3")
    image_col3 = st.empty()
    app3_monitor = st.empty()
    app3_switch_time = st.empty()


try:
    while True:
        log_time = time.time()

        image_col1.image(image1)
        app1_monitor.table(pd.DataFrame(app1_status))
        app1_switch_time.markdown(
            f"**Fail detection time**: {app1_failover_time:.6f}ms"
        )
        # current_stat = app1_status[app1_address_idx]
        # results1.append(
        #     {
        #         "time": log_time,
        #         "app": "app1",
        #         "flag": current_stat["flag"],
        #         "service_time": current_stat["service_time"],
        #         "node": current_stat["node"],
        #     }
        # )

        image_col2.image(image2)
        app2_monitor.table(pd.DataFrame(app2_status))
        app2_switch_time.markdown(
            f"**Fail detection time**: {app2_failover_time:.6f}ms"
        )
        # current_stat = app2_status[app2_address_idx]
        # results.append(
        #     {
        #         "time": log_time,
        #         "app": "app2",
        #         "flag": current_stat["flag"],
        #         "service_time": current_stat["service_time"],
        #         "node": current_stat["node"]
        #
        #     }
        # )

        image_col3.image(image3)
        app3_monitor.table(pd.DataFrame(app3_status))
        app3_switch_time.markdown(
            f"**Fail detection time**: {app3_failover_time:.6f}ms"
        )
        # current_stat = app3_status[app3_address_idx]
        # results.append(
        #     {
        #         "time": log_time,
        #         "app": "app3",
        #         "flag": current_stat["flag"],
        #         "service_time": current_stat["service_time"],
        #         "node": current_stat["node"]
        #     }
        # )
finally:

    with open(os.path.join(app1_result_dir, "results.json"), 'w') as f:
        json.dump(results1, f, indent=4)

    with open(os.path.join(app2_result_dir, "results.json"), 'w') as f:
        json.dump(results2, f, indent=4)

    with open(os.path.join(app3_result_dir, "results.json"), 'w') as f:
        json.dump(results3, f, indent=4)
    # result_df = pd.DataFrame(results1)
    # result_df.to_csv(output_file)
