#!/usr/bin/env python3
"""
    Created date: 9/11/23
"""

import base64
import io
import time
import threading
import numpy as np

import paho.mqtt.client as mqtt

TOPICS = ["/dvpg_gq_orin_1/zed/rgb_left/compressed",
          "/dvpg_gq_orin_2/zed/rgb_left/compressed",
          "/dvpg_gq_orin_4/zed/rgb_left/compressed"
          ]

SERVER_IP = "192.168.70.51"
SERVER_PORT = 1883


class MQTTSource(threading.Thread):
    """ Fetch input using MQTT """
    def __init__(self):
        super().__init__()
        self._frame_count = {}
        self._frames = {}
        self._is_stop = True

        self._client = mqtt.Client()
        self._client.on_connect = self._on_mqtt_connect
        self._client.on_message = self._on_mqtt_message
        self._client.connect(SERVER_IP, SERVER_PORT, keepalive=60)

        for topic in TOPICS:
            self._frame_count[topic] = 0
            self._frames[topic] = None

    def _on_mqtt_connect(self, client: mqtt.Client, userdata, flags, rc):
        """ Callback function on MQTT connect """
        print("Connected to MQTT result code " + str(rc))
        # client.subscribe([
        #     ("/dvpg_gq_1/zed/rgb_left/compressed", 0),
        #     ("/dvpg_gq_2/zed/rgb_left/compressed", 0),
        #     ("/dvpg_gq_3/zed/rgb_left/compressed", 0)
        # ])

        client.subscribe([
            ("/dvpg_gq_orin_1/zed/rgb_left/compressed", 0),
            ("/dvpg_gq_orin_2/zed/rgb_left/compressed", 0),
            ("/dvpg_gq_orin_4/zed/rgb_left/compressed", 0)
        ])

    def _on_mqtt_message(self, client: mqtt.Client, userdata, msg):
        """ Callback function when receiving MQTT message """
        # if msg.topic in self._frames:
        self._frames[msg.topic] = msg
        self._frame_count[msg.topic] += 1

        print(f"Got frame {msg.topic} #{self._frame_count[msg.topic]}")

    def run(self):
        """ Start listening messages """
        self._is_stop = False

        while not self._is_stop:
            self._client.loop()
            time.sleep(0.01)

    def stop(self):
        """ Stop fetching data """
        self._is_stop = True

    def get_message(self, source: str):
        """ Get message by source name """
        if source not in self._frames:
            raise ValueError(f"Unknown source {source}.")

        source_message = self._frames[source]
        source_num = self._frame_count[source]
        # if not source_message:
        #     raise RuntimeError(f"Source {source} is not available.")

        return source_num, source_message
