import os

topic = "/dvpg_gq_orin_1/zed/rgb_left/compressed"


# # Jetson Orin Node 10 DVPG
# server1_addr = "192.168.76.12:8181"
# server2_addr = "192.168.75.12:8182"
# server12_addr = "192.168.79.12:8180"
# server_orig_addr = "192.168.77.12:8190"
# server_task2 = "192.168.77.12:8190"

# IoBT 2026 experiment nodes (overridable via env vars)
server_orig_addr = os.environ.get("ORIGINAL", "192.168.85.12:8190")


server1_addr = os.environ.get("SERVER1", "192.168.82.12:8181")
server2_addr = os.environ.get("SERVER2", "192.168.84.12:8182")
server12_addr = os.environ.get("HEAD", "192.168.81.12:8180")
# server_task2 = os.environ.get("ORIGINAL", "192.168.85.12:8190")

# # Orin2 (overridable via env vars)
# server1_addr = os.environ.get("SERVER1", "127.0.0.1:8181")
# server2_addr = os.environ.get("SERVER2", "127.0.0.1:8182")
# server12_addr = os.environ.get("HEAD", "127.0.0.1:8180")
# server_orig_addr = os.environ.get("ORIGINAL", "127.0.0.1:8190")
# server_task2 = os.environ.get("ORIGINAL", "127.0.0.1:8190")

# Local testing
# server1_addr = "localhost:8181"
# server2_addr = "localhost:8182"
# server12_addr = "localhost:8180"
# server_orig_addr = "localhost:8190"

# # Jetson Orin LASS
# server1_addr = "192.168.0.175:8181"
# server2_addr = "192.168.0.175:8182"
# server12_addr = "192.168.0.175:8180"
# server_orig_addr = "192.168.0.175:8190"

# heartbeat interval in milliseconds
heartbeat_interval = 10
requests = 1000