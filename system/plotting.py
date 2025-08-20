import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df1 = pd.read_csv("/Users/kgudipaty/Desktop/work/iobt25-demo/system/results/2025-08-20_11:45:18/response_times.csv")
df2 = pd.read_csv("/Users/kgudipaty/Desktop/work/iobt25-demo/system/results/2025-08-20_11:45:19/response_times.csv")

# Convert timestamps to datetime (optional, makes the x-axis readable)
df1['datetime'] = pd.to_datetime(df1['timestamps'], unit='s')
df2['datetime'] = pd.to_datetime(df2['timestamps'], unit='s')

# Plot response_time vs time
plt.figure(figsize=(10, 5))
plt.plot(df1['datetime'], df1['response_time'], marker='o', linestyle='-', label='Task 2')
plt.plot(df2['datetime'], df2['response_time'], marker='o', linestyle='-', label='Task 1')
plt.xlabel("Time")
plt.ylabel("Service Time (s)")
plt.ylim(0, 2)
plt.title("Response Time over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
