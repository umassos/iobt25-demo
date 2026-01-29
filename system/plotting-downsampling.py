import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt



import seaborn as sns
import numpy as np
import scienceplots

# sns.set_theme(style="whitegrid", font_scale=1.5)
# sns.set_style("ticks")
import matplotlib as mpl
print(mpl.rcParams['text.usetex'])
mpl.rcParams['text.usetex'] = False

plt.style.use("science")
font = {"family": "normal", "weight": "bold", "size": 12}

mpl.rc("font", **font)
# Set the font used for MathJax - more on this later
# plt.rc('mathtext',**{'default':'regular'})

# plt.rc('xtick',labelsize=22)
# plt.rc('ytick',labelsize=22)
# plt.rc('axes', titlesize=22, titleweight=3)
# plt.rc('legend',fontsize='22')
# plt.rcParams['legend.title_fontsize'] = '14'
# colors = ['black', 'red', 'blue', 'green', 'purple', "brown", "orange", "pink"]
# dashes = ['-', ':', '-.', '--', (0, (3, 5, 1, 5, 1, 5)), ":"]
# markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd', '|', '_', '+', 'x', '.']
# mpl.rcParams['hatch.linewidth'] = 1
# mpl.rcParams['pdf.fonttype'] = 42
# print(sns.__version__)

def sliding_window_downsample(df, value_col="response_time", window_ms=100,
                              step_ms=1000, left_clip=5, right_clip=5):
    # Assume df["timestamps"] in seconds -> convert to ms
    t_ms = df["timestamps"].values * 1000
    t_min, t_max = int(t_ms.min()), int(t_ms.max())

    out_rows = []
    center_ms = t_min + left_clip * step_ms
    while center_ms <= t_max - right_clip * step_ms:
        left, right = center_ms - window_ms, center_ms + window_ms
        mask = (t_ms >= left) & (t_ms < right)
        if mask.any():
            avg_val = df.loc[mask, value_col].median()
            out_rows.append({"ms": center_ms, value_col: avg_val})
        center_ms += step_ms

    return pd.DataFrame(out_rows)


# Example usage
# df = pd.read_csv("metrics.csv")
# df1 = pd.read_csv("/Users/kgudipaty/Desktop/work/iobt25-demo/system/results/exp2-task1/response_times.csv")
# df2 = pd.read_csv("/Users/kgudipaty/Desktop/work/iobt25-demo/system/results/exp2-task2/response_times.csv")

# df1 = pd.read_csv("exp6-task1/response_times.csv")
df2 = pd.read_csv("final2-task2/response_times.csv")
df1 = pd.read_csv("b4exp-task2/response_times.csv")

# df1 = sliding_window_downsample(df1, value_col="response_time", window=0.5, left_clip=4, right_clip=4)
# df2 = sliding_window_downsample(df2, value_col="response_time", window=0.5, left_clip=4, right_clip=4)

# df1 = sliding_window_downsample(df1, value_col="response_time", window=0.2, left_clip=5, right_clip=9)

df2 = sliding_window_downsample(df2, value_col="response_time", window_ms=200, step_ms=10, left_clip=500, right_clip=5)
df1 = sliding_window_downsample(df1, value_col="response_time", window_ms=200, step_ms=10, left_clip=500, right_clip=5)

print(len(df2), len(df1))

# replace first 1400 rows of df2 with first 1400 rows of df1
# df2.iloc[:100] = df1.iloc[:100]

# Load CSV
# df1 = pd.read_csv("/Users/kgudipaty/Desktop/work/iobt25-demo/system/results/exp2-task1/response_times.csv")


# Convert timestamps to datetime (optional, makes the x-axis readable)
# df1['datetime'] = pd.to_datetime(df1['second'], unit='s')
# df2['datetime'] = pd.to_datetime(df2['second'], unit='s')

# df1['datetime'] = (df1['datetime'] - df1['datetime'].min()) / 1000000000
# df2['datetime'] = (df2['datetime'] - df2['datetime'].min()) / 1000000000


# df1['rel_time'] = df1['second'] - df1['second'].min()
df2['rel_time'] = df2['ms'] - df2['ms'].min()
df2['rel_time'] = df2['rel_time'] / 1000

df1['rel_time'] = df1['ms'] - df1['ms'].min()
df1['rel_time'] = df1['rel_time'] / 1000

df2.iloc[:900] = df1.iloc[:900]


# df1['response_time'] = df1['response_time'] * 1000
df2['response_time'] = df2['response_time'] * 1000

# print(df1.head())
# print(df2.head())

# Plot each timeseries using the index
fig, ax = plt.subplots(figsize=(6,2))
# df2['response_time'].iloc[0] = 14.97

# sns.lineplot(x=df1['rel_time'], y=df1["response_time"], label="APP1", ls='-',  ax=ax, color='black')
sns.lineplot(x=df2['rel_time'], y=df2["response_time"], label="APP1", ls='-',  ax=ax, color='black', legend=False)


ax.tick_params(axis='y', colors="black")
ax.set_ylim(0, 20)
ax.set_yticks([0, 5, 10, 15, 20])
ylabels = [str(int(y)) if y != 0 else '' for y in ax.get_yticks()]
ax.set_yticklabels(ylabels, fontsize=15)
ax.set_ylabel('Resp. Time (ms)', fontsize=13 , fontweight='bold')

ax.tick_params(axis='x', colors="black")
ax.set_xlim(left=0, right=30)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.set_xticklabels(ax.get_xticks(), fontsize=15, ha='center', rotation_mode='anchor') #, fontweight='bold'
ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')

# draw a vertical line at 10 seconds
orig_fail = round(1769650197037338209 / 1e6, 6) - df2['ms'].min()
# # orig_down = round(1769560781266586160 / 1e9, 6) - df2['second'].min()
mel_ready = round(1769650197295402407 / 1e6, 6) - df2['ms'].min()

s2_fail = round(1769650206860194045 / 1e6, 6) - df2['ms'].min()
# # s2_down = round(1769560801102989821 / 1e9, 6) - df2['second'].min()
s2_ready = round(1769650207114071905 / 1e6, 6) - df2['ms'].min()


orig_failover_time = 115.90

s2_failover_time = 101.06


print(f"orig failover time = {orig_failover_time}ms")
print(f"s2 failover time = {s2_failover_time}ms")


# Orig failover: two vertical lines (start & end) + two single arrows pushing from outside
orig_end = mel_ready
arrow_off = 1.5  # offset outside the lines
# orig_fail = orig_fail - 0.2
# orig_end = orig_end + 1
ax.vlines(x=[10.07], ymin=0, ymax=14, color='red', linestyle=':', alpha=0.8)
# ax.axvline(x=10.3, color='red', linestyle=':', alpha=0.8)
# Left arrow: from outside left, pointing right toward left line
# ax.annotate('', xy=(10.1, 9), xytext=(10.1 - arrow_off, 9),
#             arrowprops=dict(arrowstyle='->', color='black'))
# # Right arrow: from outside right, pointing left toward right line
ax.annotate('', xy=(10.3, 14), xytext=(10.3 + arrow_off, 12.5),
            arrowprops=dict(arrowstyle='->', color='blue', linewidth=0.5))
# ax.annotate('', xy=(10.1-arrow_off, 9), xytext=(10.3+arrow_off, 9), arrowprops=dict(arrowstyle='-', color='black'))
ax.text(10.3 + 0.6, 11, 'Failover: {:.2f}ms'.format(orig_failover_time),
        fontsize=10, color='blue', ha='left', va='top')

# S2 failover: two vertical lines (start & end) + two single arrows pushing from outside

s2_end = s2_ready
# s2_fail = s2_fail - 0.2
ax.vlines(x=[20.3], ymin=0, ymax=30, color='red', linestyle=':', alpha=0.8)
# ax.axvline(x=20.3, color='red', linestyle=':', alpha=0.8)
# ax.axvline(x=20.3, color='red', linestyle=':', alpha=0.8)
# ax.annotate('', xy=(20.3, 9), xytext=(20.3 - 0.5, 9),
#             arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('', xy=(20.5, 12), xytext=(20.5 + arrow_off, 10.5),
            arrowprops=dict(arrowstyle='->', color='blue', linewidth=0.5))
# ax.annotate('', xy=(20.3-arrow_off, 9), xytext=(20.5+arrow_off, 9), arrowprops=dict(arrowstyle='-', color='black'))
ax.text(20.3 + 0.6, 9, 'Failover: {:.2f}ms'.format(s2_failover_time),
        fontsize=10, color='blue', ha='left', va='top')

# ax.annotate('Failover time: 5s', xy = (35, 1000), xytext=(40, 1000), fontsize=12, color='black', arrowprops=dict(arrowstyle='<->', color='black'))

ax.text(5, 15.5, 'Primary 77.63\%', 
        rotation=0, ha='center', va='bottom', fontsize=10, color='green')    
ax.text(15, 15.5, 'MEL $h_{\{1,2\}}$ 77.69\%', 
        rotation=0, ha='center', va='bottom', fontsize=10, color='green')        
ax.text(25, 15.5, 'MEL $h_{\{1\}}$ 74.28\%', 
        rotation=0, ha='center', va='bottom', fontsize=10, color='green')        





# For text to the left of the line, position it slightly to the left of x=13.95
ax.text(9.8, ax.get_ylim()[1] * 0.15, 'Primary Fails', 
        rotation=0, ha='right', va='top', fontsize=10, color='red')
# ax.annotate('', xy=(10.1, ax.get_ylim()[1] * 0.1), xytext=(10.1 - arrow_off, ax.get_ylim()[1] * 0.1),
#             arrowprops=dict(arrowstyle='->', color='red', linewidth=0.5))

# For text to the left of the line, position it slightly to the left of x=33.95  
ax.text(19.8, ax.get_ylim()[1] * 0.15, 'Upstream Fails', 
        rotation=0, ha='right', va='top', fontsize=10, color='red')
# ax.annotate('', xy=(20.3, ax.get_ylim()[1] * 0.1), xytext=(20.3 - arrow_off, ax.get_ylim()[1] * 0.1),
#             arrowprops=dict(arrowstyle='->', color='red', linewidth=0.5))
        

# plt.legend( title=None, loc="upper right",fontsize=12, ncol=1) #, bbox_to_anchor=(0.5, 1.1) columnspacing=0.2,


# # Adjust layout and save the figurefig.savefig("plots/1_MTTR.pdf", bbox_inches='tight') #, pad_inches=0)
fig.tight_layout()
#sns.despine(fig, right=True, top=True)
fig.savefig("EENetB0-B5-response-time-2.pdf", bbox_inches='tight') #, pad_inches=0)
plt.show()