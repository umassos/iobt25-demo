import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", font_scale=1.5)
sns.set_style("ticks")
plt.rcParams["font.family"] = "Times New Roman"
# Set the font used for MathJax - more on this later
plt.rc('mathtext',**{'default':'regular'})

plt.rc('xtick',labelsize=22)
plt.rc('ytick',labelsize=22)
plt.rc('axes', titlesize=22, titleweight=3)
plt.rc('legend',fontsize='22')
plt.rcParams['legend.title_fontsize'] = '14'
colors = ['black', 'red', 'blue', 'green', 'purple', "brown", "orange", "pink"]
dashes = ['-', ':', '-.', '--', (0, (3, 5, 1, 5, 1, 5)), ":"]
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd', '|', '_', '+', 'x', '.']
mpl.rcParams['hatch.linewidth'] = 1
mpl.rcParams['pdf.fonttype'] = 42
print(sns.__version__)

def sliding_window_downsample(df, value_col="response_time", window=0.5, left_clip=5, right_clip=5):
    """
    Downsample a time series to 1Hz using sliding windows.
    
    Args:
        df (pd.DataFrame): Must have 'timestamps' (unix seconds float) and a value column.
        value_col (str): Column to aggregate (e.g., response_time).
        window (float): Half-window size in seconds (0.5 => ±0.5 sec).
        clip (int): Seconds to trim at start and end.

    Returns:
        pd.DataFrame with one data point per second.
    """
    # Convert timestamp float to datetime
    df["datetime"] = pd.to_datetime(df["timestamps"], unit="s")

    # Define integer second bins
    df["second"] = df["timestamps"].astype(int)

    # Aggregate using sliding window
    out_rows = []
    for sec in range(df["second"].min(), df["second"].max() + 1):
        center_time = sec + 0.0  # current second
        left, right = center_time - window, center_time + window
        mask = (df["timestamps"] >= left) & (df["timestamps"] < right)
        if mask.any():
            avg_val = df.loc[mask, value_col].mean()
            out_rows.append({"second": sec, value_col: avg_val})

    out_df = pd.DataFrame(out_rows)

    # Clip off first/last `clip` seconds
    clipped = out_df.iloc[left_clip:-right_clip].reset_index(drop=True)
    return clipped


# Example usage
# df = pd.read_csv("metrics.csv")
# df1 = pd.read_csv("/Users/kgudipaty/Desktop/work/iobt25-demo/system/results/exp2-task1/response_times.csv")
# df2 = pd.read_csv("/Users/kgudipaty/Desktop/work/iobt25-demo/system/results/exp2-task2/response_times.csv")

df1 = pd.read_csv("/Users/kgudipaty/Desktop/work/iobt25-demo/system/results/exp2-task1/response_times.csv")
df2 = pd.read_csv("/Users/kgudipaty/Desktop/work/iobt25-demo/system/results/exp2-task2/response_times.csv")

# df1 = sliding_window_downsample(df1, value_col="response_time", window=0.5, left_clip=4, right_clip=4)
# df2 = sliding_window_downsample(df2, value_col="response_time", window=0.5, left_clip=4, right_clip=4)

df1 = sliding_window_downsample(df1, value_col="response_time", window=0.5, left_clip=4, right_clip=4)
df2 = sliding_window_downsample(df2, value_col="response_time", window=0.5, left_clip=4, right_clip=4)




# Load CSV
# df1 = pd.read_csv("/Users/kgudipaty/Desktop/work/iobt25-demo/system/results/exp2-task1/response_times.csv")


# Convert timestamps to datetime (optional, makes the x-axis readable)
# df1['datetime'] = pd.to_datetime(df1['second'], unit='s')
# df2['datetime'] = pd.to_datetime(df2['second'], unit='s')

# df1['datetime'] = (df1['datetime'] - df1['datetime'].min()) / 1000000000
# df2['datetime'] = (df2['datetime'] - df2['datetime'].min()) / 1000000000


df1['rel_time'] = df1['second'] - df1['second'].min()
df2['rel_time'] = df2['second'] - df2['second'].min()


df1['response_time'] = df1['response_time'] * 1000
df2['response_time'] = df2['response_time'] * 1000

print(df1.head())
print(df2.head())

# Plot each timeseries using the index
fig, ax = plt.subplots(figsize=(10,5))

sns.lineplot(x=df1['rel_time'], y=df1["response_time"], label="APP1", ls='-',  ax=ax, lw=2, color='black')
sns.lineplot(x=df2['rel_time'], y=df2["response_time"], label="APP2", ls='--',  ax=ax, lw=2, color='red')

ax.tick_params(axis='x', colors="black")
# ax.set_xlim(0, 60)
# ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xticklabels(ax.get_xticks(), fontsize=18, ha='center', rotation_mode='anchor') #, fontweight='bold'
ax.tick_params(axis='y', colors="black")
ax.set_ylabel('Response Time (ms)', fontsize=30 , fontweight='bold')
ax.set_xlabel('Time (s)', fontsize=30, fontweight='bold')

# # ax.set_yscale('log')
# ax.set_ylim(0, 25)
# ax.set_yticks([0, 5, 10, 15, 20, 25])


ax.set_ylim(0, 1500)
ax.set_yticks([0, 500, 1000, 1500])

ax.annotate('', xy = (15.15, 1000), xytext=(22, 1000), fontsize=12, color='black', arrowprops=dict(arrowstyle='<->', color='black'))
ax.text(15.15, 900, 'Failover time: 6s', fontsize=12, color='black')
ax.annotate('Failover time: 5s', xy = (35, 1000), xytext=(40, 1000), fontsize=12, color='black', arrowprops=dict(arrowstyle='<->', color='black'))

ax.set_yticklabels(ax.get_yticks(), fontsize=18) #, fontweight='bold'

# draw a vertical line at 10 seconds
ax.axvline(x=15, color='blue', linestyle='-.')
ax.axvline(x=35, color='green', linestyle='-.')

# For text to the left of the line, position it slightly to the left of x=13.95
ax.text(14, ax.get_ylim()[1] * 0.95, 'Original Fails', 
        rotation=0, ha='right', va='top', fontsize=18, color='blue')

# For text to the left of the line, position it slightly to the left of x=33.95  
ax.text(33, ax.get_ylim()[1] * 0.95, 'Upstream Fails', 
        rotation=0, ha='right', va='top', fontsize=18, color='green')

plt.legend( title=None, loc="upper right",   fontsize=14, ncol=1) #, bbox_to_anchor=(0.5, 1.1) columnspacing=0.2,


# # Adjust layout and save the figurefig.savefig("plots/1_MTTR.pdf", bbox_inches='tight') #, pad_inches=0)
fig.tight_layout()
sns.despine(fig, right=True, top=True)
fig.savefig("system/results/iobt_mel_response_time.png", bbox_inches='tight') #, pad_inches=0)
plt.show()