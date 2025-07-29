import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01.processed.pkl")
df = df[df["label"] != "rest"]



# Compute acceleration magnitude
df["acc_r"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)

# Compute gyroscope magnitude
df["gyr_r"] = np.sqrt(df["gyr_x"]**2 + df["gyr_y"]**2 + df["gyr_z"]**2)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
ohp_df   = df[df["label"] == "ohp"]
dead_df  = df[df["label"] == "dead"]
row_df  = df[df["label"] == "row"]


# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = bench_df

# can count reps using peaks
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot() 
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot() 
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot() 
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()
 
 
#  very disoriented i.e cant tell no. of reps
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot() 
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot() 
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot() 
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot() 

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000 / 200
LowPass = LowPassFilter()


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------


bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set   = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set   = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set  = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]


bench_set["acc_r"].plot()


column = "acc_r"
LowPass.low_pass_filter(squat_set,col=column , sampling_frequency=fs , 
                        cutoff_frequency= 0.4 , order=10)[column + "_lowpass"].plot()



# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------

# argrelextrema(bench_set["acc_r"].values, np.greater)
data = LowPass.low_pass_filter(squat_set,col=column , sampling_frequency=fs , 
                        cutoff_frequency= 0.4 , order=10)


indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
peaks = data.iloc[indexes]


def count_reps(dataset, column='acc_r', cutoff=0.4, order=10):
    # Apply low-pass filter to smooth the signal
    data = LowPass.low_pass_filter(
        dataset,
        column,
        sampling_frequency=fs,
        cutoff_frequency=cutoff,
        order=order
    )
      # Find local maxima in the smoothed signal
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)[0]
    
    # Extract peak rows (optional: you can return len(indexes) directly)
    peaks = data.iloc[indexes]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot the low-pass filtered signal
    ax.plot(dataset[f"{column}_lowpass"], label='Filtered Signal')
    
    # Plot the peaks
    ax.plot(peaks.index, peaks[f"{column}_lowpass"], 'ro', label='Peaks')
    
    # Axis label
    ax.set_ylabel(f"{column}_lowpass")
    
    # Title using exercise and category
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    rep_count = len(peaks)
    ax.set_title(f"{category} {exercise}: {rep_count} Reps")
    
    ax.legend()
    plt.tight_layout()
    plt.show()
    
  
    
    # Return the number of detected peaks as rep count
    return len(peaks)

count_reps(bench_set)


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"] = df["category"].apply(lambda x : 5 if x == "heavy" else 10 )

reps_df =  df.groupby(["category", "label", "set"])["reps"].max().reset_index()



for s in df["set"].unique():
    subset = df[df["set"] == s]

    # Set default values
    column = "acc_r"
    cutoff = 0.4

    # Adjust cutoff based on label or category
    if subset["label"].iloc[0]== "squat":
        cutoff = 0.35
    elif subset["label"].iloc[0] == "row":
        cutoff = 0.65
        column = "gyr_x"
    elif subset["label"].iloc[0] == "ohp":
        cutoff = 0.35

    # Count reps
    reps = count_reps(subset, cutoff=cutoff, column=column)

    # Update rep_df if needed
    reps_df.loc[reps_df["set"] == s, "reps_pred"] = reps


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------


error = mean_absolute_error(reps_df["reps"], reps_df["reps_pred"]).round(2)


reps_df.groupby(["label" ,"category"])["reps" , "reps_pred"].mean().plot.bar()


reps_df["error"] = (reps_df["reps"] - reps_df["reps_pred"]).abs().round(2)
