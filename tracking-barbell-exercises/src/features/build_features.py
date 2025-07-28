import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
#from FrequencyAbstraction import FourierTransformation

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")

predictors_column = list(df.columns[:6])


plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictors_column:
    df[col] = df[col].interpolate()

df.info()
    
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

df[df["set"] == 1]
duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]

duration.seconds




for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    end =  df[df["set"] == s].index[-1]
    
    duration = end - start
    
    df.loc[(df["set"] == s) , "duration"] = duration.seconds


duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

    
# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()

LowPass= LowPassFilter()

fs = 1000/200

cutoff = 1.5

df_lowpass = LowPass.low_pass_filter( df_lowpass , "acc_y" , fs , cutoff , order=5)


subset = df_lowpass[df_lowpass["set"]==45]

#print(subset["label"][0])


fig,ax = plt.subplots(nrows=2, sharex= True , figsize = (20,10))

ax[0].plot(subset["acc_y"].reset_index(drop=True), label = "raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label = "butterworth filter")
ax[0].legend(loc = "upper center"  , bbox_to_anchor = (0.5,1.15) , fancybox= True , shadow = True)
ax[1].legend(loc = "upper center"  , bbox_to_anchor = (0.5,1.15) , fancybox= True , shadow = True)

for col in predictors_column:
    df_lowpass = LowPass.low_pass_filter(df_lowpass,col,fs,cutoff,order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()


pc_values = PCA.determine_pc_explained_variance(df_pca,predictors_column)


plt.figure(figsize=(10,10))
plt.plot(range(1, len(predictors_column)+1) , pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variable")
plt.show()

# elbow on 3  i.e 3 variables captures most info and adding more wont affect much
df_pca = PCA.apply_pca(df_pca, predictors_column , 3)


subset = df_pca[df_pca["set"]==35]

subset[["pca_1","pca_2","pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# # --------------------------------------------------------------

#impartial to device orientation and can handle dynamic re-orientations.

df_squared = df_pca.copy()

acc_r =  df_squared["acc_x"]**2 +  df_squared["acc_y"]**2 +  df_squared["acc_z"]**2
gyr_r =  df_squared["gyr_x"]**2 +  df_squared["gyr_y"]**2 +  df_squared["gyr_z"]**2

df_squared["acc_r"]  =  np.sqrt(acc_r)
df_squared["gyr_r"]  =  np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 16]

subset[["acc_r","gyr_r"]].plot(subplots = True)


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()

NumAbs = NumericalAbstraction()

predictors_column = predictors_column +  ["acc_r", "gyr_r"]

ws = int(1000/200)   # window size is 1 sec so we take the value and 4 previous observation according to 200ms step size

for col in predictors_column:
    df_temporal = NumAbs.abstract_numerical(df_temporal,[col],ws,"mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal,[col],ws,"std")
    

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"]==s].copy()
    for col in predictors_column:
        subset= NumAbs.abstract_numerical(subset,[col],ws,"mean")
        subset = NumAbs.abstract_numerical(subset,[col],ws,"std")
    df_temporal_list.append(subset)   

df_temporal = pd.concat(df_temporal_list)


df_temporal.info()
        
subset[["acc_y","acc_y_temp_mean_ws_5"	,"acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y","gyr_y_temp_mean_ws_5"	,"gyr_y_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_freq = df_temporal.copy().reset_index()

import importlib

import FrequencyAbstraction
importlib.reload(FrequencyAbstraction)

from FrequencyAbstraction import FourierTransformation

# Check preconditions
print("Data length:", len(df_freq))
print("NaNs in acc_y:", df_freq["acc_y"].isna().sum())

FreqAbs = FourierTransformation()

fs = int(1000/200)  # 5 Hz sampling rate
ws = int(2800/200)  # safer window size for FFT

try:
    df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)
except Exception as e:
    print("Error during frequency abstraction:", e)

print("Computed columns:", df_freq.columns)


#Visualise results
subset = df_freq[df_freq["set"]==15]
subset["acc_y"].plot()
subset["acc_y_max_freq"].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse","acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()


subset = df_freq[df_freq["set"] == 17]
subset[["acc_y", "acc_y_max_freq"]].plot()




df_freq_list = []
for s in  df_freq["set"].unique():
    print(f"Applying fourier transform to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictors_column, ws, fs)
    df_freq_list.append(subset)
    

df_freq =  pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)  

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq = df_freq.dropna()

# remove adjcaent rows 
df_freq =  df_freq.iloc[::2] 



# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

from sklearn.cluster import KMeans
# Make a copy of your DataFrame
df_cluster = df_freq.copy()

# Select columns for clustering
cluster_columns = ["acc_x", "acc_y", "acc_z"]
# X = df_cluster[cluster_columns]

# Elbow method to find the optimal k
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=20)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias, marker='o')
plt.xlabel("k")
plt.ylabel("Sum of squared distances (Inertia)")

plt.grid(True)
plt.show()
# elbow at 5

kmeans = KMeans(n_clusters=5, random_state=0, n_init=20)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plotting is supported

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(
        subset["acc_x"],
        subset["acc_y"],
        subset["acc_z"],
        label=c
    )

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()





fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for c in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == c]
    ax.scatter(
        subset["acc_x"],
        subset["acc_y"],
        subset["acc_z"],
        label=c
    )

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------


df_cluster.to_pickle("../../data/interim/03_data_features.pkl")