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

FreqAbs = FourierTransformation()

fs = int(1000/200)
ws = int(1000/200)

 
df_freq = FreqAbs.abstract_frequency(df_freq,["acc_y"],ws,fs)

df_freq.columns

#Visualise results
subset = df_freq[df_freq["set"]==15]


class FourierTransformation:
    
    def __init__(self):
        self.temp_list = []
        self.freqs = None

    # Find the amplitudes of the different frequencies using a fast fourier transformation. Here,
    # the sampling rate expresses
    # the number of samples per second (i.e. Frequency is Hertz of the dataset).
    
    def find_fft_transformation(self, data):
        # Create the transformation, this includes the amplitudes of both the real
        # and imaginary part.
        # print(data.shape)
        transformation = np.fft.rfft(data, len(data))
        # real
        real_ampl = transformation.real
        # max
        max_freq = self.freqs[np.argmax(real_ampl[0:len(real_ampl)])]
        # weigthed
        freq_weigthed = float(np.sum(self.freqs * real_ampl)) / np.sum(real_ampl)

        # pse

        PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
        PSD_pdf = np.divide(PSD, np.sum(PSD))

        # Make sure there are no zeros.
        if np.count_nonzero(PSD_pdf) == PSD_pdf.size:
            pse = -np.sum(np.log(PSD_pdf) * PSD_pdf)
        else:
            pse = 0

        real_ampl = np.insert(real_ampl, 0, max_freq)
        real_ampl = np.insert(real_ampl, 0, freq_weigthed)
        row = np.insert(real_ampl, 0, pse)

        self.temp_list.append(row)

        return 0

    # Get frequencies over a certain window.
    def abstract_frequency(self, data_table, columns, window_size, sampling_rate):
        self.freqs = (sampling_rate * np.fft.rfftfreq(int(window_size))).round(3)

        for col in columns:
            collist = []
            # prepare column names
            collist.append(col + '_max_freq')
            collist.append(col + '_freq_weighted')
            collist.append(col + '_pse')
            
            collist = collist + [col + '_freq_' +
                    str(freq) + '_Hz_ws_' + str(window_size) for freq in self.freqs]
           
            # rolling statistics to calculate frequencies, per window size. 
            # Pandas Rolling method can only return one aggregation value. 
            # Therefore values are not returned but stored in temp class variable 'temp_list'.

            # note to self! Rolling window_size would be nicer and more logical! In older version windowsize is actually 41. (ws + 1)
            data_table[col].rolling(
                window_size + 1).apply(self.find_fft_transformation)

            # Pad the missing rows with nans
            frequencies = np.pad(np.array(self.temp_list), ((window_size, 0), (0, 0)),
                        'constant', constant_values=np.nan)
            # add new freq columns to frame
            
            data_table[collist] = pd.DataFrame(frequencies, index=data_table.index)

            # reset temp-storage array
            del self.temp_list[:]
            

        
        return data_table

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
