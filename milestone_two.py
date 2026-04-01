# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


# pd.set_option('display.max_columns', None)

# df = pd.read_csv("Azure_Based_Demand_Forecasting_Data.csv") # loads our csv file inot a dataframe
# df.columns = df.columns.str.strip().str.lower()
# print(df)

# df = df.sort_values("timestamp").reset_index(drop=True)
# print(df)
# # ensures that there is a crt chronologically order
# # required for the lag, rolling, and spike calcul



"""
Milestone 2 - Feature Engineering & Data Wrangling

This script:
1. Loads the Azure demand dataset
2. Performs data cleaning
3. Extracts time-based features
4. Creates lag & rolling statistical features
5. Detects anomaly spikes
6. Encodes categorical variables
7. Produces a final model-ready dataset
"""


#  1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents pop-up graphs (needed for server environments)
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)  # Show all columns when printing


#  2. Load Dataset

df = pd.read_csv("Azure_Based_Demand_Forecasting_Data.csv")  # Load dataset into a DataFrame
print("Dataset Loaded Successfully!\n")

# Clean column names (remove spaces, make lowercase)
df.columns = df.columns.str.strip().str.lower()
print(df)


# 3. Convert Timestamp → Datetime

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce") 

# Sort by time (always required for time series)
df = df.sort_values("timestamp").reset_index(drop=True)

print("Timestamp converted and dataset sorted!\n")
print(df)


# 4. Create Time-Based Features

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["weekday"] = df["timestamp"].dt.weekday
df["month"] = df["timestamp"].dt.month
df["quarter"] = df["timestamp"].dt.quarter
df["year"] = df["timestamp"].dt.year

# Create season column
def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"

df["season"] = df["month"].apply(get_season)

print("Time-based features created successfully!\n")
print(df)


#  5. Create Lag Features

df["lag_1_usage"] = df["usage_units"].shift(1)      # previous hour
df["lag_7_usage"] = df["usage_units"].shift(7)      # same hour one week earlier
df["lag_24_usage"] = df["usage_units"].shift(24)    # same hour previous day
df["lag_168_usage"] = df["usage_units"].shift(168)  # same hour previous week

print("Lag features created!\n")
print(df)



#  6. Rolling Statistical Features

df["rolling_mean_3"] = df["usage_units"].rolling(window=3).mean()      # short-term trend
df["rolling_mean_24"] = df["usage_units"].rolling(window=24).mean()    # daily trend
df["rolling_std_24"] = df["usage_units"].rolling(window=24).std()      # daily variability
df["rolling_mean_168"] = df["usage_units"].rolling(window=168).mean()  # weekly trend

print("Rolling features added!\n")
print(df)


#  7. Usage Spike Detection

threshold = df["usage_units"].mean() + df["usage_units"].std()
df["usage_spike"] = np.where(df["usage_units"] > threshold, 1, 0)


# Peak hour they normally see high load
df["peak_hour_flag"] = df["hour"].apply(lambda x: 1 if x in [9,10,11,18,19,20] else 0)

print("Anomaly detection features added!\n")
print(df)


#  8. One-Hot Encode Categorical Features

df = pd.get_dummies(df, columns=["region", "service_type", "season"], drop_first=True)

print("Categorical variables encoded!\n")

#  9. Remove Rows with NaN

df = df.dropna().reset_index(drop=True)

print("Final dataset cleaned and ready for modeling!\n")


#  10. Print Summary

print("\n== FINAL DATASET INFO ==\n")
print(df.info())
print("\n== FIRST 5 ROWS ==\n")
print(df.head())
print("\n== NULL VALUES CHECK ==\n")
print(df.isnull().sum())