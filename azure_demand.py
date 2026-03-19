import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)

df = pd.read_csv("Azure_Based_Demand_Forecasting_Data.csv") # loads our csv file inot a dataframe
df.columns = df.columns.str.strip().str.lower()
print(df)

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') #converts the string data inot datetime objects
df = df.sort_values(by='timestamp') # sorting is mandatory for the time-series
print(df)

df['region'] = (df['region'].str.strip().str.lower().str.replace(" ", "-"))
# str.lower- removes the case diff
#.replaces- standaridizes the formatting


df['region'] = df['region'].replace({ # .replace- maps the variants to canoniical names.
    'central-india': 'Central-India',
    'west-us': 'West-US',
    'east-us': 'East-US',
    'east-asia': 'East-Asia',
    'uk-south': 'UK-South',  
})

print(df)

df = df.drop_duplicates() # keeps the first occurances
print(df)

numeric_cols = [
    'usage_units',
    'provisioned_capacity',
    'cost_usd',
    'availability_pct',
    'economic_growth_index',
    'it_spending_growth'
]

for col in numeric_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r'[^0-9\.\-]', '', regex=True)
        .replace('', np.nan)
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')


df['usage_units'] = df['usage_units'].interpolate()# usage units- time dependent
# handling missing value- col wise
# interpolate preserves trend

df['cost_usd'] = df['cost_usd'].fillna(
    df['usage_units'] * 0.5
)
#cost- derived from the usage unit
# recomputing instead of guessing mean

df['availability_pct'] = df['availability_pct'].fillna(method='ffill')
# fowrard fill is reasonable

df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)

print(df.isnull().sum())

# ==============================
# CLEAN SEPARATE PLOTS PER REGION
# ==============================
regions = df['region'].unique()

for region in regions:
    sub = df[df['region'] == region]

    if sub.empty:
        print(f"Skipping {region} — no data")
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(sub['timestamp'], sub['usage_units'], linewidth=1)
    
    plt.title(f"Usage Units Over Time - {region}", fontsize=14)
    plt.xlabel("Timestamp")
    plt.ylabel("Usage Units")
    plt.grid(True)

    filename = f"usage_units_{region}.png"
    plt.savefig(filename)
    plt.close()

    print(f"Saved: {filename}")

