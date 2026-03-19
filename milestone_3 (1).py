# 1. Import Required Libraries


# Data manipulation library
import pandas as pd

# Numerical operations
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt

# Time series forecasting model
from statsmodels.tsa.arima.model import ARIMA

# Machine learning regression model
from xgboost import XGBRegressor

# Model evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Model tuning
from sklearn.model_selection import GridSearchCV

# Traintest split
from sklearn.model_selection import train_test_split

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")



# 2. Load Dataset


# Load the Azure demand dataset
df = pd.read_csv("Azure_Based_Demand_Forecasting_Data.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)

# Display first 5 rows
print(df.head())



# 3. Data Preparation


# Convert timestamp column to datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort dataset based on time to maintain timeseries order
df = df.sort_values("timestamp")
df = df.reset_index(drop=True)

# Check missing values in dataset
print(df.isnull().sum())

# Fill missing numeric values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Display first 5 rows
print(df.head())



# 4. Feature Engineering


# Extract time components from timestamp

df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek



#Encode Categorical Variables


# Convert region and service_type into dummy variables
df = pd.get_dummies(df, columns=["region","service_type"], drop_first=True)



#Define Features and Target Variable


# X contains independent variables
X = df.drop(["usage_units","timestamp"], axis=1)

# y contains target variable (demand to predict)
y = df["usage_units"]



#TrainTest Split


# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)



#Baseline ARIMA Model


# Train ARIMA model with initial parameters
arima_model = ARIMA(y_train, order=(1,1,1))

# Fit model
arima_fit = arima_model.fit()

# Forecast demand for test data
arima_pred = arima_fit.forecast(steps=len(y_test))



#Baseline XGBoost Model


# Initialize XGBoost regressor
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    objective="reg:squarederror"
)

# Train model
xgb_model.fit(X_train, y_train)

# Make predictions
xgb_pred = xgb_model.predict(X_test)



#Baseline Model Evaluation


def evaluate(true, pred):

    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    bias = np.mean(pred - true)

    return mae, rmse, bias

arima_mae, arima_rmse, arima_bias = evaluate(y_test, arima_pred)
xgb_mae, xgb_rmse, xgb_bias = evaluate(y_test, xgb_pred)

print("Baseline ARIMA RMSE:", arima_rmse)
print("Baseline XGBoost RMSE:", xgb_rmse)



#ARIMA Hyperparameter Tuning


p = range(0,4)
d = range(0,2)
q = range(0,4)

best_rmse = float("inf")
best_order = None



#Grid Search for ARIMA


for i in p:
    for j in d:
        for k in q:
            try:

                model = ARIMA(y_train, order=(i,j,k))
                model_fit = model.fit()

                pred = model_fit.forecast(steps=len(y_test))

                rmse = np.sqrt(mean_squared_error(y_test, pred))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = (i,j,k)

            except:
                continue

print("Best ARIMA Order:", best_order)



#Train Best ARIMA Model


best_arima = ARIMA(y_train, order=best_order).fit()

arima_tuned_pred = best_arima.forecast(steps=len(y_test))



#XGBoost Hyperparameter Tuning


param_grid = {

    "n_estimators":[100,200,300],
    "max_depth":[3,5,7],
    "learning_rate":[0.01,0.1],
    "subsample":[0.8,1]

}


#GridSearchCV


grid_search = GridSearchCV(
    estimator=XGBRegressor(objective="reg:squarederror"),
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1
)

grid_search.fit(X_train, y_train)



#Best XGBoost Model
 

best_xgb = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)



#Tuned XGBoost Prediction


xgb_tuned_pred = best_xgb.predict(X_test)



#Final Model Evaluation


arima_mae, arima_rmse, arima_bias = evaluate(y_test, arima_tuned_pred)
xgb_mae, xgb_rmse, xgb_bias = evaluate(y_test, xgb_tuned_pred)

results = pd.DataFrame({

    "Model":["ARIMA","XGBoost"],
    "MAE":[arima_mae,xgb_mae],
    "RMSE":[arima_rmse,xgb_rmse],
    "Forecast Bias":[arima_bias,xgb_bias]

})

print(results)


# Model Performance Comparison


plt.figure(figsize=(8,5))

plt.bar(results["Model"], results["RMSE"])

plt.title("Model RMSE Comparison")
plt.xlabel("Model")
plt.ylabel("RMSE (Lower is Better)")

plt.grid(axis="y")

# Save image
plt.savefig("model_rmse_comparison.png", dpi=300)

plt.show()



#Visualization


plt.figure(figsize=(12,6))

plt.plot(y_test.values, label="Actual Demand", linewidth=2)
plt.plot(arima_tuned_pred.values, label="ARIMA Prediction", linestyle="-")
plt.plot(xgb_tuned_pred, label="XGBoost Prediction", linestyle=":")

plt.title("Azure Based Demand Forecast Comparison")
plt.xlabel("Time Index")
plt.ylabel("Usage Units")

plt.legend()
plt.grid(True)

# Save image to project folder
plt.savefig("demand_forecast_comparison.png", dpi=300)

plt.show()

# Feature importance visualization

importance = best_xgb.feature_importances_

features = X.columns

plt.figure(figsize=(10,6))

plt.barh(features, importance)
plt.tight_layout()

plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance Score")

plt.savefig("feature_importance.png", dpi=300)
plt.show()



# Actual vs Predicted Plot


plt.figure(figsize=(6,6))

plt.scatter(y_test, xgb_tuned_pred)

plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")

plt.title("Actual vs Predicted Demand (XGBoost)")

plt.grid(True)

plt.savefig("actual_vs_predicted.png", dpi=300)

plt.show()



#Best Model Selection


best_model = results.sort_values("RMSE").iloc[0]

print("Best Performing Model:")
print(best_model)



import joblib

if xgb_rmse < arima_rmse:
    joblib.dump(best_xgb, "best_xgboost_model.pkl")
    print("XGBoost model saved.")
else:
    joblib.dump(best_arima, "best_arima_model.pkl")
    print("ARIMA model saved.")


