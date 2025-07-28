
# --------------------------------------------
# 01 - DATA PREPARATION
# --------------------------------------------
# This notebook loads macroeconomic and company stock data,
# merges them by date, applies MinMax scaling, and saves
# the processed datasets for later modeling.
# --------------------------------------------

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


# Set Paths (relative to notebook location)
macro_path = os.path.join("..", "data", "raw", "daily_dataset_final.csv")
trades_root = os.path.join("..", "data", "raw", "trades_data")
output_dir = os.path.join("..", "data", "processed")
model_dir = os.path.join("..", "models")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# Load Macroeconomic Dataset
macro_df = pd.read_csv(macro_path)
macro_df['Date'] = pd.to_datetime(macro_df['Date'], errors='coerce')
macro_df.dropna(subset=['Date'], inplace=True)

macro_columns = ['Tourism Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation rate']


# Loop through each company’s daily stock data
for company_name in os.listdir(trades_root):
    company_path = os.path.join(trades_root, company_name, "Trades")
    daily_file = os.path.join(company_path, "daily.csv")

    # Fallback in case of filename typo
    if not os.path.exists(daily_file):
        daily_file = os.path.join(company_path, "dalily.csv")
    if not os.path.exists(daily_file):
        print(f"Skipping {company_name}: daily.csv not found.")
        continue

    print(f"Processing: {company_name}")

    stock_df = pd.read_csv(daily_file)

    if 'Trade Date' not in stock_df.columns or 'Close (Rs.)' not in stock_df.columns:
        print(f"Skipping {company_name}: required columns missing.")
        continue

    stock_df['Date'] = pd.to_datetime(stock_df['Trade Date'], format='%m/%d/%y', errors='coerce')
    stock_df.dropna(subset=['Date'], inplace=True)

    # Merge stock and macroeconomic data by date
    merged_df = pd.merge(stock_df, macro_df, on='Date', how='inner')
    merged_df.sort_values('Date', inplace=True)

    if len(merged_df) < 60:
        print(f"Skipping {company_name}: not enough merged data.")
        continue

    
    # Apply MinMax scaling to stock and macro data
    stock_scaler = MinMaxScaler()
    macro_scaler = MinMaxScaler()

    merged_df['Close_scaled'] = stock_scaler.fit_transform(merged_df[['Close (Rs.)']])
    merged_df[macro_columns] = macro_scaler.fit_transform(merged_df[macro_columns])

    
    # Save processed dataset and scalers
    clean_name = company_name.replace(" ", "_").replace("(", "").replace(")", "")
    processed_file = os.path.join(output_dir, f"{clean_name}_scaled.csv")

    merged_df.to_csv(processed_file, index=False)
    joblib.dump(stock_scaler, os.path.join(model_dir, f"{clean_name}_stock_scaler.save"))
    joblib.dump(macro_scaler, os.path.join(model_dir, f"{clean_name}_macro_scaler.save"))

    print(f"Saved: {processed_file}")


----------------------------# --------------------------------------------
# 02 - SEQUENCE BUILDER
# --------------------------------------------
# This notebook transforms merged & scaled stock-macro
# data into LSTM-ready sequences for forecasting.
# --------------------------------------------

import os
import numpy as np
import pandas as pd

# Settings
input_dir = os.path.join("..", "data", "processed")
output_dir = os.path.join(input_dir, "sequences")
os.makedirs(output_dir, exist_ok=True)

# Features to extract
macro_features = ['Tourism Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation rate']
target_col = "Close_scaled"
window_size = 30

# Loop through each company file
for file in os.listdir(input_dir):
    if not file.endswith("_scaled.csv"):
        continue

    df = pd.read_csv(os.path.join(input_dir, file))

    if len(df) <= window_size:
        print(f"Skipping {file}: not enough data.")
        continue

    X_stock = []
    X_macro = []
    y = []

    for i in range(window_size, len(df)):
        # 30-day stock price history
        X_stock.append(df[target_col].values[i-window_size:i])

        # 30-day macroeconomic context
        X_macro.append(df[macro_features].values[i-window_size:i])

        # Target is the next day’s stock price
        y.append(df[target_col].values[i])

    X_stock = np.array(X_stock)[..., np.newaxis]  # Shape: (samples, 30, 1)
    X_macro = np.array(X_macro)                  # Shape: (samples, 30, 4)
    y = np.array(y)

    # Save files
    company = file.replace("_scaled.csv", "")
    np.save(os.path.join(output_dir, f"{company}_X_stock.npy"), X_stock)
    np.save(os.path.join(output_dir, f"{company}_X_macro.npy"), X_macro)
    np.save(os.path.join(output_dir, f"{company}_y.npy"), y)

    print(f"Saved sequences for: {company}")


------------------------
# --------------------------------------------
# 03 - TRAIN DUAL-INPUT LSTM MODEL
# --------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Load a specific company’s sequences
company = "DOLPHIN_HOTELS_PLC"  # Change as needed
seq_path = os.path.join("..", "data", "processed", "sequences")
model_path = os.path.join("..", "models", f"{company}_lstm_model.keras")

X_stock = np.load(os.path.join(seq_path, f"{company}_X_stock.npy"))
X_macro = np.load(os.path.join(seq_path, f"{company}_X_macro.npy"))
y = np.load(os.path.join(seq_path, f"{company}_y.npy"))

print(f"X_stock shape: {X_stock.shape}, X_macro shape: {X_macro.shape}, y shape: {y.shape}")


# Train/test split
split_idx = int(0.8 * len(y))
X_stock_train, X_stock_test = X_stock[:split_idx], X_stock[split_idx:]
X_macro_train, X_macro_test = X_macro[:split_idx], X_macro[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]


# Build dual-input LSTM model
input_stock = Input(shape=(X_stock.shape[1], 1), name="Stock_Input")
input_macro = Input(shape=(X_macro.shape[1], X_macro.shape[2]), name="Macro_Input")

x1 = LSTM(64, return_sequences=False)(input_stock)
x2 = LSTM(64, return_sequences=False)(input_macro)

x = Concatenate()([x1, x2])
x = Dense(32, activation='relu')(x)
output = Dense(1, name='Forecast')(x)

model = Model(inputs=[input_stock, input_macro], outputs=output)
model.compile(optimizer=Adam(0.001), loss="mse")

model.summary()


# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    [X_stock_train, X_macro_train],
    y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

model.save(model_path)
print(f"Model saved to: {model_path}")


# Evaluate the model
y_pred = model.predict([X_stock_test, X_macro_test])
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")


# Plot actual vs predicted (scaled)
plt.figure(figsize=(12, 5))
plt.plot(y_test, label="Actual", color="black")
plt.plot(y_pred, label="Predicted", color="blue", alpha=0.7)
plt.title(f"{company} - Scaled Stock Price Forecast")
plt.xlabel("Time")
plt.ylabel("Scaled Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

----------------------
# --------------------------------------------
# 05 - EVALUATE & INVERSE TRANSFORM RESULTS
# --------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Setup
company = "DOLPHIN_HOTELS_PLC"
seq_dir = os.path.join("..", "data", "processed", "sequences")
model_dir = os.path.join("..", "models")
results_dir = os.path.join("..", "outputs")
os.makedirs(results_dir, exist_ok=True)

# Load model and scalers
model_path = os.path.join(model_dir, f"{company}_lstm_model.keras")
model = load_model(model_path, compile=False)

stock_scaler_path = os.path.join(model_dir, f"{company}_stock_scaler.save")
stock_scaler = joblib.load(stock_scaler_path)

# Load sequences
X_stock = np.load(os.path.join(seq_dir, f"{company}_X_stock.npy"))
X_macro = np.load(os.path.join(seq_dir, f"{company}_X_macro.npy"))
y_scaled = np.load(os.path.join(seq_dir, f"{company}_y.npy"))

# Predict
y_pred_scaled = model.predict([X_stock, X_macro])
y_pred = stock_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_actual = stock_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


# Save results as CSV
results_df = pd.DataFrame({
    "Actual Price": y_actual,
    "Predicted Price": y_pred
})

results_path = os.path.join(results_dir, f"{company}_forecast_vs_actual.csv")
results_df.to_csv(results_path, index=False)
print(f"Saved results to: {results_path}")


# Evaluation metrics on actual prices
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)

print(f"Actual Price RMSE: {rmse:.2f}")
print(f"Actual Price MAE : {mae:.2f}")


# Plot Actual vs Predicted
plt.figure(figsize=(12, 5))
plt.plot(y_actual, label="Actual", color="black")
plt.plot(y_pred, label="Predicted", color="blue", alpha=0.7)
plt.title(f"{company} - Stock Price Forecast (Rs.)")
plt.xlabel("Time")
plt.ylabel("Price (Rs.)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


---------------

# --------------------------------------------
# 04 - SHAP ANALYSIS FOR DUAL LSTM MODEL
# --------------------------------------------
# Explains the macroeconomic influence on LSTM predictions
# using SHAP GradientExplainer
# --------------------------------------------

import os
import shap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Set paths and load data
company = "DOLPHIN_HOTELS_PLC" 
seq_path = os.path.join("..", "data", "processed", "sequences")
model_path = os.path.join("..", "models", f"{company}_lstm_model.keras")

# Load the trained model (compile=False avoids loss reloading issues)
model = load_model(model_path, compile=False)

X_stock = np.load(os.path.join(seq_path, f"{company}_X_stock.npy"))
X_macro = np.load(os.path.join(seq_path, f"{company}_X_macro.npy"))
y = np.load(os.path.join(seq_path, f"{company}_y.npy"))

print(f"X_stock shape: {X_stock.shape}, X_macro shape: {X_macro.shape}, y shape: {y.shape}")


# SHAP: Use a subset for explanation
X_stock_sample = X_stock[:200]
X_macro_sample = X_macro[:200]

explainer = shap.GradientExplainer(
    (model.inputs, model.output),
    data=[X_stock_sample, X_macro_sample]
)

# Explain last 50 samples
shap_values = explainer.shap_values([X_stock_sample[-50:], X_macro_sample[-50:]])

# --------------------------------------------
# SHAP Summary Plot for Macroeconomic Inputs
# --------------------------------------------
# Convert 3D macro input (samples, time, features) → (samples, features)
# by averaging over the 30-day time window
X_macro_flat = X_macro_sample[-50:].mean(axis=1)

macro_feature_names = ['Tourism Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation rate']

# Plot SHAP values
# Average SHAP values across time dimension
shap_values_macro_mean = shap_values[1].mean(axis=1)

# Plot using time-averaged values
shap.summary_plot(
    shap_values_macro_mean,
    features=X_macro_flat,
    feature_names=macro_feature_names
)




-------------------------------
----
# --------------------------------------------
# 06 - TRAIN LSTM MODELS FOR ALL COMPANIES
# --------------------------------------------

import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model


# Paths
seq_path = os.path.join("..", "data", "processed", "sequences")
model_dir = os.path.join("..", "models")
os.makedirs(model_dir, exist_ok=True)


# Loop over available sequence files
companies = set()
for fname in os.listdir(seq_path):
    if fname.endswith("_X_stock.npy"):
        company = fname.replace("_X_stock.npy", "")
        companies.add(company)


# Model Training Loop
for company in sorted(companies):
    try:
        print(f"Training for: {company}")

        X_stock = np.load(os.path.join(seq_path, f"{company}_X_stock.npy"))
        X_macro = np.load(os.path.join(seq_path, f"{company}_X_macro.npy"))
        y = np.load(os.path.join(seq_path, f"{company}_y.npy"))

        # Split
        split_idx = int(0.8 * len(y))
        X_stock_train, X_stock_test = X_stock[:split_idx], X_stock[split_idx:]
        X_macro_train, X_macro_test = X_macro[:split_idx], X_macro[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Build model
        input_stock = Input(shape=(X_stock.shape[1], 1), name="Stock_Input")
        input_macro = Input(shape=(X_macro.shape[1], X_macro.shape[2]), name="Macro_Input")

        x1 = LSTM(64)(input_stock)
        x2 = LSTM(64)(input_macro)

        x = Concatenate()([x1, x2])
        x = Dense(32, activation='relu')(x)
        output = Dense(1, name='Forecast')(x)

        model = Model(inputs=[input_stock, input_macro], outputs=output)
        model.compile(optimizer=Adam(0.001), loss="mse")

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(
            [X_stock_train, X_macro_train],
            y_train,
            validation_split=0.1,
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0  # Set to 1 to watch progress
        )

        model_path = os.path.join(model_dir, f"{company}_lstm_model.keras")
        model.save(model_path)
        print(f"Saved model: {model_path}")

        # Optional evaluation print
        y_pred = model.predict([X_stock_test, X_macro_test])
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE: {rmse:.4f}")

    except Exception as e:
        print(f"Failed for {company}: {str(e)}")


# --------------------------

# --------------------------------------------
# 07 - EVALUATE ALL MODELS AND EXPORT RESULTS
# --------------------------------------------

import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Paths
seq_dir = os.path.join("..", "data", "processed", "sequences")
model_dir = os.path.join("..", "models")
output_dir = os.path.join("..", "outputs")
os.makedirs(output_dir, exist_ok=True)


# Identify all companies with trained models
companies = [
    f.replace("_lstm_model.keras", "")
    for f in os.listdir(model_dir)
    if f.endswith("_lstm_model.keras")
]

summary_metrics = []


# Loop through companies
for company in sorted(companies):
    try:
        print(f"Evaluating: {company}")

        # Load model and data
        model_path = os.path.join(model_dir, f"{company}_lstm_model.keras")
        model = load_model(model_path, compile=False)

        X_stock = np.load(os.path.join(seq_dir, f"{company}_X_stock.npy"))
        X_macro = np.load(os.path.join(seq_dir, f"{company}_X_macro.npy"))
        y_scaled = np.load(os.path.join(seq_dir, f"{company}_y.npy"))

        # Load stock scaler
        scaler_path = os.path.join(model_dir, f"{company}_stock_scaler.save")
        stock_scaler = joblib.load(scaler_path)

        # Predict
        y_pred_scaled = model.predict([X_stock, X_macro])
        y_pred = stock_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_actual = stock_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

        # Save CSV
        result_df = pd.DataFrame({
            "Actual Price": y_actual,
            "Predicted Price": y_pred
        })

        csv_path = os.path.join(output_dir, f"{company}_forecast_vs_actual.csv")
        result_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        summary_metrics.append({
            "Company": company,
            "RMSE": rmse,
            "MAE": mae
        })

    except Exception as e:
        print(f" Skipped {company}: {str(e)}")


# Save summary metrics
summary_df = pd.DataFrame(summary_metrics)
summary_path = os.path.join(output_dir, "summary_model_performance.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary to: {summary_path}")


# ---------------------------


# --------------------------------------------
# 08 - PLOT AND SAVE ALL MODEL FORECASTS
# --------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
import math

# Paths
output_dir = os.path.join("..", "outputs")
plot_dir = os.path.join(output_dir, "plots/predictions")
os.makedirs(plot_dir, exist_ok=True)

forecast_files = [f for f in os.listdir(output_dir) if f.endswith("_forecast_vs_actual.csv")]
forecast_files.sort()
num_companies = len(forecast_files)
cols = 3
rows = math.ceil(num_companies / cols)

fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3), sharex=False)
axes = axes.flatten()

for i, filename in enumerate(forecast_files):
    df = pd.read_csv(os.path.join(output_dir, filename))
    company_name = filename.replace("_forecast_vs_actual.csv", "").replace("_", " ")

    ax = axes[i]
    ax.plot(df["Actual Price"], label="Actual", color="black")
    ax.plot(df["Predicted Price"], label="Predicted", color="blue", alpha=0.7)
    ax.set_title(company_name, fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Finalize and save combined grid image
fig.suptitle("All Company Forecasts - Actual vs Predicted", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
combined_path = os.path.join(plot_dir, "all_forecasts_grid.png")
fig.savefig(combined_path)
print(f"Saved grid image: {combined_path}")
plt.show()


# -------------------

# --------------------------------------------
# 08 - SAVE INDIVIDUAL PREDICTION PLOTS
# --------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
output_dir = os.path.join("..", "outputs")
plot_dir = os.path.join(output_dir, "plots/predictions")
os.makedirs(plot_dir, exist_ok=True)

# Find all forecast result files
forecast_files = [
    f for f in os.listdir(output_dir)
    if f.endswith("_forecast_vs_actual.csv")
]

# Sort for consistency
forecast_files.sort()

# Loop through each forecast file
for filename in forecast_files:
    try:
        company_name = filename.replace("_forecast_vs_actual.csv", "").replace("_", " ")
        df = pd.read_csv(os.path.join(output_dir, filename))

        # Create plot
        plt.figure(figsize=(10, 4))
        plt.plot(df["Actual Price"], label="Actual", color="black")
        plt.plot(df["Predicted Price"], label="Predicted", color="blue", alpha=0.7)
        plt.title(company_name)
        plt.xlabel("Time")
        plt.ylabel("Stock Price (Rs.)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(plot_dir, f"{company_name}.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

        # Show plot
        plt.show()

    except Exception as e:
        print(f" Error in {filename}: {str(e)}")


# -----------------------

# --------------------------------------------
# 9 ---- Step 1: Run Granger Causality Tests
# --------------------------------------------

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# Configuration
company_list = [d for d in os.listdir("../data/processed/sequences") if d.endswith("_X_stock.npy")]
macro_feature_names = ['Tourist Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation']
max_lag = 10  # You can tune this

# Output dataframe
results = []

for file_prefix in company_list:
    company = file_prefix.replace("_X_stock.npy", "")
    print(f"Testing Granger causality for: {company}")

    # Load sequences and targets
    X_stock = np.load(f"../data/processed/sequences/{company}_X_stock.npy")
    X_macro = np.load(f"../data/processed/sequences/{company}_X_macro.npy")
    y = np.load(f"../data/processed/sequences/{company}_y.npy")

    # Flatten sequences into 1D for analysis (simplification)
    stock_series = y.reshape(-1)

    for i, feature in enumerate(macro_feature_names):
        macro_series = X_macro[:, -1, i]  # Use most recent time step in each window
        df = pd.DataFrame({"stock": stock_series, "macro": macro_series})

        try:
            # Run Granger test
            test_result = grangercausalitytests(df[["stock", "macro"]], maxlag=max_lag, verbose=False)
            p_vals = [round(test_result[lag][0]["ssr_chi2test"][1], 4) for lag in range(1, max_lag+1)]
            min_p = min(p_vals)

            results.append({
                "Company": company.replace("_", " "),
                "Feature": feature,
                "Min_PValue": min_p
            })
        except Exception as e:
            print(f"Failed for {company} - {feature}: {e}")
            results.append({
                "Company": company.replace("_", " "),
                "Feature": feature,
                "Min_PValue": np.nan
            })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("../outputs/granger_causality_results.csv", index=False)
df_results.head()




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Granger causality results
df = pd.read_csv("../outputs/granger_causality_results.csv")

# Pivot for heatmap: rows = companies, columns = macro features
heatmap_data = df.pivot(index="Company", columns="Feature", values="Min_PValue")

# Plot
plt.figure(figsize=(10, len(heatmap_data) * 0.6 + 2))
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="coolwarm_r",
    fmt=".3f",
    linewidths=0.5,
    cbar_kws={"label": "Min P-Value"},
    vmax=0.1  # cap upper bound to highlight low p-values
)

plt.title("Granger Causality Heatmap – Min P-Values", fontsize=14)
plt.xlabel("Macroeconomic Feature")
plt.ylabel("Company")
plt.tight_layout()

# Save plot
plt.savefig("../outputs/plots/granger_heatmap.png")
plt.show()


# -----------------------------

# 10_Flexible_Time_Window_granger_analysis
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# ----------------------------
#  Flexible Time Configuration
# ----------------------------
months_per_window = 6         # e.g., 3, 6, or 12
months_stride = 3             # step between windows / Overlap or step in months (e.g., 3 for quarterly)
trading_days_per_month = 21

company = "DOLPHIN_HOTELS_PLC"  
window_days = months_per_window * trading_days_per_month
stride_days = months_stride * trading_days_per_month

# ----------------------------
#  Granger Rolling Window Test
# ----------------------------
macro_feature_names = ['Tourist Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation']
max_lag = 10

# Load data
X_stock = np.load(f"../data/processed/sequences/{company}_X_stock.npy")
X_macro = np.load(f"../data/processed/sequences/{company}_X_macro.npy")
y = np.load(f"../data/processed/sequences/{company}_y.npy")
stock_series = y.reshape(-1)

output_rows = []

for start in range(0, len(stock_series) - window_days, stride_days):
    end = start + window_days
    time_label = f"Day {start}-{end}"

    for i, feature in enumerate(macro_feature_names):
        macro_series = X_macro[start:end, -1, i]
        stock_window = stock_series[start:end]
        df = pd.DataFrame({"stock": stock_window, "macro": macro_series})

        try:
            result = grangercausalitytests(df[["stock", "macro"]], maxlag=max_lag, verbose=False)
            min_p = min([round(result[lag][0]["ssr_chi2test"][1], 4) for lag in range(1, max_lag + 1)])
        except Exception:
            min_p = np.nan

        output_rows.append({
            "Window": time_label,
            "Feature": feature,
            "Min_PValue": min_p
        })

# Save results
df_windowed = pd.DataFrame(output_rows)
out_path = f"../outputs/granger_{company}_{months_per_window}month.csv"
df_windowed.to_csv(out_path, index=False)
df_windowed.head()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the rolling Granger causality result CSV
company = "DOLPHIN_HOTELS_PLC"  
months = 6  # 3, 6, 12 etc. depending on your config
csv_path = f"../outputs/granger_{company}_{months}month.csv"

# Read results
df = pd.read_csv(csv_path)

# Pivot for heatmap: rows = macro features, columns = rolling time windows
heatmap_df = df.pivot(index="Feature", columns="Window", values="Min_PValue")

# Plot the heatmap
plt.figure(figsize=(14, 4))
sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".3f",
    cmap="coolwarm_r",
    linewidths=0.5,
    cbar_kws={"label": "Min P-Value"},
    vmax=0.1  # highlight lower p-values
)

plt.title(f"Granger Causality Heatmap – {months}-Month Rolling Window ({company.replace('_', ' ')})", fontsize=13)
plt.xlabel("Time Window (Day Range)")
plt.ylabel("Macroeconomic Feature")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save plot
plot_path = f"../outputs/plots/rolling_granger_heatmap_{company}_{months}month.png"
plt.savefig(plot_path)
plt.show()

# ----------------------------------------
# 11_all_Flexible_Time_Window_granger_analysis
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

def plot_granger_heatmap(company, months_per_window=6, months_stride=3,
                          input_dir="../data/processed/sequences",
                          output_dir="../outputs",
                          save_dir="../outputs/plots/rolling_ganager_heatmap"):
    """
    Computes rolling-window Granger causality and generates heatmap for a company.
    
    Parameters:
    - company (str): Company name prefix used in sequence file names
    - months_per_window (int): Size of the rolling window in months
    - months_stride (int): Step size in months for rolling window
    - input_dir (str): Directory with input .npy sequences
    - output_dir (str): Directory to save causality CSV
    - save_dir (str): Directory to save heatmap image
    """
    trading_days_per_month = 21
    window_days = months_per_window * trading_days_per_month
    stride_days = months_stride * trading_days_per_month
    macro_feature_names = ['Tourist Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation']
    max_lag = 10

    # Load data
    try:
        X_stock = np.load(os.path.join(input_dir, f"{company}_X_stock.npy"))
        X_macro = np.load(os.path.join(input_dir, f"{company}_X_macro.npy"))
        y = np.load(os.path.join(input_dir, f"{company}_y.npy"))
    except FileNotFoundError:
        print(f"Sequence files for '{company}' not found in {input_dir}")
        return

    stock_series = y.reshape(-1)
    output_rows = []

    for start in range(0, len(stock_series) - window_days, stride_days):
        end = start + window_days
        time_label = f"Day {start}-{end}"

        for i, feature in enumerate(macro_feature_names):
            macro_series = X_macro[start:end, -1, i]
            stock_window = stock_series[start:end]
            df = pd.DataFrame({"stock": stock_window, "macro": macro_series})

            try:
                result = grangercausalitytests(df[["stock", "macro"]], maxlag=max_lag, verbose=False)
                min_p = min([round(result[lag][0]["ssr_chi2test"][1], 4) for lag in range(1, max_lag + 1)])
            except Exception:
                min_p = np.nan

            output_rows.append({
                "Window": time_label,
                "Feature": feature,
                "Min_PValue": min_p
            })

    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"granger_{company}_{months_per_window}month.csv")
    df_windowed = pd.DataFrame(output_rows)
    df_windowed.to_csv(csv_path, index=False)
    print(f"Granger CSV saved: {csv_path}")

    # Generate heatmap
    heatmap_data = df_windowed.pivot(index="Feature", columns="Window", values="Min_PValue")
    plt.figure(figsize=(14, 4))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="coolwarm_r",
        linewidths=0.5,
        cbar_kws={"label": "Min P-Value"},
        vmax=0.1
    )
    plt.title(f"Granger Causality Heatmap – {months_per_window}-Month Window ({company.replace('_', ' ')})", fontsize=13)
    plt.xlabel("Time Window (Day Range)")
    plt.ylabel("Macroeconomic Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"rolling_granger_heatmap_{company}_{months_per_window}month.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Heatmap saved: {plot_path}")

# -------------------------------

# 12_all_shap_analysis

import os
import shap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def generate_shap_plot(company, seq_path="../data/processed/sequences", model_path="../models", save_dir="../outputs/plots/shap_plots"):
    """
    Generates and saves a SHAP summary plot for macroeconomic features
    influencing LSTM stock price predictions for a specific company.

    Parameters:
    - company (str): Company name prefix
    - seq_path (str): Path to directory containing the .npy sequence files
    - model_path (str): Path to directory containing the trained .keras model
    - save_dir (str): Directory where the SHAP plot will be saved
    """
    print(f" Generating SHAP plot for {company}")

    # Load model and data
    model_file = os.path.join(model_path, f"{company}_lstm_model.keras")
    model = load_model(model_file, compile=False)

    X_stock = np.load(os.path.join(seq_path, f"{company}_X_stock.npy"))
    X_macro = np.load(os.path.join(seq_path, f"{company}_X_macro.npy"))

    # Subsample for SHAP
    X_stock_sample = X_stock[:200]
    X_macro_sample = X_macro[:200]

    # SHAP GradientExplainer
    explainer = shap.GradientExplainer(
        (model.inputs, model.output),
        data=[X_stock_sample, X_macro_sample]
    )

    shap_values = explainer.shap_values([X_stock_sample[-50:], X_macro_sample[-50:]])

    # Average over time steps
    X_macro_flat = X_macro_sample[-50:].mean(axis=1)
    shap_values_macro_mean = shap_values[1].mean(axis=1)

    # Plot
    plt.figure()
    macro_feature_names = ['Tourism Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation rate']
    shap.summary_plot(
        shap_values_macro_mean,
        features=X_macro_flat,
        feature_names=macro_feature_names,
        show=False  # prevent auto-display
    )

    # Save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"shap_summary_macro_{company}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f" SHAP summary plot saved to: {out_path}")


# -----------------------------------------------------
#  13_all_shap_analysis_bar_plots
import os
import shap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

def generate_shap_plot(company,
                             seq_path="../data/processed/sequences",
                             model_path="../models",
                             save_dir="../outputs/plots/shap_bar_plots"):
    """
    Generates a 2x2 SHAP grid plot using bar plots for each macroeconomic feature
    and saves it as a single image.

    Parameters:
    - company (str): Company identifier (prefix)
    - seq_path (str): Directory with .npy sequence files
    - model_path (str): Directory with trained model
    - save_dir (str): Directory to save grid image
    """
    print(f"Generating SHAP bar grid plot for: {company}")

    # Load data
    X_stock = np.load(os.path.join(seq_path, f"{company}_X_stock.npy"))
    X_macro = np.load(os.path.join(seq_path, f"{company}_X_macro.npy"))
    model = load_model(os.path.join(model_path, f"{company}_lstm_model.keras"), compile=False)

    # Sample for SHAP
    X_stock_sample = X_stock[:200]
    X_macro_sample = X_macro[:200]

    # Explain last 50 samples
    explainer = shap.GradientExplainer(
        (model.inputs, model.output),
        data=[X_stock_sample, X_macro_sample]
    )
    shap_values = explainer.shap_values([X_stock_sample[-50:], X_macro_sample[-50:]])
    shap_values_macro = shap_values[1].mean(axis=1)
    X_macro_flat = X_macro_sample[-50:].mean(axis=1)

    feature_names = ['Tourism Arrivals', 'Money Supply', 'Exchange Rate', 'Inflation rate']

    tmp_files = []

    # Generate individual bar plots and save temporarily
    for i, name in enumerate(feature_names):
        shap.summary_plot(
            shap_values_macro[:, i].reshape(-1, 1),
            features=X_macro_flat[:, i].reshape(-1, 1),
            feature_names=[name],
            show=False,
            plot_type="bar"
        )
        temp_path = f"temp_shap_bar_{i}.png"
        plt.title(f"{name} – SHAP (Bar)", fontsize=11)
        plt.tight_layout()
        plt.savefig(temp_path, bbox_inches="tight")
        plt.clf()
        tmp_files.append(temp_path)

    # Create grid of saved bar plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        img = plt.imread(tmp_files[i])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(feature_names[i], fontsize=12)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    grid_path = os.path.join(save_dir, f"shap_bar_grid_macro_{company}.png")
    plt.savefig(grid_path)
    plt.show()
    plt.close()

    # Remove temp files
    for file in tmp_files:
        os.remove(file)

    print(f"Bar grid saved to: {grid_path}")
