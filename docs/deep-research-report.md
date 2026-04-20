# Solar Energy Forecasting Project Outline

## Step 2: Insights from Literature  
Solar forecasting has seen a range of approaches.  Simple statistical/regression baselines (e.g. linear or ARIMA models) can be used as an initial benchmark.  Many studies use decision-tree ensembles and support vector machines (SVR/XGBoost) as medium-complexity models.  Recent work highlights deep learning models: LSTM (recurrent) and CNN-based models have emerged as *“promising approaches for solar irradiance forecasting”*【15†L64-L72】.  For example, a stacked LSTM model (32–16–8 units) achieved very low error (RMSE≈37.1 W/m², MAE≈15.2 W/m²) when predicting one-hour-ahead irradiance【79†L210-L218】.  Comparisons in the literature show that Random Forest (RF) often outperforms other algorithms (ANN, LSTM, etc.) on solar PV prediction【73†L89-L97】.  One recent study even built an ensemble of eight diverse ML models to boost performance【76†L85-L92】.  In summary, candidate models include simple linear regression, SVR/XGBoost, tree ensembles (Random Forest), and advanced deep nets (LSTM, CNN, hybrid CNN–LSTM) – guided by these findings【15†L64-L72】【73†L89-L97】.

## Step 3: Dataset Selection  
We choose the **High-Resolution Floating Solar PV Data** from OpenEI, which contains multi-year solar PV generation data at 1-minute intervals.  This dataset covers over two years of operation at floating PV sites (Oakville CA, Orlando FL, etc.), including detailed power output and module/weather measurements【64†L89-L97】.  It is large and clean (CSV format) with relevant features (timestamp, PV output, ambient and module temperatures, solar irradiance, wind, etc.).  The high temporal resolution and rich features make it well-suited for time-series forecasting of solar power production【64†L89-L97】.  

*Final dataset:* the “Floating PV Oakville CA” (and related site) CSV files were downloaded from OpenEI. (They include ~1-min PV output and meteorological data.)

## Step 4: Problem Definition  
**Problem:** “I will predict solar PV power output using the floating PV dataset.”  In particular, the task is to forecast the plant’s future generation (kW or MW) one step ahead (e.g. next minute/hour) given current and past data.  
- **Inputs (X):** time index and meteorological/system features (e.g. timestamp, ambient irradiance, ambient temperature, module temperatures, wind speed, etc.).  
- **Output (y):** PV power generation (DC or AC output) at the next time step.  

Thus we will build regression models to map environmental/time features to future PV output.

## Step 5: Algorithm Selection  
We will compare a range of models from simple to advanced:  
- **Linear Regression** – simple baseline.  
- **Random Forest (RF)** or **XGBoost** – ensemble tree-based model.  
- **Support Vector Regression (SVR)** – non-linear ML model.  
- **LSTM (Long Short-Term Memory network)** – RNN for sequence forecasting.  
- **1D Convolutional Neural Network (CNN)** – to capture local temporal patterns (or CNN+LSTM hybrid).  

This covers a spectrum: a simple linear baseline, tree ensemble (RF) which literature finds strong【73†L89-L97】, and deep models (LSTM/CNN) shown to improve accuracy【15†L64-L72】【79†L210-L218】.  

**Final algorithms:** Linear Regression, Random Forest (or XGBoost), LSTM, and 1D-CNN.

## Step 6: Evaluation Metrics  
Models will be compared using standard regression error metrics.  Specifically:  
- **Root Mean Square Error (RMSE)** and **Mean Absolute Error (MAE)** – primary accuracy measures (both used widely for solar forecasting【79†L210-L218】).  
- **Mean Absolute Percentage Error (MAPE)** – to gauge relative error, and possibly **$R^2$ or Pearson correlation** as a goodness-of-fit measure.  

We adopt RMSE/MAE as fixed metrics for ranking models (they quantify typical error in watts or kW)【79†L210-L218】【80†L37-L41】.  MAPE and correlation will provide additional insight【80†L37-L41】.

## Step 7: Comparison Plan  
I will compare models based on:  
- **Predictive accuracy:** RMSE and MAE on test data (lower is better)【79†L210-L218】.  MAPE and correlation $r$ will also be checked【80†L37-L41】.  
- **Computational efficiency:** training time and inference speed (important for real-time forecasts).  
- **Robustness:** stability of performance (e.g. consistency of error across different conditions, variance over cross-validation folds).  

Models will be ranked primarily by error metrics (RMSE/MAE) since the goal is accurate forecasting【79†L210-L218】, with secondary consideration for efficiency and robustness.  

**Sources:** The above choices and evaluations are guided by recent studies in solar forecasting【15†L64-L72】【73†L89-L97】【79†L210-L218】【80†L37-L41】, which report on model performance and metrics.