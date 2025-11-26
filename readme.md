# Traffic Volume Forecasting using ARIMA & Temporal Fusion Transformer (TFT)

This project demonstrates an end-to-end **traffic forecasting system** using:
- **ARIMA** – classical statistical model  
- **TFT (Temporal Fusion Transformer)** – state-of-the-art deep learning model  

It includes:
- Full preprocessing pipeline  
- Baseline model (ARIMA)  
- Advanced deep learning model (TFT)  
- Performance comparison  
- Streamlit web dashboard  



## 📘 Dataset

- **Source:** UCI Machine Learning Repository  
- **Name:** Metro Interstate Traffic Volume  
- **Link:** https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume  
- **Total Rows:** **48,205**
- **Frequency:** Hourly traffic volume  
- **Target Variable:** `traffic_volume`

This dataset contains weather, time, holidays, and hourly traffic flow measurements.


## 🎯 Project Objective

To build a robust multi-step (24-hour) traffic forecasting application and demonstrate why **TFT significantly outperforms ARIMA** for real-world traffic prediction.


## 🛠 Tech Stack

### **Machine Learning**
- ARIMA
- Temporal Fusion Transformer (PyTorch Forecasting)
- PyTorch Lightning

### **Data Engineering**
- Pandas, Numpy
- Feature engineering (lags, rolling windows, embeddings)

### **Deployment**
- Streamlit web app  
- Model checkpoint loading (no retraining required)



## 🧹 Data Preprocessing

Key feature engineering steps:

- Parsed timestamps → extracted:  
  `hour`, `weekday`, `weekend`, `month`, `rush_hour`
- Encoded categorical features:  
  `weather_main`, `weather_description`, `holiday`
- Added lag features:  
  - `lag_1` (previous hour)  
  - `lag_24` (previous day same hour)
- Added rolling averages:  
  - `roll_3` (3-hour mean)  
  - `roll_24` (24-hour mean)
- Train-val-test split:  
  - Train: 2012–2016  
  - Val: 2017  
  - Test: 2018  



## 📈 Models Implemented

### **1. ARIMA (Baseline)**
- Univariate  
- Fast to train  
- Struggles with non-linear seasonal behavior  

**ARIMA RMSE:**  ≈ 1989


---

### **2. Temporal Fusion Transformer (TFT)**
A powerful attention-based architecture for multi-horizon forecasting.

Uses:
- Covariates (weather, time, rolling features)
- Variable selection networks  
- Static & dynamic embeddings  
- Multi-head temporal attention  

**TFT Performance :**
- Overall RMSE ≈ 287
- t+1 RMSE ≈ 169
- t+6 RMSE ≈ 457
- t+24 RMSE ≈ 63


This represents **over 85% performance improvement** compared to ARIMA.



## Why This Project Matters (Real-World Use Cases)

Traffic forecasting is critical for:

### **Smart Traffic Signals**
Adaptive signal control reduces congestion.

### **Route Optimization**
Used by Google Maps & Waze for accurate ETA prediction.

### **Public Transport Scheduling**
Demand-aware bus/train scheduling.

### **Emergency Services**
Optimized routing for ambulances, fire services, and police.

### **Urban Planning**
Helps identify bottlenecks, plan flyovers, signal placements.

### **Pollution Reduction**
Less idling → lower emissions → healthier cities.

This project replicates the core logic used in **Intelligent Transportation Systems (ITS)** and **Smart City infrastructure**.


## 🖥 Streamlit Dashboard

The Streamlit app includes:

- ARIMA forecast visualization  
- TFT multi-horizon forecast  
- Model performance metrics  
- Comparison plots  
- Dataset overview  
- GitHub link and modern UI layout  

Run locally:

```bash
git clone https://github.com/a-anuj/tft-traffic-volume-analysis.git
cd tft-traffic-volume-analysis
streamlit run app.py
```

## Results Summary

- TFT outperforms ARIMA **across all forecast horizons**  
- Learns complex seasonal and weather-driven traffic patterns  
- Shows strong stability and accuracy for long-term predictions  
- Suitable for real-world deployment in smart transportation systems  
- Produces reliable and consistent 24-hour forecasts  


