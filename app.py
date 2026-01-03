import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import plotly.express as px
import plotly.graph_objects as go

from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Traffic Forecasting Dashboard – ARIMA vs TFT",
    page_icon="🚦",
    layout="wide"
)

# ----------------------------------------------------------


st.title("Traffic Volume Forecasting Dashboard")
st.markdown(
        """
        <div style="padding-top: 10px;padding-bottom: 20px;padding-left: 5px;">
            <a href="https://github.com/a-anuj/tft-traffic-volume-analysis" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" 
                     width="30" title="View on GitHub">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown("""
This interactive dashboard compares **ARIMA** and **Temporal Fusion Transformer (TFT)** models  
for **multi-step traffic forecasting** on the Metro Interstate dataset.
""")

    


# ----------------------------------------------------------
# LOAD MODELS & DATA
# ----------------------------------------------------------
@st.cache_resource
def load_arima():
    return pickle.load(open("models/arima.pkl", "rb"))

def load_tft():
    return TemporalFusionTransformer.load_from_checkpoint(
        "models/tft.ckpt",
        map_location="cpu"
    )


@st.cache_resource
def load_test_df():
    return pd.read_csv("data/test_data.csv")


arima_model = load_arima()
tft_model = load_tft()
test_df = load_test_df()

# ----------------------------------------------------------
# SIDEBAR: CONTROLS
# ----------------------------------------------------------
st.sidebar.header("Settings")
forecast_hours = st.sidebar.slider("Forecast horizon (hours)", 1, 24, 24)

# ----------------------------------------------------------
# RUN PREDICTIONS
# ----------------------------------------------------------
def predict_arima(horizon):
    return arima_model.forecast(steps=horizon)

def predict_tft(horizon):
    # ---- Load FULL dataset object (has all training metadata) ----
    with open("models/full_dataset.pkl", "rb") as f:
        full_dataset = pickle.load(f)

    # ---- Ensure categorical columns are strings ----
    cat_cols = ["weather_main_id", "weather_desc_id", "holiday_id"]
    for col in cat_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str)

    # ---- Build test dataset from FULL dataset ----
    test_dataset = TimeSeriesDataSet.from_dataset(
        full_dataset,
        test_df,
        predict=True,
        stop_randomization=True
    )

    # ---- Create dataloader ----
    test_loader = test_dataset.to_dataloader(train=False, batch_size=1)

    # ---- Predict ----
    preds = tft_model.predict(test_loader)
    preds = preds.detach().cpu().numpy()[0]

    return preds[:horizon]




# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "📈 ARIMA Forecast", "🤖 TFT Forecast", "⚔ Model Comparison"])

# ----------------------------------------------------------
# TAB 1 — OVERVIEW
# ----------------------------------------------------------
with tab1:

    col1, col2 = st.columns([1.2, 1])

    # LEFT SIDE -----------------------------------------
    with col1:
        st.markdown("""
### What This Dashboard Does

This dashboard demonstrates a complete **end-to-end traffic forecasting system** using:

- **ARIMA** — a classical statistical model  
- **TFT (Temporal Fusion Transformer)** — a state-of-the-art deep learning model  

You can interactively compare both approaches and understand why TFT  
delivers superior performance for **multi-step traffic forecasting**.

---

### Real-Life Importance of Traffic Forecasting

Accurate traffic forecasting is critical for modern cities:

#### 1. **Smart Traffic Signals**
Forecasted flow prevents congestion by adjusting green-time dynamically.

#### 2. **Route Optimization**
Maps (Google Maps, Waze) use traffic predictions for ETA accuracy.

#### 3. **Public Transport Planning**
Buses/trains can be scheduled based on expected rush-hour load.

#### 4. **Emergency Response**
Ambulances & fire services can plan fastest routes in advance.

#### 5. **Pollution Reduction**
Traffic jams increase emissions — forecasting helps avoid peak congestion.

This project replicates the core logic used in **Intelligent Transportation Systems (ITS)** and **Smart City Solutions**.
        """)

    # RIGHT SIDE ----------------------------------------
    with col2:
        st.markdown("### 📊 Dataset Summary")
        st.info("""
**Dataset:** Metro Interstate Traffic Volume  
**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume  
""")

        st.metric("Total Records", "48,205 rows")
        st.metric("Features", str(test_df.shape[1]))

        st.markdown("---")

        st.markdown("### 🔍 Model Summary")
        st.warning("""
**ARIMA**
- Univariate  
- Assumes linear relationships  
- Works for short-term, simple patterns  

**TFT**
- Multi-variable + attention  
- Learns seasonality, weather effects, and rush-hour patterns  
- Ideal for long-range forecasting  
""")

        st.markdown("---")

        st.markdown("### Key Insight")
        st.success("""
TFT extracts deep temporal patterns from **time, weather, seasonality, and lag features**,  giving highly reliable 24-hour forecasts — 
                   far outperforming ARIMA.
""")



# ----------------------------------------------------------
# TAB 2 — ARIMA FORECAST
# ----------------------------------------------------------
with tab2:
    st.subheader("📈 ARIMA Forecast")

    arima_pred = predict_arima(forecast_hours)

    df_arima = pd.DataFrame({
        "Hour": np.arange(1, forecast_hours + 1),
        "Traffic Volume": arima_pred
    })
    df_arima["Traffic Volume"] = pd.to_numeric(df_arima["Traffic Volume"], errors="coerce")
    df_arima = df_arima.dropna()


    fig = px.line(
        df_arima,
        x="Hour",
        y="Traffic Volume",
        title="ARIMA: Traffic Volume vs Hour",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_arima, use_container_width=True)

# ----------------------------------------------------------
# TAB 3 — TFT FORECAST
# ----------------------------------------------------------
with tab3:
    st.subheader("🤖 TFT Forecast")

    tft_pred = predict_tft(forecast_hours)

    df_tft = pd.DataFrame({
        "Hour": np.arange(1, forecast_hours + 1),
        "Traffic Volume": tft_pred
    })

    df_tft["Traffic Volume"] = pd.to_numeric(df_tft["Traffic Volume"], errors="coerce")
    df_tft = df_tft.dropna()


    fig = px.line(
        df_tft,
        x="Hour",
        y="Traffic Volume",
        title="TFT: Traffic Volume vs Hour",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_tft, use_container_width=True)

# ----------------------------------------------------------
# TAB 4 — MODEL COMPARISON
# ----------------------------------------------------------
with tab4:
    st.subheader("⚔ ARIMA vs TFT Comparison")

    arima_pred = predict_arima(forecast_hours)
    tft_pred = predict_tft(forecast_hours)

    df_compare = pd.DataFrame({
        "Hour": np.arange(1, forecast_hours + 1),
        "ARIMA": arima_pred,
        "TFT": tft_pred
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_compare["Hour"], y=df_compare["ARIMA"],
                             mode='lines+markers', name='ARIMA'))
    fig.add_trace(go.Scatter(x=df_compare["Hour"], y=df_compare["TFT"],
                             mode='lines+markers', name='TFT'))

    fig.update_layout(
        title="ARIMA vs TFT Forecast Curve",
        xaxis_title="Hour Ahead",
        yaxis_title="Traffic Volume",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_compare, use_container_width=True)

    st.markdown("""
    ### 📊 Key Insight

    **TFT dramatically outperforms ARIMA across all forecast horizons.**

    #### ARIMA Performance
    - Overall RMSE: **≈ 1989**

    #### TFT Performance
    - Overall RMSE: **≈ 287**
    - RMSE at **t+1 hour:** ≈ **169**
    - RMSE at **t+6 hours:** ≈ **457**
    - RMSE at **t+24 hours:** ≈ **63**

    #### Why this matters
    - TFT learns complex temporal patterns (seasonality, rush-hour impact, weather correlations).
    - ARIMA struggles because it is **univariate** and cannot use covariates.
    - TFT provides **more stable long-range forecasts**, with t+24 RMSE being the lowest.

    **Bottom line:**  
    The TFT model reduces forecasting error by **over 85%** compared to ARIMA,  
    making it far more reliable for real-world traffic prediction.
    """)




