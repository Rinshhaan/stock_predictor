# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
import requests
import pickle
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Prophet may not be installed in your environment by default.
# Install it with: pip install prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# --------- CONFIG ----------
st.set_page_config(page_title="Stock Predictor + News", layout="wide")

# --------- FINNHUB (News) ----------
# Replace with your Finnhub API key. If empty, the app falls back to a sample.
FINNHUB_API_KEY = "d3443fhr01qqt8snegsgd3443fhr01qqt8snegt0"  # <-- replace with your real key
FINNHUB_NEWS_URL = f"https://finnhub.io/api/v1/business-insider-news?token={FINNHUB_API_KEY}"

def fetch_finnhub_news(limit=30):
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY":
        # fallback sample
        return [{
            "headline": "Sample market news headline",
            "summary": "Sample description. Place your Finnhub API key into the code to fetch live news.",
            "url": "https://example.com",
            "datetime": pd.Timestamp.now().isoformat()
        }]
    try:
        r = requests.get(FINNHUB_NEWS_URL, timeout=10)
        if r.status_code == 200:
            return r.json()[:limit]
        else:
            return [{
                "headline": f"News fetch failed: {r.status_code}",
                "summary": "",
                "url": "",
                "datetime": ""
            }]
    except Exception as e:
        return [{
            "headline": f"News fetch error: {e}",
            "summary": "",
            "url": "",
            "datetime": ""
        }]

# --------- DATA & UTIL ----------
@st.cache_data
def list_csv_files(data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]

@st.cache_data
def load_csv(filepath):
    df = pd.read_csv(filepath)
    # try to parse Date column
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        # normalize Date column name to 'Date'
        if date_col != "Date":
            df = df.rename(columns={date_col: "Date"})
    else:
        # if no date present, create one
        df["Date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
    # ensure Close exists
    if "Close" not in df.columns:
        raise ValueError("CSV must contain a 'Close' column")
    df = df.reset_index(drop=True)
    return df

# Technical indicators used for RF features
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

# --------- FEATURE ENGINEERING & RF ----------
def prepare_features(df, target_col="Close", lags=5, rolling_windows=(3,7,14), add_indicators=True):
    df2 = df.copy().sort_values("Date").reset_index(drop=True)
    if add_indicators:
        df2["EMA_12"] = ema(df2[target_col], 12)
        df2["EMA_26"] = ema(df2[target_col], 26)
        macd_line, macd_sig, macd_hist = macd(df2[target_col])
        df2["MACD"] = macd_line
        df2["MACD_signal"] = macd_sig
        df2["MACD_hist"] = macd_hist
        df2["RSI_14"] = rsi(df2[target_col], 14)
    X = pd.DataFrame()
    for lag in range(1, lags+1):
        X[f"lag_{lag}"] = df2[target_col].shift(lag)
    for w in rolling_windows:
        X[f"roll_mean_{w}"] = df2[target_col].rolling(window=w).mean().shift(1)
        X[f"roll_std_{w}"] = df2[target_col].rolling(window=w).std().shift(1)
    # include other numeric columns if present
    numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
    for c in numeric_cols:
        if c != target_col:
            X[c] = df2[c]
    y = df2[target_col]
    X = X.dropna()
    y = y.loc[X.index]
    # keep Date aligned for plotting later
    X["Date"] = df2.loc[X.index, "Date"].values
    return X.reset_index(drop=True), y.reset_index(drop=True)

def train_random_forest(X, y):
    feat_cols = [c for c in X.columns if c not in ("Date",)]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feat_cols].values)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y.values)
    return model, scaler, feat_cols

def iterative_forecast_rf(last_df, model, scaler, feat_cols, horizon, lags=5, rolling_windows=(3,7,14), add_indicators=True):
    """
    Iteratively forecast horizon days using RF.
    last_df: original dataframe with Date & Close (and other relevant numeric cols)
    """
    df = last_df.copy().sort_values("Date").reset_index(drop=True)
    preds = []
    for step in range(horizon):
        X_all, _ = prepare_features(df, lags=lags, rolling_windows=rolling_windows, add_indicators=add_indicators)
        if X_all.empty:
            raise ValueError("Not enough history to build features for iterative forecasting.")
        X_next = X_all[feat_cols].iloc[-1:].values
        X_next_scaled = scaler.transform(X_next)
        y_pred = model.predict(X_next_scaled)[0]
        next_date = df["Date"].iloc[-1] + pd.Timedelta(days=1)
        # append predicted row (carry forward non-Close numeric columns if any)
        new_row = {"Date": next_date, "Close": y_pred}
        for col in df.columns:
            if col not in ("Date", "Close") and pd.api.types.is_numeric_dtype(df[col]):
                new_row[col] = df[col].iloc[-1]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        preds.append((next_date, y_pred))
    pred_df = pd.DataFrame(preds, columns=["ds", "yhat"])
    return pred_df

# --------- PROPHET FORECAST ----------
def prophet_forecast(df, horizon):
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not installed. Install with: pip install prophet")
    df_prop = df[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"})
    m = Prophet(daily_seasonality=True)
    # add regressors or holidays here if needed
    m.fit(df_prop)
    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)  # contains yhat, yhat_lower, yhat_upper
    return forecast

# --------- SUGGESTION RULE ----------
def suggestion_from_prices(current_price, predicted_price, investor_type="long"):
    # Simple heuristic thresholds (tunable)
    expected_return = (predicted_price - current_price) / current_price
    if investor_type == "long":
        if expected_return > 0.25:
            return "Buy (Long-term) ✅ — expects >25% gain"
        elif expected_return > 0.05:
            return "Hold (Long-term) 🟡 — moderate expected gain"
        else:
            return "Sell (Long-term) ❌ — little or negative expected gain"
    else:  # short-term
        if expected_return > 0.05:
            return "Buy (Short-term) ✅ — good for trading"
        elif expected_return > 0.01:
            return "Hold (Short-term) 🟡 — weak short-term move"
        else:
            return "Sell (Short-term) ❌ — not attractive for trading"

# --------- UI: Sidebar Controls ----------
st.sidebar.title("Settings")
csv_files = list_csv_files("data")
if not csv_files:
    st.sidebar.error("No CSV files found in data/ — drop your OHLCV CSVs there (aapl.csv, tsla.csv, etc.).")
selected_csv = st.sidebar.selectbox("Select Stocks", csv_files) if csv_files else None

model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Prophet (time-series)"])
timeframe = st.sidebar.selectbox("Forecast timeframe", [
    "1 Week", "1 Month", "6 Months", "1 Year", "3 Years", "10 Years", "10+ Years"
])
timeframe_map = {
    "1 Week": 7,
    "1 Month": 30,
    "6 Months": 180,
    "1 Year": 365,
    "3 Years": 365*3,
    "10 Years": 365*10,
    "10+ Years": 365*15
}
forecast_horizon = timeframe_map[timeframe]

add_indicators = st.sidebar.checkbox("Add technical indicators for RF (EMA/MACD/RSI)", value=True)
use_scaler = st.sidebar.checkbox("Scale features (RF)", value=True)



# --------- MAIN: Home (Prediction) ----------
st.title("📈 Stock Prediction Dashboard — Home")

if selected_csv is None:
    st.warning("No CSV selected. Place stock CSV files in the `data/` folder and refresh.")
    st.stop()

# Load selected CSV
data_path = os.path.join("data", selected_csv)
try:
    df_raw = load_csv(data_path)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

st.subheader(f"Loaded: {selected_csv}")
st.dataframe(df_raw.tail(6))

# Live price via yfinance (try to guess ticker from filename, fallback to last CSV Close)
ticker_guess = os.path.splitext(selected_csv)[0].upper().split("_")[0]
live_price = None
try:
    t = yf.Ticker(ticker_guess)
    hist = t.history(period="1d", interval="1m")
    if not hist.empty:
        live_price = hist["Close"].iloc[-1]
except Exception:
    live_price = None

if live_price is None:
    live_price = df_raw["Close"].iloc[-1]

st.metric(label=f"{ticker_guess} — Current price (realtime via yfinance or CSV last)", value=f"{live_price:.2f}")

# Historical interactive chart
st.subheader("Historical Price (interactive)")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df_raw["Date"], y=df_raw["Close"], mode="lines", name="Historical Close"))
fig_hist.update_layout(height=420, xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_hist, use_container_width=True)

# MODEL: Train & Forecast
st.subheader(f"Forecast ({model_choice}) — {timeframe}")

if model_choice == "Random Forest":
    # Prepare RF features & train
    with st.spinner("Preparing features and training Random Forest..."):
        X, y = prepare_features(df_raw, target_col="Close", lags=7, rolling_windows=(7,14,30), add_indicators=add_indicators)
        if X.empty or len(X) < 50:
            st.error("Not enough data after feature creation. Try a different CSV or reduce lags/windows.")
        else:
            # drop Date from training features but keep order
            feat_cols = [c for c in X.columns if c not in ("Date",)]
            model, scaler, trained_feat_cols = train_random_forest(X, y)
            # iterative forecast
            with st.spinner(f"Generating {forecast_horizon}-day forecast with Random Forest..."):
                try:
                    preds_rf = iterative_forecast_rf(df_raw, model, scaler, trained_feat_cols, horizon=forecast_horizon, lags=7, rolling_windows=(7,14,30), add_indicators=add_indicators)
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
                    preds_rf = pd.DataFrame(columns=["ds","yhat"])
            # Plot historical + forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_raw["Date"], y=df_raw["Close"], mode="lines", name="Historical"))
            if not preds_rf.empty:
                fig.add_trace(go.Scatter(x=preds_rf["ds"], y=preds_rf["yhat"], mode="lines+markers", name="RF Forecast"))
            fig.update_layout(height=480, xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
            # predicted price and suggestion
            if not preds_rf.empty:
                predicted_price = preds_rf["yhat"].iloc[-1]
                st.metric("Predicted price (end of horizon)", f"{predicted_price:.2f}")
                st.write("**Suggestions (heuristic)**")
                st.write(suggestion_from_prices(live_price, predicted_price, investor_type="long"))
                st.write(suggestion_from_prices(live_price, predicted_price, investor_type="short"))
                # show simple RF metrics on test split
                # quick approximate test: last 10% as test
                try:
                    split_idx = int(len(y) * 0.9)
                    X_train_vals = X[trained_feat_cols].values[:split_idx]
                    X_test_vals = X[trained_feat_cols].values[split_idx:]
                    if use_scaler:
                        X_train_vals = scaler.transform(X_train_vals)
                        X_test_vals = scaler.transform(X_test_vals)
                    y_train_vals = y.values[:split_idx]
                    y_test_vals = y.values[split_idx:]
                    y_pred_test = model.predict(X_test_vals)
                    rmse = (mean_squared_error(y_test_vals, y_pred_test))**0.5
                    mae = mean_absolute_error(y_test_vals, y_pred_test)
                    st.write(f"Model test RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                except Exception:
                    pass
            # allow model download
            try:
                with open("rf_model.pkl", "wb") as f:
                    pickle.dump({"model": model, "scaler": scaler, "features": trained_feat_cols}, f)
                st.download_button("Download Random Forest model (pickle)", data=open("rf_model.pkl","rb"), file_name="rf_model.pkl")
            except Exception:
                pass

elif model_choice == "Prophet (time-series)":
    if not PROPHET_AVAILABLE:
        st.error("Prophet library not available. Install with: pip install prophet")
    else:
        with st.spinner("Fitting Prophet and forecasting..."):
            try:
                forecast = prophet_forecast(df_raw, forecast_horizon)
                # Plot historical + forecast
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_raw["Date"], y=df_raw["Close"], mode="lines", name="Historical"))
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Prophet Forecast"))
                # uncertainty band
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], fill="tonexty", line=dict(width=0), name="Uncertainty"))
                fig.update_layout(height=480, xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
                predicted_price = forecast["yhat"].iloc[-1]
                st.metric("Predicted price (end of horizon)", f"{predicted_price:.2f}")
                st.write("**Suggestions (heuristic)**")
                st.write(suggestion_from_prices(live_price, predicted_price, investor_type="long"))
                st.write(suggestion_from_prices(live_price, predicted_price, investor_type="short"))
                # save prophet model? (Prophet objects are picklable)
                try:
                    with open("prophet_model.pkl", "wb") as f:
                        pickle.dump(forecast, f)
                    st.download_button("Download Prophet forecast (pickle)", data=open("prophet_model.pkl","rb"), file_name="prophet_forecast.pkl")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Prophet forecast failed: {e}")




# --------- FOOTER / DISCLAIMER ----------
st.markdown("""
### ⚠️ Important Notice — Read carefully
- **Predictions are not guaranteed.** They are based on historical patterns and statistical models.
- **Real-world events** (new tax rules, regulatory changes, company earnings surprises, geopolitical events, or market breakdowns) can change prices quickly and invalidate predictions.
- **Do not rely solely on these predictions** for investment decisions. Always consult up-to-date news, company fundamentals, and financial professionals.
- **Use this app as one tool among many** — combine technical forecasts, news, risk management, and your own research.
- For serious backtesting / trading, include transaction costs, slippage, taxes, and a proper risk model.
""")
