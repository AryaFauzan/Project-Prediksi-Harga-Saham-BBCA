import os
import json
import joblib
import warnings
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


def download_data(ticker="BBCA.JK", period="10y"):
    """
    Mengambil data historis saham dari Yahoo Finance.
    auto_adjust=True digunakan agar harga sudah menyesuaikan corporate action.
    """
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError("Data kosong. Periksa koneksi internet atau ticker saham.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna()

    return df


def add_features(df):
    """
    Membuat fitur tambahan dari data OHLCV:
    - lag harga
    - return
    - moving average
    - volatilitas
    - volume ratio
    - rasio harga harian
    """
    df = df.copy()

    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"close_lag_{lag}"] = df["Close"].shift(lag)

    for period in [1, 3, 5, 10, 20]:
        df[f"return_{period}"] = df["Close"].pct_change(period)

    for window in [5, 10, 20, 50]:
        df[f"ma_{window}"] = df["Close"].rolling(window).mean()
        df[f"close_to_ma_{window}"] = (df["Close"] / df[f"ma_{window}"]) - 1

    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = df["Close"].pct_change().rolling(window).std()

    for window in [5, 10, 20]:
        df[f"volume_ma_{window}"] = df["Volume"].rolling(window).mean()
        df[f"volume_ratio_{window}"] = df["Volume"] / df[f"volume_ma_{window}"]

    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["open_close_ratio"] = df["Open"] / df["Close"]
    df["high_close_ratio"] = df["High"] / df["Close"]
    df["low_close_ratio"] = df["Low"] / df["Close"]
    df["candle_body"] = (df["Close"] - df["Open"]) / df["Open"]

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def prepare_training_data(raw_df):
    """
    Membuat target prediksi.
    Target = harga Close hari perdagangan berikutnya.
    """
    df = add_features(raw_df)

    df["target"] = df["Close"].shift(-1)

    df = df.dropna()

    return df


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


def main():
    os.makedirs("model", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    ticker = "BBCA.JK"

    print("Mengambil data BBCA dari Yahoo Finance...")
    raw_df = download_data(ticker=ticker, period="10y")
    raw_df.to_csv("data/bbca_raw.csv")

    print("Membuat fitur dan target...")
    df = prepare_training_data(raw_df)
    df.to_csv("data/bbca_features.csv")

    feature_cols = [col for col in df.columns if col != "target"]

    X = df[feature_cols]
    y = df["target"]

    # Split time series: 80% awal training, 20% akhir testing.
    # Tidak menggunakan random split agar tidak terjadi data leakage.
    train_size = int(len(df) * 0.8)

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]

    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    test_dates = df.index[train_size:]

    print(f"Jumlah data total   : {len(df)}")
    print(f"Jumlah data training: {len(X_train)}")
    print(f"Jumlah data testing : {len(X_test)}")

    # Menggunakan parameter terbaik dari hasil tuning sebelumnya untuk mempercepat training
    best_params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": None
    }

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            random_state=42,
            n_jobs=1,
            **best_params
        ))
    ])

    print("Melatih model Random Forest Regressor dengan parameter statis (Training Cepat)...")
    pipeline.fit(X_train, y_train)

    best_model = pipeline

    print("Training selesai.")
    
    # Membuat objek mock untuk search.best_params_ agar struktur kode penyimpanan metadata tidak error
    class MockSearch:
        pass
    search = MockSearch()
    search.best_params_ = {f"model__{k}": v for k, v in best_params.items()}

    y_pred = best_model.predict(X_test)

    metrics = calculate_metrics(y_test.values, y_pred)

    print("\nHasil Evaluasi Model:")
    print(f"MAE  : {metrics['MAE']:.2f}")
    print(f"RMSE : {metrics['RMSE']:.2f}")
    print(f"MAPE : {metrics['MAPE']:.2f}%")
    print(f"R2   : {metrics['R2']:.4f}")

    evaluation_df = pd.DataFrame([{
        "Model": "Random Forest Regressor",
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "R2": metrics["R2"],
        "Best Parameters": str(search.best_params_)
    }])

    evaluation_df.to_csv("output/evaluation_result.csv", index=False)

    prediction_df = pd.DataFrame({
        "Date": test_dates,
        "Current_Close": X_test["Close"].values,
        "Actual_Close_Next_Day": y_test.values,
        "Predicted_Close_Next_Day": y_pred,
        "Prediction_Error": y_test.values - y_pred
    })

    prediction_df.to_csv("output/prediction_result.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(prediction_df["Date"], prediction_df["Actual_Close_Next_Day"], label="Actual")
    plt.plot(prediction_df["Date"], prediction_df["Predicted_Close_Next_Day"], label="Predicted")
    plt.title("Actual vs Predicted Harga Penutupan Saham BBCA")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga Penutupan")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/actual_vs_predicted.png")
    plt.close()

    rf_model = best_model.named_steps["model"]

    feature_importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    feature_importance_df.to_csv("output/feature_importance.csv", index=False)

    joblib.dump(best_model, "model/bbca_random_forest_model.pkl")

    with open("model/features.json", "w") as f:
        json.dump(feature_cols, f, indent=4)

    metadata = {
        "project": "Prediksi Harga Saham BBCA",
        "ticker": ticker,
        "data_source": "Yahoo Finance via yfinance",
        "model": "Random Forest Regressor",
        "target": "Harga penutupan BBCA hari perdagangan berikutnya",
        "features": feature_cols,
        "split_method": "Time-based split, 80% training dan 20% testing",
        "best_parameters": search.best_params_,
        "metrics": {
            "MAE": float(metrics["MAE"]),
            "RMSE": float(metrics["RMSE"]),
            "MAPE": float(metrics["MAPE"]),
            "R2": float(metrics["R2"])
        },
        "last_data_date": str(df.index[-1].date()),
        "last_run_date": str(datetime.date.today())
    }

    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nFile berhasil dibuat:")
    print("model/bbca_random_forest_model.pkl")
    print("model/features.json")
    print("model/metadata.json")
    print("data/bbca_raw.csv")
    print("data/bbca_features.csv")
    print("output/evaluation_result.csv")
    print("output/prediction_result.csv")
    print("output/actual_vs_predicted.png")
    print("output/feature_importance.csv")


if __name__ == "__main__":
    main()