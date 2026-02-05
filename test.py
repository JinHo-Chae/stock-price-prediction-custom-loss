import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import yfinance as yf

from src.data import normalize_test, create_dataset_test
from src.loss import custom_loss, price_mse_metric, direction_loss_metric
from src.io import load_stats
from src.scaling import z_score_denormalize

def main():
    stock = "000990.KS"
    test_ID = 3
    test_start = "2025-01-01"

    window = 10
    horizon = 4

    all_stats = load_stats("artifacts/stats.json")
    train_stats = all_stats[stock]

    model = keras.models.load_model(
        "artifacts/model.keras",
        custom_objects={
            "loss_fn": custom_loss(lambda_direction=3.0, k=1.0),
            "price_mse_metric": price_mse_metric,
            "direction_loss_metric": direction_loss_metric,
        }
    )

    test_raw = yf.download(stock, start=test_start)
    date = test_raw.index.date

    test_df, test_vol = normalize_test(test_raw, train_stats)
    X_test, y_test, X2_test = create_dataset_test(test_df, test_vol, window=window, horizon=horizon)
    X3_test = np.full((X_test.shape[0], 1), test_ID)

    y_pred = model.predict([X_test, X2_test, X3_test], verbose=0)

    open_m, open_s   = train_stats["open"]
    high_m, high_s   = train_stats["high"]
    low_m, low_s     = train_stats["low"]
    close_m, close_s = train_stats["close"]

    # real (원본 가격)
    real_open  = test_raw["Open"].to_numpy()[window:]
    real_high  = test_raw["High"].to_numpy()[window:]
    real_low   = test_raw["Low"].to_numpy()[window:]
    real_close = test_raw["Close"].to_numpy()[window:]

    # pred (정규화 space → 원가격)
    pred_open  = z_score_denormalize(y_pred[:,0,0], open_m, open_s)
    pred_high  = z_score_denormalize(y_pred[:,0,1], high_m, high_s)
    pred_low   = z_score_denormalize(y_pred[:,0,2], low_m, low_s)
    pred_close = z_score_denormalize(y_pred[:,0,3], close_m, close_s)

    plt.figure(figsize=(16, 12))
    titles = ["Open", "High", "Low", "Close"]
    real_values = [real_open, real_high, real_low, real_close]
    pred_values = [pred_open, pred_high, pred_low, pred_close]

    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(date[window:], real_values[i], label="Real", linewidth=2)
        plt.plot(date[window:], pred_values[i], label="Pred", linewidth=2)
        plt.title(f"{titles[i]} - Real vs Pred")
        plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("✅ test done")

if __name__ == "__main__":
    main()
