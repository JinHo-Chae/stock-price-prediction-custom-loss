import os
import numpy as np
from tensorflow import keras

from src.data import download_stock, normalize_train, make_df, create_dataset
from src.loss import custom_loss, price_mse_metric, direction_loss_metric
from src.model import build_model
from src.io import save_stats

def main():
    os.makedirs("artifacts", exist_ok=True)

    train_start = "2020-01-01"
    train_end   = "2024-12-31"

    tickers = {
        "019170.KS": 0,
        "090360.KQ": 1,
        "023910.KQ": 2,
        "000990.KS": 3,
        "210980.KS": 4,
    }

    window = 10
    horizon = 4

    all_X, all_y, all_X2, all_X3 = [], [], [], []
    all_stats = {}

    for ticker, stock_id in tickers.items():
        raw = download_stock(ticker, start=train_start, end=train_end)

        o, h, l, c, v, stats = normalize_train(raw)
        all_stats[ticker] = stats

        df = make_df(o, h, l, c)
        X, y, X2 = create_dataset(df, v, window=window, horizon=horizon)
        X3 = np.full((X.shape[0], 1), stock_id)

        all_X.append(X); all_y.append(y); all_X2.append(X2); all_X3.append(X3)

    X_train  = np.concatenate(all_X, axis=0)
    y_train  = np.concatenate(all_y, axis=0)
    X2_train = np.concatenate(all_X2, axis=0)
    X3_train = np.concatenate(all_X3, axis=0)

    model = build_model(
        window=window,
        num_stocks=len(tickers),
        embedding_dim=32,
        lr=1e-3,
        loss_fn=custom_loss(lambda_direction=3.0, k=1.0),
        metrics=[price_mse_metric, direction_loss_metric]
    )

    model.fit([X_train, X2_train, X3_train], y_train, epochs=50, batch_size=32)

    model.save("artifacts/model.keras")
    save_stats("artifacts/stats.json", all_stats)

    print("✅ saved: artifacts/model.keras, artifacts/stats.json")

if __name__ == "__main__":
    main()
