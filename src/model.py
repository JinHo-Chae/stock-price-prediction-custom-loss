from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Bidirectional

def build_model(window: int, num_stocks=5, embedding_dim=32, lr=1e-3, loss_fn=None, metrics=None):
    input_price = Input(shape=(window, 4), name="price_input")
    input_volume = Input(shape=(window, 1), name="volume_input")
    input_stock_id = Input(shape=(1,), dtype="int32", name="stock_id_input")

    # price branch
    x1 = layers.Conv1D(32, 3, activation="sigmoid")(input_price)
    x1 = Bidirectional(layers.LSTM(64, return_sequences=True))(x1)
    x1 = layers.Dropout(0.1)(x1)
    x1 = Bidirectional(layers.LSTM(64))(x1)
    x1 = layers.Dense(256)(x1)
    x1 = layers.LeakyReLU(alpha=0.2)(x1)

    # volume branch
    x2 = layers.Conv1D(32, 3, activation="sigmoid")(input_volume)
    x2 = Bidirectional(layers.LSTM(64, return_sequences=True))(x2)
    x2 = layers.Dropout(0.1)(x2)
    x2 = Bidirectional(layers.LSTM(64))(x2)
    x2 = layers.Dense(128)(x2)
    x2 = layers.LeakyReLU(alpha=0.2)(x2)

    # stock id embedding
    e = layers.Embedding(input_dim=num_stocks, output_dim=embedding_dim)(input_stock_id)
    e = layers.Flatten()(e)
    e = layers.Dense(64)(e)
    e = layers.LeakyReLU(alpha=0.2)(e)

    merged = layers.concatenate([x1, x2, e], axis=-1)
    merged = layers.Dense(256)(merged)
    merged = layers.LeakyReLU(alpha=0.2)(merged)
    merged = layers.Dense(16)(merged)
    output = layers.Reshape((4, 4))(merged)

    model = Model(inputs=[input_price, input_volume, input_stock_id], outputs=output)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics or [])
    return model
