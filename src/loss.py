import tensorflow as tf

def custom_loss(lambda_direction=3.0, k=1.0):
    def loss_fn(y_true, y_pred):
        price_mse = tf.reduce_mean(tf.square(y_true - y_pred))

        true_delta = y_true[:, 1:, :] - y_true[:, :-1, :]
        pred_delta = y_pred[:, 1:, :] - y_pred[:, :-1, :]

        true_sign = tf.tanh(k * true_delta)
        pred_sign = tf.tanh(k * pred_delta)

        direction_loss = tf.reduce_mean(tf.square(true_sign - pred_sign))
        return price_mse + lambda_direction * direction_loss
    return loss_fn

def direction_loss_metric(y_true, y_pred, k=1.0):
    true_delta = y_true[:, 1:, :] - y_true[:, :-1, :]
    pred_delta = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    true_sign = tf.tanh(k * true_delta)
    pred_sign = tf.tanh(k * pred_delta)
    return tf.reduce_mean(tf.square(true_sign - pred_sign))

def price_mse_metric(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
