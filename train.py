import tensorflow as tf
from comet_ml import Experiment
from keras.losses import mean_squared_error

import settings
from LRMultiplierSGD import LRMultiplierSGD
from data_load import load_data
from model import get_model

experiment = Experiment(api_key="o1YplG6lPqzCodSyFRtQvkys1", project_name="srcnn")


def psnr(y_true, y_pred):  # peak signal-to-noise ratio
    mse = mean_squared_error(y_true, y_pred)
    mse = tf.reduce_mean(mse)
    return 10 * log10(1 / mse)


def log10(x):
    n = tf.log(x)
    d = tf.log(tf.constant(10, dtype=n.dtype))
    return n / d


def main():
    model = get_model(c=1, n1=64, n2=32, f1=9, f2=1, f3=5, fsub=33)
    model.summary()

    x, y = load_data('../data/91sr/sub')

    opt = LRMultiplierSGD(lr=1e-4, momentum=0.9, multipliers=[1, 1, 1, 1, 0.1, 0.1])
    model.compile(optimizer=opt, loss=mean_squared_error, metrics=[psnr])
    model.fit(x, y, batch_size=settings.BATCH_SIZE, epochs=settings.EPOCHS)


if __name__ == "__main__":
    main()
