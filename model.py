from keras import Input, Model
from keras.layers import Conv2D
from keras.initializers import random_normal


def get_model(c, n1, n2, f1, f2, f3, fsub):
    input = Input(shape=(fsub, fsub, c))
    x = Conv2D(filters=n1, kernel_size=f1, activation='relu', kernel_initializer=random_normal(stddev=1e-3))(input)
    x = Conv2D(filters=n2, kernel_size=f2, activation='relu', kernel_initializer=random_normal(stddev=1e-3))(x)
    x = Conv2D(filters=c,  kernel_size=f3, activation='relu', kernel_initializer=random_normal(stddev=1e-3))(x)
    return Model(inputs=input, outputs=x)


def get_model_default():
    return get_model(c=1, n1=64, n2=32, f1=9, f2=1, f3=5, fsub=33)
