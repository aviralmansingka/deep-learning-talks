from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.utils import np_utils


def build_model(nb_words):
    model = Sequential()
    model.add(Dense(100, input_dim=nb_words, init='uniform', activation='relu'))
    # model.add(Dense(50, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
