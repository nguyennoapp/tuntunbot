#!/usr/bin/env python3

import ccxt
import os
import numpy as np
from random import shuffle
from configparser import ConfigParser
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense

TIME_FRAMES = ['15m', '1h', '4h', '1d']
FEATURE_SIZE = 5
STEP_SIZE = 5
LSTM_SIZE = FEATURE_SIZE * STEP_SIZE
LSTM_MODEL = 'lstm'
SAVE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = '{}/config.ini'.format(SAVE_PATH)

config = ConfigParser()
config.read(CONFIG_FILE)
exchange = ccxt.binance({'apiKey': config['CONFIG']['API_KEY'], 'secret': config['CONFIG']['API_SECRET']})


def open_model(new=False):
    name = LSTM_MODEL
    model = None
    filename = '{}/{}.h5'.format(SAVE_PATH, name)
    print('load LSTM model from file: {}'.format(filename))
    try:
        model = load_model(filename)
        print('load LSTM model from file success: {}'.format(name))
    except Exception as e:
        print('load LSTM model from file error: {}'.format(str(e)))
    new_model = Sequential()
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True,
                       batch_input_shape=(LSTM_SIZE * 100, STEP_SIZE, FEATURE_SIZE)))
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    new_model.add(LSTM(FEATURE_SIZE, stateful=True))
    new_model.add(Dense(FEATURE_SIZE, activation='linear'))
    new_model.add(Dense(1))
    new_model.compile(optimizer='adagrad', metrics=['accuracy'], loss='mse')
    if not new and model is not None:
        new_model.set_weights(model.get_weights())
    else:
        print('create LSTM model success: {}'.format(name))
    return new_model


def save_model(model):
    new_model = Sequential()
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True,
                       batch_input_shape=(1, STEP_SIZE, FEATURE_SIZE)))
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    new_model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    new_model.add(LSTM(FEATURE_SIZE, stateful=True))
    new_model.add(Dense(FEATURE_SIZE, activation='linear'))
    new_model.add(Dense(1))
    new_model.compile(optimizer='adagrad', metrics=['accuracy'], loss='mse')  # adagrad
    new_model.set_weights(model.get_weights())
    name = LSTM_MODEL
    filename = '{}/{}.h5'.format(SAVE_PATH, name)
    try:
        print('save LSTM model to file: {}'.format(filename))
        new_model.save(filename)
        print('save LSTM model to file success')
    except Exception as e:
        print('save LSTM model to file error: {}'.format(str(e)))


def symbol_data(symbol, time_frame):
    data = exchange.fetch_ohlcv(symbol, time_frame)
    df = DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df.set_index('time')
    df.replace({0: np.nan}, inplace=True)
    df['price'] = df[['open', 'high', 'low', 'close']].mean(axis=1)
    df['price_change'] = df['price'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df = df.assign(**{'volatility': lambda x: (x['high'] - x['low']) / x['open']})
    df = df.assign(**{'convergence': lambda x: (x['open'] - x['close']) / (x['high'] - x['low'])})
    df = df.assign(**{'predisposition': lambda x: 1 - 2 * (x['high'] - x['close']) / (x['high'] - x['low'])})
    df.dropna(axis=0, how='any', inplace=True)
    sc = MinMaxScaler(feature_range=(-1, 1))
    na = sc.fit_transform(df[['price_change', 'volume_change', 'volatility', 'convergence', 'predisposition']])
    return na, na[:, 0]


def save_data():
    exchange.load_markets(reload=True)
    train_x = []
    train_y = []
    for time_frame in TIME_FRAMES:
        print('start collect data with time frame: {}'.format(time_frame))
        symbols = exchange.symbols
        shuffle(symbols)
        for symbol in symbols:
            print('start collect data from symbol: {}'.format(symbol))
            input_data, output_data = symbol_data(symbol, time_frame)
            if len(input_data) == 0:
                continue
            for i in range(len(input_data) - STEP_SIZE - 1):
                train_x.append(input_data[i:i + STEP_SIZE])
                train_y.append(output_data[i + STEP_SIZE])
    length = int(len(train_x) // (LSTM_SIZE * 100) * (LSTM_SIZE * 100))
    train_x = train_x[-length:]
    train_y = train_y[-length:]
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    np.save('{}/train_x.npy'.format(SAVE_PATH), train_x)
    np.save('{}/train_y.npy'.format(SAVE_PATH), train_y)
    print('save train data success')


def train():
    model = open_model()
    train_x = np.load('{}/train_x.npy'.format(SAVE_PATH))
    train_y = np.load('{}/train_y.npy'.format(SAVE_PATH))
    print('train LSTM model with data length: %g' % len(train_x))
    model.fit(train_x, train_y, epochs=LSTM_SIZE * 10, batch_size=LSTM_SIZE * 100, verbose=1)
    print('train LSTM model success')
    save_model(model)


def main():
    save_data()
    train()


if __name__ == "__main__":
    main()