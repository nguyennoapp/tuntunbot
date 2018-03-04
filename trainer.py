#!/usr/bin/env python3

import ccxt
import os
import numpy as np
from tqdm import tqdm
from configparser import ConfigParser
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

TIME_FRAMES = ['5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
FEATURE_SIZE = 5
STEP_SIZE = 5
LSTM_SIZE = FEATURE_SIZE * STEP_SIZE
LSTM_MODEL = 'lstm'
SAVE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = '{}/config.ini'.format(SAVE_PATH)

config = ConfigParser()
config.read(CONFIG_FILE)
exchange = ccxt.binance({'apiKey': config['CONFIG']['API_KEY'], 'secret': config['CONFIG']['API_SECRET']})


def create_model(train=False):
    shape = 1
    if train:
        shape = LSTM_SIZE * 100
    model = Sequential()
    model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True, batch_input_shape=(shape, STEP_SIZE, FEATURE_SIZE)))
    model.add(LSTM(LSTM_SIZE, stateful=True, return_sequences=True))
    model.add(LSTM(FEATURE_SIZE, stateful=True))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adagrad', metrics=['accuracy'], loss='mse')
    return model


def open_model():
    new_model = create_model(train=True)
    filename = '{}/{}.h5'.format(SAVE_PATH, LSTM_MODEL)
    print('Load model from file: {}'.format(filename))
    try:
        model = load_model(filename)
        new_model.set_weights(model.get_weights())
        print('Load and copy weights from file success')
    except Exception as e:
        print('Load and copy weights from file error: {}'.format(str(e)))
    print('Create new model success: {}'.format(LSTM_MODEL))
    return new_model


def save_model(model):
    new_model = create_model(train=False)
    new_model.set_weights(model.get_weights())
    filename = '{}/{}.h5'.format(SAVE_PATH, LSTM_MODEL)
    try:
        print('Save model to file: {}'.format(filename))
        new_model.save(filename)
        print('Save model to file success')
    except Exception as e:
        print('Save model to file error: {}'.format(str(e)))


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
    if len(df) > 0:
        sc = MinMaxScaler(feature_range=(-1, 1))
        na = sc.fit_transform(df[['price_change', 'volume_change', 'volatility', 'convergence', 'predisposition']])
        return na, na[:, 0]
    else:
        return [], []


def save_data():
    print('Start download data:')
    exchange.load_markets(reload=True)
    train_x = []
    train_y = []
    with tqdm(total=len(exchange.symbols) * len(TIME_FRAMES)) as bar:
        for time_frame in TIME_FRAMES:
            for symbol in exchange.symbols:
                bar.update(1)
                input_data, output_data = symbol_data(symbol, time_frame)
                if len(input_data) < STEP_SIZE + 1:
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
    print('Save train data success')


def train():
    model = open_model()
    train_x = np.load('{}/train_x.npy'.format(SAVE_PATH))
    train_y = np.load('{}/train_y.npy'.format(SAVE_PATH))
    print('Train model with data length: %g' % len(train_x))
    model.fit(train_x, train_y, epochs=LSTM_SIZE * 10, batch_size=LSTM_SIZE * 100, verbose=2)
    print('Train model success')
    save_model(model)


def main():
    save_data()
    train()


if __name__ == "__main__":
    main()