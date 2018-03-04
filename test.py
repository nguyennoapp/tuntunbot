#!/usr/bin/env python3

import ccxt
import os
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

BASE = 'ETH'
QUOTE = 'BTC'
TIME_FRAME = '1h'
STEP_SIZE = 5
SAVE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = '{}/config.ini'.format(SAVE_PATH)

config = ConfigParser()
config.read(CONFIG_FILE)
exchange = ccxt.binance({'apiKey': config['CONFIG']['API_KEY'], 'secret': config['CONFIG']['API_SECRET']})


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


def main():
    exchange.load_markets()
    input_data, output_data = symbol_data('{}/{}'.format(BASE, QUOTE), TIME_FRAME)
    model = load_model('{}/lstm.h5'.format(SAVE_PATH))
    mean = np.mean(output_data, axis=0)
    output_data = output_data.tolist()
    predicts_change = [0, 0, 0, 0, 0]
    for i in range(len(input_data) - STEP_SIZE + 1):
        predict_change = model.predict(np.array([input_data[i:i + STEP_SIZE]]), batch_size=1) - mean
        predicts_change.append(predict_change)
    for i in range(len(output_data) - 1):
        output_data[i] = output_data[i] - mean
    output_data.append(0)
    index = np.arange(1, len(output_data) + 1, 1)
    plt.plot(index, output_data, label='Original price change')
    plt.plot(index, predicts_change, label='LSTM predict change')
    plt.legend()
    plt.axhline(0)
    plt.axhline(-.5)
    plt.axhline(-1)
    plt.axhline(.5)
    plt.axhline(1)
    plt.show()


if __name__ == "__main__":
    main()
