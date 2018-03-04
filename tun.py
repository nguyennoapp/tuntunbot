#!/usr/bin/env python3

import os
import pickle
import time
import numpy as np
import talib.abstract as ta
import tensorflow as tf
from random import randint
from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

RETRY_LIMIT = 10
SAVE_FILE = '{}/buy_prices.dict'.format(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = '{}/lstm.h5'.format(os.path.dirname(os.path.abspath(__file__)))


class Tun(object):
    def __init__(self, exchange):
        self.exchange = exchange
        self.model = load_model(MODEL_FILE)
        self.graph = tf.get_default_graph()
        self.exchange.load_markets(reload=True)

    def save_buy_price(self, symbol, buy_price):
        try:
            with open(SAVE_FILE, 'rb') as f:
                buy_prices = pickle.load(f)
        except Exception:
            buy_prices = {}
        buy_prices[symbol] = buy_price
        with open(SAVE_FILE, 'wb') as f:
            pickle.dump(buy_prices, f)

    def get_buy_price(self, symbol):
        try:
            with open(SAVE_FILE, 'rb') as f:
                buy_prices = pickle.load(f)
        except Exception:
            buy_prices = {}
        if symbol in buy_prices:
            return buy_prices[symbol]
        return 0

    def recharge_fee(self, fee, quote):
        symbol = '{}/{}'.format(fee, quote)
        last_price = self.exchange.fetch_ticker(symbol)['last']
        min_cost = self.exchange.market(symbol)['limits']['cost']['min']
        amount = min_cost // last_price + 1
        if self.exchange.fetch_balance()['free'][fee] < amount:
            self.exchange.create_market_buy_order(symbol, amount)

    def price_calculate(self, symbol):
        s = self.exchange.market(symbol)
        order_book = self.exchange.fetch_order_book(symbol)
        buy_price = round(order_book['bids'][0][0] + s['limits']['price']['min'] * randint(2, 5), s['precision']['price'])
        sell_price = round(order_book['asks'][0][0] - s['limits']['price']['min'] * randint(2, 5), s['precision']['price'])
        return buy_price, sell_price

    def order_status(self, order_id, symbol):
        order = self.exchange.fetch_order(order_id, symbol)
        status = order['status']
        filled = order['filled']
        remaining = order['remaining']
        if status == 'open' and filled > 0:
            status = 'parted'
        return status, filled, remaining

    def buy(self, symbol, budget, update, retry=0):
        retry += 1
        if retry > RETRY_LIMIT:
            return
        s = self.exchange.market(symbol)
        buy_price, sell_price = self.price_calculate(symbol)
        amount = round(budget / buy_price // s['limits']['amount']['min'] * s['limits']['amount']['min'],
                       s['precision']['amount'])
        if amount == 0 or amount * buy_price < s['limits']['cost']['min']:
            return
        update.message.reply_text(
            '%s buy amount:%.8f price:%.8f total:%.8f' % (symbol, amount, buy_price, amount * buy_price))
        order = self.exchange.create_limit_buy_order(symbol, amount, buy_price)
        time.sleep(1)
        order_id = order['id']
        wait = 0
        while True:
            status, filled, remaining = self.order_status(order_id, symbol)
            if status == 'open':
                wait += 1
                if wait > RETRY_LIMIT:
                    self.exchange.cancel_order(order_id, symbol)
                    time.sleep(1)
                    self.buy(symbol, budget, update, retry)
                    break
                else:
                    time.sleep(1)
                    continue
            elif status == 'parted':
                update.message.reply_text('%s buy partially filled, amount:%.8f' % (symbol, filled))
                wait += 1
                if wait > RETRY_LIMIT:
                    self.exchange.cancel_order(order_id, symbol)
                    time.sleep(1)
                    self.save_buy_price(symbol, buy_price)
                    self.buy(symbol, remaining * buy_price, update, retry)
                    break
                else:
                    time.sleep(1)
                    continue
            elif status == 'closed':
                update.message.reply_text('%s buy filled, amount:%.8f' % (symbol, amount))
                self.save_buy_price(symbol, buy_price)
            else:
                update.message.reply_text('%s buy failed, status:%s' % (symbol, status))
                self.exchange.cancel_order(order_id, symbol)
            break

    def sell(self, symbol, update):
        s = self.exchange.market(symbol)
        buy_price, sell_price = self.price_calculate(symbol)
        balance = self.exchange.fetch_balance()
        amount = round(balance['free'][s['base']] // s['limits']['amount']['min'] * s['limits']['amount']['min'],
                       s['precision']['amount'])
        if amount == 0 or amount * sell_price < s['limits']['cost']['min']:
            return
        update.message.reply_text(
            '%s sell amount:%.8f price:%.8f total:%.8f' % (symbol, amount, sell_price, amount * sell_price))
        order = self.exchange.create_limit_sell_order(symbol, amount, sell_price)
        time.sleep(1)
        order_id = order['id']
        wait = 0
        while True:
            status, filled, remaining = self.order_status(order_id, symbol)
            if status == 'open':
                wait += 1
                if wait > RETRY_LIMIT:
                    self.exchange.cancel_order(order_id, symbol)
                    time.sleep(1)
                    self.sell(symbol, update)
                else:
                    time.sleep(1)
                    continue
            elif status == 'parted':
                update.message.reply_text('%s sell partially filled, amount:%.8f' % (symbol, filled))
                wait += 1
                if wait > RETRY_LIMIT:
                    self.exchange.cancel_order(order_id, symbol)
                    buy_price = self.get_buy_price(symbol)
                    if buy_price > 0:
                        update.message.reply_text('%s possible profit: %.8f %.2f%%' % (
                        symbol, (sell_price - buy_price) * filled, (sell_price / buy_price - 1) * 100))
                    time.sleep(1)
                    self.sell(symbol, update)
                else:
                    time.sleep(1)
                    continue
            elif status == 'closed':
                update.message.reply_text('%s sell filled, amount:%.8f' % (symbol, amount))
                buy_price = self.get_buy_price(symbol)
                if buy_price > 0:
                    update.message.reply_text('%s possible profit: %.8f %.2f%%' % (
                    symbol, (sell_price - buy_price) * amount, (sell_price / buy_price - 1) * 100))
            else:
                update.message.reply_text('%s sell failed, status:%s' % (symbol, status))
                self.exchange.cancel_order(order_id, symbol)
            break

    def clean_sell(self, symbol):
        try:
            last_price = self.exchange.fetch_ticker(symbol)['last']
        except Exception:
            return
        s = self.exchange.market(symbol)
        min_amount = s['limits']['amount']['min']
        precision = s['precision']['amount']
        min_cost = s['limits']['cost']['min']
        amount = self.exchange.fetch_balance()['free'][s['base']]
        amount = round(amount // min_amount * min_amount, precision)
        if amount == 0 or amount * last_price > min_cost:
            return
        self.exchange.create_market_sell_order(symbol, amount)

    def get_values(self, base, quote, amount):
        symbol = '{}/{}'.format(base, quote)
        ticker = self.exchange.fetch_ticker(symbol)
        quote_value = ticker['last'] * amount
        usdt_value = self.exchange.fetch_ticker('{}/USDT'.format(quote))['last'] * quote_value
        return ticker['last'], ticker['change'], quote_value, usdt_value

    def balance(self, quote, update):
        self.exchange.load_markets(reload=True)
        balance = self.exchange.fetch_balance()['total']
        quote_total = 0
        usdt_total = 0
        text = 'Your account balance:  \n'
        text += '%s amount: %g  \n' % (quote, balance[quote])
        for base in sorted(balance.keys()):
            amount = balance[base]
            try:
                price, change, quote_value, usdt_value = self.get_values(base, quote, amount)
            except Exception:
                continue
            symbol = '{}/{}'.format(base, quote)
            min_cost = self.exchange.market(symbol)['limits']['cost']['min']
            if quote_value < min_cost:
                self.clean_sell(symbol)
            else:
                buy_price = self.get_buy_price(symbol)
                if buy_price > 0:
                    profit = (price / buy_price - 1) * 100
                else:
                    profit = 0
                text += '%s amount: %.4f, price: %.8f,  value(%s): %.4f, value(USDT): %.2f, change(24h): %.2f%%, profit: %.2f%%  \n' % \
                        (base, amount, price, quote, quote_value, usdt_value, change, profit)
                quote_total += quote_value
                usdt_total += usdt_value
        quote_total += balance[quote]
        usdt_total += balance[quote] * self.exchange.fetch_ticker('{}/USDT'.format(quote))['last']
        text += 'Total in %s: %.8f, in USDT: %.2f' % (quote, quote_total, usdt_total)
        update.message.reply_text(text)

    def crossed(self, series1, series2, direction=None):
        if isinstance(series1, np.ndarray):
            series1 = Series(series1)
        if isinstance(series2, int) or isinstance(series2, float) or isinstance(series2, np.ndarray):
            series2 = Series(index=series1.index, data=series2)
        if direction is None or direction == "above":
            above = Series((series1 > series2) & (
                    series1.shift(1) <= series2.shift(1)))
        if direction is None or direction == "below":
            below = Series((series1 < series2) & (
                    series1.shift(1) >= series2.shift(1)))
        if direction is None:
            return above or below
        return above if direction is "above" else below

    def crossed_above(self, series1, series2):
        return self.crossed(series1, series2, "above")

    def crossed_below(self, series1, series2):
        return self.crossed(series1, series2, "below")

    def st_signal(self, symbol, stop_loss, take_profit):
        s = self.exchange.market(symbol)
        if self.exchange.fetch_balance()['free'][s['base']] > s['limits']['amount']['min']:
            buy_price = self.get_buy_price(symbol)
            if buy_price > 0:
                sell_price = self.exchange.fetch_ticker(symbol)['last']
                if (sell_price / buy_price - 1) * 100 <= stop_loss:
                    return 'STOP LOSS'
                if (sell_price / buy_price - 1) * 100 >= take_profit:
                    return 'TAKE PROFIT'
        return 'neutral'

    def ta_signal(self, symbol, time_frame):
        data = self.exchange.fetch_ohlcv(symbol, time_frame)
        df = DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df.set_index('time', inplace=True, drop=True)
        df['rsi'] = ta.RSI(df)
        df['adx'] = ta.ADX(df)
        df['plus_di'] = ta.PLUS_DI(df)
        df['minus_di'] = ta.MINUS_DI(df)
        df['fastd'] = ta.STOCHF(df)['fastd']
        df.loc[
            (
                    (df['rsi'] < 35) &
                    (df['fastd'] < 35) &
                    (df['adx'] > 30) &
                    (df['plus_di'] > 0.5)
            ) |
            (
                    (df['adx'] > 65) &
                    (df['plus_di'] > 0.5)
            ),
            'buy'] = 1
        df.loc[
            (
                    (
                            (self.crossed_above(df['rsi'], 70)) |
                            (self.crossed_above(df['fastd'], 70))
                    ) &
                    (df['adx'] > 10) &
                    (df['minus_di'] > 0)
            ) |
            (
                    (df['adx'] > 70) &
                    (df['minus_di'] > 0.5)
            ),
            'sell'] = 1
        buy_signal, sell_signal = df.iloc[-1]['buy'], df.iloc[-1]['sell']
        if buy_signal == 1:
            return 'BUY'
        elif sell_signal == 1:
            return 'SELL'
        return 'neutral'

    def dl_signal(self, symbol, time_frame):
        data = self.exchange.fetch_ohlcv(symbol, time_frame)
        df = DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df.set_index('time', inplace=True, drop=True)
        df.replace({0: np.nan}, inplace=True)
        df['price'] = df[['open', 'high', 'low', 'close']].mean(axis=1)
        df['price_change'] = df['price'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df = df.assign(**{'volatility': lambda x: (x['high'] - x['low']) / x['open']})
        df = df.assign(**{'convergence': lambda x: (x['open'] - x['close']) / (x['high'] - x['low'])})
        df = df.assign(**{'predisposition': lambda x: 1 - 2 * (x['high'] - x['close']) / (x['high'] - x['low'])})
        df.dropna(axis=0, how='any', inplace=True)
        sc = MinMaxScaler(feature_range=(-1, 1))
        input_data = sc.fit_transform(df[['price_change', 'volume_change', 'volatility', 'convergence', 'predisposition']])
        if len(input_data) >= 5:
            output_data = input_data[:, 0]
            mean = np.mean(output_data, axis=0)
            last_change = output_data[-1] - mean
            with self.graph.as_default():
                predict_change = self.model.predict(np.array([input_data[-5:]]), batch_size=1)[0][0] - mean
            if last_change < 0 < .1 < predict_change:
                return 'BUY'
            elif last_change > 0 > -.1 > predict_change:
                return 'SELL'
        return 'neutral'

    def scan(self, quote, update, time_frames=['15m', '1h', '4h', '1d'], stop_loss=-5, take_profit=5, auto_st=False):
        self.exchange.load_markets(reload=True)
        for symbol in self.exchange.symbols:
            if symbol.split('/')[1] == quote:
                time.sleep(0.1)
                change = self.exchange.fetch_ticker(symbol)['change']
                st = self.st_signal(symbol, stop_loss, take_profit)
                tas = []
                dls = []
                ta_text = ''
                dl_text = ''
                buy_score = 0
                for time_frame in time_frames:
                    time.sleep(0.1)
                    t = self.ta_signal(symbol, time_frame)
                    d = self.dl_signal(symbol, time_frame)
                    tas.append(t)
                    dls.append(d)
                    ta_text += '{}: {}, '.format(time_frame, t)
                    dl_text += '{}: {}, '.format(time_frame, d)
                    if t == 'BUY':
                        buy_score += 1
                    elif t == 'SELL':
                        buy_score -= 1
                    if d == 'BUY':
                        buy_score += 1
                    elif d == 'SELL':
                        buy_score -= 1
                buy_score = buy_score / len(time_frames) / 2 * 100
                if st != 'neutral' or buy_score > 0:
                    text = '%s change(24h): %.2f%%  \n' \
                           'ST signal: %s  \n' \
                           'TA signal: %s  \n' \
                           'DL signal: %s  \n' \
                           'BUY score: %d%%'% (symbol, change, st, ta_text, dl_text, buy_score)
                    update.message.reply_text(text)
                    if auto_st:
                        self.sell(symbol, update)
