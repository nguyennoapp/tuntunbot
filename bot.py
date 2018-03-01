#!/usr/bin/env python3
import ccxt
import json
import logging
import os
import pickle
import time
import threading
from configparser import ConfigParser
from random import randint
from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, RegexHandler, ConversationHandler)

QUOTE, COMMAND, BASE, BUDGET, CONFIRM = range(5)
SAVE_FILE = '{}/buy_prices.dict'.format(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = '{}/config.ini'.format(os.path.dirname(os.path.abspath(__file__)))

config = ConfigParser()
config.read(CONFIG_FILE)
exchange = ccxt.binance({'apiKey': config['CONFIG']['API_KEY'], 'secret': config['CONFIG']['API_SECRET']})
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
buy_prices = {}
trade_type = ''
trade_quote = ''
trade_base = ''
trade_budget = ''


class Symbol(object):
    def __init__(self, obj):
        self.__dict__ = json.loads(json.dumps(obj))


def save_buy_price(symbol, buy_price):
    buy_prices[symbol] = buy_price
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump(buy_prices, f)


def get_buy_price(symbol):
    global buy_prices
    if len(buy_prices) == 0:
        with open(SAVE_FILE, 'rb') as f:
            buy_prices = pickle.load(f)
    if symbol in buy_prices:
        return buy_prices[symbol]
    return 0
    

def price_calculate(symbol):
    order_book = exchange.fetch_order_book(symbol.symbol)
    buy_price = round(order_book['bids'][0][0] + symbol.limits['price']['min'] * randint(2, 5), symbol.precision['price'])
    sell_price = round(order_book['asks'][0][0] - symbol.limits['price']['min'] * randint(2, 5), symbol.precision['price'])
    return buy_price, sell_price


def order_status(order_id, symbol):
    order = exchange.fetch_order(order_id, symbol.symbol)
    status = order['status']
    filled = order['filled']
    remaining = order['remaining']
    if status == 'open' and filled > 0:
        status = 'parted'
    return status, filled, remaining


def buy(symbol, budget, update, panic=0):
    panic += 1
    if panic > int(config['CONFIG']['RETRY_LIMIT']):
        return
    buy_price, sell_price = price_calculate(symbol)
    amount = round(int(float(budget) / buy_price / symbol.limits['amount']['min'])
                   * symbol.limits['amount']['min'], symbol.precision['amount'])
    if amount < symbol.limits['amount']['min']:
        return
    if amount * buy_price < symbol.limits['cost']['min']:
        return
    update.message.reply_text('%s buy amount:%.8f price:%.8f total:%.8f' %
                              (symbol.symbol, amount, buy_price, amount * buy_price))
    order = exchange.create_limit_buy_order(symbol.symbol, amount, buy_price)
    time.sleep(1)
    order_id = order['id']
    panic_buy = 0
    while True:
        status, filled, remaining = order_status(order_id, symbol)
        if status == 'open':
            panic_buy += 1
            if panic_buy > int(config['CONFIG']['RETRY_LIMIT']):
                exchange.cancel_order(order_id, symbol.symbol)
                time.sleep(1)
                buy(symbol, budget, update, panic)
                break
            else:
                time.sleep(1)
                continue
        elif status == 'parted':
            update.message.reply_text('%s buy partially filled, amount:%.8f' % (symbol.symbol, filled))
            panic_buy += 1
            if panic_buy > int(config['CONFIG']['RETRY_LIMIT']):
                exchange.cancel_order(order_id, symbol.symbol)
                time.sleep(1)
                buy(symbol, remaining * buy_price, update, panic)
                break
            else:
                time.sleep(1)
                continue
        elif status == 'closed':
            update.message.reply_text('%s buy filled, amount:%.8f' % (symbol.symbol, amount))
            save_buy_price(symbol.symbol, buy_price)
        else:
            update.message.reply_text('%s buy failed, status:%s' % (symbol.symbol, status))
            exchange.cancel_order(order_id, symbol.symbol)
        break


def sell(symbol, update):
    buy_price, sell_price = price_calculate(symbol)
    balance = exchange.fetch_balance()
    amount = round(int(balance['free'][symbol.base] / symbol.limits['amount']['min'])
                   * symbol.limits['amount']['min'], symbol.precision['amount'])
    if amount < symbol.limits['amount']['min']:
        return
    if amount * sell_price < symbol.limits['cost']['min']:
        return
    update.message.reply_text('%s sell amount:%.8f price:%.8f total:%.8f' %
                (symbol.symbol, amount, sell_price, amount * sell_price))
    order = exchange.create_limit_sell_order(symbol.symbol, amount, sell_price)
    time.sleep(1)
    order_id = order['id']
    panic_sell = 0
    while True:
        status, filled, remaining = order_status(order_id, symbol)
        if status == 'open':
            panic_sell += 1
            if panic_sell > int(config['CONFIG']['RETRY_LIMIT']):
                exchange.cancel_order(order_id, symbol.symbol)
                time.sleep(1)
                sell(symbol, update)
            else:
                time.sleep(1)
                continue
        elif status == 'parted':
            update.message.reply_text('%s sell partially filled, amount:%.8f' % (symbol.symbol, filled))
            panic_sell += 1
            if panic_sell > int(config['CONFIG']['RETRY_LIMIT']):
                exchange.cancel_order(order_id, symbol.symbol)
                buy_price = get_buy_price(symbol.symbol)
                if buy_price > 0:
                    update.message.reply_text('%s possible profit: %.8f %.2f%%' %
                                              (symbol.symbol, (sell_price - buy_price) * filled, (sell_price / buy_price - 1) * 100))
                time.sleep(1)
                sell(symbol, update)
            else:
                time.sleep(1)
                continue
        elif status == 'closed':
            update.message.reply_text('%s sell filled, amount:%.8f' % (symbol.symbol, amount))
            buy_price = get_buy_price(symbol.symbol)
            if buy_price > 0:
                update.message.reply_text('%s possible profit: %.8f %.2f%%' %
                                          (symbol.symbol, (sell_price - buy_price) * amount, (sell_price / buy_price - 1) * 100))
        else:
            update.message.reply_text('%s sell failed, status:%s' % (symbol.symbol, status))
            exchange.cancel_order(order_id, symbol.symbol)
        break


def clean_sell(base, quote, update):
    try:
        symbol = '{}/{}'.format(base, quote)
        amount = exchange.fetch_balance()['free'][base]
        if amount == 0:
            return
        last_price = exchange.fetch_ticker(symbol)['last']
        min_amount = exchange.market(symbol)['limits']['amount']['min']
        precision = exchange.market(symbol)['precision']['amount']
        min_cost = exchange.market(symbol)['limits']['cost']['min']
        amount = round(int(amount / min_amount) * min_amount, precision)
        if amount < min_amount:
            return
        if amount * last_price > min_cost:
            return
        update.message.reply_text('%s clean sell amount: %.8f' % (symbol, amount))
        exchange.create_market_sell_order(symbol, amount)
    except Exception:
        return


def to_quote(base, quote, amount):
    try:
        symbol = '{}/{}'.format(base, quote)
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last'], ticker['last'] * amount, ticker['change']
    except Exception:
        return 0, 0, 0


def start(bot, update):
    reply_keyboard = [['USDT', 'BTC', 'ETH', 'BNB']]
    update.message.reply_text('Hello! Set your quote first, sir.',
                              reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
    return QUOTE


def quote(bot, update):
    global trade_quote
    trade_quote = update.message.text
    return restart(bot, update)


def restart(bot, update):
    reply_keyboard = [['Scan', 'Balance', 'Buy', 'Sell']]
    update.message.reply_text('Your current quote is {}. What do you want me to do?'.format(trade_quote),
                              reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
    return COMMAND


def command(bot, update):
    global trade_type
    cmd = update.message.text
    if cmd == 'Scan':
        update.message.reply_text('I\'m going scan market to find profitable trade.')
        return scan(bot, update)
    elif cmd == 'Balance':
        update.message.reply_text('I see, you want to check your account balance.')
        return balance(bot, update)
    elif cmd == 'Buy' or cmd == 'Sell':
        trade_type = cmd.lower()
        update.message.reply_text('Which symbol do you want to {}, sir?'.format(trade_type))
        return BASE
    else:
        return restart(bot, update)


def scan(bot, update):
    exchange.load_markets(reload=True)
    # to be implemented
    return restart(bot, update)


def balance(bot, update):
    balance = exchange.fetch_balance()['total']
    quote_total = 0
    text = 'Total balance in %s:  \n' % trade_quote
    text += '%s amount: %g  \n' % (trade_quote, balance[trade_quote])
    for key in sorted(balance.keys()):
        if key in ['USDT', 'BTC', 'ETH', 'BNB']:
            continue
        amount = balance[key]
        if amount > 0:
            price, value, change = to_quote(key, trade_quote, balance[key])
            buy_price = get_buy_price('{}/{}'.format(key, trade_quote))
            if buy_price > 0:
                profit = (price / buy_price - 1) * 100
            else:
                profit = 0
            if value >= exchange.market('{}/{}'.format(key, trade_quote))['limits']['cost']['min']:
                text += '%s amount: %g price: %g value: %g change: %.2f%% profit: %.2f%%  \n' % \
                        (key, amount, price, value, change, profit)
                quote_total += value
            else:
                clean_sell(key, trade_quote, update)
            time.sleep(0.1)
    quote_total += balance[trade_quote]
    text += 'Total in %s: %g' % (trade_quote, quote_total)
    update.message.reply_text(text)
    return restart(bot, update)


def base(bot, update):
    global trade_base
    trade_base = update.message.text.upper()
    symbol = '{}/{}'.format(trade_base, trade_quote)
    if symbol not in exchange.symbols:
        update.message.reply_text('{} is not exists.'.format(symbol))
        return restart(bot, update)
    if trade_type == 'buy':
        update.message.reply_text('Enter your quote budget, sir.')
        return BUDGET
    elif trade_type == 'sell':
        reply_keyboard = [['GO', 'Cancel']]
        update.message.reply_text('So, you want to {} {}!?'.
                                  format(trade_type, trade_base),
                                  reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
        return CONFIRM


def budget(bot, update):
    global trade_budget
    trade_budget = update.message.text
    if not trade_budget.isdigit():
        update.message.reply_text('{} is not a number.'.format(trade_budget))
        return restart(bot, update)
    reply_keyboard = [['GO', 'Cancel']]
    update.message.reply_text('So, you want to {} {} with budget {} {}!?'.
                              format(trade_type, trade_base, trade_budget, trade_quote),
                              reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
    return CONFIRM


def confirm(bot, update):
    global trade_type
    global trade_base
    global trade_budget
    cmd = update.message.text
    if cmd == 'GO':
        symbol = Symbol(exchange.market('{}/{}'.format(trade_base, trade_quote)))
        if trade_type == 'buy':
            thread = threading.Thread(target=buy, args=(symbol, float(trade_budget), update))
            thread.start()
        elif trade_type == 'sell':
            thread = threading.Thread(target=sell, args=(symbol, update))
            thread.start()
    elif cmd == 'Cancel':
        update.message.reply_text('Nope, let\'s try it again?')
        trade_type = ''
        trade_base = ''
        trade_budget = ''
    return restart(bot, update)


def cancel(bot, update):
    global trade_quote
    global trade_type
    global trade_base
    global trade_budget
    trade_quote = ''
    trade_type = ''
    trade_base = ''
    trade_budget = ''
    update.message.reply_text('Hmm, will we do it again?', reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


def error(bot, update, error):
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    exchange.load_markets()
    updater = Updater(config['CONFIG']['BOT_TOKEN'])
    dp = updater.dispatcher
    handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            QUOTE: [RegexHandler('^(USDT|BTC|ETH|BNB)$', quote)],
            COMMAND: [RegexHandler('^(Scan|Balance|Buy|Sell)$', command)],
            BASE: [MessageHandler(Filters.text, base)],
            BUDGET: [MessageHandler(Filters.text, budget)],
            CONFIRM: [RegexHandler('^(GO|Cancel)$', confirm)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    dp.add_handler(handler)
    dp.add_error_handler(error)
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()