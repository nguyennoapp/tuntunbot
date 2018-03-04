#!/usr/bin/env python3

import ccxt
import os
import threading
from configparser import ConfigParser
from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, RegexHandler, ConversationHandler)

from tun import Tun

QUOTE, COMMAND, BASE, BUDGET, CONFIRM = range(5)
CONFIG_FILE = '{}/config.ini'.format(os.path.dirname(os.path.abspath(__file__)))

config = ConfigParser()
config.read(CONFIG_FILE)
updater = Updater(config['CONFIG']['BOT_TOKEN'], request_kwargs={'read_timeout': 30, 'connect_timeout': 60})
exchange = ccxt.binance({'apiKey': config['CONFIG']['API_KEY'], 'secret': config['CONFIG']['API_SECRET']})
tun = Tun(exchange)

buy_prices = {}
trade_type = ''
trade_quote = ''
trade_base = ''
trade_budget = ''


def start(bot, update):
    reply_keyboard = [['USDT', 'BTC', 'ETH', 'BNB']]
    update.message.reply_text('Hello! Set your quote first, sir.',
                              reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
    return QUOTE


def quote(bot, update, job_queue):
    global trade_quote
    trade_quote = update.message.text
    #job_queue.run_repeating(tun.balance(trade_quote, update), interval=60 * 15, first=0)
    #job_queue.run_repeating(tun.scan(trade_quote, update), interval=60 * 15, first=5)
    return restart(bot, update)


def restart(bot, update):
    reply_keyboard = [['Scan', 'Balance', 'Buy', 'Sell']]
    update.message.reply_text('Your current quote is {}. What do you want me to do?'.format(trade_quote),
                              reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))
    return COMMAND


def command(bot, update):
    global trade_type
    cmd = update.message.text.lower()
    if cmd == 'scan':
        update.message.reply_text('I\'m going scan the market to find signals, please wait a few minutes.')
        thread = threading.Thread(target=tun.scan, args=(trade_quote, update))
        thread.start()
    elif cmd == 'balance':
        update.message.reply_text('I see, you wanna check your account balance, wait a moment.')
        thread = threading.Thread(target=tun.balance, args=(trade_quote, update))
        thread.start()
    elif cmd == 'buy' or cmd == 'sell':
        trade_type = cmd
        update.message.reply_text('Which symbol do you want to {}, sir?'.format(trade_type))
        return BASE
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
    try:
        float(trade_budget)
    except ValueError:
        update.message.reply_text('{} is not a float.'.format(trade_budget))
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
        tun.recharge_fee('BNB', trade_quote)
        symbol = '{}/{}'.format(trade_base, trade_quote)
        if trade_type == 'buy':
            thread = threading.Thread(target=tun.buy, args=(symbol, float(trade_budget), update))
            thread.start()
        elif trade_type == 'sell':
            thread = threading.Thread(target=tun.sell, args=(symbol, update))
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
    print('Error: {}'.format(error))


def main():
    dp = updater.dispatcher
    handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            QUOTE: [RegexHandler('^(USDT|BTC|ETH|BNB)$', quote, pass_job_queue=True)],
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