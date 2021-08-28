# -*- coding: utf-8 -*-
import datetime
from .TradingBot import TradingBot
from binance.client import Client
from django.core.management.base import BaseCommand
import time
# def gold(start_date, end_date, company, sector_id, sector_name):

class Command(BaseCommand):

    def handle(self, *args, **options):
        # Api information from throwaway account

        args = {
            # Stochastic RSI Attributes
            'rsi_period': 14,
            'stochastic_period': 9,
            'k_slow_period': 3,
            'd_slow_period': 3,

            # Bollinger Band Attributes
            'simple_moving_average_period': 21,
            'bollinger_band_standard_deviation': 2,

            # Buy and Sell Attributes
            'stochastic_upper_band': 80,
            'stochastic_lower_band': 20,
            # 'time_look_back' : '1 month ago UTC',
            'time_look_back': '1 day ago UTC',
            # 'asset_interval' : Client.KLINE_INTERVAL_5MINUTE,
            'asset_interval': Client.KLINE_INTERVAL_1MINUTE,
            'show_times': 500
        }

        # while True:
        # time.sleep(40)
        bot = TradingBot(args)
        bot.backtest()