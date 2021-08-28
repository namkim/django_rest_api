from binance.client import Client, BinanceAPIException
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as plt_dates
import datetime as dt
from datetime import datetime
import time
import json
from cutter.models import Price, BuySell
from ... import util
import time
import environ
env = environ.Env()


class TradingBot:

    def __init__(self, args):
        #Bot information
        self.client = Client(env("API_KEY"), env("API_SECRET"))

        #Stoch RSI Information
        self.avg_gain, self.avg_loss = 0, 0
        self.rsi,  self.rsi_array = 0, []
        self.rsi_period, self.stoch_period = args['rsi_period'], args['stochastic_period']
        self.k_slow_period, self.d_slow_period = args['k_slow_period'], args['d_slow_period']
        self.k_fast_array, self.k_slow_array, self.d_slow_array = {'time':[],'k_fast':[]}, {'time':[],'k_slow':[]}, {'time':[],'d_slow':[]}
        self.stoch_lower, self.stoch_upper = args['stochastic_upper_band'], args['stochastic_lower_band']

        #Bollinger Band Information
        self.sma_period, self.sma_array = args['simple_moving_average_period'], []
        self.deviation = args['bollinger_band_standard_deviation']
        self.bb = {'time':[],'sma': [],'lower_band': [],'upper_band': []}

        #Buy Sell Algorithm Information
        self.orders = {'time':[],'order_limit': [],'order_type': []}
        self.time_look_back = args['time_look_back']
        self.asset_interval = args['asset_interval']
        self.round_cnt = int(env("ROUND_CNT"))
        self.buy_quantity = float(env("BUY_QUANTITY"))
        self.sell_per = float(env("SELL_PER"))
        self.buy_per = float(env("BUY_PER"))
        self.sleeptime = float(env("SLEEPTIME"))

        #Binance trading fees and general information
        self.pair = env("PAIR")
        self.show_times = args['show_times']
        self.closing_price_array, self.closing_times = [], []
        self.checked_prices, self.checked_times = [], []
        self.general_trade_fee = 0.001 # 0.1% trading fee Maker/Taker --> Seller/Buyer

    @staticmethod
    def simple_moving_average(arr, period):
        """Simple_moving_average = (A1+A2+...Ai)/i
        :param arr: An array of numeric values
        :param period: Number of numeric values in arr
        :return: simple moving average
        """
        return np.sum(arr) / period

    @staticmethod
    def wilders_moving_average(period, close, prev_moving_average):
        """
        :param period: Number of numeric values in arr
        :param close: The current closing price
        :param prev_moving_average: The previous moving average
        :return: wilders moving average
        """
        return ((prev_moving_average * (period - 1)) + close) / period

    @staticmethod
    def rsi_calc(avg_gain, avg_loss):
        """Calculates single rsi value
        :param avg_gain: The moving average of the sum of positive price changes over a period n
        :param avg_loss: The moving average of the sum of abs(negative) price changes over a period n
        :return: rsi value
        """

        rs = avg_gain / (avg_loss + 0.00000001)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def k_fast_stoch(rsi_array):
        """Calculates the k_fast value for stochastic indicator
        :param rsi_array: an array of stoch_period number of rsi values
        :return: k_fast value
        """
        close = rsi_array[len(rsi_array) - 1]
        high = np.amax(rsi_array)
        low = np.amin(rsi_array)
        return abs(((close - low) / (high - low))) * 100

    def bollinger_bands(self, kline_array, sma_period, deviation_number, time):
        """Calculates the values for the bollinger band indicator
        :param kline_array:
        :param sma_period: The period for the simple_moving_average
        :param deviation_number: The number of standard deviations away the upper and lower bands are from the sma
        :param time: The time when this value is calculated
        :return: simple_moving_average, bb upper band, bb lower band, deviation value
        """
        recent_numbers = kline_array[len(kline_array) - sma_period:]
        sma = self.simple_moving_average(sum(recent_numbers), sma_period)
        self.bb['sma'].append(sma)
        squared_errors = []
        for x in range(0, len(recent_numbers)):
            squared_errors.append(pow(recent_numbers[x] - sma, 2))
        standard_deviation = (sum(squared_errors) / len(squared_errors)) ** (1 / 2)
        upper_band = sma + standard_deviation * deviation_number
        lower_band = sma - standard_deviation * deviation_number
        self.bb['lower_band'].append(lower_band)
        self.bb['upper_band'].append(upper_band)
        self.bb['time'].append(time)
        return sma, upper_band, lower_band, standard_deviation

    def print_values(self):
        """Displays the current values for each indicator in the terminal
        :return: Nothing
        """
        print('Previous Stoch_RSI(%K_FAST)', self.k_fast_array['k_fast'][-1])
        print('Previous Stoch_RSI(%K_SLOW)', self.k_slow_array['k_slow'][-1])
        print('Previous Stoch_RSI(%D_SLOW)', self.d_slow_array['d_slow'][-1])
        print('RSI(Smoothed) ', self.rsi)
        print('Bollinger Lower Band', self.bb['lower_band'][-1])
        print('Bollinger Upper Band', self.bb['upper_band'][-1])
        print('Simple Moving Average', self.bb['sma'][-1], '\n')

    def backtest(self):
        # Cut off most recent history closing price since it is not complete and would effect the calculations
        #kline_array = self.client.get_historical_klines(symbol=pair, interval=Client.KLINE_INTERVAL_5MINUTE, start_str= '1' + ' month ago UTC')

        try:
            kline_array_temp = self.client.get_historical_klines(symbol=self.pair, interval=self.asset_interval, start_str=self.time_look_back)
        except Exception as e:
            print(e)
            util.slack_send(str(f"================ {env('LOG_TITLE_BUY')}]======================" + str(e)))
            time.sleep(300)
            return None
        # kline_array = self.client.get_klines(symbol=self.pair, interval=self.asset_interval, limit=240)
        # self.closing_times = [dt.datetime.utcfromtimestamp(x[6]/1000) for x in kline_array][0:-1]
        # self.closing_price_array = [float(x[4]) for x in kline_array][0:-1]
        jsonDec = json.decoder.JSONDecoder()


        # 바이낸스에서 최근 30건 가격만 디비 저장
        # timestamp_id 가 가장 높은수치가 가장 최근 비트코인 가격
        # 일단 디비에 다 저장
        for k in kline_array_temp[-30:]:
            price, is_new = Price.objects.get_or_create(timestamp=k[0])
            if is_new:
                price.kline_array = json.dumps(k)
                price.btc_datetime_at = str(datetime.fromtimestamp(int(price.timestamp) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
                price.save()

        kline_array = []
        # DB에서 가져온 30건 list형변환
        for p in Price.objects.all().order_by('-id')[:30]:
            # print(p.kline_array, p.timestamp, datetime.fromtimestamp(int(p.timestamp) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
            kline_array.append(jsonDec.decode(p.kline_array))

        # 최신값 가져오는지 확인
        kline_array.sort(reverse=False)
        self.closing_times = [dt.datetime.utcfromtimestamp(x[6]/1000) for x in kline_array][0:-1]
        self.closing_price_array = [float(x[4]) for x in kline_array][0:-1]
        self.checked_prices = []
        gain, loss = 0, 0
        #가장 확실한건 buy 이뤄지고 그게 최신 값이면 끝
        # 여기서 확인할것은 처음 루프돌때 처음의 값이 오래된 과거의 값이면 될듯
        util.slack_send_log(str(f"================ {env('LOG_TITLE_BUY')}]======================"))

        for x in range(0, len(self.closing_price_array)-1):
            # print(self.closing_price_array[x])
            change = self.closing_price_array[x+1] - self.closing_price_array[x]
            self.checked_prices.append(self.closing_price_array[x+1])
            self.checked_times.append(self.closing_times[x+1])

            if change > 0:
                gain += change
            elif change < 0:
                loss += abs(change)
            #Get first rsi simple moving average
            # print(x, self.rsi_period)
            if x == self.rsi_period:
                self.avg_gain = self.simple_moving_average(gain, self.rsi_period)
                self.avg_loss = self.simple_moving_average(loss, self.rsi_period)
                self.rsi = self.rsi_calc(self.avg_gain, self.avg_loss)
                self.rsi_array.append(self.rsi)
                gain, loss = 0, 0

            #Use wilders moving average to continue calculating rsi values
            elif x > self.rsi_period:
                self.avg_gain = self.wilders_moving_average(self.rsi_period, gain, self.avg_gain)
                self.avg_loss = self.wilders_moving_average(self.rsi_period, loss, self.avg_loss)
                self.rsi = self.rsi_calc(self.avg_gain, self.avg_loss)
                self.rsi_array.append(self.rsi)
                gain, loss = 0, 0
                # When there are enough rsi values begin to calculate stoch_rsi
                if len(self.rsi_array) >= self.stoch_period:
                    k_fast = self.k_fast_stoch(self.rsi_array[len(self.rsi_array) - self.stoch_period:])
                    self.k_fast_array['k_fast'].append(k_fast)
                    self.k_fast_array['time'].append(self.closing_times[x])
                    # When there are enough %K_FAST values begin to calculate %K_SLOW values = sma of n %K_FAST values
                    if len(self.k_fast_array['k_fast']) >= self.k_slow_period:
                        k_slow = self.simple_moving_average(self.k_fast_array['k_fast'][-1*self.k_slow_period:], self.k_slow_period)
                        self.k_slow_array['k_slow'].append(k_slow)
                        self.k_slow_array['time'].append(self.closing_times[x])
                        # When there are enough %K_SLOW values begin to calculate %D_SLOW values = sma of n %K_SLOW values
                        if len(self.k_slow_array['k_slow']) >= self.d_slow_period:
                            d_slow = self.simple_moving_average(self.k_slow_array['k_slow'][-1*self.d_slow_period:], self.d_slow_period)
                            self.d_slow_array['d_slow'].append(d_slow)
                            self.d_slow_array['time'].append(self.closing_times[x])
                            self.bollinger_bands(self.checked_prices, self.sma_period, self.deviation, self.checked_times[x])
                            #Once all values start to be calculated we can determine whether to buy or sell until we hit the last
                            self.buy_sell(current_time = self.checked_times[x])

        # self.plot_orders() #Plot orders on graph

    def buy_sell(self, current_time):
        print(f"================ {env('LOG_TITLE_BUY')}]======================")
        # Setting buy limit conditional
        next_price = self.checked_prices[-1]
        if self.k_slow_array['k_slow'][-1] <= self.stoch_lower and self.d_slow_array['d_slow'][-1] <= self.stoch_lower and next_price <= self.bb['lower_band'][-1]:
            print("buy.....")
            lower_band_price = round(self.bb['lower_band'][-1], self.round_cnt)
            buy_price = round(lower_band_price - (lower_band_price * self.buy_per), self.round_cnt)

            str_param = \
                f'buy_quantity:{self.buy_quantity} ' \
                f'next_price:{next_price} ' \
                f'buy_price:{buy_price} ' \
                f'bb.lower_band:{lower_band_price}  '
            print(str_param)
            util.slack_send(str(f"================ {env('LOG_TITLE_BUY')}]======================" + str(str_param)))
            try:
                print(
                'BUY Current Price: ', next_price, '\nCreated Buy Order: ', buy_price, '\nQuantity: ',
                self.buy_quantity, '\nTime: ', current_time)
                order = self.client.create_order(symbol=self.pair, type="LIMIT_MAKER", side="BUY",quantity=self.buy_quantity, price=buy_price)

                util.slack_send(str(f"================ {env('LOG_TITLE_BUY')}]======================"+ str(order)))
                bs, is_new = BuySell.objects.get_or_create(timestamp=str(current_time))
                bs.price = round(lower_band_price)
                bs.coin = self.pair
                bs.status = "buy"
                bs.order = str(order)
                bs.order_id = str(order['orderId'])
                bs.save()
                time.sleep(self.sleeptime)
                self.orders['time'].append(current_time)
                self.orders['order_limit'].append(self.bb['lower_band'][-1])
                self.orders['order_type'].append('buy')
                self.print_values()

            except Exception as e:
                print(e)
                util.slack_send(str(f"================ {env('LOG_TITLE_BUY')}]======================" + str(e)))
                for b in BuySell.objects.filter(coin=self.pair, status="buy").order_by('-id')[:5]:
                    if b.cancel_flag == "Waiting":
                        try:
                            util.slack_send(str(f"================ cancel buy======================"))
                            b.cancel_flag = "done"
                            b.save()
                            self.client.cancel_order(symbol=self.pair, orderId=b.order_id)
                        except Exception as oe:
                            util.slack_send(str(f"================ cancel error======================" + str(oe)))
                time.sleep(self.sleeptime)

        # Setting sell limit conditional
        elif self.k_slow_array['k_slow'][-1] >= self.stoch_upper and self.d_slow_array['d_slow'][-1] >= self.stoch_upper and next_price >= self.bb['upper_band'][-1]:
            print("sell.....")
            # order = self.client.create_order(symbol=self.pair, side=Client.SIDE_SELL, type=Client.ORDER_TYPE_MARKET, quantity=self.buy_quantity)
            upper_band_price = round(self.bb['upper_band'][-1], self.round_cnt)
            sell_price = round(upper_band_price + (upper_band_price * self.sell_per), self.round_cnt)

            str_param = \
                f'buy_quantity:{self.buy_quantity} ¥n' \
                f'next_price:{next_price}¥n' \
                f'sell_price:{sell_price}¥n' \
                f'bb.lower_band:{upper_band_price}¥n'
            print(str_param)
            util.slack_send(str(f"================ {env('LOG_TITLE_SELL')}]======================" + str(str_param)))
            try:
                print(
                'SELL Current Price: ', next_price, '\nCreated Sell Order: ',
                upper_band_price, '\nQuantity: ',
                sell_price, '\nsell_price: ',
                self.buy_quantity, '\nTime: ', current_time)
                order = self.client.create_order(symbol=self.pair, type="LIMIT_MAKER", side="SELL", quantity=self.buy_quantity,price=sell_price)
                util.slack_send(str(f"================ {env('LOG_TITLE_SELL')}======================" + str(order)))
                dicts = self.client.get_account()["balances"]
                btc = str(next(item for item in dicts if item["asset"] == "BTC"))
                ada = str(next(item for item in dicts if item["asset"] == "ADA"))
                doge = str(next(item for item in dicts if item["asset"] == "DOGE"))
                busd = str(next(item for item in dicts if item["asset"] == "BUSD"))
                util.slack_send_price(str(f"=={env('LOG_TITLE_SELL')}=== {btc, doge, ada, busd}================"))
                before_price = BuySell.objects.last()
                bs, is_new = BuySell.objects.get_or_create(timestamp=str(current_time))
                bs.price = round(upper_band_price)
                bs.coin = self.pair
                bs.stochastic_period = self.stoch_period
                bs.rsi_period = self.rsi_period
                bs.upper_band = self.stoch_upper
                bs.lower_band = self.stoch_lower
                bs.total = int(float(before_price.price)) - round(upper_band_price)
                bs.status = "sell"
                bs.order = str(order)
                bs.order_id = str(order['orderId'])
                bs.save()
                time.sleep(self.sleeptime)

                self.orders['time'].append(current_time)
                self.orders['order_limit'].append(self.bb['upper_band'][-1])
                self.orders['order_type'].append('sell')
                self.print_values()

            except Exception as e:
                print(e)
                util.slack_send(str(f"================ {env('LOG_TITLE_SELL')}]======================" + str(e)))
                for b in BuySell.objects.filter(coin=self.pair, status="sell").order_by('-id')[:5]:
                    if b.cancel_flag == "Waiting":
                        try:
                            b.cancel_flag = "done"
                            b.save()
                            util.slack_send(str(f"================ cancel sell======================"))
                            self.client.cancel_order(symbol=self.pair, orderId=b.order_id)
                        except Exception as oe:
                            util.slack_send(str(f"================ cancel error======================" + str(oe)))

                time.sleep(self.sleeptime)

    def plot_orders(self):
        """
        Plot all closing prices and indicators
        :return: Nothing
        """
        fig, axis = plt.subplots(2)

        #Only showing last s values for all values
        s = -1*self.show_times

        #Plot all checked prices and times, this leaves out the current incomplete closing price
        axis[0].plot_date(plt_dates.date2num(self.checked_times[s:]), self.checked_prices[s:], xdate=True, fmt='c-')

        #Plot bollinger bands
        axis[0].plot_date(plt_dates.date2num(self.bb['time'][s:]), self.bb['sma'][s:], xdate=True, fmt='k-') #-1 since sma not calculate for recent incomplete closing price
        axis[0].plot_date(plt_dates.date2num(self.bb['time'][s:]), self.bb['upper_band'][s:], xdate=True, fmt='k--')
        axis[0].plot_date(plt_dates.date2num(self.bb['time'][s:]), self.bb['lower_band'][s:], xdate=True, fmt='k--')

        #Plot stoch_rsi (%K_Slow, %D_Slow) in different subplot from time when buy_sell algorithm starts
        axis[1].plot_date(plt_dates.date2num(self.k_slow_array['time'][s:]), self.k_slow_array['k_slow'][s:], xdate=True, fmt='g-')
        axis[1].plot_date(plt_dates.date2num(self.d_slow_array['time'][s:]), self.d_slow_array['d_slow'][s:], xdate=True, fmt='b-')

        #Plot stoch_rsi user set upper and lower limits
        upper_limit = np.ones(shape=np.asarray(self.k_slow_array['time'][s:]).shape) * self.stoch_upper
        lower_limit = np.ones(shape=np.asarray(self.k_slow_array['time'][s:]).shape) * self.stoch_lower
        axis[1].plot_date(plt_dates.date2num(self.k_slow_array['time'][s:]), upper_limit, xdate=True, fmt='r--')
        axis[1].plot_date(plt_dates.date2num(self.k_slow_array['time'][s:]), lower_limit, xdate=True, fmt='r--')

        #Plot buy and sell orders
        orders = np.asarray(self.orders)
        buy_times = np.where( (np.asarray(self.orders['time']) > self.d_slow_array['time'][s]) & (np.asarray(self.orders['order_type']) == 'buy') )
        sell_times = np.where( (np.asarray(self.orders['time']) > self.d_slow_array['time'][s]) & (np.asarray(self.orders['order_type']) == 'sell') )
        if len(buy_times[0] > 0):
            axis[0].plot_date(plt_dates.date2num(np.take(self.orders['time'], buy_times[0])), np.take(self.orders['order_limit'], buy_times[0]), xdate=True, fmt='go')
        if len(sell_times[0] > 0):
            axis[0].plot_date(plt_dates.date2num(np.take(self.orders['time'], sell_times[0])), np.take(self.orders['order_limit'], sell_times[0]), xdate=True, fmt='ro')

        #Plot attributes
        axis[0].legend(labels=('Closing Prices', 'SMA', 'Bolling Upper Band', 'Bollinger Lower Band', 'Buy Orders', 'Sell Order'), loc='upper right', prop={'size':5})
        axis[1].legend(labels=('%k slow', '%d slow', 'user limits'), loc='upper right', prop={'size': 5})

        axis[0].set_xlabel('Date')
        axis[0].set_ylabel('USD')

        axis[1].set_xlabel('Date')
        axis[1].set_ylabel('%')

        fig.autofmt_xdate() #Auto aligns dates on x axis
        plt.show()
