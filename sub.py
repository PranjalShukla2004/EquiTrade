import datetime as dt
import time
import random
import logging
import numpy as np
from optibook.synchronous_client import Exchange
from libs import print_positions_and_pnl, round_down_to_tick, round_up_to_tick
from IPython.display import clear_output

class LinearSeqReg:
    def __init__(self,tim=50, seq_values=None, weight_values=None):
        self.sxw = 0.0 
        self.sx2w = 0.0 
        self.sw = 0
        self.idle_time = tim
        self.x = 0
        self.w = 0
        if seq_values is not None and weight_values is not None:
            self.idle_time = len(seq_values)-1
            self.x = np.array(seq_values)
            self.w = np.array(weight_values)
        else:
            self.x = np.array([i+1 for i in range(self.idle_time)])
            self.w = np.array([1 for i in range(self.idle_time)])
        self.sxw = sum(self.x*self.w)
        self.sx2w = sum(self.x*self.x*self.w)
        self.sw = sum(self.w)
    
    def estimate(self,prev_values, seq_no):
        j = seq_no - self.idle_time
        xyw_sum = sum(np.array(prev_values[-self.idle_time:])*self.x*self.w)
        wy_sum = sum(np.array(prev_values[-self.idle_time:])*self.w)
        c = (self.sxw*xyw_sum - self.sx2w*wy_sum) / (self.sxw*self.sxw - self.sw * self.sx2w)
        m = (self.sxw*wy_sum - self.sw*xyw_sum) /(self.sxw*self.sxw-self.sw*self.sx2w)
        return m*self.x[-1]+c

def insert_quotes(exchange, instrument, bid_price, ask_price, bid_volume, ask_volume):
    if bid_volume > 0:
        # Insert new bid limit order on the market
        exchange.insert_order(
            instrument_id=instrument.instrument_id,
            price=bid_price,
            volume=bid_volume,
            side='bid',
            order_type='limit',
        )
        
        # Wait for some time to avoid breaching the exchange frequency limit
        time.sleep(0.05)

    if ask_volume > 0:
        # Insert new ask limit order on the market
        exchange.insert_order(
            instrument_id=instrument.instrument_id,
            price=ask_price,
            volume=ask_volume,
            side='ask',
            order_type='limit',
        )

        # Wait for some time to avoid breaching the exchange frequency limit
        time.sleep(0.05)    

def compute_price(order_book, max_num_orders=5):
    max_num_orders = min(max_num_orders,len(order_book.bids),len(order_book.asks))
    bidvolsum = sum(x.volume for x in order_book.bids[:max_num_orders])
    askvolsum = sum(x.volume for x in order_book.asks[:max_num_orders])
    minvol = min(bidvolsum,askvolsum)
    if minvol==0.0:
        return 0.0
    total = 0.0
    x = 0
    y = 0
    for i in range(max_num_orders):
        if x <= minvol:
            total = total + min(minvol-x,order_book.bids[i].volume)*order_book.bids[i].price
        if y <= minvol:
            total = total + min(minvol-y,order_book.asks[i].volume)*order_book.bids[i].price
    return total/(2.0 * minvol)

def calculate_volatility(price_changes):
    """standard deviation of price changes."""
    return np.std(price_changes)

def adjust_safety_margin(order_book, price_changes, base_margin=0.01, volatility_factor=1.5, spread_factor=0.5):
    """Dynamically adjust the safety margin based on volatility and bid-ask spread."""
    # Calculate historical volatility
    volatility = calculate_volatility(price_changes)
    
    # current bid-ask spread as a percentage of mid-price
    bid_price = order_book.bids[0].price
    ask_price = order_book.asks[0].price
    mid_price = (bid_price + ask_price) / 2.0
    spread_percentage = (ask_price - bid_price) / mid_price
    
    # Adjust safety margin based on volatility and spread
    adjusted_margin = base_margin + (volatility * volatility_factor) + (spread_percentage * spread_factor)
    
    return adjusted_margin

def compute_and_decide_price_with_dynamic_margin(order_book, price_changes, max_num_orders=5):
    # Dynamically adjust the safety margin
    safety_margin = adjust_safety_margin(order_book, price_changes)
    
    fair_price = compute_price(order_book, max_num_orders) 
    
    if fair_price is not None:
        adjusted_buy_price = fair_price * (1 - safety_margin)
        adjusted_sell_price = fair_price * (1 + safety_margin)
        
        if order_book.bids[0].price > adjusted_buy_price:
            return (adjusted_buy_price)
        elif order_book.asks[0].price < adjusted_sell_price:
            return (adjusted_sell_price)
        else:
            return (fair_price)
    else:
        return (None)

def get_price_changes(instrument_id, new_price, max_history=5):
    historical_prices = {}
    # Ensure the list for this instrument exists in the historical_prices dictionary
    if instrument_id not in historical_prices:
        historical_prices[instrument_id] = []
    
    # Add the new price to the history, ensuring we don't exceed max_history items
    prices = historical_prices[instrument_id]
    prices.append(new_price)
    if len(prices) > max_history:
        # Remove the oldest price if we exceed the history limit
        prices.pop(0)
    
    # Calculate percentage changes between consecutive prices
    price_changes = [(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
    
    return price_changes

def company_name(social_feeds):
    company_name_mapping = {
        "Cisco": "CSCO",
        "CSCO": "CSCO",
        "CISCO": "CSCO",
        "NVDA": "NVDA",
        "NVDIA": "NVDA",
        "Nvidia": "NVDA",
        "Pfizer": "PFE",
        "PFE": "PFE",
        "ING": "ING",
        "Santander" : "SAN",
        "Banco" : "SAN",
        "SAN": "SAN"
    }
    while True:
            if not social_feeds:
                print(f'{dt.datetime.now()}: no new messages')
                print(social_feeds)
            else:
                for feed in social_feeds:
                    post = determiner(feed, ['NVDA', 'ING','CSCO', 'PFE', 'SAN'])
                    return post.labels[0]


def company_status(social_feeds):

        if not social_feeds:
            print(f'{dt.datetime.now()}: no new messages')
            return -1
        else:
            for feed in social_feeds:
                print(f'{feed.timestamp}: {feed.post}')
                post = determiner(feed, ['Good', 'Bad'])
                max_score = post.scores[0]
                max_label = post.labels[0]
                if max_label == 'Good' and max_score > 0.75:
                    return 'Good', max_score
                elif max_label == 'Bad' and max_score > 0.75:
                    return 'Bad', max_score
                else:
                    return -1

    
exchange = Exchange()
exchange.connect()

INSTRUMENTS = exchange.get_instruments()

QUOTED_VOLUME = 10
FIXED_MINIMUM_CREDIT = 0.15
PRICE_RETREAT_PER_LOT = 0.005
POSITION_LIMIT = 50

seq_no = 0
idle_time = 50
past_prices = dict()
for i in INSTRUMENTS.values():
    past_prices[i.instrument_id] = [0.0]
model = LinearSeqReg()

while True:
    print(f'')
    print(f'-----------------------------------------------------------------')
    print(f'TRADE LOOP ITERATION ENTERED AT {str(dt.datetime.now()):18s} UTC.')
    print(f'-----------------------------------------------------------------')

    # Display our own current positions in all stocks, and our PnL so far
    print_positions_and_pnl(exchange)
    print(f'')
    print(f'          (ourbid) mktbid :: mktask (ourask)')
    curr_time = time.time()
    socialfeed = exchange.poll_new_social_media_feeds()
    company_str = company_name(socialfeed)
    alpha = company_status(socialfeed)    
    if alpha != -1:
        sentiment, score = alpha
        to_trade[company_str][0] = True
        to_trade[company_str][1] = curr_time
        curr_instrument = None
        for i in INSTRUMENTS.values():
            if i.instrument_id==company_str:
                curr_instrument = i
                break
        exchange.delete_orders(company_str)
        position = exchange.get_positions()[company_str]
        instrument_order_book = exchange.get_last_price_book(company_str)
        if sentiment == 'Bad':
            best_bid_price = instrument_order_book.bids[-1].price
            exchange.insert_order(
            instrument_id=instrument.instrument_id,
            price=best_bid_price,
            volume= (int) (position + np.ceil((1+score)*QUOTED_VOLUME) if (position > 0) else np.ceil((1+score)*QUOTED_VOLUME)),
            side='ask',
            order_type='ioc',
        )
        else:
            best_ask_price = instrument_order_book.asks[-1].price
            exchange.insert_order(
            instrument_id=instrument.instrument_id,
            price=best_ask_price,
            volume= (int)((-position + np.ceil((1+score)*QUOTED_VOLUME)) if (position < 0) else np.ceil((1+score)*QUOTED_VOLUME)),
            side='bid',
            order_type='ioc',
        )
    else:
            pass
    for instrument in INSTRUMENTS.values():
        # Remove all existing (still) outstanding limit orders
        exchange.delete_orders(instrument.instrument_id)
        #seq_no = seq_no + 1
        # Obtain order book and only skip this instrument if there are no bids or offers available at all on that instrument,
        # as we we want to use the mid price to determine our own quoted price
        instrument_order_book = exchange.get_last_price_book(instrument.instrument_id)
        if not (instrument_order_book and instrument_order_book.bids and instrument_order_book.asks):
            print(f'{instrument.instrument_id:>6s} --     INCOMPLETE ORDER BOOK')
            past_prices[instrument.instrument_id].append(past_prices[-1])
            #print(f"with seq no : {seq_no}, instrument : {instrument.instrument_id}, size of fair prices:{len(past_prices[instrument.instrument_id])}")
            continue
        our_mid_price = compute_price(instrument_order_book)
        past_prices[instrument.instrument_id].append(our_mid_price)
        if seq_no < idle_time+1:
            #print(f"with seq no : {seq_no}, instrument : {instrument.instrument_id}, size of fair prices:{len(past_prices[instrument.instrument_id])}")
            continue
        if to_trade[instrument.instrument_id][0]:
            if curr_time - to_trade[instrument.instrument_id][1] > 15:
                to_trade[instrument.instrument_id][0] = True
            else:
                continue
        # Obtain own current position in instrument
        position = exchange.get_positions()[instrument.instrument_id]
        # Obtain best bid and ask prices from order book to determine mid price
        best_bid_price = instrument_order_book.bids[0].price
        best_ask_price = instrument_order_book.asks[0].price
        mid_price = (best_bid_price + best_ask_price) / 2.0 
        #our_mid_price = compute_price(instrument_order_book)
        print(f"with seq no : {seq_no}, instrument : {instrument.instrument_id}, past 5 fair prices:{past_prices[instrument.instrument_id][-5:]}")
        print(f"best bid and best ask prices for the same are : {best_bid_price}, {best_ask_price}")
        next_est_price = model.estimate(past_prices[instrument.instrument_id],seq_no)
        print(f"next estimate is : {next_est_price}")
        bid_ask_spread = best_ask_price - best_bid_price
        # Calculate our fair/theoretical price based on the market mid price and our current position
        theoretical_price = our_mid_price - PRICE_RETREAT_PER_LOT * position
        bid_price = -1
        ask_price = -1
        if abs(position) > 20:
            if position > 0:
                ask_price = round_up_to_tick(next_est_price + instrument.tick_size, instrument.tick_size)
                ask_volume = QUOTED_VOLUME
                bid_volume = 0
            else:
                bid_price = round_down_to_tick(next_est_price-instrument.tick_size , instrument.tick_size)
                bid_volume = QUOTED_VOLUME
                ask_volume = 0
            insert_quotes(exchange, instrument, bid_price, ask_price, bid_volume, ask_volume)
            print("closing out")
            print(f'{instrument.instrument_id:>6s} -- ({bid_price:>6.2f}) {best_bid_price:>6.2f} :: {best_ask_price:>6.2f} ({ask_price:>6.2f})')   
        elif next_est_price >= best_bid_price and next_est_price <= best_ask_price :
            # Calculate final bid and ask prices to insert
            bid_price = round_down_to_tick((next_est_price+best_bid_price)/2 , instrument.tick_size)
            ask_price = round_up_to_tick((next_est_price + best_ask_price)/2, instrument.tick_size)
            # Calculate bid and ask volumes to insert, taking into account the exchange position_limit
            max_volume_to_buy = POSITION_LIMIT - position
            max_volume_to_sell = POSITION_LIMIT + position
            bid_volume = min(QUOTED_VOLUME, max_volume_to_buy)
            ask_volume = min(QUOTED_VOLUME, max_volume_to_sell)
            print("BINGO")
            # Insert new quotes
            insert_quotes(exchange, instrument, bid_price, ask_price, bid_volume, ask_volume)
            print(f'{instrument.instrument_id:>6s} -- ({bid_price:>6.2f}) {best_bid_price:>6.2f} :: {best_ask_price:>6.2f} ({ask_price:>6.2f})')
        else:
            price_changes = get_price_changes(instrument.instrument_id, our_mid_price)
    
                # Compute and decide price with dynamic margin
            dynamic_price = compute_and_decide_price_with_dynamic_margin(instrument_order_book, price_changes)
    
            if dynamic_price is not None:
            # Insert quotes with dynamic price
                bid_price = round_down_to_tick(dynamic_price - instrument.tick_size, instrument.tick_size)
                ask_price = round_up_to_tick(dynamic_price + instrument.tick_size, instrument.tick_size)
                max_volume_to_buy = POSITION_LIMIT - position
                max_volume_to_sell = POSITION_LIMIT + position
                bid_volume = min(QUOTED_VOLUME, max_volume_to_buy)
                ask_volume = min(QUOTED_VOLUME, max_volume_to_sell)
                insert_quotes(exchange, instrument, bid_price, ask_price, bid_volume, ask_volume)
                print(f'{instrument.instrument_id:>6s} -- ({bid_price:>6.2f}) {best_bid_price:>6.2f} :: {best_ask_price:>6.2f} ({ask_price:>6.2f})')
            else:
            # Handle case where dynamic price couldn't be computed
                pass

            # Display information for tracking the algorithm's actions
            #print(f'{instrument.instrument_id:>6s} -- ({bid_price:>6.2f}) {best_bid_price:>6.2f} :: {best_ask_price:>6.2f} ({ask_price:>6.2f})')

    seq_no = seq_no + 1    
    # Wait for a few seconds to refresh the quotes
    print(f'\nWaiting for 0.2 seconds.')
    time.sleep(0.2)

    # Clear the displayed information after waiting
    clear_output(wait=True)
                