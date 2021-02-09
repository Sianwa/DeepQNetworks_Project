import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import keras
from keras.models import load_model

from agent import AI_Trader
from functions import *
from tqdm import tqdm
import sys



def SimulateTrade(currency_pair):
    #hyperparameters
    window_size = 7
    currency_pair=currency_pair
    data = forexdata_loader(currency_pair)
    data_samples = len(data) - 1

    trained_model = load_model("models/ai_trader_30.h5")
    trader = AI_Trader(window_size, False, "ai_trader_30.h5")

    state = CreateState(data, 0, window_size + 1)
    total_profit = 0
    trader.inventory = []
    buy_signals = []
    sell_signals = []
    tick_data = []

    #trading loop
    for t in range(data_samples):
            action = trader.act(state)
            next_state = CreateState(data, t+1, window_size + 1)
            reward = 0
            

            if action == 1:
                trader.inventory.append(data[t][0])
                buy_signals.append(data[t][0])
                tick_data.append(data[t][0])
                sell_signals.append(np.nan)
                #print("AI Trader BOUGHT:", formatPrice(data[t][0]))

            elif action == 2 and len(trader.inventory) > 0:
                buy_price = trader.inventory.pop(0)
                reward = max(data[t][0] - buy_price, 0)
                total_profit += data[t][0] - buy_price
                sell_signals.append(data[t][0])
                tick_data.append(data[t][0])
                buy_signals.append(np.nan)
                #print("AI Trader SOLD: ", formatPrice(data[t][0]), " Profit: " + formatPrice(data[t][0] - buy_price))
            else:
                tick_data.append(data[t][0])
                sell_signals.append(np.nan)
                buy_signals.append(np.nan)


            if t == data_samples - 1 :
            # this for loop basically closes all open positions as the episode comes to an end. 
        
                for x in trader.inventory:
                    buy_price = trader.inventory.pop(0)
                    reward = max(data[t][0] - buy_price, 0)
                    total_profit += data[t][0] - buy_price
                    sell_signals.append(data[t][0])
                    tick_data.append(data[t][0])
                    #print("AI Trader SOLD: ", formatPrice(data[t][0]), " Profit: " + formatPrice(data[t][0] - buy_price))

                done = True

            else:
                done = False

            trader.memory.append((state, action, reward, next_state, done))
            state = next_state 

            #Plot Signals
            df = pd.DataFrame(tick_data,columns=['Data'])
            BuySignals = pd.DataFrame(buy_signals,columns=['Buy'])
            SellSignals = pd.DataFrame(sell_signals, columns=['Sell'])
            final_data = pd.concat([df,BuySignals,SellSignals],  axis=1,join='outer', ignore_index = False)
            
            if done:
                performance = plotData(final_data)

    profit = str(total_profit)
    return "Profit/Loss generated: " + profit 