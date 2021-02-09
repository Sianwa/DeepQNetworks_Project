import keras
from keras.models import load_model

from agent import AI_Trader
from functions import *
from tqdm import tqdm
import sys

trained_model = load_model("models/ai_trader_30.h5")

#hyperparameters
window_size = 5
batch_size = 32
data = forexdata_loader("EUR/USD")
data_samples = len(data) - 1
trader = AI_Trader(window_size, False, "ai_trader_30.h5")

state = CreateState(data, 0, window_size + 1)
total_profit = 0
trader.inventory = []
buy_signals = []
sell_signals = []
tick_data = []


for t in tqdm(range(data_samples)):
        action = trader.act(state)
        next_state = CreateState(data, t+1, window_size + 1)
        reward = 0
        

        if action == 1:
            trader.inventory.append(data[t][0])
            buy_signals.append(data[t][0])
            tick_data.append(data[t][0])
            sell_signals.append(np.nan)
            print("AI Trader BOUGHT:", formatPrice(data[t][0]))

        elif action == 2 and len(trader.inventory) > 0:
            buy_price = trader.inventory.pop(0)
            reward = max(data[t][0] - buy_price, 0)
            total_profit += data[t][0] - buy_price
            sell_signals.append(data[t][0])
            tick_data.append(data[t][0])
            buy_signals.append(np.nan)
            print("AI Trader SOLD: ", formatPrice(
                data[t][0]), " Profit: " + formatPrice(data[t][0] - buy_price))
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
                print("AI Trader SOLD: ", formatPrice(
                data[t][0]), " Profit: " + formatPrice(data[t][0] - buy_price))

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
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("Final Inventory", trader.inventory)
            performance = plotData(final_data)
            #print(trader.memory)
            print("########################")