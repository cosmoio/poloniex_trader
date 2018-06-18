#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import signal
import getopt
import datetime
from machine_learning.machine_learning import compute_MACD
from machine_learning.machine_learning import compute_MOM
from util.trader_impl import compute_MOM_trader
from util.trader_impl import compute_hodl
from util.trader_impl import compute_burst_detector


from numpy import array

from util.message import print_message
from util.message import print_summary
from util.message import print_logo

import os, sys
 


program_name = "[ Poloniex trader v0.9 ]"
filename_base = "/home/cosmo/Desktop/test/poloniex_trader/"
dataset_folder = "datasets/"
#dataset_name = "result4"

#dataset_name = "result2"
dataset_name = "btcd_p1800_20160630_20170812"
#dataset_name = "str_p1800_20160630_20170812"

#dataset_name = "result1"


filename_dataset = filename_base+dataset_folder+dataset_name
result_folder = "results/"
result_name = "export.gain"+"_"+dataset_name
filename_result = filename_base+result_folder+result_name
    

def main():
    
    matplotlib.style.use('ggplot')
    
    print_logo(program_name)
    print_message("Changing display options (console width, height, ..)","information")

    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 300)
    pd.set_option('display.width', 300)

    print_message("Reading dataset from "+filename_dataset,"information")
    with open(filename_dataset) as data_file:
        data = json.load(data_file)




    print_message("Creating dataframe","success")
    df = pd.DataFrame(data)

    print_message("Computing descriptive statistics","success")

    #    df = compute_MACD(df, 12,26)
    ema_short = 6
    ema_long = 200
    df = compute_MACD(df, ema_short, ema_long)
    
    #mom = 3
    #df = compute_MOM(df,mom)
 
    plotting_dataframe_macd = pd.DataFrame() #creates a new dataframe that's empty
   
    #plotting_dataframe_macd['MACD'] = df['MACD']
    #plotting_dataframe_macd['close'] = df['close']
    plotting_dataframe_macd['EMA_'+str(ema_short)] = df['EMA_'+str(ema_short)]
    plotting_dataframe_macd['EMA_'+str(ema_long)] = df['EMA_'+str(ema_long)]
   # plotting_dataframe_macd['volume'] = df['volume']
    
    #plotting_dataframe_macd['MACD_diff'] = df['MACD_diff']
    #plotting_dataframe_macd['MOMENTUM_'+str(mom)] = df['MOMENTUM_'+str(mom)]



#    plotting_dataframe_macd['MACD_sign'] = df['MACD_sign']
    #plotting_dataframe_macd['MACD_diff'] = df['MACD_diff']



    plotting_dataframe_macd.plot()

    plt.draw()
    plt.show(block=False)
    
    print_message("Testing predictors","success")

    investment = 2000
    final_investment = 0
    greed = 2
    
    dff = df.copy(deep=True)
    df_results = []
    stats_dataframe = pd.DataFrame() #creates a new dataframe that's empty

    df_results.append(compute_hodl(df,investment).copy(deep=True))
    stats_dataframe = stats_dataframe.append(get_stats(df,investment),ignore_index=True)
    print_summary(investment, stats_dataframe.iloc[0]["final_investment"],df_results[0], filename_result+"_"+"g"+str(greed)+"_"+"i"+str(investment),pd,plt)    
    df_results.append(compute_burst_detector(df,investment,greed))
    stats_dataframe = stats_dataframe.append(get_stats(df,investment),ignore_index=True)
    print_summary(investment, stats_dataframe.iloc[1]["final_investment"],df_results[1], filename_result+"_"+"g"+str(greed)+"_"+"i"+str(investment),pd,plt)    
        
    
    plotting_dataframe_investment = pd.DataFrame() #creates a new dataframe that's empty
    plotting_dataframe_investment['close'] = df['close']
    plotting_dataframe_investment['volume'] = df['volume']
    
    plotting_dataframe_action = pd.DataFrame() #creates a new dataframe that's empty
    plotting_dataframe_action['close'] = df['close']

    plotting_dataframe_investment["return_strategy"] = df['CURRENT_INVESTMENT']
    plotting_dataframe_action["DOLLAR"] = df['DOLLAR']


    for i,dataframe in enumerate(df_results):
        plotting_dataframe_investment["return_strategy_"+str(i)] = df_results[i]['CURRENT_INVESTMENT']
        plotting_dataframe_action["DOLLAR_"+str(i)] = df_results[i]['DOLLAR']

    plotting_dataframe_investment.plot()
    plotting_dataframe_action.plot()
    
    fig = plt.gcf()
    fig2 = plt.gcf()
    
    
    #print_message("Printing chart to file "+filename_result+"_strategies_combined.png","information")
    #fig.savefig(filename_result+"_strategies_combined.png")

            
    determine_winner(stats_dataframe) 
    plt.show()


def determine_winner(stats):
    index_winner = 0
    for index, row in stats.iterrows():
        print_message("Strategy: "+str(index)+" "+str(row["relative_gain"]),"information")
        if row["relative_gain"] > stats.iloc[index_winner]["relative_gain"]: 
           index_winner = index 
    print_message("The winner is strategy: "+str(index_winner),"success")
    



def get_stats(df, initial_investment):
    final_investment = df.iloc[len(df)-1]["DOLLAR"]
    if final_investment == 0 and df.iloc[len(df)-1]["CURRENCY_0"] != 0:
        print_message("No dollar amount, computing revenue based on other currency","information")
        print_message("Currency Amount: "+str(df.iloc[len(df)-1]["CURRENCY_0"])+" Exchange rate: "+str(df.iloc[len(df)-1]["close"]),"information")
        
        final_investment = df.iloc[len(df)-1]["CURRENCY_0"]*df.iloc[len(df)-1]["close"]
    
    date_from = str(datetime.datetime.fromtimestamp(  df.iloc[0]["date"]).strftime('%Y-%m-%d %H:%M:%S')) 
    date_to = str(datetime.datetime.fromtimestamp(  df.iloc[len(df)-1]["date"]).strftime('%Y-%m-%d %H:%M:%S'))
    relative_gain = 100* final_investment/initial_investment
    cum_fee = df.iloc[len(df)-1]["CUM_FEE"]
    
    df = pd.DataFrame([[date_from,date_to,initial_investment,final_investment,relative_gain,cum_fee]],columns=['date_from','date_to','initial_investment','final_investment','relative_gain','cum_fee'])
       
    return df


    
      

def set_sell_trigger(val):
    global SELL_TRIGGER
    SELL_TRIGGER = val
    print_message("Setting external sell trigger: "+str(SELL_TRIGGER),"information")



def exit_gracefully(signum, frame):
    global SELL_TRIGGER
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
#    signal.signal(signal.SIGINT, original_sigint)

    try:
#        if input("\nReally quit? (y/n)> ").lower().startswith('y'):
#            sys.exit(1)
        if input("\Sell? (y/n)> ").lower().startswith('y'):
            set_sell_trigger(True)

    except KeyboardInterrupt:
        print("Ok ok, quitting")
        sys.exit(1)
        

    
    





if __name__ == "__main__":
    #original_sigint = signal.getsignal(signal.SIGINT)
    #signal.signal(signal.SIGINT, exit_gracefully)


    main()



