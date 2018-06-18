'''
TRADERS go here
'''

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import signal
import getopt
import datetime

from numpy import array

from util.message import print_message
from util.message import print_summary
from util.message import print_logo


import os, sys

SELL_TRIGGER = False


def get_current_investment(current_close,currency_0,current_investment_arg):
    if currency_0 > 0:
        current_investment = current_close*currency_0
        return current_investment
    
    return current_investment_arg

def buy_coin(df,index,lastbuy,currency_0,BUY_FEE):
    buy_string = "Buying:   @ {:<12} Amount: {:<15} Index: {:<12}".format(str(df.ix[index,"close"]),str(df.ix[index,"DOLLAR"]/df.ix[index,"close"]),str(index))
    print_message(buy_string,"information")
#    print_message("Buying:  @ "+str(df.ix[index,"close"])+" Amount: "+str(df.ix[index,"DOLLAR"]/df.ix[index,"close"])+" Index: "+str(index),"information")
    currency_0 += (df.ix[index,"DOLLAR"]/df.ix[index,"close"])*(1-BUY_FEE)
    
    df.ix[index-1,"ACTION"] = "BUY"
    df.ix[index,"DOLLAR"] = 0
     #df.ix[index,"DOLLAR"] += currency_0*df.ix[index,"close"]*(1-BUY_FEE)
    df.ix[index,"CUM_FEE"] += currency_0*df.ix[index,"close"]*BUY_FEE
    df.ix[index,"CURRENCY_0"] = currency_0
    
    df.ix[index,"CURRENT_INVESTMENT"] = currency_0*df.ix[index,"close"]
    
    return df.ix[index,"close"], currency_0
    
    
def sell_coin(df,index,currency_0,SELL_FEE):
      sell_string = "Selling:  @ {:<12} Amount: {:<15} Index: {:<12}".format(str(df.ix[index,"close"]),str(currency_0),str(index))
      print_message(sell_string,"information")
#      print_message("Selling: @ "+str(df.ix[index,"close"])+" Amount: "+str(currency_0)+" Index: "+str(index),"information")
      df.ix[index-1,"ACTION"] = "SELL"
      df.ix[index,"CUM_FEE"] += df.ix[index,"close"]*currency_0*SELL_FEE
      df.ix[index,"DOLLAR"] += df.ix[index,"close"]*currency_0*(1-SELL_FEE)
      df.ix[index,"CURRENCY_0"] = 0


      df.ix[index,"CURRENT_INVESTMENT"] = df.ix[index,"DOLLAR"] 
      
      return 0,df.ix[index,"close"]
      
      

    # dont buy too quickly
'''
def compute_MACD_trader(df,investment,greed):
    print_message("Creating \"MACD\" estimator","information")

    #for index, row in df.iterrows():
    #        print (df["close"])
   
    percentage = greed
    margin_buy = [1.01]
    margin_buy_low = [1.001]
    
    margin_sell = [1.05]
    
    
    currency_0 = 0                              # assume we have a bitcoin
    #initial_buy_cost = df.ix[0,"close"]         # initial cost of buying
    close_column = df["close"] 
    
    df["DOLLAR"] = 0
    df["CUM_FEE"] = 0
    df["ACTION"] = "HOLD"                   # WHAT DO WE DO? Buy, Hold, or sell? Default is Hold
    df["CURRENCY_0"] = 0
    df["CURRENT_INVESTMENT"] = 0

    df.ix[0,"DOLLAR"] = investment
    
    SELL_FEE = 0.0015
    BUY_FEE = 0.0025
    lastbuy = 10000000             # Initial setting, such that the algorithm will buy
    lastsell  = 10000000           # Initial setting, such that the algorithm will buy
    
    MACD_STEEPNESS = 1
    
    # Storing the amount at which the currency was bought could make sense
    max_iter = len(df)
    index = 1
    
    ms_coeff = 0
    mb_coeff = 0
    mb_boost = 1.001
    
    while index < max_iter: 
        df.ix[index,"CUM_FEE"] = df.ix[index-1,"CUM_FEE"]
        df.ix[index,"DOLLAR"] = df.ix[index-1,"DOLLAR"]
        df.ix[index,"CURRENCY_0"] = df.ix[index-1,"CURRENCY_0"]
        df.ix[index,"CURRENT_INVESTMENT"] =  df.ix[index-1,"CURRENT_INVESTMENT"]


        #if (df.ix[index,"close"] >= margin_sell*lastbuy*(1+(percentage/100))) and currency_0 > 0:
        if  (
                ((df.ix[index,"MACD_diff"] < 0 and df.ix[index,"close"] > margin_sell[ms_coeff]*lastbuy) 
                or  (df.ix[index,"MACD_diff"] > MACD_STEEPNESS and df.ix[index,"close"] > margin_sell[ms_coeff]*lastbuy)) 
                and  currency_0 > 0
            ): 
                
            # SELL
                currency_0,lastsell = sell_coin(df,index,currency_0,SELL_FEE) 
                
        #elif df.ix[index,"close"] <= margin_buy*lastsell*(1-(percentage/100)):         # safe version
        elif (
                ((df.ix[index,"MACD_diff"] > 0 and df.ix[index,"close"] < margin_buy[mb_coeff]*lastsell) or
                (df.ix[index,"MACD_diff"] > MACD_STEEPNESS*10 and df.ix[index,"close"] < mb_boost*margin_buy_low[mb_coeff]*lastsell))                  
                and df.ix[index,"DOLLAR"] > 0
             ):
            # BUY
                mb_boost = 1.001
                lastbuy,currency_0 = buy_coin(df,index,lastbuy,currency_0,BUY_FEE)
                
                
        df.ix[index,"CURRENT_INVESTMENT"] = get_current_investment(df.ix[index,"close"],currency_0,df.ix[index,"CURRENT_INVESTMENT"])
        index += 1

        ms_coeff = (ms_coeff + 1) % len(margin_sell)
        mb_coeff = (mb_coeff + 1) % len(margin_buy)

        mb_boost += 0.1
        mb_boost = 1.001 if mb_boost > 1.5 else mb_boost
        
      #  print("ms_boost: "+str(ms_boost))

    print_message("\"MACD Estimator\": Done","success")
    return df
'''

def compute_MACD_trader(df,investment,greed):
    print_message("Creating \"MACD\" estimator","information")

    #for index, row in df.iterrows():
    #        print (df["close"])
   
    percentage = greed
    margin_above_buy = 1.01
    margin_below_buy = 0.99
    
    margin_above_sell = 1.08
    margin_below_sell = 0.98
    
    currency_0 = 0                              # assume we have a bitcoin
    #initial_buy_cost = df.ix[0,"close"]         # initial cost of buying
    close_column = df["close"] 
    
    df["DOLLAR"] = 0
    df["CUM_FEE"] = 0
    df["ACTION"] = "HOLD"                   # WHAT DO WE DO? Buy, Hold, or sell? Default is Hold
    df["CURRENCY_0"] = 0
    df["CURRENT_INVESTMENT"] = 0

    df.ix[0,"DOLLAR"] = investment
    
    SELL_FEE = 0.0015
    BUY_FEE = 0.0025
    lastbuy = 10000000             # Initial setting, such that the algorithm will buy
    lastsell  = 10000000           # Initial setting, such that the algorithm will buy
    lastdiff_buy = 0
    lastdiff_sell = 0
    
    MACD_STEEPNESS = 1
    
    # Storing the amount at which the currency was bought could make sense
    max_iter = len(df)
    index = 1
    
    import math
    
    while index < max_iter: 
        df.ix[index,"CUM_FEE"] = df.ix[index-1,"CUM_FEE"]
        df.ix[index,"DOLLAR"] = df.ix[index-1,"DOLLAR"]
        df.ix[index,"CURRENCY_0"] = df.ix[index-1,"CURRENCY_0"]
        df.ix[index,"CURRENT_INVESTMENT"] =  df.ix[index-1,"CURRENT_INVESTMENT"]


        #print_message("MACD_diff: "+str(df.ix[index,"MACD_diff"])+" "+" lastdiff_buy: "+str(lastdiff_buy)+" "+"lastdiff_sell: "+str(lastdiff_sell)+" close: "+str(df.ix[index,"close"])+" "+"lastsell: "+str(lastsell)+" "+"lastbuy: "+str(lastbuy)+" "+" index: "+str(index),"log")

       # print_message("Condition: SELL:  ("+str(df.ix[index-1,"MACD_diff"] > df.ix[index,"MACD_diff"]) + " and  " + str((df.ix[index,"close"] >= lastbuy*margin_above_sell)) + ")","log")
       # print_message("Condition: BUY:  ("+str(lastdiff_buy < df.ix[index,"MACD_diff"]) + " and " + str((df.ix[index,"close"] < lastsell*margin_above_buy)) + ") or " + str(df.ix[index,"close"] < lastsell*margin_below_buy),"log")

        if math.isnan(df.ix[index,"MACD_diff"]):
            index+=1
            continue
        

        #if (df.ix[index,"close"] >= margin_sell*lastbuy*(1+(percentage/100))) and currency_0 > 0:
        if  (
                #((df.ix[index,"MACD_diff"] > 0 and df.ix[index,"close"] >= lastbuy*margin_sell) or
                #(df.ix[index,"MACD_diff"] > 20 and df.ix[index,"close"] >= 0.94*lastbuy*margin_sell))
                
                (df.ix[index,"close"] >= lastbuy*margin_above_sell)
                and  currency_0 > 0
            ): 
                
            # SELL
                lastdiff_sell = df.ix[index,"MACD_diff"]
                currency_0,lastsell = sell_coin(df,index,currency_0,SELL_FEE) 
                
        #elif df.ix[index,"close"] <= margin_buy*lastsell*(1-(percentage/100)):         # safe version
        elif (
                #(df.ix[index,"MACD_diff"] < 0 and df.ix[index,"close"] < lastsell*margin_buy)
                df.ix[index,"DOLLAR"] > 0 and
                (df.ix[index,"close"] <= lastsell*margin_below_buy or (df.ix[index,"MOMENTUM_9"] > 0 and df.ix[index,"close"] <= lastsell*margin_above_buy))
                 
                
             ):
            # BUY
                lastdiff_buy = df.ix[index,"MACD_diff"]
                lastbuy,currency_0 = buy_coin(df,index,lastbuy,currency_0,BUY_FEE)
                
                
        df.ix[index,"CURRENT_INVESTMENT"] = get_current_investment(df.ix[index,"close"],currency_0,df.ix[index,"CURRENT_INVESTMENT"])
        index += 1

        
      #  print("ms_boost: "+str(ms_boost))

    print_message("\"MACD Estimator\": Done","success")
    return df
        




def compute_MOM_trader(df,investment,greed):
    print_message("Creating \"MOM\" estimator","information")

    #for index, row in df.iterrows():
    #        print (df["close"])
   
    percentage = greed
    margin_above_buy = 1.01
    margin_below_buy = 0.99
    
    margin_above_sell = 1.08
    margin_below_sell = 0.98
    
    currency_0 = 0                              # assume we have a bitcoin
    #initial_buy_cost = df.ix[0,"close"]         # initial cost of buying
    close_column = df["close"] 
    
    df["DOLLAR"] = 0
    df["CUM_FEE"] = 0
    df["ACTION"] = "HOLD"                   # WHAT DO WE DO? Buy, Hold, or sell? Default is Hold
    df["CURRENCY_0"] = 0
    df["CURRENT_INVESTMENT"] = 0

    df.ix[0,"DOLLAR"] = investment
    
    SELL_FEE = 0.0015
    BUY_FEE = 0.0025
    lastbuy = 10000000             # Initial setting, such that the algorithm will buy
    lastsell  = 10000000           # Initial setting, such that the algorithm will buy
    
    MACD_STEEPNESS = 1
    
    # Storing the amount at which the currency was bought could make sense
    max_iter = len(df)
    index = 1
    
    import math
    
    while index < max_iter: 
        df.ix[index,"CUM_FEE"] = df.ix[index-1,"CUM_FEE"]
        df.ix[index,"DOLLAR"] = df.ix[index-1,"DOLLAR"]
        df.ix[index,"CURRENCY_0"] = df.ix[index-1,"CURRENCY_0"]
        df.ix[index,"CURRENT_INVESTMENT"] =  df.ix[index-1,"CURRENT_INVESTMENT"]


        #print_message("MACD_diff: "+str(df.ix[index,"MACD_diff"])+" "+" lastdiff_buy: "+str(lastdiff_buy)+" "+"lastdiff_sell: "+str(lastdiff_sell)+" close: "+str(df.ix[index,"close"])+" "+"lastsell: "+str(lastsell)+" "+"lastbuy: "+str(lastbuy)+" "+" index: "+str(index),"log")

       # print_message("Condition: SELL:  ("+str(df.ix[index-1,"MACD_diff"] > df.ix[index,"MACD_diff"]) + " and  " + str((df.ix[index,"close"] >= lastbuy*margin_above_sell)) + ")","log")
       # print_message("Condition: BUY:  ("+str(lastdiff_buy < df.ix[index,"MACD_diff"]) + " and " + str((df.ix[index,"close"] < lastsell*margin_above_buy)) + ") or " + str(df.ix[index,"close"] < lastsell*margin_below_buy),"log")

        if math.isnan(df.ix[index,"MACD_diff"]):
            index+=1
            continue
        

        #if (df.ix[index,"close"] >= margin_sell*lastbuy*(1+(percentage/100))) and currency_0 > 0:
        if  (
                #((df.ix[index,"MACD_diff"] > 0 and df.ix[index,"close"] >= lastbuy*margin_sell) or
                #(df.ix[index,"MACD_diff"] > 20 and df.ix[index,"close"] >= 0.94*lastbuy*margin_sell))
                
                (df.ix[index,"close"] >= lastbuy*margin_above_sell or (df.ix[index,"MOMENTUM_3"] < 0 and df.ix[index,"close"] >= lastbuy*0.98))
                and  currency_0 > 0
            ): 
                
            # SELL
                currency_0,lastsell = sell_coin(df,index,currency_0,SELL_FEE) 
                
        #elif df.ix[index,"close"] <= margin_buy*lastsell*(1-(percentage/100)):         # safe version
        elif (
                #(df.ix[index,"MACD_diff"] < 0 and df.ix[index,"close"] < lastsell*margin_buy)
                df.ix[index,"DOLLAR"] > 0 and
                (df.ix[index,"close"] <= lastsell*margin_below_buy or (df.ix[index-1,"MACD_diff"] < df.ix[index,"MACD_diff"] and df.ix[index,"close"] <= lastsell*margin_above_buy))
                 
                
             ):
            # BUY
                lastbuy,currency_0 = buy_coin(df,index,lastbuy,currency_0,BUY_FEE)
                
                
        df.ix[index,"CURRENT_INVESTMENT"] = get_current_investment(df.ix[index,"close"],currency_0,df.ix[index,"CURRENT_INVESTMENT"])
        index += 1

        
      #  print("ms_boost: "+str(ms_boost))

    print_message("\"MOM Estimator\": Done","success")
    return df




def compute_hodl(df, investment):
    print_message("Creating \"Hodl\" estimator","information")
    
    
    currency_0 = 0                              # assume we have a bitcoin
    close_column = df["close"] 
    
    df["DOLLAR"] = 0
    df["CUM_FEE"] = 0
    df["ACTION"] = "HOLD"                   # WHAT DO WE DO? Buy, Hold, or sell? Default is Hold
    df["CURRENCY_0"] = 0
    df["CURRENT_INVESTMENT"] = 0

    df.ix[0,"DOLLAR"] = investment
    df.ix[1,"DOLLAR"] = investment
    
    BUY_FEE = 0.0025
    lastbuy = 10000000             # Initial setting, such that the algorithm will buy
    
    # Storing the amount at which the currency was bought could make sense 
    # Ideally we could try to skip long dry periods
    lastbuy, currency_0 = buy_coin(df,1,lastbuy,currency_0,BUY_FEE)
    
    df["CUM_FEE"] = df.ix[1,"CUM_FEE"]
    df["CURRENCY_0"] = currency_0
    df.ix[0,"CURRENCY_0"] = 0
    df["CURRENT_INVESTMENT"] = currency_0*df["close"]       
        
    print(df.ix[len(df)-1,"CURRENT_INVESTMENT"])
        
    print_message("\"Hodl Estimator\": Done","success")
    return df

def compute_linear(df, investment,greed):
    print_message("Creating \"Rising BTC Bet\" estimator","information")

    #for index, row in df.iterrows():
    #        print (df["close"])
   
    percentage = greed
    margin_buy = 0.99
    margin_sell = 1.02
    
    
    currency_0 = 0                              # assume we have a bitcoin
    #initial_buy_cost = df.ix[0,"close"]         # initial cost of buying
    close_column = df["close"] 
    
    df["DOLLAR"] = 0
    df["CUM_FEE"] = 0
    df["ACTION"] = "HOLD"                   # WHAT DO WE DO? Buy, Hold, or sell? Default is Hold
    df["CURRENCY_0"] = 0
    df["CURRENT_INVESTMENT"] = 0

    df.ix[0,"DOLLAR"] = investment
    
    SELL_FEE = 0.0015
    BUY_FEE = 0.0025
    lastbuy = 10000000             # Initial setting, such that the algorithm will buy
    lastsell  = 10000000           # Initial setting, such that the algorithm will buy
    
    global SELL_TRIGGER
    # Storing the amount at which the currency was bought could make sense 
    for index, row in df.iloc[1:].iterrows():
        df.ix[index,"CUM_FEE"] = df.ix[index-1,"CUM_FEE"]
        df.ix[index,"DOLLAR"] = df.ix[index-1,"DOLLAR"]
        df.ix[index,"CURRENCY_0"] = df.ix[index-1,"CURRENCY_0"]
        df.ix[index,"CURRENT_INVESTMENT"] =  df.ix[index-1,"CURRENT_INVESTMENT"]

        #if (df.ix[index,"close"] >= margin_sell*lastbuy*(1+(percentage/100))) and currency_0 > 0:
        if ((df.ix[index,"close"] >= margin_sell*lastbuy*(1+(percentage/100))) or SELL_TRIGGER) and currency_0 > 0:
            # SELL
                set_sell_trigger(False)
                currency_0,lastsell = sell_coin(df,index,currency_0,SELL_FEE) 
                
        #elif df.ix[index,"close"] <= margin_buy*lastsell*(1-(percentage/100)):         # safe version
        elif (df.ix[index,"close"] < margin_buy*lastsell) or (df.ix[index,"close"] >= lastsell*1.01): #*(1-(percentage/100))):        # buy if the price didn't double during the last period
            # BUY
            if df.ix[index,"DOLLAR"] > 0:
                lastbuy,currency_0 = buy_coin(df,index,lastbuy,currency_0,BUY_FEE)
                
                
        df.ix[index,"CURRENT_INVESTMENT"] = get_current_investment(df.ix[index,"close"],currency_0,df.ix[index,"CURRENT_INVESTMENT"])
            #df.ix[index,"LINEAR"] = (df.ix[index]["close"] - close_column.ix[index-1]) / close_column.ix[index-1]
       # print( df.ix[index]["LINEAR"] )
  #  df["close"]i-1 - df["close"] / df["close"]i-1
    
    print_message("\"Rising BTC Bet Estimator\": Done","success")
    return df



def compute_burst_detector(df, investment,greed):
    print_message("Creating \"Burst\" estimator","information")

    #for index, row in df.iterrows():
    #        print (df["close"])
   
    percentage = 1+(greed/100)
    margin_buy = 0.99
    margin_sell = 1.02
    
    
    currency_0 = 0                              # assume we have a bitcoin
    #initial_buy_cost = df.ix[0,"close"]         # initial cost of buying
    close_column = df["close"] 
    
    df["DOLLAR"] = 0
    df["CUM_FEE"] = 0
    df["ACTION"] = "HOLD"                   # WHAT DO WE DO? Buy, Hold, or sell? Default is Hold
    df["CURRENCY_0"] = 0
    df["CURRENT_INVESTMENT"] = 0

    df.ix[0,"DOLLAR"] = investment
    
    SELL_FEE = 0.0015
    BUY_FEE = 0.0025
    lastbuy = 10000000             # Initial setting, such that the algorithm will buy
    lastsell  = 10000000           # Initial setting, such that the algorithm will buy
    
    global SELL_TRIGGER
    # Storing the amount at which the currency was bought could make sense 
    index = 1
    max_iter = len(df)
    
    
    # Assumption: Burst, we buy while it's cheap then sell AND wait for things to return to being normal
    price_before_burst = 0
    target_profit = 3.0
    
    df['EMA_vol'] = pd.ewma(df["volume"], span=3)
    df["MA"] = pd.Series(pd.rolling_mean(df['close'], 400))  
    pause = 0
    
    while index < max_iter:
        df.ix[index,"CUM_FEE"] = df.ix[index-1,"CUM_FEE"]
        df.ix[index,"DOLLAR"] = df.ix[index-1,"DOLLAR"]
        df.ix[index,"CURRENCY_0"] = df.ix[index-1,"CURRENCY_0"]
        df.ix[index,"CURRENT_INVESTMENT"] =  df.ix[index-1,"CURRENT_INVESTMENT"]


       # print(str(df.ix[index,"EMA_vol"])+" "+str(df.ix[index,"volume"])+" "+str(df.ix[index,"EMA_20"])+" "+str(df.ix[index,"close"]))
          
        #if (df.ix[index,"close"] >= margin_sell*lastbuy*(1+(percentage/100))) and currency_0 > 0:
        #if currency_0 > 0 and ((df.ix[index,"close"] >= margin_sell*lastbuy*percentage*1.3) or (lastbuy > df.ix[index,"close"]*1.2 and get_current_investment(df.ix[index,"close"],currency_0,df.ix[index,"CURRENT_INVESTMENT"]) > 2000)):
        if (df.ix[index,"close"] >= lastbuy*3) and currency_0 > 0:
            # SELL
            currency_0,lastsell = sell_coin(df,index,currency_0,SELL_FEE) 
        #elif df.ix[index,"close"] <= margin_buy*lastsell*(1-(percentage/100)):         # safe version
        if df.ix[index,"close"] < lastsell:
            if df.ix[index,"DOLLAR"] > 0:     # BUY
                lastbuy,currency_0 = buy_coin(df,index,lastbuy,currency_0,BUY_FEE)
    
        
        #elif df.ix[index,"close"] < lastsell*1.05 and df.ix[index-1,"volume"]*3 < df.ix[index,"volume"]:
         #   if df.ix[index,"DOLLAR"] > 0:     # BUY
          #      lastbuy,currency_0 = buy_coin(df,index,lastbuy,currency_0,BUY_FEE)
    
        elif df.ix[index,"MA"]*2 < df.ix[index,"close"]: #and df.ix[index-1,"volume"] < df.ix[index,"volume"]:
            print_message("Too high, don't buy  MA: "+str(df.ix[index,"MA"])+" Index: "+str(index),"information")
    
    
    
        df.ix[index,"CURRENT_INVESTMENT"] = get_current_investment(df.ix[index,"close"],currency_0,df.ix[index,"CURRENT_INVESTMENT"])
        index += 1
    
    print_message("\"Burst\": Done","success")
    return df
