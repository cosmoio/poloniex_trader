#!/usr/bin/env python3

from calendar import timegm
from message import print_message

import datetime
import requests


def main():
    print_message("[ Poloniex Data Retriever v0.9 ]","log")
    
    filename = "result"
    base = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_STR'

    current_time = datetime.datetime.utcnow()
    current_unix_time = timegm(current_time.timetuple())

    start_time = current_unix_time - 3600*12*1000
    end_time = current_unix_time
    period = 1800

    api_url =  base + '&start='+ str(start_time) + '&end=' + str(end_time) + '&period='+str(period)


    print_message("Configuration","information")
    print_message("Start time: "+str(start_time),"information")
    print_message("End time: "+str(end_time),"information")
    print_message("Period: "+str(period),"information")
    print_message("API Location and Parameters: "+api_url ,"information")
    

    raw_json_data = retrieve_chart_data(api_url) 
    print(raw_json_data)
    #store_chart_data(raw_json_data,filename)


def store_chart_data(data, filename):
    current_time = datetime.datetime.utcnow()

    with open(filename+"_"+str(current_time), 'w') as f:
        f.write(data)

def retrieve_chart_data(api_url):
    print_message("Retrieving Poloniex Data","information")
    r = requests.get(api_url)
    print_message("Done: Retrieving Data","success")
 
    return r.json()
    
    

if __name__ == "__main__":
    main()
