from colorama import init, Fore, Style
import os, sys
import datetime

def print_logo(program_name):
    rows, columns = os.popen('stty size', 'r').read().split()
    size = int(columns)-10
    print("\n\n"+Style.BRIGHT+Fore.YELLOW+program_name.center(size,"#")+"\n")
    Fore.WHITE
    
def print_message(message, message_type):
    if message_type == "information":
        print(Style.BRIGHT + Fore.YELLOW + "[!] " + Style.RESET_ALL + message)
    if message_type == "success":
        print(Style.BRIGHT + Fore.GREEN + "[+] " + Style.RESET_ALL + message)
    if message_type == "error":
        print(Style.BRIGHT + Fore.RED + "[!] " + Style.RESET_ALL + message)
    if message_type == "log":
        print(Style.BRIGHT + Fore.BLUE + "[~] " + Style.RESET_ALL + message)
    

def print_summary(initial_investment,final_investment,df, filename,pd,plt):
    rows, columns = os.popen('stty size', 'r').read().split()
    size = int(columns)-10

    
    if initial_investment < final_investment:
        color = Fore.GREEN
    elif initial_investment == final_investment:
        color = Fore.YELLOW
    else:
        color = Fore.RED
    
    date_from = str(datetime.datetime.fromtimestamp(  df.iloc[0]["date"]).strftime('%Y-%m-%d %H:%M:%S')) 
    date_to = str(datetime.datetime.fromtimestamp(  df.iloc[len(df)-1]["date"]).strftime('%Y-%m-%d %H:%M:%S'))
    
    print_message("Printing results to file ...","information")
    df.to_csv(filename)
    print_message("Done: Writing "+filename,"success")
    

    gains_text = "[   GAINS   ]"    
    space = " "
    initial_investment_str =    lr_justify('Initial Investment: {:>21}'.format(str(initial_investment)),"|",size-3)
    final_investment_str =      lr_justify('Final Investment (after deduction): {:>13}'.format(str(final_investment)),"|",size-3)
    initial_close =             lr_justify('Initial Close: {:>28}'.format(str(df.iloc[0]["close"])),"|",size-3)
    final_close =             lr_justify('Final Close: {:>30}'.format(str(df.iloc[len(df)-1]["close"])),"|",size-3)
    total_fees =            lr_justify('Total Fees: {:>38}'.format(str(df.ix[len(df)-1,"CUM_FEE"])),"|",size-3)
    date_range =            lr_justify('Date Range',"|",size-3)
    from_date =             lr_justify('From: {:>50}'.format(date_from),"|",size-3)
    to_date =               lr_justify('To:   {:>50}'.format(date_to),"|",size-3)
    actions =               lr_justify('Committed Actions',"|",size-3)
    hold =                  lr_justify("Holds: ","|",size-3)
    buy =                   lr_justify("Buys:  ","|",size-3)
    sell =                  lr_justify("Sells: ","|",size-3)
    relative_gain =         lr_justify("Relative gain: {:>36}".format(str(100* final_investment/initial_investment) + "%"),"|",size-3)
    profit =                lr_justify("Profit: {:>42}".format(str(df.ix[len(df)-1,"CURRENT_INVESTMENT"]-initial_investment)),"|",size-3)
    
    
    print("\n\n"+Style.BRIGHT+color+gains_text.center(size,"#"))
    print("|"+" "*(size-2)+"|")
    print('|  '+initial_investment_str)
    print('|  '+final_investment_str)
    print('|  '+total_fees)
    print('|'+'-'*(size-1))
    print('|  '+date_range)
    print('|  '+from_date)
    print('|  '+to_date)
    print('|'+'-'*(size-1))
    print('|  '+actions)
    print('|  '+hold)
    print('|  '+buy)
    print('|  '+sell)
    print('|'+'-'*(size-1))
    print('|  '+initial_close)
    print('|  '+final_close)
    print('|  '+profit)
    print('|  '+relative_gain)
    print("#"*(size))        
    

    #df_plot = pd.concat([df['DOLLAR'], df['CUM_FEE'], df['close']], axis=1, keys=['DOLLAR', 'CUM_FEE','close'])
    #plt.figure() 
    #df_plot.plot()
    #plt.draw()




def lr_justify(left, right, width):
    return '{}{}{}'.format(left, ' '*(width-len(left+right)), right)
