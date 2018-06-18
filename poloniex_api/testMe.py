import sys, getopt

from poloniex import Poloniex

def main(argv):
  polo = Poloniex('API KEY','SECRET KEY')
  
  print (polo.returnBalances())
  
if __name__ == "__main__":
  main(sys.argv[1:])
