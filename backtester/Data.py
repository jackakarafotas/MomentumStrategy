import pandas as pd 

DEFAULT_DIRECTORY = "SP500"
START_DATE = "20050101"
END_DATE = "20180101"

class Data:
	def __init__(self,stock_universe_dir=DEFAULT_DIRECTORY,start_date = START_DATE,end_date = END_DATE):
		self.directory = stock_universe_dir

		start_date = pd.to_datetime(start_date,format='%Y%m%d', errors='ignore')
		end_date = pd.to_datetime(end_date,format='%Y%m%d', errors='ignore')
		self.dates = pd.date_range(start_date,end_date)

	def getTickerData(self,ticker):
		tickerData = pd.read_csv(self.directory+'/{0}.csv'.format(ticker),index_col=0)
		tickerData.index = pd.to_datetime(tickerData.index)
		return tickerData

	def loadData(self,tickers):
		self.prices = pd.DataFrame(index=self.dates,columns=tickers)
		for ticker in tickers:
			try:
				tick = self.getTickerData(ticker)
				idx = tick.index.intersection(self.prices.index)
				self.prices.loc[idx,ticker] = tick.loc[idx,'Close']
			except:
				pass
		self.prices.dropna(how='all',inplace=True)

data = Data()
data.loadData(["JPM","FB","GOOG"])
print(data.prices)