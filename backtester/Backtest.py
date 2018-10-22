import pandas as pd 
import numpy as np 

class Backtest:
	def __init__(self,data,strategy):
		# data 		: pd.DataFrame of all the stock data indexed by date, 
		#			  with the columns being the closing prices for each stock
		# strategy	: strategy object that has the method .run(dataSeries)
		self.data = data
		self.assets = self.data.columns.values
		self.data['cash'] = 1
		self.strategy = strategy

		self.trades = None
		self.positions = None
		self.portfolio_size = None
		self.returns = None

	def run(self):
		self.trades = pd.DataFrame(0,index=self.data.index,columns=self.data.columns)
		for date in self.data.index:
			new_positions = self.strategy.run(self.data.loc[date,self.assets]) # new posiiton = pd.Series w/ each val = trades in terms of positions (NaN filled w/ 0)
			self.trades.loc[date,self.assets] = new_positions 
			self.trades.loc[date,'cash'] = -np.dot(new_positions.values,self.data.loc[date,self.assets].fillna(0).values)

		self.positions = self.trades.cumsum() 

	def getPortfolioSize(self):
		if (not self.trades):
			self.run()
		self.portfolio_size = pd.Series(index=self.data.index)
		for date in self.data.index:
			self.portfolio_size[date] = np.dot(np.append(self.data.loc[date].values,[1]), self.trades.loc[date].values)
		return self.portfolio_size

	def getReturns(self):
		if (not self.portfolio_size):
			self.getPortfolioSize()
		self.returns = self.portfolio_size - self.portfolio_size.shift(1)
		return self.returns




