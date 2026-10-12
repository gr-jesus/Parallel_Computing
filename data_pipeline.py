import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class FinancialData():
	def __init__(self, 
				ticker='TSLA',
				start='2025-01-01',
				end='2025-12-31'):
		self.ticker = ticker
		self.start = start
		self.end = end
		self.scaler = StandardScaler()

	def fetch_single_ticker(self):
		# In this analysis we only want to predict the close, then we require only that column
		self.data = yf.download(self.ticker, start=self.start, end=self.end)['Close']
		if self.data.empty:
			raise ValueError(f"No se obtuvieron datos para el ticker '{self.ticker}' en el rango especificado.")


	def visualize_data(self):
		plt.plot(self.data)
		plt.show()

	def normalice_data(self):
		self.normalized_data = self.scaler.fit_transform(self.data)
	
	def get_data(self):
		self.fetch_single_ticker()
		self.normalice_data()
		return self.data, self.normalice_data

	



def main():
	fd=FinancialData('TSLA')
	data, normalized_data = fd.get_data()
	#print(fd.data)
	


if __name__ == '__main__':
	main()

