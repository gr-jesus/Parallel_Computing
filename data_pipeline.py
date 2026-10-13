import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

class FinancialData():
	def __init__(self, 
				ticker='TSLA',
				start='2024-01-01',
				end='2025-12-31',
				window=3):
		self.ticker = ticker
		self.start = start
		self.end = end
		self.scaler = StandardScaler()
		self.window=window

	def fetch_single_ticker(self):
		# In this analysis we only want to predict the close, then we require only that column
		self.data = yf.download(self.ticker, start=self.start, end=self.end)['Close']
		if self.data.empty:
			raise ValueError(f"No se obtuvieron datos para el ticker '{self.ticker}' en el rango especificado.")

		log_ret = np.log(self.data).diff().dropna()
		self.log_returns = log_ret.values.reshape(-1, 1)


	def visualize_data(self):
		plt.plot(self.data)
		plt.show()

	def normalice_data(self):
		self.normalized_data = self.scaler.fit_transform(self.data)
	
	def get_data(self):
		self.fetch_single_ticker()
		self.normalice_data()
		return self.data, self.normalice_data

	def _create_sequences(self, data_array):
		"""
		Crea matrices X e y basadas en la ventana temporal:
		X[i] = [p_{i}, p_{i+1}, ..., p_{i+window-1}]
		y[i] = p_{i+window}
		"""
		X, y = [], []
		for i in range(len(data_array) - self.window):
			X.append(data_array[i : i + self.window].flatten())
			y.append(data_array[i + self.window].flatten()[0])
		return np.array(X), np.array(y)

	def splits(self, train_ratio=0.70, val_ratio=0.15):
		n_samples = len(self.log_returns)
		train_end = int(n_samples * train_ratio)
		val_end = int(n_samples * (train_ratio + val_ratio))

		# Particiones temporales de retornos
		raw_train = self.log_returns[:train_end]
		raw_val = self.log_returns[train_end - self.window : val_end]
		raw_test = self.log_returns[val_end - self.window :]

        # Ajuste del StandardScaler sólo sobre entrenamiento
		scaled_train = self.scaler.fit_transform(raw_train)
		scaled_val = self.scaler.transform(raw_val)
		scaled_test = self.scaler.transform(raw_test)

        # Creación de secuencias X, y
		X_train, y_train = self._create_sequences(scaled_train)
		X_val, y_val = self._create_sequences(scaled_val)
		X_test, y_test = self._create_sequences(scaled_test)

		return (X_train, y_train), (X_val, y_val), (X_test, y_test)


	



def main():
	fd = FinancialData(ticker='TSLA', window=3)
	fd.fetch_single_ticker()
    
	(X_train, y_train), (X_val, y_val), (X_test, y_test) = fd.splits(
        train_ratio=0.70, 
        val_ratio=0.15
    )

	print(f"Dimensiones de entrenamiento: X={X_train.shape}, y={y_train.shape}")
	print(f"Dimensiones de validación:    X={X_val.shape}, y={y_val.shape}")
	print(f"Dimensiones de prueba:        X={X_test.shape}, y={y_test.shape}")
	print(f"\nEjemplo de X_train[0]: {X_train[0]} -> y_train[0]: {y_train[0]}")
	


if __name__ == '__main__':
	main()

