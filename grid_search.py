from sklearn.ensemble import RandomForestRegressor
from multiprocessing import Pool, cpu_count
import itertools
import data_pipeline
from functools import partial
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_single_combination(params, model_cls, X_train, y_train, X_val, y_val):
    """
    Función pura a nivel de módulo: fácilmente serializable por multiprocessing.
    """
    model = model_cls(**params)
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    return {
        'params': params,
        'val_score': score,
        'model': model
    }


class gridSearch():
    def __init__(self,
                 model=RandomForestRegressor,
                 hyperparameters=['n_estimators', 'criterion'],
                 values=[[10,50,100,200,250],['squared_error','absolute_error']],
                 train=None,
                 test=None,
                 validation=None,
                 ticker='NVDA'):
        self.model=model 
        self.hyperparameters=hyperparameters
        self.values=values
        self.ticker=ticker
        
    def get_splits(self):
        fd = data_pipeline.FinancialData(ticker=self.ticker, window=4)
        fd.fetch_single_ticker()
        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = fd.splits(
        train_ratio=0.70, val_ratio=0.15)

    def generate_combinations(self):
        self.grid_configs = [dict(zip(self.hyperparameters,comb)) for comb in itertools.product(*self.values)]


    def evaluate_test(self, verbose=True):
        if not hasattr(self, 'best_result') or self.best_result is None:
            raise ValueError("Debes ejecutar 'run_combinations()' antes de evaluar.")

        best_model = self.best_result['model']
        y_pred = best_model.predict(self.X_test)

        r2 = r2_score(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)

        if verbose:
            print("\n" + "="*45)
            print("      EVALUACIÓN EN CONJUNTO DE PRUEBA (TEST)     ")
            print("="*45)
            print(f"Modelo:              {self.model.__name__}")
            print(f"Mejores Parámetros:  {self.best_result['params']}")
            print(f"R² en Validación:    {self.best_result['val_score']:.4f}")
            print("-" * 45)
            print(f"R² en Prueba (Test): {r2:.4f}")
            print(f"MSE en Prueba:       {mse:.6f}")
            print(f"MAE en Prueba:       {mae:.6f}")
            print("="*45)

        return {'r2': float(r2), 'mse': float(mse), 'mae': float(mae), 'predictions': y_pred}

    def run_combinations(self):
        # Function to apply parallelism
        worker_fn = partial(evaluate_single_combination,
                            model_cls=self.model,
                            X_train=self.X_train,
                            y_train=self.y_train,
                            X_val=self.X_val,
                            y_val=self.y_val)
        # Run in parallel
        with Pool(processes=cpu_count()-1) as pool:
            results = pool.map(worker_fn, self.grid_configs)
        # Review the best configuration
        for i in results:
            print(i)
        # Evaluate in the test set
        self.best_result = max(results, key=lambda res: res['val_score'])
        self.evaluate_test()
        # Show and save the best model


from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor

def main():
    #SVM
    grid=gridSearch(model=SVR,
                    hyperparameters=['kernel','C','epsilon','gamma'],
                    values=[
                        ['rbf','linear'],
                        [0.1, 1.0, 10.0, 100.0],
                        [0.01, 0.05, 0.1, 0.2],
                        ['scale', 0.01, 0.1]
                    ],
                    ticker='NVDA')

    #grid=gridSearch(model=ElasticNet,
    #                hyperparameters=['alpha','l1_ratio'],
    #                values=[[0.01, 0.1, 1.0],[0.2, 0.5, 0.8]])

    #grid=gridSearch(model=LGBMRegressor,
    #                hyperparameters=['n_estimators','learning_rate', 'max_depth'],
    #                values=[[50,100,200],[0.01, 0.05, 0.1],[15, 30]])


    grid.get_splits()
    grid.generate_combinations()
    grid.run_combinations()
	


if __name__ == '__main__':
	main()