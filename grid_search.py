from sklearn.ensemble import RandomForestRegressor
from multiprocessing import Pool, cpu_count
import itertools
import data_pipeline
from functools import partial

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
                 values=[[10,50,100],['squared_error','absolute_error']],
                 train=None,
                 test=None,
                 validation=None,
                 ticker='TSLA'):
        self.model=model 
        self.hyperparameters=hyperparameters
        self.values=values
        self.ticker=ticker
        
    def get_splits(self):
        fd = data_pipeline.FinancialData(ticker=self.ticker, window=3)
        fd.fetch_single_ticker()
        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = fd.splits(
        train_ratio=0.70, val_ratio=0.15)

    def generate_combinations(self):
        self.grid_configs = [dict(zip(self.hyperparameters,comb)) for comb in itertools.product(*self.values)]


    def run_combinations(self):
        # Funtion to apply parallelism
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
        # Show and save the best model





def main():
    grid=gridSearch()
    grid.get_splits()
    grid.generate_combinations()
    grid.run_combinations()
    #for i in grid.grid_configs:
    #    print(i)
	


if __name__ == '__main__':
	main()