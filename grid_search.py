from sklearn.ensemble import RandomForestRegressor
from multiprocessing import Pool, cpu_count
import itertools


class gridSearch():
    def __init__(self,
                 model=RandomForestRegressor,
                 hyperparameters=['n_estimators', 'criterion'],
                 values=[[10,50,100],['squared_error','absolute_error','poisson']],
                 train=None,
                 test=None,
                 validation=None):
        self.model=model 
        self.hyperparameters=hyperparameters
        self.values=values
        self.train=train
        self.test=test
        self.validation=validation

    def generate_combinations(self):
        self.grid_configs = [dict(zip(self.hyperparameters,comb)) for comb in itertools.product(*self.values)]

    def single_combination(self, params):
        model = RandomForestRegressor(**params)
        model.fit(self.train.x, self.train.y)
        return model.score(self.validation.x, self.train.y)

    def run_combinations():
        # Funtion to apply parallelism
        worker_fn = self.single_combination()
        # Run in parallel
        with Pool(processes=cpu_count()-1) as pool:
            results = pool.map(worker_fn, self.grid_configs)
        # Review the best configuration
        # Evaluate in the test set
        # Show and save the best model
        




def main():
    grid=gridSearch()
    grid.generate_combinations()
    for i in grid.grid_configs:
        print(i)
	


if __name__ == '__main__':
	main()