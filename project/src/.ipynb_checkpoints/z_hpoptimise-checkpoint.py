import os
import sys
sys.dont_write_bytecode = True
import pickle
import joblib
import copy

import numpy as np

import optuna

from sklearn.metrics import r2_score

class DuplicateParamsPruner(optuna.pruners.BasePruner):
    def prune(self, study, trial):
        trials = study.get_trials(deepcopy=False)
        completed_trials = [t.params for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        if len(completed_trials) == 0:
            return False

        if trial.params in completed_trials:
            return True

        return False

class HPoptimise(object):

    def __init__(self, X_train, y_train, X_val, y_val, Regressor, filepath, max_trials, max_trials_no_change):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.Regressor = Regressor
        self.filepath = filepath
        self.max_trials = max_trials
        self.max_trials_no_change = max_trials_no_change
        self.study = None
        self.params = {}
        self.default_params = {
            'RandomForestRegressor': {'n_jobs':-1},
        }
    
    def objective_RandomForestRegressor(self, trial):
        self.params['n_estimators'] = 100
        self.params['max_depth'] = None
        
        self.params['criterion'] = trial.suggest_categorical('criterion', ['mse', 'mae'])
        self.params['min_samples_split'] = trial.suggest_float('min_samples_split', 0.01, 0.3, step=0.01)
        self.params['min_samples_leaf'] = trial.suggest_float('min_samples_leaf', 0.01, 0.3, step=0.01)
        self.params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        self.params['max_samples'] = trial.suggest_float('max_samples', 0.5, 0.95, step=0.05)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        Regressor = self.Regressor(**self.default_params[self.Regressor.__name__], **self.params)

        Regressor.fit(self.X_train, self.y_train)
        y_pred = Regressor.predict(self.X_val)
        
        score = r2_score(self.y_val, y_pred)

        trial.set_user_attr('params', copy.deepcopy(self.params))
        
        return score
                    
    def optimise(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        try:
            self.study = joblib.load(self.filepath)
        except FileNotFoundError:
            self.study = optuna.create_study(direction="maximize", pruner=DuplicateParamsPruner())
        
        if self.Regressor.__name__ == 'RandomForestRegressor':
            self.study.optimize(self.objective_RandomForestRegressor, n_trials=self.max_trials, callbacks=[self.print_score, self.check_early_stopping, self.save_study])
        else:
            pass

    def best_params(self):
        return self.study.best_trial.user_attrs['params']

    def best_score(self):
        return self.study.best_value
    
    def print_score(self, study, trial):
        print('Trial number: \033[1m{}\033[0m Best validation score: \033[1m{}\033[0m                           \r'.format(trial.number, study.best_trial.value), end='')
        
    def check_early_stopping(self, study, trial):
        if trial.number - study.best_trial.number == self.max_trials_no_change:
            study.stop()
            
    def save_study(self, study, trial):
        try:
            joblib.dump(study, self.filepath)
        except Exception as e:
            print('Failed to save file: "{}"!'.format(self.filepath), file=sys.stderr)
            print('Error: {}'.format(repr(e)), file=sys.stderr)
        
        
        
            
            
