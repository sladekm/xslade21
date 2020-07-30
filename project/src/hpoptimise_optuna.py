# Autor: Matyáš Sládek
# Rok: 2020

# Tento soubor obsahuje třídu se sadou optimalizačních funkcí optimalizaci parametrů klasifikačních algoritmů

import os
import sys
sys.dont_write_bytecode = True
import warnings
import pickle
import joblib
import copy

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold

class DuplicateParamsPruner(optuna.pruners.BasePruner):
    '''
    Class used for pruning trials with parameters, that were already tested.
    '''
    def prune(self, study, trial):
        '''
        A function that checks if study already contains currently selected parameter values
        
        Returns True if study already contains currently selected parameter values, False otherwise
        '''
        trials = study.get_trials(deepcopy=False)
        completed_trials = [t.params for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        if len(completed_trials) == 0:
            return False

        if trial.params in completed_trials:
            return True

        return False

class HPoptimise(object):
    '''
    Class used to optimise classifiers using either GridSampler or TPESampler algorithms.
    '''
    def __init__(self, X_train, y_train, X_val, y_val, classifier_name, classifier, filepath, other_params):
        self._X_train = X_train   # Features for training
        self._y_train = y_train   # Labels for training
        self._X_val = X_val   # Features for validation
        self._y_val = y_val   # Labels for classification
        self._classifier_name = classifier_name   # Name of the classifier to choose correct parameters
        self._classifier = classifier   # The classifier object
        self._filepath = filepath   # Path to save the optimization prgress
        self._max_trials = other_params['max_trials']   # Maximum number of trials
        self._max_trials_no_change = other_params['max_trials_no_change']   # Maximum numbers of trials with no improvement
        self._use_cross_validation = other_params['use_cross_validation']   # Whether to use cross validation or validation set
        self._cross_validation_num_of_folds = other_params['cross_validation_num_of_folds']   # Num of folds for cross validation
        self._n_jobs = None   # Stores number of cpu cores to be used
        self._study = None   # Stores the study object containing optimisation progress
        self._params = {}   # Stores chosen params for each iteration
        self._default_params = {   # Default parameter values (not to be tuned)
            'LogisticRegression': {'max_iter':10000, 'class_weight':'balanced'},
            'KNeighborsClassifier': {'n_jobs':-1, 'algorithm':'brute'},
            'MLPClassifier': {'max_iter':10000, 'random_state':42},
            'DecisionTreeClassifier': {'class_weight':'balanced', 'random_state':42},
            'SVC_linear': {'kernel':'linear', 'class_weight':'balanced'},
            'SVC_rbf': {'kernel':'rbf', 'class_weight':'balanced'},
            'RandomForestClassifier': {'n_jobs':-1, 'class_weight':'balanced', 'random_state':42},
            'XGBClassifier': {'tree_method':'gpu_hist', 'n_jobs':1, 'random_state':42},
        }
        self._search_space = {   # Search spaces for grid search (parameters are described in optimalization functions)
            'LogisticRegression': {
                'penalty': ['l2', 'none'],
                'fit_intercept': [True, False],
                'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
            },
            'KNeighborsClassifier': {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            },
            'SVC_linear': {
                'C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
            },
            'SVC_rbf': {
                'C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
                'gamma': [100, 200, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
            }
        }
        
        # Set n_jobs parameter for cross-validation based on the classifier used and number of logical cpu cores
        num_cpus = 1 if os.cpu_count() is None else os.cpu_count()
        self._n_jobs = self._cross_validation_num_of_folds if self._classifier_name in ['LogisticRegression', 'DecisionTreeClassifier', 'SVC_linear', 'SVC_rbf'] else 1
        self._n_jobs = self._n_jobs if self._n_jobs <= num_cpus else num_cpus
        
    def best_params(self):
        '''
        A function used to get the best found set of parameters
        
        Returns:
        
        self._study.best_trial.user_attrs['params']: Optimised parameters
        '''
        return self._study.best_trial.user_attrs['params']

    def best_score(self):
        '''
        A function used to get the best achieved score
        
        Returns:
        
        self._study.best_value: The best achieved score
        '''
        return self._study.best_value
    
    def optimise(self):
        '''
        A function used to optimise classifiers using either GridSampler or TPESampler algorithms.
        If previously done study is found, optimisation continues from there
        '''
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        try:   # If previously done study is found, continue optimisation from there
            self._study = joblib.load(self._filepath)
        except FileNotFoundError:   # Else create new study with either GridSampler or TPESampler based on the selected classifier
            if self._classifier_name in ['LogisticRegression', 'KNeighborsClassifier', 'SVC_linear', 'SVC_rbf']:
                print('Grid search is used for classifier \033[1m{}\033[0m, early_stopping_rounds parameter ignored.'.format(self._classifier_name))
                self._study = optuna.create_study(sampler=optuna.samplers.GridSampler(self._search_space[self._classifier_name]), direction="maximize")
            else:
                self._study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42), direction="maximize", pruner=DuplicateParamsPruner())
            
        # Start the optimisation for selected classifier
        if self._classifier_name == 'LogisticRegression':
            self._study.optimize(self._objective_LogisticRegression, n_trials=2*2*7, callbacks=[self._print_score, self._save_study])
        elif self._classifier_name == 'KNeighborsClassifier':
            self._study.optimize(self._objective_KNeighborsClassifier, n_trials=9*2*2, callbacks=[self._print_score, self._save_study])
        elif self._classifier_name == 'MLPClassifier':
            with warnings.catch_warnings():   # Suppress warnings regarding database storage which is not used
                warnings.simplefilter('ignore', category=UserWarning)
                self._study.optimize(self._objective_MLPClassifier, n_trials=self._max_trials, callbacks=[self._print_score, self._check_early_stopping, self._save_study])
        elif self._classifier_name == 'DecisionTreeClassifier':
            self._study.optimize(self._objective_DecisionTreeClassifier, n_trials=self._max_trials, callbacks=[self._print_score, self._check_early_stopping, self._save_study])
        elif self._classifier_name == 'SVC_linear':
            self._study.optimize(self._objective_SVC_linear, n_trials=6, callbacks=[self._print_score, self._save_study])
        elif self._classifier_name == 'SVC_rbf':
            self._study.optimize(self._objective_SVC_rbf, n_trials=6*8, callbacks=[self._print_score, self._save_study])
        elif self._classifier_name == 'RandomForestClassifier':
            self._study.optimize(self._objective_RandomForestClassifier, n_trials=self._max_trials, callbacks=[self._print_score, self._check_early_stopping, self._save_study])
        elif self._classifier_name == 'XGBClassifier':
            self._study.optimize(self._objective_XGBClassifier, n_trials=self._max_trials, callbacks=[self._print_score, self._check_early_stopping, self._save_study])
        else:
            pass
    
    def _objective_LogisticRegression(self, trial):
        '''
        Objective function used to optimise LogisticRegression with GridSampler.
        
        Parameters:
        
        trial: The trial object for current iteration
        
        Returns:
        
        score: Classification score achieved in current iteration
        
        '''
        self._params = {}
                
        self._params['penalty'] = trial.suggest_categorical('penalty', ['l2', 'none'])   # Whether to use regularization
        self._params['fit_intercept'] = trial.suggest_categorical('fit_intercept', [True, False])   # Whether to fit dummy feature to be tuned

        if self._params['penalty'] != 'none':
            self._params['C'] = trial.suggest_categorical('C', [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]) # Regularization strength (inverse)
        
        score = self._classify()

        # Save the parameter values from current iteration to the trial object
        trial.set_user_attr('params', copy.deepcopy(self._params))
        
        return score
    
    def _objective_KNeighborsClassifier(self, trial):
        '''
        Objective function used to optimise KNeighborsClassifier with GridSampler.
        
        Parameters:
        
        trial: The trial object for current iteration
        
        Returns:
        
        score: Classification score achieved in current iteration
        
        '''
        self._params = {}
        
        self._params['n_neighbors'] = trial.suggest_categorical('n_neighbors', [3, 5, 7, 9, 11, 13, 15, 17, 19])   #Number of neighbors to be used
        self._params['weights'] = trial.suggest_categorical('weights', ['uniform', 'distance'])   # Whether to penalize neighbors based on their distance
        self._params['p'] = trial.suggest_categorical('p', [1, 2])   # Whether to use Manhattan or Euclidian distance
        
        score = self._classify()

        # Save the parameter values from current iteration to the trial object
        trial.set_user_attr('params', copy.deepcopy(self._params))
        
        return score
    
    def _objective_MLPClassifier(self, trial):
        '''
        Objective function used to optimise MLPClassifier with TPESampler.
        
        Parameters:
        
        trial: The trial object for current iteration
        
        Returns:
        
        score: Classification score achieved in current iteration
        
        '''
        self._params = {}
        
        self._params['solver'] = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])   # Type of solver to adjust weights and biases
        
        if self._params['solver'] == 'lbfgs':
            self._params['hidden_layer_sizes'] = trial.suggest_categorical('hidden_layer_sizes_lbfgs', [(100), (100, 50), (100, 50, 25)])   # Sizes of hidden layers
            self._params['activation'] = trial.suggest_categorical('activation_lbfgs', ['logistic', 'tanh', 'relu'])   # Shape of activation function
            self._params['alpha'] = trial.suggest_categorical('alpha_lbfgs', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])   # Regularization strength (L2)
        elif self._params['solver'] == 'sgd':
            self._params['learning_rate'] = 'adaptive'   # Decreases learning rate if two epochs fail to improve score (faster convergence)
            self._params['learning_rate_init'] = 0.1   # Initial learning rate
            self._params['hidden_layer_sizes'] = trial.suggest_categorical('hidden_layer_sizes_sgd', [(100), (100, 50), (100, 50, 25)])
            self._params['activation'] = trial.suggest_categorical('activation_sgd', ['logistic', 'tanh', 'relu'])
            self._params['alpha'] = trial.suggest_categorical('alpha_sgd', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
            self._params['momentum'] = trial.suggest_float('momentum', 0.5, 1, step=0.05)   # Attempt to speed up SGD with optimal momentum
        elif self._params['solver'] == 'adam':
            self._params['hidden_layer_sizes'] = trial.suggest_categorical('hidden_layer_sizes_adam', [(100), (100, 50), (100, 50, 25)])
            self._params['activation'] = trial.suggest_categorical('activation_adam', ['logistic', 'tanh', 'relu'])
            self._params['alpha'] = trial.suggest_categorical('alpha_adam', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
            self._params['epsilon'] = trial.suggest_categorical('epsilon', [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])   # Attemt to speed up ADAM classifier
            
        # Prune the trial if the same parameter values were already tested
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        score = self._classify()

        # Save the parameter values from current iteration to the trial object
        trial.set_user_attr('params', copy.deepcopy(self._params))
        
        return score
    
    def _objective_DecisionTreeClassifier(self, trial):
        '''
        Objective function used to optimise DecisionTreeClassifier with TPESampler.
        
        Parameters:
        
        trial: The trial object for current iteration
        
        Returns:
        
        score: Classification score achieved in current iteration
        
        '''
        self._params = {}
        
        self._params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy'])   # Wheter to use gini index or information gain to evaluate split
#         self._params['max_depth'] = trial.suggest_int('max_depth', 3, 100)   # max_depth did not prove to be a viable parameter to tune
        self._params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 200)   # Minimal number of samples in node to make split
        self._params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, self._params['min_samples_split'])   # Minimum number of leaves to be left in node
                                                                                                                        # after split
        self._params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])   # Maximum number of features to be considered for splitting
        self._params['ccp_alpha'] = trial.suggest_float('ccp_alpha', 0.0, 0.03, step=0.001)   # Tree pruning to avoid overfitting
        
        # Prune the trial if the same parameter values were already tested
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        score = self._classify()

        # Save the parameter values from current iteration to the trial object
        trial.set_user_attr('params', copy.deepcopy(self._params))
        
        return score
    
    def _objective_SVC_linear(self, trial):
        '''
        Objective function used to optimise SVC with linear kernel with GridSampler.
        
        Parameters:
        
        trial: The trial object for current iteration
        
        Returns:
        
        score: Classification score achieved in current iteration
        
        '''
        self._params = {}
        
        self._params['C'] = trial.suggest_categorical('C', [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])   # Regularization strength (inverse)
        
        score = self._classify()

        # Save the parameter values from current iteration to the trial object
        trial.set_user_attr('params', copy.deepcopy(self._params))
        
        return score
    
    def _objective_SVC_rbf(self, trial):
        '''
        Objective function used to optimise SVC with RBF kernel with GridSampler.
        
        Parameters:
        
        trial: The trial object for current iteration
        
        Returns:
        
        score: Classification score achieved in current iteration
        
        '''
        self._params = {}
        
        self._params['C'] = trial.suggest_categorical('C', [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])   # Regularization strength (inverse)
        self._params['gamma'] = trial.suggest_categorical('gamma', [100, 200, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])   # How much samples affect RBF transformation
        
        # GridSampler does not support string values, so they are represented by numbers and replaced here
        if self._params['gamma'] == 100:
            self._params['gamma'] = 'scale'
        elif self._params['gamma'] == 200:
            self._params['gamma'] = 'auto'
        
        score = self._classify()

        # Save the parameter values from current iteration to the trial object
        trial.set_user_attr('params', copy.deepcopy(self._params))
        
        return score
    
    def _objective_RandomForestClassifier(self, trial):
        '''
        Objective function used to optimise RandomForestClassifier with TPESampler.
        
        Parameters:
        
        trial: The trial object for current iteration
        
        Returns:
        
        score: Classification score achieved in current iteration
        
        '''
        self._params = {}
        
        self._params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy'])   # Wheter to use gini index or information gain to evaluate split
#         self._params['max_depth'] = trial.suggest_int('max_depth', 3, 100)   # max_depth did not prove to be a viable parameter to tune
        self._params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 200)   # Minimal number of samples in node to make split
        self._params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, self._params['min_samples_split'])   # Minimum number of leaves to be left in node
        self._params['ccp_alpha'] = trial.suggest_float('ccp_alpha', 0.0, 0.03, step=0.001)   # Tree pruning to avoid overfitting
        self._params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])   # Maximum number of features to be considered for splitting
        self._params['max_samples'] = trial.suggest_float('max_samples', 0.5, 0.99, step=0.01)   # Percentage of samples for bootstrapping
        
        # Prune the trial if the same parameter values were already tested
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        score = self._classify()

        # Save the parameter values from current iteration to the trial object
        trial.set_user_attr('params', copy.deepcopy(self._params))
        
        return score
    
    def _objective_XGBClassifier(self, trial):
        '''
        Objective function used to optimise RandomForestClassifier with TPESampler.
        
        Parameters:
        
        trial: The trial object for current iteration
        
        Returns:
        
        score: Classification score achieved in current iteration
        
        '''
        self._params = {}
        self._params['n_estimators'] = 10000   # Setting maximum number of estimator to be high because early stopping is used to determine the optimal value
        self._params['learning_rate'] = 0.1   # Setting learning rate to be relatively low, but not low enough to be extremely slow
            
        self._params['max_depth'] = trial.suggest_int('max_depth', 3, 15, step=1)   # Maximum depth of each base classifier
        self._params['min_child_weight'] = trial.suggest_int('min_child_weight', 0, 10, step=1)   # Minimum weight of child node to make a split
        self._params['subsample'] = trial.suggest_float('subsample', 0.5, 1, step=0.05)   # Percentage of samples to be used to train each tree
        self._params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1, step=0.05)   # Percentage of features to be used to train each tree
        self._params['reg_alpha'] = trial.suggest_float('reg_alpha', 0, 1, step=0.001)   # Regularizetion strength (L1)
        self._params['reg_lambda'] = trial.suggest_float('reg_lambda', 0, 10, step=0.001)   # Regularization strength (L2)
        self._params['gamma'] = trial.suggest_float('gamma', 0, 1, step=0.001)   # Regularizetion strength (XGBoosts pseudo regularization)

        # Prune the trial if the same parameter values were already tested
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Initialize the classifier with default and currently selected parameter values
        classifier = self._classifier(**self._default_params[self._classifier_name], **self._params)

        best_ntree_limit = 0   # Stores the optimum number of boosting iterations (n_estimators)
        scores = []   # Stores classification scores
        
        # Validate either using cross validation or validation set while tracking the optimum number of boosting iterations
        if self._use_cross_validation:            
            for train_index, val_index in StratifiedKFold(n_splits=self._cross_validation_num_of_folds).split(self._X_train, self._y_train):
                X_train_cv, X_val = self._X_train[train_index], self._X_train[val_index]
                y_train_cv, y_val = self._y_train[train_index], self._y_train[val_index]

                classifier.fit(X_train_cv, y_train_cv, early_stopping_rounds=100, eval_set=[(X_val, y_val)], verbose=False)
                scores.append(1 - classifier.get_booster().best_score)

                if best_ntree_limit < classifier.get_booster().best_ntree_limit:
                    best_ntree_limit = classifier.get_booster().best_ntree_limit
        else:        
            classifier.fit(self._X_train, self._y_train, early_stopping_rounds=100, eval_set=[(self._X_val, self._y_val)], verbose=False)
            scores.append(1 - classifier.get_booster().best_score)
            best_ntree_limit = classifier.get_booster().best_ntree_limit
                
        # Set n_estimators to the optimum value found with early stopping
        self._params['n_estimators'] = best_ntree_limit
        
        # Save the parameter values from current iteration to the trial object
        trial.set_user_attr('params', copy.deepcopy(self._params))

        return np.mean(scores)
    
    def _classify(self):
        '''
        A function that performs classification either using cross-validation or validation set
        
        Returns:
        
        score: Classification score
        '''
        
        # Initialize the classifier with default and currently selected parameter values
        classifier = self._classifier(**self._default_params[self._classifier_name], **self._params)

        # Validate either using cross validation or validation set
        if self._use_cross_validation:
            score = cross_val_score(classifier, self._X_train, self._y_train, cv=StratifiedKFold(n_splits=self._cross_validation_num_of_folds), n_jobs=self._n_jobs).mean()
        else:
            classifier.fit(self._X_train, self._y_train)
            score = classifier.score(self._X_val, self._y_val)
            
        return score
    
    def _print_score(self, study, trial):
        '''
        A callback function used to print out current optimisation progress
        
        Parameters:
        
        study: Current study object
        trial: Current trial object
        '''
        print('Current trial number: \033[1m{}\033[0m. Best trial number: \033[1m{}\033[0m with score: \033[1m{}\033[0m                           \r'.format(trial.number, study.best_trial.number, study.best_trial.value), end='')
        
    def _check_early_stopping(self, study, trial):
        '''
        A callback function used to check if optimisation should stop (when selected number of trials with no improvement has been reached)
        
        Parameters:
        
        study: Current study object
        trial: Current trial object
        '''
        if trial.number - study.best_trial.number >= self._max_trials_no_change:
            study.stop()           
            
    def _save_study(self, study, trial):
        '''
        A callback function used to save the study object with current progress
        
        Parameters:
        
        study: Current study object
        trial: Current trial object
        '''
        try:
            joblib.dump(study, self._filepath)
        except Exception as e:
            print('Failed to save file: "{}"!'.format(self._filepath), file=sys.stderr)
            print('Error: {}'.format(repr(e)), file=sys.stderr)
        
        
        
            
            
