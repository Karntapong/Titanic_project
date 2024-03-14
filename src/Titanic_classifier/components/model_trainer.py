import os
import optuna
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from optuna.integration import SklearnPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna import Trial
from typing import Any, Dict, Union
from Titanic_classifier.entity import ModelTrainerConfig
from joblib import dump

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.study = None

    def train(self):
        self.X_train = pd.read_csv(self.config.x_train_path)
        self.X_test = pd.read_csv(self.config.x_test_path)
        self.y_train = pd.read_csv(self.config.y_train_path)
        self.y_test = pd.read_csv(self.config.y_test_path)
        print("X_train shape:", self.X_train.shape)
        print("y_train shape:", self.y_train.shape)
        print("X_test shape:", self.X_test.shape)
        print("y_test shape:", self.y_test.shape)
        if self.study is None:
            self.study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=5), sampler=TPESampler())

        self.study.optimize(self.objective, n_trials=100)

        best_params = self.study.best_params
        print('!!!!!!!!!!!!!!!!best:',best_params)
        model_type = best_params['model']
        best_params.pop('model', None)
        if model_type == 'XGBoost':
            model = XGBClassifier(**best_params)
        elif model_type == 'LightGBM':
            model = LGBMClassifier(**best_params)
        elif model_type == 'LogisticRegression':
            model = LogisticRegression(**best_params)
        elif model_type == 'RandomForest':
            model = RandomForestClassifier(**best_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(self.X_train, self.y_train)

        # Get predictions from the base models
        base_model_preds_train = {
            "XGBoost": model.predict(self.X_train),
            "LightGBM": model.predict(self.X_train),
            "LogisticRegression": model.predict(self.X_train),
            "RandomForest": model.predict(self.X_train)
        }
        

        base_model_preds_test = {
            "XGBoost": model.predict(self.X_test),
            "LightGBM": model.predict(self.X_test),
            "LogisticRegression": model.predict(self.X_test),
            "RandomForest": model.predict(self.X_test)
        }

        # Get stack features
        X_train_stacked = pd.concat([self.X_train, pd.DataFrame(base_model_preds_train)], axis=1)
        X_test_stacked = pd.concat([self.X_test, pd.DataFrame(base_model_preds_test)], axis=1)

        stacking_model = XGBClassifier()  #Can replace with prefer model
        stacking_model.fit(X_train_stacked, self.y_train)

        stacking_accuracy = accuracy_score(self.y_test, stacking_model.predict(X_test_stacked))
        print(f"Stacking Model Accuracy: {stacking_accuracy}")
        model_accuracy = accuracy_score(self.y_test, model.predict(self.X_test))
        print(f" Model Accuracy: {model_accuracy}")

        dump(model, os.path.join(self.config.root_dir, "best_model.joblib"))
        dump(stacking_model, os.path.join(self.config.root_dir, "stacking_model.joblib"))

        return X_train_stacked,X_test_stacked,model_accuracy,stacking_accuracy

    def objective(self, trial: Trial) -> float:
        model_type = trial.suggest_categorical("model", ["XGBoost", "LightGBM", "LogisticRegression", "RandomForest"])
        params = self.get_model_params(trial)
        print("!!!!!!!!!!!!!!!!!!!!!!Parameters for", model_type, ":", params)
        if model_type == 'XGBoost':
            model = XGBClassifier(**params)
        elif model_type == 'LightGBM':
            model = LGBMClassifier(**params)
        elif model_type == 'LogisticRegression':
            params.pop('model', None)  # Remove 'model' key
            model = LogisticRegression(**params)
        elif model_type == 'RandomForest':
            params.pop('model', None)  # Remove 'model' key
            model = RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(self.X_train, self.y_train)

        accuracy = accuracy_score(self.y_test, model.predict(self.X_test))

        return accuracy





    def get_model_params(self, trial: Trial) -> Dict[str, Any]:
        if trial.params['model'] == 'XGBoost':
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.5),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        elif trial.params['model'] == 'LightGBM':
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.5),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        elif trial.params['model'] == 'LogisticRegression':
            params = {
                "C": trial.suggest_loguniform("C", 0.01, 100),
                "penalty": 'l1', 
                "solver": 'liblinear' 
            }
        elif trial.params['model'] == 'RandomForest':
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
            }
        else:
            raise ValueError(f"Unknown model type: {trial.params['model']}")

        return params