import logging
from abc import ABC, abstractmethod
from typing import Any

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from numpy import mean, std
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Abstract method to build and train a model.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        pass


# Concrete Strategy for Linear Regression using scikit-learn
class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid=None, *args, **kwargs) -> Pipeline:
        """
        Builds and trains a linear regression model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Linear Regression model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Linear Regression model with scaling.")

        rfe = RFE(estimator=LinearRegression(), n_features_to_select=12)
        
        # Creating a pipeline with standard scaling and linear regression
        pipeline = Pipeline(
            [
                ("rfe", rfe),
                ("model", LinearRegression()),  # Linear regression model
            ]
        )

        logging.info("Training Linear Regression model.")
        pipeline.fit(X_train, y_train)  # Fit the pipeline with training data
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(pipeline, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

        logging.info("Model training completed.")
        return pipeline
    
class RandomForestRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid=None, *args, **kwargs) -> Pipeline:
        """
        Builds and trains a random forest regression model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Random Forest Regression model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Random Forest Regression model with scaling.")
        
        if param_grid is None:
            rfe= RFE(estimator=RandomForestRegressor(), n_features_to_select=12)
            # Creating a pipeline with standard scaling and Random Forest Regression
            pipeline = Pipeline(
                [
                    ("rfe", rfe),
                    ("model", RandomForestRegressor()),  # Random Forest Regression model
                ]
            )

            logging.info("Training Random Forest Regression model.")
            pipeline.fit(X_train, y_train)  # Fit the pipeline with training data
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(pipeline, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
            print('neg_mean_absolute_error: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

            logging.info("Model training completed.")
            return pipeline
        else:
            param_grid = param_grid
            rfe= RFE(estimator=RandomForestRegressor(), n_features_to_select=12)
            # Creating a pipeline with standard scaling and Random Forest Regression
            pipeline = Pipeline(
                [
                    ("rfe", rfe),
                    ("model", RandomForestRegressor()),  # Random Forest Regression model
                ]
            )
            logging.info("Training Random Forest Regression model.")
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            logging.info("Model training completed.")
            return best_pipeline

class CatBoostRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid=None, *args, **kwargs) -> Pipeline:
        """
        Builds and trains a CatBoost regression model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained CatBoost regression model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing CatBoost Regression model.")
        
        rfe = RFE(estimator=CatBoostRegressor(), n_features_to_select=12)
        # Define the parameter grid for CatBoost
        if param_grid is None:
            pipeline = Pipeline(
                [
                    ("rfe", rfe),
                    ("model", CatBoostRegressor(*args, **kwargs)),  # CatBoost regression model
                ]
            )
            logging.info("Training CatBoost Regression model.")
            pipeline.fit(X_train, y_train)
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(pipeline, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
            print('neg_mean_absolute_error: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
            logging.info("Model training completed.")
            return pipeline
        else:
            param_grid = param_grid

        
        # Creating a pipeline with CatBoost regression model
        pipeline = Pipeline(
            [   
                ("rfe", rfe),
                ("model", CatBoostRegressor()),  # CatBoost regression model
            ]
        )
        
        logging.info("Training CatBoost Regression model.")
        
        # Initialize GridSearchCV with the pipeline and parameter grid
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        # Get the best estimator from the grid search
        best_pipeline = grid_search.best_estimator_
        
        logging.info("Model training completed.")
        return best_pipeline

class LGBMRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid=None, *args, **kwargs) -> Pipeline:
        """
        Builds and trains a LightGBM regression model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained LightGBM regression model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing LightGBM Regression model.")

        rfe = RFE(estimator=LGBMRegressor(), n_features_to_select=12)
        
        if param_grid is None:
            # Creating a pipeline with LightGBM regression model
            pipeline = Pipeline(
                [   
                    ("rfe", rfe),
                    ("model", LGBMRegressor()),  # LightGBM regression model
                ]
            )
            
            logging.info("Training LightGBM Regression model.")
            pipeline.fit(X_train, y_train)  # Fit the pipeline with training data
            
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(pipeline, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
            print('neg_mean_absolute_error: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
            
            logging.info("Model training completed.")
            return pipeline
        else:
            # Creating a pipeline with LightGBM regression model
            pipeline = Pipeline(
                [   
                    ("rfe", rfe),
                    ("model", LGBMRegressor()),  # LightGBM regression model
                ]
            )
            
            logging.info("Training LightGBM Regression model.")
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            logging.info("Model training completed.")
            return best_pipeline
    
class XGBRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid=None, *args, **kwargs) -> Pipeline:
        """
        Builds and trains a XGBoost regression model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained XGBoost regression model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing XGBoost Regression model.")

        rfe = RFE(estimator=XGBRegressor(), n_features_to_select=10)
        
        if param_grid is None:
            # Creating a pipeline with XGBoost regression model
            pipeline = Pipeline(
                [   
                    ("rfe", rfe),
                    ("model", XGBRegressor()),  # XGBoost regression model
                ]
            )

            logging.info("Training XGBoost Regression model.")
            pipeline.fit(X_train, y_train)  # Fit the pipeline with training data
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(pipeline, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
            print('neg_mean_absolute_error: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
            
            logging.info("Model training completed.")
            return pipeline
        else:
            # Creating a pipeline with XGBoost regression model
            pipeline = Pipeline(
                [   
                    ("rfe", rfe),
                    ("model", XGBRegressor()),  # XGBoost regression model
                ]
            )
            
            logging.info("Training XGBoost Regression model.")
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            logging.info("Model training completed.")
            return best_pipeline
    
class NNRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid=None, *args, **kwargs) -> Pipeline:
        """
        Builds and trains a Neural Network regression model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Neural Network regression model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Neural Network Regression model.")

        rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=12)
        
        if param_grid is None:
        
            # Creating a pipeline with Neural Network regression model
            pipeline = Pipeline(
                [   
                    ("rfe", rfe),
                    ("model", MLPRegressor()),  # Neural Network regression model
                ]
            )

            logging.info("Training Neural Network Regression model.")
            pipeline.fit(X_train, y_train)  # Fit the pipeline with training data
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(pipeline, X_train, y_train, scoring='mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
            print('neg_mean_absolute_error: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

            logging.info("Model training completed.")
            return pipeline
        
        else:
            param_grid = param_grid
            # Creating a pipeline with Neural Network regression model
            pipeline = Pipeline(
                [   
                    ("rfe", rfe),
                    ("model", MLPRegressor()),  # Neural Network regression model
                ]
            )

            logging.info("Training Neural Network Regression model.")
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            logging.info("Model training completed.")
            return best_pipeline


# Context Class for Model Building
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initializes the ModelBuilder with a specific model building strategy.

        Parameters:
        strategy (ModelBuildingStrategy): The strategy to be used for model building.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Sets a new strategy for the ModelBuilder.

        Parameters:
        strategy (ModelBuildingStrategy): The new strategy to be used for model building.
        """
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid=None, *args, **kwargs) -> RegressorMixin:
        """
        Executes the model building and training using the current strategy.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train, param_grid=param_grid, *args, **kwargs)