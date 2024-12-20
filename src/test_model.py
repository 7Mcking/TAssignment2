import logging
from abc import ABC, abstractmethod

from matplotlib import lines, markers
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Abstract method to evaluate a model.

        Parameters:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        pass


# Concrete Strategy for Regression Model Evaluation
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def __init__(self):
        """
        Initializes the RegressionModelEvaluationStrategy.
        """
        self.metrics = None
        self.mse = None
        self.r2 = None
        self.y_pred = None
        self.y_test = None
        self.model = None
        logging.info("Regression Model Evaluation Strategy initialized.")
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """
        Evaluates a regression model using R-squared and Mean Squared Error.

        Parameters:
        model (RegressorMixin): The trained regression model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing R-squared and Mean Squared Error.
        """
        self.y_test = y_test
        self.model = model
        logging.info("Predicting using the trained model.")
        self.y_pred =self.model.predict(X_test)

        logging.info("Calculating evaluation metrics.")
        self.mse = mean_squared_error(y_test,self.y_pred)
        self.r2 = r2_score(self.y_test, self.y_pred)

        self.metrics = {"Mean Squared Error": self.mse, "R-Squared": self.r2}

        logging.info(f"Model Evaluation Metrics: {self.metrics}")
        return self.metrics
    
    def plot_predictions(self):
        """
        Plots the actual vs predicted values for the regression model.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, color='blue', label='Predictions')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 
                 color='red', lw=2, linestyle='--', label='Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.legend()
        plt.show()
    
    def get_predictions(self):
        return self.y_pred
    
    def get_actual(self):
        return self.y_test
    
    def plot_residuals(self):
        """
        Plots the residuals for the regression model.
        """
        import matplotlib.pyplot as plt
        residuals = self.y_test - self.y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, residuals, color='blue')
        plt.xlabel('Actual')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()
        
    def plot_prediction_curve(self, X_test: pd.DataFrame):
        """
        Plots the prediction curve based on the model's coefficients.
        """
        if hasattr(self.model.named_steps['model'], 'coef_'):
            import matplotlib.pyplot as plt
            coef = self.model.named_steps['model'].coef_
            intercept = self.model.named_steps['model'].intercept_
            X_test = X_test[X_test.columns[self.model.named_steps['rfe'].get_support()]]
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test.iloc[:, 0], self.y_test, color='blue', label='Actual')
            plt.plot(X_test.iloc[:, 0], X_test @ coef + intercept, color='green', 
                     linestyle='None', marker='x',
                     label='Prediction Curve')
            plt.xlabel('Actual')
            plt.ylabel('Target')
            plt.title('Prediction Curve')
            plt.legend()
            plt.show()
        else:
            logging.warning("The model does not have coefficients to plot the prediction curve.")


# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy.

        Parameters:
        strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
        strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)
    
    def plot_predictions(self):
        self._strategy.plot_predictions()
        
    def plot_residuals(self):
        self._strategy.plot_residuals()
        
    def get_predictions(self):
        return self._strategy.get_predictions()
    
    def get_actual(self):
        return self._strategy.get_actual()
    
    def plot_prediction_curve(self, X_test: pd.DataFrame):
        self._strategy.plot_prediction_curve(X_test)
    
    


# Example usage
if __name__ == "__main__":
    # Example trained model and data (replace with actual trained model and data)
    # model = trained_sklearn_model
    # X_test = test_data_features
    # y_test = test_data_target

    # Initialize model evaluator with a specific strategy
    # model_evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
    # evaluation_metrics = model_evaluator.evaluate(model, X_test, y_test)
    # print(evaluation_metrics)

    pass