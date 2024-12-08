import logging
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.decorators import dag, task
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from build_model import (
    LinearRegressionStrategy,
    RandomForestRegressionStrategy,
    XGBRegressionStrategy,
    CatBoostRegressionStrategy,
    LGBMRegressionStrategy,
    ModelBuilder,
)
from test_model import ModelEvaluator, RegressionModelEvaluationStrategy
from read_data import DataLoader
from preprocessing import FeatureEngineer, IQROutlierDetection, LogTransformation, MinMaxScaling, MissingValueHandler, FillMissingValuesStrategy, OutlierDetector
import mlflow
import mlflow.sklearn

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with your MLflow tracking server URI

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG using the @dag decorator
@dag(
    default_args=default_args,
    description='A regression pipeline with MLflow logging',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)
def regression_pipeline():
    # Define your tasks using the @task decorator
    @task
    def load_data():
        data_loader = DataLoader()
        data = data_loader.get_ingestor(".csv").ingest("C:\\Documents\\Code\\TVaritAssignment2\\DSData_Assignments 1.csv")
        return data

    @task
    def preprocess_data(data: pd.DataFrame):
        # Fill missing values
        missing_value_handler = MissingValueHandler(FillMissingValuesStrategy(method='mean'))
        data['HM_TEMP'] = missing_value_handler.handle_missing_values(data[['HM_TEMP']])['HM_TEMP']
        
        # Handle outliers
        outlier_handler = OutlierDetector(IQROutlierDetection())
        data_scaled = data.dropna()
        data_scaled = data_scaled.reset_index(drop=True)
        
        # Log transformation
        columns_to_log_transform = ['CAC2', 'MG']
        feature_engineering = FeatureEngineer(LogTransformation(features=columns_to_log_transform))
        data_scaled = feature_engineering.apply_feature_engineering(data_scaled)
        
        # Min-Max scaling
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_engineering.set_strategy(MinMaxScaling(features=numerical_columns[:-1]))
        data_scaled = feature_engineering.apply_feature_engineering(data_scaled)
        
        return data_scaled

    @task
    def split_data(data_scaled: pd.DataFrame):
        X_train, X_test, y_train, y_test = train_test_split(data_scaled.drop(columns=['DS_S']), data_scaled['DS_S'], test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    @task
    def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_strategy: ModelBuilder, experiment_name: str):
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            model = model_strategy.build_model(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")
            return model

    @task
    def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, experiment_name: str):
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
            metrics = evaluator.evaluate(model, X_test, y_test)
            evaluator.plot_predictions()
            evaluator.plot_residuals()
            mlflow.log_metrics(metrics)

    # Define the task dependencies
    data = load_data()
    data_scaled = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data_scaled)
    
    # Linear Regression
    model = train_model(X_train, y_train, model_strategy=ModelBuilder(LinearRegressionStrategy()), experiment_name='linear_regression_experiment')
    evaluate_model(model, X_test, y_test, experiment_name='linear_regression_experiment')

    # Random Forest
    model_rf = train_model(X_train, y_train, model_strategy=ModelBuilder(RandomForestRegressionStrategy()), experiment_name='random_forest_experiment')
    evaluate_model(model_rf, X_test, y_test, experiment_name='random_forest_experiment')

    # XGBoost
    model_xgb = train_model(X_train, y_train, model_strategy=ModelBuilder(XGBRegressionStrategy()), experiment_name='xgboost_experiment')
    evaluate_model(model_xgb, X_test, y_test, experiment_name='xgboost_experiment')

    # CatBoost
    model_catboost = train_model(X_train, y_train, model_strategy=ModelBuilder(CatBoostRegressionStrategy()), experiment_name='catboost_experiment')
    evaluate_model(model_catboost, X_test, y_test, experiment_name='catboost_experiment')

    # LGBM
    model_lgbm = train_model(X_train, y_train, model_strategy=ModelBuilder(LGBMRegressionStrategy()), experiment_name='lgbm_experiment')
    evaluate_model(model_lgbm, X_test, y_test, experiment_name='lgbm_experiment')

# Instantiate the DAG
regression_pipeline_dag = regression_pipeline()