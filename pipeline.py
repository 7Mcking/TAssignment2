import logging
import pandas as pd
from zenml.pipelines import pipeline
from zenml.steps import step, Output
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
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
from preprocessing import FeatureEngineer, LogTransformation, MinMaxScaling, MissingValueHandler, FillMissingValuesStrategy
import smogn
import mlflow

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with your MLflow tracking server URI

# Define your steps
@step
def load_data() -> Output(data=pd.DataFrame):
    data_loader = DataLoader()
    data = data_loader.get_ingestor(".csv").ingest("C:\\Documents\\Code\\TVaritAssignment2\\DSData_Assignments 1.csv")
    return data

@step
def preprocess_data(data: pd.DataFrame) -> Output(data_scaled=pd.DataFrame):
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

@step
def split_data(data_scaled: pd.DataFrame) -> Output(X_train=pd.DataFrame, X_test=pd.DataFrame, y_train=pd.Series, y_test=pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(data_scaled.drop(columns=['DS_S']), data_scaled['DS_S'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@enable_mlflow
@step
def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_strategy: ModelBuilder) -> Output(model=Any):
    mlflow.set_experiment("train_model_experiment")
    model = model_strategy.build_model(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")
    return model

@enable_mlflow
@step
def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Output(metrics=dict):
    mlflow.set_experiment("evaluate_model_experiment")
    evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
    metrics = evaluator.evaluate(model, X_test, y_test)
    evaluator.plot_predictions()
    evaluator.plot_residuals()
    mlflow.log_metrics(metrics)
    return metrics

# Define your pipelines
@pipeline
def linear_regression_pipeline(load_data, preprocess_data, split_data, train_model, evaluate_model):
    data = load_data()
    data_scaled = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data_scaled)
    model = train_model(X_train, y_train, model_strategy=ModelBuilder(LinearRegressionStrategy()))
    evaluate_model(model, X_test, y_test)

@pipeline
def random_forest_pipeline(load_data, preprocess_data, split_data, train_model, evaluate_model):
    data = load_data()
    data_scaled = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data_scaled)
    model = train_model(X_train, y_train, model_strategy=ModelBuilder(RandomForestRegressionStrategy()))
    evaluate_model(model, X_test, y_test)

@pipeline
def xgboost_pipeline(load_data, preprocess_data, split_data, train_model, evaluate_model):
    data = load_data()
    data_scaled = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data_scaled)
    model = train_model(X_train, y_train, model_strategy=ModelBuilder(XGBRegressionStrategy()))
    evaluate_model(model, X_test, y_test)

@pipeline
def catboost_pipeline(load_data, preprocess_data, split_data, train_model, evaluate_model):
    data = load_data()
    data_scaled = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data_scaled)
    model = train_model(X_train, y_train, model_strategy=ModelBuilder(CatBoostRegressionStrategy()))
    evaluate_model(model, X_test, y_test)

@pipeline
def lgbm_pipeline(load_data, preprocess_data, split_data, train_model, evaluate_model):
    data = load_data()
    data_scaled = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data_scaled)
    model = train_model(X_train, y_train, model_strategy=ModelBuilder(LGBMRegressionStrategy()))
    evaluate_model(model, X_test, y_test)

# Run the pipelines
if __name__ == "__main__":
    linear_regression_pipeline.run()
    random_forest_pipeline.run()
    xgboost_pipeline.run()
    catboost_pipeline.run()
    lgbm_pipeline.run()