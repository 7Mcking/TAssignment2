# README.md

## Project Overview

This project is a machine learning pipeline for regression tasks, utilizing various regression models and logging results with MLflow. The pipeline is orchestrated using Apache Airflow.

## Repository Structure

```
TVaritAssignment2/
├── .gitignore
├── LICENSE
├── README.md
├── airflowPipeline.py
├── requirements.txt
├── src/
│   ├── build_model.py
│   ├── preprocessing.py
│   ├── read_data.py
│   └── test_model.py
|   ├── data_eda.py
|   └── feature_engineering.py
└── DSData_Assignments 1.csv
└── main.ipynb
└── logs.log
└── Assignment 2.pdf
```

### File Descriptions

- **.gitignore**: Specifies files and directories to be ignored by Git, such as virtual environments, compiled Python files, and MLflow runs.

- **LICENSE**: Contains the MIT License under which this project is distributed.

- **README.md**: Provides an overview of the project, repository structure, and descriptions of each file.

- **airflowPipeline.py**: Defines the Airflow DAG for the regression pipeline. It includes tasks for loading data, preprocessing, splitting data, training models, and evaluating models. The results are logged using MLflow.

- **requirements.txt**: Lists the Python dependencies required for the project, including libraries for machine learning, data processing, and Airflow.

- **src/**: Directory containing the source code for various components of the project.
    - **build_model.py**: Contains classes and methods for building different regression models, including Linear Regression, Random Forest, XGBoost, CatBoost, and LightGBM.
    - **preprocessing.py**: Includes classes and methods for data preprocessing, such as handling missing values, outlier detection, log transformation, and scaling.
    - **read_data.py**: Defines classes for reading data from different file formats, including CSV and ZIP files.
    - **test_model.py**: Contains classes and methods for evaluating regression models, including metrics calculation and plotting.

- **DSData_Assignments 1.csv**: The dataset used for the regression tasks in the pipeline.
- **main.ipynb**: Jupyter notebook containing the code for the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Apache Airflow
- MLflow

### Installation

1. Clone the repository:
     ```sh
     git clone https://github.com/yourusername/TVaritAssignment2.git
     cd TVaritAssignment2
     ```

2. Install the required Python packages:
     ```sh
     pip install -r requirements.txt
     ```

### Running the Pipeline

1. Start the Airflow web server and scheduler:
     ```sh
     airflow webserver --port 8080
     airflow scheduler
     ```

2. Access the Airflow web UI at `http://localhost:8080` and trigger the `regression_pipeline` DAG.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
