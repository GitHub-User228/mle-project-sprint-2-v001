# MLE MLflow Project
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![S3](https://img.shields.io/badge/S3-003366?style=for-the-badge)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-219ebc?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-yellow?style=for-the-badge)
![Pydantic](https://img.shields.io/badge/Pydantic-CC0066?style=for-the-badge)


## Description

This project covers the buisiness problem of improving the key metrics of the model for predicting the value of Yandex Real Estate flats.
The goal is to make the training process and other related processes easily repeatable and improve key model metrics that impact the company's business metrics, particularly the increase in successful transactions. MLflow framework is considered in order to run a large number of experiments and ensure reproducibility.

In order to achieve this goal, the following steps were taken:
- Deploying MLflow Tracking Server and MLflow Model Registry
- Logging the baseline model with it's metrics and parameters using MLflow
- EDA
- Feature Engineering (TODO)
- Feature Selection (TODO)
- Hyperparameter Tuning (TODO)

## Project Structure

**[requirements.txt](requirements.txt)**: This file contains the list of Python packages required for the project.

**[mlflow_server](mlflow_server)**: This directory contains shell scripts to start the MLflow server:
- [run_mlflow_server.sh](mlflow_server/run_mlflow_server.sh): Starts MlFlow server

**[config](config)**: This directory contains configuration files:
- [config.yaml](config/config.yaml): Configuration for the project
- [logger_config.yaml](config/logger_config.yaml): Configuration for the logger

**[scripts](scripts)**: This directory contains Python scripts:
- [init.py](scripts/__init__.py): Initialization of a logger
- [env.py](scripts/env.py): Initialization of pydantic settings with environment variables
- [utils.py](scripts/utils.py): Utility functions
- [utils_mlflow.py](scripts/utils_mlflow.py): Utility functions for interacting with MLflow
- [metrics.py](scripts/metrics.py): Definition of custom metrics
- [evaluate.py](scripts/evaluate.py): Function for metrics calculation or evaluation
- [plotters.py](scripts/plotters.py): Functions for plotting

**[notebooks](notebooks)**: This directory contains Jupyter notebooks which cover the following topics:
- [1_register_baseline.ipynb](notebooks/1_register_baseline.ipynb): Registering the baseline model
- [2_model_improvement.ipynb](notebooks/2_model_improvement.ipynb): EDA stage and further stages


## Project Stages

### Stage I. Deploying MLflow Tracking Server and MLflow Model Registry. Registering the existing baseline model

1. Clone the repository and cd to it:
```bash
git clone https://github.com/GitHub-User228/mle-project-sprint-2-v001
cd mle-project-sprint-2-v001
```
2. Create a virtual environment and activate it:
```bash 
python3.10 -m venv .venv
source .venv/bin/activate
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```
4. Configure the environment variables by editing the template file [env_template](env_template). Then rename the file to `.env`
5. Export the environment variables:
```bash
export $(cat .env | xargs)
```
6. Start the MLflow server:
```bash
sh ./mlflow_server/run_mlflow_server.sh
```

As a result, the MLflow server will be started and the MLflow UI will be available at http://127.0.0.1:5000.

In order to register the baseline model, follow the [1_register_baseline.ipynb](notebooks/1_register_baseline.ipynb) notebook. It is required for the baseline model to be already saved in the project directory.

---

### Stage II. EDA

EDA stage is covered in detail in the [2_model_improvement.ipynb](notebooks/2_model_improvement.ipynb) notebook.

The following steps were taken:
- A brief description of the data
- Individual and paired feature analysis
- Individual target feature analysis
- Paired analysis of the target feature with other features
- Correlation analysis
- Logging artifacts via MLflow

Here are the main results of the analysis:  
1. The following features have a high correlation with the target feature:
- total_area, ceiling_height, living_area, kitchen_area, rooms
2. The following features have a very low correlation with the target feature:
- is_duplicated, is_apartment
3. An average flat can be described as follows:
- located within Moscow
- located in a building built in the 70s-80s with a typical characteristics of the residential buildings built in that period
- ceiling height is about 2.6-2.7m
- number of floors is about 14-17
- total_area is about 52-57 $m^2%$ with half of the total area being the living area
- price is about 15 million rubles
4. The distributon of the target feature is close to be normal, which is a good sign.  
5. The distribution of all features is similar if to compare train and test subsets  
6. The price is higher, if:
- the area is larger
- the ceiling is higher
- the number of rooms is larger
- the flat is closer to the center of Moscow
- the building is too old or built in the near past
- the building type is 3
- it is an apartment
7. A flat is rare if:
- it is an apartment
- the building type is 0 or 3
- the number of rooms is larger than 4
- the building is too old or too high
8. There is a significant amount of flats with 0 kitchen and living areas.  
These properties are probably used for commercial purposes.
9. It is a good idea to create the following features:
- the distance from the center of Moscow to the flat
- the distance from the nearest metro station to the flat
- a feature which would indicate whether the value is 0 for both living_area and kitchen_area

---

Bucket: s3-student-mle-20240730-73c4e0c760