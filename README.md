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
- Feature Engineering
- Feature Selection
- Hyperparameter Tuning

All experiments were done under the `experiment_name` = `flat_price`  
ML models were logged under the `model_name` = `flat_price_model`  
The transforming pipeline obtained after feature engineering was logged under the `model_name` = `flat_price_feat_eng_pipeline`  

## Project Structure

**[requirements.txt](requirements.txt)**: This file contains the list of Python packages required for the project.

**[mlflow_server](mlflow_server)**: This directory contains shell scripts to start the MLflow server:
- [run_mlflow_server.sh](mlflow_server/run_mlflow_server.sh): Starts MlFlow server

**[config](config)**: This directory contains configuration files:
- [config.yaml](config/config.yaml): Configuration for the project
- [logger_config.yaml](config/logger_config.yaml): Configuration for the logger
- [fe_config.yaml](config/fe_config.yaml): Configuration for the feature engineering

**[scripts](scripts)**: This directory contains Python scripts:
- [init.py](scripts/__init__.py): Initialization of a logger
- [env.py](scripts/env.py): Initialization of pydantic settings with environment variables
- [utils.py](scripts/utils.py): Utility functions
- [utils_mlflow.py](scripts/utils_mlflow.py): Utility functions for interacting with MLflow
- [metrics.py](scripts/metrics.py): Definition of custom metrics
- [evaluate.py](scripts/evaluate.py): Function for metrics calculation or evaluation
- [plotters.py](scripts/plotters.py): Functions for plotting
- [transformers.py](scripts/transformers.py): Functions and classes for data transformation

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
The following important notes can be made:
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
<font color='#d0863b'> Properties are probably used for commercial purposes with 0 value for living area.  
The corresponding rows have been removed from the dataset. </font>
9. It is a good idea to create the following features:
- the distance from the center of Moscow to the flat
- the distance from the nearest metro station to the flat
10. The following high-correlated features will be removed in the featuure engineering stage:
- has_elevator
- living_area

---

### Stage III. Feature Engineering

<p style="font-size: 10; color: white; font-family: Verdana; font-weight: bold;">Part 1. Manual Feature Engineering</p>

As a result of EDA, the following new features will be considered:
- `dt.dist_to_center` = the distance from the center of Moscow to the flat
  This feature is calculated using a hand-written `DistanceTransformer`
- `cdt.dist_to_metro` = the distance from the nearest metro station to the flat
  This feature is calculated using a hand-written `ClosestDistanceTransformer`

Also, the following `PolynomialFeatures`, `KBinsDiscretizer` and `SplineTransformer` will be applied for the most important features according to EDA + latitude:
- `total_area`
- `kitchen_area`
- `ceiling_height`
- `build_year` (only `KBinsDiscretizer`)  

The resulting features will have `poly.`, `kbins.` and `spline.` prefixes, respectively.

The following categorical features will be encoded via `CatBoostEncoder`:
- `building_type_int`
- `kbins.total_area`
- `kbins.kitchen_area`
- `kbins.ceiling_height`
- `kbins.build_year`  

The resulting features will have `cbe.` prefix.

The following binary features will be encoded via `OrdinalEncoder` in order to move from boolean to numeric values:
- `is_apartment`
- `is_duplicated`

The resulting features will have `oe.` prefix.

Apart from that, let's consider new features, which are denoted as a normalization of the `total_area` based on group-wise aggregation in the following way for each observation:
$$fnew_{i} = \frac{f_{i} - f_{g, i}}{f_{g, i}}$$
where $i$ is the index of the observation, $f_{i}$ is the original feature, $f_{g, i}$ is the mean value of $f$ over a grouping column $g$ and $fnew_{i}$ is the new feature.
The following features will be considered as a grouping column:
- `kbins.build_year`
- `building_type_int`

<p style="font-size: 10;; font-family: Verdana; font-weight: bold;">Part 2. Auto Feature Engineering</p>

`featuretools` library will be utilised. It will be used only for creating the building age feature via `scalar_subtract_numeric_feature` primitive for `build_year` feature.
The resulting features will have `ft.` prefix.

It is a good practice to generate new features using some common transformations.  
`autofeat` package can do so automatically. Moreover, it performs feature selection  
internally so that only the most important new features are left as a result.  
Let's consisder the following transformations:
- '1/'
- 'log'
- 'cos'
- 'sin'
Apart from that, `autofeat` automatically generates multiplication or division for all different pairs of features.


It is possible to consider a lot more transformations, but it leads to a problem with a computational resources' limitations. <font color='#d0863b'> As for features, all numerical features will be considered except for ones that must be dropped according to correlation analysis in EDA </font>   
- `ft.2024 - build_year` (generated by `featuretools`)
- `flats_count`
- `total_area`
- `kitchen_area`
- `ceiling_height`
- `latitude`
- `longitude`
- `floors_total`
- `rooms`
- `floor`  
Since there is no a prebuilt transformer for `autofeat`, a hand-written `AutoFeatTransformer` will be used.
<font color='#d0863b'> Additionally, all highly correlated features will be dropped - this functionality is implemented in the transformer. </font> 
The resulting features will have `aft.` prefix.


<p style="font-size: 10; font-family: Verdana; font-weight: bold;">Part 3. Dropping features</p>

Finally, it is required to drop unnecessary features:
- `living_area` (high correlation with `total_area`)
- `has_elevator` (high correlation with `floors_total`)
- `id` (it is just an identifier)
- `building_id` (it is just an identifier)
- `subset` (it is just an identifier)
- `build_year` (a new building age feature is calculated using this feature)
- `is_apartment` (it is encoded via `OrdinalEncoder`)
- `is_duplicated` (it is encoded via `OrdinalEncoder`)
- `kbins.total_area` (it is encoded via `CatBoostEncoder`)
- `kbins.kitchen_area` (it is encoded via `CatBoostEncoder`)
- `kbins.ceiling_height` (it is encoded via `CatBoostEncoder`)
- `kbins.build_year` (it is encoded via `CatBoostEncoder`)
- `building_type_int` (it is encoded via `CatBoostEncoder`)

<font color='#d0863b'> Additionally, as the last step, a correlation filter is applied to drop highly correlated features. It is implemented as a part of `PassthroughTransformer`. </font>

`flat_id` column will not be dropped because at least one identifier column might be 
necessary later in order to separate one observation from another.



<p style="font-size: 10; color: white; font-family: Verdana; font-weight: bold;">Part 4. Results and comments </p>

Numeric features were not scaled since only tree-based models were used later.
The whole feature engineering pipeline is parametrized via the configuration file [fe_config.yaml](config/fe_config.yaml) with a help of functions and classes defined in the [transformers.py](scripts/transformers.py) module. This way the pipeline is more flexible and easy to maintain.

As a result, the model with all features generated after feature engineering is slightly worse than a baseline model. It is highly likely that this happened because there are too many features and the model overfits on the training data - the model is not able to generalize well on a separate data. Therefore it is important to perform a feature selection.

Feature Engineering run names:
- `fe_preprocessor2` - for logging the transforming pipeline
- `fe_model2` - for logging the ML model trained with all features

---

### Stage IV. Feature Selection

<p style="font-size: 10; color: white; font-family: Verdana; font-weight: bold;">Part 1. Feature Importance based methods </p>

It is possible to evaluate the importance of each feature and see what features contribute more into the performance of the ML model. There following two ways of evaluation were used:
- `internal feature importance` - calculated for tree-based models
- `permutation feature importance` - calculated via applying permutations to each feature and calculating the difference in the model performance

These techniques were applied to the model obtained as a result of feature engineering. 

Since we have a list of features sorted in the importance order, we can iteravely add the most important feature and see how the model performance changes. As a result, the following number of top features was selected for each approach:
- `internal feature importance` - 27
- `permutation feature importance` - 14

<p style="font-size: 10; color: white; font-family: Verdana; font-weight: bold;">Part 2. Sequential Feature Selection </p>

These group of methods are based on the greedy approach. Given the current state (some features are already selected), the algorithm tries to find the best feature to add (or delete) at the current moment according to some metric. This process is repeated until the stopping criterion is met.

The following two methods were used:
- `Sequential Forward Selection` - starts with an empty set of features and iteratively adds the feature which increases the performance the most
- `Sequential Backward Selection` - starts with all features and iteratively removes the feature which decreases the performance the least

As a result, the following number of top features was selected for each approach:
- `Sequential Forward Selection` - 46
- `Sequential Backward Selection` - 18

Additionaly, an intersection and a union of the feature sets was calculated:
- `Intersection` - 47
- `Union` - 17

<p style="font-size: 10; color: white; font-family: Verdana; font-weight: bold;">Part 3. Comparison </p>

Features obtained via `Permutation Importance` method were the best choice, although by a small margin.

The ML model with the selected features was trained and evaluated. 
It showed a slight improvement in the performance compared to models from previous stages:
- `baseline`: test_r2 = 0.822439
- `after feature engineering`: test_r2 = 0.825053
- `after feature selection`: test_r2 = 0.825172

This means, that both feature engineering and feature selection stages were successful.

The corresponding run name: `fs2`

---

### Stage V. Hyperparameter Tuning

Optuna with Random Sampler and TPE Sampler were considered.
Therefore, a distribution needs to be defined for each hyperparameter. Following the recommendations from the `CatBoost` documentation (`CatBoostRegressor` model was considered), the following distributions were used for tuning hyperparameters:
- `iterations`: int from 100 to 500
- `learning_rate`: float from 0.01 to 1
- `max_depth`: int from 4 to 10
- `l2_leaf_reg`: loguniform from 0.0001 to 1
- `loss_function`: `RMSE` (found to be generally better)
- `random_strength`: int from 1 to 10
- `bagging_temperature`: float from 0 to 2
- `border_count`: int from 64 to 255
- `grow_policy`: `Depthwise` (found to be generally better)

As a result, the following interesting patterns were observed:
- `max_depth`: higher depth was better - a less biased model with high depth is crucial for the task
- `learning_rate`: lower learning rate was better up to 0.1, but lower rates were decreasing the performance
- `iterations`: it was not a critical parameter, although too low values were decreasing the performance
- `border_count`: it was not a critical parameter
- `bagging_temperature`: it was not a critical parameter
- `random_strength`: higher values can more likely decrease 

Unsurprisingly, TPE Sampler provided a better hyperparameter tuning results.

The ML model with the selected hyperparameters was trained and evaluated.
- `baseline`: test_r2 = 0.822439
- `after feature engineering`: test_r2 = 0.825053
- `after feature selection`: test_r2 = 0.825172
- `after hyperparameter tuning`: test_r2 = 0.833082

The final model showed some improvement in the performance compared to models from previous stages, although the margin is not that great since the CatBoost model is very powerful even on the default hyperparameters.

The corresponding run names: `tuning_random2`, `tuning_tpe2`, `tuning_final_model2`

---

Bucket: s3-student-mle-20240730-73c4e0c760