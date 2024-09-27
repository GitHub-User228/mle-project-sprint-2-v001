import os
from pathlib import Path

import yaml
import joblib
import psycopg
import requests
import itertools
import numpy as np
import pandas as pd
from typing import *
from scipy import stats
from tqdm.auto import tqdm

from scripts import logger
from scripts.env import env_vars


def save_yaml(data: dict, path: Path) -> None:
    """
    Saves a dictionary to a YAML file.

    Args:
        data (dict):
            The dictionary to be saved as YAML.
        path (Path):
            The path to the YAML file where the data will be saved.

    Raises:
        ValueError:
            If the file is not a YAML file.
        IOError:
            If an I/O error occurs during the saving process.
        yaml.YAMLError:
            If there is an error encoding the dictionary to YAML.
    """
    if path.suffix not in [".yaml", ".yml"]:
        msg = f"The file {path} is not a YAML file"
        logger.error(f"{msg}: {e}")
        raise ValueError(msg)
    try:
        with open(path, "w") as file:
            yaml.safe_dump(data, file)
        logger.info(f"Dictionary has been saved to YAML file {path}")
    except IOError as e:
        msg = f"An I/O error occurred while saving the dictionary to {path}"
        logger.error(f"{msg}: {e}")
        raise IOError(msg) from e
    except yaml.YAMLError as e:
        msg = f"Error encoding dictionary to YAML file {path}"
        logger.error(f"{msg}: {e}")
        raise yaml.YAMLError(msg) from e
    except Exception as e:
        msg = (
            f"An unexpected error occurred while saving the dictionary"
            f"to {path}"
        )
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def read_yaml(path: Path, verbose: bool = True) -> Dict:
    """
    Reads a yaml file, and returns a dict.

    Args:
        path_to_yaml (Path):
            Path to the yaml file

    Returns:
        Dict:
            The yaml content as a dict.
        verbose:
            Whether to do any info logs

    Raises:
        ValueError:
            If the file is not a YAML file
        FileNotFoundError:
            If the file is not found.
        yaml.YAMLError:
            If there is an error parsing the yaml file.
    """
    if path.suffix not in [".yaml", ".yml"]:
        msg = f"The file {path} is not a YAML file"
        logger.error(f"{msg}: {e}")
        raise ValueError(msg)
    try:
        with open(path, "r") as file:
            content = yaml.safe_load(file)
        if verbose: 
            logger.info(f"YAML file {path} has been loaded")
        return content
    except FileNotFoundError as e:
        msg = f"File {path} not found"
        logger.error(f"{msg}: {e}")
        raise FileNotFoundError(msg) from e
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML file {path}"
        logger.error(f"{msg}: {e}")
        raise yaml.YAMLError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while reading YAML file {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def read_pkl(path: Path) -> object:
    """
    Reads a model object from a file using joblib.

    Args:
        path (Path):
            The path to the file with the model to load.

    Returns:
        object:
            The loaded model object.

    Raises:
        ValueError:
            If the file does not have a .pkl extension.
        FileNotFoundError:
            If the file does not exist.
        IOError:
            If an I/O error occurs during the loading process.
        Exception:
            If an unexpected error occurs while loading the model.
    """

    if path.suffix != ".pkl":
        msg = f"The file {path} is not a pkl file"
        logger.error(f"{msg}: {e}")
        raise ValueError(msg)

    try:
        with open(path, "rb") as f:
            model = joblib.load(f)
        logger.info(f"Model {path} has been loaded")
        return model
    except FileNotFoundError as e:
        msg = f"File '{path}' does not exist"
        logger.error(f"{msg}: {e}")
        raise FileNotFoundError(msg) from e
    except IOError as e:
        msg = f"An I/O error occurred while loading a model from {path}"
        logger.error(f"{msg}: {e}")
        raise IOError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while loading a model from {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def read_table_from_db(
    table_name: str,
    connection: dict = {
        "sslmode": "require",
        "target_session_attrs": "read-write",
    },
    date_columns: List[str] | None = None,
) -> pd.DataFrame:
    """
    Reads a table from a PostgreSQL database and returns it as a pandas
    DataFrame.

    Args:
        table_name (str):
            The name of the table to read from the database.
        connection (dict, optional):
            A dictionary of connection parameters for the database.
            Defaults to `{"sslmode": "require",
            "target_session_attrs": "read-write"}`.
        date_columns (List[str] | None, optional):
            A list of column names that should be converted to datetime
            objects. Defaults to `None`.

    Returns:
        pd.DataFrame:
            A pandas DataFrame containing the data from the specified
            table.

    Raises:
        psycopg.Error:
            If an error occurs while reading the table from the
            database.
    """

    postgres_credentials = {
        "host": env_vars.db_destination_host,
        "port": env_vars.db_destination_port,
        "dbname": env_vars.db_destination_name,
        "user": env_vars.db_destination_user,
        "password": env_vars.db_destination_password,
    }

    connection.update(postgres_credentials)

    try:
        with psycopg.connect(**connection) as conn:

            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {table_name}")
                data = cur.fetchall()
                columns = [col[0] for col in cur.description]
    except psycopg.Error as e:
        msg = f"Error reading table {table_name} from database"
        logger.error(f"{msg}: {e}")
        raise psycopg.Error(msg) from e

    df = pd.DataFrame(data, columns=columns)
    if date_columns:
        df[date_columns] = pd.to_datetime(df[date_columns])

    logger.info(f"Table {table_name} has been loaded")

    return df


def get_bins(x: int) -> int:
    """
    Calculates the appropriate number of bins for the histogram
    according to the number of the observations

    Args:
        x (int):
            Number of the observations

    Returns:
        int:
            Number of bins
    """
    if x > 0:
        n_bins = max(int(1 + 3.2 * np.log(x)), int(1.72 * x ** (1 / 3)))
    else:
        msg = (
            "An invalid input value passed. Expected a positive "
            + "integer, but got {x}"
        )
        logger.error(f"{msg}: {e}")
        raise ValueError(msg)
    return n_bins


def compare_distributions(
    data: pd.DataFrame,
    features: List[str],
    hue: str,
    significance_level: float = 0.05,
    are_categorical: bool = False,
) -> pd.DataFrame:
    """
    Compares the distributions on the dataset using
    multiple statistical tests.

    Args:
        data (pd.DataFrame):
            A DataFrame with the data.
        features (List[str]):
            A list of feature names to compare the distributions for.
        hue (str):
            The name of the column in the DataFrame that represents
            the "hue" or grouping variable.
        significance_level (float, optional):
            The significance level to use for the hypothesis tests.
            Defaults to 0.05.
        are_categorical (bool, optional):
            Whether the features are categorical or not.
            Defaults to False.

    Returns:
        pd.DataFrame:
            A DataFrame containing the results of the hypothesis tests,
            including the p-values and whether the distributions are
            considered similar or not.
    """

    def get_pvalues_num(
        data1: np.ndarray | pd.Series, data2: np.ndarray | pd.Series
    ) -> Dict[str, float]:
        """
        Calculates the p-values for the Kolmogorov-Smirnov (KS) test,
        the Anderson-Darling test, Mann-Whitney U test and Wilcoxon
        rank-sum test.

        Args:
            data1 (np.ndarray | pd.Series):
                The first dataset.
            data2 (np.ndarray | pd.Series):
                The second dataset.

        Returns:
            Dict[str, float]:
                A dictionary containing the p-values for tests
        """
        res = {}
        res["ks_2samp"] = stats.ks_2samp(data1, data2)[1]
        res["anderson_ksamp"] = stats.anderson_ksamp([data1, data2]).pvalue
        res["mannwhitneyu"] = stats.mannwhitneyu(data1, data2)[1]
        res["ranksums"] = stats.ranksums(data1, data2)[1]
        return res

    def get_pvalues_cat(
        data: pd.DataFrame, feature: str, hue: str
    ) -> Dict[str, float]:
        """
        Calculates the p-values for the Fisher's exact test and the
        Barnard's exact test for a 2x2 contingency table with small
        sample size, or the p-values for the Chi-square test and the
        Chi-square test with likelihood ratio for larger samples.

        Args:
            data (pd.DataFrame):
                The DataFrame containing the data.
            feature (str):
                The name of the feature column.
            hue (str):
                The name of the hue column.

        Returns:
            Dict[str, float]:
                 A dictionary containing the p-values for the tests.
        """

        observed = pd.crosstab(data[feature], data[hue]).values
        res = {}
        if len(data) < 30:
            if observed.shape == (2, 2):
                res["fisher_exact"] = stats.fisher_exact(observed)[1]
                res["barnard_exact"] = stats.barnard_exact(observed)[1]
        else:
            res["chi2_contingency"] = stats.chi2_contingency(observed)[1]
            res["chi2_contingency_likelihood"] = stats.chi2_contingency(
                observed, lambda_="log-likelihood"
            )[1]
        return res

    pv = pd.DataFrame(
        [
            (value,) + combination
            for value in features
            for combination in itertools.combinations(data[hue].unique(), 2)
        ],
        columns=["feature", "hue1", "hue2"],
    )

    pv["significance_level"] = significance_level

    for feature in features:

        for i in pv.loc[pv["feature"] == feature].index:
            hue1 = pv.loc[i, "hue1"]
            hue2 = pv.loc[i, "hue2"]
            if are_categorical:
                vals = get_pvalues_cat(
                    data=data.loc[
                        data[hue].isin([hue1, hue2]), [feature, hue]
                    ],
                    feature=feature,
                    hue=hue,
                )
            else:
                vals = get_pvalues_num(
                    data.query(f"subset == '{hue1}'")[feature],
                    data.query(f"subset == '{hue2}'")[feature],
                )
            for key, value in vals.items():
                pv.loc[i, f"{key}.pv"] = value

    for col in pv.columns:
        if col.endswith(".pv"):
            pv[f"{col.replace('.pv', '')}.similar"] = (
                pv[col] >= significance_level
            )
    return pv


def reduce_size(df: pd.DataFrame) -> None:
    """
    Reduces the size of the DataFrame by converting integer
    and float columns to smaller data types.

    This function iterates through each column in the DataFrame and
    checks the minimum and maximum values. It then converts the column
    to a smaller data type if possible, such as `uint8`, `uint16`,
    `int8`, or `int16`, to reduce the memory footprint of the DataFrame.

    Args:
        df (pd.DataFrame):
            The DataFrame to be reduced in size.
    """
    print(
        "Dataframe memory usage before optimisation:",
        df.memory_usage().sum() / (1024**2),
        "MB",
    )
    for col in tqdm(df.columns):
        if "int" in df[col].dtype.name:
            if df[col].min() >= 0:
                if df[col].max() <= 255:
                    df[col] = df[col].astype("uint8")
                elif df[col].max() <= 65535:
                    df[col] = df[col].astype("uint16")
                else:
                    df[col] = df[col].astype("uint32")
            else:
                if max(abs(df[col].min()), df[col].max()) <= 127:
                    df[col] = df[col].astype("int8")
                elif max(abs(df[col].min()), df[col].max()) <= 32767:
                    df[col] = df[col].astype("int16")
                else:
                    df[col] = df[col].astype("int32")
        elif "float" in df[col].dtype.name:
            df[col] = df[col].astype("float32")
    print(
        "Dataframe memory usage after optimisation:",
        df.memory_usage().sum() / (1024**2),
        "MB",
    )


def retrieve_moscow_metro_stations_data() -> None:
    """
    Retrieves data for Moscow metro stations from the HH.ru API.
    """
    try:
        with requests.Session() as s:
            response = s.get(
                url="https://api.hh.ru/metro/1",
            )
            response.raise_for_status()
    except Exception as e:
        msg = f"Error retrieving Moscow metro stations data"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e
    try:
        data = pd.DataFrame(
            [
                (v["id"], v["name"], k["name"], k["lat"], k["lng"], k["order"])
                for v in response.json()["lines"]
                for k in v["stations"]
            ],
            columns=[
                "line_id",
                "line_name",
                "station_name",
                "lat",
                "long",
                "order",
            ],
        )
        data = data.query("(long > 30) and (long < 38.5)").reset_index(
            drop=True
        )
        data["lat"] = data["lat"].astype(np.float32)
        data["long"] = data["long"].astype(np.float32)
    except Exception as e:
        msg = f"Error processing Moscow metro stations data"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e
    try:
        data.to_csv(
            os.path.join(
                env_vars.fe_artifacts_dir, "moscow_metro_stations.csv"
            ),
            index=False,
        )
    except Exception as e:
        msg = f"Error saving Moscow metro stations data"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e
    logger.info(f"Moscow metro stations data has been retrieved and saved")
