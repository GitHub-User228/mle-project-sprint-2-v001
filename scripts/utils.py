from pathlib import Path

import yaml
import joblib
import psycopg
import itertools
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import ks_2samp, anderson_ksamp

from scripts import logger
from scripts.env import env_vars


def read_yaml(path: Path) -> dict:
    """
    Reads a yaml file, and returns a dict.

    Args:
        path_to_yaml (Path):
            Path to the yaml file

    Returns:
        Dict:
            The yaml content as a dict.

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
        logger.error(msg)
        raise ValueError(msg)
    try:
        with open(path, "r") as file:
            content = yaml.safe_load(file)
        logger.info(f"YAML file {path} has been loaded")
        return content
    except FileNotFoundError as e:
        msg = f"File {path} not found"
        logger.error(msg)
        raise FileNotFoundError(msg) from e
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML file {path}"
        logger.error(msg)
        raise yaml.YAMLError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while reading YAML file {path}"
        logger.error(msg)
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
        logger.error(msg)
        raise ValueError(msg)

    try:
        with open(path, "rb") as f:
            model = joblib.load(f)
        logger.info(f"Model {path} has been loaded")
        return model
    except FileNotFoundError as e:
        msg = f"File '{path}' does not exist"
        logger.error(msg)
        raise FileNotFoundError(msg) from e
    except IOError as e:
        msg = f"An I/O error occurred while loading a model from {path}"
        logger.error(msg)
        raise IOError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while loading a model from {path}"
        logger.error(msg)
        raise Exception(msg) from e


def read_table_from_db(
    table_name: str,
    connection: dict = {
        "sslmode": "require",
        "target_session_attrs": "read-write",
    },
    date_columns: list[str] | None = None,
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
        date_columns (list[str] | None, optional):
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
        logger.error(msg)
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
        logger.error(msg)
        raise ValueError(msg)
    return n_bins


def compare_disributions(
    data: dict[str, np.ndarray | pd.Series],
    significance_level: float = 0.05,
):
    """
    Compares the distributions of multiple datasets using the
    Kolmogorov-Smirnov (KS) test and the Anderson-Darling test.

    Args:
        data (dict[str, np.ndarray | pd.Series]):
            A dictionary where the keys are the names of the datasets
            and the values are the corresponding data.
        significance_level (float, optional):
            The significance level to use for the hypothesis tests.
            Defaults to 0.05.

    Returns:
        pd.DataFrame:
            A DataFrame containing the results of the hypothesis tests,
            including the p-values and whether the distributions are
            considered similar or not.
    """

    pvalues = pd.DataFrame(
        list(itertools.combinations(list(data.keys()), 2)),
        columns=["data1", "data2"],
    )
    pvalues["ks_2samp.pvalue"] = None
    pvalues["anderson_ksamp.pvalue"] = None
    pvalues["significance_level"] = significance_level

    for i in tqdm(pvalues.index):
        subset1 = pvalues.loc[i, "data1"]
        subset2 = pvalues.loc[i, "data2"]
        _, p_value = ks_2samp(
            data[subset1],
            data[subset2],
        )
        pvalues.loc[i, "ks_2samp.pvalue"] = p_value

        res = anderson_ksamp(
            [
                data[subset1],
                data[subset2],
            ]
        )
        pvalues.loc[i, "anderson_ksamp.pvalue"] = res.pvalue

    pvalues["ks_2samp.are_similar"] = (
        pvalues["ks_2samp.pvalue"] >= significance_level
    )
    pvalues["anderson_ksamp.are_similar"] = (
        pvalues["anderson_ksamp.pvalue"] >= significance_level
    )

    return pvalues
