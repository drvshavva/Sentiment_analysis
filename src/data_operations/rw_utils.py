from os.path import dirname
import warnings

from sqlalchemy import create_engine
from pandas import DataFrame
import pandas as pd
import pickle

warnings.filterwarnings(action='ignore')
DATA_PATH = dirname(dirname(dirname(__file__))) + "/data/"


def get_db_connection(db_name: str):
    """
    This is for to connect db

    Params:
        db_name: Name of the db to connect

    Returns:
        connection
    """
    return create_engine(f'sqlite:///{DATA_PATH}{db_name}').connect()


def read_from_db(table_name: str = "relation_1", db_name: str = "fiiler_relations.db") -> DataFrame:
    """
    Read data from db and returns dataFrame

    Args:
         table_name: table name, :type str
         db_name: db name, type: str

    Returns:
         DataFrame
    """
    con = create_engine(f'sqlite:///{DATA_PATH}{db_name}').connect()
    df = pd.read_sql_table(f"{table_name}", con)
    df = df.drop(["index"], axis=1)
    print(f"Data is read. Len of the data {len(df)} and columns {df.columns}")
    return df


def read_from_csv(csv_name: str, sep: str = ",") -> DataFrame:
    """
    This method read data from csv file and  returns DataFrame

    Args:
         sep: csv seperator, :type str
         csv_name: name of the csv, :type str
    Returns:
         DataFrame
    """
    df = pd.read_csv(f"{DATA_PATH}{csv_name}", sep=sep)
    print(f"Data is read. Len of the data {len(df)} and columns {df.columns}")
    return df


def read_from_excel(file_name: str, cols) -> DataFrame:
    """
    This method read data from xlsx file and  returns DataFrame

    Args:
         file_name: name of the csv, :type str
    Returns:
         DataFrame
    """
    df = pd.read_excel(f"{DATA_PATH}{file_name}", names=cols)
    print(f"Data is read. Len of the data {len(df)} and columns {df.columns}")
    return df


def write_to_csv(csv_name: str, data: DataFrame):
    """
    This method write data from csv file and  returns DataFrame

    Args:
         data: data to save, :type str
         csv_name: name of the csv, :type str
    Returns:
         None
    """
    data.to_csv(f"{DATA_PATH}{csv_name}", index=False)
    print(f"Data is wrote to path {DATA_PATH}, with name {csv_name}")


def load_model(path: str):
    """
    This method loads the model

    :param path: path of the mode, :type str
    :return:
    """
    with open(path, 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model
