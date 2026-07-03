import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import RANDOM_STATE

def load_dataset(dataset_path):
    """
    Loads the dataset from disk.

    Returns
    -------
    pandas.DataFrame
    """

    df = pd.read_csv(dataset_path)

    return df