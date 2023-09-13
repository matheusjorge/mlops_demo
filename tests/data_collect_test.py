import os

os.sys.path.insert(0, "src")

from data_collect import data_collect
import pandas as pd

def test_return_is_dataframe():
    data = data_collect()
    assert isinstance(data, pd.DataFrame)