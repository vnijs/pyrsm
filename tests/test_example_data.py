from pyrsm.example_data import load_data
import pandas as pd


def test_all_data():
    data, descriptions = load_data()
    assert isinstance(data, dict), "Return value should be a dictionary"
    assert len(data) == 28, "Where datasets added or removed?"
    assert [
        isinstance(v, str) for v in descriptions.values()
    ], "Descriptions should all be strings"
    assert isinstance(descriptions, dict), "Return value should be a dictionary"


def test_package():
    data, description = load_data(pkg="data")
    assert isinstance(data, dict), "Return value should be a dictionary"
    assert len(data) == 5, "Where datasets added or removed to the data package?"


def test_dataset():
    data, description = load_data(pkg="model", name="dvd")
    assert isinstance(data, pd.DataFrame), "Return value should be a dictionary"
    assert data.shape == (
        20_000,
        5,
    ), "DVD data doesn't have the right shape?"


if __name__ == "__main__":
    test_all_data()
    test_package()
    test_dataset()
