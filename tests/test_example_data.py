import os
from importlib import import_module
import pandas as pd
from pyrsm.example_data import load_data


def test_all_data():
    all_data = load_data()
    assert isinstance(all_data, dict), "Return value should be a dictionary"
    assert len(all_data) == 28, "Where datasets added or removed?"


def test_package():
    data_data = load_data(pkg="data")
    assert isinstance(data_data, dict), "Return value should be a dictionary"
    assert len(data_data) == 5, "Where datasets added or removed to the data package?"


def test_dataset():
    model_data = load_data(pkg="model", name="dvd")
    assert isinstance(model_data, dict), "Return value should be a dictionary"
    assert model_data["dvd"].shape == (
        20_000,
        5,
    ), "DVD data doesn't have the right shape?"
