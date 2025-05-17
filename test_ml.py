from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model
from sklearn.model_selection import train_test_split
import pytest
import numpy as np
import pandas as pd

cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

@pytest.fixture(scope="session")
def data():
    # code to load in the data.
    datapath = "./data/census.csv"
    return pd.read_csv(datapath)

def test_features_exists(data):
    """
    # Test the features that are required for our training model exists in our dataset
    """

    assert check_features(data, cat_features) == True


def test_compute_model_metrics():
    """
    # Test the compute_model_metrics method with random array of 0, 1
    # output should always provide a precision, recall, and fbeta
    """
    y_slice = np.random.randint(2, size= 10)
    preds = np.random.randint(2, size= 10)

    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None


def test_inference(data):
    """
    # Test training a model and a prediction from the model will have the same shape
    """
    train, test = train_test_split(data,
                            test_size=0.2,
                            random_state=42,
                            stratify=data['salary'])

    X, y, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    y_preds = inference(model, X)
    assert y.shape == y_preds.shape

def check_features(df, features):
    return all(feature in df.columns for feature in features)