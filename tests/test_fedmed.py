import os

from server.client import FedMedClient


def test_fedmed_project_structure():
    assert os.path.exists("server")
    assert os.path.exists("client")
    assert os.path.exists("model")
    assert os.path.exists("data")
    assert os.path.exists("docs")


def test_fedmed_client_training():
    client = FedMedClient()

    parameters = []
    updated_parameters, num_examples, metrics = client.fit(
        parameters,
        {}
    )

    assert updated_parameters == parameters
    assert num_examples == 1
    assert metrics == {}


def test_fedmed_client_evaluation():
    client = FedMedClient()

    loss, num_examples, metrics = client.evaluate(
        [],
        {}
    )

    assert loss == 0.0
    assert num_examples == 1
    assert metrics["accuracy"] == 0.0