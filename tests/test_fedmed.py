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


def test_three_hospital_configuration():
    from src.config import NUM_CLIENTS, CLIENT_CLASS_DISTRIBUTIONS

    assert NUM_CLIENTS == 3
    assert set(CLIENT_CLASS_DISTRIBUTIONS.keys()) == {0, 1, 2}

    for distribution in CLIENT_CLASS_DISTRIBUTIONS.values():
        assert set(distribution.keys()) == {0, 1}
        assert sum(distribution.values()) == 1.0


def test_server_address_from_environment(monkeypatch):
    monkeypatch.setenv("FEDMED_SERVER_ADDRESS", "192.168.1.100:8080")

    import importlib
    import src.config as config

    importlib.reload(config)

    assert config.SERVER_ADDRESS == "192.168.1.100:8080"


def test_server_address_default(monkeypatch):
    monkeypatch.delenv("FEDMED_SERVER_ADDRESS", raising=False)

    import importlib
    import src.config as config

    importlib.reload(config)

    assert config.SERVER_ADDRESS == "127.0.0.1:8080"