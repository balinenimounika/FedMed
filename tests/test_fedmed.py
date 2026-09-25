def test_fedmed_basic():
    assert True


def test_fedmed_project_structure():
    import os

    assert os.path.exists("server")
    assert os.path.exists("client")
    assert os.path.exists("model")
    assert os.path.exists("data")
    assert os.path.exists("docs")