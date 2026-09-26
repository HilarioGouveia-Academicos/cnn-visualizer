import pytest
from app.settings import training_allowed


@pytest.mark.parametrize("secrets,expected", [
    ({}, True), ({"app": {}}, True),
    ({"app": {"allow_training": True}}, True),
    ({"app": {"allow_training": False}}, False),
    ({"app": {"allow_training": "false"}}, False),
])
def test_training_configuration(monkeypatch, secrets, expected):
    monkeypatch.setattr("app.settings.st.secrets", secrets)
    assert training_allowed() is expected


def test_missing_secrets_file(monkeypatch):
    class MissingSecrets:
        def get(self, *args):
            raise FileNotFoundError()
    monkeypatch.setattr("app.settings.st.secrets", MissingSecrets())
    assert training_allowed()
