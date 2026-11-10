"""The API tests' client: the app on fakes, with its lifespan (and so its worker) running."""

import pytest
from api_helpers import make_client


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c
