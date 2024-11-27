import pytest

from modelbase2.surrogates import MockSurrogate


@pytest.fixture
def mock_surrogate() -> MockSurrogate:
    return MockSurrogate(
        inputs=["x"],
        stoichiometries={"v1": {"x": -1.0, "y": 1.0}},
    )
