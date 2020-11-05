# content of conftest.py

import pytest


def pytest_addoption(parser):
    """
    Adds a pytest_addoption.

    Args:
        parser: (todo): write your description
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--use_cuda", action="store_true", default=False, help="run tests on gpu"
    )


def pytest_configure(config):
    """
    Èi̇·åıĸ pytest_configure. py

    Args:
        config: (todo): write your description
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """
    Modify all items in the given config.

    Args:
        config: (todo): write your description
        items: (todo): write your description
    """
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

@pytest.fixture
def use_cuda(request):
    """ Run test on gpu """
    return request.config.getoption("--use_cuda")
