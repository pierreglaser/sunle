"""Allow to specify compute budgets for running/comparing inference results in pytest"""
from .test_utils import get_param_sets


def pytest_addoption(parser):
    parser.addoption(
        "--large-budget",
        action="store_true",
        default=False,
        help="run and compare large training/simulation/inference budget settings",
    )


def pytest_generate_tests(metafunc):
    """Adapt tests to compute budget specifications.

    Determine whether to run and compare SUNLE with previous output
    using a large compute budget scenario (good for validating accuracy), or
    a small compute budget scenario (good for quickly validating reproducibility
    when using CI runners).
    """
    large_budget = metafunc.config.getoption("--large-budget")

    if metafunc.function.__name__ in ("test_unle", "test_sunle", "test_sunle_vi"):
        method = metafunc.function.__name__[5:]
        param_sets = get_param_sets(method, large_budget)
        metafunc.parametrize("param_set", param_sets)
