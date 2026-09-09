"""Smoke test verifying the project and test setup work."""

import analyst


def test_analyst_package_imports():
    assert analyst.__version__ == "0.1.0"
