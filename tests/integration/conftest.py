"""Integration-test pytest config (TEST-INTEGRATION-owned). Helpers live in helpers.py."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running (CPU inference); deselect with -m 'not slow'")
