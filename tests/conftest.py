collect_ignore_glob = ["devtests/*", "project_validation/*"]


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'slow: marks tests as slow (deselect with \'-m "not slow"\')'
    )