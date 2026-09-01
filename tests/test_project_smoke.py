"""Smoke tests for the Recoverly project scaffold."""


def test_backend_packages_importable() -> None:
    import backend
    import backend.data

    assert backend.__name__ == "backend"
    assert backend.data.__name__ == "backend.data"
