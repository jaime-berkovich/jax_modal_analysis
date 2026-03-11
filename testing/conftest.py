from pathlib import Path


_TESTING_DIR = Path(__file__).resolve().parent
_VENDORED_UPSTREAM_TESTS = (_TESTING_DIR / "jax-fem" / "tests").resolve()


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path)).resolve()
    return path == _VENDORED_UPSTREAM_TESTS or _VENDORED_UPSTREAM_TESTS in path.parents
