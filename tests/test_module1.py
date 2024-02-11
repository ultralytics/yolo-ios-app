# Ultralytics YOLO 🚀, AGPL-3.0 license

from module1 import add_numbers


def test_add_numbers():
    """Test the add_numbers function from module1."""
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(-1, -1) == -2
