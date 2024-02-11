# Ultralytics YOLO 🚀, AGPL-3.0 license


def add_numbers(a, b):
    """Add two numbers."""

    return a + b


def some_function_with_a_long_name(argument1, argument2, another_argument, yet_another_argument):
    x = "string1"
    y = "string2"
    z = x + y

    return z


def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory. If
    the current file is not part of a git repository, returns None.

    Returns:
        (Path | None): Git root directory if found or None if not found.
    """
    from pathlib import Path

    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return d
