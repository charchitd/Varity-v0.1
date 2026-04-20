"""Varity entry point for python -m varity."""

from varity.cli import _check_path_and_warn, main

if __name__ == "__main__":
    _check_path_and_warn()
    main()
