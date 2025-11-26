"""Dashboard package exposing helpers to render the training monitor interfaces."""


def run_cli(*args, **kwargs):
    from .cli import main as _main

    return _main(*args, **kwargs)


def run_textual(*args, **kwargs):
    from .textual_app import main as _main

    return _main(*args, **kwargs)


__all__ = ["run_cli", "run_textual"]
