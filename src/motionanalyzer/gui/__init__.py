"""
GUI support: model runners and shared constants.

Model modes are fully separated; the desktop GUI calls runners.run(mode, ...)
so that each mode (physics, dream, patchcore, grid_search, bayesian) has
a single entry point and no cross-dependencies in the GUI layer.
"""

from motionanalyzer.gui.runners import run_training_or_optimization

__all__ = ["run_training_or_optimization"]
