"""The Basic Model Interface (BMI) for Python."""

from __future__ import annotations

from bmipy._version import __version__
from bmipy.bmi import Bmi
from bmipy.bmi import BmiExtension

__all__ = ["__version__", "Bmi", "BmiExtension"]
