from __future__ import annotations

import pytest

from bmipy import Bmi
from bmipy import BmiExtension
from bmipy import BmiPointers


@pytest.mark.parametrize(
    "name",
    (
        "get_input_item_count",
        "get_output_item_count",
        "get_input_var_names",
        "get_output_var_names",
        "get_value_ptr",
    ),
)
def test_bmi_v2_methods_removed(name):
    assert not hasattr(Bmi, name)


@pytest.mark.parametrize(
    "name",
    ("get_varset_member_count", "get_varset_members", "get_extension"),
)
def test_bmi_v3_methods_abstract(name):
    assert name in Bmi.__abstractmethods__


def test_pointers_is_an_extension():
    assert issubclass(BmiPointers, BmiExtension)
    assert BmiPointers.__abstractmethods__ == frozenset({"get_value_ptr"})
