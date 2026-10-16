from __future__ import annotations

import inspect

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
    (
        "get_varset_member_count",
        "get_varset_members",
        "get_extension",
        "get_var_index",
    ),
)
def test_bmi_v3_methods_abstract(name):
    assert name in Bmi.__abstractmethods__


@pytest.mark.parametrize(
    "cls,name",
    (
        (Bmi, "get_var_grid"),
        (Bmi, "get_var_type"),
        (Bmi, "get_var_units"),
        (Bmi, "get_var_itemsize"),
        (Bmi, "get_var_nbytes"),
        (Bmi, "get_var_location"),
        (Bmi, "get_value"),
        (Bmi, "get_value_at_indices"),
        (Bmi, "set_value"),
        (Bmi, "set_value_at_indices"),
        (BmiPointers, "get_value_ptr"),
    ),
)
def test_variable_methods_take_index(cls, name):
    params = list(inspect.signature(getattr(cls, name)).parameters)
    assert params[1] == "index"


def test_pointers_is_an_extension():
    assert issubclass(BmiPointers, BmiExtension)
    assert BmiPointers.__abstractmethods__ == frozenset({"get_value_ptr"})
