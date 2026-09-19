import numpy as np
import pytest

from laya_coreml.inputs import collate_items
from laya_coreml.prompt import PromptMixin


def shape(**kwargs):
    return {
        "batch_size": 3,
        "max_length": 128,
        "min_length": 16,
        "max_options": 4,
        "flexible": True,
        **kwargs,
    }


def item(length=17, markers=None):
    return {"ids": list(range(length)), "markers": markers or [3, 6], "qtype": 2}


def test_partial_batch_and_padding():
    arrays = collate_items([item()], 9, shape=shape())
    assert arrays["input_ids"].shape == (3, 32)
    assert all(v.dtype == np.int32 for v in arrays.values())
    assert arrays["attention_mask"].sum(axis=1).tolist() == [17, 1, 1]
    assert arrays["marker_mask"].sum(axis=1).tolist() == [2, 0, 0]
    assert arrays["input_ids"][0, 17:].tolist() == [9] * 15
    assert arrays["qtype"].tolist() == [2, 0, 0]


@pytest.mark.parametrize("length,expected", [(4, 16), (16, 16), (17, 32), (64, 64), (127, 128)])
def test_flexible_length_buckets(length, expected):
    batch = collate_items([item(length)], 0, shape=shape())
    assert batch["input_ids"].shape[1] == expected


def test_fixed_length_does_not_silently_truncate():
    batch = collate_items([item(17)], 0, shape=shape(flexible=False))
    assert batch["input_ids"].shape[1] == 128
    with pytest.raises(ValueError, match="129 tokens"):
        collate_items([item(129)], 0, shape=shape(flexible=False))


@pytest.mark.parametrize("items", [[], [item()] * 4, [item(markers=[1, 2, 3, 4, 5])]])
def test_reject_export_capacity_overflow(items):
    with pytest.raises(ValueError):
        collate_items(items, 0, shape=shape())


@pytest.mark.parametrize(
    "definition",
    [
        {"type": "freeform", "instructions": "hi"},
        {"type": "choice", "criteria": ["a"]},
        {"type": "choice", "instructions": "hi", "criteria": []},
        {"type": "choice", "instructions": "hi", "criteria": ["a", "a"]},
        {"type": "score", "instructions": "hi", "criteria": {}},
    ],
)
def test_reject_invalid_question(definition):
    with pytest.raises(ValueError):
        PromptMixin._to_internal(definition)


def test_structured_question_criteria_preserved():
    result = PromptMixin._to_internal(
        {
            "type": "choice",
            "instructions": {"task": "pick"},
            "criteria": {"zero": 0, "false": False, "object": {"x": 1}},
        }
    )
    assert result["crit"] == {"zero": 0, "false": False, "object": {"x": 1}}


def test_known_failed_gpu_configuration_is_rejected_before_loading(tmp_path):
    import json

    from laya_coreml import Agent

    (tmp_path / "coreml_config.json").write_text(
        json.dumps(
            {
                "format": "laya-coreml",
                "format_version": 1,
                "shape": shape(),
            }
        )
    )
    with pytest.raises(ValueError, match="failed local fidelity"):
        Agent(tmp_path, compute_units="cpu_gpu")


def test_enumerated_shapes_select_next_bucket():
    batch = collate_items([item(65)], 0, shape=shape(lengths=[16, 32, 64, 96, 128]))
    assert batch["input_ids"].shape[1] == 96
