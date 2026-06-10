import pytest

from slime.utils.dynamic_batch import (
    compute_dynamic_global_batch_size,
    get_aligned_train_data_trim_len,
    trim_sample_aligned_data,
)


@pytest.mark.unit
def test_dynamic_gbs_uses_filtered_sample_count():
    dynamic_gbs, desired_steps, realized_steps, wasted = compute_dynamic_global_batch_size(
        503,
        dp_size=2,
        target_steps=2,
    )

    assert dynamic_gbs == 250
    assert desired_steps == 2
    assert realized_steps == 2
    assert wasted == 3
    assert get_aligned_train_data_trim_len(503, global_batch_size=dynamic_gbs, dp_size=2) == 500
    assert get_aligned_train_data_trim_len(131, global_batch_size=65, dp_size=2) == 130


@pytest.mark.unit
def test_trim_sample_aligned_data_only_trims_per_sample_lists():
    data = {
        "tokens": list(range(503)),
        "raw_reward": list(range(503)),
        "metadata": [{"i": i} for i in range(503)],
        "not_per_sample": ["keep", "me"],
    }

    trim_sample_aligned_data(data, 500, 503)

    assert len(data["tokens"]) == 500
    assert len(data["raw_reward"]) == 500
    assert data["metadata"][-1] == {"i": 499}
    assert data["not_per_sample"] == ["keep", "me"]
