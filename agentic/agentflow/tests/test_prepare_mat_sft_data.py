import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_mat_sft_data import convert_sft
from test_prepare_mat_data import FIXTURE


def test_convert_sft_keeps_every_teacher_step():
    output = convert_sft(FIXTURE, image_root="/data/images")
    assert len(output) == 8
    assert [item["metadata"]["step_type"] for item in output].count("code") == 2


def test_convert_sft_uses_step_specific_images_and_targets():
    output = convert_sft(FIXTURE, image_root="/data/images")
    over = [item for item in output if item["metadata"]["trajectory_id"] == "overexposure_60"]
    assert over[0]["image_path"] == ["/data/images/overexposure_60_proc.png"]
    assert over[1]["image_path"] == ["/data/images/overexposure_60_proc.png"]
    assert over[2]["image_path"] == ["/data/images/overexposure_60_ori.png"]
    assert "<problem>" in over[0]["messages"][1]["content"]
    assert "<code>" in over[1]["messages"][1]["content"]
    assert "<answer>" in over[2]["messages"][1]["content"]


def test_convert_sft_emits_one_image_placeholder_per_sample():
    for sample in convert_sft(FIXTURE):
        assert sample["messages"][0]["content"].count("<image>") == 1
        assert sample["messages"][0]["role"] == "user"
        assert sample["messages"][1]["role"] == "assistant"


def test_convert_sft_preserves_teacher_context():
    output = convert_sft(FIXTURE)
    code = next(item for item in output if item["metadata"]["step_type"] == "code")
    assert code["messages"][0]["content"].endswith("...")
