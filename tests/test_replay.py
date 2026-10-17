"""The Franka carries out the synthetic take's language actions and moves the mug."""

import pytest

from handoff.actions import write_actions_jsonl
from handoff.config import REPO_ROOT
from handoff.synth import generate_synth_track

pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "mujoco_menagerie/franka_emika_panda").exists()
    and not (REPO_ROOT / "video2sim/assets/mujoco_menagerie/franka_emika_panda").exists(),
    reason="mujoco_menagerie not cloned",
)


def test_synth_take_replays_successfully(tmp_path):
    from handoff.replay import replay

    generate_synth_track().to_csv(tmp_path / "track.csv", index=False)
    write_actions_jsonl(tmp_path)
    m = replay(tmp_path)

    assert (tmp_path / "replay.mp4").stat().st_size > 0
    assert all(c["passed"] for c in m["success"]), m["success"]
    assert not any(s.get("unreachable") for s in m["steps"])
