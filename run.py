"""python run.py takes/<take> [--video raw/<file>.mp4] [--from STAGE] [--to STAGE] [--force]

Runs the Handoff pipeline stages, in order, over a single take directory.
"""

import argparse
from pathlib import Path

STAGES = ["perception", "transcribe", "actions", "vlm", "export", "replay", "compose"]
OUTPUTS = {"actions": "actions.jsonl", "replay": "replay.mp4"}


def run_stage(stage, take_dir, video, force):
    out = Path(take_dir) / OUTPUTS.get(stage, "")
    if stage in OUTPUTS and out.exists() and not force:
        print(f"[{stage}] {out} exists, skipping (--force to rerun)")
        return
    if stage == "actions":
        from handoff.actions import write_actions_jsonl
        from handoff.config import load_config
        cfg = load_config()
        print(f"[actions] wrote {write_actions_jsonl(take_dir, window_s=cfg.window_s, fps=cfg.fps)}")
    elif stage == "replay":
        from handoff.replay import replay
        m = replay(take_dir)
        checks = ", ".join(f"{c['type']}={'pass' if c['passed'] else 'fail'}" for c in m["success"])
        print(f"[replay] wrote {out} ({checks})")
    else:
        print(f"[{stage}] take={take_dir} video={video} force={force} -- not yet implemented")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take", help="take directory, e.g. takes/mytake")
    parser.add_argument("--video", help="source video for the perception stage")
    parser.add_argument("--from", dest="from_stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to", dest="to_stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--force", action="store_true", help="rerun stages even if outputs exist")
    args = parser.parse_args()

    start = STAGES.index(args.from_stage)
    end = STAGES.index(args.to_stage)
    if start > end:
        parser.error(f"--from {args.from_stage} comes after --to {args.to_stage}")

    for stage in STAGES[start:end + 1]:
        run_stage(stage, args.take, args.video, args.force)


if __name__ == "__main__":
    main()
