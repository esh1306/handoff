"""Smoke-tests the environment before any pipeline module is built.

Checks: mediapipe import, torch device order, Depth Anything V2 Small loads,
MuJoCo loads the Menagerie Panda scene, ArUco dictionary exists, lerobot
imports, OPENAI_API_KEY present. Prints an OK/FAIL table; exits 1 on any FAIL.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

results = []


def check(name):
    def decorator(fn):
        try:
            detail = fn()
            results.append((name, True, detail or "ok"))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return decorator


@check("mediapipe import")
def _():
    import mediapipe as mp
    return f"mediapipe {mp.__version__}"


@check("torch device (cuda > mps > cpu)")
def _():
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return f"torch {torch.__version__}, device={device}"


@check("ArUco dictionary (DICT_4X4_50)")
def _():
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return f"{type(d).__name__} loaded"


@check("lerobot import")
def _():
    import lerobot
    return f"lerobot {getattr(lerobot, '__version__', 'unknown version')}"


@check("OPENAI_API_KEY present")
def _():
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("not set (copy .env.example to .env and fill it in)")
    return "set"


@check("MuJoCo loads Menagerie Panda scene")
def _():
    import mujoco
    scene = REPO_ROOT / "mujoco_menagerie/franka_emika_panda/scene.xml"
    if not scene.exists():
        raise FileNotFoundError(
            f"{scene} missing (git clone https://github.com/google-deepmind/mujoco_menagerie)"
        )
    model = mujoco.MjModel.from_xml_path(str(scene))
    return f"nq={model.nq} nbody={model.nbody}"


@check("Depth Anything V2 Small loads")
def _():
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    return f"loaded on {device}"


def main():
    name_w = max(len(n) for n, _, _ in results)
    print(f"{'CHECK':{name_w}}  STATUS  DETAIL")
    print("-" * (name_w + 60))
    n_fail = 0
    for name, ok, detail in results:
        status = "OK" if ok else "FAIL"
        n_fail += 0 if ok else 1
        print(f"{name:{name_w}}  {status:6}  {detail}")
    print("-" * (name_w + 60))
    print(f"{len(results) - n_fail}/{len(results)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
