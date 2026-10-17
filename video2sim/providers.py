"""Swappable vision-language model backends.

Every provider takes the same input: a text prompt, labelled JPEG images and
(optionally) the source video, and returns the model's text reply.

    get_provider("gemini-4-argon")   -> Gemini (GEMINI_API_KEY or GOOGLE_API_KEY)
    get_provider("claude-opus-5-5")  -> Anthropic (ANTHROPIC_API_KEY)
    get_provider("gpt-6.1-sol")      -> OpenAI (OPENAI_API_KEY)

Keys come from the environment or from video2sim/.env.
    get_provider("mock")             -> canned answers, no network (pipeline testing)
"""
import base64
import json
import os
import time
from pathlib import Path


def load_env(path=Path(__file__).parent / ".env"):
    """Read KEY=value lines from video2sim/.env into the environment (real env vars win)."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())


load_env()


class Provider:
    native_video = False  # True: the raw video is sent alongside the frames

    def __init__(self, model):
        self.model = model

    def generate(self, prompt, images, video_path=None):
        raise NotImplementedError


class GeminiProvider(Provider):
    native_video = True

    def __init__(self, model):
        super().__init__(model)
        from google import genai
        self.genai = genai
        self.client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        self._video = {}

    def _upload(self, path):
        if path not in self._video:
            f = self.client.files.upload(file=path)
            while f.state.name == "PROCESSING":
                time.sleep(2)
                f = self.client.files.get(name=f.name)
            if f.state.name != "ACTIVE":
                raise RuntimeError(f"Gemini could not process the video: {f.state}")
            self._video[path] = f
        return self._video[path]

    def generate(self, prompt, images, video_path=None):
        from google.genai import types
        parts = []
        if video_path:
            parts += ["Source video:", self._upload(video_path)]
        for label, jpg in images:
            parts += [label, types.Part.from_bytes(data=jpg, mime_type="image/jpeg")]
        parts.append(prompt)
        resp = self.client.models.generate_content(model=self.model, contents=parts)
        return resp.text


class ClaudeProvider(Provider):
    # Models that accept the server-side refusal fallback.
    FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}

    def __init__(self, model):
        super().__init__(model)
        import anthropic
        self.client = anthropic.Anthropic()

    def generate(self, prompt, images, video_path=None):
        content = []
        for label, jpg in images:
            content += [{"type": "text", "text": label},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": base64.standard_b64encode(jpg).decode()}}]
        content.append({"type": "text", "text": prompt})
        kwargs = dict(model=self.model, max_tokens=64000, thinking={"type": "adaptive"},
                      output_config={"effort": "high"},
                      messages=[{"role": "user", "content": content}])
        if self.model in self.FALLBACK_MODELS:
            stream = self.client.beta.messages.stream(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            stream = self.client.messages.stream(**kwargs)
        with stream as s:
            msg = s.get_final_message()
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the request: {msg.stop_details}")
        return "".join(b.text for b in msg.content if b.type == "text")


class OpenAIProvider(Provider):
    def __init__(self, model):
        super().__init__(model)
        from openai import OpenAI
        self.client = OpenAI()

    def generate(self, prompt, images, video_path=None):
        content = []
        for label, jpg in images:
            content += [{"type": "input_text", "text": label},
                        {"type": "input_image",
                         "image_url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}]
        content.append({"type": "input_text", "text": prompt})
        resp = self.client.responses.create(model=self.model, input=[{"role": "user", "content": content}])
        return resp.output_text


class MockProvider(Provider):
    """Canned pick-and-place answers so the loop can be tested offline."""

    def __init__(self, model, robot="panda"):
        super().__init__(model)
        self.robot = robot
        self.calls = 0

    def generate(self, prompt, images, video_path=None):
        self.calls += 1
        spec = json.loads(json.dumps(MOCK_SPECS[self.robot]))
        spec["done"] = self.calls > 1
        spec["changes"] = "initial guess" if self.calls == 1 else "no changes needed"
        return "```json\n" + json.dumps(spec, indent=1) + "\n```"


def _step(action, language, reason, **kw):
    return {"action": action, "language": language, "reason": reason, **kw}


MOCK_SPECS = {
    "panda": {
        "task_description": "Pick up the red cube and drop it into the blue bin.",
        "reasoning": "mock",
        "camera": {"lookat": [0.5, 0.0, 0.05], "distance": 1.5, "azimuth": 160, "elevation": -30},
        "objects": [
            {"name": "red_cube", "shape": "box", "size": [0.025, 0.025, 0.025], "pos": [0.5, -0.15, 0.025],
             "rgba": [0.9, 0.15, 0.1, 1], "mass": 0.05},
            {"name": "blue_bin", "shape": "container", "size": [0.08, 0.08, 0.04], "pos": [0.5, 0.2, 0.04],
             "rgba": [0.15, 0.35, 0.9, 1], "mass": 0.3},
        ],
        "plan": [
            _step("move_to", "move above the cube", "approach from above so the fingers clear the cube", pos=[0.5, -0.15, 0.2], yaw=0),
            _step("move_by", "move down 17 cm", "bring the finger pads level with the cube centre", delta_cm=[0, 0, -17]),
            _step("close", "close gripper", "squeeze the cube between the pads"),
            _step("move_by", "move up 20 cm", "lift clear of the table before transporting", delta_cm=[0, 0, 20]),
            _step("move_to", "move above the bin", "carry the cube over the bin opening", pos=[0.5, 0.2, 0.23]),
            _step("move_by", "move down 8 cm", "lower so the cube drops only a short distance", delta_cm=[0, 0, -8]),
            _step("open", "open gripper", "release the cube into the bin"),
            _step("move_by", "move up 10 cm", "retreat without hitting the bin walls", delta_cm=[0, 0, 10]),
        ],
        "success": [{"type": "object_on", "object": "red_cube", "support": "blue_bin"},
                    {"type": "object_lifted", "object": "red_cube", "min_height": 0.1}],
    },
    "qarm": {
        "task_description": "Pick up the red cube from the left stand and place it on the right stand.",
        "reasoning": "mock",
        "camera": {"lookat": [0.3, 0.0, 0.15], "distance": 1.3, "azimuth": 150, "elevation": -25},
        "objects": [
            {"name": "stand_a", "shape": "box", "size": [0.04, 0.04, 0.1], "pos": [0.38, 0.0, 0.1],
             "rgba": [0.9, 0.55, 0.15, 1], "fixed": True},
            {"name": "stand_b", "shape": "box", "size": [0.04, 0.04, 0.1], "pos": [0.25, 0.28, 0.1],
             "rgba": [0.2, 0.7, 0.3, 1], "fixed": True},
            {"name": "red_cube", "shape": "box", "size": [0.02, 0.02, 0.02], "pos": [0.38, 0.0, 0.22],
             "rgba": [0.9, 0.15, 0.1, 1], "mass": 0.03},
        ],
        "plan": [
            _step("move_to", "move behind the cube", "line the open fingers up with the cube", pos=[0.32, 0.0, 0.25]),
            _step("move_to", "move forward to the cube", "slide the fingers around the cube", pos=[0.38, 0.0, 0.225]),
            _step("close", "close gripper", "pinch the cube"),
            _step("move_by", "move up 8 cm", "lift off the stand", delta_cm=[0, 0, 8]),
            _step("move_to", "move above stand b", "carry the cube to the second stand", pos=[0.25, 0.28, 0.3]),
            _step("move_by", "move down 7 cm", "set the cube onto the stand", delta_cm=[0, 0, -7]),
            _step("open", "open gripper", "release"),
            _step("move_by", "move up 8 cm", "retreat", delta_cm=[0, 0, 8]),
        ],
        "success": [{"type": "object_on", "object": "red_cube", "support": "stand_b"}],
    },
}


# Environment variable each provider reads its API key from.
KEY_ENV = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def resolve_provider(model, provider="auto"):
    if provider != "auto":
        return provider
    if model == "mock":
        return "mock"
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt", "o1", "o3", "o4", "o5")):
        return "openai"
    raise ValueError(f"Can't infer provider for {model!r}; pass --provider")


def get_provider(model, provider="auto", robot="panda"):
    provider = resolve_provider(model, provider)
    return {"gemini": GeminiProvider, "anthropic": ClaudeProvider, "openai": OpenAIProvider,
            "mock": lambda m: MockProvider(m, robot)}[provider](model)
