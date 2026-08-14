"""Two-model voice-controlled runner for SO-ARM 101 + SmolVLA.

Keeps a per-color SmolVLA policy resident in VRAM and switches which one drives
the arm based on the live task written by voice_task.py to current_task.txt:

  "grab the red ball ..."   -> red policy
  "grab the black ball ..." -> black policy
  __STOP__     -> ease the arm back to its home pose and wait (say a color to resume)
  __COMPLETE__ -> shut down cleanly

This is a pure inference/control loop: no dataset is recorded. It reuses the exact
lerobot building blocks that lerobot-record uses for observation/action framing and
inference, so per-model behavior matches a normal `lerobot-record --policy.path=...`
run. Robot arguments are parsed the same way too, e.g.:

  python voice_run.py \
    --robot.type=so101_follower --robot.port=COM3 --robot.id=ty_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --red_path=outputs/train/smolvla_grab_the_redball/checkpoints/030000/pretrained_model \
    --black_path=outputs/train/smolvla_grab_the_blackball/checkpoints/030000/pretrained_model \
    --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.top": "observation.images.camera2"}'
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401  (registers "opencv")
from lerobot.configs import parser
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so_follower,  # registers the "so101_follower" robot type
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging

STOP_TASK = "__STOP__"
COMPLETE_TASK = "__COMPLETE__"
HOME_ALPHA = 0.1  # fraction of remaining distance to home moved per step (smooth ease-in)


@dataclass
class VoiceRunConfig:
    robot: RobotConfig
    # Per-color checkpoint dirs (the ".../pretrained_model" folder). black may be omitted
    # until that model is trained; "black" commands then just hold at home.
    red_path: str
    black_path: str | None = None
    task_file: str = str(Path(__file__).parent / "current_task.txt")
    fps: int = 30
    device: str = "cuda"
    use_amp: bool = True
    # Rename raw camera keys to the names the policy was trained on (front->camera1, top->camera2).
    rename_map: dict[str, str] = field(default_factory=dict)


def _repair_rename_map(rm: dict[str, str]) -> dict[str, str]:
    # PowerShell strips JSON quotes, so draccus parses '{"a":"b"}' as {'a:b': 'None'}. Repair it.
    if rm and all(v in (None, "None") for v in rm.values()) and any(":" in k for k in rm):
        return dict(k.split(":", 1) for k in rm if ":" in k)
    return rm


def _read_task(task_file: str) -> str | None:
    try:
        return Path(task_file).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _color_of(task: str) -> str | None:
    t = task.lower()
    if "black" in t:
        return "black"
    if "red" in t:
        return "red"
    return None


def _load_model(path: str, cfg: VoiceRunConfig):
    logging.info(f"loading policy: {path}")
    policy = SmolVLAPolicy.from_pretrained(path)
    policy.to(cfg.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=path,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    return policy, preprocessor, postprocessor


@parser.wrap()
def voice_run(cfg: VoiceRunConfig):
    init_logging()
    cfg.rename_map = _repair_rename_map(cfg.rename_map)
    device = get_safe_torch_device(cfg.device)

    # Clear any stale command left over from a previous session so we don't act on an
    # old __COMPLETE__/__STOP__/color the moment we start. Wait for a fresh utterance.
    Path(cfg.task_file).write_text("", encoding="utf-8")

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    _, robot_action_processor, robot_observation_processor = make_default_processors()

    # Same feature construction lerobot-record uses, but without creating a dataset. Serves both
    # build_dataset_frame (observation) and make_robot_action (action names/order).
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    models = {"red": _load_model(cfg.red_path, cfg)}
    if cfg.black_path:
        models["black"] = _load_model(cfg.black_path, cfg)
    else:
        logging.warning("black_path not set - 'black' commands will hold at home until it is provided.")

    home_pose = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
    active = None  # currently driving color, or None while idle
    idle = False
    logging.info("voice runner ready. say a color to act, 'stop' to hold, 'complete' to quit.")

    try:
        while True:
            t0 = time.perf_counter()
            task = _read_task(cfg.task_file)

            if task == COMPLETE_TASK:
                logging.info("COMPLETE received - shutting down.")
                break

            color = _color_of(task) if task and task != STOP_TASK else None

            # Stop, nothing yet, or a color with no loaded model -> ease toward home and wait.
            if color is None or color not in models:
                if not idle:
                    if task and task != STOP_TASK:
                        logging.warning(f"no model for task {task!r} - holding at home.")
                    else:
                        logging.info("STOP - returning to home pose and waiting.")
                    idle = True
                    active = None
                present = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
                goal = {k: present[k] + HOME_ALPHA * (home_pose[k] - present[k]) for k in home_pose if k in present}
                robot.send_action(goal)
                precise_sleep(max(1 / cfg.fps - (time.perf_counter() - t0), 0.0))
                continue

            policy, preprocessor, postprocessor = models[color]
            if color != active or idle:  # switched model or resumed from idle -> re-plan
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                active = color
                idle = False
                logging.info(f"active model: {color}")

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task,
                robot_type=robot.robot_type,
            )
            act = make_robot_action(action_values, dataset_features)
            robot.send_action(robot_action_processor((act, obs)))
            precise_sleep(max(1 / cfg.fps - (time.perf_counter() - t0), 0.0))
    finally:
        if robot.is_connected:
            robot.disconnect()
        logging.info("bye")


if __name__ == "__main__":
    voice_run()
