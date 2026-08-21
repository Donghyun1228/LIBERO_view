"""
Re-render OpenVLA-style LIBERO HDF5 trajectories with LIBERO_view cameras.

This script expects raw LIBERO HDF5 files with `data/demo_*/states` plus actions
and robot observations. It replays each demo, applies OpenVLA's no-op filtering,
keeps successful replays, and writes view-specific RLDS/TFDS directly. It can
also replay inverse-rotated actions in a robot-base-yaw action-frame shift. The
public `openvla/modified_libero_rlds` dataset is already rendered RLDS and does
not contain MuJoCo simulator states, so it cannot be used as the input.

Output layout intentionally keeps the original OpenVLA dataset names under
separate view roots:

    <tfds-output-root>/original/libero_10_no_noops/1.0.0/...
    <tfds-output-root>/small/libero_10_no_noops/1.0.0/...
    <tfds-output-root>/medium/libero_10_no_noops/1.0.0/...
    <tfds-output-root>/large/libero_10_no_noops/1.0.0/...

That lets the existing OpenPI LeRobot converter read each view by changing only
`--data_dir`.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import h5py
import init_path  # noqa: F401
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import robosuite.utils.transform_utils as transform_utils

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
VIEWS = {
    "original": "agentview",
    "small": "agentview_small",
    "medium": "agentview_medium",
    "large": "agentview_large",
}
VERSION = "1.0.0"


@dataclass(frozen=True)
class TaskHdf5:
    suite: str
    task_id: int
    task_name: str
    description: str
    bddl_file: str
    hdf5_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-hdf5-root",
        "--openvla-hdf5-root",
        dest="libero_hdf5_root",
        type=Path,
        required=True,
        help=(
            "Root containing raw LIBERO HDF5 files. Expected layouts include "
            "<root>/<suite>/*_demo.hdf5 or <root>/<suite>_no_noops/*_demo.hdf5."
        ),
    )
    parser.add_argument(
        "--tfds-output-root",
        type=Path,
        required=True,
        help="Output root for view-specific prepared TFDS/RLDS datasets.",
    )
    parser.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=tuple(VIEWS))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--max-tasks-per-suite", type=int, default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument(
        "--robot-base-yaw-deg",
        type=float,
        default=0.0,
        help="Robot-base yaw used by the shifted environment, in degrees.",
    )
    parser.add_argument(
        "--compensate-robot-base-yaw",
        action="store_true",
        help="Compensate the restored robot root joint after applying robot-base yaw.",
    )
    parser.add_argument(
        "--rotate-actions-with-robot-base-yaw",
        action="store_true",
        help="Let the environment rotate policy-facing actions by robot-base yaw.",
    )
    parser.add_argument(
        "--replay-action-yaw-deg",
        type=float,
        default=0.0,
        help=(
            "Yaw applied to each source action before it is saved and passed to the "
            "environment. Use the inverse of --robot-base-yaw-deg to collect successful "
            "policy-facing demonstrations in the shifted action frame."
        ),
    )
    parser.add_argument("--noop-threshold", type=float, default=1e-4)
    parser.add_argument(
        "--skip-noop-filter",
        action="store_true",
        help="Keep no-op actions instead of applying OpenVLA's no-op filter.",
    )
    parser.add_argument(
        "--keep-unsuccessful",
        action="store_true",
        help="Keep replayed demos even when the final LIBERO done flag is false.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-rotate-180",
        action="store_true",
        help="Disable the 180 degree image rotation used by OpenVLA's LIBERO RLDS conversion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate HDF5 discovery/schema only; do not render or write TFDS.",
    )
    return parser.parse_args()


def reject_rlds_input(root: Path) -> None:
    rlds_markers = list(root.glob("*/1.0.0/features.json")) + list(
        root.glob("*/1.0.0/dataset_info.json")
    )
    if rlds_markers:
        marker = rlds_markers[0]
        raise SystemExit(
            f"{root} looks like prepared RLDS/TFDS ({marker}). Re-rendering new "
            "camera views requires raw LIBERO HDF5 files with data/demo_*/states."
        )


def suite_dataset_name(suite: str) -> str:
    return f"{suite}_no_noops"


def task_hdf5_candidates(
    root: Path,
    suite: str,
    task_name: str,
    problem_folder: str,
) -> list[Path]:
    file_name = f"{task_name}_demo.hdf5"
    return [
        root / suite_dataset_name(suite) / file_name,
        root / f"{suite}_regen" / file_name,
        root / suite / file_name,
        root / problem_folder / file_name,
        root / file_name,
    ]


def find_task_hdf5(root: Path, suite: str, task: Any) -> Path:
    for candidate in task_hdf5_candidates(root, suite, task.name, task.problem_folder):
        if candidate.exists():
            return candidate
    candidates = "\n".join(str(p) for p in task_hdf5_candidates(root, suite, task.name, task.problem_folder))
    raise FileNotFoundError(f"Missing HDF5 for task {suite}/{task.name}. Tried:\n{candidates}")


def require_demo_schema(path: Path) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"{path} is not a LIBERO HDF5: missing group 'data'")
        demos = sorted(handle["data"].keys(), key=lambda name: int(name.split("_")[-1]))
        if not demos:
            raise ValueError(f"{path} has no demos under group 'data'")
        total_steps = 0
        for demo_name in demos:
            demo = handle["data"][demo_name]
            for key in ("states", "actions", "obs"):
                if key not in demo:
                    raise ValueError(f"{path}:{demo_name} missing '{key}'")
            obs = demo["obs"]
            for key in ("ee_states", "gripper_states", "joint_states"):
                if key not in obs:
                    raise ValueError(f"{path}:{demo_name}/obs missing '{key}'")
            n_actions = len(demo["actions"])
            if len(demo["states"]) != n_actions:
                raise ValueError(
                    f"{path}:{demo_name} length mismatch: "
                    f"states={len(demo['states'])}, actions={n_actions}"
                )
            total_steps += n_actions
        return len(demos), total_steps


def collect_tasks(root: Path, suites: list[str], max_tasks_per_suite: int | None) -> list[TaskHdf5]:
    benchmark_dict = benchmark.get_benchmark_dict()
    tasks: list[TaskHdf5] = []
    for suite in suites:
        task_suite = benchmark_dict[suite]()
        suite_tasks = range(task_suite.n_tasks)
        if max_tasks_per_suite is not None:
            suite_tasks = range(min(max_tasks_per_suite, task_suite.n_tasks))
        for task_id in suite_tasks:
            task = task_suite.get_task(task_id)
            hdf5_path = find_task_hdf5(root, suite, task)
            require_demo_schema(hdf5_path)
            tasks.append(
                TaskHdf5(
                    suite=suite,
                    task_id=task_id,
                    task_name=task.name,
                    description=task.language,
                    bddl_file=task_suite.get_task_bddl_file_path(task_id),
                    hdf5_path=hdf5_path,
                )
            )
    return tasks


def make_env(
    task: TaskHdf5,
    camera_name: str,
    image_size: int,
    render_gpu_device_id: int,
    *,
    robot_base_yaw_deg: float,
    compensate_robot_base_yaw: bool,
    rotate_actions_with_robot_base_yaw: bool,
) -> OffScreenRenderEnv:
    env_args = {
        "bddl_file_name": task.bddl_file,
        "camera_heights": image_size,
        "camera_widths": image_size,
        "camera_names": [camera_name, "robot0_eye_in_hand"],
        "render_gpu_device_id": render_gpu_device_id,
        "robot_base_yaw": np.deg2rad(robot_base_yaw_deg),
        "compensate_robot_base_yaw": compensate_robot_base_yaw,
        "rotate_actions_with_robot_base_yaw": rotate_actions_with_robot_base_yaw,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


def dummy_action() -> list[float]:
    return [0, 0, 0, 0, 0, 0, -1]


def rotate_pose_action_yaw(action: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate translation and axis-angle components while preserving gripper action."""
    transformed_action = np.asarray(action, dtype=np.float32).copy()
    if transformed_action.ndim != 1 or transformed_action.shape[0] < 6:
        raise ValueError(
            "Expected a one-dimensional pose action with at least 6 values, "
            f"got shape {transformed_action.shape}"
        )
    if yaw_deg == 0.0:
        return transformed_action

    yaw_rad = np.deg2rad(yaw_deg)
    cosine = np.cos(yaw_rad)
    sine = np.sin(yaw_rad)
    yaw_rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    transformed_action[:3] = yaw_rotation @ transformed_action[:3]
    transformed_action[3:6] = yaw_rotation @ transformed_action[3:6]
    return transformed_action


def is_noop(action: np.ndarray, prev_action: np.ndarray | None, threshold: float) -> bool:
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]


def image_from_obs(obs: dict[str, Any], key: str, *, rotate_180: bool) -> np.ndarray:
    image = obs[key]
    if rotate_180:
        image = image[::-1, ::-1]
    return np.asarray(image, dtype=np.uint8)


def state_from_env_obs(obs: dict[str, Any]) -> np.ndarray:
    ee_state = np.hstack(
        (
            obs["robot0_eef_pos"],
            transform_utils.quat2axisangle(obs["robot0_eef_quat"]),
        )
    )
    gripper_state = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    return np.concatenate([ee_state, gripper_state], axis=0).astype(np.float32)


def joint_state_from_env_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(obs["robot0_joint_pos"], dtype=np.float32)


def iter_task_examples(
    task: TaskHdf5,
    *,
    view_key: str,
    image_size: int,
    render_gpu_device_id: int,
    max_episodes_per_task: int | None,
    rotate_180: bool,
    filter_noops: bool,
    noop_threshold: float,
    keep_unsuccessful: bool,
    settle_steps: int,
    robot_base_yaw_deg: float,
    compensate_robot_base_yaw: bool,
    rotate_actions_with_robot_base_yaw: bool,
    replay_action_yaw_deg: float,
) -> Iterator[tuple[str, dict[str, Any]]]:
    camera_name = VIEWS[view_key]
    env = make_env(
        task,
        camera_name,
        image_size,
        render_gpu_device_id,
        robot_base_yaw_deg=robot_base_yaw_deg,
        compensate_robot_base_yaw=compensate_robot_base_yaw,
        rotate_actions_with_robot_base_yaw=rotate_actions_with_robot_base_yaw,
    )
    try:
        with h5py.File(task.hdf5_path, "r") as handle:
            demos = sorted(handle["data"].keys(), key=lambda name: int(name.split("_")[-1]))
            if max_episodes_per_task is not None:
                demos = demos[:max_episodes_per_task]

            for demo_name in demos:
                demo = handle["data"][demo_name]
                states = demo["states"][()]
                actions = demo["actions"][()]
                env.reset()
                obs = env.set_init_state(states[0])
                done = False
                for _ in range(settle_steps):
                    obs, _, step_done, _ = env.step(dummy_action())
                    done = done or step_done

                steps = []
                kept_actions: list[np.ndarray] = []
                for raw_action in actions:
                    action = rotate_pose_action_yaw(raw_action, replay_action_yaw_deg)
                    prev_action = kept_actions[-1] if kept_actions else None
                    if filter_noops and is_noop(action, prev_action, noop_threshold):
                        # No-op filtering changes the saved dataset, not the physical
                        # replay. Execute every source action so timing stays faithful.
                        obs, _, step_done, _ = env.step(action.tolist())
                        done = done or step_done
                        continue

                    steps.append(
                        {
                            "observation": {
                                "image": image_from_obs(
                                    obs,
                                    f"{camera_name}_image",
                                    rotate_180=rotate_180,
                                ),
                                "wrist_image": image_from_obs(
                                    obs,
                                    "robot0_eye_in_hand_image",
                                    rotate_180=rotate_180,
                                ),
                                "state": state_from_env_obs(obs),
                                "joint_state": joint_state_from_env_obs(obs),
                            },
                            "action": action,
                            "discount": np.float32(1.0),
                            "reward": np.float32(0.0),
                            "is_first": len(steps) == 0,
                            "is_last": False,
                            "is_terminal": False,
                            "language_instruction": task.description,
                        }
                    )
                    kept_actions.append(action)
                    obs, _, step_done, _ = env.step(action.tolist())
                    done = done or step_done

                if not steps:
                    print(f"[skip] {task.suite}/{task.task_name}/{demo_name}: no kept actions")
                    continue

                if not done and not keep_unsuccessful:
                    print(f"[skip] {task.suite}/{task.task_name}/{demo_name}: replay unsuccessful")
                    continue

                steps[-1]["reward"] = np.float32(1.0 if done else 0.0)
                steps[-1]["is_last"] = True
                steps[-1]["is_terminal"] = bool(done)

                key = f"{task.suite}_{task.task_id:02d}_{demo_name}"
                yield key, {
                    "steps": steps,
                    "episode_metadata": {
                        "file_path": str(task.hdf5_path),
                        "robot_base_yaw_deg": np.float32(robot_base_yaw_deg),
                        "replay_action_yaw_deg": np.float32(replay_action_yaw_deg),
                        "compensate_robot_base_yaw": compensate_robot_base_yaw,
                        "rotate_actions_with_robot_base_yaw": rotate_actions_with_robot_base_yaw,
                    },
                }
    finally:
        env.close()


def iter_suite_examples(
    tasks: list[TaskHdf5],
    suite: str,
    *,
    view_key: str,
    image_size: int,
    render_gpu_device_id: int,
    max_episodes_per_task: int | None,
    rotate_180: bool,
    filter_noops: bool,
    noop_threshold: float,
    keep_unsuccessful: bool,
    settle_steps: int,
    robot_base_yaw_deg: float,
    compensate_robot_base_yaw: bool,
    rotate_actions_with_robot_base_yaw: bool,
    replay_action_yaw_deg: float,
) -> Iterator[tuple[str, dict[str, Any]]]:
    for task in tasks:
        if task.suite != suite:
            continue
        yield from iter_task_examples(
            task,
            view_key=view_key,
            image_size=image_size,
            render_gpu_device_id=render_gpu_device_id,
            max_episodes_per_task=max_episodes_per_task,
            rotate_180=rotate_180,
            filter_noops=filter_noops,
            noop_threshold=noop_threshold,
            keep_unsuccessful=keep_unsuccessful,
            settle_steps=settle_steps,
            robot_base_yaw_deg=robot_base_yaw_deg,
            compensate_robot_base_yaw=compensate_robot_base_yaw,
            rotate_actions_with_robot_base_yaw=rotate_actions_with_robot_base_yaw,
            replay_action_yaw_deg=replay_action_yaw_deg,
        )


def make_features(image_size: int):
    import tensorflow_datasets as tfds

    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            "image": tfds.features.Image(
                                shape=(image_size, image_size, 3),
                                dtype=np.uint8,
                                encoding_format="jpeg",
                            ),
                            "wrist_image": tfds.features.Image(
                                shape=(image_size, image_size, 3),
                                dtype=np.uint8,
                                encoding_format="jpeg",
                            ),
                            "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                            "joint_state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                        }
                    ),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }
            ),
            "episode_metadata": tfds.features.FeaturesDict(
                {
                    "file_path": tfds.features.Text(),
                    "robot_base_yaw_deg": tfds.features.Scalar(dtype=np.float32),
                    "replay_action_yaw_deg": tfds.features.Scalar(dtype=np.float32),
                    "compensate_robot_base_yaw": tfds.features.Scalar(dtype=np.bool_),
                    "rotate_actions_with_robot_base_yaw": tfds.features.Scalar(dtype=np.bool_),
                }
            ),
        }
    )


def write_tfds_dataset(
    *,
    name: str,
    view_output_root: Path,
    examples: Iterator[tuple[str, dict[str, Any]]],
    image_size: int,
    overwrite: bool,
) -> None:
    import tensorflow_datasets as tfds

    dataset_dir = view_output_root / name / VERSION
    if dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{dataset_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(view_output_root / name)

    view_output_root.mkdir(parents=True, exist_ok=True)
    tfds.dataset_builders.store_as_tfds_dataset(
        name=name,
        version=VERSION,
        features=make_features(image_size),
        split_datasets={"train": examples},
        data_dir=view_output_root,
        description=(
            "LIBERO_view re-rendering of raw LIBERO HDF5 trajectories with OpenVLA "
            "no-op filtering and optional robot-base-yaw action-frame replay."
        ),
        homepage="https://github.com/openvla/openvla",
        disable_shuffling=True,
    )


def main() -> None:
    args = parse_args()
    hdf5_root = args.libero_hdf5_root.expanduser().resolve()
    output_root = args.tfds_output_root.expanduser().resolve()
    if not hdf5_root.exists():
        raise SystemExit(f"Missing --libero-hdf5-root: {hdf5_root}")
    for name, value in (
        ("robot_base_yaw_deg", args.robot_base_yaw_deg),
        ("replay_action_yaw_deg", args.replay_action_yaw_deg),
    ):
        if not np.isfinite(value):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite; got {value}")
    reject_rlds_input(hdf5_root)

    if args.rotate_actions_with_robot_base_yaw:
        net_action_yaw_deg = args.robot_base_yaw_deg + args.replay_action_yaw_deg
        print(
            "[action-frame] "
            f"environment_yaw={args.robot_base_yaw_deg:g}deg "
            f"replay_yaw={args.replay_action_yaw_deg:g}deg "
            f"net_controller_yaw={net_action_yaw_deg:g}deg "
            f"compensate={args.compensate_robot_base_yaw}"
        )

    tasks = collect_tasks(hdf5_root, args.suites, args.max_tasks_per_suite)
    totals: dict[str, tuple[int, int]] = {}
    for suite in args.suites:
        suite_tasks = [task for task in tasks if task.suite == suite]
        demos = 0
        steps = 0
        for task in suite_tasks:
            task_demos, task_steps = require_demo_schema(task.hdf5_path)
            if args.max_episodes_per_task is None:
                demos += task_demos
                steps += task_steps
            else:
                selected = min(task_demos, args.max_episodes_per_task)
                demos += selected
                with h5py.File(task.hdf5_path, "r") as handle:
                    demo_names = sorted(handle["data"].keys(), key=lambda name: int(name.split("_")[-1]))
                    for demo_name in demo_names[:selected]:
                        steps += len(handle["data"][demo_name]["actions"])
        totals[suite] = (len(suite_tasks), demos)
        print(f"[validate] {suite}: tasks={len(suite_tasks)} demos={demos} source_steps={steps}")

    if args.dry_run:
        print("[dry-run] HDF5 discovery/schema validation passed.")
        return

    rotate_180 = not args.no_rotate_180
    for view_key in args.views:
        view_output_root = output_root / view_key
        for suite in args.suites:
            print(f"[write] view={view_key} dataset={suite_dataset_name(suite)}")
            write_tfds_dataset(
                name=suite_dataset_name(suite),
                view_output_root=view_output_root,
                examples=iter_suite_examples(
                    tasks,
                    suite,
                    view_key=view_key,
                    image_size=args.image_size,
                    render_gpu_device_id=args.render_gpu_device_id,
                    max_episodes_per_task=args.max_episodes_per_task,
                    rotate_180=rotate_180,
                    filter_noops=not args.skip_noop_filter,
                    noop_threshold=args.noop_threshold,
                    keep_unsuccessful=args.keep_unsuccessful,
                    settle_steps=args.settle_steps,
                    robot_base_yaw_deg=args.robot_base_yaw_deg,
                    compensate_robot_base_yaw=args.compensate_robot_base_yaw,
                    rotate_actions_with_robot_base_yaw=args.rotate_actions_with_robot_base_yaw,
                    replay_action_yaw_deg=args.replay_action_yaw_deg,
                ),
                image_size=args.image_size,
                overwrite=args.overwrite,
            )
    print(f"[done] wrote view-specific TFDS datasets under {output_root}")


if __name__ == "__main__":
    main()
