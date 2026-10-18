# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Load Unitree Go2 quadrupeds and hold their default joint pose.")
parser.add_argument("--config", type=Path, default=Path(__file__).with_name("landing_config.json"),
                    help="JSON configuration file (defaults to landing_config.json beside this script).")
args_cli = parser.parse_args()
try:
    with args_cli.config.open(encoding="utf-8") as config_file:
        settings = json.load(config_file)
except (OSError, ValueError) as exc:
    parser.error(f"Cannot read configuration: {exc}")
numeric_fields = {
    "num_envs", "steps", "height_min", "height_max", "stable_time", "stabilization_timeout",
    "episode_timeout", "contact_threshold", "joint_angle_tolerance_deg", "simulation_dt",
}
expected_fields = numeric_fields | {"device", "headless", "kit_args"}
if not isinstance(settings, dict) or set(settings) != expected_fields:
    parser.error(f"Configuration must contain exactly these keys: {sorted(expected_fields)}")
for key in numeric_fields:
    if type(settings[key]) not in (int, float) or not math.isfinite(settings[key]):
        parser.error(f"{key} must be a finite number")
if type(settings["num_envs"]) is not int or type(settings["steps"]) is not int:
    parser.error("num_envs and steps must be integers")
if type(settings["headless"]) is not bool:
    parser.error("headless must be a JSON boolean")
if not isinstance(settings["device"], str) or not isinstance(settings["kit_args"], str):
    parser.error("device and kit_args must be strings")
config = SimpleNamespace(**settings)
if config.num_envs < 1 or config.steps < 0:
    parser.error("num_envs must be positive and steps must be nonnegative")
if not 0 < config.height_min <= config.height_max:
    parser.error("Require 0 < height_min <= height_max; choose heights that clear the feet from the floor")
if not 0 < config.stable_time <= config.stabilization_timeout <= config.episode_timeout:
    parser.error("Require 0 < stable_time <= stabilization_timeout <= episode_timeout")
if config.contact_threshold <= 0 or config.simulation_dt <= 0 or config.joint_angle_tolerance_deg <= 0:
    parser.error("contact_threshold, simulation_dt, and joint_angle_tolerance_deg must be positive")

# launch omniverse app
app_launcher = AppLauncher(device=config.device, headless=config.headless, kit_args=config.kit_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

##
# Pre-defined configs
##
from isaaclab_assets import UNITREE_GO2_CFG  # isort:skip


@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    """Unitree Go2 robots on a flat ground plane."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=0
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Drop each robot, wait for touchdown, then time continuous four-foot contact."""
    robot = scene["robot"]
    contact_sensor = scene["contact_sensor"]
    body_ids, body_names = contact_sensor.find_bodies("base")
    foot_ids, foot_names = contact_sensor.find_bodies(".*_foot")
    leg_joint_ids, leg_joint_names = robot.find_joints([".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])
    if len(leg_joint_ids) != 12:
        raise RuntimeError(f"Expected 12 Go2 leg joints; found {leg_joint_names}")
    joint_tolerance = math.radians(config.joint_angle_tolerance_deg)
    if len(foot_ids) != 4 or len(body_ids) != 1:
        raise RuntimeError(f"Expected four feet and one base; found {foot_names=} and {body_names=}")

    RESET, FALLING, STABILIZING = 0, 1, 2
    phase = torch.full((scene.num_envs,), RESET, dtype=torch.long, device=sim.device)
    episode_time = torch.zeros(scene.num_envs, device=sim.device)
    stabilization_time = torch.zeros_like(episode_time)
    contact_time = torch.zeros_like(episode_time)
    sim_dt = sim.get_physics_dt()
    count = 0
    successes = failures = timeouts = 0

    while simulation_app.is_running() and (config.steps == 0 or count < config.steps):
        reset_ids = (phase == RESET).nonzero(as_tuple=False).flatten()
        if reset_ids.numel():
            heights = torch.empty(reset_ids.numel(), device=sim.device).uniform_(
                config.height_min, config.height_max
            )
            root_state = robot.data.default_root_state[reset_ids].clone()
            root_state[:, :3] += scene.env_origins[reset_ids]
            # Height is an absolute torso height above each environment's floor.
            root_state[:, 2] = scene.env_origins[reset_ids, 2] + heights
            root_state[:, 7:] = 0.0
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=reset_ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=reset_ids)
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos[reset_ids].clone(),
                robot.data.default_joint_vel[reset_ids].clone(),
                env_ids=reset_ids,
            )
            scene.reset(reset_ids)
            episode_time[reset_ids] = 0.0
            stabilization_time[reset_ids] = 0.0
            contact_time[reset_ids] = 0.0
            phase[reset_ids] = FALLING
            print(f"[INFO]: Reset environments {reset_ids.tolist()}, torso heights {heights.tolist()}", flush=True)

        # Keep physics advancing during both falling and stabilization.
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        episode_time += sim_dt

        # Read current forces only, after the physics step and scene update.
        forces = contact_sensor.data.net_forces_w
        foot_contact = forces[:, foot_ids, :].norm(dim=-1) > config.contact_threshold
        body_contact = (forces[:, body_ids, :].norm(dim=-1) > config.contact_threshold).any(dim=-1)
        touchdown = (phase == FALLING) & foot_contact.any(dim=-1) & ~body_contact
        # Count elapsed stabilization time only for robots already in that phase.
        stabilization_time[phase == STABILIZING] += sim_dt
        phase[touchdown] = STABILIZING
        if touchdown.any():
            print(f"[INFO]: First foot contact: {touchdown.nonzero().flatten().tolist()}", flush=True)

        # Joint positions are radians. Every leg joint must be near its own default angle.
        joint_error = (robot.data.joint_pos[:, leg_joint_ids] - robot.data.default_joint_pos[:, leg_joint_ids]).abs()
        standing_pose = (joint_error <= joint_tolerance).all(dim=-1)
        holding_contact = (phase == STABILIZING) & foot_contact.all(dim=-1) & ~body_contact & standing_pose
        contact_time = torch.where(holding_contact, contact_time + sim_dt, 0.0)
        
        # Do not credit an entire interval for the first touchdown sample.
        contact_time[touchdown] = 0.0
        timed_out = (episode_time >= config.episode_timeout) | (
            (phase == STABILIZING) & (stabilization_time >= config.stabilization_timeout)
        )
        # Failure and timeout take precedence over success on the same step.
        success = (contact_time >= config.stable_time) & holding_contact & ~timed_out
        timeout = timed_out & ~body_contact
        for label, mask in (("body-contact failure", body_contact), ("timeout", timeout), ("stable contact", success)):
            if mask.any():
                print(f"[INFO]: {label}: environments {mask.nonzero().flatten().tolist()}", flush=True)
        successes += int(success.sum().item())
        failures += int(body_contact.sum().item())
        timeouts += int(timeout.sum().item())
        phase[body_contact | timeout | success] = RESET

    print(f"[INFO]: Completed {count} quadruped simulation steps.", flush=True)
    print(f"[INFO]: Outcomes: {successes=}, {failures=}, {timeouts=}", flush=True)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=config.simulation_dt, device=config.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 1.5], [0.0, 0.0, 0.3])
    # Design scene
    scene_cfg = QuadrupedSceneCfg(num_envs=config.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    try:
        main()
    finally:
        simulation_app.close()
    
