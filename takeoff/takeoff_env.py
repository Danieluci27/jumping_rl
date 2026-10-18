import json
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

config_path = Path(__file__).resolve().with_name("takeoff_config.json")
try:
    with config_path.open(encoding="utf-8") as config_file:
        settings = json.load(config_file)
except (OSError, ValueError) as exc:
    raise RuntimeError(f"Failed to load {config_path}: {exc}") from exc

config = SimpleNamespace(**settings)

app_launcher = AppLauncher(device=config.device, headless=config.headless, kit_args=config.kit_args)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import torch
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from takeoff_scene_config import TakeoffSceneCfg

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=config.simulation_dt, device=config.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 1.5], [0.0, 0.0, 0.3])
    # Design scene
    scene_cfg = TakeoffSceneCfg(num_envs=config.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Advance physics and hold the robot's default pose until the app closes.
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    with torch.inference_mode():
        while simulation_app.is_running():
            robot.set_joint_position_target(robot.data.default_joint_pos)
            scene.write_data_to_sim()
            sim.step()
            if not simulation_app.is_running():
                break
            scene.update(sim_dt)

if __name__ == "__main__":
    # run the main function
    try:
        main()
    finally:
        simulation_app.close()
