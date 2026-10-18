from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_assets import UNITREE_GO2_CFG
import isaaclab.terrains as terrain_gen


@configclass
class LandingSceneCfg(InteractiveSceneCfg):
    """Unitree Go2 robots on a flat ground plane."""

    # ground plane
    ground = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            seed=42,
            size=(4.0, 4.0),       # Size of each terrain patch, meters
            num_rows=4,
            num_cols=4,
            border_width=2.0,
            horizontal_scale=0.05,
            vertical_scale=0.002,
            use_cache=False,
            sub_terrains={
                "rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    noise_range=(-0.02, 0.02),  # Heights from -2 to +2 cm
                    noise_step=0.002,          # Height sampling increments
                    downsampled_scale=0.20,    # Spacing of coarse noise samples
                    border_width=0.25,
                ),
            },
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.7,
            restitution=0.0,
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=0
    )