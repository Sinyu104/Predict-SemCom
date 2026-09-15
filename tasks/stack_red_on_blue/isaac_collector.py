"""
tasks/stack_red_on_blue/isaac_collector.py
Entry point: stack the red cube on top of the blue cube.

The blue cube is a FixedCuboid (cannot be knocked over).
The scripted policy uses the same RMPFlow pick-and-place state machine,
targeting the top surface of the blue cube as the goal.
"""

import argparse, os, sys

if __name__ == "__main__":
    import h5py
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--headless", action="store_true")
    _pre_args, _ = _pre.parse_known_args()
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": _pre_args.headless, "renderer": "RayTracedLighting", "anti_aliasing": 0})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from isaac_sim.base_scenes import StackScene
from isaac_sim.collector_runner import collect, parse_collector_args
import numpy as np


class StackRedOnBlueScene(StackScene):
    def __init__(self, **kwargs):
        super().__init__(target_color="red", base_color="blue", **kwargs)

    def scripted_action(self) -> np.ndarray:
        return self._rmpflow_pick_place_action(
            self.cubes["red"], self._stack_goal_pos
        )


if __name__ == "__main__":
    args = parse_collector_args(
        default_output      = "data/stack_red_on_blue/demos.hdf5",
        default_instruction = "stack the red cube on top of the blue cube",
    )
    scene = StackRedOnBlueScene(camera_ids=args.camera)
    collect(scene, args)
    simulation_app.close()
