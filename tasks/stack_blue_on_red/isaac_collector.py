"""
tasks/stack_blue_on_red/isaac_collector.py
Entry point: stack the blue cube on top of the red cube.

The red cube is a FixedCuboid (cannot be knocked over).
The scripted policy uses the same RMPFlow pick-and-place state machine,
targeting the top surface of the red cube as the goal.
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


class StackBlueOnRedScene(StackScene):
    def __init__(self, **kwargs):
        super().__init__(target_color="blue", base_color="red", **kwargs)

    def scripted_action(self) -> np.ndarray:
        return self._rmpflow_pick_place_action(
            self.cubes["blue"], self._stack_goal_pos
        )


if __name__ == "__main__":
    args = parse_collector_args(
        default_output      = "data/stack_blue_on_red/demos.hdf5",
        default_instruction = "stack the blue cube on top of the red cube",
    )
    scene = StackBlueOnRedScene(camera_ids=args.camera)
    collect(scene, args)
    simulation_app.close()
