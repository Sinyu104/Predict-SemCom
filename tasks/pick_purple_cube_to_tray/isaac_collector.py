"""
tasks/pick_purple_cube_to_tray/isaac_collector.py
Entry point: pick up the purple cube and place it on the tray.
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
from isaac_sim.base_scenes import PickAndPlaceScene
from isaac_sim.collector_runner import collect, parse_collector_args
import numpy as np


class PickPurpleCubeToTrayScene(PickAndPlaceScene):
    def __init__(self, **kwargs):
        super().__init__(goal_type="tray", **kwargs)
        self._target_cube = self.cubes["purple"]

    def scripted_action(self) -> np.ndarray:
        return self._rmpflow_pick_place_action(self.cubes["purple"], self.goal_pos)


if __name__ == "__main__":
    args = parse_collector_args(
        default_output      = "data/pick_purple_cube_to_tray/demos.hdf5",
        default_instruction = "pick up the purple cube and place it on the tray",
    )
    scene = PickPurpleCubeToTrayScene(camera_ids=args.camera)
    collect(scene, args)
    simulation_app.close()
