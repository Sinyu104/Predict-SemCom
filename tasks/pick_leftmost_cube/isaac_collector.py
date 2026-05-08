"""
tasks/pick_leftmost_cube/isaac_collector.py
Entry point: pick up the leftmost cube and place it on the tray.

"Leftmost" is determined dynamically at each reset() by finding the cube with
the largest Y coordinate in world space, which corresponds to the leftmost
position from the camera's perspective.  The identity is locked in for the
duration of the episode so scripted_action() and is_success() target the same
cube throughout.
"""

import argparse, os, sys
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


class PickLeftmostCubeScene(PickAndPlaceScene):
    """
    Identifies the leftmost cube (largest Y in world space) at reset time
    and stores it as self._target_cube for the episode.
    """

    def __init__(self):
        super().__init__(goal_type="tray")
        # _target_cube will be set in reset()

    def reset(self, randomise: bool = True):
        super().reset(randomise)
        # Identify leftmost cube: largest Y value = leftmost from camera view.
        # Locked in here so both scripted_action() and is_success() use the
        # same cube for the entire episode.
        max_y = -float("inf")
        self._target_cube = None
        for cube in self.cubes.values():
            pos, _ = cube.get_world_pose()
            if pos[1] > max_y:
                max_y = pos[1]
                self._target_cube = cube

    def scripted_action(self) -> np.ndarray:
        if self._target_cube is None:
            return np.zeros(7, dtype=np.float32)
        return self._rmpflow_pick_place_action(self._target_cube, self.goal_pos)


if __name__ == "__main__":
    args = parse_collector_args(
        default_output      = "data/pick_leftmost_cube/demos.hdf5",
        default_instruction = "pick up the leftmost cube and place it on the tray",
    )
    scene = PickLeftmostCubeScene()
    collect(scene, args)
    simulation_app.close()
