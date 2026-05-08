"""
tasks/sort_two_cubes/isaac_collector.py
Entry point: sort the red and blue cubes into their matching trays.

The scene has a red tray and a blue tray.  The scripted policy performs two
sequential pick-and-place operations:
  1. Pick the red cube → red tray  (phases 0-5 of the shared state machine)
  2. Pick the blue cube → blue tray (phases 0-5, second pass)

self._sort_phase tracks which operation is active (0 = red, 1 = blue).
is_success() requires BOTH cubes on their correct trays.

Note: double the default episode_length (240) to allow both operations to
complete within one episode.
"""

import argparse, os, sys
import h5py

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--headless", action="store_true")
_pre_args, _ = _pre.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": _pre_args.headless, "renderer": "RayTracedLighting", "anti_aliasing": 0})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from isaac_sim.base_scenes import SortScene
from isaac_sim.collector_runner import collect, parse_collector_args
import numpy as np


class SortTwoCubesScene(SortScene):
    def scripted_action(self) -> np.ndarray:
        if self._sort_phase == 0:
            action = self._rmpflow_pick_place_action(
                self.cubes["red"], self.RED_TRAY_POS
            )
            # When the first pick-and-place finishes (phase >= 5), start the second.
            if self._phase >= 5:
                self._sort_phase = 1
                self._phase      = 0
                self._grip_timer = 0
                self._grasp_pos  = None
        else:
            action = self._rmpflow_pick_place_action(
                self.cubes["blue"], self.BLUE_TRAY_POS
            )
        return action


if __name__ == "__main__":
    args = parse_collector_args(
        default_output      = "data/sort_two_cubes/demos.hdf5",
        default_instruction = "sort the red and blue cubes into their matching trays",
    )
    # Override default episode length — two sequential pick-and-place ops needed.
    if args.episode_length == 120:
        args.episode_length = 240
    scene = SortTwoCubesScene()
    collect(scene, args)
    simulation_app.close()
