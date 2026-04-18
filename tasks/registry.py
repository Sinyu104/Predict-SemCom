"""
tasks/registry.py  —  Central registry mapping task names to scene classes.

Call get_scene() AFTER SimulationApp has been created.
"""

TASK_NAMES = [
    "pick_red_cube_to_tray",
    "pick_blue_cube_to_tray",
    "pick_yellow_cube_to_tray",
    "pick_orange_cube_to_tray",
    "pick_purple_cube_to_tray",
    "pick_red_cube_to_front_left_corner",
    "pick_red_cube_to_front_right_corner",
    "pick_red_cube_to_back_left_corner",
    "pick_red_cube_to_back_right_corner",
    "pick_leftmost_cube",
    "stack_red_on_blue",
    "stack_blue_on_red",
    "sort_two_cubes",
]


def get_scene(task_name: str):
    """
    Instantiate and return the scene for the given task name.
    Must be called after SimulationApp() has been created.
    """
    if task_name == "pick_red_cube_to_tray":
        from tasks.pick_red_cube_to_tray.isaac_collector import PickRedCubeToTrayScene
        return PickRedCubeToTrayScene()
    elif task_name == "pick_blue_cube_to_tray":
        from tasks.pick_blue_cube_to_tray.isaac_collector import PickBlueCubeToTrayScene
        return PickBlueCubeToTrayScene()
    elif task_name == "pick_yellow_cube_to_tray":
        from tasks.pick_yellow_cube_to_tray.isaac_collector import PickYellowCubeToTrayScene
        return PickYellowCubeToTrayScene()
    elif task_name == "pick_orange_cube_to_tray":
        from tasks.pick_orange_cube_to_tray.isaac_collector import PickOrangeCubeToTrayScene
        return PickOrangeCubeToTrayScene()
    elif task_name == "pick_purple_cube_to_tray":
        from tasks.pick_purple_cube_to_tray.isaac_collector import PickPurpleCubeToTrayScene
        return PickPurpleCubeToTrayScene()
    elif task_name == "pick_red_cube_to_front_left_corner":
        from tasks.pick_red_cube_to_front_left_corner.isaac_collector import PickRedCubeToFrontLeftCornerScene
        return PickRedCubeToFrontLeftCornerScene()
    elif task_name == "pick_red_cube_to_front_right_corner":
        from tasks.pick_red_cube_to_front_right_corner.isaac_collector import PickRedCubeToFrontRightCornerScene
        return PickRedCubeToFrontRightCornerScene()
    elif task_name == "pick_red_cube_to_back_left_corner":
        from tasks.pick_red_cube_to_back_left_corner.isaac_collector import PickRedCubeToBackLeftCornerScene
        return PickRedCubeToBackLeftCornerScene()
    elif task_name == "pick_red_cube_to_back_right_corner":
        from tasks.pick_red_cube_to_back_right_corner.isaac_collector import PickRedCubeToBackRightCornerScene
        return PickRedCubeToBackRightCornerScene()
    elif task_name == "pick_leftmost_cube":
        from tasks.pick_leftmost_cube.isaac_collector import PickLeftmostCubeScene
        return PickLeftmostCubeScene()
    elif task_name == "stack_red_on_blue":
        from tasks.stack_red_on_blue.isaac_collector import StackRedOnBlueScene
        return StackRedOnBlueScene()
    elif task_name == "stack_blue_on_red":
        from tasks.stack_blue_on_red.isaac_collector import StackBlueOnRedScene
        return StackBlueOnRedScene()
    elif task_name == "sort_two_cubes":
        from tasks.sort_two_cubes.isaac_collector import SortTwoCubesScene
        return SortTwoCubesScene()
    else:
        raise ValueError(
            f"Unknown task: {task_name!r}\nKnown tasks: {TASK_NAMES}"
        )
