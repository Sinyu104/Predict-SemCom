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


def get_scene(task_name: str, camera_ids: list | tuple = (1,)):
    """
    Instantiate and return the scene for the given task name.
    Must be called after SimulationApp() has been created.

    camera_ids : which cameras to activate (1=fixed cam1, 2=fixed cam2,
                 3=wrist). Forwarded to the scene's camera_ids kwarg.
    """
    if task_name == "pick_red_cube_to_tray":
        from tasks.pick_red_cube_to_tray.isaac_collector import PickRedCubeToTrayScene
        return PickRedCubeToTrayScene(camera_ids=camera_ids)
    elif task_name == "pick_blue_cube_to_tray":
        from tasks.pick_blue_cube_to_tray.isaac_collector import PickBlueCubeToTrayScene
        return PickBlueCubeToTrayScene(camera_ids=camera_ids)
    elif task_name == "pick_yellow_cube_to_tray":
        from tasks.pick_yellow_cube_to_tray.isaac_collector import PickYellowCubeToTrayScene
        return PickYellowCubeToTrayScene(camera_ids=camera_ids)
    elif task_name == "pick_orange_cube_to_tray":
        from tasks.pick_orange_cube_to_tray.isaac_collector import PickOrangeCubeToTrayScene
        return PickOrangeCubeToTrayScene(camera_ids=camera_ids)
    elif task_name == "pick_purple_cube_to_tray":
        from tasks.pick_purple_cube_to_tray.isaac_collector import PickPurpleCubeToTrayScene
        return PickPurpleCubeToTrayScene(camera_ids=camera_ids)
    elif task_name == "pick_red_cube_to_front_left_corner":
        from tasks.pick_red_cube_to_front_left_corner.isaac_collector import PickRedCubeToFrontLeftCornerScene
        return PickRedCubeToFrontLeftCornerScene(camera_ids=camera_ids)
    elif task_name == "pick_red_cube_to_front_right_corner":
        from tasks.pick_red_cube_to_front_right_corner.isaac_collector import PickRedCubeToFrontRightCornerScene
        return PickRedCubeToFrontRightCornerScene(camera_ids=camera_ids)
    elif task_name == "pick_red_cube_to_back_left_corner":
        from tasks.pick_red_cube_to_back_left_corner.isaac_collector import PickRedCubeToBackLeftCornerScene
        return PickRedCubeToBackLeftCornerScene(camera_ids=camera_ids)
    elif task_name == "pick_red_cube_to_back_right_corner":
        from tasks.pick_red_cube_to_back_right_corner.isaac_collector import PickRedCubeToBackRightCornerScene
        return PickRedCubeToBackRightCornerScene(camera_ids=camera_ids)
    elif task_name == "pick_leftmost_cube":
        from tasks.pick_leftmost_cube.isaac_collector import PickLeftmostCubeScene
        return PickLeftmostCubeScene(camera_ids=camera_ids)
    elif task_name == "stack_red_on_blue":
        from tasks.stack_red_on_blue.isaac_collector import StackRedOnBlueScene
        return StackRedOnBlueScene(camera_ids=camera_ids)
    elif task_name == "stack_blue_on_red":
        from tasks.stack_blue_on_red.isaac_collector import StackBlueOnRedScene
        return StackBlueOnRedScene(camera_ids=camera_ids)
    elif task_name == "sort_two_cubes":
        from tasks.sort_two_cubes.isaac_collector import SortTwoCubesScene
        return SortTwoCubesScene(camera_ids=camera_ids)
    else:
        raise ValueError(
            f"Unknown task: {task_name!r}\nKnown tasks: {TASK_NAMES}"
        )
