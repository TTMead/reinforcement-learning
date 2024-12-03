from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv


def make_env(time_scale, file_path, no_graphics, seed):
    """Creates a petting-zoo style environment using a port connection or file 
    instance of a godot scene"""
    config = {
        "env_path": file_path,
        "show_window": (not no_graphics),
        "speedup": time_scale
    }
    return GDRLPettingZooEnv(config=config, seed=seed)
