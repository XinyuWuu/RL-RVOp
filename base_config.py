import numpy as np
PARAMs = {
    "seed": 0,
    "isdraw": False,
    "isrender": False,
    "codec": 'h264',
    "framerate": 10,
    "dreach": 0.075,
    "tolerance": 0.04,
    "rreach": 30.0,
    "a": 2,
    "b": 1,
    "c": 10,
    "d": 1,
    "e": 1,
    "f": 20,
    "g": 1,
    "eta": 0.5,
    "h": 0.5,
    "mu": 0.75,
    "dmax": 3.0,
    "vmax": 1.0,
    "tau": 0.5,
    "remix": True,
    "rm_middle": 5,
    "w": 5,
    "device": "cuda",
    "gamma": 0.99,
    "polyak": 0.995,
    "lr": 5e-4,
    "alpha": 1,
    "act_dim": 2,
    "max_obs": 16,
    "obs_self_dim": 4,
    "obs_sur_dim": 10,
    "hidden_sizes": [1024, 1024, 1024],
    "replay_size": int(1e6),
    "batch_size": 1024,
    "max_simu_second": 30,
    "steps_per_epoch": 20000,
    "epochs": 1000,
    "update_every": 50,
}
PARAMs["obs_dim"] = PARAMs["obs_self_dim"] + \
    PARAMs["max_obs"] * (PARAMs["obs_sur_dim"] + 1)
PARAMs["act_limit"] = np.array(
    [PARAMs["vmax"], PARAMs["vmax"]], dtype=np.float32)
PARAMs["max_ep_len"] = int(PARAMs["max_simu_second"] * PARAMs["framerate"])
PARAMs["random_steps"] = PARAMs["max_ep_len"] * 10
PARAMs["update_after"] = PARAMs["max_ep_len"] * 2
PARAMs["total_steps"] = PARAMs["steps_per_epoch"] * PARAMs["epochs"]
