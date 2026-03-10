# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from . import agents

gym.register(
    id="Atom01-Standup",
    entry_point=f"{__name__}.standup_env:StandupEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standup_env_cfg:Atom01StandupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_standup_agent_cfg:RslRlOnPolicyRunnerStandupCfg",
    },
)

gym.register(
    id="Atom01-Standup-Play",
    entry_point=f"{__name__}.standup_env:StandupEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standup_env_cfg:Atom01StandupEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_standup_agent_cfg:RslRlOnPolicyRunnerStandupCfg",
    },
)
