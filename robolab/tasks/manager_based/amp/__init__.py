# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gymnasium as gym
from . import agents

gym.register(
    id="Atom01-AMP",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_env_cfg:Atom01AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="Atom01-AMP-Play",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_env_cfg:Atom01AmpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

# Walk + Getup combined skill
gym.register(
    id="Atom01-AMP-Getup",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_getup_env_cfg:Atom01AmpGetupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="Atom01-AMP-Getup-Play",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_getup_env_cfg:Atom01AmpGetupEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

# PHUMA dataset integration
gym.register(
    id="Atom01-AMP-Phuma",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_phuma_env_cfg:Atom01AmpPhumaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="Atom01-AMP-Phuma-Play",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_phuma_env_cfg:Atom01AmpPhumaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="Atom01-AMP-Lab-Phuma",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_phuma_env_cfg:Atom01AmpLabPhumaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="Atom01-AMP-Lab-Phuma-Play",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_phuma_env_cfg:Atom01AmpLabPhumaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

# Walk + Getup + PHUMA combined skill
gym.register(
    id="Atom01-AMP-Getup-Phuma",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_getup_phuma_env_cfg:Atom01AmpGetupPhumaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="Atom01-AMP-Getup-Phuma-Play",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_getup_phuma_env_cfg:Atom01AmpGetupPhumaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="Atom01-AMP-Getup-Phuma-Single",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_getup_phuma_env_cfg:Atom01AmpGetupPhumaEnvCfg_SINGLE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="Atom01-AMP-Getup-Phuma-Paper",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_getup_phuma_env_cfg:Atom01AmpGetupPhumaEnvCfg_PAPER",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

# V3: reduced push frequency (25-40s) for more natural walking gait
gym.register(
    id="Atom01-AMP-Getup-Phuma-V3",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_getup_phuma_env_cfg:Atom01AmpGetupPhumaEnvCfgV3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="Atom01-AMP-Getup-Phuma-V3-Play",
    entry_point=f"{__name__}.amp_env:AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_amp_getup_phuma_env_cfg:Atom01AmpGetupPhumaEnvCfgV3_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_amp_agent_cfg:RslRlOnPolicyRunnerAmpCfg",
    },
)