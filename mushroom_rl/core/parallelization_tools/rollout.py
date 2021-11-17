# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import time
import torch as to
import torch.nn as nn
from matplotlib import pyplot as plt
from typing import Callable, Tuple, Optional, Union
from .step_sequence import StepSequence
from mushroom_rl.utils.seeds import fix_random_seed
# from tabulate import tabulate

# import pyrado
# from pyrado.environment_wrappers.action_delay import ActDelayWrapper
# from pyrado.environments.base import Env
# from pyrado.environments.real_base import RealEnv
# from pyrado.environments.sim_base import SimEnv
# from pyrado.environment_wrappers.utils import inner_env, typed_env
# from pyrado.plotting.curve import draw_dts
# from pyrado.plotting.policy_parameters import draw_policy_params
# from pyrado.plotting.rollout_based import (
#     plot_observations_actions_rewards,
#     plot_actions,
#     plot_observations,
#     plot_rewards,
#     plot_potentials,
#     plot_features,
# )
# from pyrado.policies.base import Policy, TwoHeadedPolicy
# from pyrado.policies.recurrent.potential_based import PotentialBasedPolicy
# from pyrado.sampling.step_sequence import StepSequence
# from pyrado.utils.data_types import RenderMode
# from pyrado.utils.input_output import print_cbt, color_validity


def rollout(
    env: None,
    policy: None,
    preprocessors: None,
    eval: Optional[bool] = False,
    max_steps: Optional[int] = None,
    reset_kwargs: Optional[dict] = None,
    render_mode: Optional[bool] = False,
    render_step: Optional[int] = 1,
    no_reset: Optional[bool] = False,
    no_close: Optional[bool] = False,
    record_dts: Optional[bool] = False,
    stop_on_done: Optional[bool] = True,
    seed: Optional[int] = None,
) -> StepSequence:
    """
    Perform a rollout (i.e. sample a trajectory) in the given environment using given policy.

    :param env: environment to use (`SimEnv` or `RealEnv`)
    :param policy: policy to determine the next action given the current observation.
                   This policy may be wrapped by an exploration strategy.
    :param eval: pass `False` if the rollout is executed during training, else `True`. Forwarded to PyTorch `Module`.
    :param max_steps: maximum number of time steps, if `None` the environment's property is used
    :param reset_kwargs: keyword arguments passed to environment's reset function
    :param render_mode: determines if the user sees an animation, console prints, or nothing
    :param render_step: rendering interval, renders every step if set to 1
    :param no_reset: do not reset the environment before running the rollout
    :param no_close: do not close (and disconnect) the environment after running the rollout
    :param record_dts: flag if the time intervals of different parts of one step should be recorded (for debugging)
    :param stop_on_done: set to false to ignore the environments's done flag (for debugging)
    :param seed: seed value for the random number generators, pass `None` for no seeding
    :return paths of the observations, actions, rewards, and information about the environment as well as the policy
    """

    # Initialize the paths
    obs_hist = []
    act_hist = []
    explored_act_hist = []
    explored_next_state = []
    explored_sing_step_r = []
    explored_next_states_absorbing = []
    explored_next_states_value = []
    rew_hist = []
    next_obs_hist = []
    absorbing = []
    last = []

    t_hist = []

    # Override the number of steps to execute
    if max_steps is not None:
        env.max_steps = max_steps

    # Set all rngs' seeds
    if seed is not None:
        fix_random_seed(seed,env)

    # Reset the environment and pass the kwargs
    if reset_kwargs is None:
        reset_kwargs = {}

    if not no_reset:
        obs = env.reset(**reset_kwargs)
    else:
        print ("CAREFUL: not really resetting,...")
        obs = np.zeros(env.obs_space.shape)

    # Initialize the main loop variables
    done = False
    t = 0.0  # time starts at zero
    t_hist.append(t)
    if record_dts:
        t_post_step = time.time()  # first sample of remainder is useless

    # ----------
    # Begin loop
    # ----------

    # Terminate if the environment signals done, it also keeps track of the time
    while not (done and stop_on_done):

        # Check observations
        if np.isnan(obs).any():
            env.render(render_mode, render_step=1)
            raise pyrado.ValueErr(
                msg=f"At least one observation value is NaN!"
                + tabulate(
                    [list(env.obs_space.labels), [*color_validity(obs, np.invert(np.isnan(obs)))]], headers="firstrow"
                )
            )

        # Get the agent's action
        obs = _preprocess(obs.copy(), preprocessors)
        obs_hist.append(obs)

        obs_to = to.from_numpy(obs).type(to.get_default_dtype())  # policy operates on PyTorch tensors
        with to.no_grad():
            act_to = policy.draw_action(obs_to)
            if (len(act_to)==6):
                explored_act_hist.append([*act_to[1]])
                explored_next_state.append([*act_to[2]])
                explored_sing_step_r.append([*act_to[3]])
                explored_next_states_absorbing.append([*act_to[4]])
                explored_next_states_value.append([*act_to[5]])
                act_to = act_to[0]
            else:
                explored_act_hist.append([act_to])
                explored_next_state.append([np.asarray(obs_to)])
                explored_sing_step_r.append([None])
                explored_next_states_absorbing.append([None])
                explored_next_states_value.append([None])

        act = act_to

        # Ask the environment to perform the simulation step
        obs_next, rew, done, env_info = env.step(act)

        # Store the observation for next step (if done, this is the final observation)
        obs_next = _preprocess(obs_next.copy(), preprocessors)
        obs = obs_next
        act_hist.append(np.asarray(list(act)))
        rew_hist.append(rew)
        next_obs_hist.append(obs_next)
        # in this case here of collecting complete rollouts absorbing and last is identical
        absorbing.append(done)
        last.append(done)

        t_hist.append(t)

    # --------
    # End loop
    # --------

    if not no_close:
        # Disconnect from EnvReal instance (does nothing for EnvSim instances)
        env.close()

    # Return result object
    res = StepSequence(
        observations=obs_hist,
        next_obs=next_obs_hist,
        actions=act_hist,
        rewards=rew_hist,
        absorbing=absorbing,
        last=last,
        #states=state_hist,
        time=t_hist,
        explored_act_hist=explored_act_hist,
        explored_next_state=explored_next_state,
        explored_sing_step_r=explored_sing_step_r,
        explored_next_states_absorbing=explored_next_states_absorbing,
        explored_next_states_value=explored_next_states_value,
        # rollout_info=rollout_info,
        #env_infos=env_info_hist,
        complete=True,  # the rollout function always returns complete paths
    )

    return res


def _preprocess(state, preprocessors):
    """
    Method to apply state preprocessors.

    Args:
        state (np.ndarray): the state to be preprocessed.

    Returns:
         The preprocessed state.

    """
    for p in preprocessors:
        state = p(state)

    return state


def after_rollout_query(
    env: None, policy: None, rollout: StepSequence
) -> Tuple[bool, Optional[np.ndarray], Optional[dict]]:
    """
    Ask the user what to do after a rollout has been animated.

    :param env: environment used for the rollout
    :param policy: policy used for the rollout
    :param rollout: collected data from the rollout
    :return: done flag, initial state, and domain parameters
    """
    # Fist entry contains hotkey, second the info text
    options = [
        ["C", "continue simulation (with domain randomization)"],
        ["N", "set domain parameters to nominal values, and continue"],
        ["F", "fix the initial state"],
        ["I", "print information about environment (including randomizer), and policy"],
        ["S", "set a domain parameter explicitly"],
        ["P", "plot all observations, actions, and rewards"],
        ["PO [indices]", "plot all observations, or selected ones by passing separated integers"],
        ["PA", "plot actions"],
        ["PR", "plot rewards"],
        ["PF", "plot features (for linear policy)"],
        ["PPOT", "plot potentials, stimuli, and actions (for potential-based policies)"],
        ["PDT", "plot time deltas (profiling of a real system)"],
        ["E", "exit"],
    ]

    # Ask for user input
    ans = input(tabulate(options, tablefmt="simple") + "\n").lower()

    if ans == "c" or ans == "":
        # We don't have to do anything here since the env will be reset at the beginning of the next rollout
        return False, None, None

    elif ans == "f":
        try:
            if isinstance(inner_env(env), RealEnv):
                raise pyrado.TypeErr(given=inner_env(env), expected_type=SimEnv)
            elif isinstance(inner_env(env), SimEnv):
                # Get the user input
                usr_inp = input(
                    f"Enter the {env.obs_space.flat_dim}-dim initial state "
                    f"(format: each dim separated by a whitespace):\n"
                )
                state = list(map(float, usr_inp.split()))
                if isinstance(state, list):
                    state = np.array(state)
                    if state.shape != env.obs_space.shape:
                        raise pyrado.ShapeErr(given=state, expected_match=env.obs_space)
                else:
                    raise pyrado.TypeErr(given=state, expected_type=list)
                return False, state, {}
        except (pyrado.TypeErr, pyrado.ShapeErr):
            return after_rollout_query(env, policy, rollout)

    elif ans == "n":
        # Get nominal domain parameters
        if isinstance(inner_env(env), SimEnv):
            dp_nom = inner_env(env).get_nominal_domain_param()
            if typed_env(env, ActDelayWrapper) is not None:
                # There is an ActDelayWrapper in the env chain
                dp_nom["act_delay"] = 0
        else:
            dp_nom = None
        return False, None, dp_nom

    elif ans == "i":
        # Print the information and return to the query
        print(env)
        if hasattr(env, "randomizer"):
            print(env.randomizer)
        print(policy)
        return after_rollout_query(env, policy, rollout)

    elif ans == "p":
        plot_observations_actions_rewards(rollout)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "pa":
        plot_actions(rollout, env)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "po":
        plot_observations(rollout)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif "po" in ans and any(char.isdigit() for char in ans):
        idcs = [int(s) for s in ans.split() if s.isdigit()]
        plot_observations(rollout, idcs_sel=idcs)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "pf":
        plot_features(rollout, policy)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "pp":
        draw_policy_params(policy, env.spec, annotate=False)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "pr":
        plot_rewards(rollout)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "pdt":
        draw_dts(rollout.dts_policy, rollout.dts_step, rollout.dts_remainder)
        plt.show()
        return (after_rollout_query(env, policy, rollout),)

    elif ans == "ppot":
        plot_potentials(rollout)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "s":
        if isinstance(env, SimEnv):
            dp = env.get_nominal_domain_param()
            for k, v in dp.items():
                dp[k] = [v]  # cast float to list of one element to make it iterable for tabulate
            print("These are the nominal domain parameters:")
            print(tabulate(dp, headers="keys", tablefmt="simple"))

        # Get the user input
        strs = input("Enter one new domain parameter\n(format: key whitespace value):\n")
        try:
            param = dict(str.split() for str in strs.splitlines())
            # Cast the values of the param dict from str to float
            for k, v in param.items():
                param[k] = float(v)
            return False, None, param
        except (ValueError, KeyError):
            print_cbt(f"Could not parse {strs} into a dict.", "r")
            after_rollout_query(env, policy, rollout)

    elif ans == "e":
        env.close()
        return True, None, {}  # breaks the outer while loop

    else:
        return after_rollout_query(env, policy, rollout)  # recursion
