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
import pickle
import sys
import copy
import torch.multiprocessing as mp
from functools import partial
from itertools import product
from math import ceil
from tqdm import tqdm
from typing import List, Optional

from .sampler import SamplerBase
from init_args_serializer import Serializable

from .step_sequence import StepSequence
from .sampler_pool import SamplerPool
from .rollout import rollout

# import pyrado
# from pyrado.environments.base import Env
# from pyrado.policies.base import Policy
# from pyrado.sampling.sampler_pool import SamplerPool
# from pyrado.sampling.step_sequence import StepSequence
# from pyrado.sampling.rollout import rollout
# from pyrado.sampling.sampler import SamplerBase
from mushroom_rl.utils.function_calls import wrapped_call

def _ps_init(G, env, mdp_args,policy,preprocessor):
    """ Store pickled (and thus copied) environment and policy. """
    G.env = wrapped_call(type(env),mdp_args[0],mdp_args[1])
    G.policy = pickle.loads(policy)
    G.policy.set_mdp(G.env)
    G.preprocessor = pickle.loads(preprocessor)

def _ps_update_policy(G, policy,preprocessor):
    G.policy = pickle.loads(policy)
    G.policy.set_mdp(G.env)
    G.preprocessor = pickle.loads(preprocessor)
    del preprocessor

def _ps_sample_one(G, eval: bool):
    """
    Sample one rollout and return step count if counting steps, rollout count (1) otherwise.
    This function is used when a minimum number of steps was given.
    """
    ro = rollout(G.env, G.policy, G.preprocessor, eval=eval)
    return ro, len(ro)


def _ps_run_one(G, num: int, eval: bool):
    """
    Sample one rollout without specifying the initial state or the domain parameters.
    This function is used when a minimum number of rollouts was given.
    """
    return rollout(G.env, G.agent, eval=eval)


def _ps_run_one_init_state(G, init_state: np.ndarray, eval: bool):
    """
    Sample one rollout with given init state.
    This function is used when a minimum number of rollouts was given.
    """
    return rollout(G.env, G.agent, eval=eval, reset_kwargs=dict(init_state=init_state))


def _ps_run_one_domain_param(G, domain_param: dict, eval: bool):
    """
    Sample one rollout with given domain parameters.
    This function is used when a minimum number of rollouts was given.
    """
    return rollout(G.env, G.agent, eval=eval, reset_kwargs=dict(domain_param=domain_param))


def _ps_run_one_reset_kwargs(G, reset_kwargs: tuple, eval: bool):
    """
    Sample one rollout with given init state and domain parameters, passed as a tuple for simplicity at the other end.
    This function is used when a minimum number of rollouts was given.
    """
    if len(reset_kwargs) != 2:
        raise pyrado.ShapeErr(given=reset_kwargs, expected_match=(2,))
    if not isinstance(reset_kwargs[0], np.ndarray):
        raise pyrado.TypeErr(given=reset_kwargs[0], expected_type=np.ndarray)
    if not isinstance(reset_kwargs[1], dict):
        raise pyrado.TypeErr(given=reset_kwargs[1], expected_type=dict)
    return rollout(
        G.env, G.agent, eval=eval, reset_kwargs=dict(init_state=reset_kwargs[0], domain_param=reset_kwargs[1])
    )


class ParallelRolloutSampler(SamplerBase, Serializable):
    """ Class for sampling from multiple environments in parallel """

    def __init__(
        self,
        env,
        agent,
        use_cuda,
        core_obj,
        num_workers: int,
        *,
        min_rollouts: int = None,
        min_steps: int = None,
        show_progress_bar: bool = True,
        seed: int = None,
        mdp_args: int=None,
    ):
        """
        Constructor

        :param env: environment to sample from
        :param agent: agent to act in the environment (can also be an exploration strategy)
        :param num_workers: number of parallel samplers
        :param min_rollouts: minimum number of complete rollouts to sample
        :param min_steps: minimum total number of steps to sample
        :param show_progress_bar: it `True`, display a progress bar using `tqdm`
        :param seed: seed value for the random number generators, pass `None` for no seeding
        """
        Serializable._init(self, locals())
        super().__init__(min_rollouts=min_rollouts, min_steps=min_steps)

        self.env = env
        self.agent = agent
        self.show_progress_bar = show_progress_bar
        self.core_obj = core_obj
        self.mdp_args = mdp_args
        self.num_workers = num_workers
        self.seed = seed

        # # Set method to spawn if using cuda
        if use_cuda and mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)

        # Create parallel pool. We use one thread per env because it's easier.
        self.pool = SamplerPool(num_workers)

        # Set all rngs' seeds
        if seed is not None:
            self.set_seed(seed)

        # Distribute environments. We use pickle to make sure a copy is created for n_envs=1
        # self.pool.invoke_all(_ps_init, pickle.dumps(self.env), pickle.dumps(self.policy),mdp_args)
        self.pool.invoke_all(_ps_init, self.env, self.mdp_args, pickle.dumps(self.agent.policy),pickle.dumps(self.core_obj._preprocessors))

    def set_seed(self, seed):
        """
        Set a deterministic seed on all workers.

        :param seed: seed value for the random number generators
        """
        self.pool.set_seed(seed)

    def set_num_steps(self, num_steps):
        self.min_steps = num_steps


    def reinit(self, env: None, agent: None):
        """
        Re-initialize the sampler.

        :param env: the environment which the agent operates
        :param agent: the agent used for sampling
        """
        # Update env and agent if passed
        if env is not None:
            self.env = env
        if agent is not None:
            self.agent = agent

        # Always broadcast to workers
        self.pool.invoke_all(_ps_init, self.env, pickle.dumps(self.agent),self.mdp_args, pickle.dumps(self.agent.policy),pickle.dumps(self.core_obj._preprocessors))

    def sample(
        self,
        init_states: Optional[List[np.ndarray]] = None,
        domain_params: Optional[List[dict]] = None,
        eval: Optional[bool] = False,
        update: Optional[bool] = True,
    ) -> List[StepSequence]:
        """
        Do the sampling according to the previously given environment, agent, and number of steps/rollouts.

        :param init_states: initial states forw `run_map()`, pass `None` (default) to sample from the environment's
                            initial state space
        :param domain_params: domain parameters for `run_map()`, pass `None` (default) to not explicitly set them
        :param eval: pass `False` if the rollout is executed during training, else `True`. Forwarded to `rollout()`.
        :return: list of sampled rollouts
        """

        if (True):
            self.pool.invoke_all(_ps_update_policy, pickle.dumps(self.agent.policy), pickle.dumps(self.core_obj._preprocessors))


        # Collect samples
        with tqdm(
            leave=False,
            file=sys.stdout,
            desc="Sampling",
            disable=(not self.show_progress_bar),
            unit="steps" if self.min_steps is not None else "rollouts",
        ) as pb:

            if self.min_steps is None:
                if init_states is None and domain_params is None:
                    # Simply run min_rollouts times
                    func = partial(_ps_run_one, eval=eval)
                    arglist = range(self.min_rollouts)
                elif init_states is not None and domain_params is None:
                    # Run every initial state so often that we at least get min_rollouts trajectories
                    func = partial(_ps_run_one_init_state, eval=eval)
                    rep_factor = ceil(self.min_rollouts / len(init_states))
                    arglist = rep_factor * init_states
                elif init_states is None and domain_params is not None:
                    # Run every domain parameter set so often that we at least get min_rollouts trajectories
                    func = partial(_ps_run_one_domain_param, eval=eval)
                    rep_factor = ceil(self.min_rollouts / len(domain_params))
                    arglist = rep_factor * domain_params
                elif init_states is not None and domain_params is not None:
                    # Run every combination of initial state and domain parameter so often that we at least get
                    # min_rollouts trajectories
                    func = partial(_ps_run_one_reset_kwargs, eval=eval)
                    allcombs = list(product(init_states, domain_params))
                    rep_factor = ceil(self.min_rollouts / len(allcombs))
                    arglist = rep_factor * allcombs

                # Only minimum number of rollouts given, thus use run_map
                return self.pool.run_map(func, arglist, pb)

            else:
                # Minimum number of steps given, thus use run_collect (automatically handles min_runs=None)
                if init_states is None:
                    return self.pool.run_collect(
                        self.min_steps,
                        partial(_ps_sample_one, eval=eval),
                        collect_progressbar=pb,
                        min_runs=self.min_rollouts,
                    )[0]
                else:
                    raise NotImplementedError
                    # return self.pool.run_collect(
                    #     self.min_steps,
                    #     _ps_run_one_init_state,
                    #     init_states,  # *args
                    #     collect_progressbar=pb,
                    #     min_runs=self.min_rollouts
                    # )[0]
