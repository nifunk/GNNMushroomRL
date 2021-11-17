from tqdm import tqdm
from .parallelization_tools.parallel_rollout_sampler import ParallelRolloutSampler
from .core import Core

# this core parallel is another implementation of the core which allows sampling in parallel from different instances
# of the environment
# However, note that now the functinality is slightly changed as now all samples are collected with the old policy,
# then all the updates are performed and then new samples are gathered
# this is a bit orthogonal to the behavior of the sampling in the core in which after the update directly the newly
# obtained policy is used in the environment.
class CoreParallel(Core):
    """
    Implements the functions to run a generic algorithm.

    """
    def __init__(self, agent, mdp, use_cuda, callbacks_fit=None, callback_step=None,
                 preprocessors=None, mdp_args=None, num_workers=1):
        """
        Constructor.

        Args:
            agent (Agent): the agent moving according to a policy;
            mdp (Environment): the environment in which the agent moves;
            callbacks_fit (list): list of callbacks to execute at the end of
                each fit;
            callback_step (Callback): callback to execute after each step;
            preprocessors (list): list of state preprocessors to be
                applied to state variables before feeding them to the
                agent.

        """
        self.agent = agent
        self.mdp = mdp
        self.ac_space_shape = self.mdp.info.action_space.shape[0]
        self.callbacks_fit = callbacks_fit if callbacks_fit is not None else list()
        self.callback_step = callback_step if callback_step is not None else lambda x: None
        self._preprocessors = preprocessors if preprocessors is not None else list()
        self.use_cuda = use_cuda
        self.mdp_args = mdp_args
        self.num_workers = num_workers

        self._state = None

        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0
        self._episode_steps = None
        self._n_episodes = None
        self._n_steps_per_fit = None
        self._n_episodes_per_fit = None

        self.sampler = ParallelRolloutSampler(
            self.mdp, self.agent, use_cuda, self, num_workers=self.num_workers, min_steps=1000, mdp_args=mdp_args)

    def learn(self, n_steps=None, n_episodes=None, n_steps_per_fit=None,
              n_episodes_per_fit=None, render=False, quiet=False, reset=False):

        """
        This function moves the agent in the environment and fits the policy
        using the collected samples. The agent can be moved for a given number
        of steps or a given number of episodes and, independently from this
        choice, the policy can be fitted after a given number of steps or a
        given number of episodes. By default, the environment is reset.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            n_steps_per_fit (int, None): number of steps between each fit of the
                policy;
            n_episodes_per_fit (int, None): number of episodes between each fit
                of the policy;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        if (reset):
            # in case of reset delete the current sampler and obtain a new set of parallel workers and environments
            # to collect the experience
            del self.sampler
            self.sampler = ParallelRolloutSampler(
                self.mdp, self.agent, self.use_cuda, self, num_workers=self.num_workers, min_steps=1000, mdp_args=self.mdp_args)

        self.sampler.set_num_steps(n_steps)
        assert (n_episodes_per_fit is not None and n_steps_per_fit is None)\
            or (n_episodes_per_fit is None and n_steps_per_fit is not None)

        self._n_steps_per_fit = n_steps_per_fit
        self._n_episodes_per_fit = n_episodes_per_fit

        if n_steps_per_fit is not None:
            fit_condition =\
                lambda: self._current_steps_counter >= self._n_steps_per_fit
        else:
            fit_condition = lambda: self._current_episodes_counter\
                                     >= self._n_episodes_per_fit

        return self._run(n_steps, n_episodes, fit_condition, render, quiet)

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None,
                 render=False, quiet=False):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from
        a set of initial states for the whole episode. By default, the
        environment is reset.

        Args:
            initial_states (np.ndarray, None): the starting states of each
                episode;
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        self.sampler.set_num_steps(n_steps)
        fit_condition = lambda: False

        return self._run(n_steps, n_episodes, fit_condition, render, quiet,
                         initial_states)

    def _run(self, n_steps, n_episodes, fit_condition, render, quiet,
             initial_states=None):
        assert n_episodes is not None and n_steps is None and initial_states is None\
            or n_episodes is None and n_steps is not None and initial_states is None\
            or n_episodes is None and n_steps is None and initial_states is not None

        self._n_episodes = len(
            initial_states) if initial_states is not None else n_episodes

        if n_steps is not None:
            move_condition =\
                lambda: self._total_steps_counter < n_steps

            steps_progress_bar = tqdm(total=n_steps,
                                      dynamic_ncols=True, disable=quiet,
                                      leave=False)
            episodes_progress_bar = tqdm(disable=True)
        else:
            move_condition =\
                lambda: self._total_episodes_counter < self._n_episodes

            steps_progress_bar = tqdm(disable=True)
            episodes_progress_bar = tqdm(total=self._n_episodes,
                                         dynamic_ncols=True, disable=quiet,
                                         leave=False)

        return self._run_impl(move_condition, fit_condition, steps_progress_bar,
                              episodes_progress_bar, render, initial_states)

    def preprocess_ac(self,action,idx):
        if (self.ac_space_shape==1):
            return action[idx]
        else:
            return action[idx,:]

    def _run_impl(self, move_condition, fit_condition, steps_progress_bar,
                  episodes_progress_bar, render, initial_states, reset=False):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        first_time_false = False
        stopping = False
        self._current_episodes_counter = 0
        self._current_steps_counter = 0

        dataset = list()
        dataset_orig = list()

        # first collect all samples -> ros contains all the rollouts gathered from the parallel workers
        ros = self.sampler.sample()

        # return state, action, reward, next_state, absorbing, last -> loop through all of the collected rollouts
        for ro in ros:
            if (stopping):
                break
            for i in range(len(ro.observations)):
                if (stopping):
                    break

                sample = ro.observations[i], self.preprocess_ac(ro.actions,i), ro.rewards[i], ro.next_obs[i], \
                         ro.explored_act_hist[i,:], ro.explored_next_state[i,:], ro.explored_sing_step_r[i,:], ro.explored_next_states_absorbing[i,:], \
                         ro.explored_next_states_value[i,:], ro.absorbing[i], ro.last[i]

                # "Problem:" If sampling deparallelized -> there is no feedback from rollouts to standard agent
                # -> to updated epsilon, this call here is needed that results in linearily decreasing the value,... (each call value is adapted...)
                if hasattr(self.agent.policy, '_epsilon'):
                    self.agent.policy._epsilon(0)

                self.callback_step([sample])
                self._total_steps_counter += 1
                self._current_steps_counter += 1
                steps_progress_bar.update(1)

                if sample[-1]:
                    self._total_episodes_counter += 1
                    self._current_episodes_counter += 1
                    episodes_progress_bar.update(1)

                dataset.append(sample)
                dataset_orig.append(sample)
                if fit_condition():
                    self.agent.fit(dataset)
                    self._current_episodes_counter = 0
                    self._current_steps_counter = 0

                    for c in self.callbacks_fit:
                        c(dataset)

                    dataset = list()

                last = sample[-1]

                # if move condition not satisfied for first time -> go into first time false mode, i.e. adapt label of upcoming sample
                # if not satisfied for second time -> stop everything -> this is to ensure that only the specified number of samples
                # is used to update the policy and not more,...
                if not(move_condition()):
                    if not(first_time_false):
                        first_time_false = True
                    else:
                        stopping = True

        self.agent.stop()
        self.mdp.stop()

        steps_progress_bar.close()
        episodes_progress_bar.close()

        # return dataset
        return dataset_orig