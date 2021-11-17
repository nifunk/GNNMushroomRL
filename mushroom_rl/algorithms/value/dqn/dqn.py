from copy import deepcopy

import numpy as np

from mushroom_rl.algorithms.agent import Agent
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory, ReplayMemory
from mushroom_rl.utils.parameters import to_parameter


class AbstractDQN(Agent):
    def __init__(self, mdp_info, policy, approximator, approximator_params,
                 batch_size, target_update_frequency,
                 replay_memory=None, initial_replay_size=500,
                 max_replay_size=5000, fit_params=None, clip_reward=False, mdp=None):
        """
        Constructor.

        Args:
            approximator (object): the approximator to use to fit the
               Q-function;
            approximator_params (dict): parameters of the approximator to
                build;
            batch_size ((int, Parameter)): the number of samples in a batch;
            target_update_frequency (int): the number of samples collected
                between each update of the target network;
            replay_memory ([ReplayMemory, PrioritizedReplayMemory], None): the
                object of the replay memory to use; if None, a default replay
                memory is created;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            fit_params (dict, None): parameters of the fitting algorithm of the
                approximator;
            clip_reward (bool, False): whether to clip the reward or not.

        """
        self.mdp = mdp
        self._fit_params = dict() if fit_params is None else fit_params

        self._batch_size = to_parameter(batch_size)
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency

        if replay_memory is not None:
            self._replay_memory = replay_memory
            if isinstance(replay_memory, PrioritizedReplayMemory):
                self._fit = self._fit_prioritized
            else:
                self._fit = self._fit_standard
        else:
            self._replay_memory = ReplayMemory(initial_replay_size,
                                               max_replay_size)
            self._fit = self._fit_standard

        self._n_updates = 0

        apprx_params_train = deepcopy(approximator_params)
        apprx_params_train['loglist'] = approximator_params['loglist']
        apprx_params_target = deepcopy(approximator_params)
        apprx_params_target['loglist'] = apprx_params_target['loglist']

        self._initialize_regressors(approximator, apprx_params_train,
                                    apprx_params_target)
        policy.set_q(self.approximator)
        # policy.set_q_target(self.target_approximator)

        self._add_save_attr(
            _fit_params='pickle',
            _batch_size='mushroom',
            _n_approximators='primitive',
            _clip_reward='primitive',
            _target_update_frequency='primitive',
            _replay_memory='mushroom',
            _n_updates='primitive',
            approximator='mushroom',
            target_approximator='mushroom'
        )

        super().__init__(mdp_info, policy)

    def fit(self, dataset):
        self._fit(dataset)

        self._n_updates += 1
        if self._n_updates % self._target_update_frequency == 0:
            self._update_target()

    def _fit_standard(self, dataset, approximator=None):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, expl_ac, expl_next_s, expl_next_sr, expl_next_sa, expl_next_sv, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            # first do "pre-filtering" - this is a caveat of our special parallelized implementation. All samples for
            # which there has been no search conducted are marked with this special observation which can be identified
            # by this prefilter,... (cf. line 146 in td_policy.py)
            pre_filter = deepcopy(expl_next_sa != np.array(-100))

            expl_next_s = expl_next_s[pre_filter].astype(float)
            expl_ac = expl_ac[pre_filter].astype(int)
            expl_next_sr = expl_next_sr[pre_filter].astype(float)
            expl_next_sv = expl_next_sv[pre_filter].astype(float)
            expl_next_sa = expl_next_sa[pre_filter].astype(bool)

            expl_next_s = expl_next_s.reshape(-1,np.shape(state)[-1])

            expl_ac = expl_ac.reshape(-1,np.shape(action)[-1])
            num_explored_actions = np.shape(pre_filter)[1]

            add_state = deepcopy(state)
            add_state = add_state[pre_filter[:,0]]
            add_state = np.expand_dims(add_state,axis=1)
            add_state = np.tile(add_state, (1, num_explored_actions, 1))
            add_state = add_state.reshape((-1,np.shape(state)[-1]))

            state = np.vstack((state,add_state))


            expl_next_sr = expl_next_sr.reshape(-1)
            expl_next_sa = expl_next_sa.reshape(-1)

            # state = np.vstack((state, expl_next_s))
            action = np.vstack((action,expl_ac))

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            standard = False
            if (standard):
                # treat additional samples as additional experience, i.e. dynamically compute new target value (HAS TO BE TESTED,...)
                next_state = np.vstack((next_state, expl_next_s))
                reward = np.hstack((reward, expl_next_sr))
                absorbing = np.hstack((absorbing, expl_next_sa))
                q_next = self._next_q(next_state, absorbing)
                q = reward + self.mdp_info.gamma * q_next
            else:
                # calculate only the q-value dynamically for the action that has been taken, while the experience
                # collected during search is interpreted as a constant regularization,...
                q_next = self._next_q(next_state, absorbing)
                q = reward + self.mdp_info.gamma * q_next
                # -> add target values from mcts search statically -> they are a static regularizer
                q = np.hstack((q,expl_next_sv))

            if approximator is None:
                self.approximator.fit(state, action, q, **self._fit_params)
            else:
                approximator.fit(state, action, q, **self._fit_params)

    def _fit_prioritized(self, dataset, approximator=None):
        self._replay_memory.add(
            dataset, np.ones(len(dataset)) * self._replay_memory.max_priority)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _, idxs, is_weight = \
                self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next
            td_error = q - self.approximator.predict(state, action)

            self._replay_memory.update(td_error, idxs)

            if approximator is None:
                self.approximator.fit(state, action, q, weights=is_weight,
                                      **self._fit_params)
            else:
                approximator.fit(state, action, q, weights=is_weight,
                                 **self._fit_params)

    def draw_action(self, state):
        action = super().draw_action(np.array(state))

        return action

    def _initialize_regressors(self, approximator, apprx_params_train,
                               apprx_params_target):
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             **apprx_params_target)
        self._update_target()

    def _update_target(self):
        """
        Update the target network.

        """
        self.target_approximator.set_weights(self.approximator.get_weights())

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Maximum action-value for each state in ``next_state``.

        """
        raise NotImplementedError

    def _post_load(self):
        if isinstance(self._replay_memory, PrioritizedReplayMemory):
            self._fit = self._fit_prioritized
        else:
            self._fit = self._fit_standard

        self.policy.set_q(self.approximator)


class DQN(AbstractDQN):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """
    def _next_q(self, next_state, absorbing):
        q = self.target_approximator.predict(next_state)
        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1)

        return np.max(q, axis=1)

class DQNMultidim(AbstractDQN):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """
    def _next_q(self, next_state, absorbing):
        q = self.target_approximator.predict(next_state)

        target_elements, elements_placed, elements_to_be_placed, num_actions, target_arr, elements_placed_arr, elements_to_be_placed_arr = self.mdp._decode_observation_multidim(next_state)
        # use those masking arrays in order to only get the q values that are allowed,...
        mask1 = np.zeros(q.shape) # to filter out which block can be placed in the next step
        mask2 = np.zeros(q.shape) # to filter out with respect to which block we can place the things in the next step
        mask1[elements_to_be_placed_arr,:,:] = 1
        mask2[elements_placed_arr,:,:] = 1
        mask2 = np.transpose(mask2, (0, 2, 1, 3))

        # fuse the two masks
        mask = np.logical_and((mask1>0.1),(mask2>0.1))

        # for all invalid actions set to minimal q value (hopefully better than setting to arbitrary negative number or so)
        q[np.logical_not(mask)] = np.min(q)

        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1, 1, 1)

        return np.max(np.max(np.max(q, axis=3),axis=2),axis=1)