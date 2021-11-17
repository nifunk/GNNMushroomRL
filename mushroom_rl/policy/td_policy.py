import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from .policy import Policy
import copy

from mushroom_rl.utils.parameters import Parameter, to_parameter
from mushroom_rl.policy.mcts.mcts_base import MCTS


class TDPolicy(Policy):
    def __init__(self):
        """
        Constructor.

        """
        self._approximator = None
        # self._target_approximator = None

        self._add_save_attr(_approximator='mushroom!')
        # self._add_save_attr(_approximator='mushroom!',_target_approximator='mushroom!')


    def set_q(self, approximator):
        """
        Args:
            approximator (object): the approximator to use.

        """
        self._approximator = approximator

    def set_q_target(self, approximator):

        self._target_approximator = approximator

    def get_q(self):
        """
        Returns:
             The approximator used by the policy.

        """
        return self._approximator


class EpsGreedy(TDPolicy):
    """
    Epsilon greedy policy.

    """
    def __init__(self, epsilon):
        """
        Constructor.

        Args:
            epsilon ((float, Parameter)): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step.

        """
        super().__init__()

        self._epsilon = to_parameter(epsilon)

        self._add_save_attr(_epsilon='mushroom')

    def __call__(self, *args):
        state = args[0]
        q = self._approximator.predict(np.expand_dims(state, axis=0)).ravel()
        max_a = np.argwhere(q == np.max(q)).ravel()

        p = self._epsilon.get_value(state) / self._approximator.n_actions

        if len(args) == 2:
            action = args[1]
            if action in max_a:
                return p + (1. - self._epsilon.get_value(state)) / len(max_a)
            else:
                return p
        else:
            probs = np.ones(self._approximator.n_actions) * p
            probs[max_a] += (1. - self._epsilon.get_value(state)) / len(max_a)

            return probs

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            q = self._approximator.predict(state)
            max_a = np.argwhere(q == np.max(q)).ravel()

            if len(max_a) > 1:
                max_a = np.array([np.random.choice(max_a)])

            return max_a

        return np.array([np.random.choice(self._approximator.n_actions)])

    def set_epsilon(self, epsilon):
        """
        Setter.

        Args:
            epsilon ((float, Parameter)): the exploration coefficient. It indicates the
            probability of performing a random actions in the current step.

        """
        self._epsilon = to_parameter(epsilon)

    def update(self, *idx):
        """
        Update the value of the epsilon parameter at the provided index (e.g. in
        case of different values of epsilon for each visited state according to
        the number of visits).

        Args:
            *idx (list): index of the parameter to be updated.

        """
        self._epsilon.update(*idx)


class EpsGreedyMultidimensional(TDPolicy):
    """
    Epsilon greedy policy on a multidimensional task

    """
    def __init__(self, epsilon, mdp):
        print("USING NORMAL EPLSILON GREEDY")
        """
        Constructor.

        Args:
            epsilon ((float, Parameter)): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step.

        """
        super().__init__()

        self._epsilon = to_parameter(epsilon)
        self.mdp = mdp # this is needed to find out the allowed actions from the environment,...
        self._add_save_attr(_epsilon='mushroom')

        # Note: to allow compatibility with a filled replay buffer from MCTS here the value of 1 has to be changed to the actual search budget
        self.dummy_obs = [self.mdp.info.observation_space.low]*1
        self.dummy_act = [np.zeros(self.mdp.info.action_space.shape)]*1
        self.dummy_none = [-100]*1

    def __call__(self, *args):
        state = args[0]
        q = self._approximator.predict(np.expand_dims(state, axis=0)).ravel()

        max_a = np.argwhere(q == np.max(q)).ravel()

        p = self._epsilon.get_value(state) / self._approximator.n_actions

        if len(args) == 2:
            action = args[1]
            if action in max_a:
                return p + (1. - self._epsilon.get_value(state)) / len(max_a)
            else:
                return p
        else:
            probs = np.ones(self._approximator.n_actions) * p
            probs[max_a] += (1. - self._epsilon.get_value(state)) / len(max_a)

            return probs

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            q = self._approximator.predict(state)

            elements_placed, elements_to_be_placed, num_actions = self.identify_valid_actions(state)

            q = q[elements_to_be_placed][:,elements_placed]

            max_a = np.argwhere(q == np.max(q))
            #print (max_a)
            if len(max_a) > 1:
                max_a = max_a[np.random.choice(len(max_a))]
            else:
                max_a = max_a[0]

            # to account for the filtering we did before
            max_a[0] = elements_to_be_placed[max_a[0]]
            max_a[1] = elements_placed[max_a[1]]

            return copy.deepcopy([max_a, self.dummy_act, self.dummy_obs, self.dummy_none, self.dummy_none, self.dummy_none])

        elements_placed, elements_to_be_placed, num_actions = self.identify_valid_actions(state)

        return copy.deepcopy([np.array([np.random.choice(elements_to_be_placed), np.random.choice(elements_placed), np.random.choice(num_actions)]),\
                              self.dummy_act, self.dummy_obs, self.dummy_none, self.dummy_none, self.dummy_none])

    def identify_valid_actions(self,state):
        target_elements, elements_placed, elements_to_be_placed, num_actions = self.mdp._decode_observation(state)
        return elements_placed, elements_to_be_placed, num_actions


    def set_epsilon(self, epsilon):
        """
        Setter.

        Args:
            epsilon ((float, Parameter)): the exploration coefficient. It indicates the
            probability of performing a random actions in the current step.

        """
        self._epsilon = to_parameter(epsilon)

    def set_mdp(self,mdp):
        self.mdp = mdp

    def update(self, *idx):
        """
        Update the value of the epsilon parameter at the provided index (e.g. in
        case of different values of epsilon for each visited state according to
        the number of visits).

        Args:
            *idx (list): index of the parameter to be updated.

        """
        self._epsilon.update(*idx)


class EpsGreedyMultidimensionalMCTS(EpsGreedyMultidimensional):
    """
    Epsilon-MCTS implementation
    """

    def __init__(self, epsilon, mdp, normalizer, mdp_args, mcts_args):
        print ("USING EPSILON_MCTS")
        """
        Constructor.

        Args:
            epsilon ((float, Parameter)): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step.

        """
        super().__init__(epsilon, mdp)

        self._epsilon = to_parameter(epsilon)
        self.mdp = mdp # this is needed to find out the allowed actions from the environment,...
        self.mdp_args = mdp_args
        self.mcts_args = mcts_args
        self.normalizer = normalizer
        # save search budget and create dummy items which are needed to be returned in case no search is being
        # conducted
        self.mcts_search_budget = (self.mcts_args[1]['searchBudget_iter'])
        self.dummy_obs = [self.mdp.info.observation_space.low]*self.mcts_search_budget
        self.dummy_act = [np.zeros(self.mdp.info.action_space.shape)]*self.mcts_search_budget
        self.dummy_none = [-100]*self.mcts_search_budget

        self.mcts_searcher = MCTS(mdp=mdp, mdp_params=mdp_args, normalizer=normalizer, ref_policy=self, **self.mcts_args[1])

        self._add_save_attr(_epsilon='mushroom')

    def set_mdp(self,mdp):
        # functionality to set the mdp of this searcher. this is needed as we will later spawn multiple workers that
        # are conducting this epsilon-MCTS procedure,...
        self.mdp = mdp
        self.mcts_searcher = MCTS(mdp=self.mdp, mdp_params=self.mdp_args, normalizer=self.normalizer, ref_policy=self, **self.mcts_args[1])

    def draw_action(self, state):
        # EPSILON GREEDY - MCTS, in case of smaller than epsilon -> conduct search
        if not np.random.uniform() < self._epsilon(state):
            next_action, transitions, state_counter, explored_actions, expl_actions_next_states, explored_sing_step_r, explored_next_states_absorbing, explored_next_states_value = self.mcts_searcher.search(initialPosition=state)

            return [next_action, explored_actions, expl_actions_next_states, explored_sing_step_r, explored_next_states_absorbing, explored_next_states_value]

        # else: get all available actions and select a random one,...
        elements_placed, elements_to_be_placed, num_actions = self.identify_valid_actions(state)

        return copy.deepcopy([np.array([np.random.choice(elements_to_be_placed), np.random.choice(elements_placed), np.random.choice(num_actions)]),\
    self.dummy_act, self.dummy_obs, self.dummy_none, self.dummy_none, self.dummy_none])

        # # Potential slightly different implementation - even though taking a random action -> always return all actions that one has searched for
        # elements_placed, elements_to_be_placed, num_actions = self.identify_valid_actions(state)
        # next_action, transitions, state_counter, explored_actions, expl_actions_next_states, explored_sing_step_r, explored_next_states_absorbing, explored_next_states_value = self.mcts_searcher.search(
        #     initialPosition=state)
        #
        # return copy.deepcopy([np.array([np.random.choice(elements_to_be_placed), np.random.choice(elements_placed),
        #                                 np.random.choice(num_actions)]), \
        #                       explored_actions, expl_actions_next_states, explored_sing_step_r,
        #                       explored_next_states_absorbing, explored_next_states_value])

    def get_epsilon_random_action(self):
        if not np.random.uniform() < self._epsilon(None):
            return False # i.e. no random action
        else:
            return True

    def draw_action_epsilon(self, state, blocked_actions=[], use_mean_std=False):
        # draw an action using epsilon greedy, however some actions might be blocked as they have already been explored
        # case 1: choose maximum action:
        if not np.random.uniform() < self._epsilon(state):
            if (not use_mean_std):
                q = self._approximator.predict(state)
            else:
                q = self._approximator.predict(state,mean_std=True)
            if (len(blocked_actions)>0):
                for i in range(len(blocked_actions)):
                    q[blocked_actions[i]] = -np.inf

            elements_placed, elements_to_be_placed, num_actions = self.identify_valid_actions(state)

            q = q[elements_to_be_placed][:,elements_placed]

            max_a = np.argwhere(q == np.max(q))
            if len(max_a) > 1:
                max_a = max_a[np.random.choice(len(max_a))]
            else:
                max_a = max_a[0]

            # to account for the filtering we did before
            max_a[0] = elements_to_be_placed[max_a[0]]
            max_a[1] = elements_placed[max_a[1]]

            return max_a

        # case 2: choose random action
        elements_placed, elements_to_be_placed, num_actions = self.identify_valid_actions(state)

        if (len(blocked_actions) == 0):
            return np.array([np.random.choice(elements_to_be_placed), np.random.choice(elements_placed), np.random.choice(num_actions)])
        else:
            # random choice until valid action which is not inside the blocked ones
            while (True):
                guess = np.array([np.random.choice(elements_to_be_placed), np.random.choice(elements_placed), np.random.choice(num_actions)])
                if (self.to_tupel(guess) not in (blocked_actions)):
                    return guess

    def to_tupel(self, input):
        return tuple(list(input))


    def get_value_estimate(self,state):
        q = self._approximator.predict(state)
        elements_placed, elements_to_be_placed, num_actions = self.identify_valid_actions(state)
        q = q[elements_to_be_placed][:, elements_placed]
        return np.max(q)

    def get_q_estimate(self,state,idx):
        q = self._approximator.predict(state)

        if (len(np.shape(idx))==1):
            return q[idx]
        else:
            shape_q = np.shape(q)
            idx = idx[:,0]*(shape_q[-1]*shape_q[-2])+idx[:,1]*shape_q[-1]+idx[:,2]
            q = q.reshape(-1)
            return (q[idx])


class QMultidimensionalMCTS(EpsGreedyMultidimensionalMCTS):
    """
    Q-MCTS policy implementation for a multidimensional task
    """

    def __init__(self, epsilon, mdp, normalizer, mdp_args, mcts_args):
        print("USING Q_MCTS")
        """
        Constructor.

        Args:
            epsilon ((float, Parameter)): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step.

        """
        super().__init__(epsilon, mdp, normalizer, mdp_args, mcts_args)

    def draw_action(self, state):
        # in Q-MCTS we always conduct search
        next_action, transitions, state_counter, explored_actions, expl_actions_next_states, explored_sing_step_r, explored_next_states_absorbing, explored_next_states_value = self.mcts_searcher.search(initialPosition=state)
        return [next_action, explored_actions,expl_actions_next_states,explored_sing_step_r, explored_next_states_absorbing, explored_next_states_value]




class Boltzmann(TDPolicy):
    """
    Boltzmann softmax policy.

    """
    def __init__(self, beta):
        """
        Constructor.

        Args:
            beta ((float, Parameter)): the inverse of the temperature distribution. As
            the temperature approaches infinity, the policy becomes more and
            more random. As the temperature approaches 0.0, the policy becomes
            more and more greedy.

        """
        super().__init__()
        self._beta = to_parameter(beta)

        self._add_save_attr(_beta='mushroom')

    def __call__(self, *args):
        state = args[0]
        q_beta = self._approximator.predict(state) * self._beta(state)
        q_beta -= q_beta.max()
        qs = np.exp(q_beta)

        if len(args) == 2:
            action = args[1]

            return qs[action] / np.sum(qs)
        else:
            return qs / np.sum(qs)

    def draw_action(self, state):
        return np.array([np.random.choice(self._approximator.n_actions,
                                          p=self(state))])

    def set_beta(self, beta):
        """
        Setter.

        Args:
            beta ((float, Parameter)): the inverse of the temperature distribution.

        """
        self._beta = to_parameter(beta)

    def update(self, *idx):
        """
        Update the value of the beta parameter at the provided index (e.g. in
        case of different values of beta for each visited state according to
        the number of visits).

        Args:
            *idx (list): index of the parameter to be updated.

        """
        self._beta.update(*idx)


class Mellowmax(Boltzmann):
    """
    Mellowmax policy.
    "An Alternative Softmax Operator for Reinforcement Learning". Asadi K. and
    Littman M.L.. 2017.

    """

    class MellowmaxParameter(Parameter):
        def __init__(self, outer, omega, beta_min, beta_max):
            self._omega = omega
            self._outer = outer
            self._beta_min = beta_min
            self._beta_max = beta_max

            self._add_save_attr(
                _omega='primitive',
                _outer='primitive',
                _beta_min='primitive',
                _beta_max='primitive',
            )

        def __call__(self, state):
            q = self._outer._approximator.predict(state)
            mm = (logsumexp(q * self._omega(state)) - np.log(
                q.size)) / self._omega(state)

            def f(beta):
                v = q - mm
                beta_v = beta * v
                beta_v -= beta_v.max()

                return np.sum(np.exp(beta_v) * v)

            try:
                beta = brentq(f, a=self._beta_min, b=self._beta_max)
                assert not (np.isnan(beta) or np.isinf(beta))

                return beta
            except ValueError:
                return 0.

    def __init__(self, omega, beta_min=-10., beta_max=10.):
        """
        Constructor.

        Args:
            omega (Parameter): the omega parameter of the policy from which beta
                of the Boltzmann policy is computed;
            beta_min (float, -10.): one end of the bracketing interval for
                minimization with Brent's method;
            beta_max (float, 10.): the other end of the bracketing interval for
                minimization with Brent's method.

        """
        beta_mellow = self.MellowmaxParameter(self, omega, beta_min, beta_max)

        super().__init__(beta_mellow)

    def set_beta(self, beta):
        raise RuntimeError('Cannot change the beta parameter of Mellowmax policy')

    def update(self, *idx):
        raise RuntimeError('Cannot update the beta parameter of Mellowmax policy')
