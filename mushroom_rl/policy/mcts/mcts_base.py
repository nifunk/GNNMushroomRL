from __future__ import division
from mushroom_rl.policy.mcts.state_interface import GridState
import time
import math
import random
import numpy as np
from mushroom_rl.utils.function_calls import wrapped_call
import copy


class treeNode:
    """
    Class contaning the nodes of the search tree built by MCTS
    """
    def __init__(self, state, parent):
        """
        Initialize Node of Tree
        :param state: the state ID
        :param parent: the parant of the node
        """
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.stateValue = 0
        self.children = {}

    def __str__(self):
        s=[]
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))


class MCTS:
    """
    Class containing the MCTS algorithm
    """
    def __init__(self, mdp, mdp_params, normalizer, ref_policy, rolloutPolicy = 'e_greedy', searchBudget_iter=5, \
                 rollout_depth=0, num_avg_rollouts=1, expansionPolicy="e_greedy", allowRepeatBeforeExpansion=False, \
                 uctExplConst=1.0, act_select_eps=True, priorEstimate=True, eval=False):
        """
        Initialize all relevant hyperparameters for MCTS algorithm
        :param experiment: name of gridworld (environment) to be used
        :param timeLimit: timeLimit for one MCTS run
        :param iterationLimit: iterationLimit for one MCTS run
        :param explorationConstant: value for PUCT
        :param rolloutPolicy: can be either e_greedy or randomPolicy
        :param usePrior: if True the prior is used for PUCT
        :param actionValueFct: the current action-value function used for rollouts
        :param epsilon_start: starting value for e_greedy (greediness: usually starting at 1.0 only exploration)
        :param gamma: discount factor
        :param num_rollouts: number of rollouts to be carried out for one node evaluation
        :param options: options to be used alongside the primitive actions
        :param opt_act_steps: steps of how long an option is being followed
        :param laplaceLearn: if True data is collected in the rollouts
        :param maxTransSamples: number of transition samples to be collected
        """
        # self.mdp = grid_generator.generate_grid_world(grid=experiment, prob=1.0, pos_rew=1, neg_rew=0)

        mdp_params[1]["visualize"] = False
        # self.mdp = wrapped_call(type(mdp),mdp_params[0],mdp_params[1])
        self.mdp = wrapped_call(type(mdp),mdp_params[0],mdp_params[1])
        self.mdp.reset()
        self.normalizer = normalizer
        self.reference_policy = ref_policy
        rolloutPolicy = rolloutPolicy #'e_greedy' # 'randomPolicy'
        # exploration constant is c in the additional term,...
        explorationConstant = -1.0
        timeLimit = None #-> only take iteration limit into account in this setting
        iterationLimit = searchBudget_iter
        gamma = self.mdp.info.gamma
        self.act_select_eps = act_select_eps
        self.priorEstimate = priorEstimate
        self.priorProbabilities = None
        self.actionValueFct = None
        self.epsilon = 1.0
        self.epsilon_initial = 1.0
        self.num_rollouts = num_avg_rollouts
        self.rollout_depth = rollout_depth
        self.options = None
        self.opt_act_steps = 6
        self.collectData = False
        self.maxTransSamples = 1000000
        self.state_counter = None
        self.transitions = []
        self.eval = eval

        # either use time or iteration limit or both,...
        if timeLimit is not None:
            if iterationLimit is not None:
                self.timeLimit = timeLimit
                self.searchLimit = iterationLimit
                self.limitType = 'iterations_timed'
            # time taken for each MCTS search in milliseconds
            else:
                self.timeLimit = timeLimit
                self.limitType = 'time'
        else:
            if iterationLimit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        if rolloutPolicy == "randomPolicy":
            self.rollout = self.randomPolicy
        elif rolloutPolicy == "e_greedy":
            self.rollout = self.e_greedy_policy

        if expansionPolicy == "e_greedy":
            self.expansion_fct = self.epsilon_expansion
        elif expansionPolicy == "e_greedy_special":
            self.expansion_fct = self.epsilon_expansion_special
        elif expansionPolicy == "random":
            self.expansion_fct = self.random_expansion
        elif expansionPolicy == "uct":
            self.expansion_fct = self.uct_expansion


        self.allow_repeat_before_expansion = allowRepeatBeforeExpansion
        self.uct_expl_const = uctExplConst


        self.step_count = 0
        self.gamma = gamma
        self.e_threshold = None

    def search(self, initialPosition, usePrior=None, actionValueFct=None, step_count=None,
               start_e_decay=False):
        """
        Run MCTS from current state to find best action
        :param initialPosition: current state (root node)
        :param usePrior: if True the priors are used in PUCT
        :param actionValueFct: current action value function
        :param step_count: current count of steps taken in the environment
        :param start_e_decay: if True the e-decay is started
        :return: best action, transitions and statistics
        """
        # FIRST: UNNORMALIZE
        # here case distinction is necessary that both the evaluation and training work
        if (self.eval):
            initialPosition = self.normalizer.unnormalize(copy.deepcopy(initialPosition))
        else:
            initialPosition = self.normalizer.unnormalize(copy.deepcopy(initialPosition.detach().cpu().numpy()))

        self.root = treeNode(GridState(initialPosition, 0, False, self.mdp, self.options, self.opt_act_steps), None)
        self.priorProbabilities = usePrior
        self.actionValueFct = actionValueFct
        self.step_count = step_count

        if start_e_decay:
            if self.e_threshold is None:
                self.e_threshold = step_count
            self.epsilon = self.epsilon_initial*np.exp(-(1/((20+1000/self.searchLimit)*self.e_threshold))*(step_count-self.e_threshold))

            if self.epsilon < 0.1:
                self.epsilon = 0.1
        else:
            self.epsilon = 1.0

        # run MCTS with specified Limit type
        # here we basically do the expansion, i.e. the actual search,...
        num_iters = 0
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
                num_iters += 1
        elif self.limitType == 'iterations_timed':
            timeLimit = time.time() + self.timeLimit / 1000
            for i in range(self.searchLimit):
                if time.time() > timeLimit:
                    break
                self.executeRound()
                num_iters += 1
        else:
            for i in range(self.searchLimit):
                self.executeRound()
                num_iters += 1


        # evaluate / get the best action and convert it to the right format
        bestChild = self.getBestChild(self.root, False, self.priorProbabilities, final=True, eps_selection=self.act_select_eps)
        explored_actions = []
        explored_next_states = []
        explored_sing_step_r = []
        explored_next_states_absorbing = []
        explored_next_states_value = []
        for action, node in self.root.children.items():
            # different selection mode depending on the expansion type
            if (self.expansion_fct == self.uct_expansion):
                if not(node.state.position is None):
                    explored_next_states.append(node.state.position)
                    explored_actions.append(np.asarray(list(action)))
                    explored_sing_step_r.append(node.state.reward)
                    explored_next_states_absorbing.append(node.state.isterminal)
                    explored_next_states_value.append(node.stateValue)
            else:
                explored_next_states.append(node.state.position)
                explored_actions.append(np.asarray(list(action)))
                explored_sing_step_r.append(node.state.reward)
                explored_next_states_absorbing.append(node.state.isterminal)
                explored_next_states_value.append(node.stateValue)
        action = (action for action, node in self.root.children.items() if node is bestChild).__next__()
        return action, self.transitions, self.state_counter, explored_actions, explored_next_states, explored_sing_step_r, explored_next_states_absorbing, explored_next_states_value

    def executeRound(self):
        """
        execute a selection-expansion-simulation-backpropagation round
        """
        # selects which action to take and simulates the action -> we end up in a node state that is returned
        node = self.selectNode(self.root)
        # we now determine the reward of this node state
        reward = self.rollout(node.state, self.mdp)
        # we backpropagate the information from the node and the reward
        self.backpropogate(node, reward)

    def selectNode(self, node):
        """
        Selection step: first node that is not fully expanded (traversing the tree using getBestChild --> PUCT)
        :param node: node to start the selection process
        :return: selected node
        """
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, False, self.priorProbabilities)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        """
        Expansion step: expand current node by choosing a new action and adding the new Node to the Tree
        :param node: node to expand
        :return: newNode that is expanded
      actions  """
        actions = node.state.getPossibleActions()
        len_act = len(actions)
        return self.expansion_fct(actions,node,len_act)

    def random_expansion(self, actions, node, len_act):
        # expand randomly
        while len(actions) > 0:
            action = actions[np.random.randint(0, len(actions))]

            if action not in node.children:
                env_steps = node.state.simulate_step(action)

                newNode = treeNode(GridState(env_steps[0][0], env_steps[0][1], env_steps[0][2],
                                             self.mdp, self.options, self.opt_act_steps), node)
                # add prior guess of reward and also increment visitation count:
                if (self.priorEstimate):
                    newNode.numVisits += 1
                    newNode.totalReward += self.reference_policy.get_q_estimate(self.normalizer(copy.deepcopy(node.state.position)),action)
                    node.numVisits += 1
                    next_node = node.parent
                    # recursively increase visit counts for all nodes
                    while next_node is not None:
                        next_node.numVisits += 1
                        next_node = next_node.parent

                node.children[action] = newNode

                if len_act == len(node.children):
                    node.isFullyExpanded = True

                return newNode
            else:
                if (self.allow_repeat_before_expansion):
                    # allow recursive calling
                    next_root = node.children[action]
                    return self.selectNode(next_root)
                else:
                    actions.remove(action)

        raise Exception("Should never reach here")

    def to_tupel(self,input):
        return tuple(list(input))

    def to_list(self,input):
        return list(input)

    def epsilon_expansion_special(self, actions, node, len_act):
        return self.epsilon_expansion(actions, node, len_act, use_mean_std=True)

    def epsilon_expansion(self, actions, node, len_act,use_mean_std=False):
        forbidden_actions = []  # keep track of "invalid actions"
        while len(actions) > 0:
            encoded_obs = self.normalizer(copy.deepcopy(node.state.position))
            action = self.reference_policy.draw_action_epsilon(encoded_obs,blocked_actions=forbidden_actions, use_mean_std=use_mean_std)

            if self.to_tupel(action) not in node.children:
                env_steps = node.state.simulate_step(action)

                newNode = treeNode(GridState(env_steps[0][0], env_steps[0][1], env_steps[0][2],
                                             self.mdp, self.options, self.opt_act_steps), node)
                # add prior guess of reward and visitation count:
                if (self.priorEstimate):
                    newNode.numVisits += 1
                    newNode.totalReward += self.reference_policy.get_q_estimate(self.normalizer(copy.deepcopy(node.state.position)),self.to_tupel(action))
                    node.numVisits += 1
                    next_node = node.parent
                    # recursively increase visit counts for all nodes
                    while next_node is not None:
                        next_node.numVisits += 1
                        next_node = next_node.parent

                node.children[self.to_tupel(action)] = newNode

                # nods that are not at the end of an option are set to be fully expanded (not further evaluated)
                if len_act == len(node.children):
                    node.isFullyExpanded = True

                return newNode
            else:
                if (self.allow_repeat_before_expansion):
                    # allow recursive calling
                    next_root = node.children[self.to_tupel(action)]
                    return self.selectNode(next_root)
                else:
                    forbidden_actions.append(self.to_tupel(action))
                    actions.remove(self.to_tupel(action))

        raise Exception("Should never reach here")

    def uct_expansion(self, actions, node, len_act):
        # uct expansion
        # if not initial expansion - add all potential children - however without actually exploring them, simply init
        # their value from the Q-fct estimate,...
        if not(node.state.initialexpansion):
            all_values = self.reference_policy.get_q_estimate(self.normalizer(copy.deepcopy(node.state.position)),np.asarray(actions))
            for i in range(len(actions)):
                newNode = treeNode(GridState(None, None, None,
                                             self.mdp, self.options, self.opt_act_steps), node)
                # add prior guess of reward and visitation count:
                if (self.priorEstimate):
                    newNode.numVisits += 1
                    newNode.totalReward += all_values[i]
                    node.numVisits += 1
                    next_node = node.parent
                    # recursively increase visit counts for all nodes
                    while next_node is not None:
                        next_node.numVisits += 1
                        next_node = next_node.parent
                else:
                    newNode.totalReward += all_values[i]
                    newNode.numVisits += 1

                node.children[actions[i]] = newNode
            node.state.initialexpansion = True

        while len(actions) > 0:
            # initialize the uct search with the first action that is possible
            if (node.numVisits==0):
                init_bound = node.children[actions[0]].totalReward / node.children[actions[0]].numVisits
                init_ac = actions[0]
            else:
                init_bound = node.children[actions[0]].totalReward / node.children[actions[0]].numVisits + self.uct_expl_const \
                                * math.sqrt(math.log(node.numVisits) / node.children[actions[0]].numVisits)
                init_ac = actions[0]

            # now actually do the uct search by looping through all possible actions
            for i in range(len(actions)-1):
                if (node.numVisits == 0):
                    new_bound = node.children[actions[i + 1]].totalReward / node.children[
                        actions[i + 1]].numVisits
                else:
                    new_bound = node.children[actions[i+1]].totalReward / node.children[actions[i+1]].numVisits + self.uct_expl_const \
                                 * math.sqrt(math.log(node.numVisits) / node.children[actions[i+1]].numVisits)
                # if there is a new bound that is better than the previous one - update the action that should be taken,...
                if (new_bound>init_bound or (new_bound==init_bound and np.random.rand()<0.5)):
                    init_bound = new_bound
                    init_ac = actions[i+1]

            # from above we now actually have an action that is to be taken
            # -> now determine what to do next
            # is the state position of this Node is still None -> this transition has not been simulated before (i.e.
            # we might only already have assigned a value as done above) -> do the actual simulation now!
            if node.children[init_ac].state.position is None:
                env_steps = node.state.simulate_step(self.to_list(init_ac))
                # create proper node
                newNode = treeNode(GridState(env_steps[0][0], env_steps[0][1], env_steps[0][2],
                                             self.mdp, self.options, self.opt_act_steps), node)

                if (self.priorEstimate):
                    # transfer here the prior knowledge
                    newNode.numVisits = node.children[init_ac].numVisits
                    newNode.totalReward = node.children[init_ac].totalReward
                    node.children[init_ac] = newNode
                else:
                    newNode.numVisits = 0
                    newNode.totalReward = 0
                    node.children[init_ac] = newNode

                fully_expanded = True
                for key in node.children:
                    if (node.children[key].state.position is None):
                        fully_expanded = False
                        break # can immediately stop here

                if (fully_expanded):
                    node.isFullyExpanded = True
                return newNode
            else:
                if (self.allow_repeat_before_expansion):
                    # allow recursive calling
                    next_root = node.children[init_ac]
                    return self.selectNode(next_root)
                else:
                    actions.remove(init_ac)

        raise Exception("Should never reach here")


    def backpropogate(self, node, reward):
        """
        Backpropagation step: backpropagate rewards in the tree
        :param node: node to start backpropagation from
        :param reward: reward obtained for that node
        """
        discount = 0
        while node is not None:
            node.numVisits += 1
            if (discount==0):
                node.totalReward += (self.gamma**discount) * reward
                reward = (self.gamma**discount) * reward
                discount += 1
            else:
                node.totalReward += node.state.reward + (self.gamma) * reward
                reward = node.state.reward + (self.gamma) * reward
            node.stateValue = node.totalReward/node.numVisits
            node = node.parent

    def getBestChild(self, node, zero_exp, priorProbabilities, final=False, eps_selection=False):
        """
        Choose best child for current node based on the decision rule UCT or PUCT
        :param node: node to be evaluatedd
        :param zero_exp: if True set exploration Value to zero in order to only do exploitation
        :param priorProbabilities: probabilities of different actions
        :return: best action for node
        """
        bestValue = float("-inf")
        bestNodes = []
        if self.explorationConstant == -1:
            explorationValue = np.log((1 + node.numVisits))
        elif self.explorationConstant >= 0:
            explorationValue = self.explorationConstant
        else:
            raise Exception("Wrong explorationValue")
        if zero_exp:
            explorationValue = 0

        # Note: in the following there are always case distinctions between uct_expansion and non uct_expansion. The
        # reason for this is that for doing the uct expansion we had to add nodes to the search tree although they
        # have not been visited before and should therefore not be considered in this process of selecting the best child.

        if (final and eps_selection):
            explorationValue = 0
            rand_act = self.reference_policy.get_epsilon_random_action()
            if (rand_act):
                for child in node.children.values():
                    # choose randomly, i.e. append all childs as we anyways do a random selection at the return statement
                    if (self.expansion_fct == self.uct_expansion):
                        if not(child.state.position is None):
                            bestNodes.append(child)
                    else:
                        bestNodes.append(child)
            else:
                for child in node.children.values():
                    if (self.expansion_fct == self.uct_expansion):
                        if not(child.state.position is None):
                            nodeValue = child.totalReward / child.numVisits + explorationValue \
                                        * math.sqrt(math.log(node.numVisits) / child.numVisits)
                            if nodeValue > bestValue or (nodeValue==bestValue and np.random.rand()<0.5):
                                bestValue = nodeValue
                                bestNodes = [child]
                    else:
                        if (self.expansion_fct == self.random_expansion):
                            nodeValue = child.totalReward / child.numVisits + explorationValue \
                                        * math.sqrt(math.log(node.numVisits) / child.numVisits)
                        else:
                            nodeValue = child.totalReward / child.numVisits
                        if nodeValue > bestValue or (nodeValue==bestValue and np.random.rand()<0.5):
                            bestValue = nodeValue
                            bestNodes = [child]

        elif priorProbabilities is None: # UCT rule
            for child in node.children.values():
                if (self.expansion_fct == self.uct_expansion):
                    if not(child.state.position is None):
                        nodeValue = child.totalReward / child.numVisits + explorationValue \
                                    * math.sqrt(math.log(node.numVisits) / child.numVisits)
                        if nodeValue > bestValue or (nodeValue==bestValue and np.random.rand()<0.5):
                            bestValue = nodeValue
                            bestNodes = [child]
                else:
                    if (self.expansion_fct == self.random_expansion):
                        nodeValue = child.totalReward / child.numVisits + explorationValue \
                                    * math.sqrt(math.log(node.numVisits) / child.numVisits)
                    else:
                        nodeValue = child.totalReward / child.numVisits
                    if nodeValue > bestValue or (nodeValue==bestValue and np.random.rand()<0.5):
                        bestValue = nodeValue
                        bestNodes = [child]

        elif priorProbabilities is not None: # PUCT rule
            for child in node.children:
                if (self.expansion_fct == self.uct_expansion):
                    if not(child.state.position is None):
                        nodeValue = node.children[child].totalReward / node.children[child].numVisits + \
                                    explorationValue * self.priorProbabilities[node.state.position, child] \
                                    * math.sqrt(math.log(node.numVisits) / node.children[child].numVisits)
                        if nodeValue > bestValue or (nodeValue==bestValue and np.random.rand()<0.5):
                            bestValue = nodeValue
                            bestNodes = [node.children[child]]
                    else:
                        nodeValue = node.children[child].totalReward / node.children[child].numVisits + \
                                    explorationValue * self.priorProbabilities[node.state.position, child] \
                                    * math.sqrt(math.log(node.numVisits) / node.children[child].numVisits)
                        if nodeValue > bestValue or (nodeValue==bestValue and np.random.rand()<0.5):
                            bestValue = nodeValue
                            bestNodes = [node.children[child]]

        return random.choice(bestNodes)

    def randomPolicy(self, state, mdp):
        """
        Rollout step: using random policy to calculate rollouts
        :param state: state from which to start the rollouts
        :param mdp: mdp to use for rollouts
        :return: mean reward over all rollouts
        """
        rewards = []
        # do multiple rollouts and take the mean over them to improve accuracy
        for _ in range(self.num_rollouts):
            roll_state = GridState(state.position, state.reward, state.isterminal, mdp, self.options, self.opt_act_steps)
            discount = 0
            reward = state.reward

            while not roll_state.isterminal:
                try:
                    # select random action
                    action = random.choice(roll_state.getPossibleActions(primitive=True))
                except IndexError:
                    raise Exception("Non-terminal state has no possible actions: " + str(roll_state))
                if (discount==self.rollout_depth and not roll_state.isterminal):
                    #abort here -> use the estimation from the q-network, i.e. the value estimation and cut the rollout
                    reward += (self.gamma ** (discount+1)) * self.reference_policy.get_value_estimate(self.normalizer(copy.deepcopy(roll_state.position)))
                    break

                sing_step_reward, num_act_steps = roll_state.rollAction(action)
                reward += (self.gamma ** (discount+1)) * sing_step_reward
                discount += num_act_steps

            rewards.append(reward)
        rewards = np.array(rewards)
        return np.mean(rewards)

    def e_greedy_policy(self, state, mdp):
        """
        Rollout step: using e-greedy policy to calculate rollouts
        :param state: state from which to start the rollouts
        :param mdp: mdp to use for rollouts
        :return: mean reward over all rollouts
        """
        rewards = []
        # do multiple rollouts and take the mean over them to improve accuracy
        for _ in range(self.num_rollouts):
            roll_state = GridState(state.position, state.reward, state.isterminal, mdp, self.options, self.opt_act_steps)
            discount = 0
            reward = state.reward

            while not roll_state.isterminal:
                if (discount==self.rollout_depth and not roll_state.isterminal):
                    #abort here -> use the estimation from the q-network, i.e. the value estimation and cut the rollout
                    reward += (self.gamma ** (discount+1)) * self.reference_policy.get_value_estimate(self.normalizer(copy.deepcopy(roll_state.position)))
                    break
                encoded_obs = self.normalizer(copy.deepcopy(roll_state.position))
                action = self.reference_policy.draw_action_epsilon(encoded_obs)
                sing_step_reward, num_act_steps = roll_state.rollAction(action)
                reward += (self.gamma ** (discount+1)) * sing_step_reward
                discount += num_act_steps


            rewards.append(reward)

        rewards = np.array(rewards)
        return np.mean(rewards)
