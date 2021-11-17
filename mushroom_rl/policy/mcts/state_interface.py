import copy
import numpy as np


class GridState:
    """
    State of the environment. This connects the mdp environment with the required variables and calls needed for the
    MCTS to run in this environment.
    """
    def __init__(self, position, reward, isterminal, mdp, options, opt_act_steps):
        """
        Initialize object with all relevant variable
        :param position: state ID of the current grid state
        :param reward: reward obtained in that state
        :param isterminal: if True the state of the environment is terminal
        :param mdp: the environment
        :param options: set of options
        :param opt_act_steps: number of steps taken in each option, can be array with multiple possible step lengths
        """
        self.position = position
        self.reward = reward
        self.isterminal = isterminal
        self.initialexpansion = False # this is only needed for special expansion policies,...
        self.mdp = mdp
        self.options = options
        self.opt_act_steps = opt_act_steps
        if isinstance(self.opt_act_steps, int):
            self.len_opt_act_steps = 1
        else:
            self.len_opt_act_steps = len(opt_act_steps)

    def getPossibleActions(self, primitive=False):
        """
        Returns the set of possible actions. This can include options and also options with different step length
        :param primitive: if True only primitive actions are returned
        :return: set of possible actions
        """
        if self.isTerminal():
            return []
        else:
            # print (self.position)
            target, placed_blocks, to_be_placed, num_actions = self.mdp._decode_observation(self.position)
            actionset = []
            for i in range(len(to_be_placed)):
                for j in range(len(placed_blocks)):
                    for k in range(len(num_actions)):
                        actionset.append((to_be_placed[i],placed_blocks[j],k))

            return actionset

    def rollAction(self, action, steps=None):
        """
        Takes action that changes this GridState
        :param action: actionID indicating the action to be taken
        :param steps: if defined: number of steps that should be taken
        :return: reward and number of steps that were taken
        """
        self.mdp.reset(self.position)
        position, reward, isterminal, x = self.mdp.step(action)
        self.isterminal = isterminal
        self.reward = reward
        self.position = position
        num_act_steps = 1
        return self.reward, num_act_steps

    def takeStepAction(self, action):
        """
        Takes action that changes this GridState but only one step even for option
        :param action: actionID indicating the action to be taken
        :return:
        """
        self.mdp.reset(self.position)

        position, reward, isterminal, x = self.mdp.step(action)
        self.isterminal = isterminal
        self.reward = reward
        self.position = position


    def simulate_step(self, action):
        """
        Does not change the GridState itself but only simulates the return values for the next states
        :param action: actionID indicating the action to be taken
        :return: tuples of position, reward and isterminal for each step taken
        """
        self.mdp.reset(self.position)

        env_steps = []

        position, reward, isterminal, x = self.mdp.step(action)

        env_steps.append((position, reward, isterminal))

        return env_steps

    def isTerminal(self):
        """
        :return: returns true if the GridState is terminal
        """
        return self.isterminal

    def getReward(self):
        """
        :return: Returns reward of the GridState
        """
        return self.reward


class GridWorld:
    """
    GridWorld is a second class linking the mdp environment to the algorithm.
    It is based on a 2d-array grid that can be used for visualization.
    """
    def __init__(self, experiment):
        """
        Initialize GridWorld object with given parameters.
        :param experiment: name of the gridworld (environment) to be used
        """
        # initialize grid
        grid_txt = open(experiment)
        self.grid = []
        for line in grid_txt:
            self.grid.append([a for a in line.rstrip()])

        # set starting pos
        self.initial_state = (0, 0)
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if self.grid[x][y] == 'S':
                    self.initial_state = (x, y)

        self.goal_state = (0, 0)
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if self.grid[x][y] == 'G':
                    self.goal_state = (x, y)

        self.goal_reward = 10

        self.agent_pos = self.initial_state

    def state_terminal(self, state):
        """
        :param state: query state
        :return: True if query state is goal state (terminal), false if not
        """
        if state[0] == self.goal_state[0] and state[1] == self.goal_state[1]:
            return True
        else:
            return False

    def set_state(self, state, value):
        """
        Sets a value in the 2d-array grid for a given stateID.
        :param state: stateID to be set
        :param value: value to set to the state
        """
        counter = 0
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if self.grid[x][y] != '#':
                    if counter == state:
                        self.grid[x][y] = value
                        return
                    else:
                        counter += 1

    def get_stateID(self, position):
        """
        :param position: x,y position
        :return: Returns the state ID for a given position
        """
        counter = 0
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if self.grid[x][y] != '#':
                    if position[0] == x and position[1] == y:
                        return counter
                    else:
                        counter += 1

    def get_pos(self, stateID):
        """
        :param stateID: ID of state of environment
        :return: x,y tuple of grid position for given state
        """
        counter = 0
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if self.grid[x][y] != '#':
                    if counter == stateID:
                        return x, y
                    else:
                        counter += 1

    def state_reward(self, state):
        """
        :param state: query state in x,y position
        :return: Returns the reward of the query state
        """
        if state[0] == self.goal_state[0] and state[1] == self.goal_state[1]:
            return self.goal_reward
        else:
            return 0

    def env_step(self, action):
        """
        Takes an environment step for a given action
        :param action: primitive action 0-3
        :return: the agent position after executing the action
        """
        act = 0
        if (action == 0):
            act = (-1, 0)
        elif (action == 1):
            act = (1, 0)
        elif (action == 2):
            act = (0, -1)
        elif (action == 3):
            act = (0, 1)
        else:
            print("ERROR: Implausible Action!")

        desired_pos = (self.agent_pos[0] + act[0], self.agent_pos[1] + act[1])
        if self.grid[desired_pos[0]][desired_pos[1]] != '#':
            self.agent_pos = desired_pos
            return desired_pos
        else:
            return self.agent_pos
