import sys
sys.path.append("../../")
from mushroom_env import fill_volume_env_3D_multiple_obj_3Dac
from mushroom_env import fill_volume_env_3D_multiple_obj_3Dac_4
from mushroom_env import fill_volume_env_3D_multiple_obj_3Dac_wrap_robot
from mushroom_env import fill_volume_env_3D_multiple_obj_3Dac_wrap_robot_1
from mushroom_env import fill_volume_env_3D_multiple_obj_3Dac_wrap_robot_4
from mushroom_env import fill_volume_env_3D_multiple_obj_3Dac_more_objects_wrap_robot
from mushroom_env import fill_volume_env_3D_multiple_obj_3Dac_more_objects2_20acs
from mushroom_env import fill_volume_env_3D_multiple_obj_3Dac_more_objects

import argparse
import datetime
import pathlib
import os
import shutil
import inspect
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict


from mushroom_rl.algorithms.value import AveragedDQN, CategoricalDQN, DQN,\
    DoubleDQN, MaxminDQN, DuelingDQN, DQNMultidim
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core, Logger, CoreParallel
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy, EpsGreedyMultidimensional, EpsGreedyMultidimensionalMCTS, QMultidimensionalMCTS
from mushroom_rl.utils.dataset import compute_metrics
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory
from mushroom_rl.utils.seeds import fix_random_seed
from mushroom_rl.utils.function_calls import wrapped_call

from mushroom_rl.utils.preprocessors import MinMaxPreprocessor

from custom_networks.mpnn_multidim_attention import MPNN_Multidim_Attention, MPNN_Multidim_Full_Attention, MPNN_Single_Full_Attention, Pytorch_Transformer, MPNN_Multidim_Full_Attention_Multiple
from custom_networks.mpnn_multidim import MPNN_Multidim, MPNN_Multidim_MORE

import gc

"""
This script runs a custom env
"""


# This is the loss function that is used for training the agents
# Note: for the q- and e-MCTS agents, there is also a cross entropy regularization possible, which simply gets
# activated by setting the loss on the cross entropy loss != 0
class SpecialLoss(nn.Module):
    def __init__(self,batch_size,search_budget, weight_normal_loss=0.5,weight_ce_loss=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.search_budget = search_budget
        self.m = nn.Softmax(dim=1)
        self.weight_normal_loss = weight_normal_loss
        self.weight_ce_loss = weight_ce_loss
        print ("WEIGHTS NORMAL " + str(self.weight_normal_loss) + " WEIGHT CE: " + str(self.weight_ce_loss))

    def forward(self, input, target):
        normal_input = input[:self.batch_size]
        normal_target = target[:self.batch_size]
        loss1 = F.smooth_l1_loss(normal_input,normal_target)

        if (self.weight_ce_loss!=0):
            add_input = input[self.batch_size:]

            add_target = target[self.batch_size:]

            add_input = add_input.reshape((-1,int(self.search_budget)))
            add_target = add_target.reshape((-1,int(self.search_budget)))
            soft_input = torch.log(self.m(add_input))
            soft_target = self.m(add_target)
            loss2 = -(1/self.batch_size)*torch.sum((torch.sum(torch.multiply(soft_target,soft_input),dim=1)))
        if (self.weight_ce_loss!=0):
            return self.weight_normal_loss*loss1+self.weight_ce_loss*loss2
        else:
            return self.weight_normal_loss * loss1

def print_epoch(epoch, logger):
    logger.info('################################################################')
    logger.info('Epoch: %d' % epoch)
    logger.info('----------------------------------------------------------------')


def get_stats(dataset, logger):
    score = compute_metrics(dataset)
    logger.info(('min_reward: %f, max_reward: %f, mean_reward: %f,'
                ' games_completed: %d' % score))

    return score


def experiment():
    # Argument parser
    parser = argparse.ArgumentParser()

    arg_game = parser.add_argument_group('Game')
    arg_game.add_argument("--name",
                          type=str,
                          default='Custom Cube 2 ENV',
                          help='Gym ID of the Atari game.')

    arg_game.add_argument("--num-blocks",
                          type=int,
                          default=None,
                          help='Number of blocks to be placed')

    arg_game.add_argument("--env-grid",
                          type=int,
                          default=3,
                          help='Size of grid of env')

    arg_game.add_argument("--graph-fconnected",
                          type=int,
                          default=1,
                          help='Specify if graph should be fully connected (1) - or not (0)')

    arg_mem = parser.add_argument_group('Replay Memory')

    arg_mem.add_argument("--initial-replay-size", type=int, default=1000,
                         help='Initial size of the replay memory.')
    arg_mem.add_argument("--max-replay-size", type=int, default=150000,
                         help='Max size of the replay memory.')
    arg_mem.add_argument("--prioritized", action='store_true',
                         help='Whether to use prioritized memory or not.')

    arg_net = parser.add_argument_group('Deep Q-Network')
    arg_net.add_argument("--optimizer",
                         choices=['adadelta',
                                  'adam',
                                  'rmsprop',
                                  'rmspropcentered'],
                         default='adam',
                         help='Name of the optimizer to use.')
    arg_net.add_argument("--learning-rate", type=float, default=.0001,
                         help='Learning rate value of the optimizer.')
    arg_net.add_argument("--decay", type=float, default=.95,
                         help='Discount factor for the history coming from the'
                              'gradient momentum in rmspropcentered and'
                              'rmsprop')
    arg_net.add_argument("--epsilon", type=float, default=1e-8,
                         help='Epsilon term used in rmspropcentered and'
                              'rmsprop')

    arg_alg = parser.add_argument_group('Algorithm')
    arg_alg.add_argument("--algorithm", choices=['dqn', 'ddqn', 'adqn', 'mmdqn',
                                                 'cdqn', 'dueldqn'],
                         default='dqn',
                         help='Name of the algorithm. dqn is for standard'
                              'DQN, ddqn is for Double DQN and adqn is for'
                              'Averaged DQN.')
    arg_alg.add_argument("--n-approximators", type=int, default=1,
                         help="Number of approximators used in the ensemble for"
                              "AveragedDQN or MaxminDQN.")
    arg_alg.add_argument("--batch-size", type=int, default=32,
                         help='Batch size for each fit of the network.')
    arg_alg.add_argument("--target-update-frequency", type=int, default=5000,
                         help='Number of collected samples before each update'
                              'of the target network.')
    arg_alg.add_argument("--evaluation-frequency", type=int, default=100,
                         help='Number of collected samples before each'
                              'evaluation. An epoch ends after this number of'
                              'steps')
    arg_alg.add_argument("--train-frequency", type=int, default=4,
                         help='Number of collected samples before each fit of'
                              'the neural network.')
    arg_alg.add_argument("--max-steps", type=int, default=50000000,
                         help='Total number of collected samples.')
    arg_alg.add_argument("--final-exploration-frame", type=int, default=200000,
                         help='Number of collected samples until the exploration'
                              'rate stops decreasing.')
    arg_alg.add_argument("--initial-exploration-rate", type=float, default=1.,
                         help='Initial value of the exploration rate.')
    arg_alg.add_argument("--final-exploration-rate", type=float, default=.05,#.1
                         help='Final value of the exploration rate. When it'
                              'reaches this values, it stays constant.')
    arg_alg.add_argument("--test-exploration-rate", type=float, default=.05,
                         help='Exploration rate used during evaluation.')
    arg_alg.add_argument("--test-samples", type=int, default=10,
                         help='Number of collected samples for each'
                              'evaluation.')
    arg_alg.add_argument("--n-atoms", type=int, default=51,
                         help='Number of atoms for Categorical DQN.')
    arg_alg.add_argument("--v-min", type=int, default=-10,
                         help='Minimum action-value for Categorical DQN.')
    arg_alg.add_argument("--v-max", type=int, default=10,
                         help='Maximum action-value for Categorical DQN.')

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--use-cuda', action='store_true',
                           help='Flag specifying whether to use the GPU.')
    arg_utils.add_argument('--save', action='store_true',
                           help='Flag specifying whether to save the model.')
    arg_utils.add_argument('--save-path', type=str, default='',
                           help='Flag specifying whether to save the model.')
    arg_utils.add_argument('--load-path', type=str,
                           help='Path of the model to be loaded.')
    arg_utils.add_argument('--render', action='store_true',
                           help='Flag specifying whether to render the game.')
    arg_utils.add_argument('--quiet', action='store_true',
                           help='Flag specifying whether to hide the progress'
                                'bar.')
    arg_utils.add_argument('--debug', action='store_true',
                           help='Flag specifying whether the script has to be'
                                'run in debug mode.')
    arg_utils.add_argument('--seed', type=int, default=-1,
                           help='Flag specifying whether to save the model.')
    arg_utils.add_argument('--model', type=str, default='s2v',
                           help='Flag specifying whether to save the model.')
    arg_utils.add_argument('--env', type=str, default='default',
                           help='Specify which environment to use')

    arg_mcts = parser.add_argument_group('MCTS')
    arg_mcts.add_argument('--use-mcts', type=int, default=0,
                          help='Specify whether to use mcts or not')
    arg_mcts.add_argument('--num-workers', type=int, default=4,
                           help='Specify the number of workers for parallel sampling')
    arg_mcts.add_argument('--mcts-type', type=str, default='e_mcts',
                           help='Specify which MCTS version to run - e_mcts or q_mcts')
    arg_mcts.add_argument('--rollout-policy', type=str, default='e_greedy',
                           help='Specify the policy to use when performing rollouts/playouts')
    arg_mcts.add_argument('--search-budget-iter', type=int, default=10,
                           help='Specify the search budget for MCTS search')
    arg_mcts.add_argument('--rollout-depth', type=int, default=0,
                           help='Specify the depth of the rollout before using q-estimate')
    arg_mcts.add_argument('--num-avg-rollouts', type=int, default=1,
                           help='Specify over how much rollouts to normalize to estimate a nodes value')
    arg_mcts.add_argument('--expansion-policy', type=str, default='e_greedy',
                           help='Specify which policy to use to expand nodes')
    arg_mcts.add_argument('--allow-repeat-before-expansion', type=int, default=0,
                           help='Specify whether it is allowed to further explore although not all nodes have been visited')
    arg_mcts.add_argument('--uct-expl-const', type=float, default=2.0,
                           help='Specify exploration constant for uct expansion policy')
    arg_mcts.add_argument('--act-select-eps', type=int, default=1,
                           help='whether or not to use eps greedy selection inside search module')
    arg_mcts.add_argument('--prior-estimate', type=int, default=1,
                           help='whether or not to use the Q-prior, i.e. Q(s,a) in the search for estimating the value')

    arg_mcts.add_argument('--w-normal-loss', type=float, default=0.5,
                          help='Weight of the normal loss')
    arg_mcts.add_argument('--w-ce-loss', type=float, default=0.5,
                          help='Weight of the special ce loss')

    args = parser.parse_args()

    scores = list()

    optimizer = dict()
    if args.optimizer == 'adam':
        optimizer['class'] = optim.Adam
        optimizer['params'] = dict(lr=args.learning_rate,
                                   eps=args.epsilon)
    elif args.optimizer == 'adadelta':
        optimizer['class'] = optim.Adadelta
        optimizer['params'] = dict(lr=args.learning_rate,
                                   eps=args.epsilon)
    elif args.optimizer == 'rmsprop':
        optimizer['class'] = optim.RMSprop
        optimizer['params'] = dict(lr=args.learning_rate,
                                   alpha=args.decay,
                                   eps=args.epsilon)
    elif args.optimizer == 'rmspropcentered':
        optimizer['class'] = optim.RMSprop
        optimizer['params'] = dict(lr=args.learning_rate,
                                   alpha=args.decay,
                                   eps=args.epsilon,
                                   centered=True)
    else:
        raise ValueError

    # Summary folder -> path where to store results of the training
    if (args.save_path==''):
        folder_name = './logs/custom_' + args.algorithm + '_' + args.name +\
            '_' + datetime.datetime.now().strftime('%Y-%m-%d--%H_%M_%S_%f')
    else:
        folder_name = args.save_path + '/custom_' + args.algorithm + '_' + args.name +\
            '_' + datetime.datetime.now().strftime('%Y-%m-%d--%H_%M_%S_%f')

    if (args.save):
        pathlib.Path(folder_name).mkdir(parents=True)
        logger = Logger(DQN.__name__, results_dir=folder_name, log_console=True)
        writer = SummaryWriter(folder_name)
        loglist = defaultdict(list)
    else:
        logger = Logger(DQN.__name__, results_dir=None)
        writer = None
        loglist = None


    logger.strong_line()
    logger.info('Experiment Algorithm: ' + DQN.__name__)

    # Settings
    if args.debug:
        initial_replay_size = 50
        max_replay_size = 500
        train_frequency = 5
        target_update_frequency = 10
        test_samples = 20
        evaluation_frequency = 50
        max_steps = 1000
    else:
        initial_replay_size = args.initial_replay_size
        max_replay_size = args.max_replay_size
        train_frequency = args.train_frequency
        target_update_frequency = args.target_update_frequency
        test_samples = args.test_samples
        evaluation_frequency = args.evaluation_frequency
        max_steps = args.max_steps

    # Define the properties of the MDP:
    if (args.num_blocks is None):
        number_of_parts = (9+5)
    else:
        number_of_parts = args.num_blocks

    # define size of the grid, i.e. how large the potential target shape to be built can be
    env_grid_size = args.env_grid

    ensemble = True
    # every node in the graph has 5 features: 1) position (x,y,z) and two booleans indicating: target element / block ; block placed / unplaced
    dim_individual_feature = 5
    # each block has an observation dimension of its own features plus the adjaceny matrix, i.e. if it is connected
    # to other elements
    dim_obs_per_block = dim_individual_feature + number_of_parts


    # create a list which contains all of the arguments for the MDP as well as MCTS as due to parallelization we will
    # need them multiple times -> passing list will make creation of MDP's and MCTS agents simpler
    mdp_args=[]
    unnamed_args = [number_of_parts]
    named_args = {"visualize": False, "add_connectivity": True, "ensemble": ensemble, "env_grid_size": env_grid_size, "fully_connected": bool(args.graph_fconnected), \
                  "load_path": args.load_path}
    mdp_args.append(unnamed_args)
    mdp_args.append(named_args)

    mcts_args=[]
    unnamed_args = []
    named_args = {"rolloutPolicy": args.rollout_policy, "searchBudget_iter": args.search_budget_iter, "rollout_depth": args.rollout_depth, \
                  "num_avg_rollouts": args.num_avg_rollouts, "expansionPolicy": args.expansion_policy, \
                  "allowRepeatBeforeExpansion": bool(args.allow_repeat_before_expansion), "uctExplConst": args.uct_expl_const, \
                  "act_select_eps": bool(args.act_select_eps), "priorEstimate": bool(args.prior_estimate), "eval": bool(args.load_path)}
    mcts_args.append(unnamed_args)
    mcts_args.append(named_args)

    # if the load path is specified -> we want to evaluate -> we enable visualization
    if args.load_path:
        mdp_args[1]["visualize"] = True

    # select which environment we want to train on
    # TODO: make this selection process cleaner!

    # variable robot_state_dim is needed to indicate whether the robot's state is also included in the observation
    # or not -> having robot state in observation is needed to conduct proper MCTS search as environment has to be
    # reset / set appropriately
    if (args.env == 'default'):
        print ("No environment has been specified -> terminating")
        return 0
    elif (args.env == '2-wo-robo'):
        mdp = wrapped_call(fill_volume_env_3D_multiple_obj_3Dac.StackBoxesEnv3D_multiple_obj,mdp_args[0],mdp_args[1])
        robot_state_dim = 0
    elif (args.env == '4-wo-robo'):
        mdp = wrapped_call(fill_volume_env_3D_multiple_obj_3Dac_4.StackBoxesEnv3D_multiple_obj_4,mdp_args[0],mdp_args[1])
        robot_state_dim = 0
    elif (args.env == '1-wo-robo'):
        mdp = wrapped_call(fill_volume_env_3D_multiple_obj_3Dac_more_objects.StackBoxesEnv3D_multiple_obj,mdp_args[0],mdp_args[1])
        robot_state_dim = 0
    elif (args.env == '2-wo-robo-more-obj'):
        mdp = wrapped_call(fill_volume_env_3D_multiple_obj_3Dac_more_objects2_20acs.StackBoxesEnv3D_multiple_obj,mdp_args[0],mdp_args[1])
        robot_state_dim = 0
    elif (args.env == '1-w-robo'):
        mdp = wrapped_call(fill_volume_env_3D_multiple_obj_3Dac_wrap_robot_1.StackBoxesEnv3D_multiple_obj_w_robot,mdp_args[0],mdp_args[1])
        robot_state_dim = 2 * 9
    elif (args.env == '2-w-robo'):
        mdp = wrapped_call(fill_volume_env_3D_multiple_obj_3Dac_wrap_robot.StackBoxesEnv3D_multiple_obj_w_robot,mdp_args[0],mdp_args[1])
        robot_state_dim = 2 * 9
    elif (args.env == '4-w-robo'):
        mdp = wrapped_call(fill_volume_env_3D_multiple_obj_3Dac_wrap_robot_4.StackBoxesEnv3D_multiple_obj_w_robot,mdp_args[0],mdp_args[1])
        robot_state_dim = 2 * 9
    elif (args.env == '2-w-robo-more-obj'):
        mdp = wrapped_call(fill_volume_env_3D_multiple_obj_3Dac_more_objects_wrap_robot.StackBoxesEnv3D_multiple_obj_w_robot,mdp_args[0],mdp_args[1])
        robot_state_dim = 2 * 9

    # set visualization false in the argument list -> all other envs except for the one created above do not have any
    # visualization
    mdp_args[1]["visualize"] = False
    # normalization callback
    normalizer = MinMaxPreprocessor(mdp_info=mdp.info)

    # do the seeding:
    if (args.seed==-1 and args.load_path):
        seed = 0
    elif (args.seed==-1):
        seed = random.randint(0, 1000)
    else:
        seed = args.seed
    fix_random_seed(seed,mdp)

    # if loading an agent -> this basically corresponds to evaluating it
    if args.load_path:
        # Load Agent
        agent = DQN.load(args.load_path)
        # # this has to be set when wanting to solve a task of different size,...
        agent.approximator.model.network.robot_state_dim = robot_state_dim
        agent.approximator.model.network.dim_whole_obs = dim_obs_per_block
        # # set the network to evaluation mode as for some models, dropout might be involved,...
        agent.approximator.model.network.eval()

        epsilon_test = Parameter(value=args.test_exploration_rate)

        if (args.use_mcts==0):
            pi = EpsGreedyMultidimensional(epsilon=epsilon_test, mdp=mdp)
        else:
            if (args.mcts_type=='e_mcts'):
                pi = EpsGreedyMultidimensionalMCTS(epsilon=0.05, mdp=mdp, normalizer=normalizer, mdp_args=mdp_args, mcts_args=mcts_args)
            elif (args.mcts_type=='q_mcts'):
                pi = QMultidimensionalMCTS(epsilon=0.05, mdp=mdp, normalizer=normalizer, mdp_args=mdp_args, mcts_args=mcts_args)

        agent.policy = pi
        pi.set_q(agent.approximator)
        agent.policy.set_epsilon(epsilon_test)
        agent.policy.set_mdp(mdp)


        # Algorithm
        core_test = Core(agent, mdp, preprocessors=[normalizer])

        # Evaluate model
        dataset = core_test.evaluate(n_steps=args.test_samples,
                                     render=args.render,
                                     quiet=args.quiet)
        get_stats(dataset,logger)

    else:
        # do the training
        # Policy
        epsilon = LinearParameter(value=args.initial_exploration_rate,
                                  threshold_value=args.final_exploration_rate,
                                  n=args.final_exploration_frame)
        epsilon_test = Parameter(value=args.test_exploration_rate)
        epsilon_random = Parameter(value=1)
        # depending on input argument choose appropriate policy to be used,...
        if (args.use_mcts==0):
            pi = EpsGreedyMultidimensional(epsilon=epsilon_random, mdp=mdp)
        else:
            if (args.mcts_type=='e_mcts'):
                pi = EpsGreedyMultidimensionalMCTS(epsilon=0.05, mdp=mdp, normalizer=normalizer, mdp_args=mdp_args, mcts_args=mcts_args)
            elif (args.mcts_type=='q_mcts'):
                pi = QMultidimensionalMCTS(epsilon=0.05, mdp=mdp, normalizer=normalizer, mdp_args=mdp_args, mcts_args=mcts_args)

        pi.set_epsilon(epsilon)

        # Approximator -> choose which model is to be used
        model_to_be_used = None
        if (args.model=="s2v"):
            model_to_be_used = MPNN_Multidim
        elif (args.model=="s2v_new"):
            model_to_be_used = MPNN_Multidim_MORE
        elif (args.model=="mha"):
            model_to_be_used = MPNN_Multidim_Attention
        elif (args.model=="mha_full"):
            model_to_be_used = MPNN_Multidim_Full_Attention
        elif (args.model=="sha_full"):
            model_to_be_used = MPNN_Single_Full_Attention
        elif (args.model=="mha_full_multiple"):
            model_to_be_used = MPNN_Multidim_Full_Attention_Multiple
        elif (args.model == "torch_transformer"):
            model_to_be_used = Pytorch_Transformer


        num_actions_available = mdp.info.action_space.high[-1] + 1

        # Approximator -> create the appropriate one by passing as input arguments the problem's properties
        approximator_params = dict(
            network=model_to_be_used,
            input_shape=mdp.info.observation_space.shape,
            output_shape=mdp.info.action_space.shape,
            #n_actions=mdp.info.action_space.shape,
            n_obs_in = dim_individual_feature,
            n_layers = 3,
            n_features=64,
            tied_weights=False,
            n_hid_readout = [],
            dim_whole_obs=dim_obs_per_block,
            num_actions_avail = num_actions_available,
            robot_state_dim = robot_state_dim,
            optimizer=optimizer,
            loss=SpecialLoss(args.batch_size,args.search_budget_iter,args.w_normal_loss,args.w_ce_loss) if  args.use_mcts!=0 else SpecialLoss(args.batch_size,args.search_budget_iter,1.0,0.0),#F.smooth_l1_loss,
            use_cuda=args.use_cuda,
            loglist=loglist
        )

        approximator = TorchApproximator

        if args.prioritized:
            print ("prioritized replay memory currently not supported in this parallel implementation, still has to \
                   be implemented")
            # replay_memory = PrioritizedReplayMemory(
            #     initial_replay_size, max_replay_size, alpha=.6,
            #     beta=LinearParameter(.4, threshold_value=1,
            #                          n=max_steps // train_frequency)
            # )
            replay_memory = None
        else:
            replay_memory = None

        # Agent
        algorithm_params = dict(
            batch_size=args.batch_size,
            target_update_frequency=target_update_frequency // train_frequency,
            replay_memory=replay_memory,
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size
        )

        if args.algorithm == 'dqn':
            agent = DQNMultidim(mdp.info, pi, approximator, mdp=mdp,
                        approximator_params=approximator_params,
                        **algorithm_params)
        elif args.algorithm == 'ddqn':
            print ("currently not supported properly in this repo")
            # agent = DoubleDQN(mdp.info, pi, approximator,
            #                   approximator_params=approximator_params,
            #                   **algorithm_params)
        elif args.algorithm == 'adqn':
            print ("currently not supported properly in this repo")
            # agent = AveragedDQN(mdp.info, pi, approximator,
            #                     approximator_params=approximator_params,
            #                     n_approximators=args.n_approximators,
            #                     **algorithm_params)
        elif args.algorithm == 'mmdqn':
            print ("currently not supported properly in this repo")
            # agent = MaxminDQN(mdp.info, pi, approximator,
            #                   approximator_params=approximator_params,
            #                   n_approximators=args.n_approximators,
            #                   **algorithm_params)
        elif args.algorithm == 'dueldqn':
            print ("currently not supported properly in this repo")
            # agent = DuelingDQN(mdp.info, pi,
            #                    approximator_params=approximator_params,
            #                    **algorithm_params)
        elif args.algorithm == 'cdqn':
            print ("currently not supported properly in this repo")
            # agent = CategoricalDQN(mdp.info, pi,
            #                        approximator_params=approximator_params,
            #                        n_atoms=args.n_atoms, v_min=args.v_min,
            #                        v_max=args.v_max, **algorithm_params)

        # Algorithm
        # here use the parallel version which collects the experience in parallel, i.e. we spawn multiple environments
        # in which we collect the experience. Afterwards, we do the update with all of the samples that have been
        # acquired
        core = CoreParallel(agent, mdp, args.use_cuda, preprocessors=[normalizer], mdp_args=mdp_args, num_workers=args.num_workers)

        # RUN

        # Fill replay memory with initial dataset
        print_epoch(0, logger)
        core.learn(n_steps=initial_replay_size,
                   n_steps_per_fit=initial_replay_size, quiet=args.quiet)

        if args.save:
            agent.save(folder_name + '/agent_0.msh')
            normalizer.save(folder_name + '/normalizer_0.msh')
            # copy the training file (i.e. the file here)
            shutil.copy2(os.path.abspath(__file__),folder_name + '/')
            # copy the environment file
            shutil.copy2(inspect.getfile(mdp.__class__), folder_name + '/')
            # copy the network setup:
            shutil.copy2(inspect.getfile(agent.approximator.model.network.__class__), folder_name + '/')
            # write the args to file:
            f = open(folder_name + '/arguments.txt', "a")
            f.write(str(sys.argv[1:]) + '\n')
            f.write('seed ' + str(seed))
            f.close()


        # Evaluate initial policy
        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_steps=test_samples, render=args.render,
                                quiet=args.quiet)
        scores.append(get_stats(dataset, logger))

        if (args.save):
            np.save(folder_name + '/scores.npy', scores)

        best_mean_rew = -1000
        # train for 5000 epochs -> this is hardcoded for now
        for n_epoch in range(1, 5000 + 1):
            # every 25 iterations create new workers. this "hack" was necessary as otherwise in the parallel sampling
            # procedure we observed memory leackage that could not be tracked down,...
            gc.collect()
            if (n_epoch%25==0):
                do_reset = True
            else:
                do_reset = False

            print_epoch(n_epoch, logger)
            logger.info('- Learning:')
            # learning step
            print (epsilon.get_value())
            pi.set_epsilon(epsilon)
            # mdp.set_episode_end(True)
            dataset_train = core.learn(n_steps=evaluation_frequency,
                       n_steps_per_fit=train_frequency, quiet=args.quiet, reset=do_reset)

            # only store every 100th model, but make a full save which stores everything, including the state of the replay
            # memory
            if args.save and (n_epoch%100)==0:
                agent.save(folder_name + '/agent_full_' + str(n_epoch) + '.msh', full_save=True)
                normalizer.save(folder_name + '/normalizer_full_' + str(n_epoch) + '.msh')
                # however, as storing the full model consumes lots of space -> periodically remove earlier checkpoints
                # to not consume too much space
                if (n_epoch>299) and not(((n_epoch-200)%500)==0):
                    os.remove(folder_name + '/agent_full_' + str(n_epoch-200) + '.msh')
                    os.remove(folder_name + '/normalizer_full_' + str(n_epoch-200) + '.msh')
            # only store every 50th model but this time without the full save -> this is feasible
            if args.save and (n_epoch%50)==0:
                agent.save(folder_name + '/agent_' + str(n_epoch) + '.msh')
                normalizer.save(folder_name + '/normalizer_' + str(n_epoch) + '.msh')

            logger.info('- Evaluation:')
            # evaluation step
            pi.set_epsilon(epsilon_test)
            # mdp.set_episode_end(False)
            # to speed up training the test set is exactly the same as the training set,...
            import copy
            dataset = copy.deepcopy(dataset_train)
            scores.append(get_stats(dataset, logger))


            # add logging to tensorboard file
            if (writer is not None):
                if (loglist is not None):
                    attribute_list = list(loglist.keys())
                    for i in range(len(attribute_list)):
                        # write mean of logging
                        writer.add_scalar(str(attribute_list[i]), np.mean(loglist[attribute_list[i]]), n_epoch)
                        # empty list again -> ready for new values,...
                        loglist[attribute_list[i]] = []

                # Add a few fixed things to the tensorboard log
                min_train_rew, max_train_rew, mean_train_rew, num_train_games_completed = compute_metrics(dataset_train)
                min_test_rew, max_test_rew, mean_test_rew, num_test_games_completed = compute_metrics(dataset)
                writer.add_scalar('Train/min_rew', min_train_rew, n_epoch)
                writer.add_scalar('Train/max_rew', max_train_rew, n_epoch)
                writer.add_scalar('Train/mean_rew', mean_train_rew, n_epoch)
                writer.add_scalar('Train/comp_games', num_train_games_completed, n_epoch)

                writer.add_scalar('Test/min_rew', min_test_rew, n_epoch)
                writer.add_scalar('Test/max_rew', max_test_rew, n_epoch)
                writer.add_scalar('Test/mean_rew', mean_test_rew, n_epoch)
                writer.add_scalar('Test/comp_games', num_test_games_completed, n_epoch)

            # this is here to always keep the best model
            if (scores[-1][2]>best_mean_rew and args.save):
                agent.save(folder_name + '/agent_' + 'BEST' + '.msh')
                normalizer.save(folder_name + '/normalizer_' + 'BEST' + '.msh')
                best_mean_rew = scores[-1][2]

            if (args.save):
                np.save(folder_name + '/scores.npy', scores)

    return scores


if __name__ == '__main__':
    experiment()
