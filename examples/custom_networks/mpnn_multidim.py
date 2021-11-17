# structure 2 vec implementation based on https://github.com/tomdbar/eco-dqn/blob/134df732cbdc32ad840ee2c05079fb2dbb6dd6d0/src/networks/mpnn.py#L5
# but heavily adapted for our own purposes and special usecase

import torch
import torch.nn as nn
import torch.nn.functional as F

# This is the standard implementation of S2V
class MPNN_Multidim(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 n_obs_in=7,
                 n_layers=3,
                 n_features=64,
                 tied_weights=False,
                 n_hid_readout=[],
                 dim_whole_obs = 10,
                 num_actions_avail=None,
                 robot_state_dim=0,
                 **kwargs,):

        super().__init__()

        self.dim_whole_obs = dim_whole_obs
        self.n_obs_in = n_obs_in
        self.n_layers = n_layers
        self.n_features = n_features
        self.tied_weights = tied_weights
        self.output_dim = num_actions_avail
        self.robot_state_dim = robot_state_dim

        self.node_init_embedding_layer = nn.Sequential(
            nn.Linear(n_obs_in, n_features, bias=False),
            nn.ReLU()
        )
        self.node_init_embedding_layer.apply(self.weights_init)


        self.edge_embedding_layer = EdgeAndNodeEmbeddingLayer(n_obs_in, n_features)

        if self.tied_weights:
            self.update_node_embedding_layer = UpdateNodeEmbeddingLayer(n_features)
        else:
            self.update_node_embedding_layer = nn.ModuleList(
                [UpdateNodeEmbeddingLayer(n_features) for _ in range(self.n_layers)])

        self.readout_layer = ReadoutLayerMultidim(n_features, n_hid_readout, self.output_dim)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight,
                                    gain=nn.init.calculate_gain('relu'))

    @torch.no_grad()
    def get_normalisation(self, adj):
        # add up the number of connections
        norm = torch.sum((adj != 0), dim=1).unsqueeze(-1)
        # even if no connections set to 1 as otherwise numerical errors
        norm[norm == 0] = 1
        return norm.float()

    def forward(self, obs, action=None):
        if (self.robot_state_dim>0):
            obs = obs[:,:-(self.robot_state_dim)]
        batch_size = obs.size(0)
        # reshape to account for individual nodes
        obs = obs.contiguous().view(batch_size, -1, self.dim_whole_obs)

        # Extract the features from the observation which are used by the network
        node_features = obs[:, :, 0:self.n_obs_in]

        # Get graph adj matrix. - they are also part of the observation and just appended behind the "normal" features
        adj = obs[:, :, self.n_obs_in:]
        # get the normalization for the adjacency matrix -> how many connections does each node have?
        norm = self.get_normalisation(adj)

        # simply calculate feature representation of the nodes
        init_node_embeddings = self.node_init_embedding_layer(node_features)
        # calculate the edge embeddings and add them per node already -> returns the sum of added edge weights per node
        # - adjacency matrix tells whether there is an edge or not
        # -> precisely: returns the per node edge embedding,...
        edge_embeddings = self.edge_embedding_layer(node_features, adj, norm)

        # Initialise the node embeddings.
        current_node_embeddings = init_node_embeddings

        # perform message passing. Note: in this architecture the edge weights are not recalculated but kept constant
        if self.tied_weights:
            for _ in range(self.n_layers):
                current_node_embeddings = self.update_node_embedding_layer(current_node_embeddings,
                                                                           edge_embeddings,
                                                                           norm,
                                                                           adj)
        else:
            for i in range(self.n_layers):
                current_node_embeddings = self.update_node_embedding_layer[i](current_node_embeddings,
                                                                              edge_embeddings,
                                                                              norm,
                                                                              adj)

        out = self.readout_layer(current_node_embeddings)

        if action is None:
            return out
        else:
            action = action.long()
            q_prefilter = out[torch.arange(out.size(0)),action[:,0],action[:,1],action[:,2]]

            return q_prefilter

# This implements a slightly modified version in which the edge embeddings are updated more iteratively compared to the
# above implementation of MPNN_Multidim
class MPNN_Multidim_MORE(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 n_obs_in=7,
                 n_layers=3,
                 n_features=64,
                 tied_weights=False,
                 n_hid_readout=[],
                 dim_whole_obs = 10,
                 num_actions_avail=None,
                 robot_state_dim=0,
                 **kwargs,):

        super().__init__()

        self.dim_whole_obs = dim_whole_obs
        self.n_obs_in = n_obs_in
        self.n_layers = n_layers
        self.n_features = n_features
        self.tied_weights = tied_weights
        self.output_dim = num_actions_avail
        self.robot_state_dim = robot_state_dim

        self.node_init_embedding_layer = nn.Sequential(
            nn.Linear(n_obs_in, n_features, bias=False),
            nn.ReLU()
        )
        self.node_init_embedding_layer.apply(self.weights_init)


        if self.tied_weights:
            self.edge_embedding_layer = EdgeAndNodeEmbeddingLayer(n_features,n_features)
        else:
            self.edge_embedding_layer = nn.ModuleList(
                [EdgeAndNodeEmbeddingLayer(n_features,n_features) for _ in range(self.n_layers)])

        if self.tied_weights:
            self.update_node_embedding_layer = UpdateNodeEmbeddingLayer(n_features)
        else:
            self.update_node_embedding_layer = nn.ModuleList(
                [UpdateNodeEmbeddingLayer(n_features) for _ in range(self.n_layers)])

        self.readout_layer = ReadoutLayerMultidim(n_features, n_hid_readout, self.output_dim)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight,
                                    gain=nn.init.calculate_gain('relu'))

    @torch.no_grad()
    def get_normalisation(self, adj):
        # add up the number of connections
        norm = torch.sum((adj != 0), dim=1).unsqueeze(-1)
        # even if no connections set to 1 as otherwise numerical errors
        norm[norm == 0] = 1
        return norm.float()

    def forward(self, obs, action=None):
        if (self.robot_state_dim>0):
            obs = obs[:,:-(self.robot_state_dim)]
        batch_size = obs.size(0)
        # reshape to account for individual nodes
        obs = obs.contiguous().view(batch_size, -1, self.dim_whole_obs)

        # Extract the features from the observation which are used by the network
        node_features = obs[:, :, 0:self.n_obs_in]

        # Get graph adj matrix. - they are also part of the observation and just appended behind the "normal" features
        adj = obs[:, :, self.n_obs_in:]
        # get the normalization for the adjacency matrix -> how many connections does each node have?
        norm = self.get_normalisation(adj)

        # simply calculate feature representation of the nodes
        init_node_embeddings = self.node_init_embedding_layer(node_features)
        # calculate the edge embeddings and add them per node already -> returns the sum of added edge weights per node
        # - adjacency matrix tells whether there is an edge or not
        # -> precisely: returns the per node edge embedding,...


        # Initialise the node embeddings.
        current_node_embeddings = init_node_embeddings

        # perform message passing. Note: in this optimized version, the edge weights are actually being updated based on
        # the current node embeddings -> this is the minor difference.
        # However, for our experiments this did not result in big improvements and we thus stayed with the original
        # implementation.
        if self.tied_weights:
            for _ in range(self.n_layers):
                prev_embedding = current_node_embeddings
                # unlike in original version: here now update the edge embeddings based on the current node embedding
                edge_embeddings = self.edge_embedding_layer(current_node_embeddings, adj, norm)
                current_node_embeddings = self.update_node_embedding_layer(current_node_embeddings,
                                                                           edge_embeddings,
                                                                           norm,
                                                                           adj)
                current_node_embeddings = current_node_embeddings + prev_embedding

        else:
            for i in range(self.n_layers):
                prev_embedding = current_node_embeddings
                # unlike in original version: here now update the edge embeddings based on the current node embedding
                edge_embeddings = self.edge_embedding_layer[i](current_node_embeddings, adj, norm)
                current_node_embeddings = self.update_node_embedding_layer[i](current_node_embeddings,
                                                                              edge_embeddings,
                                                                              norm,
                                                                              adj)
                current_node_embeddings = current_node_embeddings + prev_embedding

        out = self.readout_layer(current_node_embeddings)

        if action is None:
            return out
        else:
            action = action.long()
            q_prefilter = out[torch.arange(out.size(0)),action[:,0],action[:,1],action[:,2]]

            return q_prefilter



class EdgeAndNodeEmbeddingLayer(nn.Module):

    def __init__(self, n_obs_in, n_features):
        super().__init__()
        self.n_obs_in = n_obs_in
        self.n_features = n_features
        # add oned dimenstion as there is an indicator whether edge is there or not, remove one feature as the normalization is appended
        self.edge_embedding_NN = nn.Linear(int(n_obs_in + 1), n_features - 1, bias=False)
        self.edge_feature_NN = nn.Linear(n_features, n_features, bias=False)

    def forward(self, node_features, adj, norm):
        edge_features = torch.cat([adj.unsqueeze(-1),
                                   node_features.unsqueeze(-2).transpose(-2, -3).repeat(1, adj.shape[-2], 1, 1)],
                                  dim=-1)

        # multiplies in the connectivity -> all entries where there are no connections are set to 0,...
        edge_features *= (adj.unsqueeze(-1) != 0).float()

        # unroll all the edges -> flatten the middle dimension to pass through NN
        edge_features_unrolled = torch.reshape(edge_features, (
        edge_features.shape[0], edge_features.shape[1] * edge_features.shape[1], edge_features.shape[-1]))

        # embed the unrolled edges
        embedded_edges_unrolled = F.relu(self.edge_embedding_NN(edge_features_unrolled))
        # bring back into rolled shale
        embedded_edges_rolled = torch.reshape(embedded_edges_unrolled,
                                              (adj.shape[0], adj.shape[1], adj.shape[1], self.n_features - 1))

        # sum the edge weights per node (add values of all edges that go into node) and normalize by connectivity (=norm)
        embedded_edges = embedded_edges_rolled.sum(dim=2) / norm

        # apply nonlinear transform to the embedded edges (per node) and also add the norm information (tells about connectivity,..)
        edge_embeddings = F.relu(self.edge_feature_NN(torch.cat([embedded_edges, norm / norm.max()], dim=-1)))

        return edge_embeddings


class UpdateNodeEmbeddingLayer(nn.Module):

    def __init__(self, n_features):
        super().__init__()

        self.message_layer = nn.Linear(2 * n_features, n_features, bias=False)
        self.update_layer = nn.Linear(2 * n_features, n_features, bias=False)

    def forward(self, current_node_embeddings, edge_embeddings, norm, adj):
        # for each note calculate new embedding based on the connected nodes and normalize it
        node_embeddings_aggregated = torch.matmul(adj, current_node_embeddings) / norm

        # message is a nonlinear transform of aggregated node and edge embeddings (at each node)
        message = F.relu(self.message_layer(torch.cat([node_embeddings_aggregated, edge_embeddings], dim=-1)))
        # update the current node embeddings with the message
        new_node_embeddings = F.relu(self.update_layer(torch.cat([current_node_embeddings, message], dim=-1)))

        return new_node_embeddings


class ReadoutLayer(nn.Module):
    '''
    Original implementation of the readout layer which did not suit our purpose as it did not compute values per edge,..
    '''

    def __init__(self, n_features, n_hid=[], output_dim=1, bias_pool=False, bias_readout=True):

        super().__init__()

        self.layer_pooled = nn.Linear(int(n_features), int(n_features), bias=bias_pool)

        if type(n_hid) != list:
            n_hid = [n_hid]

        n_hid = [2 * n_features] + n_hid + [output_dim]

        self.layers_readout = []
        for n_in, n_out in list(zip(n_hid, n_hid[1:])):
            layer = nn.Linear(n_in, n_out, bias=bias_readout)
            self.layers_readout.append(layer)

        self.layers_readout = nn.ModuleList(self.layers_readout)

    def forward(self, node_embeddings):

        f_local = node_embeddings
        # h_pooled -> sum over all node embeddings (normalized) -> can be seen as computing global feature
        h_pooled = self.layer_pooled(node_embeddings.sum(dim=1) / node_embeddings.shape[1])
        # scale up again to the number of nodes
        f_pooled = h_pooled.repeat(1, node_embeddings.shape[1]).view(node_embeddings.shape)

        # combine global and local features
        features = F.relu(torch.cat([f_pooled, f_local], dim=-1))

        # propagate through layers
        for i, layer in enumerate(self.layers_readout):
            features = layer(features)
            if i < len(self.layers_readout) - 1:
                features = F.relu(features)
            else:
                out = features

        return out


class ReadoutLayerMultidim(nn.Module):
    '''
    Scaled up version where one value for each edge is returned,...
    '''

    def __init__(self, n_features, n_hid=[], output_dim=1, bias_pool=False, bias_readout=True):

        super().__init__()

        self.layer_pooled = nn.Linear(int(n_features), int(n_features), bias=bias_pool)

        if type(n_hid) != list:
            n_hid = [n_hid]

        n_hid = [3 * n_features] + n_hid + [output_dim]

        self.layers_readout = []
        for n_in, n_out in list(zip(n_hid, n_hid[1:])):
            layer = nn.Linear(n_in, n_out, bias=bias_readout)
            self.layers_readout.append(layer)

        self.layers_readout = nn.ModuleList(self.layers_readout)

    def forward(self, node_embeddings):

        f_local = node_embeddings

        # Scale up such that one value per node-node pair,... the first one is const along 2nd dim, second one along 3rd,..
        f_expanded = f_local.repeat(1,f_local.shape[1],1).view(f_local.shape[0],f_local.shape[1],f_local.shape[1],-1).transpose(1,2).reshape(f_local.shape[0],f_local.shape[1]**2,-1)
        f_expanded_diff = f_local.repeat(1,f_local.shape[1],1).view(f_local.shape[0],f_local.shape[1],f_local.shape[1],-1).reshape(f_local.shape[0],f_local.shape[1]**2,-1)

        # h_pooled -> sum over all node embeddings (normalized) -> can be seen as computing global feature
        h_pooled = self.layer_pooled(node_embeddings.sum(dim=1) / node_embeddings.shape[1])
        # scale up again to the number of nodes
        f_pooled = h_pooled.repeat(1, f_expanded.shape[1]).view(f_expanded.shape)

        # combine global and local features
        features = F.relu(torch.cat([f_pooled, f_expanded, f_expanded_diff], dim=-1))

        # propagate through layers
        for i, layer in enumerate(self.layers_readout):
            features = layer(features)
            if i < len(self.layers_readout) - 1:
                features = F.relu(features)
            else:
                out = features

        out = out.view(f_local.shape[0],f_local.shape[1],f_local.shape[1],-1)

        return out