# MHA GNN implementation based on https://github.com/wouterkool/attention-learn-to-route/blob/master/nets/graph_encoder.py
# The implementation has been modified for our purposes and for instance now allows to pass an attention mask which has
# not been possible previously
# also, we added a specialized readout


import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):
    '''
    Skip Connection Module compatible to FF modules that return themselves the output and the masking,...
    '''

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input_0, input_1):
        res1, res2 = self.module(input_0,input_1)
        return (input_0 + res1), input_1

class SkipConnectionSimple(nn.Module):
    '''
    Skip Connection Module compatible to "simple" FF modules that return themselves only the output and also cannot
    take the masking as the input,...
    '''
    def __init__(self, module):
        super(SkipConnectionSimple, self).__init__()
        self.module = module

    def forward(self, input_0, input_1):
        res1 = self.module(input_0)
        return (input_0 + res1), input_1


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    # def forward(self, q, h=None, mask=None):
    def forward(self, q, mask):
        h = None
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[torch.logical_not(mask.bool())] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[torch.logical_not(mask.bool())] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)
        # print (out)
        # input ('out')

        return out, mask


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input

class mySequential(nn.Sequential):
    '''
    Custom definition of a sequential layer that allows to pass multiple arguments in the forward call
    '''
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input

# This is lightweight version of attention: -> after attention there is not this additional FF layer
class MultiHeadAttentionLayer(mySequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
        )

# This is standard version of attention: -> after attention there is additional FF layer
class MultiHeadAttentionLayerFull(mySequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayerFull, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            SkipConnectionSimple(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
        )


# this class uses the lightweight attention implementation
class MPNN_Multidim_Attention(nn.Module):
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
                 robot_state_dim = 0,
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

        n_heads = 4
        embed_dim = 64
        feed_forward_hidden = 512
        normalization = 'batch'
        self.layers = mySequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

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

        # - adjacency matrix tells whether there is an edge or not
        # this command computes the updated node embeddings,..
        current_node_embeddings = self.layers(init_node_embeddings,adj)

        # finally we compute the appropriate values from the "embedded" / feature graph representation
        out = self.readout_layer(*current_node_embeddings)

        if action is None:
            return out
        else:
            action = action.long()
            q_prefilter = out[torch.arange(out.size(0)),action[:,0],action[:,1],action[:,2]]

            return q_prefilter

# this class uses the full attention implementation
class MPNN_Multidim_Full_Attention(MPNN_Multidim_Attention):
    '''
    Differs compared to the "not full" implementation in the way that the Attention Mechanism is implemented,...
    -> in this version the number of parameters is more "blown up"
    '''
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

        super().__init__(input_shape, output_shape, n_obs_in, n_layers, n_features, tied_weights, n_hid_readout, dim_whole_obs, num_actions_avail, robot_state_dim, **kwargs)

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

        n_heads = 4
        embed_dim = 64
        feed_forward_hidden = 512
        normalization = 'batch'
        self.layers = mySequential(*(
            MultiHeadAttentionLayerFull(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

        self.readout_layer = ReadoutLayerMultidim(n_features, n_hid_readout, self.output_dim)

## this is the single head attention implementation -> there is only one single attention head, see variable n_heads
class MPNN_Single_Full_Attention(MPNN_Multidim_Attention):
    '''
    Differs compared to the "not full" implementation in the way that the Attention Mechanism is implemented,...
    -> in this version the number of parameters is more "blown up"
    '''
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

        super().__init__(input_shape, output_shape, n_obs_in, n_layers, n_features, tied_weights, n_hid_readout, dim_whole_obs, num_actions_avail, robot_state_dim, **kwargs)

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

        n_heads = 1
        embed_dim = 64
        feed_forward_hidden = 512
        normalization = 'batch'
        self.layers = mySequential(*(
            MultiHeadAttentionLayerFull(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

        self.readout_layer = ReadoutLayerMultidim(n_features, n_hid_readout, self.output_dim)

# this is a special implementation in which we create multiple full attention models, i.e. the result is an ensebmble
# of those models. The idea was that the uncertainty estimate might be useful in the MCTS pipeline of our method.
# However, results with this method have also not been included in the publication.
class MPNN_Multidim_Full_Attention_Multiple(MPNN_Multidim_Full_Attention):
    '''
    Differs compared to the "not full" implementation in the way that the Attention Mechanism is implemented,...
    -> in this version the number of parameters is more "blown up"
    '''
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

        self.num_models = 5
        self.weighting_factor = 1.0
        self.list_model = []
        super().__init__(input_shape, output_shape, n_obs_in, n_layers, n_features, tied_weights, n_hid_readout, dim_whole_obs, num_actions_avail, robot_state_dim, **kwargs)

        for i in range(self.num_models):
            exec("self.abc{} = MPNN_Multidim_Full_Attention(input_shape, output_shape, n_obs_in, n_layers, n_features, tied_weights, n_hid_readout, dim_whole_obs, num_actions_avail, robot_state_dim, **kwargs)".format(i))

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

        n_heads = 4
        embed_dim = 64
        feed_forward_hidden = 512
        normalization = 'batch'
        self.layers = mySequential(*(
            MultiHeadAttentionLayerFull(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

        self.readout_layer = ReadoutLayerMultidim(n_features, n_hid_readout, self.output_dim)

    def forward(self, obs, action=None,mean_std=False):
        if not(mean_std):
            if (next(self.readout_layer.parameters()).is_cuda) and not(next(self.abc0.readout_layer.parameters()).is_cuda):
                device = torch.device("cuda")
                for i in range(self.num_models):
                    a = getattr(self,"abc{}".format(i))
                    a.to(device)

            init_out = 1.0/self.num_models * self.abc0(obs,action=action)
            out = init_out
            for i in range(self.num_models-1):
                a = getattr(self, "abc{}".format(i+1))
                out = out + 1.0 / self.num_models * a(obs, action=action)

            return out
        else:
            if (next(self.readout_layer.parameters()).is_cuda) and not (
            next(self.abc0.readout_layer.parameters()).is_cuda):
                device = torch.device("cuda")
                for i in range(self.num_models):
                    a = getattr(self,"abc{}".format(i))
                    a.to(device)

            all_outputs = []
            a = getattr(self, "abc{}".format(0))
            init_out = 1.0 / self.num_models * a(obs, action=action)
            out = init_out
            all_outputs.append(init_out.detach().cpu().numpy())
            for i in range(self.num_models - 1):
                a = getattr(self, "abc{}".format(i+1))
                res = 1.0 / self.num_models * a(obs, action=action)
                all_outputs.append(res.detach().cpu().numpy())
                out = out + res

            std_dev = np.std(np.asarray(all_outputs), axis=0)
            out = out.detach().cpu().numpy()

            return torch.from_numpy(out + self.weighting_factor * std_dev)


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

    def forward(self, node_embeddings, mask):
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

# Note: this pytorch transformer implementation has not been used in this work so there are no guarantees on performance
class Pytorch_Transformer(nn.Module):
    def __init__(self,
                 n_obs_in=7,
                 n_layers=3,
                 n_features=64,
                 tied_weights=False,
                 n_hid_readout=[],
                 dim_whole_obs = 10,
                 **kwargs,):

        super().__init__()
        print ("not used in this work - so no guarantee on performance")

        n_obs_in = 4
        n_layers = 3
        n_features = 64
        tied_weights = False
        n_hid_readout = []
        output_dim = 3

        self.dim_whole_obs = dim_whole_obs
        self.n_obs_in = n_obs_in
        self.n_layers = n_layers
        self.n_features = n_features
        self.tied_weights = tied_weights
        self.output_dim = output_dim

        self.node_init_embedding_layer = nn.Sequential(
            nn.Linear(n_obs_in, n_features, bias=False),
            nn.ReLU()
        )
        self.node_init_embedding_layer.apply(self.weights_init)

        n_heads = 4
        n_layers = 3
        feed_forward_hidden = 512
        dropout = 0.0

        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.normalization = nn.LayerNorm(n_features)
        encoder_layers = TransformerEncoderLayer(n_features, n_heads, feed_forward_hidden, dropout)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers, norm=self.normalization)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

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
        #print (action)
        #print (obs.shape)
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
        # print (init_node_embeddings)
        current_node_embeddings = self.transformer_encoder(init_node_embeddings)
        # print (current_node_embeddings)
        # input ("WAIT")

        # input ("FINAL WAITING")

        # TODO: retrieving the right, allowed action is not implemented correctly for now,...
        out = self.readout_layer(current_node_embeddings)

        if action is None:
            return out
        else:
            action = action.long()
            q_prefilter = out[torch.arange(out.size(0)),action[:,0],action[:,1],action[:,2]]

            return q_prefilter