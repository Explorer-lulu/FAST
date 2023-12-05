from mxnet import nd
from mxnet.gluon import nn

import numpy as np

import mxnet.gluon.rnn as mrnn

class Spatial_Attention_layer(nn.Block):
    def __init__(self, **kwargs):
        super(Spatial_Attention_layer, self).__init__(**kwargs)
        with self.name_scope():
            self.W_1 = self.params.get('W_1', allow_deferred_init=True)
            self.W_2 = self.params.get('W_2', allow_deferred_init=True)
            self.W_3 = self.params.get('W_3', allow_deferred_init=True)
            self.b_s = self.params.get('b_s', allow_deferred_init=True)
            self.V_s = self.params.get('V_s', allow_deferred_init=True)

    def forward(self, x):
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        self.W_1.shape = (num_of_timesteps, )
        self.W_2.shape = (num_of_features, num_of_timesteps)
        self.W_3.shape = (num_of_features, )
        self.b_s.shape = (1, num_of_vertices, num_of_vertices)
        self.V_s.shape = (num_of_vertices, num_of_vertices)
        for param in [self.W_1, self.W_2, self.W_3, self.b_s, self.V_s]:
            param._finish_deferred_init()
        lhs = nd.dot(nd.dot(x, self.W_1.data()), self.W_2.data())

        # shape of rhs is (batch_size, T, V)
        rhs = nd.dot(self.W_3.data(), x.transpose((2, 0, 3, 1)))

        # shape of product is (batch_size, V, V)
        product = nd.batch_dot(lhs, rhs)

        # try tanh/relu as a new active function
                
        S = nd.dot(self.V_s.data(),
                   nd.relu(product + self.b_s.data())
                     .transpose((1, 2, 0))).transpose((2, 0, 1))

        # normalization
        S = S - nd.max(S, axis=1, keepdims=True)
        exp = nd.exp(S)
        S_normalized = exp / nd.sum(exp, axis=1, keepdims=True)
        return S_normalized


class cheb_conv_with_SAt(nn.Block):
    '''
    K-order chebyshev graph convolution with Spatial Attention scores to finish the spatial temporal attention
    '''
    def __init__(self, num_of_filters, K, cheb_polynomials, **kwargs):
        super(cheb_conv_with_SAt, self).__init__(**kwargs)
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        with self.name_scope():
            self.Theta = self.params.get('Theta', allow_deferred_init=True)

    def forward(self, x, spatial_attention):
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape

        self.Theta.shape = (self.K, num_of_features, self.num_of_filters)
        self.Theta._finish_deferred_init()

        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = nd.zeros(shape=(batch_size, num_of_vertices,
                                     self.num_of_filters), ctx=x.context)
            for k in range(self.K):

                T_k = self.cheb_polynomials[k]
                T_k_with_at = T_k * spatial_attention
                theta_k = self.Theta.data()[k]
                rhs = nd.batch_dot(T_k_with_at.transpose((0, 2, 1)),
                                   graph_signal)

                output = output + nd.dot(rhs, theta_k)
            outputs.append(output.expand_dims(-1))
        return nd.relu(nd.concat(*outputs, dim=-1))


class Temporal_Attention_layer(nn.Block):
    '''
    compute temporal attention scores
    '''
    def __init__(self, **kwargs):
        super(Temporal_Attention_layer, self).__init__(**kwargs)
        with self.name_scope():
            self.U_1 = self.params.get('U_1', allow_deferred_init=True)
            self.U_2 = self.params.get('U_2', allow_deferred_init=True)
            self.U_3 = self.params.get('U_3', allow_deferred_init=True)
            self.b_e = self.params.get('b_e', allow_deferred_init=True)
            self.V_e = self.params.get('V_e', allow_deferred_init=True)

    def forward(self, x):
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # defer shape
        self.U_1.shape = (num_of_vertices, )
        self.U_2.shape = (num_of_features, num_of_vertices)
        self.U_3.shape = (num_of_features, )
        self.b_e.shape = (1, num_of_timesteps, num_of_timesteps)
        self.V_e.shape = (num_of_timesteps, num_of_timesteps)
        for param in [self.U_1, self.U_2, self.U_3, self.b_e, self.V_e]:
            param._finish_deferred_init()

        lhs = nd.dot(nd.dot(x.transpose((0, 3, 2, 1)), self.U_1.data()),
                     self.U_2.data())

        rhs = nd.dot(self.U_3.data(), x.transpose((2, 0, 1, 3)))
        product = nd.batch_dot(lhs, rhs)

        E = nd.dot(self.V_e.data(),
                   nd.relu(product + self.b_e.data())
                     .transpose((1, 2, 0))).transpose((2, 0, 1))


        # normailzation
        E = E - nd.max(E, axis=1, keepdims=True)
        exp = nd.exp(E)
        E_normalized = exp / nd.sum(exp, axis=1, keepdims=True)
        return E_normalized


class DSTEGCN_block(nn.Block):
    def __init__(self, backbone, **kwargs):
        '''
        Contain a temporal enhanced block to capture the temporal information

        '''
        super(DSTEGCN_block, self).__init__(**kwargs)

        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
        time_conv_strides = backbone['time_conv_strides']
        cheb_polynomials = backbone["cheb_polynomials"]

        with self.name_scope():
            self.SAt = Spatial_Attention_layer()
            self.cheb_conv_SAt = cheb_conv_with_SAt(
                num_of_filters=num_of_chev_filters,
                K=K,
                cheb_polynomials=cheb_polynomials)
            self.TAt = Temporal_Attention_layer()
            self.time_conv = nn.Conv2D(
                channels=num_of_time_filters,
                kernel_size=(1, 3),
                padding=(0, 1),
                strides=(1, time_conv_strides))
            self.residual_conv = nn.Conv2D(
                channels=num_of_time_filters,
                kernel_size=(1, 1),
                strides=(1, time_conv_strides))

            self.ln = nn.LayerNorm(axis=2)

            self.GRU = mrnn.GRU(hidden_size=64)
            self.MLP_in = nn.Dense(64, flatten=False, activation='relu')
            self.MLP_out = nn.Dense(12, flatten=False)

    def forward(self, x):
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape
        
        gru_data = x[:, :, :, -1]
        gru_result = self.GRU(gru_data.transpose((1, 0, 2)))
        gru_result = gru_result.transpose((1, 0, 2))
       

        temporal_At = self.TAt(x)
        
        x_TAt = nd.batch_dot(x.reshape(batch_size, -1, num_of_timesteps),
                             temporal_At)\
                  .reshape(batch_size, num_of_vertices,
                           num_of_features, num_of_timesteps)


        spatial_At = self.SAt(x_TAt)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)


        time_conv_output = (self.time_conv(spatial_gcn.transpose((0, 2, 1, 3)))
                            .transpose((0, 2, 1, 3)))

        x_residual = (self.residual_conv(x.transpose((0, 2, 1, 3)))
                      .transpose((0, 2, 1, 3)))


        gru_result = gru_result.expand_dims(-1)

        V = nd.concat(x_residual, gru_result, dim= -1)

        V = self.MLP_in(V)
        V = self.MLP_out(V)


        return self.ln(nd.relu(V + time_conv_output))



class DSTEGCN_submodule(nn.Block):
    '''
    a module in DSTEGCN
    '''
    def __init__(self, num_for_prediction, backbones, **kwargs):

        super(DSTEGCN_submodule, self).__init__(**kwargs)

        self.blocks = nn.Sequential()
        for backbone in backbones:
            self.blocks.add(DSTEGCN_block(backbone))

        with self.name_scope():
            self.final_conv = nn.Conv2D(
                channels=num_for_prediction,
                kernel_size=(1, backbones[-1]['num_of_time_filters']))
            self.W = self.params.get("W", allow_deferred_init=True)

    def forward(self, x):

        x_blocks = self.blocks(x)
        module_output = (self.final_conv(x_blocks.transpose((0, 3, 1, 2)))
                         [:, :, :, -1].transpose((0, 2, 1)))
        _, num_of_vertices, num_for_prediction = module_output.shape
        self.W.shape = (num_of_vertices, num_for_prediction)
        self.W._finish_deferred_init()

        return ( module_output * self.W.data()) 

class DSTEGCN(nn.Block):
    '''
    DSTEGCN, 3 sub-modules, for hour, day, week respectively
    '''
    def __init__(self, num_for_prediction, all_backbones, **kwargs):
        '''
        all_backbones: list[list],
                       3 backbones for "hour", "day", "week" 
        '''
        super(DSTEGCN, self).__init__(**kwargs)
        if len(all_backbones) <= 0:
            raise ValueError("The length of all_backbones "
                             "must be greater than 0")

        self.submodules = []
        with self.name_scope():
            for backbones in all_backbones:
                self.submodules.append(
                    DSTEGCN_submodule(num_for_prediction, backbones))
                self.register_child(self.submodules[-1])

    def forward(self, x_list):
        if len(x_list) != len(self.submodules):
            raise ValueError("num of submodule not equals to "
                             "length of the input list")

        num_of_vertices_set = {i.shape[1] for i in x_list}
        if len(num_of_vertices_set) != 1:
            raise ValueError("Different num_of_vertices detected! "
                             "Check if your input data have same size"
                             "at axis 1.")

        batch_size_set = {i.shape[0] for i in x_list}
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have same batch size!")

        submodule_outputs = [self.submodules[idx](x_list[idx])
                             for idx in range(len(x_list))]

        return nd.add_n(*submodule_outputs)

