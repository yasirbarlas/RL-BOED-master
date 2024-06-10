import copy

import torch
import torch.nn as nn

from garage.torch import NonLinearity


class MultiHeadedLSTMModule(nn.Module):
    """MultiHeadedLSTMModule Model.

    A PyTorch module composed of multiple LSTM layers with multiple parallel 
    output layers which maps real-valued inputs to real-valued outputs. 
    The length of outputs is n_heads and shape of each output element depends 
    on each output dimension.

    Args:
        n_heads (int): Number of different output layers.
        input_dim (int): Dimension of the network input.
        output_dims (int or list or tuple): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of LSTM layer(s).
            For example, (32, 32) means this LSTM consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module or list or tuple):
            Activation function for intermediate LSTM layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate LSTM layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate LSTM layer(s). The function should return a
            torch.Tensor.
        output_nonlinearities (callable or torch.nn.Module or list or tuple):
            Activation function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
            Size of the parameter should be 1 or equal to n_heads.
        output_w_inits (callable or list or tuple): Initializer function for
            the weight of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_heads.
        output_b_inits (callable or list or tuple): Initializer function for
            the bias of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_heads.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 n_heads,
                 input_dim,
                 output_dims,
                 hidden_sizes,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearities=None,
                 output_w_inits=nn.init.xavier_normal_,
                 output_b_inits=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__()

        self._layers = nn.ModuleList()

        output_dims = self._check_parameter_for_output_layer(
            'output_dims', output_dims, n_heads)
        output_w_inits = self._check_parameter_for_output_layer(
            'output_w_inits', output_w_inits, n_heads)
        output_b_inits = self._check_parameter_for_output_layer(
            'output_b_inits', output_b_inits, n_heads)
        output_nonlinearities = self._check_parameter_for_output_layer(
            'output_nonlinearities', output_nonlinearities, n_heads)

        self._layers = nn.ModuleList()

        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            if layer_normalization:
                hidden_layers.add_module('layer_normalization',
                                         nn.LayerNorm(prev_size))
            lstm_layer = nn.LSTM(prev_size, size, batch_first=True)
            
            for weight in lstm_layer._all_weights:
                if "weight" in weight:
                    hidden_w_init(getattr(lstm_layer, weight))
                if "bias" in weight:
                    hidden_b_init(getattr(lstm_layer, weight))

            hidden_layers.add_module('lstm', lstm_layer)

            self._layers.append(hidden_layers)
            prev_size = size

        self._output_layers = nn.ModuleList()
        for i in range(n_heads):
            output_layer = nn.Sequential()
            linear_layer = nn.Linear(prev_size, output_dims[i])
            output_w_inits[i](linear_layer.weight)
            output_b_inits[i](linear_layer.bias)
            output_layer.add_module('linear', linear_layer)

            if output_nonlinearities[i]:
                output_layer.add_module('non_linearity',
                                        NonLinearity(output_nonlinearities[i]))

            self._output_layers.append(output_layer)

    @classmethod
    def _check_parameter_for_output_layer(cls, var_name, var, n_heads):
        """Check input parameters for output layer are valid.

        Args:
            var_name (str): Variable name.
            var (any): Variable to be checked.
            n_heads (int): Number of heads.

        Returns:
            list: List of variables (length of n_heads).

        Raises:
            ValueError: If the variable is a list but the length of the variable
                is not equal to n_heads.

        """
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                return list(var) * n_heads
            if len(var) == n_heads:
                return var
            msg = ('{} should be either an integer or a collection of length '
                   'n_heads ({}), but {} provided.')
            raise ValueError(msg.format(var_name, n_heads, var))
        return [copy.deepcopy(var) for _ in range(n_heads)]

    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, T, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values.

        """
        x = input_val
        for layer in self._layers:
            x, _ = layer(x)

        return [output_layer(x) for output_layer in self._output_layers]