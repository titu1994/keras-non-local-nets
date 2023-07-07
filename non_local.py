from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Reshape, dot, Activation, Lambda, MaxPool1D, add
from tensorflow.keras import backend as K

class NonLocalBlock:
    def __init__(self, intermediate_dim=None, compression=2, mode='embedded', add_residual=True):
        """
        Initializes a NonLocalBlock instance.

        Parameters
        ----------
        intermediate_dim: None / int
            The dimension of the intermediate representation. Can be `None` or a positive integer greater than 0. If `None`, computes the intermediate dimension as half of the input channel dimension.
        compression: None or positive integer. 
            Compresses the intermediate representation during the dot products to reduce memory consumption. Default is set to 2, which states halve the time/space/spatio-time dimension for the intermediate step. Set to 1 to prevent computation compression. None or 1 causes no reduction.
        mode: str
            Mode of operation. Can be one of `embedded`, `gaussian`, `dot` or `concatenate`.
        add_residual: bool
            Decides if the residual connection should be added or not. Default is True for ResNets, and False for Self Attention.
        """
        self.intermediate_dim = intermediate_dim
        self.compression = compression
        self.mode = mode
        self.add_residual = add_residual

    def __call__(self, ip):
        """
        Applies the Non-Local block to the input tensor.

        Returns:
            Tensor: Output tensor of the Non-Local block with the same shape as the input.

        Parameters
        ----------
        ip: array
            Input tensor.
        """
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(ip)

        if self.mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        if self.compression is None:
            self.compression = 1

        # check rank and calculate the input shape
        rank = len(input_shape)
        if rank not in [3, 4, 5]:
            raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

        elif rank == 3:
            batchsize, dims, channels = input_shape

        else:
            if channel_dim == 1:
                batchsize, channels, *dims = input_shape
            else:
                batchsize, *dims, channels = input_shape

        # verify correct intermediate dimension specified
        if self.intermediate_dim is None:
            self.intermediate_dim = channels // 2

            if self.intermediate_dim < 1:
                self.intermediate_dim = 1

        else:
            self.intermediate_dim = int(self.intermediate_dim)

            if self.intermediate_dim < 1:
                raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

        if self.mode == 'gaussian':  # Gaussian instantiation
            x1 = Reshape((-1, channels))(ip)  # xi
            x2 = Reshape((-1, channels))(ip)  # xj
            f = dot([x1, x2], axes=2)
            f = Activation('softmax')(f)

        elif self.mode == 'dot':  # Dot instantiation
            # theta path
            theta = self._convND(ip, rank, self.intermediate_dim)
            theta = Reshape((-1, self.intermediate_dim))(theta)

            # phi path
            phi = self._convND(ip, rank, self.intermediate_dim)
            phi = Reshape((-1, self.intermediate_dim))(phi)

            f = dot([theta, phi], axes=2)

            size = K.int_shape(f)

            # scale the values to make it size invariant
            f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

        elif self.mode == 'concatenate':  # Concatenation instantiation
            raise NotImplementedError('Concatenate model has not been implemented yet')

        else:  # Embedded Gaussian instantiation
            # theta path
            theta = self._convND(ip, rank, self.intermediate_dim)
            theta = Reshape((-1, self.intermediate_dim))(theta)

            # phi path
            phi = self._convND(ip, rank, self.intermediate_dim)
            phi = Reshape((-1, self.intermediate_dim))(phi)

            if self.compression > 1:
                # shielded computation
                phi = MaxPool1D(self.compression)(phi)

            f = dot([theta, phi], axes=2)
            f = Activation('softmax')(f)

        # g path
        g = self._convND(ip, rank, self.intermediate_dim)
        g = Reshape((-1, self.intermediate_dim))(g)

        if self.compression > 1 and self.mode == 'embedded':
            # shielded computation
            g = MaxPool1D(self.compression)(g)

        # compute output path
        y = dot([f, g], axes=[2, 1])

        # reshape to input tensor format
        if rank == 3:
            y = Reshape((dims, self.intermediate_dim))(y)
        else:
            if channel_dim == -1:
                y = Reshape((*dims, self.intermediate_dim))(y)
            else:
                y = Reshape((self.intermediate_dim, *dims))(y)

        # project filters
        y = self._convND(y, rank, channels)

        # residual connection
        if self.add_residual:
            y = add([ip, y])

        return y
    

    def _convND(self, ip, rank, channels):
        """
        Applies a convolution operation based on the rank of the input tensor.

        Returns:
            Tensor: Output of the convolution operation.

        Parameters
        ----------
        ip: array
            Input tensor.
        rank: int
            Rank of the input tensor. Must be 3, 4, or 5.
        channels: int 
            Number of output channels for the convolution.
        """
            
        assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

        if rank == 3:
            x = Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
        elif rank == 4:
            x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
        else:
            x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
        return x
