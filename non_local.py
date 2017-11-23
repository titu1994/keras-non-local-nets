from keras.layers import Activation, Reshape, Lambda, concatenate, dot, add
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D
from keras import backend as K


def non_local_block(ip, shield_computation=True, mode='embedded'):
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    dim1, dim2, dim3 = None, None, None

    if len(ip_shape) == 3:  # time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # Video / Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        if rank == 3:
            theta = Conv1D(channels // 2, 1, padding='same', use_bias=False)(ip)
        elif rank == 4:
            theta = Conv2D(channels // 2, (1, 1), padding='same', use_bias=False)(ip)
        else:
            theta = Conv3D(channels // 2, (1, 1, 1), padding='same', use_bias=False)(ip)

        theta = Reshape((-1, channels // 2))(theta)

        # phi path
        if rank == 3:
            phi = Conv1D(channels // 2, 1, padding='same', use_bias=False)(ip)
        elif rank == 4:
            phi = Conv2D(channels // 2, (1, 1), padding='same', use_bias=False)(ip)
        else:
            phi = Conv3D(channels // 2, (1, 1, 1), padding='same', use_bias=False)(ip)

        phi = Reshape((-1, channels // 2))(phi)

        f = dot([theta, phi], axes=2)

        # scale the values to make it size invariant
        f = Lambda(lambda z: 1./batchsize * f)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplemented('Concatenation mode has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        if rank == 3:
            theta = Conv1D(channels // 2, 1, padding='same', use_bias=False)(ip)
        elif rank == 4:
            theta = Conv2D(channels // 2, (1, 1), padding='same', use_bias=False)(ip)
        else:
            theta = Conv3D(channels // 2, (1, 1, 1), padding='same', use_bias=False)(ip)

        theta = Reshape((-1, channels // 2))(theta)

        # phi path
        if rank == 3:
            phi = Conv1D(channels // 2, 1, padding='same', use_bias=False)(ip)
        elif rank == 4:
            phi = Conv2D(channels // 2, (1, 1), padding='same', use_bias=False)(ip)
        else:
            phi = Conv3D(channels // 2, (1, 1, 1), padding='same', use_bias=False)(ip)

        phi = Reshape((-1, channels // 2))(phi)

        if shield_computation:
            # shielded computation
            phi = MaxPool1D()(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    # g path
    if rank == 3:
        g = Conv1D(channels // 2, 1, padding='same', use_bias=False)(ip)
    elif rank == 4:
        g = Conv2D(channels // 2, (1, 1), padding='same', use_bias=False)(ip)
    else:
        g = Conv3D(channels // 2, (1, 1, 1), padding='same', use_bias=False)(ip)

    g = Reshape((-1, channels // 2))(g)

    if shield_computation and mode == 'embedded':
        # shielded computation
        g = MaxPool1D()(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, channels // 2))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, channels // 2))(y)
        else:
            y = Reshape((channels // 2, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, channels // 2))(y)
        else:
            y = Reshape((channels // 2, dim1, dim2, dim3))(y)

    # project filters
    y = Conv2D(channels, (1, 1), padding='same', use_bias=False)(y)

    # residual connection
    residual = add([ip, y])

    return residual
