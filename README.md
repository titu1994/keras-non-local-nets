# Keras Non-Local Neural Networks
Keras implementation of Non-local blocks from the paper ["Non-local Neural Networks"](https://arxiv.org/abs/1711.07971)

- Support for `"Gaussian"`, `"Embedded Gaussian"` and `"Dot"` instantiations of the Non-Local block. 
- Support for variable shielded computation mode (reduces computation by N**2 x, where N is default to 2)
- Support for `"Concatenation"` instantiation will be supported when authors release their code.

# Usage

The script `non_local.py` contains a single function : `non_local_block` which takes in an input tensor and wraps a non-local block around it.

```python
from non_local import NonLocalBlock
from tensorflow.keras.layers import Input, Conv1D, Conv2D, Conv3D

ip = Input(shape=(##))  # input tensor with an "N" rank order of 3, 4 or 5
x = ConvND(...)         # convolution operation with aforementioned rank 
...
non_local_block = NonLocalBlock(intermediate_dim=None, compression=2, mode='embedded', add_residual=True)
x = non_local_block(x)
...
```

# Basic block
From the paper, a basic Non-Local block looks like below (with the Embedded Gaussian instantiation)

<img src="https://github.com/titu1994/keras-non-local-nets/blob/master/images/non-local-block.PNG?raw=true" width=100% height=100%>
