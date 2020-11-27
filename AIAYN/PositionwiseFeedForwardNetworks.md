In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.   
$$
FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

> **[info]** [ReLU](https://windmissing.github.io/Bible-DeepLearning/Chapter6/3Hidden/1ReLU.html)  

While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality df f = 2048. 