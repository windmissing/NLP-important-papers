# A Mean-field Iteration as a Stack of CNN Layers

A key contribution of this paper is to show that **the mean-field CRF inference can be reformulated as a Recurrent Neural Network (RNN)**.   

> **[success]** 这一节的目标：用RNN实现mean-field CRF inference  

To this end, we first consider individual steps of the mean-field algorithm summarized in Algorithm 1 [27], and describe them as CNN layers.   

> **[success]** 第一步：用CNN层描述mean-field CRF算法    

Our contribution is based on the observation that **filter-based approximate mean-field inference approach for dense CRFs relies on applying Gaussian spatial and bilateral filters on the mean-field approximates in each iteration**.   

> **[info]** bilateral：双边的  

Unlike the standard convolutional layer in a CNN, in which filters are fixed after the training stage, we use edge-preserving Gaussian filters [54, 40], coefficients of which depend on the original spatial and appearance information of the image.   

> **[info]** edge-preserving：保边去噪  

These filters have the additional advantages of requiring a smaller set of parameters, despite the filter size being potentially as big as the image.  

> **[success]**  
这里的CNN与标准CNN的区别：  
CNN：coefficients依赖于图像的original spatial and appearance information  
标准CNN：训练完成后filters are fixed  
CNN的优点：需要较少的参数集  

While reformulating the steps of the inference algorithm as CNN layers, it is essential to be able to calculate error differentials in each layer w.r.t. its inputs in order to be able to back-propagate the error differentials to previous layers during training.   

> **[success]**  
w.r.t：关于，with respect/regard/reference to   
用CNN层描述mean-field CRF算法的关键是计算每一个的错误偏导  

We also discuss how to calculate error differentials w.r.t. the parameters in each layer, enabling their optimization through the back-propagation algorithm. Therefore, in our formulation, CRF parameters such as the weights of the Gaussian kernels and the label compatibility function can also be optimized automatically during the training of the full network.  

Once the individual steps of the algorithm are broken down as CNN layers, the full algorithm can then be formulated as an RNN.   

> **[success]** 第二步：把CNN reformulate 成RNN  

We explain this in Section 5 after discussing the steps of Algorithm 1 in detail below. In Algorithm 1 and the remainder of this paper, we use $U_i(l)$ to denote the negative of the unary energy introduced in the previous section, i.e., $U_i(l) = -\psi_u(X_i = l)$. In the conventional CRF setting, this input $U_i(l)$ to the mean-field algorithm is obtained from an independent classifier.  

## Initialization

In the initialization step of the algorithm, the operation $Q_i(l) \leftarrow \frac{1}{Z_i}\exp(U_i(l))$, where $Z_i = \sum_l \exp(U_i(l))$, is performed. Note that this is equivalent to applying a softmax function over the unary potentials U across all the labels at each pixel. The softmax function has been extensively used in CNN architectures before and is therefore well known in the deep learning community. This operation does not include any parameters and the error differentials received at the output of the step during back-propagation could be passed down to the unary potential inputs after performing usual backward pass calculations of the softmax
transformation.  

> **[success]**  
$Q_i(l)$是对$U_i(l)$做softmax的结果  
这一步在BP算法中不需要特殊处理。  

## Message Passing

In the dense CRF formulation of [27], message passing is implemented by **applying M Gaussian filters on Q values**. Gaussian filter coefficients are derived based on image features such as the pixel locations and RGB values, which reflect how strongly a pixel is related to other pixels. Since the CRF is potentially fully connected, each filter’s receptive field spans the whole image, making it infeasible to use a brute-force implementation of the filters. Fortunately, several approximation techniques exist to make computation of high dimensional Gaussian filtering significantly faster. Following[27], we use the **permutohedral lattice implementation** [1], which can compute the filter response in O(N) time, where N is the number of pixels of the image [1].

> **[success] 前向传播过程：**  
方法：applying M Gaussian filters on Q values  
过程：permutohedral lattice implementation  
复杂度：O(N)  

During back-propagation, error derivatives w.r.t. the filter inputs are calculated by **sending the error derivatives** w.r.t. the filter outputs through the same M Gaussian filters **in reverse direction**. In terms of permutohedral lattice operations, this can be accomplished by only **reversing the order of the separable filters in the blur stage**, while building the permutohedral lattice, splatting, and slicing in the same way as in the forward pass. Therefore, back-propagation through this filtering stage can also be performed in O(N) time. Following [27], we use two Gaussian kernels, a spatial kernel and a bilateral kernel. In this work, for simplicity, we keep the bandwidth values of the filters fixed.

> **[success] 反向传播过程：**  
方法：sending the error derivatives in reverse direction   
过程：reversing the order of the separable filters in the blur stage    
复杂度：O(N)  

## Weighting Filter Outputs

The next step of the mean-field iteration is **taking a weighted sum of the M filter outputs from the previous step**, for each class label l. When each class label is considered individually, this can be viewed as usual convolution with a 1 × 1 filter with M input channels, and one output channel.   

> **[success]** 这一步可以看作是M输入1输出的1x1的卷积层  

Since both inputs and the outputs to this step are known during back-propagation, the error derivative w.r.t. the **filter weights** can be computed, making it possible to **automatically learn the filter weights** (relative contributions from each Gaussian filter output from the previous stage). **Error derivative** w.r.t. the inputs can also be computed **in the usual manner** to pass the error derivatives down to the previous stage.   

> **[success]**  
filter weights的计算方法：automatically learn the filter weights  
Error derivative的计算方法：in the usual manner  

To obtain a higher number of tunable parameters, in contrast to [27], we use independent kernel weights for each class label. The intuition is that the relative importance of the spatial kernel vs the bilateral kernel depends on the visual class. For example, bilateral kernels may have on the one hand a high importance in bicycle detection, because similarity of colours is determinant; on the other hand they may have low importance for TV detection, given that whatever is inside the TV screen may have different colours.

> **[warning]** 上面这一段没看懂  

## Compatibility Transform

In the compatibility transform step, outputs from the previous step (denoted by $\check Q$ in Algorithm 1) are shared between the labels to a varied extent, depending on the compatibility between these labels.   

> **[warning]** [?] 这一句没看懂  

Compatibility between the two labels l and l ' is parameterized by the label compatibility function µ(l,l ' ). The Potts model, given by $\mu(l,l')=[l\neq l']$, where [.] is the Iverson bracket, assigns a fixed penalty if different labels are assigned to pixels with similar properties. A limitation of this model is that it **assigns the same penalty for all different pairs** of labels. Intuitively, better results can be obtained **by taking the compatibility between different label pairs into account and penalizing the assignments accordingly**. For example, assigning labels “person” and “bicycle” to nearby pixels should have a lesser penalty than assigning “sky” and “bicycle”. Therefore, learning the function µ from data is preferred to fixing it in advance with Potts model. We also relax our compatibility transform model by assuming $\mu(l,l') \neq \mu(l' ,l)$ in general.

Compatibility transform step can be viewed as another convolution layer where the spatial receptive field of the filter is 1 × 1, and the number of input and output channels are both L. Learning the weights of this filter is equivalent to learning the label compatibility function µ. Transferring
error differentials from the output of this step to the input can be done since this step is a usual convolution operation.  

> **[success]** 这一步可以看作是L输入L输出1 × 1的卷积层，前向和后面传播都不需要特殊处理  

## Adding Unary Potentials

In this step, the output from the compatibility transform stage is subtracted element-wise from the unary inputs U. While no parameters are involved in this step, transferring error differentials can be done trivially by copying the differentials at the output of this step to both inputs with the
appropriate sign.  

> **[success]** 这一层没有要计算的参数，直接使用上一层的偏导  

## Normalization

Finally, the normalization step of the iteration can be considered as another softmax operation with no parameters. Differentials at the output of this step can be passed on to the input using the softmax operation’s backward pass.

> **[success]** 这一层是softmax，也没有要计算的参数，直接使用上一层的偏导  

