Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.   
> **[warning]**  
为什么说这种方法不能利用位置信息呢？self-attention看上去有点像CNN，也是有序列顺序关系的。  

To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed.   
> **[success]**  
目的：用Positional Encoding来表达序列顺序信息  
用Input Embedding + Positional Encoding的方法来表达带序列顺序信息的输入数据
用Output Embedding + Positional Encoding的方法来表达带序列顺序信息的上一个时间步的输出数据  
![](/images/2020/7.png)   
具体步骤：  
1. 生成Positional Encoding  
2. 把Positional Encoding与Input Embedding结合起来，结合的方式是向量相加  

There are many choices of positional encodings, learned and fixed [9].   
In this work, we use sine and cosine functions of different frequencies:   

$$
\begin{aligned}
    PE(pos, i) = \sin\left(\frac{pos}{10000^{\frac{i}{d_{model}}}}\right)   &&   i \text{ is even}\\
    PE(pos, i) = \cos\left(\frac{pos}{10000^{\frac{i-1}{d_{model}}}}\right)   &&   i \text{ is odd}
\end{aligned}
$$

> **[success]**  
pos：代表序列中的第几个时间步  
i：代表某个时间步的Encoding Vector中的第几个值，i的最大值为PE的维度，也就是Input Embedding的维度。    
$d_{model}$：PE Vector的长度，与Embedding的长度相同   

where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of P Epos.    
> **[success]**  
以上是本文使用的PE计算方法，称为sinusoidal PE，具有以下特点：  
- [?] 波长形成从2π到10000·2π的[几何级数](https://windmissing.github.io/mathematics_basic_for_ML/Numbers/GeometricProgression.html)。这个序列跟几何级数什么关系？    
- [?] 对于任何固定偏移k，PE（pos + k）可以表示为PE（pos）的线性函数。这个线性关系的公式推导不出来。[三角函数公式](https://windmissing.github.io/mathematics_basic_for_ML/Mathematics/Formula/trigonometric.html)    
- 推断序列可以长于训练序列  

We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training. 

> **[success]**  
除了本文所用的sinusoidal PE，还有其它PE计算方式，例如[learned positional attention](https://arxiv.org/pdf/1705.03122)。  
从本文实验上看，两种PE算法的性能差不多，选择sinusoidal PE是因为它的第三个特点。  