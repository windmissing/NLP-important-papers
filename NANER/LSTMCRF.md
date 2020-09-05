# LSTM - CRF Model

We provide a brief description of LSTMs and CRFs, and present a hybrid tagging architecture. This architecture is similar to the ones presented by Collobert et al. (2011) and Huang et al. (2015).  

## LSTM

Recurrent neural networks (RNNs) are a family of neural networks that operate on sequential data. They take as input a sequence of vectors (x 1 ,x 2 ,...,x n ) and return another sequence (h 1 ,h 2 ,...,h n ) that represents some information about the sequence at every step in the input. Although RNNs can, in theory, learn long dependencies, in practice they fail to do so and tend to be biased towards their most recent inputs in the sequence (Bengio et al., 1994). Long Short-term Memory Networks (LSTMs) have been designed to combat this issue by incorporating a memory-cell and have been shown to capture long-range dependencies. They do so using several gates that control the proportion of the input to give to the memory cell, and the proportion from the previous state to forget (Hochreiter and Schmidhuber, 1997). We use the following implementation:

> **[success]**   
输入向量序列x分别正向LSTM和反向LSTM得到向量h1和向量h2，  
h1和h2合并成一向量，称为h，是关于输入序列的some information  

$$
\begin{aligned}
i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)  \\
c_t = (1-i_t)\odot c_{t-1} + i_t \odot \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)  \\
o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)   \\
h_t = o_t \odot \tanh(c_t)
\end{aligned}
$$

> **[success]**   
LSTM的过程：  
1. 把$x_t$和$h_{t-1}$合并成一个大的输入向量$[x_t, h_{t-1}]$   
2. 根据输入向量$[x_t, h_{t-1}]$和缓存状态$C_{t-1}$计算gate  
3. gate决定输入向量$[x_t, h_{t-1}]$和缓存状态$C_{t-1}$各起多少作用，得到新的缓存状态$C_t$  
4. 根据输入向量$[x_t, h_{t-1}]$和新的缓存状态$C_t$计算另一个gate  
5. gate决定有多个$C_t$进入$h_t$

where σ is the element-wise sigmoid function, and $\odot$ is the element-wise product.
For a given sentence (x 1 ,x 2 ,...,x n ) containing n words, each represented as a d-dimensional vector, an LSTM computes a representation $h_t$ of the left context of the sentence at every word t. Naturally, generating a representation of the right context $h_t$ as well should add useful information. This can be achieved using a second LSTM that reads the same sequence in reverse. We will refer to the former as the forward LSTM and the latter as the backward LSTM. These are two distinct networks with different parameters. This forward and backward LSTM pair is referred to as a bidirectional LSTM (Graves and Schmidhuber, 2005).  

The representation of a word using this model is obtained by concatenating its left and right context representations, $h_t = [h_t;h_t]$. These representations effectively include a representation of a word in context, which is useful for numerous tagging applications.

## CRF Tagging Models

> **[success]**  
CRF 条件随机场
$$
P(Y|X)
$$

> 说明：  
公式代表在已知X的情况下预测Y的分布。  
X是输入变量，即观测序列   
Y是输出变量，即标记序列   

A very simple—but surprisingly effective—tagging model is to use the h t ’s as features to make independent tagging decisions for each output y t (Ling et al., 2015b). Despite this model’s success in simple problems like POS tagging, its independent classification decisions are limiting when there are **strong dependencies across output labels**. NER is one such task, since the “grammar” that characterizes interpretable sequences of tags imposes several hard constraints (e.g., I-PER cannot follow B-LOC; see §2.4 for details) that would be impossible to model with independence assumptions.

> **[success]**  
由于output之间有strong dependency，不能直接根据$h_t$预测$y_t$  
[?] Pos tagging

Therefore, instead of modeling tagging decisions independently, we **model them jointly using a conditional random field** (Lafferty et al., 2001).   
> **[success]**  
解决方法：
使用CRF做tagging决定“jointly”

For an input sentence

$$
X = (x_1, x_2, \cdots, x_n),
$$

we consider P to be the matrix of scores output by the bidirectional LSTM network. P is of size n × k, where k is the number of distinct tags, and $P_{i,j}$ corresponds to the score of the j th tag of the i th word in a sentence.   

> **[success]**  
P是$n \times k$的矩阵，  
$P_{ij}$代表第i个单词打第j个tag的分数   
n代表句子长度，k代表tag个数  

For a sequence of predictions   
$$
y = (y 1 ,y 2 ,...,y n ),
$$

we define its score to be   
$$
s(X,y) = \sum_{i=0}^n A_{y_i, y_{i+1}} + \sum_{i=1}^n P_{i, y_i}
$$

> **[success]**  
公式说明：  
x: 输入的word组成的序列  
y: 由预测的tag组成的序列  
第一项：tag前后的关联性分数  
第二项：独立分类的分数  
P在上一段已经解释，A将在下一段解释。


where A is a matrix of transition scores such that A i,j represents the score of a transition from the tag i to tag j. y 0 and y n are the start and end tags of a sentence, that we add to the set of possible tags. A is therefore a square matrix of size k+2.

> **[success]**  
A：$(k+2) \times (k+2)$状态转移矩阵  
$A_{ij}$代表tag i向tag j转移的分数  
矩阵A中增加了start和end两个tag，因此size是k+2  

A softmax over all possible tag sequences yields a probability for the sequence y:

$$
\begin{aligned}
p(y|X) = \frac{e^{s(X, y)}}{\sum_{\tilde y \in Y_X} e^{s(X, \tilde y)}}
\end{aligned}
$$

> **[success]**  
$\sum$代表遍历所有的可能的y

During training, we maximize the log-probability of the correct tag sequence:  
$$
\begin{aligned}
\log(p(y|X)) &=& s(X, y) - \log\left(\sum_{\tilde y \in Y_X} e^{s(X, \tilde y)}\right)  \\
&=& s(X, y) - \text{logadd}_{\tilde y \in Y_X} s(X, \tilde y), && (1)
\end{aligned}
$$

> **[warning]** [?] 把上面的公式对数化，以防下溢？  

where $Y_X$ represents all possible tag sequences (even those that do not verify the IOB format) for a sentence X.   
> **[warning]** 为什么要包含明显不可能存在的y？这种y出现的概率为0，没有必要算进去呀？  

From the formulation above, it is evident that we encourage our network to produce a valid sequence of output labels.   
> **[warning]** 没看出来怎么鼓励产生合法sequence的。  

While decoding, we predict the output sequence that obtains the maximum score given by:  

$$
y^* = \arg\max s(X, \tilde y)
$$

> **[success]**   
在所有y中找到一个使s最大的y  

Since we are only modeling bigram interactions between outputs, both the summation in Eq. 1 and the maximum a posteriori sequence y ∗ in Eq. 2 can be computed using dynamic programming.

> **[success]**   
bigram interaction是指$s_{i+1}$只依赖于$s_i$的结果，因此可以用DP   

## Parameterization and Training

The scores associated with each tagging decision for each token (i.e., the $P_{i,y}$’s) are defined to be the dot product between the embedding of a word-in-context computed with a bidirectional LSTM—exactly the same as the POS tagging model of Ling et al. (2015b) and these are combined with bigram compatibility scores (i.e., the A y,y 0 ’s). This architecture is shown in figure 1. Circles represent observed variables, diamonds are deterministic functions of their parents, and double circles are random variables.
![](/NANER/assets/2.png)  

> **[success]**   
![](/NANER/assets/1.png)  
虚线圆圈代表向量，实线圆圈代表unit。  
x为输入层  
l为正向LSTM的一个时间步    
r为反向LSTM的一个时间步  
f为一个普通的隐藏层    
c为crf层    
y为输出层  
用crf层代表softmax层生成y。   

The parameters of this model are thus the matrix of bigram compatibility scores A, and the parameters that give rise to the matrix P, namely the parameters of the bidirectional LSTM, the linear feature weights, and the word embeddings. As in part 2.2, let x i denote the sequence of word embeddings for every word in a sentence, and y i be their associated tags. We return to a discussion of how the embeddings x i are modeled in Section 4. The sequence of word embeddings is given as input to a bidirectional LSTM, which returns a representation of the left and right context for each word as explained in 2.1.

> **[success]**   
要训练的参数：  
A、双向LSTM的参数  

These representations are concatenated (c i ) and linearly projected onto a layer whose size is equal to the number of distinct tags. Instead of using the softmax output from this layer, we use a CRF as previously described to take into account neighboring tags, yielding the final predictions for every word y i . Additionally, we observed that adding a hidden layer between c i and the CRF layer marginally improved our results. All results reported with this model incorporate this extra-layer. The parameters are trained to maximize Eq. 1 of observed sequences of NER tags in an annotated corpus, given the observed words.

## Tagging Schemes

The task of named entity recognition is to assign a named entity label to every word in a sentence. A single named entity could span several tokens within a sentence. Sentences are usually represented in the IOB format (Inside, Outside, Beginning) where every token is labeled as B-label if the token is the beginning of a named entity, I-label if it is inside a named entity but not the first token within the named entity, or O otherwise. However, we decided to use the IOBES tagging scheme, a variant of IOB commonly used for named entity recognition, which encodes information about singleton entities (S) and explicitly marks the end of named entities (E). Using this scheme, tagging a word as I-label with high-confidence narrows down the choices for the subsequent word to I-label or E-label, however, the IOB scheme is only capable of determining that the subsequent word cannot be the interior of another label. Ratinov and Roth (2009) and Dai et al. (2015) showed that using a more expressive tagging scheme like IOBES improves model performance marginally. However, we did not observe a significant improvement over the IOB tagging scheme.

> **[success]**   
IOB format: 
Inside, Outside, Beginning  
IOBES tagging scheme：  
Single Entity, End of Entity