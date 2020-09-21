# Neural Machine Translation

A  neural  machine  translation  system  is  a  neural network that **directly models the conditional probability p(y|x) of  translating  a  source  sentence,x1, . . . , xn,  to a target  sentence,y1, . . . , ym**.3 A basic form of NMT consists of two components:(a) an encoder which computes a representations for each source sentence and (b) a decoder which generates one target word at a time and hence decomposes the conditional probability as:  
$$
\begin{aligned}
\log p(y|x) =\sum^m_{j=1}\log p(y_j|y_{<j},s)  && (1)
\end{aligned}
$$

> **[success]**  
NMT是指计算给定输入序列对应输出序列的概率分布。  
NMT模型分为encoder和decoder两部分。  
encoder用于把输入序列的信息打包。  
decoder用于生成另一个序列。  
以上公式是decoder的公式。  

A   natural    choice    to   model    such    a   decomposition    in    the    decoder    is    to    use    recurrent neural network(RNN)architecture,   which   most   of   the   recent   NMT   work such as(Kalchbrenner and Blunsom, 2013;Sutskever et al., 2014;Cho et al., 2014;Bahdanau et al., 2015;Luong et al., 2015;Jean et al., 2015)  have  in  common.   They,  however,  differ in terms of which RNN architectures are  used  for  the  decoder  and  how  the  encoder computes the source sentence representation $s$.  

> **[success]**  
encoder：不同的模型对s的编码方式不同。  
decoder：NMT常用docoder模型是RNN，但在RNN的结构上有所不同   

Kalchbrenner and Blunsom (2013)used an RNN   with   the   standard   hidden   unit   for   the decoder  and  a  convolutional  neural  network  for encoding the source sentence representation.   

> **[success]** encoder: CNN, decoder: RNN(standard unit)  

On the  other  hand,  both  Sutskever et al. (2014)  and Luong et al. (2015) stacked  multiple layers  of anRNN with a Long Short-Term  Memory  (LSTM)hidden unit for both the encoder and the decoder.  

> **[success]** encoder & decoder：多层LSTM  

Cho et al. (2014),Bahdanau et al. (2015),andJean et al. (2015) all adopted a different version ofthe RNN with an LSTM-inspired hidden unit, thegated recurrent unit (GRU), for both components.4  

> **[success]** encoder & decoder：多层GRU    

In more detail, one can parameterize the probability of decoding each word yj as:  
$$
\begin{aligned}
p(y_j|y_{<j},s) = softmax (g(h_j))  &&  (2)
\end{aligned}
$$

with g being the **transformation function** that outputs a vocabulary-sized  vector.5 Here,hj is the RNN hidden unit, abstractly computed as:  
$$
\begin{aligned}
h_j=f(h_{j−1},s) && (3)
\end{aligned}
$$

> **[warning]** [?]图（1）中是怎么体现s这个条件的？   

where f computes   the   **current   hidden   state** given   the   previous   hidden   state   and   can   be either  a  vanilla  RNN  unit,  a  GRU, or  an  LSTMunit. In    (Kalchbrenner and Blunsom, 2013;Sutskever et al., 2014;Cho et al., 2014;Luong et al., 2015),the source representations is   only   used   once   to   initialize   the decoder  hidden  state. On  the  other  hand,   in(Bahdanau et al., 2015;Jean et al., 2015)and this  work,s,  in  fact,  implies  a  set  of  source hidden states which are consulted throughout the entire course of the translation  process.  Such an approach is referred to as an attention mechanism,which we will discuss next.  

> **[success]**  
是否使用注意力机制的区别：  
无注意力机制：s只用于初始化hidden state  
有注意力机制：[?]没看懂  

In this work, following (Sutskever et al., 2014;Luong et al., 2015),  we  use  the  stacking  LSTM architecture  for  our  NMT systems,  as  illustrated in  Figure  1.   

> **[warning]** [?]怎么体现出stacking的作用？  

We  use  the  LSTM  unit  defined  in(Zaremba et al., 2015).   Our  training  objective  isformulated as follows:  
$$
\begin{aligned}
J_t=\sum_{(x,y)\in D} −\log p(y|x) && (4)
\end{aligned}
$$

with D being our parallel training corpus.

> **[success]** 对所有样本，让经验输出的对数似然对大  

