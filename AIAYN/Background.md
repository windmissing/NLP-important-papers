The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions.   
> **[warning]** [?] sequential computation是指什么？  

    
> **[success]**  
除了样本间的并行性。还要考虑同一个样本内不同时间步的计算的并行性。每个时间步都要根据输入向量计算出hidden向量。且hidden向量不止与当前时间步有并，还是前面的时间步有关。  
RNN采样recurrent的方法来让前面的时间步影响当前时间步，ht依赖的是$h_{t-1}$。但这种方法导致不同时间步无法并行计算。  
CNN则采用卷积的方法来达到这个目的，ht依赖的是$x_{t-1}$，这种方法下，不同时间步可以并行计算其ht。  
[Extended Neural GPU](https://proceedings.neurips.cc/paper/2016/file/fb8feff253bb6c834deb61ec76baa893-Paper.pdf)  
[ByteNet](https://arxiv.org/pdf/1610.10099)  
[ConvS2S](https://arxiv.org/pdf/1705.03122)   

In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet.   
> **[info]**  
翻译：在这些模型中，关联来自两个任意输入或输出位置的信号所需的操作数在位置之间的距离中增加，对于ConvS2S线性增长，而对于ByteNet则对数增长。  

This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.   
> **[info]**  
albeit：尽管  
conteract：低消  

　　
> **[warning]**  
[?] 对于不同模型，“关联来自两个任意输入或输出位置的信号所需的操作数”是怎么计算出来的？  
[?] 操作数增加会有什么问题？  
[?] effective resolution是指什么？为什么会下降？  
[?] 为什么Multi-Head Attention会解决reduced effective resolution的问题？  

**Self-attention**, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].   
> **[info]**  
[reading comprehension](https://arxiv.org/pdf/1601.06733.pdf?source=post_page---------------------------)  
[abstractive summarization]()  
[textual entailment]()  
[learning task-independent sentence representations]()  

End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].   
> **[success]**   
[End-to-end memory networks](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)  
这里提到了两种注意力机制：Self-attention和recurrent attention mechanism。Transformer里这两种注意力机制都用到了，应注意力区分。  

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

> **[success]**  
background介绍了：  
1. 一些基于CNN来并行计算输入序列的hidden state的方法，以及这些方法存在的问题    
2. Self-attention的概念和应用  
3. End-to-end memory networks的概念和应用  