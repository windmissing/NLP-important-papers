# Related Work

In this section we describe previous work relevant to the approaches discussed in this paper. A more detailed discussion on language modeling research is provided in (Mikolov, 2012).

## Language Models

Language Modeling (LM) has been a central task in NLP. The goal of LM is to **learn a probability distribution over sequences of symbols pertaining to a language**.   

> **[success]**  
LM的目标：计算一个语言中各种序列的概率分布  

Much work has been done on both parametric (e.g., log-linear models) and non-parametric approaches (e.g., count-based LMs). Count-based approaches (based on statistics of N-grams) typically add smoothing which account for unseen(yet possible) sequences, and have been quite successful.   

> **[success]**  
已有的LM算法：  
（1）参数算法：log-linear模型  
（2）非参数算法：基于统计的LM，例如N-grams+平滑  

To this extent, Kneser-Ney smoothed 5-gram models (Kneser & Ney, 1995) are a fairly strong baseline which, for large amounts of training data, have challenged other parametric approaches based on Neural Networks (Bengio et al., 2006).  

> **[success]**  
本文使用smoothed 5-gram模型作为baseline。  
这是一种性能非常好的非参数模型。  

Most of our work is based on **Recurrent Neural Networks (RNN)** models which **retain long term dependencies**. To this extent, we used the **Long-Short Term Memory** model (Hochreiter & Schmidhuber, 1997) which uses a gating mechanism (Gers et al., 2000) to ensure proper propagation of information through many time steps.   

> **[success]**  
To this extent：在这个程度上  
本文使用RNN+LSTM以保持长期依赖  

Much work has been done on small and large scale RNN-based LMs (Mikolov et al., 2010; Mikolov, 2012; Chelba et al., 2013; Zaremba et al., 2014; Williams et al., 2015; Ji et al., 2015a; Wang & Cho, 2015; Ji et al., 2015b). The architectures that we considered in this paper are represented in Figure 1.  

In our work, we train models on the popular One Billion Word Benchmark, which can be considered to be **a medium-sized data set for count-based LMs but a very large data set for NN-based LMs**. This regime is most interesting to us as we believe learning a very good model of human language is a complex task which will require large models, and thus large amounts of data.  

> **[info]** regime：政权

Further advances in data availability and computational resources helped our study. We argue this leap in scale enabled tremendous advances in deep learning. A clear example found in computer vision is Imagenet (Deng et al., 2009), which enabled learning complex vision models from large amounts of data (Krizhevsky et al., 2012).

A crucial aspect which we discuss in detail in later sections is the size of our models. Despite the large number of parameters, we try to minimize computation as much as possible by adopting a strategy proposed in (Sak et al., 2014) of **projecting a relatively big recurrent state space down** so that the matrices involved remain relatively small, yet the model has large memory capacity.

> **[success]**  
matrices：矩阵
使用“projecting a relatively big recurrent state space down”的算法可以用较小的矩阵而得到较大的容量。  

## Convolutional Embedding Models

There is an increased interest in incorporating character-level inputs to build word embeddings for various NLP problems, including part-of-speech tagging, parsing and language modeling (Ling et al., 2015; Kim et al., 2015;  

> **[info]** incorporating：合并  

Ballesteros et al., 2015). The additional character information has been shown useful on relatively small benchmark data sets.   

> **[success]**  
在word embedding中加入字符级表示，在小数据集场景中非常有用。  

The approach proposed in (Ling et al., 2015) builds word embeddings using bidirectional LSTMs (Schuster & Paliwal, 1997; Graves & Schmidhuber, 2005) over the characters. The recurrent networks process sequences of characters from both sides and their final state vectors are concatenated. The resulting representation is then fed to a Neural Network. This model achieved very good results on a part-of-speech tagging task.  

> **[success]**  
用双向LSTM生成字符级表示  
过程：两个RNN分别从字符序列的两端开始读入字符，最后的向量是两个RNN向量的拼接。   
性能：在“part-of-speech tagging”任务中有好的性能。  

In (Kim et al., 2015), the words characters are processed by a 1-d CNN (Le Cun et al., 1990) with max-pooling across the sequence for each convolutional feature. The resulting features are fed to a 2-layer highway network (Srivastava et al., 2015b), which allows the embedding to learn semantic representations.   

> **[success]**  
用卷积生成字符级表示   
过程：字符序列 -- [1维CNN](https://windmissing.github.io/Bible-DeepLearning/Chapter9/7Data.html) -- [最大池](https://windmissing.github.io/Bible-DeepLearning/Chapter9/3Pooling.html) --- 2层[highway](https://windmissing.github.io/Bible-DeepLearning/Chapter9/Highway.html) -- embedding   

The model was evaluated on small scale language modeling experiments for various languages and matched the best results on the PTB data set despite having 60% fewer parameters.  

> **[warning]**  
性能：  
（1）在PTB上有好的效果  
（2）参数减少60%，[?]跟谁比少60%？

## Softmax Over Large Vocabularies

Assigning probability distributions over large vocabularies is computationally challenging. For modeling language, maximizing log-likelihood of a given word sequence leads to optimizing cross-entropy between the target probability distribution (e.g., the target word we should be predicting), and our model predictions p. Generally, predictions come from a linear layer followed by a Softmax non-linearity:$p(w) = \frac{\exp(z_w)}{\sum_{w'\in V}\exp(z_{w'})}$where $z_w$ is the logit corresponding to a word w. The logit is generally computed as an inner product $z_w = h^\top e_w$ where h is a context vector and $e_w$ is a “word embedding” for w.  

> **[success]**  
目标是最大化[对数似然估计](https://windmissing.github.io/mathematics_basic_for_ML/Probability/likelihood.html)，即“目标概率分布”和“预测概率分布”的[交叉熵](https://windmissing.github.io/mathematics_basic_for_ML/Information/Divergence.html)  
在NN中通常使用[softmax](https://windmissing.github.io/Bible-DeepLearning/Chapter6/2Gradient/2OutputUnit/3Softmax.html)来计算多分类问题的“预测概率分布”。  

The main challenge when |V| is very large (in the order of one million in this paper) is the fact that computing all inner products between h and all embeddings becomes prohibitively slow during training (even when exploiting matrix-matrix multiplications and modern GPUs).   

> **[success]**  
prohibitively slow：太慢了   
大规模LM的瓶颈在于用softmax计算“预测概率分布”太慢了。  

Several approaches have been proposed to cope with the scaling issue: **importance sampling** (Bengio et al., 2003; Bengio & Senécal, 2008), **Noise Contrastive Estimation (NCE)** (Gutmann & Hyvärinen, 2010; Mnih & Kavukcuoglu, 2013), **self normalizing partition functions** (Vincent et al., 2015) or **Hierarchical Softmax** (Morin & Bengio, 2005; Mnih & Hinton, 2009) – they all offer good solutions to this problem.   

> **[success]**  
Contrastive：对比的   
解决以上问题的已有方法包括：  
（1）importance sampling  
（2）Noise Contrastive Estimation (NCE)  
（3）self normalizing partition functions  
（4）Hierarchical Softmax  

We found importance sampling to be quite effective on this task, and explain the connection between it and NCE in the following section, as they are closely related.

> **[success]**  
本文结论：  
（1）importance sampling在本文中有效  
（2）importance sampling和NCE有联系  


