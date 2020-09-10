# Introduction

Progress on the path from shallow bag-of-words information retrieval algorithms to machines capable of reading and understanding documents has been slow. Traditional approaches to machine reading and comprehension have been based on either **hand engineered grammars**[1], or **information extraction methods** of detecting predicate argument triples that can later be queried as a relational database [2].   

> **[success] 常见的阅读理解模型训练方法：**  
（1）hand engineered grammars  
（2）information extraction methods  

Supervised machine learning approaches have largely been absent from this space due
to both the lack of large scale training datasets, and the difficulty in structuring statistical models flexible enough to learn to exploit document structure.  

> **[success]**  
常见方法都是基于非监督学习的，因为可用于监督学习的数据太少。  

While obtaining supervised natural language reading comprehension data has proved difficult, some researchers have explored generating synthetic narratives and queries [3, 4].   

> **[info]**  
synthetic：合成的  
narratives：叙述  

Such approaches allow the generation of almost unlimited amounts of supervised data and enable researchers to isolate the performance of their algorithms on individual simulated phenomena. Work on such data has shown that neural network based models hold promise for modelling reading comprehension, something that we will build upon here. Historically, however, many similar approaches in Computational Linguistics have failed to manage the transition from synthetic data to real environments, as such
closed worlds inevitably fail to capture the complexity, richness, and noise of natural language [5].  

> **[success]**  
Computational Linguistics：计算机语言  
生成监督数据的方法1：生成synthetic narratives and queries  
优点：  
（1）无限制的supervised data  
（2）让算法性能与实验环境解耦  
（3）让基于NN的阅读理解模型hold promise  
缺点：  
在合成数据上工作很好的模型，无法用于真实环境。因为自然语言的complexity, richness, noise。  

In this work we seek to directly address the lack of real natural language training data by introducing a novel approach to building a supervised reading comprehension data set. We observe that summary and paraphrase sentences, with their associated documents, can be readily converted to context–query–answer triples using simple entity detection and anonymisation algorithms.   

> **[success]**  
paraphrase：改述  
readily：便利地  
anonymisation：匿名  
生成监督数据的方法2（本文推荐的方法）：  
基于文章对应的summary and paraphrase生成(context, query, answer)元组。  

Using this approach we have collected two new corpora of roughly a million news stories with associated queries from the CNN and Daily Mail websites.

We demonstrate the efficacy of our new corpora by building novel deep learning models for reading comprehension.   

> **[success]**  
efficacy：功效  
问：怎么证明生成的labelled data对训练阅读理解模型有帮助？  
答：使用基于些labelled data的监督学习模型的训练结果优于使用原始数据的非监督数据的结果。  

These models draw on recent developments for incorporating attention mechanisms into recurrent neural network architectures [6, 7, 8, 4]. This allows a model to focus on the aspects of a document that it believes will help it answer a question, and also allows us to visualises its inference process.   

> **[success]**  
基于labelled data的监督学习模型：RNN + 注意力机制  

We compare these neural models to a range of baselines and heuristic benchmarks based upon a traditional frame semantic analysis provided by a state-of-the-art natural language processing (NLP) pipeline.   

> **[success]**  
benchmarks：基准点   
frame semantic analysis：框架语义分析  
问：baseline VS benchmarks  
答：一个算法被称为baseline，基本上表示比这个算法性能还差的基本上不能接受的，除非方法上有革命性的创新点，而且还有巨大的改进空间和超越benchmark的潜力，只是因为是发展初期而性能有限。所以baseline有一个自带的含义就是“性能起点”。benchmark一般是和同行中比较牛的算法比较，比牛算法还好，那你可以考虑发好一点的会议/期刊；baseline一般是自己算法优化和调参过程中自己和自己比较，目标是越来越好，当性能超过benchmark时，可以发表了，当性能甚至超过SOTA时，恭喜你，考虑投顶会顶刊啦。  
基于原始数据的非监督学习模型：  
a traditional frame semantic analysis provided by a state-of-the-art natural language processing (NLP) pipeline。  

Our results indicate that the neural models achieve a higher accuracy, and do so
without any specific encoding of the document or query structure.


