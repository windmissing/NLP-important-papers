# Abstract

State-of-the-art named entity recognition systems rely heavily on hand-crafted features and domain-specific knowledge in order to learn effectively from the small, supervised training corpora that are available.  
> **[info]**  
state-of-the art：最先进的  
hand-crafted features：手动设计的特征  
corpora：语料库

In this paper, we introduce two new neural architectures—one based on **bidirectional LSTMs and conditional random fields**, and the other that **constructs and labels segments using a transition-based approach inspired by shift-reduce parsers**.   

> **[success]**   
传统方法：基于人肉选feature和领域知识  
新方法：  
（1）双向LSTM + [CRF](https://windmissing.github.io/LiHang-TongJiXueXiFangFa/Chapter11/crf.html)  
（2）移位归约分析 + 基于转移的方法  

Our models rely on two sources of information about words: character-based word representations learned from the supervised corpus and unsupervised word representations learned from unannotated corpora.  
> **[success]**   
使用两种表示方法：  
（1）基于标注语料库的字符级单词表示   
（2）基于未标注语料库的单词级表示  

Our models obtain state-of-the-art performance in NER in four languages without resorting to any language-specific knowledge or resources such as gazetteers.

> **[success]**   
优点：  
（1）性能好  
（2）不需要特定语言的相关知识   
（3）不需要gazetteers资源  