# Introduction

Named entity recognition (NER) is a challenging learning problem. One the one hand, in most languages and domains, there is only a very small amount of supervised training data available. On the other, there are few constraints on the kinds of words that can be names, so generalizing from this small sample of data is difficult. As a result, carefully constructed orthographic features and language-specific knowledge resources, such as gazetteers, are widely used for solving this task.   
> **[info]**  
orthographic features：正交特征  

Unfortunately, language-specific resources and features are costly to develop in new languages and new domains, making NER a challenge to adapt. Unsupervised learning from unannotated corpora offers an alternative strategy for obtaining better generalization from small amounts of supervision.   
> **[info]**  
unannotated corpora：未标注样本  

However, even systems that have relied extensively on unsupervised features (Collobert et al., 2011; Turian et al., 2010; Lin and Wu, 2009; Ando and Zhang, 2005b, inter alia) have used these to **augment, rather than replace**, hand-engineered features (e.g., knowledge about capitalization patterns and character classes in a particular language) and specialized knowledge resources (e.g., gazetteers).  

> **[success]**   
NER的Challenge：  
1. 带标注的训练样本少，name没有规律，因此难以识别name   
2. gazetteer是预定义的name资源，无法用于新语言、新领域  
3. 监督学习+非监督学习的方法仍摆脱不了gazetteer  

In this paper, we present neural architectures for NER that use **no language-specific resources or features** beyond **a small amount of supervised training data** and **unlabeled corpora**.   

> **[success]**   
本文NER算法所需要的资源：  
（1） 少量的带标注data  
（2） 未标注的corpora  
（3） 无language-specific资源/知识  

Our models are designed to capture two intuitions. First, since names often consist of multiple tokens, reasoning jointly over tagging decisions for each token is important. We compare two models here, (i) a bidirectional LSTM with a sequential conditional random layer above it (LSTM-CRF; §2), and (ii) a new model that constructs and labels chunks of input sentences using an algorithm inspired by transition-based parsing with states represented by stack LSTMs (S-LSTM; §3).   
> **[warning]** 这一段没看懂？  
先验假设1：name都是多token的   
[?] reasoning jointly over tagging decisions for each token这句话是什么意思？  
[?] 这个先验假设与这两个模型是什么关系？  

Second, token-level evidence for “being a name” includes both orthographic evidence (what does the word being tagged as a name look like?) and distributional evidence (where does the word being tagged tend to occur in a corpus?). To capture orthographic sensitivity, we use character-based word representation model (Ling et al., 2015b) to capture distributional sensitivity, we combine these representations with distributional representations (Mikolov et al., 2013b).   

> **[warning]**     
先验假设2：name具有正交敏感性和分布敏感性。  
使用“字符级词表示”捕获正交敏感性，使用“分布表示”捕获分布敏感性。  
[?]分布表示？  

Our word representations combine both of these, and dropout training is used to encourage the model to learn to trust both sources of evidence (§4).

> **[success]**     
dropout: 信任所有source  

Experiments in English, Dutch, German, and Spanish show that we are able to obtain state-of-the-art NER performance with the LSTM-CRF model in Dutch, German, and Spanish, and very near the state-of-the-art in English without any hand-engineered features or gazetteers (§5). The transition-based algorithm likewise surpasses the best previously published results in several languages, although it performs less well than the LSTM-CRF model.
