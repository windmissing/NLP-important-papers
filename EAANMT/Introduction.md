# Introduction

Neural   Machine   Translation   (NMT)   achieved state-of-the-art performances in large-scale translation   tasks   such   as   from   English   to   French(Luong et al., 2015)    and    English    to    German(Jean et al., 2015).  NMT is appealing since it requires minimal domain knowledge and is conceptually simple.   The model by Luong et al. (2015)reads through all the source words until the end-of-sentence symbol<eos>is reached. It then starts1All  our   code   and  models  are   publicly  available  athttp://nlp.stanford.edu/projects/nmt.BCD<eos>XYZXYZ<eos>AFigure 1:Neural machine translation– a stacking recurrent architecture for translating a sourcesequenceA B C Dinto  a  target  sequenceX YZ. Here,<eos>marks the end of a sentence.emitting one target word at a time, as illustrated inFigure 1. NMT is often a large neural network thatis trained in an end-to-end fashion and has the ability to generalize well to very long word sequences.This means the model does not have to explicitly store gigantic phrase tables and language modelsas in  the case  of standard  MT; hence,  NMT hasa small memory footprint.    

> **[info]**   
gigantic：巨大的   
memory footprint：内存占用  

Lastly,  implementingNMT decoders is easy unlike the highly intricate decoders in standard MT (Koehn et al., 2003).

> **[success]**  
intricate：复杂的  
NMT的特点：  
（1）在大规模翻译任务上性能好  
（2）需要较少的领域知识  
（3）conceptually simple  
（4）通常是一个很大的神经网络  
（5）能够生成长的序列   
（6）内存占用少  
（7）decoder实现简单  

In   parallel,   the   concept   of   “attention”   has gained popularity  recently in training  neural networks,  allowing  models  to  learn  alignments  between  different  modalities,  e.g.,  between  image objects  and  agent  actions  in  the  dynamic  control problem  (Mnih et al., 2014),  between speechframes  and  text  in  the  speech  recognition  task(?),  or  between  visual  features  of  a  picture  andits  text  description  in  the  image  caption  generation  task  (Xu et al., 2015).  

> **[warning]**  
modalities：形态  
attention的优势是learn  alignments  between  different  modalities  
后面是一些具体的应用，怎么alignment的不懂。  

In  the  context  of NMT, Bahdanau et al. (2015) has successfully applied such attentional mechanism to jointly translate and  align  words.   To the  best  of our knowledge, there has not been any other work exploringthe use of attention-based architectures for NMT.  

> **[success]**  
注意力机制在NMT中的应用：jointly translate and  align  words  

In this work, we design, with simplicity and effectiveness in mind, two novel types of attention-based  models:   a global approach  in  which  all source words are attended and alocalone whereby only a subset of source words are considered at a time.  **The former approach resembles the modelof (Bahdanau et al., 2015) but is simpler architecturally.  The latter can be viewed as an interesting blend between the hard and soft attention models** proposed  in  (Xu et al., 2015):   

> **[warning]** [?] the hard and soft attention models?  

it  is  computationally  less  expensive  than  the  global  model  or  thesoft attention; at the same time, unlike the hard attention, the local attention is differentiable almost every where,  making  it  easier  to  implement  and train.2  

> **[success]**  
local appproach的特点:  
（1）比global和soft的计算复杂度低  
（2）比hard容易实现和训练  

Besides,  we  also  examine  various  alignment functions for our attention-based models.  

> **[success]**  
alignment function：见3.1  

Experimentally,  we  demonstrate  that  both  of our  approaches  are  effective  in  the  WMT  translation tasks between English and German in bothdirections.   Our attentional  models  yield  a boost of up to 5.0 BLEU over non-attentional  systemswhich already incorporate known techniques suchas  dropout.   For  English  to  German  translation,we  achieve  new  state-of-the-art  (SOTA)  resultsfor  both  WMT’14  and  WMT’15,  outperformingprevious  SOTA  systems,  backed  by  NMT  mod-els  andn-gram  LM rerankers,  by  more than  1.0BLEU. We conduct extensive analysis to evaluateour models in terms of learning, the ability to han-dle long sentences, choices of attentional architec-tures, alignment quality, and translation outputs.
