# Abstract

An attentional mechanism has lately be enused  to  improve  neural  machine  translation  (NMT)  by  selectively  focusing  onparts of the source sentence during translation. However,   there  has  been  little work  exploring  useful  **architectures  for attention-based  NMT**.  This  paper  examines two simple and effective classes of attentional  mechanism:   **a global approach** which always attends to all source words and **a local one** that only looks at a subsetof source words at a time. We demonstratethe effectiveness of both approaches on the WMT  translation  tasks  between  Englishand German in both directions. With local attention, we achieve a significant gain of5.0 BLEU points over non-attentional systems that already incorporate known techniques  such  as  dropout. Our  ensemble model  using  **different  attention  architectures** yields a new state-of-the-art result in the WMT’15 English  to German  translation task with 25.9 BLEU points,  an improvement  of  1.0  BLEU  points  over  the existing best system backed by NMT and an n-gram reranker.11      

> **[info]**  
基于注意力机制的NMT问题的模型结构：  
global方法、local方法、结果更好的新方法  

