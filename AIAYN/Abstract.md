The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose **a new simple network architecture, the Transformer, based solely on attention mechanisms**, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

> **[success]**  
要解决什么问题：一种新的序列转换模型  
已有的方法：encoder + decoder。其中encoder和decoder是复杂的CNN或RNN，encoder和decoder中间的连接是attention。  
本文的主要贡献：一种新的序列转换模型，encoder、decoder以及它们中间的连接，都是基于attention。没有CNN或RNN。  
得到了什么结果：在机器翻译任务上使用这种模型，性能优于STOA、平行程度高、训练时间短。  
