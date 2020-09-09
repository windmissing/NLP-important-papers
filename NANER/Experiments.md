# 5 Experiments
This section presents the methods we use to train our models, the results we obtained on various tasks and the impact of our networks’ configuration on model
performance.

## 5.1 Training

For both models presented, we train our networks using the **back-propagation** algorithm updating our parameters on every training example, one at a time, using stochastic gradient descent (SGD) with a learning rate of 0.01 and a gradient clipping of 5.0.   

> **[warning]** 
one at a time的意思是on-line算法？  

Several methods have been proposed to enhance the performance of SGD, such as Adadelta (Zeiler, 2012) or Adam (Kingma and Ba, 2014). Although we observe faster convergence using these methods, none of them perform as well as SGD with gradient clipping.  

> **[success]**  
sgd的加速算法虽然收敛更快，但结果不如sgd+clipping  

Our LSTM-CRF model uses a single layer for the forward and backward LSTMs whose dimensions are set to 100. Tuning this dimension did not significantly impact model performance. We set the dropout rate to 0.5. Using higher rates negatively impacted our results, while smaller rates led to longer training time.  

> **[success]**  
dropout rate:  
大 --> 性能变差  
小 --> 收敛变慢

The stack-LSTM model uses two layers each of dimension 100 for each stack. The embeddings of the actions used in the composition functions have 16 dimensions each, and the output embedding is of dimension 20. We experimented with different dropout rates and reported the scores using the best dropout rate for each language. 3 It is a greedy model that apply locally optimal actions until the entire sentence is processed, further improvements might be obtained with **beam search** (Zhang and Clark, 2011) or training with exploration (Ballesteros et al., 2016).

## 5.2 Data Sets

We test our model on different datasets for named entity recognition. To demonstrate our model’s ability to generalize to different languages, we present results on the **CoNLL-2002** and **CoNLL-2003** datasets (Tjong Kim Sang, 2002; Tjong Kim Sang and De Meulder, 2003) that contain independent named entity labels for English, Spanish, German and Dutch. All datasets contain four different types of named entities: **locations, persons, organizations, and miscellaneous entities that do not belong in any of the three previous categories**. Although POS tags were made available for all datasets, we did not include them in our models. We did not perform any dataset preprocessing, apart from replacing every digit with a zero in the English NER dataset.  


## 5.3 Results

Table 1 presents our comparisons with other models for named entity recognition in English. To make the comparison between our model and others fair, we report the scores of other models with and without the use of external labeled data such as gazetteers and knowledge bases. Our models do not use gazetteers or any external labeled resources. The best score reported on this task is by Luo et al. (2015). They obtained a F 1 of 91.2 by jointly modeling the NER and **entity linking tasks** (Hoffart et al., 2011). **Their model uses a lot of hand-engineered features** including spelling features, WordNet clusters, Brown clusters, POS tags, chunks tags, as well as stemming and external knowledge bases like Freebase and Wikipedia. Our LSTM-CRF model outperforms all other systems, including the ones using external labeled data like gazetteers. Our Stack-LSTM model also outperforms all previous models that do not incorporate external features, apart from the one presented by Chiu and Nichols (2015).

Tables 2, 3 and 4 present our results on NER for German, Dutch and Spanish respectively in comparison to other models. On these three languages, the LSTM-CRF model significantly out performs all previous methods, including the ones using external labeled data. The only exception is Dutch, where the model of Gillick et al. (2015) can perform better by leveraging the information from other NER datasets. The Stack-LSTM also consistently presents state-the-art (or close to) results compared to systems that do not use external data.
As we can see in the tables, the Stack-LSTM model is more dependent on character-based representations to achieve competitive performance; we hypothesize that the LSTM-CRF model requires less orthographic information since it gets more contextual information out of the bidirectional LSTMs; however, the Stack-LSTM model consumes the words one by one and it just relies on the word representations when it chunks words.

## 5.4 Network architectures

Our models had several components that we could tweak to understand their impact on the overall performance.  

> **[info]** tweak：微调  

We explored the impact that **the CRF, the character-level representations, pretraining of our word embeddings and dropout** had on our LSTM-CRF model. We observed that pretraining our word embeddings gave us the biggest improvement in overall performance of +7.31 in F 1 . The CRF layer gave us an increase of +1.79, while using dropout resulted in a difference of +1.17 and finally learning character-level word embeddings resulted in an increase of about +0.74. For the Stack-LSTM we performed a similar set of experiments. Results with different architectures are given in table 5.

> **[success]**  
dropout > CRF layer > 字符级embedding
