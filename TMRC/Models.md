# Models

So far we have motivated the need for better datasets and tasks to evaluate the capabilities of machine reading models.  

> **[success]**  
motivated：激发  
目标：better datasets and tasks  

We proceed by describing a number of baselines, benchmarks and new models to evaluate against this paradigm. We define two simple baselines, the majority baseline (**maximum frequency**) picks the entity most frequently observed in the context document, whereas the exclusive majority (**exclusive frequency**) chooses the entity most frequently observed in the context but not observed in the query. The idea behind this exclusion is that the placeholder is unlikely to be mentioned twice in a single Cloze form query.  

> **[success]**  
baseline 1：maximum frequency，选择文本中频率最高的entity  
baseline 2：exclusive frequency，选择文本中频率最高且没有在query中出现的entity  

## Symbolic Matching Models

Traditionally, a pipeline of NLP models has been used for attempting question answering, that is models that make heavy use of **linguistic annotation, structured world knowledge and semantic parsing** and similar NLP pipeline outputs.  

> **[success]**  
annotation：注释  
传统方法所依据的东西：linguistic annotation, structured world knowledge and semantic parsing  

Building on these approaches, we define a number of NLP-centric models for our machine reading task.  

**Frame-Semantic Parsing** Frame-semantic parsing attempts to identify predicates and their arguments, allowing models access to information about “who did what to whom”.   

> **[success]**  
predicate：述语，谓语   
argument：论点  
框架语言分析的目标：提取出“who did what to whom”  

Naturally this kind of annotation lends itself to being exploited for question answering. We develop a benchmark that makes use of frame-semantic annotations which we obtained by parsing our model with a state-of-the-art frame-semantic parser [13, 14].   

> **[success]** 使用state-of-the-art frame-semantic parser作为benchmark  

As the parser makes extensive use of linguistic information we run these benchmarks **on the unanonymised version** of our corpora. There is no significant advantage in this as the frame-semantic approach used here does not possess the capability to generalise through a language model beyond exploiting one during the parsing phase. Thus, the key objective of evaluating machine comprehension abilities is maintained.   

> **[success]**  
这个benchmark要求使用原始数据，而不是2.1中提供的anonymised数据。  
但是由于benchmark不能“generalise through a language model”，让benchmark使用原始数据不影响比较的公平性。  

Extracting entity-predicate triples—denoted as (e 1 ,V,e 2 )—from both the query q and context document d, we attempt to resolve queries using a number of rules with an increasing recall/precision trade-off as follows (Table 4).  

> **[success]**  
benchmark具体做的事情：  
（1）从问题中提取三元组(p, V, y)  
（2）从文本中提取三元组(x, V, y)  
（3）x是p的候选答案，基于策略选择合适的x  
（4）recall/precision trade-off

For reasons of clarity, we pretend that all PropBank triples are of the form (e 1 ,V,e 2 ). In practice, we take the argument numberings of the parser into account and only compare like with like, except in cases such as the permuted frame rule, where ordering is relaxed. In the case of multiple possible answers from a single rule, we randomly choose one.

> **[warning]** 这一段没看懂  

**Word Distance Benchmark** We consider another baseline that relies on word distance measurements. Here, we align the placeholder of the Cloze form question with each possible entity in the context document and calculate a distance measure between the question and the context around the aligned entity. This score is calculated by summing the distances of every word in q to their nearest aligned word in d, where alignment is defined by matching words either directly or as aligned by the coreference system. We tune the maximum penalty per word (m = 8) on the validation data.

> **[success]**  
for each entity：  
（1）把entity代入cloze中  
（2）计算文本中entity的附近单词和问题中entity的附近单词的距离  
[?]distance怎么算没看懂  

## Neural Network Models

Neural networks have successfully been applied to a range of tasks in NLP. This includes classification tasks such as sentiment analysis [15] or POS tagging [16], as well as generative problems such as language modelling or machine translation [17].   

> **[info]** sentiment：情感  

We propose three neural models for estimating the probability of word type a from document d answering query q:  

$$
p(a|d,q) \propto \exp(W(a)g(d,q)), s.t. a \in V,
$$

where V is the vocabulary 4 , and W(a) indexes row a of weight matrix W and through a slight abuse of notation word types double as indexes. Note that we do not privilege entities or variables, the model must learn to differentiate these in the input sequence. The function g(d,q) returns a vector embedding of a document and query pair.  

> **[success]**  
这个公式不是太懂，可能意思是：  
g是vector embedding，代表文本和问题之间的联系点  
W是权重矩阵，代表某个单词在这些联系点上的权重。  

**The Deep LSTM Reader** Long short-term memory (LSTM, [18]) networks have recently seen considerable success in tasks such as machine translation and language modelling [17]. When used for translation, Deep LSTMs [19] have shown a remarkable ability to embed long sequences into a vector representation which contains enough information to generate a full translation in another language.   

> **[success] Deep LSTM的特点：**  
（1）embed long sequences into a vector  
（2）包含原序列的信息  
（3）用另一种形式表达信息  

Our first neural model for reading comprehension tests the ability of Deep LSTM encoders to handle significantly longer sequences. We feed our documents one word at a time into a Deep LSTM encoder, after a delimiter we then also feed the query into the encoder. Alternatively we also experiment with processing the query then the document.  

> **[success]**  
把Deep LSTM当作一个encoder来用。  
输入可以是“文本+问题”或“问题+文本”。  

The result is that this model processes each document query pair as a single long sequence. Given the embedded document and query the network predicts which token in the document answers the query. 
We employ a Deep LSTM cell with skip connections from each input x(t) to every hidden layer,
and from every hidden layer to the output y(t):
x 0 (t,k) = x(t)||y 0 (t,k ? 1), y(t) = y 0 (t,1)||...||y 0 (t,K)
i(t,k) = ? (W kxi x 0 (t,k) + W khi h(t ? 1,k) + W kci c(t ? 1,k) + b ki )
f(t,k) = ? (W kxf x(t) + W khf h(t ? 1,k) + W kcf c(t ? 1,k) + b kf )
c(t,k) = f(t,k)c(t ? 1,k) + i(t,k)tanh(W kxc x 0 (t,k) + W khc h(t ? 1,k) + b kc )
o(t,k) = ? (W kxo x 0 (t,k) + W kho h(t ? 1,k) + W kco c(t,k) + b ko )
h(t,k) = o(t,k)tanh(c(t,k))
y 0 (t,k) = W ky h(t,k) + b ky
where || indicates vector concatenation h(t,k) is the hidden state for layer k at time t, and i, f,
o are the input, forget, and output gates respectively. Thus our Deep LSTM Reader is defined by
g LSTM (d,q) = y(|d|+|q|) with input x(t) the concatenation of d and q separated by the delimiter |||.
The Attentive Reader The Deep LSTM Reader must propagate dependencies over long distances
in order to connect queries to their answers. The fixed width hidden vector forms a bottleneck for
this information flow that we propose to circumvent using an attention mechanism inspired by recent
results in translation and image recognition [6, 7]. This attention model first encodes the document
and the query using separate bidirectional single layer LSTMs [19].
We denote the outputs of the forward and backward LSTMs as
? !
y (t) and
?
y (t) respectively. The
encoding u of a query of length |q| is formed by the concatenation of the final forward and backward
outputs, u =
? !
y q (|q|) ||
?
y q (1).
For the document the composite output for each token at position t is, y d (t) =
? !
y d (t) ||
?
y d (t). The
representation r of the document d is formed by a weighted sum of these output vectors. These
weights are interpreted as the degree to which the network attends to a particular token in the docu-
ment when answering the query:
m(t) = tanh(W ym y d (t) + W um u),
s(t) / exp(w |
ms m(t)),
r = y d s,
where we are interpreting y d as a matrix with each column being the composite representation y d (t)
of document token t. The variable s(t) is the normalised attention at token t. Given this attention
5
score the embedding of the document r is computed as the weighted sum of the token embeddings.
The model is completed with the definition of the joint document and query embedding via a non-
linear combination:
g AR (d,q) = tanh(W rg r + W ug u).
The Attentive Reader can be viewed as a generalisation of the application of Memory Networks to
question answering [3]. That model employs an attention mechanism at the sentence level where
each sentence is represented by a bag of embeddings. The Attentive Reader employs a finer grained
token level attention mechanism where the tokens are embedded given their entire future and past
context in the input document.
The Impatient Reader The Attentive Reader is able to focus on the passages of a context doc-
ument that are most likely to inform the answer to the query. We can go further by equipping the
model with the ability to reread from the document as each query token is read. At each token i
of the query q the model computes a document representation vector r(i) using the bidirectional
embedding y q (i) =
? !
y q (i) ||
?
y q (i):
m(i,t) = tanh(W dm y d (t) + W rm r(i ? 1) + W qm y q (i)), 1  i  |q|,
s(i,t) / exp(w |
ms m(i,t)),
r(0) = r 0 , r(i) = y |
d s(i) + tanh(W rr r(i ? 1))
1  i  |q|.
The result is an attention mechanism that allows the model to recurrently accumulate information
from the document as it sees each query token, ultimately outputting a final joint document query
representation for the answer prediction,
g IR (d,q) = tanh(W rg r(|q|) + W qg u).
4 Empirical Evaluation
Having described a number of models in the previous section, we next evaluate these models on our
reading comprehension corpora. Our hypothesis is that neural models should in principle be well
suited for this task. However, we argued that simple recurrent models such as the LSTM probably
have insufficient expressive power for solving tasks that require complex inference. We expect that
the attention-based models would therefore outperform the pure LSTM-based approaches.
Considering the second dimension of our investigation, the comparison of traditional versus neural
approaches to NLP, we do not have a strong prior favouring one approach over the other. While nu-
merous publications in the past few years have demonstrated neural models outperforming classical
methods, it remains unclear how much of that is a side-effect of the language modelling capabilities
intrinsic to any neural model for NLP. The entity anonymisation and permutation aspect of the task
presented here may end up levelling the playing field in that regard, favouring models capable of
dealing with syntax rather than just semantics.
With these considerations in mind, the experimental part of this paper is designed with a three-
fold aim. First, we want to establish the difficulty of our machine reading task by applying a wide
range of models to it. Second, we compare the performance of parse-based methods versus that of
neural models. Third, within the group of neural models examined, we want to determine what each
component contributes to the end performance; that is, we want to analyse the extent to which an
LSTM can solve this task, and to what extent various attention mechanisms impact performance.
All model hyperparameters were tuned on the respective validation sets of the two corpora. 5 Our
experimental results are in Table 5, with the Attentive and Impatient Readers performing best across
both datasets.
5 For the Deep LSTM Reader, we consider hidden layer sizes [64,128,256], depths [1,2,4], initial learning
rates [1 E ?3,5 E ?4,1 E ?4,5 E ?5], batch sizes [16,32] and dropout [0.0,0.1,0.2]. We evaluate two types of
feeds. In the cqa setup we feed first the context document and subsequently the question into the encoder,
while the qca model starts by feeding in the question followed by the context document. We report results on
the best model (underlined hyperparameters, qca setup). For the attention models we consider hidden layer
sizes [64,128,256], single layer, initial learning rates [1 E ?4,5 E ?5,2.5 E ?5,1 E ?5], batch sizes [8,16,32]
and dropout [0,0.1,0.2,0.5]. For all models we used asynchronous RmsProp [20] with a momentum of 0.9
and a decay of 0.95. See Appendix A for more details of the experimental setup.
6
CNN Daily Mail
valid test valid test
Maximum frequency 30.5 33.2 25.6 25.5
Exclusive frequency 36.6 39.3 32.7 32.8
Frame-semantic model 36.3 40.2 35.5 35.5
Word distance model 50.5 50.9 56.4 55.5
Deep LSTM Reader 55.0 57.0 63.3 62.2
Uniform Reader 39.0 39.4 34.6 34.4
Attentive Reader 61.6 63.0 70.5 69.0
Impatient Reader 61.8 63.8 69.0 68.0
Table 5: Accuracy of all the models and bench-
marks on the CNN and Daily Mail datasets. The
Uniform Reader baseline sets all of the m(t) pa-
rameters to be equal.
Figure 2: Precision@Recall for the attention
models on the CNN validation data.
Frame-semantic benchmark While the one frame-semantic model proposed in this paper is
clearly a simplification of what could be achieved with annotations from an NLP pipeline, it does
highlight the difficulty of the task when approached from a symbolic NLP perspective.
Two issues stand out when analysing the results in detail. First, the frame-semantic pipeline has a
poor degree of coverage with many relations not being picked up by our PropBank parser as they
do not adhere to the default predicate-argument structure. This effect is exacerbated by the type
of language used in the highlights that form the basis of our datasets. The second issue is that
the frame-semantic approach does not trivially scale to situations where several sentences, and thus
frames, are required to answer a query. This was true for the majority of queries in the dataset.
Word distance benchmark More surprising perhaps is the relatively strong performance of the
word distance benchmark, particularly relative to the frame-semantic benchmark, which we had
expected to perform better. Here, again, the nature of the datasets used can explain aspects of this
result. Wheretheframe-semanticmodelsufferedduetothelanguageusedinthehighlights, theword
distance model benefited. Particularly in the case of the Daily Mail dataset, highlights frequently
have significant lexical overlap with passages in the accompanying article, which makes it easy for
the word distance benchmark. For instance the query “Tom Hanks is friends with X’s manager,
Scooter Brown” has the phrase “... turns out he is good friends with Scooter Brown, manager for
Carly Rae Jepson” in the context. The word distance benchmark correctly aligns these two while
the frame-semantic approach fails to pickup the friendship or management relations when parsing
the query. We expect that on other types of machine reading data where questions rather than Cloze
queries are used this particular model would perform significantly worse.
Neural models Within the group of neural models explored here, the results paint a clear picture
with the Impatient and the Attentive Readers outperforming all other models. This is consistent with
our hypothesis that attention is a key ingredient for machine reading and question answering due to
the need to propagate information over long distances. The Deep LSTM Reader performs surpris-
ingly well, once again demonstrating that this simple sequential architecture can do a reasonable
job of learning to abstract long sequences, even when they are up to two thousand tokens in length.
However this model does fail to match the performance of the attention based models, even though
these only use single layer LSTMs. 6
The poor results of the Uniform Reader support our hypothesis of the significance of the attention
mechanism in the Attentive model’s performance as the only difference between these models is
that the attention variables are ignored in the Uniform Reader. The precision@recall statistics in
Figure 2 again highlight the strength of the attentive approach.
We can visualise the attention mechanism as a heatmap over a context document to gain further
insight into the models’ performance. The highlighted words show which tokens in the document
were attended to by the model. In addition we must also take into account that the vectors at each
6 Memory constraints prevented us from experimenting with deeper Attentive Readers.
7
. . . . . .
Figure 3: Attention heat maps from the Attentive Reader for two correctly answered validation set
queries (the correct answers are ent23 and ent63, respectively). Both examples require significant
lexicalgeneralisationandco-referenceresolutioninordertobeansweredcorrectlybyagivenmodel.
token integrate long range contextual information via the bidirectional LSTM encoders. Figure 3
depicts heat maps for two queries that were correctly answered by the Attentive Reader. 7 In both
cases confidently arriving at the correct answer requires the model to perform both significant lexical
generalsiation, e.g. ‘killed’ ! ‘deceased’, and co-reference or anaphora resolution, e.g. ‘ent119 was
killed’ ! ‘he was identified.’ However it is also clear that the model is able to integrate these signals
with rough heuristic indicators such as the proximity of query words to the candidate answer.
5 Conclusion
The supervised paradigm for training machine reading and comprehension models provides a
promising avenue for making progress on the path to building full natural language understanding
systems. We have demonstrated a methodology for obtaining a large number of document-query-
answer triples and shown that recurrent and attention based neural networks provide an effective
modelling framework for this task. Our analysis indicates that the Attentive and Impatient Read-
ers are able to propagate and integrate semantic information over long distances. In particular we
believe that the incorporation of an attention mechanism is the key contributor to these results.
The attention mechanism that we have employed is just one instantiation of a very general idea
which can be further exploited. However, the incorporation of world knowledge and multi-document
queries will also require the development of attention and embedding mechanisms whose complex-
ity to query does not scale linearly with the data set size. There are still many queries requiring
complex inference and long range reference resolution that our models are not yet able to answer.
As such our data provides a scalable challenge that should support NLP research into the future. Fur-
ther, significantly bigger training data sets can be acquired using the techniques we have described,
undoubtedly allowing us to train more expressive and accurate models.
7 Note that these examples were chosen as they were short, the average CNN validation document contained
763 tokens and 27 entities, thus most instances were significantly harder to answer than these examples.
8
References
[1] Ellen Riloff and Michael Thelen. A rule-based question answering system for reading com-
prehension tests. In Proceedings of the ANLP/NAACL Workshop on Reading Comprehension
Tests As Evaluation for Computer-based Language Understanding Sytems.
[2] Hoifung Poon, Janara Christensen, Pedro Domingos, Oren Etzioni, Raphael Hoffmann, Chloe
Kiddon, Thomas Lin, Xiao Ling, Mausam, Alan Ritter, Stefan Schoenmackers, Stephen Soder-
land, Dan Weld, Fei Wu, and Congle Zhang. Machine reading at the University of Washing-
ton. In Proceedings of the NAACL HLT 2010 First International Workshop on Formalisms and
Methodology for Learning by Reading.
[3] Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. CoRR, abs/1410.3916,
2014.
[4] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory
networks. CoRR, abs/1503.08895, 2015.
[5] Terry Winograd. Understanding Natural Language. Academic Press, Inc., Orlando, FL, USA,
1972.
[6] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by
jointly learning to align and translate. CoRR, abs/1409.0473, 2014.
[7] Volodymyr Mnih, Nicolas Heess, Alex Graves, and Koray Kavukcuoglu. Recurrent models of
visual attention. In Advances in Neural Information Processing Systems 27.
[8] Karol Gregor, Ivo Danihelka, Alex Graves, and Daan Wierstra. DRAW: A recurrent neural
network for image generation. CoRR, abs/1502.04623, 2015.
[9] Matthew Richardson, Christopher J. C. Burges, and Erin Renshaw. Mctest: A challenge dataset
for the open-domain machine comprehension of text. In Proceedings of EMNLP.
[10] Krysta Svore, Lucy Vanderwende, and Christopher Burges. Enhancing single-document sum-
marization by combining RankNet and third-party sources. In Proceedings of EMNLP/CoNLL.
[11] Kristian Woodsend and Mirella Lapata. Automatic generation of story highlights. In Proceed-
ings of ACL, 2010.
[12] Wilson L Taylor. “Cloze procedure”: a new tool for measuring readability. Journalism Quar-
terly, 30:415–433, 1953.
[13] Dipanjan Das, Desai Chen, André F. T. Martins, Nathan Schneider, and Noah A. Smith. Frame-
semantic parsing. Computational Linguistics, 40(1):9–56, 2013.
[14] Karl Moritz Hermann, Dipanjan Das, Jason Weston, and Kuzman Ganchev. Semantic frame
identification with distributed word representations. In Proceedings of ACL, June 2014.
[15] Nal Kalchbrenner, Edward Grefenstette, and Phil Blunsom. A convolutional neural network
for modelling sentences. In Proceedings of ACL, 2014.
[16] Ronan Collobert, Jason Weston, Léon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel
Kuksa. Natural language processing (almost) from scratch. Journal of Machine Learning
Research, 12:2493–2537, November 2011.
[17] Ilya Sutskever, Oriol Vinyals, and Quoc V. V Le. Sequence to sequence learning with neural
networks. In Advances in Neural Information Processing Systems 27.
[18] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Computation,
9(8):1735–1780, November 1997.
[19] Alex Graves. Supervised Sequence Labelling with Recurrent Neural Networks, volume 385 of
Studies in Computational Intelligence. Springer, 2012.
[20] T. Tieleman and G. Hinton. Lecture 6.5—RmsProp: Divide the gradient by a running average
of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 2012.