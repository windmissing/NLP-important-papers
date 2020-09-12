# Related Work
In this section we describe previous work relevant to the
approaches discussed in this paper. A more detailed dis-
cussion on language modeling research is provided in
(Mikolov, 2012).
## Language Models
Language Modeling (LM) has been a central task in NLP.
The goal of LM is to learn a probability distribution over
sequences of symbols pertaining to a language. Much work
has been done on both parametric (e.g., log-linear models)
and non-parametric approaches (e.g., count-based LMs).
Count-based approaches (based on statistics of N-grams)
typicallyaddsmoothingwhichaccountforunseen(yetpos-
sible) sequences, and have been quite successful. To this
extent, Kneser-Ney smoothed 5-gram models (Kneser &
Ney, 1995) are a fairly strong baseline which, for large
amounts of training data, have challenged other paramet-
ric approaches based on Neural Networks (Bengio et al.,
2006).
Most of our work is based on Recurrent Neural Networks
(RNN) models which retain long term dependencies. To
this extent, we used the Long-Short Term Memory model
(Hochreiter & Schmidhuber, 1997) which uses a gating
mechanism (Gers et al., 2000) to ensure proper propaga-
tion of information through many time steps. Much work
has been done on small and large scale RNN-based LMs
(Mikolov et al., 2010; Mikolov, 2012; Chelba et al., 2013;
Zaremba et al., 2014; Williams et al., 2015; Ji et al., 2015a;
Wang & Cho, 2015; Ji et al., 2015b). The architectures that
we considered in this paper are represented in Figure 1.
In our work, we train models on the popular One Bil-
lion Word Benchmark, which can be considered to be a
medium-sizeddatasetforcount-basedLMsbutaverylarge
data set for NN-based LMs. This regime is most interesting
to us as we believe learning a very good model of human
languageisacomplextaskwhichwillrequirelargemodels,
and thus large amounts of data. Further advances in data
availability and computational resources helped our study.
We argue this leap in scale enabled tremendous advances in
deep learning. A clear example found in computer vision is
Imagenet (Deng et al., 2009), which enabled learning com-
plex vision models from large amounts of data (Krizhevsky
et al., 2012).
A crucial aspect which we discuss in detail in later sections
is the size of our models. Despite the large number of pa-
rameters, we try to minimize computation as much as pos-
sible by adopting a strategy proposed in (Sak et al., 2014)
of projecting a relatively big recurrent state space down so
that the matrices involved remain relatively small, yet the
model has large memory capacity.
## Convolutional Embedding Models
There is an increased interest in incorporating character-
level inputs to build word embeddings for various NLP
problems, including part-of-speech tagging, parsing and
language modeling (Ling et al., 2015; Kim et al., 2015;
Ballesteros et al., 2015). The additional character informa-
tion has been shown useful on relatively small benchmark
data sets.
The approach proposed in (Ling et al., 2015) builds word
embeddings using bidirectional LSTMs (Schuster & Pali-
wal, 1997; Graves & Schmidhuber, 2005) over the charac-
ters. The recurrent networks process sequences of charac-
tersfrombothsidesandtheirfinalstatevectorsareconcate-
nated. The resulting representation is then fed to a Neural
Network. This model achieved very good results on a part-
of-speech tagging task.
In (Kim et al., 2015), the words characters are processed by
a 1-d CNN (Le Cun et al., 1990) with max-pooling across
the sequence for each convolutional feature. The result-
ing features are fed to a 2-layer highway network (Srivas-
tava et al., 2015b), which allows the embedding to learn se-
mantic representations. The model was evaluated on small-
scalelanguagemodelingexperimentsforvariouslanguages
and matched the best results on the PTB data set despite
having 60% fewer parameters.
## Softmax Over Large Vocabularies
Assigning probability distributions over large vocabularies
is computationally challenging. For modeling language,
maximizing log-likelihood of a given word sequence leads
to optimizing cross-entropy between the target probability
distribution (e.g., the target word we should be predicting),
and our model predictions p. Generally, predictions come
from a linear layer followed by a Softmax non-linearity:
p(w) =
exp(z w )
P
w 0 ∈V
exp(z w 0 )
where z w is the logit correspond-
ing to a word w. The logit is generally computed as an
inner product z w = h T e w where h is a context vector and
e w is a “word embedding” for w.
The main challenge when |V | is very large (in the order
of one million in this paper) is the fact that computing
all inner products between h and all embeddings becomes
prohibitively slow during training (even when exploiting
matrix-matrix multiplications and modern GPUs). Several
approaches have been proposed to cope with the scaling is-
sue: importance sampling (Bengio et al., 2003; Bengio &
Senécal, 2008), Noise Contrastive Estimation (NCE) (Gut-
mann & Hyvärinen, 2010; Mnih & Kavukcuoglu, 2013),
self normalizing partition functions (Vincent et al., 2015)
or Hierarchical Softmax (Morin & Bengio, 2005; Mnih &
Hinton, 2009) – they all offer good solutions to this prob-
lem. We found importance sampling to be quite effective
on this task, and explain the connection between it and
NCE in the following section, as they are closely related.
