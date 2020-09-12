# Language Modeling Improvements
Recurrent Neural Networks based LMs employ the chain
rule to model joint probabilities over word sequences:
p(w 1 ,...,w N ) =
N
Y
i=1
p(w i |w 1 ,...,w i−1 )
where the context of all previous words is encoded with an
LSTM, and the probability over words uses a Softmax (see
Figure 1(a)).
## Relationship between Noise Contrastive
Estimation and Importance Sampling
As discussed in Section 2.3, a large scale Softmax is neces-
sary for training good LMs because of the vocabulary size.
A Hierarchical Softmax (Mnih & Hinton, 2009) employs
a tree in which the probability distribution over words is
decomposed into a product of two probabilities for each
word, greatly reducing training and inference time as only
the path specified by the hierarchy needs to be computed
and updated. Choosing a good hierarchy is important for
obtaining good results and we did not explore this approach
further for this paper as sampling methods worked well for
our setup.
Sampling approaches are only useful during training, as
they propose an approximation to the loss which is cheap to
compute (also in a distributed setting) – however, at infer-
ence time one still has to compute the normalization term
over all words. Noise Contrastive Estimation (NCE) pro-
poses to consider a surrogate binary classification task in
which a classifier is trained to discriminate between true
data, or samples coming from some arbitrary distribution.
If both the noise and data distributions were known, the
optimal classifier would be:
p(Y = true|w) =
p d (w)
p d (w) + kp n (w)
where Y is the binary random variable indicating whether
w comes from the true data distribution, k is the number of
negative samples per positive word, and p d and p n are the
data and noise distribution respectively (we dropped any
dependency on previous words for notational simplicity).
It is easy to show that if we train a logistic classifier
p θ (Y = true|w) = σ(s θ (w,h) − logkp n (w)) where σ
is the logistic function, then, p 0 (w) = softmax(s θ (w,h))
is a good approximation of p d (w) (s θ is a logit which e.g.
an LSTM LM computes).
The other technique, which is based on importance sam-
pling (IS), proposes to directly approximate the partition
function (which comprises a sum over all words) with an
estimate of it through importance sampling. Though the
methods look superficially similar, we will derive a similar
surrogate classification task akin to NCE which arrives at
IS, showing a strong connection between the two.
Suppose that, instead of having a binary task to decide if
a word comes from the data or from the noise distribution,
we want to identify the words coming from the true data
distribution in a set W = {w 1 ,...,w k+1 }, comprised of
k noise samples and one data distribution sample. Thus,
we can train a multiclass loss over a multinomial random
variable Y which maximizes logp(Y = 1|W), assuming
w.l.o.g. that w 1 ∈ W is always the word coming from true
data. By Bayes rule, and ignoring terms that are constant
with respect to Y , we can write:
p(Y = k|W) ∝ Y
p d (w k )
p n (w k )
and, following a similar argument than for NCE, if we de-
fine p(Y = k|W) = softmax(s θ (w k )−logp n (w k )) then
p 0 (w) = softmax(s θ (w,h)) is a good approximation of
p d (word). Note that the only difference between NCE and
IS is that, in NCE, we define a binary classification task
between true or noise words with a logistic loss, whereas
in IS we define a multiclass classification problem with a
Softmax and cross entropy loss. We hope that our deriva-
tion helps clarify the similarities and differences between
the two. In particular, we observe that IS, as it optimizes
a multiclass classification task (in contrast to solving a bi-
nary task), may be a better choice. Indeed, the updates to
the logits with IS are tied whereas in NCE they are inde-
pendent.
## CNN Softmax
The character-level features allow for a smoother and com-
pact parametrization of the word embeddings. Recent ef-
forts on small scale language modeling have used CNN
character embeddings for the input embeddings (Kim et al.,
2015). Although not as straightforward, we propose an ex-
tension to this idea to also reduce the number of param-
eters of the Softmax layer. Recall from Section 2.3 that
the Softmax computes a logit as z w = h T e w where h is
a context vector and e w the word embedding. Instead of
building a matrix of |V | × |h| (whose rows correspond to
e w ), we produce e w with a CNN over the characters of w as
e w = CNN(chars w ) – we call this a CNN Softmax. We
used the same network architecture to dynamically gener-
ate the Softmax word embeddings without sharing the pa-
rameters with the input word-embedding sub-network. For
inference, thevectorse w canbeprecomputed, sothereisno
computational complexity increase w.r.t. the regular Soft-
max.
We note that, when using an importance sampling loss such
as the one described in Section 3.1, only a few logits have
non-zerogradient(thosecorrespondingtothetrueandsam-
pled words). With a Softmax where e w are independently
learned word embeddings, this is not a problem. But we
observed that, when using a CNN, all the logits become
tied as the function mapping from w to e w is quite smooth.
As a result, a much smaller learning rate had to be used.
Even with this, the model lacks capacity to differentiate
between words that have very different meanings but that
are spelled similarly. Thus, a reasonable compromise was
to add a small correction factor which is learned per word,
such that:
z w = h T CNN(chars w ) + h T Mcorr w
where M is a matrix projecting a low-dimensional embed-
ding vector corr w back up to the dimensionality of the pro-
jected LSTM hidden state of h. This amounts to adding a
bottleneck linear layer, and brings the CNN Softmax much
closer to our best result, as can be seen in Table 1, where
adding a 128-dim correction halves the gap between regu-
lar and the CNN Softmax.
Aside from a big reduction in the number of parameters
and incorporating morphological knowledge from words,
the other benefit of this approach is that out-of-vocabulary
(OOV) words can easily be scored. This may be useful for
other problems such as Machine Translation where han-
dling out-of-vocabulary words is very important (Luong
et al., 2014). This approach also allows parallel training
over various data sets since the model is no longer explic-
itly parametrized by the vocabulary size – or the language.
This has shown to help when using byte-level input embed-
dings for named entity recognition (Gillick et al., 2015),
and we hope it will enable similar gains when used to map
onto words.
## Char LSTM Predictions
The CNN Softmax layer can handle arbitrary words and is
much more efficient in terms of number of parameters than
the full Softmax matrix. It is, though, still considerably
slow, as to evaluate perplexities we need to compute the
partition function. A class of models that solve this prob-
lem more efficiently are character-level LSTMs (Sutskever
et al., 2011; Graves, 2013). They make predictions one
character at a time, thus allowing to compute probabili-
ties over a much smaller vocabulary. On the other hand,
these models are more difficult to train and seem to per-
form worse even in small tasks like PTB (Graves, 2013).
Most likely this is due to the sequences becoming much
longer on average as the LSTM reads the input character
by character instead of word by word.
Thus, we combine the word and character-level models by
feeding a word-level LSTM hidden state h into a small
LSTM that predicts the target word one character at a time
(see Figure 1(c)). In order to make the whole process rea-
sonably efficient, we train the standard LSTM model un-
til convergence, freeze its weights, and replace the stan-
dard word-level Softmax layer with the aforementioned
character-level LSTM.
The resulting model scales independently of vocabulary
size – both for training and inference. However, it does
seem to be worse than regular and CNN Softmax – we are
hopeful that further research will enable these models to
replace fixed vocabulary models whilst being computation-
ally attractive.
