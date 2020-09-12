# Results and Analysis
In this section we summarize the results of our experiments
and do an in-depth analysis. Table 1 contains all results for
our models compared to previously published work. Ta-
ble 2 shows previous and our own work on ensembles of
models. We hope that our encouraging results, which im-
proved the best perplexity of a single model from 51.3 to
30.0 (whilst reducing the model size considerably), and set
a new record with ensembles at 23.7, will enable rapid re-
search and progress to advance Language Modeling. For
this purpose, we will release the model weights and recipes
upon publication.
## Size Matters
Unsurprisingly, size matters: when training on a very large
and complex data set, fitting the training data with an
LSTM is fairly challenging. Thus, the size of the LSTM
layer is a very important factor that influences the results,
as seen in Table 1. The best models are the largest we were
able to fit into a GPU memory. Our largest model was a 2-
layer LSTM with 8192+1024 dimensional recurrent state
in each of the layers. Increasing the embedding and projec-
tion size also helps but causes a large increase in the num-
ber of parameters, which is less desirable. Lastly, training
an RNN instead of an LSTM yields poorer results (about 5
perplexity worse) for a comparable model size.
## Regularization Importance
As shown in Table 1, using dropout improves the results.
To our surprise, even relatively small models (e.g., single
layer LSTM with 2048 units projected to 512 dimensional
outputs) can over-fit the training set if trained long enough,
eventually yielding holdout set degradation.
Using dropout on non-recurrent connections largely miti-
gates these issues. While over-fitting still occurs, there is
no more need for early stopping. For models that had 4096
or less units in the LSTM layer, we used 10% dropout prob-
ability. For larger models, 25% was significantly better.
Even with such regularization, perplexities on the training
set can be as much as 6 points below test.
In one experiment we tried to use a smaller vocabulary
comprising of the 100,000 most frequent words and found
the difference between train and test to be smaller – which
suggests that too much capacity is given to rare words. This
is less of an issue with character CNN embedding models
as the embeddings are shared across all words.
## Importance Sampling is Data Efficient
Table 3 shows the test perplexities of NCE vs IS loss after a
few epochs of 2048 unit LSTM with 512 projection. The IS
objective significantly improves the speed and the overall
performance of the model when compared to NCE.
## Word Embeddings vs Character CNN
Replacing the embedding layer with a parametrized neural
network that process characters of a given word allows the
model to consume arbitrary words and is not restricted to
a fixed vocabulary. This property is useful for data sets
with conversational or informal text as well as for mor-
phologically rich languages. Our experiments show that
using character-level embeddings is feasible and does not
degrade performance – in fact, our best single model uses
a Character CNN embedding.
Anadditionaladvantageisthatthenumberofparametersof
the input layer is reduced by a factor of 11 (though training
speed is slightly worse). For inference, the embeddings
can be precomputed so there is no speed penalty. Overall,
the embedding of the best model is parametrized by 72M
weights (down from 820M weights).
Table 4 shows a few examples of nearest neighbor embed-
dings for some out-of-vocabulary words when character
CNNs are used.
## Smaller Models with CNN Softmax
Even with character-level embeddings, the model is still
fairly large (though much smaller than the best competing
models from previous work). Most of the parameters are in
the linear layer before the Softmax: 820M versus a total of
1.04B parameters.
In one of the experiments we froze the word-LSTM after
convergence and replaced the Softmax layer with the CNN
Softmax sub-network. Without any fine-tuning that model
was able to reach 39.8 perplexity with only 293M weights
(as seen in Table 1).
As described in Section 3.2, adding a “correction” word
embedding term alleviates the gap between regular and
CNN Softmax. Indeed, we can trade-off model size versus
perplexity. For instance, by adding 100M weights (through
a 128 dimensional bottleneck embedding) we achieve 35.8
perplexity (see Table 1).
To contrast with the CNN Softmax, we also evaluated a
modelthatreplacestheSoftmaxlayerwithasmallerLSTM
that predicts one character at a time (see Section 3.3). Such
a model does not have to learn long dependencies because
the base LSTM still operates at the word-level (see Fig-
ure 1(c)). With a single-layer LSTM of 1024 units we
reached 49.0 test perplexity, far below the best model. In
order to make the comparisons more fair, we performed a
very expensive marginalization over the words in the vo-
cabulary (to rule out words not in the dictionary which the
character LSTM would assign some probability). When
doing this marginalization, the perplexity improved a bit
down to 47.9.
Words buckets of equal size (less frequent words on the right)
0.0
0.5
1.0
1.5
2.0
2.5
Mean difference in log perplexity
Figure 2. The difference in log probabilities between the best
LSTM and KN-5 (higher is better). The words from the hold-
out set are grouped into 25 buckets of equal size based on their
frequencies.
## Training Speed
We used 32 Tesla K40 GPUs to train our models. The
smaller version of the LSTM model with 2048 units and
512 projections needs less than 10 hours to reach below
45 perplexity and after only 2 hours of training the model
beats previous state-of-the art on this data set. The best
model needs about 5 days to get to 35 perplexity and 10
days to 32.5. The best results were achieved after 3 weeks
of training. See Table 3 for more details.
## Ensembles
We averaged several of our best models and we were able
to reach 23.7 test perplexity (more details and results can
be seen in Table 2), which is more than 40% improve-
ment over previous work. Interestingly, including the best
N-gram model reduces the perplexity by 1.2 point even
though the model is rather weak on its own (67.6 perplex-
ity). Most previous work had to either ensemble with the
best N-gram model (as their RNN only used a limited out-
put vocabulary of a few thousand words), or use N-gram
features as additional input to the RNN. Our results, on
the contrary, suggest that N-grams are of limited benefit,
and suggest that a carefully trained LSTM LM is the most
competitive model.
## LSTMs are best on the tail words
Figure 2 shows the difference in log probabilities between
ourbestmodel(at30.0perplexity)andtheKN-5. Ascanbe
seenfromtheplot, theLSTMisbetteracrossallthebuckets
and significantly outperforms KN-5 on the rare words. This
is encouraging as it seems to suggest that LSTM LMs may
fareevenbetterforlanguagesordatasetswherethenumber
of rare words is larger than traditional N-gram models.
## Samples from the model
To qualitatively evaluate the model, we sampled many sen-
tences. We discarded short and politically incorrect ones,
but the sample shown below is otherwise “raw” (i.e., not
hand picked). The samples are of high quality – which is
not a surprise, given the perplexities attained – but there are
still some occasional mistakes.
Sentences generatedby theensemble (about26 perplexity):
< S > With even more new technologies coming onto the market
quickly during the past three years , an increasing number of compa-
nies now must tackle the ever-changing and ever-changing environ-
mental challenges online . < S > Check back for updates on this
breaking news story . < S > About 800 people gathered at Hever
Castle on Long Beach from noon to 2pm , three to four times that of
the funeral cortège . < S > We are aware of written instructions
from the copyright holder not to , in any way , mention Rosenberg ’s
negative comments if they are relevant as indicated in the documents
, ” eBay said in a statement . < S > It is now known that coffee and
cacao products can do no harm on the body . < S > Yuri Zhirkov
was in attendance at the Stamford Bridge at the start of the second
half but neither Drogba nor Malouda was able to push on through the
Barcelona defence .
