# Experiments

All experiments were run using the TensorFlow system (Abadi et al., 2015), with the exception of some older models which were used in the ensemble.  

## Data Set

The experiments are performed on the 1B Word Benchmark data set introduced by (Chelba et al., 2013), which is a publicly available benchmark for measuring progress of statistical language modeling. The data set contains about 0.8B words with a vocabulary of 793471 words, including sentence boundary markers.   

> **[success]** 数据集介绍  

All the sentences are shuffled and the duplicates are removed. The words that are out of vocabulary (OOV) are marked with a special UNK token (there are approximately 0.3% such words).

## Model Setup

The typical measure used for reporting progress in language modeling is **perplexity**, which is the average per-word log-probability on the holdout data set:$e^{−\frac{1}{N}\sum_i\ln p_{w_i}}$. We follow the standard procedure and sum over all the words (including the end of sentence symbol).  

> **[success]**  
衡量指标：perplexity  

We used the 1B Word Benchmark data set without any preprocessing. Given the shuffled sentences, they are input to the network as a batch of independent streams of words. Whenever a sentence ends, a new one starts without any padding (thus maximizing the occupancy per batch).

For the models that consume characters as inputs or as targets, each word is fed to the model as a sequence of character IDs of prespecified length (see Figure 1(b)).   

> **[info]**   
consume：消费   
prespecified：预定义的  

The words were processed to include special begin and end of word tokens and were padded to reach the expected length. I.e. if the maximum word length was 10, the word “cat” would be transformed to “$catˆ ” due to the CNN model.

In our experiments we found that limiting the maximum word length in training to 50 was sufficient to reach very good results while 32 was clearly insufficient. We used 256 characters in our vocabulary and the non-ascii symbols were represented as a sequence of bytes.

## Model Architecture

We evaluated many variations of RNN LM architectures. These include the dimensionalities of the embedding layers, the state, projection sizes, and number of LSTM layers to use. Exhaustively trying all combinations would be extremely time consuming for such a large data set, but our findings suggest that LSTMs with a projection layer (i.e., a bottleneck between hidden states as in (Sak et al., 2014)) trained with truncated BPTT (Williams & Peng, 1990) for 20 steps performed well.  

> **[warning]** [?] projection layer  

Following (Zaremba et al., 2014) we use dropout (Srivastava, 2013) before and after every LSTM layer. The biases of LSTM forget gate were initialized to 1.0 (Jozefowicz et al., 2015). The size of the models will be described in more detail in the following sections, and the choices of hyper-parameters will be released as open source upon publication.

For any model using character embedding CNNs, we closely follow the architecture from (Kim et al., 2015). The only important difference is that we use a larger number of convolutional features of 4096 to give enough capacity to the model. The resulting embedding is then linearly transformed to match the LSTM projection sizes. This allows it to match the performance of regular word embeddings but only uses a small fraction of parameters.

## Training Procedure

The models were trained until convergence with an AdaGrad optimizer using a learning rate of 0.2. In all the experiments the RNNs were unrolled for 20 steps without ever resetting the LSTM states. We used a batch size of 128. We clip the gradients of the LSTM weights such that their norm is bounded by 1.0 (Pascanu et al., 2012).

Using these hyper-parameters we found large LSTMs to be relatively easy to train. The same learning rate was used in almost all of the experiments. In a few cases we had to reduce it by an order of magnitude. Unless otherwise stated, the experiments were performed with 32 GPU workers and asynchronous gradient updates. Further details will be fully specified with the code upon publication.

Training a model for such large target vocabulary (793471 words) required to be careful with some details about the approximation to full Softmax using importance sampling.

We used a large number of negative (or noise) samples: 8192 such samples were drawn per step, but were shared across all the target words in the batch (2560 total, i.e. 128 times 20 unrolled steps). This results in multiplying (2560 x 1024) times (1024 x (8192+1)) (instead of (2560 x 1024)
times (1024 x 793471)), i.e. about 100-fold less computation.
