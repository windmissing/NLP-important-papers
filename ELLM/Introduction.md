# Introduction

Language Modeling (LM) is a task central to Natural Language Processing (NLP) and Language Understanding. Models which can accurately place distributions over sentences not only encode complexities of language such as grammatical structure, but also distill a fair amount of information about the knowledge that a corpora may contain.   

> **[success]**   
什么是LM：  
（1）accurately place distributions over sentences  
（2）encode complexities of language  
（3）distill a fair amount of information  

Indeed, models that are able to assign a low probability to sentences that are grammatically correct but unlikely may help other tasks in fundamental language understanding like question answering, machine translation, or text summarization.  

> **[success]** 关于LM作用的一个具体的例子  

LMs have played a key role in traditional NLP tasks such as speech recognition (Mikolov et al., 2010; Arisoy et al., 2012), machine translation (Schwenk et al., 2012; Vaswani et al.), or text summarization (Rush et al., 2015; Filippova et al., 2015). Often (although not always), training better language models **improves the underlying metrics of the downstream task** (such as word error rate for speech recognition, or BLEU score for translation), which makes the task of training better LMs valuable by itself.  

> **[success]**   
underlying metrics：基本指标  
LM的作用1：improve其它NLP tasks的基本指标。  

Further, when trained on vast amounts of data, language models **compactly extract knowledge** encoded in the training data. For example, when trained on movie subtitles (Serban et al., 2015; Vinyals & Le, 2015), these language models are able to generate basic answers to questions about object colors, facts about people, etc.   

> **[success]**   
subtitles：字幕  
LM的作用2：从大量的数据中提炼简洁的信息  

Lastly, recently proposed sequence-to-sequence models employ **conditional language models** (Mikolov & Zweig, 2012) as their key component to solve diverse tasks like machine translation (Sutskever et al., 2014; Cho et al., 2014; Kalchbrenner et al., 2014) or video generation (Srivastava et al., 2015a).  

> **[success]**   
LM的作用3：seq2seq问题中的条件语言模型   

Deep Learning and Recurrent Neural Networks (RNNs)
have fueled language modeling research in the past years
as it allowed researchers to explore many tasks for which
the strong conditional independence assumptions are unre-
alistic.   

> **[success]**  
fuel：加剧  
conditional independence assumptions：[条件独立假设](https://windmissing.github.io/mathematics_basic_for_ML/Probability/probability_distribution.html)  
LM在不满足“条件独立假设”的场景中的应用。  

Despite the fact that simpler models, such as N-
grams, only use a short history of previous words to predict
the next word, they are still a key component to high qual-
ity, low perplexity LMs.   

> **[warning]**  
[?] N-grams是什么？  
N-grams模型简单、效果好。  

Indeed, most recent work on large
scale LM has shown that RNNs are great in combination
with N-grams, as they may have different strengths that
complement N-gram models, but worse when considered
in isolation (Mikolov et al., 2011; Mikolov, 2012; Chelba
et al., 2013; Williams et al., 2015; Ji et al., 2015a; Shazeer
et al., 2015).  

> **[success]**  
complement：补充  
RNN与N-gram结合的LM效果好  

We believe that, despite much work being devoted to small
data sets like the Penn Tree Bank (PTB) (Marcus et al.,
1993), research on larger tasks is very relevant as overfit-
ting is not the main limitation in current language model-
ing, but is the main characteristic of the PTB task. Results
on larger corpora usually show better what matters as many
ideas work well on small data sets but fail to improve on
larger data sets. Further, given current hardware trends and
vast amounts of text available on the Web, it is much more
straightforward to tackle large scale modeling than it used
to be. Thus, we hope that our work will help and motivate
researchers to work on traditional LM beyond PTB – for
this purpose, we will open-source our models and training
recipes.
We focused on a well known, large scale LM benchmark:
the One Billion Word Benchmark data set (Chelba et al.,
2013). This data set is much larger than PTB (one thou-
sand fold, 800k word vocabulary and 1B words training
data) and far more challenging. Similar to Imagenet (Deng
et al., 2009), which helped advance computer vision, we
believe that releasing and working on large data sets and
models with clear benchmarks will help advance Language
Modeling.
The contributions of our work are as follows:
• We explored, extended and tried to unify some of the
current research on large scale LM.
• Specifically, we designed a Softmax loss which is
based on character level CNNs, is efficient to train,
and is as precise as a full Softmax which has orders of
magnitude more parameters.
• Our study yielded significant improvements to the
state-of-the-art on a well known, large scale LM task:
from 51.3 down to 30.0 perplexity for single models
whilst reducing the number of parameters by a factor
of 20.
 We show that an ensemble of a number of different
models can bring down perplexity on this task to 23.7,
a large improvement compared to current state-of-art.
• We share the model and recipes in order to help and
motivate further research in this area.
In Section 2 we review important concepts and previous
work on language modeling. Section 3 presents our contri-
butions to the field of neural language modeling, emphasiz-
ing large scale recurrent neural network training. Sections
4 and 5 aim at exhaustively describing our experience and
understanding throughout the project, as well as emplacing
our work relative to other known approaches.
