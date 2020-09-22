# Experiments

We   evaluate   the   effectiveness   of   our   modelson   the   WMT   translation   tasks   between   En-glish   and   German   in   both   directions.newstest2013  (3000  sentences)  is used as a development set to select our hyperparameters.   Translation  performances  are  reported  in  case-sensitive BLEU   (Papineni et al., 2002)   on   newstest2014(2737  sentences)  and  newstest2015  (2169  sentences). Following (Luong et al., 2015), we reporttranslation  quality using two types of BLEU: (a)tokenized12BLEU to be comparable with existingNMT work and (b)NIST13BLEU to be compara-ble with WMT results.

## Training Details

All our models are trained on the WMT’14 training data consisting of 4.5M sentences pairs (116MEnglish  words,  110M  German  words).    Similarto (Jean et al., 2015), we limit our vocabularies tobe the top 50K most frequent words for both lan-guages.  Words not in these shortlisted vocabular-ies are converted into a universal token `<unk>`.

When  training  our  NMT  systems,  following(Bahdanau et al., 2015;  Jean et al., 2015),  we  filter   out   sentence   pairs   whose   lengths   exceed50  words  and  shuffle  mini-batches  as  we  proceed.    Our  stacking  LSTM  models  have  4  layers,  each with 1000  cells,  and 1000-dimensional embeddings.    We  follow  (Sutskever et al., 2014;Luong et al., 2015) in training NMT with similarsettings:  (a) our parameters are uniformly initial-ized in[−0.1,0.1], (b) we train for 10 epochs us-12All  texts  are  tokenized  withtokenizer.perlandBLEU scores are computed withmulti-bleu.perl.13With themteval-v13ascript as per WMT guideline.
SystemPplBLEUWinning WMT’14 system –phrase-based + large LM(Buck et al., 2014)20.7Existing NMT systemsRNNsearch (Jean et al., 2015)16.5RNNsearch + unk replace (Jean et al., 2015)19.0RNNsearch + unk replace + large vocab +ensemble8 models (Jean et al., 2015)21.6Our NMT systemsBase10.611.3Base + reverse9.912.6 (+1.3)Base + reverse + dropout8.114.0 (+1.4)Base + reverse + dropout + global attention (location)7.316.8 (+2.8)Base + reverse + dropout + global attention (location) + feed input6.418.1 (+1.3)Base + reverse + dropout + local-p attention (general) + feed input5.919.0 (+0.9)Base + reverse + dropout + local-p attention (general) + feed input + unk replace20.9 (+1.9)Ensemble8 models + unk replace23.0 (+2.1)Table 1:WMT’14 English-German results– shown are the perplexities (ppl) and thetokenizedBLEUscores of various systems on newstest2014.  We highlight thebestsystem in bold and giveprogressiveimprovements in italic between consecutive systems.local-preferes to the local attention with predictivealignments. We indicate for each attention model the alignment score function used in pararentheses.ing plain SGD, (c) a simple learning  rate sched-ule is employed – we start with a learning rate of1;  after 5 epochs,  we begin to halve the learningrate every epoch,  (d) our mini-batch  size is 128,and (e) the normalized gradient is rescaled when-ever  its  norm  exceeds  5.   Additionally,  we  alsouse dropout with probability0.2for our LSTMs assuggested by (Zaremba et al., 2015).  For dropoutmodels,  we train for 12 epochs and start halvingthe learning  rate after 8 epochs.   For local atten-tion models,  we empirically  set the window sizeD= 10.

Our code is implemented  in MATLAB. Whenrunning  on  a  single  GPU  device  Tesla  K40,  weachieve  a  speed  of  1Ktargetwords  per  second.It takes 7–10 days to completely train a model.

## English-German Results

We  compare  our  NMT  systems  in  the  English-German  task  with  various  other  systems.   These include the winning syste minWMT’14(Buck et al., 2014),aphrase-basedsystemwhose  language  models  were  trained  on  a  hugemonolingual  text,   the  Common  Crawl  corpus.For  end-to-end  NMT  systems,   to  the  best  ofour   knowledge,   (Jean et al., 2015)   is   the   onlywork  experimenting  with  this  language  pair  andcurrently  the  SOTA  system.We  only  presentresults for some of our attention models and willlater analyze the rest in Section 5.

As   shown   in   Table   1,    we   achieve   pro-gressive  improvements  when  (a)  reversing  thesource  sentence,   +1.3BLEU,  as  proposed   in(Sutskever et al., 2014)   and   (b)   using   dropout,+1.4BLEU. On top of that,  (c) the global atten-tion  approach  gives  a  significant  boost  of  +2.8BLEU, making our model slightly better than thebase attentional system of Bahdanau et al. (2015)(rowRNNSearch).When  (d)  using  theinput-feedingapproach,  we seize  another  notable  gainof +1.3BLEU and outperform their system.  Thelocal  attention  model  with  predictive  alignments(rowlocal-p)  proves  to  be  even  better,  givingus  a  further  improvement  of  +0.9BLEU on  topof  the  global  attention  model.  

> **[success]** 以上是对性能有提升作用的操作  

It  is  interesting  to  observe  the  trend  previously  reported  in(Luong et al., 2015) that **perplexity strongly correlates with translation quality**.  In total, we achievea  significant  gain  of  5.0  BLEU  points  over  thenon-attentional  baseline,  which  already  includesknown  techniques  such  as  source  reversing  and dropout.

The unknown replacement technique proposed in (Luong et al., 2015; Jean et al., 2015) yields another nice gain of +1.9BLEU, demonstrating that **our attentional models do learn useful alignments for  unknown  works**.   

> **[warning]** unknown replacement technique?  

Finally,  by  ensembling  8different  models  of  various  settings,  e.g.,  usingdifferent  attention  approaches,  with  and  withoutdropout etc., we were able to achieve anew SOTAresult of23.0BLEU, outperforming  the existing
best system (Jean et al., 2015) by +1.4BLEU.SystemBLEUTop –NMT + 5-gram rerank(Montreal)24.9Our ensemble 8 models + unk replace25.9Table  2:WMT’15  English-German  results–NISTBLEU   scores   of   the   winning   entry   inWMT’15 and our best one on newstest2015.

Latest results in WMT’15– despite the fact thatour models were trained on WMT’14 with slightlyless data, we test them on newstest2015 to demon-strate that they can generalize well to different testsets.   As  shown  in  Table  2,  our  best  system  es-tablishes anew SOTAperformance of25.9BLEU,outperforming the existing best system backed byNMT and a 5-gram LM reranker by +1.0BLEU.

## German-English Results

We carry out a similar set of experiments for theWMT’15  translation  task  from  German  to  English.   While  our  systems  have  not  yet  matchedthe performance  of the SOTA system,  we nevertheless  show  the effectiveness  of our approacheswith large and progressive gains in terms of BLEUas  illustrated  in  Table  3.   Theattentionalmech-anism  gives  us  +2.2BLEU  gain  and  on  top  ofthat, we obtain another boost of up to +1.0BLEUfrom  theinput-feedingapproach.   Using a betteralignment function, the content-baseddotproductone, together withdropoutyields another gain of+2.7BLEU. Lastly, when applying the unknownword  replacement  technique,  we  seize  an  additional  +2.1BLEU, demonstrating  the  usefulnessof attention in aligning rare words.

