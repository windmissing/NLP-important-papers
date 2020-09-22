# Conclusion

In this paper, we propose two simple and effectiveattentional mechanisms for neural machine trans-lation:   theglobalapproach  which  always  looksat all source positions and thelocalone that onlyattends  to a subset of source  positions  at a time.We  test  the  effectiveness  of  our  models  in  theWMT translation tasks between English and Ger-man in both directions.  Our local attention yieldslarge gains of up to5.0BLEU over non-attentional17The reference uses a more fancy translation of “incom-patible”, which is “im Widerspruch zu etwas stehen”.  Bothmodels, however, failed to translate “passenger experience”.models  which  already  incorporate  known  tech-niques such as dropout.   For the English  to Ger-man translation direction, our ensemble model hasestablished  new  state-of-the-art  results  for  bothWMT’14  and  WMT’15,  outperforming  existingbest systems, backed by NMT models andn-gramLM rerankers, by more than 1.0 BLEU.

We have compared various alignment functionsand  shed  light  on  which  functions  are  best  forwhich attentional models. Our analysis shows thatattention-based NMT models are superior to non-attentional  ones  in  many  cases,  for  example  intranslating names and handling long sentences.

# Acknowledgment

We  gratefully  acknowledge  support  from  a  giftfrom Bloomberg L.P. and the support of NVIDIA Corporation with the donation of Tesla K40 GPUs.We  thank  Andrew  Ng  and  his  group  as  well  asthe  Stanford  Research  Computing  for  letting  ususe  their  computing  resources.    We  thank  Rus-sell Stewart for helpful discussions on the models.Lastly,  we thank  Quoc  Le,  Ilya  Sutskever,  OriolVinyals,  Richard  Socher,  Michael  Kayser,  JiweiLi,  Panupong  Pasupat,  Kelvin  Guu,  members  ofthe Stanford NLP Group and the annonymous re-viewers for their valuable comments and feedback.

