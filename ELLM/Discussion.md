# Discussion and Conclusions
In this paper we have shown that RNN LMs can be trained
on large amounts of data, and outperform competing mod-
els including carefully tuned N-grams. The reduction in
perplexity from 51.3 to 30.0 is due to several key compo-
nents which we studied in this paper. Thus, a large, regular-
ized LSTM LM, with projection layers and trained with an
approximation to the true Softmax with importance sam-
pling performs much better than N-grams. Unlike previ-
ous work, we do not require to interpolate both the RNN
LM and the N-gram, and the gains of doing so are rather
marginal.
By exploring recent advances in model architectures (e.g.
LSTMs), exploiting small character CNNs, and by sharing
ourfindingsinthispaperandaccompanyingcodeandmod-
els (to be released upon publication), we hope to inspire
research on large scale Language Modeling, a problem we
consider crucial towards language understanding. We hope
for future research to focus on reasonably sized datasets
taking inspiration from recent advances seen in the com-
puter vision community thanks to efforts such as Imagenet
(Deng et al., 2009).
# Acknowledgements
We thank Ciprian Chelba, Ilya Sutskever, and the Google
Brain Team for their help and discussions. We also thank
Koray Kavukcuoglu for his help with the manuscript.
