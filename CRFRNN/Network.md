# The End-to-end Trainable Network

We now describe our end-to-end deep learning system for semantic image segmentation. To pave the way for this, we first explain **how repeated mean-field iterations can be organized as an RNN**.  

## CRF as RNN

In the previous section, it was shown that one iteration
of the mean-field algorithm can be formulated as a stack of
common CNN layers (see Fig. 1). We use the function f θ
to denote the transformation done by one mean-field iter-
ation: given an image I, pixel-wise unary potential values
U and an estimation of marginal probabilities Q in from the
FCN CRF-RNN
Figure 2. The End-to-end Trainable Network. Schematic vi-
sualization of our full network which consists of a CNN and the
CNN-CRF network. Best viewed in colour.
previous iteration, the next estimation of marginal distribu-
tionsafteronemean-fielditerationisgivenbyf θ (U,Q in ,I).
The vector θ =
? w (m) ,µ(l,l ′ ) ?
, m ∈ {1,...,M}, l,l ′ ∈
{l 1 ,...,l L } represents the CRF parameters described in
Section 4.
Multiple mean-field iterations can be implemented by re-
peating the above stack of layers in such a way that each
iteration takes Q value estimates from the previous iteration
and the unary values in their original form. This is equiva-
lent to treating the iterative mean-field inference as a Recur-
rent Neural Network (RNN). The behaviour of the network
is given by the following equations where H 1 ,H 2 are hid-
den states, and T is the number of mean-field iterations:
H 1 (t) =
(
softmax(U), t = 0
H 2 (t − 1), 0 < t ≤ T,
(3)
H 2 (t) = f θ (U,H 1 (t),I), 0 ≤ t ≤ T, (4)
Y (t) =
(
0, 0 ≤ t < T
H 2 (t), t = T.
(5)
We name this RNN structure CRF-RNN. Parameters of
the CRF-RNN are same as the mean-field parameters de-
scribed in Section 4 and denoted by θ here. Since the calcu-
lation of error differentials w.r.t. these parameters in a single
iterationwasdescribedinSection4, theycanbelearntinthe
RNN setting using the standard back-propagation through
time algorithm [46, 38]. It was shown in [27] that the mean-
field iterative algorithm for dense CRF converges in less
than 10 iterations. Furthermore, in practice, after about 5
iterations, increasing the number of iterations usually does
not significantly improve results [27]. Therefore, it does
not suffer from the vanishing and exploding gradient prob-
lem inherent to deep RNNs [7, 41]. This allows us to use a
plain RNN architecture instead of more sophisticated archi-
tectures such as LSTMs in our network.
5.2. Completing the Picture
Our approach comprises a fully convolutional network
stage, which predicts pixel-level labels without consid-
ering structure, followed by a CRF-RNN stage, which
performs CRF-based probabilistic graphical modelling for
structured prediction. The complete system, therefore, uni-
fies strengths of both CNNs and CRFs and is trainable
end-to-end using the back-propagation algorithm [32] and
the Stochastic Gradient Descent (SGD) procedure. During
training, a whole image (or many of them) can be used as
the mini-batch and the error at each pixel output of the net-
work can be computed using an appropriate loss function
such as the softmax loss with respect to the ground truth
segmentation of the image. We used the FCN-8s architec-
ture of [35] as the first part of our network, which provides
unary potentials to the CRF. This network is based on the
VGG-16 network [51] but has been restructured to perform
pixel-wise prediction instead of image classification. The
complete architecture of our network, including the FCN-
8s part can be found in the supplementary material.
In the forward pass through the network, once the com-
putation enters the CRF-RNN after passing through the
CNN stage, it takes T iterations for the data to leave the
loop created by the RNN. Neither the CNN that provides
unary values nor the layers after the CRF-RNN (i.e., the
loss layers) need to perform any computations during this
time since the refinement happens only inside the RNN’s
loop. Once the output Y leaves the loop, next stages of the
deep network after the CRF-RNN can continue the forward
pass. In our setup, a softmax loss layer directly follows the
CRF-RNN and terminates the network.
During the backward pass, once the error differentials
reach the CRF-RNN’s output Y , they similarly spend T it-
erations within the loop before reaching the RNN input U
in order to propagate to the CNN which provides the unary
input. In each iteration inside the loop, error differentials
are computed inside each component of the mean-field it-
eration as described in Section 4. We note that unnecessar-
ily increasing the number of mean-field iterations T could
potentially result in the vanishing and exploding gradient
problems in the CRF-RNN.
6. Implementation Details
In the present section we describe the implementation
details of the proposed network, as well as its training pro-
cess. The high-level architecture of our system, which was
implemented using the popular Caffe [25] deep learning li-
brary, is shown in Fig. 2. Complete architecture of the deep
network can be found in the supplementary material. The
source code and the trained models of our approach will be
made publicly available.
We initialized the first part of the network using the pub-
licly available weights of the FCN-8s network [35]. The
compatibility transform parameters of the CRF-RNN were
initialized using the Potts model, and kernel width and
weight parameters were obtained from a cross-validation
process. We found that such initialization results in faster
convergence of training. During the training phase, param-
eters of the whole network were optimized end-to-end using
the back-propagation algorithm. In particular, we used full
image training described in [35], with learning rate fixed at
10 −13 and momentum set to 0.99. These extreme values of
the parameters were used since we employed only one im-
age per batch to avoid reaching memory limits of the GPU.
In all our experiments, during training, we set the num-
ber of mean-field iterations T in the CRF-RNN to 5 to avoid
vanishing/exploding gradient problems and to reduce the
training time. During the test time, iteration count was in-
creased to 10. The effect of this parameter value on the
accuracy is discussed in Section 7.1.
Loss function During the training of the models that
achieved the best results reported in this paper, we used the
standard softmax loss function, that is, the log-likelihood
error function [28]. The standard metric used in the Pascal
VOC challenge is the average intersection over union (IU),
which we also use here to report the results. In our experi-
ments we found that high values of IU on the validation set
were associated to low values of the averaged softmax loss,
to a large extent. We also tried the robust log-likelihood
in [28] as a loss function for training. However, this did not
result in increased accuracy nor faster convergence.
Normalization techniques As described in Section 4,
we use the exponential function followed by pixel-wise nor-
malization across channels in several stages of the CRF-
RNN. Since this operation has a tendency to result in small
gradients with respect to the input when the input value is
large, we conducted several experiments where we replaced
this by a rectified linear unit (ReLU) operation followed by
a normalization across the channels. Our hypothesis was
that this approach may approximate the original operation
adequately while speeding up the training due to improved
gradients. Furthermore, ReLU would induce sparsity on the
probability of labels assigned to pixels, implicitly pruning
low likelihood configurations, which could have a positive
effect. However, this approach did not lead to better results,
obtaining 1% IU lower than that of original setting.
7. Experiments
We present experimental results with the proposed CRF-
RNN framework. We use two datasets: the Pascal VOC
2012 dataset, and the Pascal Context dataset. We use the
Pascal VOC 2012 dataset as it has become the golden stan-
dard to comprehensively evaluate any new semantic seg-
mentation approach. We also use the Pascal Context dataset
to assess how well our approach performs on a dataset with
different characteristics.
Pascal VOC Datasets
In order to evaluate our approach with existing methods un-
der the same circumstances, we conducted two main exper-
Figure 3. Qualitative results on the validation set of Pascal
VOC 2012. FCN [35] is a CNN-based model that does not em-
ploy CRF. Deeplab [9] is a two-stage approach, where the CNN is
trained first, and then CRF is applied on top of the CNN output.
Our approach is an end-to-end trained system that integrates both
CNN and CRF-RNN in one deep network. Best viewed in colour.
iments with the Pascal VOC 2012 dataset, followed by a
qualitative experiment.
In the first experiment, following [35, 36, 39], we used
a training set consisted of VOC 2012 training data (1464
images), and training and validation data of [22], which
amounts to a total of 11,685 images. After removing the
overlapping images between VOC 2012 validation data and
this training dataset, we were left with 346 images from the
original VOC 2012 validation set to validate our models on.
We call this set the reduced validation set in the sequel. An-
notations of the VOC 2012 test set, which consists of 1456
images, are not publicly available and hence the final results
on the test set were obtained by submitting the results to the
Pascal VOC challenge evaluation server [17]. Regardless
of the smaller number of images, we found that the relative
improvements of the accuracy on our validation set were in
good agreement with the test set.
As a first step we directly compared the potential advan-
tage of learning the model end-to-end with respect to alter-
natives. These are plain FCN-8s without applying CRF, and
with CRF as a postprocessing method disconnected from
the training of FCN, which is comparable to the approach
described in [9] and [39]. The results are reported in Table 1
and show a clear advantage of the end-to-end strategy over
the offline application of CRF as a post-processing method.
This can be attributed to the fact that during the SGD train-
ing of the CRF-RNN, the CNN component and the CRF
component learn how to co-operate with each other to pro-
duce the optimum output of the whole network.
We then proceeded to compare our approach with all
state-of-the-art methods that used training data from the
standard VOC 2012 training and validation sets, and from
the dataset published with [21]. The results are shown in
Table 2, above the bar, and we can see that our approach
outperforms all competitors.
In the second experiment, in addition to the above train-
ing set, we used data from the Microsoft COCO dataset [34]
as was done in [39] and [11]. We selected images from
COCO 2014 training set where the ground truth segmen-
tation has at least 200 pixels marked with classes labels
present in the VOC 2012 dataset. With this selection, we
ended up using 66,099 images from the COCO dataset and
therefore a total of 66,099 + 11,685 = 77,784 training im-
ageswereusedinthesecondexperiment. Thesamereduced
validation set was used in this second experiment as well.
In this case, we first fine-tuned the plain FCN-32s network
(without the CRF-RNN part) on COCO data, then we built
an FCN-8s network with the learnt weights and finally train
the CRF-RNN network end-to-end using VOC 2012 train-
ing data only. Since the MS COCO ground truth segmen-
tation data contains somewhat coarse segmentation masks
where objects are not delineated properly, we found that
fine-tuning our model with COCO did not yield significant
improvements. This can be understood because the primary
advantage of our model comes from delineating the objects
and improving fine segmentation boundaries. The VOC
2012 training dataset therefore helps our model learn this
task effectively. The results of this experiment are shown in
Table 2, below the bar, and we see that our approach sets a
new state-of-the-art on the VOC 2012 dataset.
Note that in both setups, our approach outperforms com-
peting methods due to the end-to-end training of the CNN
and CRF in the unified CRF-RNN framework. We also
evaluated our models on the VOC 2010, and VOC 2011 test
set (see Table 2). In all cases our method achieves the state-
of-the-art performance.
Method Without COCO With COCO
Plain FCN-8s 61.3 68.3
FCN-8s and CRF disconnected 63.7 69.5
End-to-end training of
CRF-RNN
69.6 72.9
Table 1. Mean IU accuracy of our approach, CRF-RNN, compared
with similar methods, evaluated on the reduced VOC 2012 valida-
tion set.
Pascal Context Dataset
We conducted an experiment on the Pascal Context dataset
[37], which differs from the previous one in the larger num-
ber of classes considered, 59. We used the provided parti-
tions of training and validation sets, and the obtained results
are reported in Table 3.
Method O 2 P [8] CFM[12]
FCN-
8s [35]
CRF-
RNN
Mean IU 18.1 31.5 37.78 39.28
Table 3. Mean IU accuracy of our approach, CRF-RNN, evaluated
on the Pascal Context validation set.
7.1. Effect of Design Choices
We performed a number of additional experiments on the
Pascal VOC 2012 validation set described above to study
the effect of some design choices we made.
We studied the performance gains attained by our mod-
ifications to CRF over the CRF approach [27]. We found
that using different filter weights for different classes im-
proved the performance by 1.8 percentage points, and that
introducing the asymmetric compatibility transform further
boosted the performance by 0.9 percentage points.
Regarding the RNN parameter iteration count T, incre-
menting it to T = 10 during the test time, from T = 5
during the train time, produced an accuracy improvement
of 0.2 percentage points. Setting T = 10 also during train-
ing reduced the accuracy by 0.7 percentage points. We be-
lieve that this might be due to a vanishing gradient effect
caused by using too many iterations. In practice that leads
to the first part of the network (the one producing unary po-
tentials) receiving a very weak error gradient signal during
training, thus hampering its learning capacity.
End-to-end training after the initialization of CRF pa-
rameters improved performance by 3.4 percentage points.
We also conducted an experiment where we froze the FCN-
8s part and fine-tuned only the RNN part (i.e., CRF param-
eters). It improved the performance over initialization by
only 1 percentage point. We therefore conclude that end-
to-end training helped to boost the accuracy of the system
significantly.
Treating each iteration of mean-field inference as an in-
dependent step with its own parameters, and training end-
to-end with 5 such iterations yielded a final mean IU score
of only 70.9, supporting the hypothesis that the recurrent
structure of our approach is important for its success.
8. Conclusion
We presented CRF-RNN, an interpretation of dense
CRFs as Recurrent Neural Networks. Our formulation
fully integrates CRF-based probabilistic graphical mod-
elling with emerging deep learning techniques. In partic-
ular, the proposed CRF-RNN can be plugged in as a part
of a traditional deep neural network: It is capable of pass-
ing on error differentials from its outputs to inputs dur-
ing back-propagation based training of the deep network
while learning CRF parameters. We demonstrate the use
of this approach by utilizing it for the semantic segmenta-
tion task: we form an end-to-end trainable deep network
by combining a fully convolutional neural network with the
CRF-RNN. Our system achieves a new state-of-the-art on
the popular Pascal VOC segmentation benchmark. This im-
provement can be attributed to combining the strengths of
CNNs and CRFs in a single deep network.
In the future, we plan to investigate the advan-
tages/disadvantages of restricting the capabilities of the
RNN part of our network to mean-field inference of dense
CRF. A sensible baseline to the work presented here would
be to use more standard RNNs (e.g. LSTMs) that learn to
iteratively improve the input unary potentials to make them
closer to the ground-truth.
Acknowledgement This work was supported by grants Leverhulme
Trust, EPSRC EP/I001107/2 and ERC 321162-HELIOS. We thank the
Caffe team, Baidu IDL, and the Oxford ARC team for their support. We
gratefully acknowledge GPU donations from NVIDIA.
References
[1] A. Adams, J. Baek, and M. A. Davis. Fast high-dimensional filtering
using the permutohedral lattice. CGF, 2010.
[2] P. Arbeláez, B. Hariharan, C. Gu, S. Gupta, L. Bourdev, and J. Malik.
Semantic segmentation using regions and parts. In CVPR, 2012.
[3] P. Arbeláez, M. Maire, C. Fowlkes, and J. Malik. Contour detection
and hierarchical image segmentation. TPAMI, (5), 2011.
[4] A. Barbu. Training an active random field for real-time image de-
noising. TIP, (11), 2009.
[5] S. Bell, P. Upchurch, N. Snavely, and K. Bala. Material recognition
in the wild with the materials in context database. In CVPR, 2015.
[6] Y. Bengio, Y. LeCun, and D. Henderson. Globally trained hand-
written word recognizer using spatial representation, convolutional
neural networks, and hidden markov models. In NIPS, 1994.
[7] Y. Bengio, P. Simard, and P. Frasconi. Learning long-term depen-
dencies with gradient descent is difficult. IEEE TNN, 1994.
[8] J. Carreira, R. Caseiro, J. Batista, and C. Sminchisescu. Free-form
region description with second-order pooling. TPAMI, 2014.
[9] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L.
Yuille. Semantic image segmentation with deep convolutional nets
and fully connected crfs. In ICLR, 2015.
[10] L.-C. Chen, A. G. Schwing, A. L. Yuille, and R. Urtasun. Learning
deep structured models. In ICLRW, 2015.
[11] J. Dai, K. He, and J. Sun. Boxsup: Exploiting bounding boxes to su-
pervise convolutional networks for semantic segmentation. In ICCV,
2015.
[12] J. Dai, K. He, and J. Sun. Convolutional feature masking for joint
object and stuff segmentation. In CVPR, 2015.
[13] T.-M.-T. Do and T. Artieres. Neural conditional random fields. In
NIPS, 2010.
[14] J. Domke. Learning graphical model parameters with approximate
marginal inference. 2013.
[15] J. Dong, Q. Chen, S. Yan, and A. Yuille. Towards unified object
detection and semantic segmentation. In ECCV, 2014.
[16] D. Eigen, C. Puhrsch, and R. Fergus. Depth map prediction from a
single image using a multi-scale deep network. In NIPS, 2014.
[17] M. Everingham, S. M. A. Eslami, L. Van Gool, C. K. I. Williams,
J. Winn, and A. Zisserman. The pascal visual object classes chal-
lenge: A retrospective. IJCV, 111(1).
[18] C. Farabet, C. Couprie, L. Najman, and Y. LeCun. Learning hierar-
chical features for scene labeling. TPAMI, 2013.
[19] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hier-
archies for accurate object detection and semantic segmentation. In
CVPR, 2014.
[20] R. Girshick, F. Iandola, T. Darrell, and J. Malik. Deformable part
models are convolutional neural networks. In CVPR, 2015.
[21] B. Hariharan, P. Arbelaez, L. D. Bourdev, S. Maji, and J. Malik.
Semantic contours from inverse detectors. In ICCV, 2011.
[22] B. Hariharan, P. Arbeláez, R. Girshick, and J. Malik. Simultaneous
detection and segmentation. In ECCV, 2014.
[23] B. Hariharan, P. Arbelaez, R. Girshick, and J. Malik. Hypercolumns
for object segmentation and fine-grained localization. In CVPR,
2015.
[24] M. Jaderberg, K. Simonyan, A. Vedaldi, and A. Zisserman. Deep
structured output learning for unconstrained text recognition. In
ICLR, 2015.
[25] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick,
S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for
fast feature embedding. In ACM Multimedia.
[26] M. Kiefel and P. V. Gehler. Human pose estmation with fields of
parts. In ECCV, 2014.
[27] P. Krähenbühl and V. Koltun. Efficient inference in fully connected
crfs with gaussian edge potentials. In NIPS, 2011.
[28] P. Krähenbühl and V. Koltun. Parameter learning and convergent
inference for dense random fields. In ICML, 2013.
[29] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classifica-
tion with deep convolutional neural networks. In NIPS, 2012.
[30] L. Ladicky, C. Russell, P. Kohli, and P. H. Torr. Associative hierar-
chical crfs for object class image segmentation. In ICCV, 2009.
[31] J. D. Lafferty, A. McCallum, and F. C. N. Pereira. Conditional ran-
dom fields: Probabilistic models for segmenting and labeling se-
quence data. In ICML, 2001.
[32] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based
learning applied to document recognition. Proceedings of the IEEE,
(11), 1998.
[33] G. Lin, C. Shen, I. Reid, and A. van dan Hengel. Efficient piecewise
training of deep structured models for semantic segmentation. In
arXiv:1504.01013, 2015.
[34] T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays,
P. Perona, D. Ramanan, C. L. Zitnick, and P. Dollar. Microsoft coco:
Common objects in context. In arXiv:1405.0312, 2014.
[35] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks
for semantic segmentation. In CVPR, 2015.
[36] M. Mostajabi, P. Yadollahpour, and G. Shakhnarovich. Feedforward
semantic segmentation with zoom-out features. In CVPR, 2015.
[37] R. Mottaghi, X. Chen, X. Liu, N.-G. Cho, S.-W. Lee, S. Fidler, R. Ur-
tasun, and A. Yuille. The role of context for object detection and
semantic segmentation in the wild. In CVPR, 2014.
[38] M. C. Mozer. Backpropagation. chapter A Focused Backpropagation
Algorithm for Temporal Pattern Recognition. L. Erlbaum Associates
Inc., 1995.
[39] G. Papandreou, L.-C. Chen, K. Murphy, and A. L. Yuille. Weakly-
and semi-supervised learning of a dcnn for semantic image segmen-
tation. In ICCV, 2015.
[40] S. Paris and F. Durand. A fast approximation of the bilateral filter
using a signal processing approach. (1), 2013.
[41] R. Pascanu, C. Gulcehre, K. Cho, and Y. Bengio. On the difficulty of
training recurrent neural networks. In ICML, 2013.
[42] G. S. Payman Yadollahpour, Dhruv Batra. Discriminative re-ranking
of diverse segmentations. In CVPR, 2013.
[43] J. Peng, L. Bo, and J. Xu. Conditional neural fields. In NIPS, 2009.
[44] P. H. O. Pinheiro and R. Collobert. Recurrent convolutional neural
networks for scene labeling. In ICML, 2014.
[45] S. Ross, D. Munoz, M. Hebert, and J. A. Bagnell. Learning message-
passing inference machines for structured prediction. In CVPR,
2011.
[46] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Neurocomputing:
Foundations of research. chapter Learning Internal Representations
by Error Propagation. MIT Press, 1988.
[47] A. G. Schwing and R. Urtasun. Fully connected deep structured net-
works. In arXiv:1503.02351, 2015.
[48] J. Shotton, A. Fitzgibbon, M. Cook, T. Sharp, M. Finocchio,
R. Moore, A. Kipman, and A. Blake. Real-time human pose recog-
nition in parts from single depth images. In CVPR, 2011.
[49] J. Shotton, M. Johnson, and R. Cipolla. Semantic texton forests for
image categorization and segmentation. In CVPR, 2008.
[50] J. Shotton, J. Winn, C. Rother, and A. Criminisi. Textonboost for im-
age understanding: Multi-class object recognition and segmentation
by jointly modeling texture, layout, and context. IJCV, (1), 2009.
[51] K. Simonyan and A. Zisserman. Very deep convolutional networks
for large-scale image recognition. In ICLR, 2014.
[52] V. Stoyanov, A. Ropson, and J. Eisner. Empirical risk minimization
of graphical model parameters given approximate inference, decod-
ing, and model structure. In AISTATS, 2011.
[53] S. C. Tatikonda and M. I. Jordan. Loopy belief propagation and gibbs
measures. In UAI, 2002.
[54] C. Tomasi and R. Manduchi. Bilateral filtering for gray and color
images. In CVPR, 1998.
[55] J. J. Tompson, A. Jain, Y. LeCun, and C. Bregler. Joint training
of a convolutional network and a graphical model for human pose
estimation. In NIPS, 2014.
[56] Z. Tu. Auto-context and its application to high-level vision tasks. In
CVPR, 2008.
[57] Z. Tu, X. Chen, A. L. Yuille, and S.-C. Zhu. Image parsing: Unify-
ing segmentation, detection, and recognition. IJCV, 63(2):113–140,
2005.
[58] Y. Zhang and T. Chen. Efficient inference for fully-connected crfs
with stationarity. In CVPR, 2012.