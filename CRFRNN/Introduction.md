# Introduction

Low-level computer vision problems such as semantic image segmentation or depth estimation often involve **assigning a label to each pixel in an image**.   

> **[success]**
像素级标签任务要解决的问题：assigning a label to each pixel in an image

While the feature representation used to classify individual pixels plays an important role in this task, it is similarly important to consider factors such as image edges, appearance consistency and spatial consistency while assigning labels in order to obtain accurate and precise results.

> **[success]**   
feature representation的作用：  
（1）像素级标签任务  
（2）image edges, appearance consistency and spatial consistency  

Designing a strong **feature representation** is a key challenge in pixel-level labelling problems. Work on this topic includes: TextonBoost [50], TextonForest [49], and Random Forest-based classifiers [48]. Recently, supervised deep learning approaches such as large-scale deep Convolutional Neural Networks (CNNs) have been immensely successful in many high-level computer vision tasks such as image recognition [29] and object detection [19].   

> **[info]** immensely：极其地  

This motivates exploring the use of CNNs for pixel-level labelling problems. The key insight is to learn a strong feature representation end-to-end for the pixel-level labelling task instead of hand-crafting features with heuristic parameter tuning.   

> **[info]** insight：洞察力  

In fact, a number of recent approaches including the particularly interesting works FCN [35] and DeepLab [9] have shown a significant accuracy boost by adapting state-of-the-art CNN based image classifiers to the semantic segmentation problem.

However, there are significant challenges in adapting CNNs designed for high level computer vision tasks such as object recognition to pixel-level labelling tasks.   

> **[success]** CNN是为high-level设计的算法，用于low-level的task不是很合适。  

Firstly, traditional CNNs have convolutional filters with large receptive fields and hence produce coarse outputs when restructured to produce pixel-level labels [35]. Presence of maxpooling layers in CNNs further reduces the chance of getting a fine segmentation output [9]. This, for instance, can result in non-sharp boundaries and blob-like shapes in semantic segmentation tasks.   

> **[success]**  
[?] receptive fields  
[?] blob-like shapes  
CNN的缺点1：receptive fields和maxpooling会导致边界模糊  

Secondly, CNNs lack smoothness constraints that encourage label agreement between similar pixels, and spatial and appearance consistency of the
labelling output. Lack of such smoothness constraints can result in poor object delineation and small spurious regions in the segmentation output [57, 56, 30, 37].  

> **[success]**  
spurious：假的  
CNN的缺点1：缺少“smoothness constraints”会导致边界不连续。  

On a separate track to the progress of deep learning techniques, probabilistic graphical models have been developed as effective methods to enhance the accuracy of pixellevel labelling tasks.   

> **[success]** probabilistic graphical models：概率图模型  

In particular, Markov Random Fields (MRFs) and its variant Conditional Random Fields (CRFs) have observed widespread success in this area [30, 27] and have become one of the most successful graphical models used in computer vision. The key idea of CRF inference for semantic labelling is to **formulate the label assignment problem as a probabilistic inference problem that incorporates assumptions such as the label agreement between similar pixels**.   

> **[success]** CRF可以弥补CNN的不足  

CRF inference is able to refine weak and coarse pixel-level label predictions to produce sharp boundaries and fine-grained segmentations. Therefore, intuitively, CRFs can be used to overcome the drawbacks in utilizing CNNs for pixel-level labelling tasks.

One way to utilize CRFs to improve the semantic labelling results produced by a CNN is to **apply CRF inference as a post-processing step disconnected from the training of the CNN** [9]. Arguably, this does not fully harness the strength of CRFs since it is not integrated with the deep network – the deep network cannot adapt its weights to the CRF behaviour during the training phase.  

> **[success]**  
用CRF来弥补CNN的不足，方法一：  
把CRF作为与CNN无关的post-processing  
缺点：没有充分利用CRF的能力  

In this paper, we propose an end-to-end deep learning solution for the pixel-level semantic image segmentation problem. Our formulation combines the strengths of both CNNs and CRF based graphical models in one unified framework. More specifically, we **formulate mean-field inference of dense CRF with Gaussian pairwise potentials as a Recurrent Neural Network (RNN)** which can refine coarse outputs from a traditional CNN in the forward pass, while passing error differentials back to the CNN during training. Importantly, with our formulation, the whole deep network, which comprises a traditional CNN and an RNN for CRF inference, can be trained end-to-end utilizing the usual back-propagation algorithm.

> **[success]**   
本文的方法：  
用RNN代表CRF中的Gaussian pairwise potentials  
优点：  
（1）refine coarse outputs from a traditional CNN in the forward pass  
（2）passing error differentials back to the CNN  
这就是所谓的“end-to-end”  

Arguably, when properly trained, the proposed network should outperform a system where CRF inference is applied as a post-processing method on independent pixel-level predictions produced by a pre-trained CNN.   

> **[success]** Arguably：可认证地  

Our experimental evaluation confirms that this indeed is the case.

