# Related Work

In this section we review approaches that make use of deep learning and CNNs for low-level computer vision tasks, with a focus on semantic image segmentation. A wide variety of approaches have been proposed to tackle the semantic image segmentation task using deep learning. These approaches can be categorized into two main strategies.  

The first strategy is based on **utilizing separate mechanisms for feature extraction, and image segmentation exploiting the edges of the image** [2, 36]. One representative instance of this scheme is the application of a CNN for the extraction of meaningful features, and using superpixels to account for the structural pattern of the image.   

> **[success]**  
用CNN做low-level的计算机视觉任务的策略一：
（1）用CNN提取特征  
（2）用superpixels计算structural pattern   
[?] superpixels  

Two representative examples are [18, 36], where the authors first obtained superpixels from the image and then used a feature extraction process on each of them. The main disadvantage of this strategy is that errors in the initial proposals (e.g: super-pixels) may lead to poor predictions, no matter how good the feature extraction process is.   

> **[success]**  
缺点：第（1）（2）两步分开做，（1）做得再好，如果（2）没做好，结果也不会好。  

Pinheiro and Collobert [44] employed an RNN to model the spatial dependencies during scene parsing. In contrast to their approach, we show that a typical graphical model such as a CRF can be formulated as an RNN to form a part of a deep network, to perform end-to-end training combined with a CNN.  

> **[success]**  
解决方法：用RNN实现CRF  

The second strategy is to directly learn a nonlinear model from the images to the label map. This, for example, was shown in [16], where the authors replaced the last fully connected layers of a CNN by convolutional layers to keep spatial information.   

> **[success]**  
用CNN做low-level的计算机视觉任务的策略二：directly learn a nonlinear model  
具体方法一：CNN的最后一层用卷积层代替FC，即全部都是卷积层    

An important contribution in this direction is [35], where Long et al. used the concept of fully convolutional networks, and the notion that top layers obtain meaningful features for object recognition whereas low layers keep information about the structure of the image, such as edges. In their work, connections from early layers to later layers were used to combine these cues.  

> **[success]**  
全卷积层的意义：上层提取对象识别的特征，下层提取图像结构的特征。  

Bell et al. [5] and Chen et al. [9, 39] used a CRF to refine segmentation results obtained from a CNN. Bell et al. focused on material recognition and segmentation, whereas Chen et al. reported very significant improvements on semantic image segmentation.   

> **[success]**  
具体方法二：CNN之后再使用CRF做refine  
这种方法比较好。  

In contrast to these works, which employed CRF inference as a standalone post-processing step disconnected from the CNN training, our approach is an end-to-end trainable network that jointly learns the parameters of the CNN
and the CRF in one unified deep network.  

> **[success]**   
具体方法三（本文方法）：把CNN和CRF结合到同一个网络中  

Works that **use neural networks to predict structured output** are found in different domains.   

> **[success]** 用NN预测结构化输出    

For example, Do et al. [13] proposed an approach to combine deep neural networks and Markov networks for sequence labeling tasks. Another domain which benefits from the combination of CNNs and structured loss is handwriting recognition. In[6], the authors combined a CNN with Hidden Markov Models for that purpose, whereas more recently, Peng et al. [43] used a modified version of CRFs. Related to this line of works, in [24] a joint CNN and CRF model was used for text recognition on natural images. Tompson et al. [55] showed the use of joint training of a CNN and an MRF for human pose estimation, while Chen et al. [10] focused on the image classification problem with a similar approach. Another prominent work is [20], in which the authors express Deformable Part Models, a kind of MRF, as a layer in a neural network. In our approach we cast a different graphical model as a neural network layer.

A number of approaches have been proposed for **automatic learning of graphical model parameters and joint training of classifiers and graphical models**. Barbu et al. [4] proposed a joint training of a MRF/CRF model together with an inference algorithm in their Active Random Field approach. Domke [14] advocated back-propagation based parameter optimization in graphical models when approximate inference methods such as mean-field and belief propagation are used. This idea was utilized in [26], where a binary dense CRF was used for human pose estimation. Similarly, Ross et al. [45] and Stoyanov et al. [52] showed how back-propagation through belief propagation can be used to optimize model parameters. Ross et al. [45], in particular, proposed an approach based on learning messages. Many of these ideas can be traced back to [53], which proposed unrolling message passing algorithms as simpler operations that could be performed within a CNN. In a different setup, Krähenbühl and Koltun [28] demonstrated automatic pa-
rameter tuning of dense CRF when a modified mean-field algorithm is used for inference. An alternative inference approach for dense CRF, not based on mean-field, is proposed in [58].

In contrast to the works described above, our approach shows that it is possible to **formulate dense CRF as an RNN** so that one can form an end-to-end trainable system for semantic image segmentation which combines the strengths of deep learning and graphical modelling. The concurrent and independent work [47] explores a similar joint training approach for semantic segmentation.

> **[warning]** [?] dense CRF  

