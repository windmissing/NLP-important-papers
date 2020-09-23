# Abstract

Pixel-level labelling tasks, such as semantic segmentation, play a central role in **image understanding**.  

> **[info]**  
labelling task：标签任务  
semantic segmentation：语义切分  

Recent approaches have attempted to harness the capabilities of deep learning techniques for image recognition to tackle pixel level labelling tasks.   

> **[info]** harness：利用  

One central issue in this methodology is **the limited capacity of deep learning techniques to delineate visual objects**.   

> **[info]** delineate：描绘  

To solve this problem, we introduce a new form of convolutional neural network that combines the strengths of Convolutional Neural Networks (CNNs)
and Conditional Random Fields (CRFs)-based probabilistic graphical modelling.   

> **[success]** CNN + CRF  

To this end, we formulate Conditional Random Fields with *Gaussian pairwise potentials and mean-field approximate inference* as Recurrent Neural
Networks.   

> **[warning]**  
to this end：为了这个目的  
[?] Gaussian pairwise potentials and mean-field approximate inference  

This network, called CRF-RNN, is then plugged in as a part of a CNN to obtain a deep network that has desirable properties of both CNNs and CRFs. Importantly, our system fully integrates CRF modelling with CNNs, making it possible to train the whole deep network **end-to-end** with the usual back-propagation algorithm, avoiding offline post-processing methods for object delineation.

We apply the proposed method to the problem of semantic image segmentation, obtaining top results on the challenging Pascal VOC 2012 segmentation benchmark.

