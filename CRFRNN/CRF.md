# Conditional Random Fields

In this section we provide a brief overview of CRF for pixel-wise labelling and introduce the notation used in the paper. A CRF, used in the context of pixel-wise label prediction, **models pixel labels as random variables that form a MRF when conditioned upon a global observation. The global observation is usually taken to be the image.**  

> **[success]**  
CRF用于像素级标签任务要做的事情：  
（1）把每个像素的label作为随机变量  
（2）基于global observation构建MRF  
（3）使用image什么意思global observation  

Let $X_i$ be the random variable associated to pixel i, which represents the label assigned to the pixel i and can take any value from a pre-defined set of labels $\Bbb L = {l_1 ,l_2 ,...,l_L}$. Let X be the vector formed by the random variables $X_1 ,X_2 ,...,X_N$ , where N is the number of pixels in the image. Given a graph G = (V,E), where $V = {X_1 ,X_2 ,...,X_N }$, and a global observation (image) I, the pair (I,X) can be modelled as a CRF characterized by a Gibbs distribution of the form $P(X = x|I) = \frac{1}{Z(I)}\exp(-E(x|I))$. Here E(x) is called the energy of the configuration x ∈ L N and Z(I) is the partition function [31]. From now on, we drop the conditioning on I in the notation for convenience.  

> **[success]**  
i：pixel ID  
$X_i$：与i相关的随机变量，代表i的label  
$\Bbb L$：label的集合  
$L$：共L个label  
$N$：像素点个数  
[?] X和V是什么关系？貌似都是Xi的集合？  
[?] Gibbs分布？  

In the fully connected pairwise CRF model of [27], the energy of a label assignment x is given by:  

> **[warning]** [?] fully connected pairwise CRF model？  

$$
\begin{aligned}
E(x) = \sum_i \psi_u(x_i) + \sum_{i<j} \psi_p (x_i ,x_j ),  && (1)
\end{aligned}
$$

where the unary energy components ψ u (x i ) measure the inverse likelihood (and therefore, the cost) of the pixel i taking the label x i , and pairwise energy components ψ p (x i ,x j ) measure the cost of assigning labels x i ,x j to pixels i,j simultaneously.   

> **[success]**  
unary：一元的  
[?] inverse likelihood   

In our model, unary energies are obtained from a CNN, which, roughly speaking, predicts labels for pixels without considering the smoothness and the consistency of the label assignments. The pairwise energies provide an image data-dependent smoothing term that encourages assigning similar labels to pixels with similar properties.   

> **[success]**  
上面公式（1）的第一项，代表预测i时不考虑与附近点之间的关系：（1）平滑性（2）一致性  
第二项代表同时为i, j打标签的cost。平滑性：有相似属性的像素得到相同的label  

As was done in [27], we model pairwise potentials as weighted Gaussians:  

$$
\begin{aligned}
\psi_p (x_i ,x_j ) = \mu(x_i ,x_j)\sum^M_{m-1} w^{(m)} k^{(m)}_G
(f i ,f j ),   &&   (2)
\end{aligned}
$$

where each $k^{(m)}_G$ for m = 1,...,M, is a Gaussian kernel applied on feature vectors. The feature vector of pixel i, denoted by fi , is derived from image features such as spatial location and RGB values [27]. We use the same features as in [27]. The function µ(.,.), called the label compatibility
function, captures the compatibility between different pairs of labels as the name implies.  

> **[success]**  
weighted Gaussians：不同Gaussians kernel的加权和  
$w^{(m)}$：权  
$k^{(m)}_G$：Gaussians kernel  
f：特征  
$\mu(x_i ,x_j)$：标签兼容性函数，captures the compatibility between different pairs of labels  

Minimizing the above CRF energy E(x) yields the most probable label assignment x for the given image. Since this exact minimization is intractable, a mean-field approximation to the CRF distribution is used for approximate maximum posterior marginal inference.   

> **[warning]**  
intractable：棘手的  
[?] mean-field approximation：平均场近似  
[?] posterior marginal inference?  

It consists in approximating the CRF distribution P(X) by a simpler distribution Q(X), which can be written as the product of independent
marginal distributions, i.e., $Q(X) = \prod_i Q_i(X_i)$. The steps
of the iterative algorithm for approximate mean-field inference and its reformulation as an RNN are discussed next.  

> **[success]**   
reformulation：再形成    
用近似的Q(X)代替P(X)，Q(X)为独立边缘分布，P(X)为CRF分布  

