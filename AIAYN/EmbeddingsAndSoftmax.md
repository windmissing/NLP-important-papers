Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by $\sqrt {d_{model}}$.    

> **[warning]**  
[?] 两个输入的embedding共享参数还能理解，为什么和pre-softmax linear transformation也能共享参数呢？  
[?] embedding是怎么做的？  




