Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively. 

![](/AIAYN/assets/2.png)  

> **[success]**  
Transformer摒弃了recurrent结构，这不代表在Transformer中每个时间步之间没有关系。实际上在Transformer中，还是存在从当前时间步到下一个时间步的数据流动。下一个时间步使用了当时步的输出。  