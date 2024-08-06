
# BPNetwork

一个简单数字识别的反向传播（Back Propagation, BP）神经网络。神经网络，简单讲就是~~拟合~~（找规律）。而反向传播是指，通过残差结果反向调整参数。


## 网络结构

- ### 神经元结构 

一个神经元从多个输入端感受信号刺激，通过线性系统和非线性的激活函数输出一个输出。假设信号的输入矢量是$\mathbf{X}_{in}\in\mathbb{R}^n$，其中$n$是输入信号的维度，若该神经元的参数特性有权重矢量$\mathbf{W}\in\mathbb{R}^n$，偏置量$b$，则该神经元的信号输入$\mathbf{X}_{in}$与输出$x_{out}$满足
$$
\begin{align}
x_{out} &= f( \mathbf{W}^T\cdot\mathbf{X}_{in}+b) 
\end{align}
$$

- ### 层结构

BP神经网络有三种层，分别是**输入层（Input）**、**隐含层（Hidden）**、**输出层（Output）**。隐含层可以有若干个。有关研究表明，一个隐含层的神经网络，只要神经元足够多，就可以以任意精度逼近一个非线性函数。因此，通常采用含有一个隐层的三层多输入单输出的BP神经网络建立预测模型。若是神经元数目过少，则会影响网络性能，达不到预期效果；若隐层神经元数目过多，会加大网络计算量并容易产生过度拟合问题。


## 原理推导

### 前向传播

#### 信号输入


将图像的二值化矩阵从$(m , n)$展平成$(m*n, 1)$作为输入层，则输入层的维度为$m*n$。

令$\mathbf{W_i}\in\mathbb{R}^{m*n}$为的权重矩阵，$\mathbf{B_i}\in\mathbb{R}^{m*n}$。

#### 

### 激活函数

选取**ReLU**函数作为激活函数。ReLU函数可以很有效解决[梯度消失](https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E9%97%AE%E9%A2%98)的问题。

$$

\begin{align}
ReLU(x)&=\left\{ 
    \begin{array}{rcl}
    0 && {x\leq0}\\
    x && {x>0}
    \end{array}
\right.
\end{align}

$$



$$

\begin{align}
\mathcal{f}(\mathbf{X})&=\mathbf{W}_i ~\cdot~\mathbf{X}~+~\mathbf{B} 
\end{align}
$$


$$



$$