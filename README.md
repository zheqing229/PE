# 从 RoPE 到 NoPE：长上下文中的位置建模正在重新分工

当上下文长度从几千个 token 扩展到几十万，甚至更长时，语言模型遇到的瓶颈不只是显存和计算量,还有一个更基础的问题：**模型究竟应该如何理解“位置”？**

在 Transformer 中，注意力机制本身并不知道两个 token 的先后顺序。为了让模型区分“我爱你”和“你爱我”，研究者引入了各种位置编码，RoPE之前主要的研究可以大致区分成两类：绝对位置编码和相对位置编码。而 RoPE 凭借简洁、有效以及能融合相对位置编码和绝对位置编码的自然建模，成为大语言模型中的主流方案。

RoPE 的做法很优雅：根据 token 所处的位置，对 query 和 key 进行不同角度的旋转。这样一来，两个 token 的注意力分数就会显式依赖它们之间的相对距离。在训练长度以内，这种位置先验通常表现得非常好。

问题出现在更长的上下文中。

当模型被要求处理远超训练窗口的文本时，它面对的是训练阶段没有见过的旋转角度、频率组合和注意力模式。为了延长上下文，研究者不得不引入位置插值、频率缩放、YaRN 或其他复杂的外推方法。即使上下文窗口在形式上被扩展，模型对远距离信息的检索能力仍然可能下降。

这时，一个看似激进的想法重新进入了研究视野：

> **如果直接不显式注入位置编码，会发生什么？**

这就是 NoPE 的出发点。NoPE 不再直接向注意力分数中加入旋转位置，而是让模型通过 causal mask 和注意力长短窗口，自己学习隐式的位置关系。令人意外的是，在一些长度泛化任务中，没有显式位置编码的模型反而能够处理更长的序列。

但 NoPE 也不是一个简单的替代方案。它可能带来困惑度上升、短文本任务退化，以及有限的泛化长度。RoPE 和 NoPE 似乎分别擅长不同的事情：RoPE 提供稳定的短距离位置信息，而 NoPE 更有利于长距离位置理解。

于是，研究问题开始发生变化，进一步追问：**不同层、不同维度和不同注意力范围，是否应该使用不同的位置机制？**

这也引出了近年来逐渐出现的混合方案：让部分层继续使用 RoPE，部分层采用 NoPE；让 RoPE 层负责局部和近期信息，让 NoPE 层负责全局检索；甚至只在部分 attention head 或部分维度中保留旋转位置编码。

本文将从 RoPE 的工作原理和长上下文局限出发，介绍 NoPE 如何学习隐式位置关系，并进一步讨论 RoPE 与 NoPE 的混合设计，尝试回答一个更根本的问题：

> **长上下文模型需要的，究竟是一套统一的位置编码，还是一种分层的位置分工？**

## RoPE：用绝对形式表达相对位置的优雅设计

非常推荐阅读苏剑林苏神的博客：
- [让研究人员绞尽脑汁的Transformer位置编码](https://spaces.ac.cn/archives/8130) 这篇介绍了早期的绝对和相对位置编码，以及早期关于 RoPE 想法的出现，读完之后真的会由衷赞叹苏神的想法，简直是太美妙了。
- [Transformer升级之路：2、博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265) 这篇比较详细介绍了 RoPE 的原理。
- 以及 RoPE 的论文：
[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

简单说，RoPE 的核心思想是：**让 query 和 key 在旋转角度上不同，从而让注意力分数显式依赖相对位置。**

1. 从二维旋转开始

先把 query 或 key 的两个相邻维度看成一个二维向量。对于位置 $m$，RoPE 用旋转矩阵将它旋转 $m\theta$：

$$
R(m\theta)=
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta)\\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}.
$$

因此，位于位置 $m$ 的 query 和位置 $n$ 的 key 分别变为

$$
\tilde{q}_m=R(m\theta)q_m,\qquad
\tilde{k}_n=R(n\theta)k_n.
$$

这里的 $q_m,k_n$ 是内容向量，旋转角度则携带了它们各自的绝对位置信息。

2. 为什么最终表示的是相对位置

注意力分数由 query 和 key 的内积决定。加入 RoPE 后，有

$$
\begin{aligned}
\tilde{q}_m^{\mathsf T}\tilde{k}_n
&=(R(m\theta)q_m)^{\mathsf T}(R(n\theta)k_n)\\
&=q_m^{\mathsf T}R(m\theta)^{\mathsf T}R(n\theta)k_n\\
&=q_m^{\mathsf T}R((n-m)\theta)k_n.
\end{aligned}
$$

其中用到了旋转矩阵的两个性质：

$$
R(\alpha)^{\mathsf T}=R(-\alpha),\qquad
R(\alpha)R(\beta)=R(\alpha+\beta).
$$

可以看到，最终的注意力分数不再分别依赖 $m$ 和 $n$，而只依赖两者的相对距离 $n-m$。这正是 RoPE 巧妙之处：**用绝对位置对应的旋转，得到只依赖相对位置的内积。**

继续把上面的式子展开。

记 $k_n^\perp$ 为 $k_n$ 逆时针旋转 $90^\circ$ 后的向量（即 $(-k_2,k_1)$），利用 

$$
\begin{aligned}
R(\Delta)k
&=\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta)\\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}\begin{bmatrix} k_1\\k_2\end{bmatrix}\\
&=\begin{bmatrix}k_1\cos\Delta-k_2\sin\Delta\\k_1\sin\Delta+k_2\cos\Delta\end{bmatrix}\\
&=\begin{bmatrix} k_1 \\ k_2 \end{bmatrix}\cos\Delta + \begin{bmatrix} -k_2\\k_1\end{bmatrix}\sin\Delta\\
&= k\cos\Delta+k^\perp\sin\Delta
\end{aligned},
$$

可以得到

$$
\begin{aligned}
\langle \tilde q_m,\tilde k_n\rangle
&=q_m^{\mathsf T}R((n-m)\theta)k_n\\
&=q_m^{\mathsf T}\left (k_n\cos((n-m)\theta)+k_n^\perp\sin((n-m)\theta)\right )\\
&=\underbrace{\langle q_m,k_n\rangle}_{\text{内容相似度}}\cos\bigl((n-m)\theta\bigr)
+\underbrace{\langle q_m,k_n^\perp\rangle}_{\text{正交分量}}\sin\bigl((n-m)\theta\bigr).
\end{aligned}
$$

这样就一目了然：

- 当 $m=n$ 时，$\cos 0=1,\ \sin 0=0$，分数退化为纯内容相似度，位置不干扰内容匹配；
- 距离拉开时，$\cos$ 项给内容相似度乘上一个随距离起伏的权重，$\sin$ 项掺入一个与内容正交的分量；
- 所以 RoPE 没有把位置加到分数上，而是**用相对距离去调制内容匹配的结果**——位置是乘性门控，不是加性偏置。

3. 扩展到高维向量

实际模型中的 head dimension 通常为偶数 $d$。RoPE 将向量按两维一组，给第 $i$ 组使用不同频率

$$
\theta_i=10000^{-2i/d},\qquad i=0,1,\ldots,\frac d2-1.
$$

整体旋转矩阵是一个分块对角矩阵：

$$
R_m=\mathrm{diag}\bigl(
R(m\theta_0),R(m\theta_1),\ldots,R(m\theta_{\frac{d}{2}-1})
\bigr).
$$

于是标准缩放点积注意力写成

$$
\mathrm{Attn}(m,n)
=\frac{(R_mq_m)^{\mathsf T}(R_nk_n)}{\sqrt d}
=\frac{q_m^{\mathsf T}R_{n-m}k_n}{\sqrt d}.
$$

随后再经过 causal mask 和 softmax 得到注意力权重。

4. 直观理解

**（a）复数视角：位置就是相位**

把每两个维度看成复平面上的一根指针 $z=x_1+\mathrm{i}x_2$。乘以 $e^{\mathrm{i}m\theta}$ 就是把指针逆时针转动 $m\theta$：

$$
z^{(m)}=z\ e^{\mathrm{i}m\theta}.
$$

于是**指针的绝对角度（相位）唯一地携带了绝对位置 $m$**。到这里为止，RoPE 注入的还是绝对信息。

真正的关键在内积的复数写法。实内积等于 $\langle \tilde q_m,\tilde k_n\rangle=\mathrm{Re}\bigl(z^{(m)}_q\,\overline{z^{(n)}_k}\bigr)$, 注意第二个因子要取共轭，共轭把相位取反：$e^{\mathrm{i}n\theta}\to e^{-\mathrm{i}n\theta}$。因此

$$
z^{(m)}_q\,\overline{z^{(n)}_k}
=\bigl(z_q\overline{z_k}\bigr)\,e^{\mathrm{i}m\theta}e^{-\mathrm{i}n\theta}
=\bigl(z_q\overline{z_k}\bigr)\,e^{\mathrm{i}(m-n)\theta}.
$$

两个绝对相位 $m\theta$ 与 $n\theta$ 在相乘时只剩下相位差 $(m-n)\theta$，实现了"绝对进、相对出"。

**（b）几何视角：只转弯，不改长度**

旋转是刚体运动，模长一致：$\lVert R_mx\rVert_2=\lVert x\rVert_2$。 RoPE 不会像加性位置编码那样扰动 query/key 的模长与数值分布，它只改变向量的朝向，而朝向正是内积所度量的东西。

**（c）多频率**

$d/2$ 组维度就是 $d/2$ 根转速不同的指针（$\theta_i=10000^{-2i/d}$）：

- **高频指针**转得快，相邻位置也有明显相位差，分辨率高、能精细区分近邻；但转过一整圈后相位开始"复用"，长距离上会出现周期性混叠。
- **低频指针**转得慢，在很长距离上都近似单调变化，能覆盖长程依赖，但近处区分度低。

但是问题也在这里埋下了：这既是 RoPE 表达力的来源，也是后文长上下文振荡与混叠的根源。

**（d）为什么 value 不需要旋转**

位置信息的目标只是决定"关注谁"，而这个决定已经完整编码在注意力权重 $\alpha_{mn}$ 里；输出 $\sum_n\alpha_{mn}v_n$ 通过权重就已经带上了相对位置。若再旋转 value，一是位置信息被重复注入，二是会改变输出表征空间的朝向，与残差、FFN 及后续层所期望的分布不一致。

## 重新审视 RoPE：怎么失灵了？

当我们重新审视 RoPE，会发现其位置信息依赖于频率 $\theta_i$，训练时模型只见过特定范围内的位置索引（如 4k）。而当推理长度超过训练长度时，高频维度的旋转角度会进入模型从未见过的相位区间，导致注意力分数出现剧烈震荡或完全崩溃。这就是长度泛化的问题，在更长的上下文中必须依赖额外的插值算法（如 YaRN）来强行压缩频率空间，这本质上是一种有损的修补，且需要重新微调或校准。

此外，论文 [RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably](./2605.15514v1.pdf)（Du et al., 2026，arXiv v1）提出了一个更强的观点：**当上下文不断增长时，RoPE 可能同时失去可靠区分位置和稳定区分 token 的能力；只调整 RoPE base，无法同时解决这两个问题。**

### 从旋转矩阵到振荡信号

回顾刚刚给出的二维 RoPE 展开：
$\langle \tilde q_m,\tilde k_n\rangle=\langle q_m,k_n\rangle\cos\bigl((n-m)\theta\bigr)
+\langle q_m,k_n^\perp\rangle\sin\bigl((n-m)\theta\bigr)$.

把 $\langle q_m,k_n\rangle$ 和 $\langle q_m,k_n^\perp\rangle$ 分别记作 $A$ 和 $B$，则有

$$
\begin{aligned}
\langle \tilde q_m,\tilde k_n\rangle
&=A\cos\bigl((n-m)\theta\bigr)+B\sin\bigl((n-m)\theta\bigr)\\
&=C\cos\bigl((n-m)\theta+\phi\bigr)\\
C&=\sqrt{A^2+B^2}
\end{aligned}.
$$

其中 $A$, $B$, $\phi$ 都是只和 query 和 key 的内容有关，而和位置无关的参数。

我们把二维的公式拓展到 $d=2h$ 维，把 query 与 key 的相对距离记为 $r$。每两个维度合并后，RoPE 作用下的未归一化注意力分数就可以写成

$$
\mathrm{Attn}_{q,k}(r)
=\sum_{n=0}^{h-1}C_n\cos(r\theta_n+\phi_n).
$$

$\theta_n=B^{-n/h}$ 由 RoPE base $B$ 决定。也就是说，RoPE attention score 本质上是许多不同频率余弦波的叠加。

接下来我们尝试把这些余弦波按照在上下文长度 $L$ 内转得快还是慢分成两堆。

对第 $n$ 个分量，随相对距离 $r$ 增大，相位从 $\phi_n$ 变成 $r\theta_n + \phi_n$. 可以想成单位圆上一个点：每拉开 1 个 token，就多转 $\theta_n$ 弧度。

- 如果存在 $r < L$ 使得 $r\theta_n \gtrsim 2\pi$，则这个分量**至少转完一圈**, $\cos$ 会上下摆动很多次。
- 而如果所有 $r$ 都有 $r\theta_n \ll \pi$，则这个分量**只扫过一小段圆弧**，$\cos$ 值几乎单调缓降。

在上下文长度 $L$ 内刚好转满一圈的临界条件是

$$
L \cdot \theta_n \approx 2\pi
\quad\Longrightarrow\quad
L \cdot B^{-n/h} \approx 2\pi
\Longrightarrow
n \approx h \log_B\!\frac{L}{2\pi}
= \Theta(h\log_B L).
$$

记$\lambda(L)=\Theta(h\log_B L)$：

- $ n \ll \lambda(L)$：转得比一圈还多 → 高频项旋转快，可以区分相邻位置，但也会使分数随距离剧烈振荡。
- $n \gg \lambda(L)$：转一个非常小的角度 → 低频项旋转慢，形成偏好近距离 token 的衰减趋势，同时让 token 相关性的排序更加稳定。

论文的关键分析是：当距离 $r$ 从一个足够长的区间中采样时，可以借助中心极限定理，把这个余弦和近似看成正态随机变量

$$
\widetilde{\mathrm{Attn}}_{q,k}\sim
\mathcal N\left(\mu_L,\sigma_L^2\right),
$$

其中均值 $\mu_L$ 主要由尚未充分旋转的低频项决定，方差 $\sigma_L^2$ 主要由不断振荡的高频项决定。随着上下文长度 $L$ 增大，低频项越来越少、振荡项越来越多，因此整体表现为**均值衰减、方差增大**，注意力分数也越来越难预测。

推导过程大致概括如下，严谨的证明推荐去看原论文。

当 $q,k$ 固定后，$C_n,\phi_n$ 就定了，$\mathrm{Attn}(r)$ 只是 $q$ 和 $k$ 的相对距离 $r$ 的函数：

$$
\mathrm{Attn}(r)=\sum_{n=0}^{h-1} \underbrace{C_n\cos(r\theta_n+\phi_n)}_{\Psi_n(r)}.
$$

我们尝试描述距离 $r$ 的分布是是**在上下文长度 $[0,L)$ 上均匀分布**的。那么每个 $\Psi_n$ 就是一个随机变量，从而 $\mathrm{Attn}$ 也是随机变量。所以问题变成：$\mathrm{Attn}$ 的分布长什么样？

我们先看**高频项**，高频项给我一种“洗匀”的感觉。当 $n \ll \lambda(L)$，$\theta_n$ 较大，当 $r$ 跑遍 $[0,L)$ 时，$\alpha(r) = r\theta_n + \phi_n$ 会绕单位圆转很多很多圈。于是 $\cos\alpha(m)$ 把 $[-1,1]$ 上每个取值都扫过很多次。也就是说，从随机抽 $r$ 的角度看，相位 $\alpha$ 近似**均匀分布在 $[0,2\pi)$**，概率密度 $p(\alpha)\approx\frac{1}{2\pi},\quad 0\le \alpha < 2\pi$

 
可以得到：

$$
\mathbb{E}[\cos\alpha]\approx\int_{0}^{2\pi}\cos\alpha \cdot \frac{1}{2\pi}\,d\alpha=0\\
\begin{aligned}
\mathbb{E}\left[\cos^2\alpha\right]
&\approx\int_{0}^{2\pi}\cos^2\alpha \cdot \frac{1}{2\pi}\,d\alpha \\
&=\frac{1}{2\pi}\int_{0}^{2\pi}\frac{1+\cos2\alpha}{2}\,d\alpha \\
&=\frac{1}{4\pi}\left(2\pi+\left.\frac12\sin2\alpha\right|_{0}^{2\pi}\right) \\
&=\frac12
\end{aligned}
$$ 

所以每个高频项

$$
\mathbb{E}[\Psi_n]\approx 0,\qquad
\mathrm{Var}(\Psi_n)\approx \frac{C_n^2}{2}.
$$

**转得越快，越像噪声：对均值几乎没有贡献，只贡献波动。**

论文里用 Dirichlet Kernel 证明：$r$ 在长区间上均匀时，$\mathbb{E}[\Psi_n]=O(\frac{2C_n}{L\theta_n})\to 0$，$\mathbb{E}[\Psi_n^2]\to a_n^2/2$。


再看**低频项**，低频项给我一种“冰冻”的感觉。当 $n \gg \lambda(L)$，$\theta_n$ 很小，$r\in[0,L)$ 时 $r\theta_n$ 扫过的角度非常小(O(1))：

$$
r\theta_n+\phi_n \approx \phi_n,\qquad
\cos(r\theta_n+\phi_n)\approx \cos\phi_n.
$$

也就是说这个余弦几乎不随 $m$ 变，像一个常数：

$$
\mathbb{E}[\Psi_n]\approx C_n\cos\phi_n,\qquad
\mathrm{Var}(\Psi_n)\approx 0.
$$

**转得越慢，越像常数：抬高或压低整体水平（均值），不制造波动。**

此时我们注意到：

**(a) 各频率项近似独立。**
$\theta=B^{-1/h}$ 时，序列 $1,\theta,\theta^2,\ldots$ 在有理数集 $\mathbb{Q}$ 上几乎线性无关（Weyl 等分布准则），$\theta_n=B^{n(-1/h)}=\theta^n$，于是 $m\theta_n \bmod 2\pi$ 对不同 $n$ 近似独立均匀。论文估计协方差 $\mathrm{Cov}[\Psi_n\Psi_p]=O(\frac{C_nC_p}{M\theta_n})$，高频时极小可忽略。

**(b) 中心极限定理。**
高频项 $\sum_{n<\lambda(M)} C_n\cos(r\theta_n+\phi_n)$ 是许多独立、均值为 0、无单项主导的随机项之和，渐近正态。Berry–Esseen 给出误差 $O(1/\sqrt{\lambda(L)})$。

再加上低频项近似常数，整体就是：

$$
\boxed{
\widetilde{\mathrm{Attn}} \sim \mathcal{N}(\mu_L,\ \sigma_L^2),\quad
\mu_L \approx \sum_{n\ge\lambda(M)} C_n\cos\phi_n,\quad
\sigma_L^2 \approx \frac12\sum_{n<\lambda(L)} C_n^2
}
$$

均值由**低频**决定，方差由**高频**决定。

论文给出了一个正态分布拟合注意力分数的图.

![](NormalApprox.png)

### 四种失败模式

| 现象 | 数学表现 | 直观含义 |
|---|---|---|
| 位置反转 | $r_1<r_2$，但 $\mathrm{Attn}(r_1)<\mathrm{Attn}(r_2)$ | 同一个 token 放得更远，注意力分数反而更高，局部性偏置失效 |
| 位置混叠 | $r_1\ne r_2$，但 $\mathrm{Attn}(r_1)=\mathrm{Attn}(r_2)$ | 不同位置产生相同分数，模型无法仅凭该分数区分位置 |
| token 反转 | $\mathrm{Attn}_1(0)>\mathrm{Attn}_2(0)$，但某个 $r$ 上 $\mathrm{Attn}_1(r)<\mathrm{Attn}_2(r)$ | 两个 token 原本的相关性排序被距离反转 |
| token 混叠 | $k_1\ne k_2$，但 $\mathrm{Attn}_1(r)=\mathrm{Attn}_2(r)$ | 不同 token 在某个位置得到相同分数 |

当上下文 $L$ 进一步变长时，$\lambda(L)=\Theta(h\log_B L)$ 变大，更多项从低频常数被划进高频噪声：$\mu$ 稳定项变少，$\sigma^2$ 变大，论文证明了两种“反转”的概率下界最终都可趋近 $0.5$，即排序接近抛硬币猜测；两种“混叠”还会受到 BF16 等有限数值精度的放大。

### RoPE base 不是免费的午餐

增大 base $B$ 会让各维度旋转得更慢。这有助于稳定不同 token 的相关性排序，缓解 token 反转和 token 混叠；但不同位置之间的相位差也会变小，从而加重位置反转和位置混叠。较大的 base 更有利于区分 token，却更不利于区分位置；较小的 base 则相反。

因此，位置插值、NTK scaling 或单纯增大 base 更像是在两类误差之间重新分配预算，而不是从根本上消除长上下文问题。
