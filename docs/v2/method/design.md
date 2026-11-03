# Graph-Level GDA：面向目标分类梯度的源监督联合校准

**暂定方法名称：Target-Queried Gradient Learning，TQG。修订版 v2。**

本文保留“目标图提出计算查询，源标签提供监督反馈，GNN 执行更新”的主线，将查询对象收束为当前目标分类梯度，并重构源监督读取与更新执行。训练阶段可以访问有标签源图和无标签目标图，预测对象始终为完整图。方法使用一个可训练 GNN、一项源分类主损失、一项网络辅助损失，以及一个停止梯度的内部校准问题。

本文只包含 Motivation、Research Question 和方法设计。它是一份待验证的研究候选方案，不将新的校准目标等同于已经证明的迁移效果或方法新颖性。

## 1. Motivation

### 1.1 目标图能够确定分类梯度的作用空间，但不能独立确定其中的监督系数

图级无监督域适配需要利用源图标签改善目标图分类。图分布变化会改变同一 GNN 在不同完整图上的计算响应，因此源域优化形成的平均更新，未必对应当前目标图需要的更新。

设模型参数为 $\theta$，完整图 $G$ 的类别 logits 为 $z_\theta(G)$，类别概率为 $p_\theta(G)$，输出对参数的 Jacobian 为 $J_\theta(G)$。交叉熵分类梯度为：

$$
g_T(\theta)
=\mathbb E_T\left[J_\theta(G)^\top\bigl(p_\theta(G)-Y\bigr)\right].
$$

这里存在一个可直接利用的结构：目标标签决定梯度的组合系数，但目标输入和当前模型已经确定了 Jacobian。对于一个给定的目标批次，无论其未知标签是什么，分类梯度都落在该批次 Jacobian 转置的像空间内。

因此，目标域提供的信息比“某些随机方向会改变输出”更具体：**它能够确定本轮分类梯度可能作用的完整参数空间。真正需要从源标签中估计的是这个空间里的监督系数。**

这一观察适用于完整图分类。它不要求源图与目标图来自同一个实例，不要求节点对应，也不以某一种图规模划分作为研究动机。

### 1.2 源监督的读取目标，应当由目标分类梯度决定

在源域拟合输出残差，再将拟合规则用于目标域，包含两项额外要求：源端的残差读出规则能够跨域使用，以及输出残差误差经过目标 Jacobian 后仍能形成适当的参数更新。仅仅共享探测方向和响应尺度，不能保证这两项要求成立。

本设计直接针对一个更集中的对象：

> **源标签应该怎样被组合，才能估计当前目标分类梯度所需要的监督量？**

方法不先要求恢复每张目标图的全部残差，而是让目标 Jacobian 与目标图分布共同定义一组梯度查询。源图提供这些查询的有标签观测，读取权重按照查询造成的参数梯度误差联合确定。

这种设计也改变了源训练驻点的作用。源图的分类梯度在源分布上相互抵消，不意味着这些图对目标梯度查询提供的监督也必然抵消。不过，如果逐图源残差确实全部为零，方法仍然无法凭空获得新的标签信息。

### 1.3 对源监督覆盖不足的目标计算，应当保留当前模型作为校正基准

目标分类梯度由一个已知的预测项和一个未知的标签项组成。已知预测项可以直接在目标图上计算，无需从源域估计。

若独立估计未知标签项，当源监督完全无法约束它时，估计可能趋于零，而已知预测项仍然保留。这会产生一个没有充分监督依据的目标更新。

本设计使用当前模型预测作为已知基准，源标签只估计相对于这一基准的校正。源监督无法支持查询时，额外反馈趋于零；有支持时，反馈由真实源标签对当前预测的差异提供。这一结构不将当前预测当作正确目标标签。

最终，目标图决定“需要读取什么监督”，源标签决定“读出的监督值”，参数坐标损失执行相同空间中的梯度。

### 1.4 技术依据与贡献定位

[Residual-as-Teacher，2026 公开稿](https://arxiv.org/abs/2603.25466) 已经研究通过源残差估计驱动目标输入上的函数学习。因此，使用 $Y-p$、停止梯度、核回归或没有显式神经 teacher，都不作为本设计的独立创新主张。

[TILT，2026](https://arxiv.org/abs/2605.14280) 研究目标输入如何通过训练目标改变源监督的拟合方式。[Optimal Algorithms in Linear Regression under Covariate Shift，2025 公开稿](https://arxiv.org/abs/2502.09047) 研究利用源目标信息构造凸优化下的估计变换。它们说明目标相关估计与 minimax 优化已有明确基础，不能仅凭采用这些术语主张新颖性。

本设计的贡献候选集中在同一个操作上：**在 GNN 当前目标分类梯度的完整作用空间内，联合确定源监督的读取权重，使读取准则对应参数方向上的误差，并在同一坐标中执行反馈。**

具体改动包括完整目标切空间、同时关注平均与最大方向误差的校准目标，以及带当前预测基准的参数梯度反馈。核方法、矩阵谱和控制变量分别是实现这些操作的工具。该组合是否具有充分的方法新意，仍需与广义核估计、梯度泛函估计和已有域适配方法比较，不能由流程完整性直接推出。

## 2. Research Question

### 2.1 主要研究问题

> **如何利用无标签目标图确定分类梯度所需的监督查询，并从源标签中联合估计这些查询的答案，使得到的反馈能够直接更新目标 GNN？**

英文表述：

> **How can unlabeled target graphs determine the supervision queries needed for their classification gradients, and how can source labels answer these queries through a shared, gradient-aware calibration mechanism?**

问题的重要性在于，图域适配最终需要改变目标图上的分类行为。训练信号的目标应当与这一参数更新任务一致，而不是只在源域完成一个代理拟合，再默认其输出能够跨域成为正确更新。

### 2.2 三个衔接的设计问题

**目标图如何定义查询空间？** 使用当前目标分类 Jacobian 的有效行空间，覆盖本轮目标分类梯度可能出现的参数方向。

**源标签如何回答查询？** 在完整图的计算签名上定义可迁移任务函数族，并按照目标梯度泛函的平均与最大方向误差，联合求解源监督读出权重。

**查询答案如何执行？** 将读出的低维系数直接转成参数梯度，以参数坐标上的辅助损失实现，不再经过一轮目标输出残差回归。

### 2.3 技术主线

$$
\boxed{
\text{目标分类梯度空间}
\;\longrightarrow\;
\text{目标梯度查询的源监督校准}
\;\longrightarrow\;
\text{同一参数坐标中的单步更新}
}
$$

目标图同时决定查询空间和读取准则。源标签只在读取值时进入方法。源分类主损失继续提供直接监督，目标反馈用于增加当前目标计算所要求的更新分量。

## 3. 方法设计

### 3.1 输入、输出与基本设定

输入为有标签源图集合和无标签目标图集合：

$$
\mathcal D_S=\{(G_i^S,Y_i^S)\}_{i=1}^{n_S},
\qquad
\mathcal D_T=\{G_j^T\}_{j=1}^{n_T}.
$$

每个 $G$ 是一张完整图，可以具有不同的节点数、边数、节点特征和边属性。源目标使用兼容的输入接口，共享 $C$ 个分类类别。$Y\in\{0,1\}^C$ 是 one-hot 标签。二分类也使用两个 logits。

训练输出为适配后的参数 $\theta^\star$。推断时只保留 GNN：

$$
p_{\theta^\star}(G)=\operatorname{softmax}\bigl(z_{\theta^\star}(G)\bigr),
\qquad
\widehat y(G)=\arg\max_c p_{\theta^\star,c}(G).
$$

方法属于 source-access graph-level UDA。目标标签不参与初始化、查询构造、校准求解或网络更新。

### 3.2 一个 GNN 与固定的类别对比坐标

模型由图编码器与线性分类头组成：

$$
h_\vartheta(G)=\operatorname{GraphEncoder}_\vartheta(G)\in\mathbb R^d,
\qquad
f_\theta(G)=U h_\vartheta(G)+b,
\qquad
\theta=(\vartheta,U,b)\in\mathbb R^P.
$$

这里 $P$ 为全部可训练参数的数量，包含编码器和分类头。后续梯度空间也使用这同一组参数。

对 logits 进行类别中心化：

$$
P_C=I_C-\frac1C\mathbf1_C\mathbf1_C^\top,
\qquad
z_\theta(G)=P_C f_\theta(G),
\qquad
p_\theta(G)=\operatorname{softmax}\bigl(z_\theta(G)\bigr).
$$

令 $q=C-1$。选择一个固定的正交类别对比基 $Q_C\in\mathbb R^{C\times q}$，满足：

$$
Q_C^\top Q_C=I_q,
\qquad
Q_CQ_C^\top=P_C,
\qquad
Q_C^\top\mathbf1_C=0.
$$

定义：

$$
\zeta_\theta(G)=Q_C^\top z_\theta(G)\in\mathbb R^q,
\qquad
\breve p_\theta(G)=Q_C^\top p_\theta(G),
\qquad
\breve Y=Q_C^\top Y.
$$

$\breve p$ 和 $\breve Y$ 是对比坐标，不是概率向量。分类仍使用完整的 $p_\theta$。该处理去掉所有类别共同平移的冗余方向，不涉及图分解。

### 3.3 源监督角色与初始化

保留两个互不重叠的源训练角色：

$$
\mathcal D_S=\mathcal D_S^{\mathrm{fit}}\mathbin{\dot\cup}\mathcal D_S^{\mathrm{qry}}.
$$

$\mathcal D_S^{\mathrm{fit}}$ 用于源模型初始化和适配中的源分类主损失，$\mathcal D_S^{\mathrm{qry}}$ 用于回答目标梯度查询。源查询池的具体大小不是方法定义的一部分。

先在 $\mathcal D_S^{\mathrm{fit}}$ 上优化：

$$
\mathcal L_{\mathrm{sup}}(\theta)
=-\frac1{|\mathcal B_S^{\mathrm{fit}}|}
\sum_{(G,Y)\in\mathcal B_S^{\mathrm{fit}}}
Y^\top\log p_\theta(G),
$$

得到初始模型 $\theta_0$。源查询标签随后通过反馈参与训练，因此查询池不是独立验证集。初始角色划分也不保证自适应复用后的统计独立性。

主版本使用普通分类 CE，估计对象是相应的目标 CE 梯度。若将目标梯度的估计准则改为类别加权 CE，应相应重写查询中的监督向量与目标梯度定义。仅改变源主损失的类别权重时，目标反馈仍可以估计普通目标 CE 梯度，但需要明确两项损失各自的优化职责。

### 3.4 每轮输入、快照和确定性计算

第 $t$ 轮读取源分类批次 $\mathcal B_{S,t}^{\mathrm{fit}}$、源查询批次 $\mathcal B_{S,t}^{\mathrm{qry}}$、无标签目标批次 $\mathcal B_{T,t}$。记查询图数为 $b_S$，目标图数为 $b_T$，并固定快照：

$$
\bar\theta_t=\theta_t.
$$

源查询批次至少包含两张图，以定义后续源侧尺度。它可以是完整查询池，也可以是预先约定的查询子批次。

本轮 Jacobian、图表示、核矩阵、校准权重和反馈均在同一快照上计算。关闭 dropout 的随机丢弃，固定 BatchNorm 等层的运行统计，同时保留正常的参数梯度。每次网络更新后重新构造查询，不复用旧参数位置上的反馈。

### 3.5 用目标 Jacobian 构造完整分类梯度空间

对任意完整图，定义对比 logits 的全参数 Jacobian：

$$
\widetilde J_t(G)
=\left.\frac{\partial\zeta_\theta(G)}{\partial\theta}\right|_{\theta=\bar\theta_t}
\in\mathbb R^{q\times P}.
$$

将目标批次的 Jacobian 堆叠为：

$$
\mathcal J_{T,t}
=\begin{bmatrix}
\widetilde J_t(G_1^T)\\
\vdots\\
\widetilde J_t(G_{b_T}^T)
\end{bmatrix}
\in\mathbb R^{b_Tq\times P}.
$$

通过薄 SVD 或秩揭示 QR，取得其行空间的正交参数基：

$$
V_t=\operatorname{orth}(\mathcal J_{T,t}^\top)
\in\mathbb R^{P\times r_t},
\qquad
V_t^\top V_t=I_{r_t},
\qquad
r_t\le\min\{P,b_T(C-1)\}.
$$

保留所有数值上有效的非零方向，数值零的判定使用与机器精度和最大奇异值相关的容差。该步骤不再固定抽取 $K$ 个随机方向，也不计算 odd/even 有限差分。

对当前目标批次，无论未知标签取何值，其真实分类梯度都位于 $\operatorname{col}(V_t)$。这一覆盖只针对当前批次。若为节省计算而主动截断非零奇异方向，应明确称为近似版本，不能继续声称完整覆盖。

对完整源图与目标图统一定义：

$$
A_t(G)=\widetilde J_t(G)V_t\in\mathbb R^{q\times r_t}.
$$

源端的 $A_t(G)$ 可以通过 JVP 计算，无需存储所有源图的完整参数 Jacobian。

若 $r_t=0$，本轮没有目标一阶分类作用方向，令目标反馈为零，仅执行源分类更新。

### 3.6 把目标分类梯度写成低维监督查询

为简化本轮记号，以下省略下标 $t$，记 $r=r_t$。若目标标签可见，其对比残差为：

$$
\widetilde r_j^T=Q_C^\top\bigl(Y_j^T-p_{\bar\theta}(G_j^T)\bigr).
$$

目标批次的分类梯度可写为：

$$
g_T^{\mathrm{lab}}
=-V d_T^{\mathrm{lab}},
\qquad
d_T^{\mathrm{lab}}
=\frac1{b_T}\sum_{j=1}^{b_T}A(G_j^T)^\top\widetilde r_j^T.
$$

这里的 $Y_j^T$ 只用于定义希望估计的量，算法不读取它。

对于一个 $q$ 维完整图函数 $u$，定义目标梯度查询算子：

$$
\mathcal T_T[u]
=\frac1{b_T}\sum_{j=1}^{b_T}A(G_j^T)^\top u(G_j^T)
\in\mathbb R^r.
$$

每个坐标都是目标分类更新需要的一个监督查询。方法估计的是这些查询的联合答案，而非先输出每张目标图的预测残差。

源标签存在随机性时，校准所针对的是固定目标输入上的条件期望分类梯度。有限目标标签实现产生的梯度还包含标签波动，不能将二者视为完全相同的确定量。

### 3.7 完整图计算签名与非线性任务函数族

对每张完整图，使用当前图级表示和同一参数基下的响应构造签名：

$$
R_t(G)=\bigl(h_{\bar\vartheta_t}(G),A_t(G)\bigr).
$$

其中 $h$ 提供当前模型的整图表征，$A$ 保留该图对目标分类方向的计算响应。只使用 $h$ 的条件标签关系，通常不足以约束 Jacobian 加权监督；将 $A$ 纳入签名，使查询所用的计算权重在签名中明确可见。

采用非线性的完整图核：

$$
k_t(G,G')
=\exp\left[
-\frac{\|h_t(G)-h_t(G')\|_2^2}{2s_{h,t}^2}
-\frac{\|A_t(G)-A_t(G')\|_F^2}{2s_{A,t}^2}
\right].
$$

两个正尺度由本轮源查询图的成对距离确定：

$$
s_{h,t}^2
=\max\left\{\operatorname{median}_{i<i'}\|h_t(G_i^S)-h_t(G_{i'}^S)\|_2^2,\varepsilon^2\right\},
$$

$$
s_{A,t}^2
=\max\left\{\operatorname{median}_{i<i'}\|A_t(G_i^S)-A_t(G_{i'}^S)\|_F^2,\varepsilon^2\right\}.
$$

同一组尺度用于源源、源目标和目标目标核。该规则不读取目标标签，也不改变图表示以缩小域间距离。若尺度数值退化，应保留上述显式下限并检查核矩阵是否仍提供有效的源目标交互。

主版本明确使用非线性核。线性签名核可能重新受到源分类驻点的限制，因为源残差可以同时与当前表示和一阶响应近似正交。非线性核避免这一线性代数上的必然退化，但不保证其额外分量具有可迁移的任务信息。

核定义一个标量再生核 Hilbert 空间 $\mathcal H_t$，$q$ 个类别对比坐标使用直积函数空间 $\mathcal H_t^q$。该函数族只用于规定哪些源监督读出与目标查询需要相互一致，不引入额外的可训练 GNN。

### 3.8 明确源监督能够回答目标查询的工作条件

本方法的迁移对象是完整图签名下的条件任务关系。记：

$$
\eta_{D,t}(R)=\mathbb E_D[Q_C^\top Y\mid R_t(G)=R].
$$

需要源目标上的 $\eta_{D,t}$ 在相关区域近似共用。用于估计固定完整目标输入上的条件梯度时，工作近似还要求签名保留足够的任务信息：

$$
\mathbb E_D[Q_C^\top Y\mid G]
\approx\eta_t\bigl(R_t(G)\bigr),
\qquad D\in\{S,T\}.
$$

如果这个任务充分性近似不成立，签名层面的条件梯度不能直接等同于给定完整图后的条件梯度。它与跨域条件稳定性共同构成方法的适用要求。

由于 $p_{\bar\theta_t}$ 是图表示 $h$ 的函数，本轮条件残差可写为：

$$
\mu_t(R)=\eta_t(R)-\breve p_{\bar\theta_t}(R).
$$

方法进一步要求 $\mu_t$ 能够被 $\mathcal H_t^q$ 适度复杂地近似，并且源查询图能够约束目标梯度所需的函数分量。这里不要求源目标的平均残差、平均梯度或类别比例相同。

这些是方法使用源监督的工作条件，不由响应相似、核值较大或共享类别集合自动保证。条件标签关系在同一签名下发生变化时，本方法不从无标签目标图中恢复任意新的类别规则。原始图规模是否重叠，也不能直接替代对签名任务关系与监督支持的判断。

### 3.9 用矩阵权重直接读取目标梯度查询

对源查询图计算真实标签残差：

$$
\widetilde r_i^S
=Q_C^\top\bigl(Y_i^S-p_{\bar\theta}(G_i^S)\bigr)\in\mathbb R^q.
$$

按图优先、图内按对比坐标的顺序堆叠：

$$
\mathbf r_S
=\begin{bmatrix}
\widetilde r_1^S\\\vdots\\\widetilde r_{b_S}^S
\end{bmatrix}
\in\mathbb R^N,
\qquad N=b_Sq.
$$

给每张源图一个读出矩阵 $W_i\in\mathbb R^{r\times q}$，并写成：

$$
W=[W_1,\ldots,W_{b_S}]\in\mathbb R^{r\times N}.
$$

对于任意函数 $u\in\mathcal H_t^q$，源端的查询读出为：

$$
\mathcal S_W[u]
=\sum_{i=1}^{b_S}W_i u(G_i^S).
$$

读取权重由下一节的目标梯度校准问题确定。它允许正负系数，也允许类别对比与参数方向之间的组合，不被解释为概率、密度比或每张源图的单个重要性权重。

最终监督答案为：

$$
\widehat d_T=W^\star\mathbf r_S.
$$

$W$ 已经包含源图聚合系数，外部不再额外除以 $b_S$。

### 3.10 校准目标：共同控制平均与最大方向的查询误差

构造标量核矩阵：

$$
(K_{SS})_{ii'}=k_t(G_i^S,G_{i'}^S),
\quad
(K_{ST})_{ij}=k_t(G_i^S,G_j^T),
\quad
(K_{TT})_{jj'}=k_t(G_j^T,G_{j'}^T).
$$

对应类别对比空间的块核为：

$$
\mathbb K_{SS}=K_{SS}\otimes I_q,
\qquad
\mathbb K_{ST}=K_{ST}\otimes I_q,
\qquad
\mathbb K_{TT}=K_{TT}\otimes I_q.
$$

将目标方向响应写为：

$$
B_T=\frac1{b_T}
\begin{bmatrix}
A(G_1^T)\\\vdots\\A(G_{b_T}^T)
\end{bmatrix}
\in\mathbb R^{b_Tq\times r}.
$$

定义：

$$
H_{ST}=\mathbb K_{ST}B_T\in\mathbb R^{N\times r},
\qquad
H_{TT}=B_T^\top\mathbb K_{TT}B_T\in\mathbb R^{r\times r}.
$$

源读出与目标查询之间的误差算子为 $\mathcal T_T-\mathcal S_W$。其 Gram 矩阵可以直接计算：

$$
E_t(W)
=H_{TT}-WH_{ST}-H_{ST}^\top W^\top
+W\mathbb K_{SS}W^\top
\in\mathbb R^{r\times r}.
$$

$E_t(W)$ 是半正定矩阵，描述所选任务函数族中，各参数方向上的查询误差及方向之间的关联。它不是由目标标签测得的实际误差，也不是已经校准的统计置信区间。

通过一个内部目标求解读出：

$$
\boxed{
W_t^\star
=\arg\min_W
\left\{
\mathcal Q_t(W)
=\frac1r\operatorname{tr}E_t(W)
+\lambda_{\max}\bigl(E_t(W)\bigr)
+\frac{\lambda_W}{r}\|W\|_F^2
\right\},
\qquad \lambda_W>0.
}
$$

三项分别负责：

- 平均方向误差，使每个目标参数方向都有基本的估计要求。
- 最大方向误差，避免少数高误差方向主导最终参数偏差。
- 权重正则，限制源标签波动与读出系数放大。

平均项与最大项的相对权重在主版本中固定，不增加一个混合超参数。这个比例是明确的设计选择，不被宣称为所有数据条件下唯一最优。

平均项不可省略。若只优化最大误差，一个缺少源支持、误差无法降低的方向可能占据最大值，使其他可估方向的改进不再影响目标。保留平均项，使可估方向仍能获得监督。

该问题关于 $W$ 为强凸问题，使用 $\lambda_W>0$ 时有唯一有限解。求解时固定 GNN、核与目标 Jacobian，不通过 $W_t^\star$ 对 GNN 求高阶梯度。

### 3.11 内部求解的具体计算过程

采用小矩阵变量：

$$
\Omega\succeq0,
\qquad
\operatorname{tr}\Omega=1,
\qquad
D=I_r+r\Omega.
$$

固定 $\Omega$ 后，最优读出满足：

$$
DW\mathbb K_{SS}+\lambda_W W=DH_{ST}^\top.
$$

这是一个 Sylvester 方程。若：

$$
\mathbb K_{SS}=R_K\operatorname{diag}(\kappa_b)R_K^\top,
\qquad
D=R_D\operatorname{diag}(\delta_a)R_D^\top,
$$

并记：

$$
\widetilde H=R_D^\top H_{ST}^\top R_K,
$$

则可以逐元素计算：

$$
\widetilde W_{ab}
=\frac{\delta_a\widetilde H_{ab}}
{\delta_a\kappa_b+\lambda_W},
\qquad
W=R_D\widetilde W R_K^\top.
$$

其中 $1\le\delta_a\le1+r$，每个方向都具有正的基础权重。实现中利用 Kronecker 结构复用 $K_{SS}$ 的特征分解，不必显式展开全部块矩阵。

一个完整的内部求解过程为：

1. 初始化 $\Omega=I_r/r$，并分解本轮 $K_{SS}$。
2. 根据当前 $D=I_r+r\Omega$ 求 $W$。
3. 计算 $E_t(W)$，在更新 $\Omega$ 之前，使用同一组 $\Omega$ 及其条件最优 $W$ 检查下式的 gap。
4. 若达到求解精度则返回当前 $W$。否则沿 $E_t(W)$ 做对偶梯度上升，将 $\Omega$ 投影回半正定、迹为一的集合。投影可通过特征分解及特征值的概率单纯形投影完成。
5. 更新 $\Omega$ 后返回步骤 2，重新求解与新 $\Omega$ 对应的 $W$。不能将旧 $W$ 与新 $\Omega$ 混合计算 gap。

内部求解精度使用：

$$
\operatorname{gap}_t
=\lambda_{\max}\bigl(E_t(W)\bigr)
-\operatorname{tr}\bigl(\Omega E_t(W)\bigr).
$$

对偶上升的步长采用回溯。对于每个待评估的 $\Omega$，先求解它自己的条件最优读出 $W_\Omega$，再计算对偶目标：

$$
\mathfrak d_t(\Omega)
=\frac1r\operatorname{tr}E_t(W_\Omega)
+\operatorname{tr}\bigl(\Omega E_t(W_\Omega)\bigr)
+\frac{\lambda_W}{r}\|W_\Omega\|_F^2.
$$

达到预设的绝对与相对 gap 容差，或达到明确的内部计算上限后停止。达到上限时保留当前有限的近似读出，并将其视为近似求解，不将未收敛结果描述为精确最优。

内部求解只更新 $\Omega$ 和临时 $W$，不训练一个新的映射网络。若出现非有限数值或核矩阵有超出舍入误差的非半正定异常，本轮反馈置零并处理数值问题，不将异常权重传给 GNN。

### 3.12 使用当前预测基准形成梯度反馈

当前目标预测项可以精确计算：

$$
\mathcal T_T[\breve p]
=\frac1{b_T}\sum_j A(G_j^T)^\top\breve p_{\bar\theta}(G_j^T).
$$

未知的标签查询值采用以下带基准的估计：

$$
\widehat{\mathcal T_T[\eta]}
=\mathcal T_T[\breve p]
+W_t^\star\mathbf r_S.
$$

这里 $\eta(R)=\mathbb E[Q_C^\top Y\mid R]$。两项已知预测矩抵消后，目标分类梯度估计为：

$$
\boxed{
\widehat g_{T,t}
=-V_tW_t^\star\mathbf r_S
=-V_t\widehat d_{T,t}.
}
$$

实际实现直接计算 $W_t^\star\mathbf r_S$，无需构造一个用于预测的标签矩或逐图目标伪标签。

保留 $Y-p$ 是为了进行有真实源标签支撑的校正。它不再被用作“在源响应列中最小化逐图残差 MSE”的拟合目标。读取矩阵由目标梯度查询的算子误差确定，监督值由真实源残差给出。

若源目标核交互为零，则 $H_{ST}=0$，校准最优解为 $W_t^\star=0$，反馈也为零。交互趋近于零时，反馈相应减弱。该性质说明无支持时的执行行为合理，不保证核交互非零时条件监督关系就一定可迁移。

源平均分类梯度接近零，并不自动使 $W_t^\star\mathbf r_S=0$，因为矩阵读出一般不等于源 Jacobian 的均匀聚合。若所有源查询残差逐图为零，则反馈仍然为零。

### 3.13 损失设计与实际参数更新

#### 3.13.1 源分类主损失

$$
\mathcal L_{\mathrm{sup},t}(\theta)
=-\frac1{|\mathcal B_{S,t}^{\mathrm{fit}}|}
\sum_{(G,Y)\in\mathcal B_{S,t}^{\mathrm{fit}}}
Y^\top\log p_\theta(G).
$$

#### 3.13.2 参数坐标上的查询反馈损失

固定 $V_t$、$\bar\theta_t$ 和 $\widehat d_{T,t}$，定义：

$$
\mathcal L_{\mathrm{qry},t}(\theta)
=\frac12
\left\|
\operatorname{sg}(V_t)^\top
\bigl(\theta-\operatorname{sg}(\bar\theta_t)\bigr)
-\operatorname{sg}(\widehat d_{T,t})
\right\|_2^2.
$$

在当前起点 $\theta=\bar\theta_t$：

$$
\left.\nabla_\theta\mathcal L_{\mathrm{qry},t}(\theta)\right|_{\theta=\bar\theta_t}
=-V_t\widehat d_{T,t}
=\widehat g_{T,t}.
$$

该损失在参数坐标中执行估计梯度，不再把监督先转回目标 logits 再乘一次 Jacobian。它只是当前一次更新的梯度接口，不是在整个训练期间拟合一个固定参数 teacher。

损失不除以 $r_t$，避免目标空间秩变化时额外改变反馈尺度。仅在当前起点执行一步，随后重新构造查询。

#### 3.13.3 总损失与单步规则

$$
\boxed{
\mathcal L_t(\theta)
=\mathcal L_{\mathrm{sup},t}(\theta)
+\beta\mathcal L_{\mathrm{qry},t}(\theta),
\qquad \beta\ge0.
}
$$

主版本在适配阶段采用无动量的普通梯度下降，使估计方向与实际增量一致：

$$
\boxed{
\theta_{t+1}
=\theta_t
-\eta_t\nabla_\theta\mathcal L_{\mathrm{sup},t}(\theta_t)
+\eta_t\beta V_t\widehat d_{T,t}.
}
$$

这是一项源监督更新与一项目标反馈的联合更新，不被描述为纯目标风险的无偏梯度步。即使目标估计准确，保留源 CE 和 $\beta$ 之后，实际优化对象也仍是一个源目标混合目标。

若改用 Adam 或动量优化器，辅助损失提供的梯度仍是 $\widehat g_{T,t}$，但实际参数增量通常不再等于 $-\eta_t\widehat g_{T,t}$。这属于执行规则的改变，需要明确说明。

| 优化对象 | 目标 | 是否对 GNN 反向传播 |
| --- | --- | --- |
| GNN 参数 $\theta$ | 源分类主损失 $\mathcal L_{\mathrm{sup}}$ | 是 |
| GNN 参数 $\theta$ | 查询反馈辅助损失 $\mathcal L_{\mathrm{qry}}$ | 是 |
| 本轮矩阵读出 $W$ | 内部校准目标 $\mathcal Q_t$ | 否 |

因此网络只有一项辅助损失。即使将内部校准目标也计入辅助优化目标，总数仍为两个。

### 3.14 停止梯度边界

| 量 | 本轮处理 |
| --- | --- |
| 快照 $\bar\theta_t$、类别基 $Q_C$ | 固定 |
| 目标 Jacobian、参数基 $V_t$、整图响应 $A_t$ | 计算后停止梯度 |
| 签名、核尺度、所有核矩阵 | 计算后停止梯度 |
| $H_{ST}$、$H_{TT}$、$E_t$ 与校准变量 | 仅用于内部求解 |
| 源标签残差 $\mathbf r_S$ | 在快照处计算并停止梯度 |
| $W_t^\star$、$\widehat d_{T,t}$ | 作为当前固定监督反馈 |
| 源 CE 中的当前网络前向 | 保留梯度 |
| 查询损失中的当前参数 $\theta$ | 保留梯度 |

不通过 $\partial V_t/\partial\theta$、$\partial W_t^\star/\partial\theta$ 或核矩阵的总导数修改本轮更新。只有源 CE 和参数坐标损失向 GNN 提供训练梯度。

### 3.15 从输入到输出的完整流程

**输入。** 源图与标签、无标签目标图、图编码器与分类头、源监督角色划分、校准正则 $\lambda_W$、反馈系数 $\beta$、学习率与训练预算，以及数值秩、核尺度和内部求解容差。

**初始化。** 在源 fit 图上训练 GNN，得到 $\theta_0$；构造固定类别对比基 $Q_C$。

**每轮适配。**

1. 读取源分类批次、源查询批次和无标签目标批次。
2. 固定当前参数快照与确定性网络状态。
3. 计算目标对比 logits 的全参数 Jacobian，得到有效参数基 $V_t$。
4. 若有效秩为零，执行源分类更新并进入下一轮。
5. 在完整源查询图和目标图上计算图表示 $h_t$ 与响应 $A_t=\widetilde J_tV_t$。
6. 使用共同的源侧尺度构造非线性核矩阵。
7. 根据目标 $A_t$ 构造 $B_T$、$H_{ST}$ 和 $H_{TT}$。
8. 求解平均与最大方向误差的联合校准问题，得到 $W_t^\star$。
9. 用真实源标签构造 $\mathbf r_S$，计算 $\widehat d_{T,t}=W_t^\star\mathbf r_S$。
10. 计算源 CE 和参数坐标反馈损失。
11. 合并两项梯度，执行一次参数更新。
12. 丢弃当前 Jacobian、参数基、核与读出缓存，在新参数位置重新查询。

**输出。** 适配完成后保留 GNN 参数 $\theta^\star$。源查询图、参数基、核矩阵和读出权重均不进入推断过程。

### 3.16 推断与计算实现

对新的完整目标图，只需一次常规 GNN 前向：

$$
G\longrightarrow h_{\vartheta^\star}(G)
\longrightarrow z_{\theta^\star}(G)
\longrightarrow p_{\theta^\star}(G).
$$

训练开销主要来自目标 Jacobian 的有效空间构造、源图上的 $r_t$ 个方向响应，以及源核矩阵与小规模校准求解。二分类时 $r_t$ 不超过目标批次的图数，通常远小于参数量，但额外计算并非零成本。

可以通过分块自动微分或 VJP/JVP 构造有效空间，不需要形成 $P\times P$ 矩阵。存储参数基约需 $O(Pr_t)$，标量源核约需 $O(b_S^2)$，读出约需 $O(r_tb_S(C-1))$。利用块核结构后，只需分解标量 $K_{SS}$，无需对完整 $N\times N$ 块核重复做通用分解。

GNN 前向与 Jacobian 可以使用足够精度的常规浮点计算，核矩阵和小规模校准求解建议使用 FP64。正则化使奇异源核仍有有限解；实现中使用分解或方程求解，不显式计算矩阵逆。

### 3.17 与原设计的实质差异和方法边界

| 环节 | 原设计 | 本修订版 |
| --- | --- | --- |
| 查询空间 | 固定数量的目标随机方向与 odd/even 响应 | 当前目标分类 Jacobian 的完整有效空间 |
| 源监督读取目标 | 源图上的逐图输出残差 MSE | 目标分类梯度查询的平均与最大方向误差 |
| 临时求解变量 | 对所有目标图共用的响应系数 $a$ | 目标梯度方向联合决定的源监督矩阵 $W$ |
| 目标信息的作用 | 产生方向、响应与尺度 | 同时定义查询对象、校准目标与执行空间 |
| 源监督值 | $Y_S-p_S$ | 保留 $Y_S-p_S$，用于当前预测基准上的校正 |
| 反馈执行 | 目标 logit 参考经反向传播形成参数梯度 | 在同一参数坐标中直接提供估计梯度 |

**与 KRR 的关系。** 只保留平均误差和权重正则时，校准会退化为常规核回归的对偶形式。加入最大方向误差后，目标梯度方向被联合处理；固定 $D$ 时，可理解为不同方向使用有效正则 $\lambda_W/\delta_a$。一般不再由一个共享 scalar-ridge 核回归器同时给出所有坐标，但它仍属于广义的目标相关核估计。$r_t=1$ 或校准度量各向同性时，会退化为共享 ridge 形式。

**坐标一致性。** 目标参数基的正交旋转不会改变其空间。签名中的 Frobenius 距离、误差矩阵的迹与最大特征值、权重范数均采用与这种旋转兼容的形式，避免人为给某一任意基方向设置特殊地位。

**迁移条件。** 核函数和目标计算不会自动创造源目标共同的标签规则。函数族失配、条件标签关系变化或目标方向缺乏源监督约束时，梯度估计仍可能不准确。反馈趋零只是一种合理的无支持行为，不代表方法在该目标区域已经完成适配。

**残差条件。** 本版改变的是残差被读取的目标与方式，并未取消残差。它避免将源平均梯度的抵消直接等同于没有目标监督，但不保证所有训练状态都有可迁移的非线性残差。

**统计与执行范围。** 校准风险是所选函数族下的结构性代理目标。自适应复用源标签、有限样本噪声和核失配需要另行分析，不能直接借用固定设计或同分布下的无偏结论。参数坐标损失精确提供构造出的梯度，但这一代数事实不等于真实目标风险必然下降。

**研究贡献候选。** 本版将源监督读取、目标梯度空间和实际参数反馈放进同一个计算定义中。核心研究对象是这套目标分类梯度查询如何在图分布变化下使用源标签，而非更换一个对齐距离、堆叠辅助模块，或把已有核回归重新命名。
