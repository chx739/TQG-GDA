# Graph-Level GDA: Target-Distribution-Driven Classification Boundary Adaptation

> Translation note (2026-11-09): This is an English translation of the [original Chinese design](design-zh.md). The design status below reflects the original document, not the current repository status. An implementation and synthetic engineering checks are now available; real-task effectiveness remains unverified. For current implementation details and the distinction between normalized margin changes and exact geometric boundary displacement, see the [implementation notes](implementation.md) and [code review guide](../review/code-review-guide.md).

**Tentative method name: Target-Score Boundary Adaptation (TSBA)**

**Design status: A new research proposal, not yet implemented or validated.**

This document follows the structure of the preceding design documents and covers only motivation, the research question, and method design. Its central workflow is: **source labels establish class semantics, the target distribution of whole-graph representations specifies boundary displacement requests, and the model's response converts these requests into parameter updates.**

This proposal is not a minor revision of the TQG v2 kernel reader. Instead of extrapolating the full target cross-entropy gradient from source residuals, it constructs adaptation updates driven by target distribution information and constrained by source classification supervision. The main version freezes the source GNN encoder and updates a small nonlinear classification module in a shared whole-graph coordinate system; it does not claim to implement end-to-end encoder adaptation.

## 1. Motivation

### 1.1 From Re-reading Source Corrections to Letting the Target Distribution Help Determine Corrections

Graph-level unsupervised domain adaptation involves a fundamental division of information: ground-truth class labels are available in the source domain, while the classification behavior to be improved is in the target domain. Source supervision can establish class semantics and an initial decision rule. After the graph distribution changes, however, merely recombining source training residuals may not produce the updates needed by the target task.

Existing TQG-v2 diagnostics provide two motivations for this design. First, after freezing the backbone, H and HA contain learnable label information in both within-source and within-target LR/MLP readout experiments. Second, with the same amount of supervision, a cross-domain prediction gap remains between a source-trained readout and a diagnostic readout trained within the target domain. These results support further investigation into adapting decision rules using existing whole-graph information, but they establish neither the sufficiency of the graph representations nor that the gap is entirely due to changes in the conditional label relationship. [D1]

Accordingly, this proposal no longer limits target graphs to defining a query space or adjusting source-label weights. Instead, the target input distribution directly helps determine how the classification boundary should move. Source labels retain two roles: establishing the meaning of the classes and constraining boundary motion so that it does not arbitrarily disrupt known classification relationships.

### 1.2 More Confident Predictions Do Not Necessarily Imply a Better Classification Boundary

A classification model can reduce prediction entropy by increasing the scale of its logits without changing any class decision boundary. Thus, increased confidence in target predictions is insufficient evidence that an initially incorrect target decision rule has been corrected.

This proposal focuses on a different computable change: **the actual displacement of the classification boundary in the target whole-graph representation space.** Under explicit assumptions, the distribution of target data near the boundary can provide a displacement direction; the model's parameter response determines whether the current classifier can realize that displacement.

For example, if one direction moves the boundary into a region of higher target data density while the opposite direction moves it out of that local region, the target distribution expresses a preference for boundary location. Although this is not a target class label, it can provide a classification-related constraint under the low-density separation assumption.

### 1.3 Core Working Assumption: Useful Boundaries Are Locally Compatible with Low-Density Target Regions

The proposal relies on the following working conditions.

**Whole-graph representation condition.** The fixed source GNN representation retains enough information for target classification that a nonlinear graph-level classification module can express useful target decision rules. If the encoder has discarded task information, updating only the classification module cannot recover it.

**Boundary–distribution condition.** On some valid portions of the boundary near the current classifier, moving toward lower target density may improve class separation. This is a local low-density separation assumption. It does not require each class to form a single cluster, nor does it guarantee that every low-density boundary is correct.

**Class semantics condition.** Source labels and the initial classifier can constrain the assignment of classes to target regions. Unlabeled target density alone cannot determine which class a region belongs to, or resolve arbitrary class permutations or arbitrary changes in the conditional label relationship.

**Local estimation condition.** Estimates of the target distribution gradient have some accuracy in the queried boundary neighborhoods, and boundary projection and parameter updates remain local.

Existing experimental reports support only the presence of readable task information in the signatures; they have not validated the boundary–distribution condition above. [D1] This document treats that condition as an explicit assumption of the new proposal, not as an established experimental finding.

### 1.4 Connection to GDA and Generality of the Method

The inputs are always individual, complete graphs. Through message passing and graph-level readout, a GNN maps graphs with different numbers of nodes and edges into a common continuous representation space. A shift in graph distributions can change the sample distribution in that space, making the source classifier's boundary unsuitable for the target data.

The proposal estimates the target distribution and moves the classification boundary in this whole-graph space. It does not modify graph structure, require paired source and target graphs, or use within-graph semantic components, context/evidence decomposition, or propagation-resolution routing.

The computational principle applies to differentiable classification models more generally; in this design, the graph encoder provides a shared whole-graph coordinate system. Because the encoder is fixed during adaptation, the explicit scope of the main version is **graph-level decision adaptation**, not recovery of the true target supervised gradient for the entire GNN.

### 1.5 Technical Foundations and Potential Contributions

Low-density separation and the use of unlabeled distributions to adjust classification rules have a long research history and cannot be presented as new fundamental principles. [R1] Denoising score matching provides an implementable tool for estimating the log-density gradient of a smoothed data distribution. [R2]

Recent work, ReTTA, incorporates score-matching/likelihood information into test-time adaptation together with discriminative objectives. Thus, the idea that target distribution gradients can contribute to adaptation is not an independent innovation of this proposal. [R3] Function-Space Learning Rates measures parameter updates through changes in model outputs, providing technical inspiration for understanding updates through function changes rather than parameter norms alone. [R4] Score-based distribution matching has also been explicitly studied. [R5]

The potential contribution of this proposal centers on a specific interface:

> **In fixed whole-graph coordinates, project the target distribution score onto the normal of the current classification boundary to obtain a desired boundary displacement; then realize that displacement through a normalized parameter response under source-semantic constraints.**

The method does not require the two domains to have identical representation distributions, regress target values of source residuals, or use target entropy reduction as its sole objective. Score matching, boundary projection, Gauss–Newton, and linear-system solvers are all established tools. Whether the proposal is sufficiently novel depends on the boundary update mechanism they form and its practical value; a complete workflow alone does not establish novelty.

References for this section:

- [D1] The user-provided *TQG-v2 Experimental Report*, Experiments 1–4: finite-sample learnability, the cross-domain prediction gap, and support stratification. Parts of the report that use target labels are diagnostic only and do not authorize their use in training this method.
- [R1] [Random Matrix Analysis to Balance between Supervised and Unsupervised Learning under the Low Density Separation Assumption, ICML 2023](https://proceedings.mlr.press/v202/feofanov23a.html).
- [R2] [Generative Modeling by Estimating Gradients of the Data Distribution, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html). This proposal uses only distribution score estimation at a fixed noise scale, not the generative sampling procedure.
- [R3] [Rethinking Entropy in Test-Time Adaptation: The Missing Piece from Energy Duality, NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/435422305988b73c6cc00bcb29ba2531-Abstract-Conference.html).
- [R4] [Function-Space Learning Rates, ICML 2025](https://proceedings.mlr.press/v267/milsom25a.html).
- [R5] [Expressive Score-Based Priors for Distribution Matching with Geometry-Preserving Regularization, ICML 2025](https://proceedings.mlr.press/v267/gong25c.html).

## 2. Research Question

### 2.1 Main Research Question

> **How can directions for classification boundary motion be extracted from the distribution of unlabeled target graphs and converted into effective graph-classification parameter updates under the semantic constraints supplied by source labels?**

English formulation provided in the original:

> **How can the distribution of unlabeled target graphs guide decision-boundary motion, and how can that motion be realized as model updates while preserving source-supervised class semantics?**

The goal is not to recover the true target cross-entropy gradient without assumptions, but to establish a clear information chain for adaptation: the target distribution provides preferences for boundary location, source labels provide class meanings, and the parameter response realizes the changes jointly determined by the two.

### 2.2 Three Connected Design Questions

**What new information do target graphs provide?** Estimate the score of the smoothed target distribution in fixed whole-graph coordinates to obtain directions of increasing or decreasing density near the boundary, rather than merely reassessing similarities to labeled source data.

**How does this information become a classification requirement?** Extract only the score component along the normal of a valid classification boundary to construct a small boundary displacement. This neither requires target samples to aggregate around a class prototype nor treats currently predicted classes as ground-truth labels.

**How do boundary requests become parameter updates?** Use the response of the boundary margin to parameters to jointly solve for a local update that accounts for source classification and boundary displacement, and check execution error using actual forward evaluations.

### 2.3 Unified Technical Approach

$$
\boxed{
\text{Target distribution in fixed whole-graph coordinates}
\;\longrightarrow\;
\text{Boundary-normal displacement}
\;\longrightarrow\;
\text{Parameter updates under source-semantic constraints}
}
$$

The main version uses a frozen graph encoder, a trainable nonlinear classification module, and a target score network used only during training. Training comprises one primary source classification objective and two auxiliary objectives with separate roles: target score estimation and boundary displacement fitting.

## 3. Method Design

### 3.1 Inputs, Outputs, and Task Setting

The inputs are a set of labeled source graphs and a set of unlabeled target graphs:

$$
\mathcal D_S=\{(G_i^S,Y_i^S)\}_{i=1}^{n_S},
\qquad
\mathcal D_T=\{G_j^T\}_{j=1}^{n_T}.
$$

Each $G$ is a complete attributed graph. The source and target domains share $C\ge2$ classes and compatible input feature interfaces. $Y\in\{0,1\}^C$ is a one-hot graph label. Neither graph instances nor nodes need to be paired across domains.

The method is source-access UDA. Initialization and adaptation may access the source training labels specified in advance; target class labels are excluded from score training, boundary construction, losses, step-size selection, and model selection. A change to this proposal does not authorize incorporating class labels from existing target diagnostic sets into training.

The main version addresses a fixed unlabeled target pool and makes no claim of continual learning capability. If the input target pool contains test graphs, the setting must be explicitly identified as transductive unlabeled adaptation.

The outputs are the frozen graph encoder, the fixed coordinate transformation, and the final classification parameters $\psi^\star$. Inference produces:

$$
p_{\psi^\star}(G)=\operatorname{softmax}(f_{\psi^\star}(x(G))),
\qquad
\widehat y(G)=\arg\max_c p_{\psi^\star,c}(G).
$$

### 3.2 Source Initialization and Fixed Whole-Graph Coordinates

First, use standard graph-classification cross-entropy on the authorized source training pool to obtain a source model:

$$
z_0(G)=W_0h_{\theta_0}(G)+b_0,
\qquad h_{\theta_0}(G)\in\mathbb R^d.
$$

An existing source model may be reused directly. The source validation set retains its original role and is not automatically merged into training. The new method does not require a separate source query pool for kernel reading, nor does it silently increase the source-label budget available for training.

Upon entering adaptation, freeze the graph encoder $h_{\theta_0}$ and its normalization running statistics, and disable stochastic dropout. First cache the representations of complete source and target graphs.

Compute a mean and a single global RMS scale from the source training representations:

$$
\mu_H=\frac1{n_S}\sum_i h_{\theta_0}(G_i^S),
\qquad
s_H=\sqrt{\frac1{n_Sd}\sum_i\|h_{\theta_0}(G_i^S)-\mu_H\|_2^2+\epsilon_H^2}.
$$

Define shared, fixed coordinates:

$$
x(G)=\frac{h_{\theta_0}(G)-\mu_H}{s_H}\in\mathbb R^d.
$$

The same $\mu_H,s_H$ are used for both domains and subsequent inference. The main version does not standardize the source and target separately, introduce a projection that changes during adaptation, or discard representation dimensions by default for the convenience of distribution modeling.

Using a global scale rather than arbitrary per-dimension scaling avoids directly amplifying coordinates with very small source variance. Fixed coordinates ensure that the target distribution seen by the score model does not move as the classifier is updated.

### 3.3 Trainable Nonlinear Graph-Level Classification Module

On the fixed $x(G)$, use a source linear head with a small residual branch:

$$
f_\psi(x)=P_C\left[Wx+b+U_2\operatorname{SiLU}(U_1x+a)\right],
\qquad
P_C=I_C-\frac1C\mathbf1_C\mathbf1_C^\top.
$$

Here:

- $W\in\mathbb R^{C\times d}$ and $b\in\mathbb R^C$ form the linear classification head.
- $U_1\in\mathbb R^{r_a\times d}$, $a\in\mathbb R^{r_a}$, and $U_2\in\mathbb R^{C\times r_a}$ form a small residual adapter.
- $\psi=(W,b,U_1,a,U_2)$ are the adaptation parameters, with dimension denoted by $P_\psi$.
- $r_a$ is the adapter width specified in advance and is not adjusted using target labels.

Initialize:

$$
W=s_HW_0,
\qquad
b=W_0\mu_H+b_0,
\qquad
U_2=0.
$$

Initialize $U_1$ using a fixed random seed and set $a=0$. Thus, the initial class probabilities at the start of adaptation match those of the original source model. Centering across classes with $P_C$ changes neither softmax probabilities nor the logit difference between any two classes.

All of $\psi$ can be updated, while the encoder remains frozen. SiLU makes the graph-level margin smooth with respect to its input, facilitating local boundary localization; it is not an additional graph-structure modeling module.

### 3.4 Definition and Scale of the Target Score

Let $P_T^x$ denote the target representation distribution in the fixed coordinate system. For a positive noise scale $\sigma$, define the Gaussian-smoothed distribution:

$$
p_{T,\sigma}(\widetilde x)
=
\int\mathcal N(\widetilde x;x,\sigma^2I_d)\,dP_T^x(x).
$$

The target score is:

$$
s_T^\sigma(\widetilde x)=\nabla_{\widetilde x}\log p_{T,\sigma}(\widetilde x).
$$

This gradient is taken with respect to whole-graph representation coordinates, not model parameters in a target cross-entropy objective.

The main version uses a single fixed $\sigma$ and does not construct a full diffusion generation chain. An initial scale rule that does not use target classes is:

$$
\ell_T
=
\operatorname{median}_j
\min_{k:\,\|x_j^T-x_k^T\|_2>0}
\|x_j^T-x_k^T\|_2,
\qquad
\sigma=\kappa_\sigma\frac{\ell_T}{\sqrt d}.
$$

$\kappa_\sigma>0$ is fixed before execution. This rule makes the typical noise norm comparable in scale to the spacing between target samples. It is an engineering initialization, not a claim of optimal bandwidth. Duplicate representations are excluded as positive-distance neighbors; density-driven adaptation does not start if fewer than two distinct target representations are available.

An excessively small $\sigma$ may capture only finite-sample spikes, while an excessively large one may smooth away useful class structure. The main version does not select it using target class labels.

### 3.5 Auxiliary Objective One: Target Distribution Score Estimation

Use a small, smooth MLP $s_\omega:\mathbb R^d\rightarrow\mathbb R^d$ to estimate $s_T^\sigma$. For example, it may have two SiLU hidden layers and a $d$-dimensional linear output layer; it shares no parameters with the graph encoder or classifier.

For target representations, sample:

$$
\epsilon\sim\mathcal N(0,I_d),
\qquad
\widetilde x=x+\sigma\epsilon.
$$

Use denoising score matching:

$$
\boxed{
\mathcal L_{\mathrm{score}}(\omega)
=
\mathbb E_{x\sim\mathcal D_T,\epsilon}
\left[
\frac1d
\left\|\sigma s_\omega(x+\sigma\epsilon)+\epsilon\right\|_2^2
\right].
}
$$

Only $\omega$ is updated; $\theta_0$ and $\psi$ are not. Training uses a predetermined budget. A score checkpoint may be selected using an unlabeled score validation loss with independently resampled noise, but this selection must not be interpreted as optimal for target classification.

After training, fix $s_{\omega^\star}$. Its output is an estimate of the distribution gradient, not class probabilities, pseudo-labels, or calibrated density confidence. Fitting a finite neural network also does not guarantee that the vector field is everywhere an exact conservative field.

### 3.6 Inputs and Snapshots for Each Classification Adaptation Round

Round $t$ reads:

- A source classification batch $\mathcal B_{S,t}$ containing authorized source labels.
- An unlabeled target representation batch $\mathcal B_{T,t}$ of size $b_T$.

Fix the current classifier snapshot at $\bar\psi_t=\psi_t$. All boundary points, class pairs, normals, scores, desired displacements, and response matrices for this round are computed at this snapshot. All of these quantities are refreshed after a single parameter update.

The score network and whole-graph coordinates remain fixed throughout adaptation and are not retrained as the classifier changes from round to round.

### 3.7 Locating Valid Boundary Points from Target Samples

For each target representation $x_j^T$, select the two distinct classes $c_j,d_j$ with the highest current logits. They are used only to identify a candidate class pair for the current decision surface and are not treated as true target labels.

Define the margin for this round's class pair:

$$
m_{j,t}(x)
=
[f_{\bar\psi_t}(x)]_{c_j}-[f_{\bar\psi_t}(x)]_{d_j}.
$$

In binary classification, always use the same class pair. In multiclass classification, keep the class pair fixed within the round to avoid implicitly changing the surface being sought during projection iterations.

Starting from $\xi_j^{(0)}=x_j^T$, use a finite number of approximate projection steps to seek $m_{j,t}(\xi_j)=0$:

$$
\xi_j^{(k+1)}
=
\Pi_{\mathbb B(x_j^T,R_{\mathrm{proj}})}
\left[
\xi_j^{(k)}
-
\frac{m_{j,t}(\xi_j^{(k)})}
{\|\nabla_xm_{j,t}(\xi_j^{(k)})\|_2^2+\epsilon_{\mathrm{proj}}}
\nabla_xm_{j,t}(\xi_j^{(k)})
\right].
$$

If necessary, backtrack this projection step to reduce the geometric residual $|m|/\|\nabla_xm\|$. Stop upon reaching a fixed maximum number of iterations; do not retry indefinitely.

Accept a boundary point only if all of the following conditions hold:

1. $\xi_j$ lies within the projection ball centered on the actual target representation.
2. $\kappa_{j,t}=\|\nabla_xm_{j,t}(\xi_j)\|_2$ exceeds a numerical lower bound.
3. $|m_{j,t}(\xi_j)|/\kappa_{j,t}\le\epsilon_B$.
4. In multiclass classification, $c_j,d_j$ remain the two highest-scoring classes at $\xi_j$, allowing small numerical errors consistent with $\epsilon_B\kappa_{j,t}$; otherwise, reject the candidate.

Let $\mathcal I_t$ denote the accepted set, with size $M_t$. Unsuccessful candidates produce no boundary feedback.

$R_{\mathrm{proj}}$ is specified in advance as a finite multiple of the distance scale of the target representations, for example, on the order of $\sigma\sqrt d$. It limits extrapolation distance but does not establish score accuracy at boundary points.

**Boundary points are only auxiliary computational locations in representation space; they do not correspond to generated real graphs.** This projection generally finds only a nearby root and does not guarantee the globally nearest boundary.

### 3.8 Converting the Target Score into a Directed Boundary Displacement

For each accepted boundary point, compute the unit normal:

$$
n_{j,t}
=
\frac{\nabla_xm_{j,t}(\xi_j)}{\kappa_{j,t}}.
$$

The normal component of the target score is:

$$
\gamma_{j,t}
=
n_{j,t}^\top s_{\omega^\star}(\xi_j).
$$

Let $\Delta_{\max}>0$ be the maximum boundary movement per round, and define the desired normal displacement:

$$
\boxed{
\nu_{j,t}
=
-\operatorname{clip}
\left(
\tau\sigma^2\gamma_{j,t},
-\Delta_{\max},\Delta_{\max}
\right),
\qquad
\zeta_{j,t}=-\nu_{j,t}.
}
$$

$\tau>0$ is a dimensionless movement coefficient. $\sigma^2$ converts the score's inverse coordinate scale to a displacement scale; clipping only limits the magnitude and does not change the sign of a nonzero request.

Intuitively, if density increases along $n_{j,t}$, the boundary displacement request points along $-n_{j,t}$, and vice versa. The tangential component of the score is not used directly as a boundary displacement request.

This rule provides a candidate for locally decreasing target density. It is not claimed to be the exact gradient of the full boundary-area energy, nor is it the gradient of the true target classification risk. A zero normal score component produces no displacement request, even if the current boundary may still be unsatisfactory.

### 3.9 From Boundary Displacement to Parameter Response

Let

$$
m_{\psi,j}(x)=[f_\psi(x)]_{c_j}-[f_\psi(x)]_{d_j}.
$$

At the old boundary point $\xi_j$, the margin change induced by a parameter change has a local correspondence with the boundary's normal displacement: the normalized margin change should approximate $-\nu_{j,t}=\zeta_{j,t}$.

Define the normalized response row:

$$
\boxed{
D_{j,t}
=
\frac{1}{\kappa_{j,t}}
\left.
\frac{\partial m_{\psi,j}(\xi_j)}{\partial\psi}
\right|_{\psi=\bar\psi_t}
\in\mathbb R^{1\times P_\psi}.
}
$$

Stack these to obtain $D_t\in\mathbb R^{M_t\times P_\psi}$ and $\zeta_t\in\mathbb R^{M_t}$. The local change to be realized in this round is:

$$
D_t\delta\psi\approx\zeta_t.
$$

The signs have an explicit meaning here: an increase in the margin at an old boundary point moves the boundary in the negative normal direction. In the binary case $m(x)=a(x-b)$, $a>0$, the response with respect to the boundary-position parameter $b$ is $D=-1$; thus, positive $\gamma$ corresponds to negative $\delta b$.

$D_t$ contains no ground-truth class information and is not a predicted task gradient. It is only a computational interface through which the current classifier converts parameter changes into boundary changes.

### 3.10 Loss Design: One Primary Objective and Two Auxiliary Objectives

#### 3.10.1 Primary Objective: Source Classification

$$
\boxed{
\mathcal L_{\mathrm{sup},t}(\psi)
=
-\frac1{|\mathcal B_{S,t}|}
\sum_{(G,Y)\in\mathcal B_{S,t}}
Y^\top\log\operatorname{softmax}(f_\psi(x(G))).
}
$$

The main version uses source classification CE on the original source training distribution. It does not incorporate true target class proportions, assume equal source and target priors by default, or additionally enforce uniform target class proportions. This loss maintains class semantics but does not guarantee the absence of forgetting, the absence of target class collapse, or a decrease in target risk.

#### 3.10.2 Auxiliary Objective One: Distribution Score

Use $\mathcal L_{\mathrm{score}}(\omega)$ from §3.5 to train only the independent score network.

#### 3.10.3 Auxiliary Objective Two: Realizing Boundary Displacement

At this round's old boundary points, define the normalized margin change obtained from actual forward evaluations:

$$
v_{j,t}(\psi)
=
\frac{
 m_{\psi,j}(\xi_j)
 -m_{\bar\psi_t,j}(\xi_j)
}{\kappa_{j,t}}.
$$

The loss for this round is:

$$
\boxed{
\mathcal L_{\mathrm{move},t}(\psi)
=
\frac1{2M_t}
\sum_{j\in\mathcal I_t}
\left[
 v_{j,t}(\psi)-\operatorname{sg}(\zeta_{j,t})
\right]^2.
}
$$

$\xi_j,c_j,d_j,\kappa_{j,t},\bar\psi_t,\zeta_{j,t}$ are all frozen reference quantities. The candidate classifier output $m_{\psi,j}(\xi_j)$ retains gradients with respect to the parameters. The denominator is not recomputed for the candidate model, preventing the model from evading this round's requirements by changing the margin scale.

Using the margin difference, rather than assuming that the old margin is exactly zero, accommodates numerical error in the boundary projection.

#### 3.10.4 Classifier Objective for the Current Round

$$
\boxed{
\mathcal J_t(\psi)
=
\mathcal L_{\mathrm{sup},t}(\psi)
+
\lambda_B\mathcal L_{\mathrm{move},t}(\psi),
\qquad\lambda_B>0.
}
$$

| Optimization Target | Objective | Data and Role |
| --- | --- | --- |
| Source initialization model | Source classification CE | True source graph labels, establishing initial semantics and whole-graph coordinates |
| Target score network $\omega$ | $\mathcal L_{\mathrm{score}}$ | Unlabeled target representations and the known noise distribution |
| Classification module $\psi$ | $\mathcal L_{\mathrm{sup}}+\lambda_B\mathcal L_{\mathrm{move}}$ | Source classification semantics and target boundary displacement |
| Graph encoder $\theta_0$ | Not optimized during adaptation | Fixed coordinates, preventing density and representations from chasing each other |

There are two auxiliary tasks in total. Numerical damping and local step-size restrictions are solver rules; they do not introduce a third self-supervised task or an additional prediction network.

### 3.11 A Parameter Step That Jointly Addresses Source Semantics and Boundary Movement

Boundary displacement requests are not treated as ordinary logit corrections and passed directly to Adam. The main version explicitly uses a damped Gauss–Newton step to jointly handle the source classification objective and the boundary response.

Compute at $\bar\psi_t$:

$$
g_{S,t}=\nabla_\psi\mathcal L_{\mathrm{sup},t}(\bar\psi_t).
$$

Let $J_i^S\in\mathbb R^{C\times P_\psi}$ be the source graph logits Jacobian and $p_i^S$ the source prediction. Define the source classification generalized Gauss–Newton (GGN) matrix:

$$
F_{S,t}
=
\frac1{|\mathcal B_{S,t}|}\sum_i
(J_i^S)^\top
\left[\operatorname{Diag}(p_i^S)-p_i^S(p_i^S)^\top\right]
J_i^S.
$$

Use the local model:

$$
\delta_t
=
\arg\min_\delta
\left\{
 g_{S,t}^\top\delta
 +\frac12\delta^\top F_{S,t}\delta
 +\frac{\lambda_B}{2M_t}\|D_t\delta-\zeta_t\|_2^2
 +\frac{\mu_t}{2}\|\delta\|_2^2
\right\},
\qquad\mu_t>0.
$$

The corresponding linear system is:

$$
\boxed{
\left(
F_{S,t}
+\frac{\lambda_B}{M_t}D_t^\top D_t
+\mu_t I
\right)\delta_t
=
-g_{S,t}
+\frac{\lambda_B}{M_t}D_t^\top\zeta_t.
}
$$

Here, $F_{S,t}$ is a positive semidefinite GGN approximation and is not described as the full Hessian of the nonlinear model. $\mu_t$ ensures numerical solvability and limits the local step size; it is not interpreted as confidence in target risk.

When $P_\psi$ is small, the system can be solved directly; when it is larger, use conjugate gradients and JVP/VJP to implement matrix–vector products. There is no need to explicitly store the Jacobian of the entire GNN encoder, and no target labels are required.

This solve is not a hard interpolation that guarantees $D_t\delta_t=\zeta_t$. Source semantics, parameter coupling among different boundary points, and damping jointly determine the final trade-off.

### 3.12 Controlling Finite-Step Execution Error with Actual Forward Passes

The linear system produces only a candidate step. The actual update is:

$$
\psi_{t+1}=\psi_t+\alpha_t\delta_t,
\qquad \alpha_t\in\{1,1/2,1/4,\ldots\}.
$$

During backtracking, keep all reference quantities for the current round fixed and recompute the actual source classification CE and $\mathcal L_{\mathrm{move},t}$ at the candidate parameters. Acceptance requires both of the following:

**Decrease in the current-round objective:**

$$
\mathcal J_t(\psi_t+\alpha_t\delta_t)
\le
\mathcal J_t(\psi_t)
+c_A\alpha_t
\nabla\mathcal J_t(\psi_t)^\top\delta_t,
\qquad 0<c_A<1.
$$

**Bounded normalized change:**

$$
\max_{j\in\mathcal I_t}
|v_{j,t}(\psi_t+\alpha_t\delta_t)|
\le\Delta_{\mathrm{exec}}.
$$

$\Delta_{\mathrm{exec}}$ is a predetermined budget for small movements, for example, of the same order as $\Delta_{\max}$. If the conditions remain unmet after a finite number of backtracking attempts, do not apply the parameter update for this round; sample a new target batch in the next round.

These checks control only the computable surrogate objectives for source classification and boundary realization. They are **neither a line search on the true target CE nor a safety certificate**. Normalized margin change approximates boundary displacement; the two cannot be treated as identical for arbitrarily large steps.

Refresh the boundary reference immediately after an update is accepted. Do not perform multiple rounds of classifier training using old boundary points and old normals.

### 3.13 Gradient and Stop-Gradient Boundaries

| Object | Treatment |
| --- | --- |
| $\theta_0,\mu_H,s_H,x(G)$ | Fixed during adaptation; receive neither classification nor score gradients |
| Score network $\omega$ | Updated only during the score stage; frozen during classification adaptation |
| Class pair $c_j,d_j$ | Fixed within the current round and used only for geometric localization |
| Boundary point $\xi_j$ | Differentiate with respect to the input during projection; apply stop-gradient during the main model update |
| $n_{j,t},\kappa_{j,t},\gamma_{j,t},\nu_{j,t},\zeta_{j,t}$ | Apply stop-gradient to all quantities after construction |
| Response $D_t$ and source GGN $F_{S,t}$ | Numerical operators for the current round; do not take meta-gradients through the solution procedure |
| Source classification output | Retain gradients with respect to $\psi$ |
| New model's margin at fixed boundary points | Retain gradients with respect to $\psi$ for the actual finite-step checks |
| $\delta_t$ obtained by the linear solve | Apply as the current explicit update; do not subsequently apply Adam to it |

This is not meta-learning: the GNN is not trained through total derivatives of the boundary projector, score network, or linear system. The different gradients have separate roles: the score provides input-distribution information, $\nabla_xm$ defines the boundary normal, and $\nabla_\psi m$ realizes parameter changes.

### 3.14 Empty Feedback, Numerical Anomalies, and Degenerate Cases

**No valid boundary points.** If $M_t=0$, skip the classifier update for this round. Do not present this as successful target adaptation or automatically revert to continued source training.

**No normal displacement request.** If all $\nu_{j,t}$ are numerically zero, skip the round. This means only that the current rule proposes no action, not that the true target task has reached an optimum.

**Near-zero boundary gradient.** Reject the corresponding anchor; do not amplify the parameter response through an extremely small denominator.

**Non-finite score, loss, or solution.** Record the anomaly and terminate the current execution; do not silently treat it as normal zero feedback. Keep the model snapshot unchanged.

**Conflict between source semantics and target boundary displacement requests.** The joint objective may yield a small step or a rejected step. This is the algorithm's tradeoff behavior and should not be forcibly concealed by continually increasing the feedback strength.

**Missing target classes.** Even with a shared class set, the unlabeled target pool may not cover every class. The main version does not enforce uniform predicted class proportions or directly use source class proportions as a target prior.

**Exact stationary points and symmetric cases.** A high-density stationary point can also have zero score. This design has no mechanism that automatically escapes all such points; stopping must not be interpreted as evidence of a correct low-density boundary.

### 3.15 Configuration to Fix Before Training

| Configuration | Purpose and selection rationale |
| --- | --- |
| Source model and source data pool available for training | Follow the permitted source-supervision protocol; do not automatically increase the label budget |
| Adapter width $r_a$ | Provide a small amount of nonlinear discriminative capacity, for example, a predetermined $\min(16,d)$; this is not a validated optimum |
| Score network width and training budget | Predetermine a small MLP; snapshot selection by unlabeled score validation loss is allowed |
| $\kappa_\sigma$ | Coefficient in the fixed rule for the target smoothing scale |
| $\lambda_B$ | Relative weight of source class semantics and boundary realization |
| $\tau$ | Scale for converting the score into a single-step displacement |
| $R_{\mathrm{proj}},\Delta_{\max},\Delta_{\mathrm{exec}}$ | Constrain boundary query distance, requested displacement, and actual finite change, respectively; define them using the same fixed coordinate scale |
| Number of projection steps, CG tolerance, number of backtracking attempts, and rule for $\mu_t$ | Control numerical solution budgets and stability; do not select them using target classification performance |
| Number of classification adaptation rounds | Fix before training; the main version outputs the last accepted classifier state |

To keep the minimal version clear, the projection radius, request limit, and execution limit can be tied to $\sigma$ or $\sigma\sqrt d$ through predetermined multipliers, rather than tuned independently using target labels. These scales remain design choices, not automatically correct statistics.

### 3.16 Complete Procedure from Input to Output

**Stage 1: Establish source semantics and fixed whole-graph coordinates.**

1. Train or reuse a source GNN using source classification CE to obtain $\theta_0,W_0,b_0$.
2. Freeze the encoder and its running statistics, and cache representations of the complete source and target graphs.
3. Compute fixed $\mu_H,s_H$ from the source training representations to obtain the shared coordinates $x(G)$.
4. Initialize the nonlinear classification module so that its initial predicted probabilities match those of the source model.

**Stage 2: Establish an unlabeled target distribution model.**

1. Determine $\sigma$ using the predetermined input-based rule for target representations.
2. Train $s_\omega$ using only target inputs and Gaussian noise.
3. Fix $\omega^\star$ according to the predetermined budget or unlabeled score validation rule.

**Stage 3: Perform iterative boundary adaptation.**

1. Load a source classification batch and a target input batch, and save $\bar\psi_t$.
2. Select the current top-two class pair at each target point and locate a nearby valid boundary point.
3. Compute the boundary normal, the normal component of the target score, and the desired small displacement.
4. Construct the normalized parameter response $D_t$, along with the source classification gradient and GGN.
5. Jointly solve for a candidate parameter step without constructing a higher-order computational graph that backpropagates through the solver.
6. Perform backtracking checks using the actual source classification CE and actual margin changes at fixed boundary points.
7. Update the classification module upon acceptance; otherwise, retain the current state. Locate the boundary again in the next round.

**Stage 4: Output the model.**

Retain $\theta_0,\mu_H,s_H,\psi^\star$. Discard the score network used for training, boundary points, response matrices, and local solver state.

Corresponding pseudocode:

```text
Input: labeled complete source graphs S, unlabeled complete target graphs T, predetermined training and numerical configuration

θ0, W0, b0 ← source-supervised initialization, or reuse a trained source model
Freeze θ0; cache hθ0(S), hθ0(T)
μH, sH ← source representation statistics
xS, xT ← apply the same fixed coordinate transformation
ψ ← initialize the nonlinear classification module to be equivalent to the original source predictions

σ ← predetermined scale rule depending only on target inputs
ω* ← minimize denoising score matching on xT; then freeze

for t = 0, ..., Tadapt-1:
    BS, BT ← source classification batch and unlabeled target batch
    ψbar ← current classification module parameters
    I ← empty set

    for xj in BT:
        (cj, dj) ← two highest-scoring classes in the current prediction, used only for localization
        ξj ← solve mcj,dj(ξj) ≈ 0 within a bounded neighborhood of xj
        if projection fails the root-accuracy, minimum-gradient, or valid class-face conditions:
            continue
        nj, κj ← margin normal and gradient norm of the old classifier at ξj
        γj ← njᵀ sω*(ξj)
        νj ← clipped -τ σ² γj
        ζj ← -νj
        Add to I; fix these reference quantities

    if I is empty or all displacement requests are numerically zero:
        continue

    D ← compute the normalized parameter response at fixed ξj
    gS, FS ← source CE gradient and positive semidefinite GGN
    δ ← solve (FS + λB DᵀD/|I| + μI) δ = -gS + λB Dᵀζ/|I|
    α ← backtrack using the actual objective with frozen reference quantities and the finite-change budget
    if a valid α is obtained:
        ψ ← ψbar + α δ
    else:
        ψ ← ψbar

Output: frozen graph encoder, fixed coordinate transformation, final classification module
```

### 3.17 Inference

For a new complete target graph $G$, compute:

$$
G
\;\longrightarrow\;
h_{\theta_0}(G)
\;\longrightarrow\;
x(G)
\;\longrightarrow\;
f_{\psi^\star}(x(G))
\;\longrightarrow\;
p_{\psi^\star}(G).
$$

Inference requires no source graphs, source labels, target data pool, boundary projection, or score model. The main computational overhead relative to the source model is a small residual classification branch.

### 3.18 Substantive Differences from TQG v2

| Design aspect | TQG v2 | TSBA candidate design |
| --- | --- | --- |
| Target information | Defines the gradient query space and reading geometry | Directly provides distribution gradients near the boundary |
| Role of source labels | Determines the residual/gradient coefficients to transfer | Establishes class semantics and constrains changes to the decision rule |
| Surrogate for target supervision | Target residuals or gradients read from the source domain | Boundary-normal displacement supplied by the target distribution |
| Role of the response matrix | Query function and object of cross-domain reading | Converts boundary changes into parameter changes |
| Adapted parameters | Target tangent space of the full GNN | Small nonlinear classification module on fixed whole-graph coordinates |
| Internal solve | Kernel matrix, joint reading weights, and spectral error | Local parameter step formed from the source GGN and boundary response |
| Main operating conditions | Transferable conditional relationships given signatures; source support for target queries | Representations contain task information; low-density boundaries have local task relevance; source semantics can serve as anchors |
| Claim | Recover the target classification gradient supported by source supervision | Construct boundary adaptation updates driven by the target distribution and constrained by source semantics |

This change does introduce different target information, but it is not a new correctness result established by the existing experiments. In particular, failure of source residual transfer must not be taken as evidence that target low-density separation will necessarily succeed.

### 3.19 Scope and Contribution Claims

**Limits of label information.** Density does not provide class names. When source supervision and the initial model cannot determine the semantics of a region, the target score cannot resolve arbitrary label permutations or changes in conditional rules.

**Limits of representation.** The main version freezes the backbone. The method may be insufficient when the target task requires new message-passing features; graph-level classifier adaptation must not be described as learning the entire encoder.

**Limits of low-density boundaries.** Classes can overlap heavily in representation space, and the correct boundary can pass through high-density regions. Differences in class priors can also shift the Bayes boundary away from density valleys. In these cases, movement toward low density need not improve target classification.

**Limits of density estimation.** Scores estimated from finite, high-dimensional samples may be inaccurate; Gaussian smoothing changes the distribution, and boundary points may lie off the true representation manifold. Limiting the projection range and controlling step sizes can only limit risk; they cannot prove that the score is reliable at these locations.

**Limits regarding collapse and stagnation.** Boundaries may also be pushed beyond the data or stop at incorrect locations where the score is zero. The source classification constraint provides no universal guarantee against collapse or of global optimality.

**Limits of execution.** The linear system and backtracking ensure numerical consistency with the defined surrogate objective; they do not observe target labels and cannot directly verify a decrease in true target risk.

The research claim this design can make is:

> **Use the normal component of the score of the target whole-graph distribution to propose boundary changes, then realize those changes through a neural response solve constrained by source labels; thereby investigate how the unlabeled target distribution can actually help determine classification updates, rather than merely change how source supervision is read.**

Claims that should not be made directly include: recovering the true target CE gradient; resolving arbitrary graph domain shifts; equating density reduction with classification-risk reduction; claiming that existing diagnostics establish the low-density assumption; or claiming that combining existing components alone already meets the standard for ICML acceptance.
