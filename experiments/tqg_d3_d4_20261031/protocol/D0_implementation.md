# D0：新增诊断接口与代数验收

[返回实验列表](README.md#experiment-list) · [统一协议](method.md) · 下一项：[D1](D1_gradient_quality.md)

**优先级P0；待实现、待执行；不做真实数据训练。** 目的：确保后面的“好／坏梯度”判断来自正确的诊断，而不是新脚本的错误。历史116项测试和独立求解器检查是旧证据，不冒充本项新结果。

## 假设与范围

新增诊断接口能复用当前数值方法，重构经验CE梯度并执行一次正确参数步，同时阻止Oracle进入正常训练。所有工程验收都通过后才能解释D1/2。

## 输入与小规模检查

先使用合成小网络，C=2及C=3，FP64；再用Mutagenicity种子3405的初始源fit模型，在三个任务各1个固定开发A块验收。真实图标签读取只经development Oracle接口；不访问final。

| 检查 | 具体做法 | 通过要求 |
|---|---|---|
| 数值与模型状态 | 快照复制全部可训练参数及buffer；查询和损失前向同为eval | layout、buffer、dtype不变；值有限 |
| 方向与符号 | autograd直接算g_A，与V及真实对比残差重构比较 | 相对L2误差≤1e-8；零梯度单列 |
| 源响应 | 少量源图JVP对显式J_tilde V | 相对误差≤1e-8 |
| 辅助梯度与执行 | auxiliary梯度对−Vd；真实SGD增量对−η(g_S+βg_hat) | absolute≤1e-10＋relative≤1e-8 |
| 标签隔离 | 同θ/输入打乱源query标签，重新求W | W相对变化≤1e-10；允许r_S和d改变 |
| Oracle权限 | 未开诊断权限或给入final ID时请求标签 | 必须拒绝；普通训练target batch不携带y |
| 近零边界 | g_A=0、g_hat=0、rank0、source norm0 | NA及原因明确；无NaN/Inf或伪造余弦 |
| 固定Ω等价 | Ω=I/r的条件解，与mean-only且lambda_W/2比较 | W相对误差≤1e-8 |
| 无支持 | 合成K_ST=0，保持PSD核结构 | W、d、反馈数值为零至1e-10 |
| 统一缩放 | 非退化签名A与7A分别重算各自中位数带宽 | 核相对差≤1e-10；floor生效时另报，不硬套不变性 |

这里relative分母为max(reference norm,1e-12)；reference近零时另要求绝对误差。对明显非PSD、非有限数值应停止该工程验收并定位，不以源-only回退掩盖新脚本问题。

## 实施约束

- 分离无标签查询构造、source残差读取、development Oracle、单步临时更新四个接口。
- 单步验证用clone／functional参数；每臂结束核对原θ和buffer哈希未改。
- 标签日志核对A/B⊂原development、与final无交集；D0本身也有日志。
- 使用8个预定块的第0块，不先看效果再选“能过”的真实图。
- 不以新诊断改写现有training.py或旧checkpoint；必要修复另开提交、重新校验数值核心并记录影响。

## 结果与继续条件

输出engineering_checks.json、permission_checks.json、snapshot_manifest.json、unit_checks.log；列出每例参数、容差、误差及失败原因，不能只写“通过”。

任何核心等式或权限失败：**停止D1/2**。修复后保留旧失败日志，重新冻结执行代码；工程通过不代表研究假设通过。

本项不以BA判断成功；没有“预期提升几个百分点”的要求。
