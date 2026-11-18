# 共同域解释驱动的共享分类更新

`shared-update` 是独立的图级无监督域适配方法。它复用 TSBA 的数据角色、源 GNN 和源交叉熵预训练。适配阶段固定源模型，通过共同域解释推断目标类别与图特征的联合统计，然后学习一个共享 logit 修正。原始 TSBA 仍可作为独立方法运行。

## 流程与代码

1. 从源训练图估计逐维均值和标准差，固定标准化。目标图的更新特征为标准化表示加常数项，记作 $u_i$，源模型输出原始 logits $z_{0,i}$。源验证集的标签用于源模型选模。
2. 用源类条件对角高斯、共享正对角仿射映射和目标类比例定义域解释。计算源类条件 NLL 与目标边缘 NLL，目标边缘密度包含映射的 Jacobian。拟合参考解释后，固定观测阈值为参考平均 NLL 加 `tau`。源高斯参数也是允许搜索的变量。见 [解释拟合与搜索](../../tsba/shared_update/explanations.py)。
3. 每个解释在全部目标图上计算类别概率 $q_i$，并压缩为 $S=n_T^{-1}\sum_i q_i u_i^\top$。使用有限候选的凸主问题求 $W$，目标为候选中的最坏相对交叉熵变化加 L2 正则。固定当前 $W$，以受约束的多起点局部优化搜索更不利的解释。新解释通过相容性和违约检查后加入主问题。达到搜索预算或没有发现显著违约时停止。见 [主问题与约束生成](../../tsba/shared_update/solver.py)。
4. 预测使用 $z_i=z_{0,i}+Wu_i$。保存的模型包含源模型、标准化参数和 $W$，可以独立对新图预测。见 [推断与存档](../../tsba/shared_update/predict.py)。

候选概率对应的有限更新风险差为 $A(W)-\langle W,S\rangle_F$，其中 $A$ 是平均 log-sum-exp 变化。日志记录参考损失、相容阈值、搜索过程和已纳入候选上的风险。当前求解使用有限候选与局部内层搜索，结果以实际找到的候选为范围。`tau` 是观测拟合容差，覆盖率由后续统计验证决定。

正对角映射限定各坐标尺度和偏移，保留方向并排除旋转。源均值在经验类均值的 `mean_radius` 邻域内调整，方差下限为 `variance_floor`，上限为经下限处理的类经验方差的 100 倍。实际边界记录在参考拟合日志中。主问题通过可行性、KKT 残差和零修正比较后采用候选；参考或主问题无效时，输出零修正并记录回退原因。

## 运行

```sh
python -m pip install -e '.[shared-update]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.run_task --method shared-update --synthetic --output /tmp/shared-update-smoke
```

已安装的 `tsba` 命令也接受 `--method shared-update`。真实任务使用 `--input` 传入[现有数据格式](../guides/usage.md)，为源训练、源验证和无标签目标声明独立角色。`--source-checkpoint` 可以复用源数据、角色和源配置匹配的 `tsba-source-v1` 文件。输出目录须为新目录。

`--config` 接受两个 JSON 对象。例如：

```json
{
  "source": {"seed": 3405, "emb_dim": 16, "source_epochs": 100},
  "adaptation": {"tau": 0.01, "regularization": 0.1, "inner_starts": 5, "search_rounds": 10}
}
```

`source` 可配置源模型结构、训练和设备。`adaptation` 的完整范围及数值默认值见 [SharedUpdateConfig](../../tsba/shared_update/config.py)。统计求解使用 CPU float64，图编码使用源模型指定设备。数值配置是参考实现的初始设置，真实图的模型范围与阈值仍需验证。

| 输出 | 内容 |
|---|---|
| `source.pt` | 可复用的冻结源模型 |
| `model.pt` | `shared-update-predictor-v1`，含源模型与修正 |
| `predictions.pt` | 目标图 ID、源概率与适配后概率 |
| `result.json` | 输入身份、配置、源选模、解释与求解记录 |

```python
from tsba.shared_update.predict import load_shared_predictor

model = load_shared_predictor("/tmp/shared-update-smoke/model.pt")
probabilities = model.predict(graphs)
```

合成图入口验证工程链路。真实图上集合是否覆盖任务需要的监督统计、能否产生分类纠错，留待使用封存后评价的实验验证。
