"""Build a meeting brief from existing, immutable experimental CSV/JSON data."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / 'docs/v2/results/assumptions_brief_20261105.md'
SOURCES = {}


def source(name, file):
    p = REPO / 'experiments' / name / file
    SOURCES[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def rows(name, file):
    with source(name, file).open() as f:
        return list(csv.DictReader(f))


def data(name, file):
    return json.loads(source(name, file).read_text())


def table(headers, values):
    def highlight(row):
        cells = list(map(str, row))
        marked = set()
        # Formatting only: preserve all numeric strings and all rows. Bold marks
        # discussion points, never an additional statistical success criterion.
        if headers == ['签名','读取器','源内可学','目标内可学']:
            marked.update([2, 3])
        if '迁移缺口' in headers:
            marked.add(headers.index('迁移缺口'))
        if '选参S→S' in headers:
            marked.update([headers.index('参照S→S'), headers.index('选参S→S')])
            if cells[0] in ['0_to_2','0_to_3']:
                marked.update([headers.index('选参S→T'), headers.index('诊断T→T')])
        if '最小组图数' in headers:
            if int(cells[3]) < 30:
                marked.add(3)
            marked.update([4, 6])
        if headers == ['设置','0→1','0→2','0→3']:
            if cells[0].startswith('H_128_NEAR_H_'):
                marked.update([1, 2, 3])
            else:
                marked.update(i for i in [1, 2, 3] if cells[i].startswith('-'))
        if '替代口径通过' in headers and cells[-1] != cells[-2]:
            marked.update(range(2, len(cells)))
        if headers == ['方法','0→1','0→2','0→3'] and cells[0] in ['LOCAL','KERNEL']:
            marked.update([0, 1, 2, 3])
        if '综合通过' in headers:
            marked.add(len(cells)-1)
            if cells[:2] == ['0_to_3','LOCAL']:
                marked.update(range(2,len(cells)))
        if '真标签更好种子' in headers:
            marked.update([3,4])
        if headers[-2:] == ['源内可学','目标内可学'] and 'S→S NLL' in headers:
            marked.update([len(cells)-2,len(cells)-1])
        return [f'**{cell}**' if i in marked else cell for i,cell in enumerate(cells)]
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(highlight(row)) + ' |' for row in values]) + '\n\n'


def f(value):
    return f'{float(value):.4f}'


def mean_sd(values, scale=1):
    assert len(values) == 5
    return f'{statistics.mean(values)*scale:.4f} ± {statistics.stdev(values)*scale:.4f}'


def stat(item, scale=1):
    return f"{item['mean']*scale:.5f} ± {item['std']*scale:.5f}（{item['positive_seeds']}/5）"


def main():
    assumptions = 'tqg_assumptions_20261103'
    proteins = 'tqg_assumptions_proteins_20261104'
    kernel = 'tqg_optimization_a4_20261104'
    local = 'tqg_optimization_local_20261104'
    feedback = 'tqg_optimization_feedback_20261104'
    triangle = rows(assumptions, 'triangle_summary.csv')
    support = rows(assumptions, 'support_summary.csv')
    curves = rows(kernel, 'summary.csv')
    local_raw = rows(local, 'results.csv')
    a2 = data(local, 'results_summary.json')
    a2_audit = data(local, 'summary_audit.json')
    a6 = data(feedback, 'results_summary.json')
    protein_triangle = rows(proteins, 'triangle_summary.csv')
    protein_support = rows(proteins, 'support_summary.csv')
    text = ['# §3.8 四项条件：口头汇报与真实实验数据\n\n2026-11-05。本文只汇总既有实验，不启动训练、不修改旧结果。数值由文末CSV/JSON自动读取，生成脚本为`scripts/build_assumptions_brief.py`。\n\n']
    text += ['## 一段话汇报\n\n这几天主要围绕TQG §3.8的四个前提做了验证：签名是否保留任务信息、源目标标签规律是否相容、核读取器能否用有限监督学到残差规律，以及源查询图能否约束目标需要的部分。主实验在Mutagenicity的0→1/2/3三个方向、五个模型种子上进行，冻结已有骨干：先用H和HA训练线性分类器、MLP和残差核回归，做源内、跨域和目标内的同监督量比较；再检查近远区域、近邻源图与随机源图、核带宽和正则；最后用梯度方向、乱标签/随机反馈和独立一步更新验收。结果是签名有可读标签信息，但存在稳定迁移缺口；调核改善了源内预测，近邻也在部分设置改善方向，却没有形成跨任务稳定反馈，KERNEL和LOCAL完整验收均0/3。PROTEINS基础复验的可学性和支持分组证据不足。目前不能说四个假设都不成立，更准确的结论是：从源信息可读到可靠目标更新的证据链尚未建立。\n\n']
    text += ['## 统一设置与术语\n\n']
    text += [table(['项目', '实际设置'], [
        ['主数据集/任务', 'Mutagenicity；0→1、0→2、0→3；沿用SEGA图规模域索引，不是官方自然域'],
        ['随机种子', '3405—3409，共五个模型种子；不是五次独立数据集划分'],
        ['源数据角色', 'fit779、query196、validation109；query不混入骨干已见fit'],
        ['目标开发池', 'A271、B271；目标final不读取'],
        ['初始条件探针', '冻结各自源模型；四外折/三内折，源训练147、源留出49，目标训练同样147，B评价271'],
        ['核学习曲线', '训练量32/64/128/147；带宽乘数0.5/1/2作用于标准差；λ=0.001/0.01/0.1/1'],
        ['局部源图实验', '仅初始快照；A16/fit16；64/128张随机、H近邻、HA近邻及全196；120目标块'],
        ['反馈验收', '源监督0/50/200快照；A16/fit16；SGD学习率0.0003、反馈权重0.1；360块，前4块做独立一步'],
        ['H / HA', 'H是图表示；HA额外包含沿目标查询方向的模型响应'],
        ['目标标签权限', '目标标签只用于明确登记的诊断探针、Oracle和封存后评价，不用于候选适配选参'],
        ['训练含义', '训练了小探针/读取器；没有本轮新增骨干预训练或连续TQG适配'],
    ])]
    text += ['NLL越低越好；方向余弦越高越好；相对梯度误差越低越好。文中均值按各实验原协议汇总；±为五模型种子标准差。随机bank/标签置乱/目标块/外折均先在同一模型内汇总。不同轮次的NLL不可直接拼成严格前后对照。\n\n']
    text += ['## 实验一：签名任务充分性\n\n**问题：** 压缩成H/HA后，是否还保留判断标签需要的信息？\n\n**做法：** 冻结骨干，分别训练线性分类器（LR）、小MLP、残差核回归（KRR）。与训练类比例预测和三次打乱标签对照比较。通过筛选要求NLL改善至少0.02且至少4/5种子同向；这是可学性筛选，不是数学上的充分性证明。\n\n']
    counts = []
    for sig in ['H', 'HA']:
        for family in ['KRR', 'LR', 'MLP']:
            selected = [r for r in triangle if r['signature'] == sig and r['family'] == family]
            assert len(selected) == 3
            counts.append([sig, family, f"{sum(r['S_S_learnable']=='True' for r in selected)}/3", f"{sum(r['T_T_learnable']=='True' for r in selected)}/3"])
    text += [table(['签名','读取器','源内可学','目标内可学'], counts)]
    text += ['**结果判断：** H/HA均有可读的任务信息；但能预测标签不等于保留完整图的全部必要信息。H源内三个任务共享同一源池和模型，不当三份独立证据。完整源内数值见下表。\n\n']
    text += ['## 实验二：跨域条件稳定性\n\n**问题：** 相似签名下，源标签规律能否用于目标？\n\n**做法：** S→S检查源内是否可学；S→T检查源规律迁移；T→T使用目标诊断标签作为定位参照。两种训练均147张，S→T和T→T在同一B池评价。目标训练不是可部署UDA方法。\n\n']
    text += [table(['任务','签名','读取器','S→S NLL','S→T NLL','T→T NLL','迁移缺口','缺口同向种子'],
        [[r['task'],r['signature'],r['family'],f(r['S_S_NLL']),f(r['S_T_NLL']),f(r['T_T_NLL']),f(r['transfer_gap']),r['gap_positive_seeds']+'/5'] for r in triangle])]
    text += ['**结果判断：** H/HA的LR/MLP三个任务均有稳定跨域缺口；这可能混合条件变化、覆盖不足、类比例和有限样本拟合，不能单凭分数差证明η_S≠η_T。H-LR在0→2近邻114图的局部相容是描述性线索，不是跨任务普遍稳定性。\n\n']
    text += ['## 实验三：核读取复杂度与有限监督\n\n**问题：** 信息存在，但原读取器是否没有把它读好？\n\n**做法：** 对样本量、带宽、正则做完整学习曲线；每次通过训练域内部验证选参。FIXED保持λ，NORMALIZED随子集大小m按λ×m/147调整。T→T仅诊断，候选KERNEL只用源标签选择。\n\n### n147同轮参照和优化结果\n\n“固定参数参照”指本轮同一协议的带宽1、基础λ0.01，不是上一轮A1分数。不能把内层选参改善说成跨域有效。\n\n']
    def get(task,sig,conv,n,domain,pool):
        result = [r for r in curves if r['task']==task and r['signature']==sig and r['convention']==conv and int(r['n'])==n and r['train_domain']==domain and r['pool']==pool]
        assert len(result)==1
        return result[0]
    details=[];learning=[]
    for task in ['0_to_1','0_to_2','0_to_3']:
        for sig in ['H','HA']:
            for conv in ['FIXED','NORMALIZED']:
                ss,st,tt=[get(task,sig,conv,147,*role) for role in [('S','S_audit'),('S','B'),('T','B')]]
                details.append([task,sig,conv,f(ss['reference_nll']),f(ss['nll']),f(st['nll']),f(tt['nll']),f"{float(st['outside'])*100:.2f}%"])
                for n in [32,64,128,147]:
                    rr=[get(task,sig,conv,n,*role) for role in [('S','S_audit'),('S','B'),('T','B')]]
                    learning.append([task,sig,conv,n]+[f"{float(x['nll']):.4f} ± {float(x['nll_sd']):.4f}" for x in rr])
    text += [table(['任务','签名','正则','参照S→S','选参S→S','选参S→T','诊断T→T','S→T原概率越界'],details)]
    text += ['核残差恢复出的原始概率可能越过[0,1]；NLL按预登记裁剪，原始MSE和越界率同时保存，高NLL不能全部归因于条件关系变化。\n\n<details>\n<summary>展开完整32／64／128／147学习曲线（48行，均值±标准差）</summary>\n\n',table(['任务','签名','正则','训练图数','S→S NLL','S→T NLL','T→T NLL'],learning),'</details>\n\n']
    text += ['**结果判断：** 原HA-KRR源内门槛未过而同签名LR/MLP可学；本轮调核改善源内预测，但跨域差距未消失。核函数族是否存在适度复杂的真实残差近似，尚未被直接证明或否定。\n\n']
    text += ['## 实验四：源监督支持与局部选图\n\n**问题：** 源经验是否足以回答目标问题？挑相关源图是否更好？\n\n**第一步：** 用源训练图留一近邻距离90%分位数划分目标近/远组；每折每组至少30张才作稳定筛选。比较远组迁移缺口减近组迁移缺口，正值表示远组更困难。\n\n']
    text += [table(['任务','签名','读取器','最小组图数','远−近迁移缺口','同向种子','支持代理筛选'],
        [[r['task'],r['signature'],r['family'],r['min_stratum_count'],f(r['gap_far_minus_near']),r['positive_seeds']+'/5',r['support_candidate']] for r in support])]
    text += ['**第二步：** 固定原JOINT读取，196源query中选64/128张随机或近邻图。只用A16输入选图，不用B；比较H距离与HA距离。公共带宽固定，正则为0.01或0.01×n/196。下面为初始快照，不能与A6三个快照平均的数值直接混比。\n\n']
    grouped=defaultdict(lambda:defaultdict(list))
    for r in local_raw:
        if r['arm']!='FULL':continue
        assert int(r['snapshot_step'])==0
        parts=r['profile'].split('_')
        profile='_'.join(parts[:3]+parts[4:]) if 'UNIFORM' in parts else r['profile']
        for metric in ['c_A','e_A']:
            grouped[profile,r['task'],metric][int(r['seed'])].append(float(r[metric]))
    for sig in ['H','HA']:
        names=sorted({key[0] for key in grouped if key[0].split('_')[0]==sig})
        assert len(names)==13
        for metric,title in [('c_A','方向余弦（越大越好）'),('e_A','相对梯度误差（越小越好）')]:
            values=[]
            for name in names:
                cells=[]
                for task in ['0_to_1','0_to_2','0_to_3']:
                    samples=grouped[name,task,metric];assert set(samples)==set(range(3405,3410))
                    cells.append(mean_sd([statistics.mean(samples[seed]) for seed in range(3405,3410)]))
                values.append([name]+cells)
            text += [f'### {sig}读取核：{title}\n\n',table(['设置','0→1','0→2','0→3'],values)]
    text += ['命名例：H_128_NEAR_HA_FIXED表示H读取核、128张HA近邻、固定正则；UNIFORM是随机，FULL是全196。五个随机排列先在每个模型内平均。\n\n']
    alt={(r['task'],r['profile']):r for r in a2_audit['alternate_A2_error_aggregation']}
    comparisons=[]
    for r in a2['results']:
        x=alt[r['task'],r['profile']]
        comparisons.append([r['task'],r['profile'],stat(r['c_delta']),stat(r['error_reduction'],100),f"{x['ratio_of_seed_mean_errors']*100:.3f}%",r['passed'],x['passed']])
    text += ['<details>\n<summary>展开近邻对同数量随机的完整配对差与门槛（48行）</summary>\n\n',table(['任务','配置','余弦提升','逐块误差降低%','先汇总误差再算比例','主口径通过','替代口径通过'],comparisons),'</details>\n\n']
    text += ['**结果判断：** H距离远组更困难，但没有局部配置达到至少2/3任务通过。原协议未明确误差比例的聚合顺序；替代口径使H_128_NEAR_H_FIXED在0→2单独通过，不能省略这个正信号，也不改变跨任务结论。近邻选择同时改变相关性、样本量和类别构成，不能当作数学支持充分性的证明。\n\n']
    text += ['## 最终验收：预测改善能否变成可靠目标更新？\n\nLOCAL和KERNEL各为一条预登记、只用源内验证选择的算法路线，不按目标B挑配置。LOCAL读取64/128张，但选参接触196张源标签；不能声称只需64/128个监督标签。两条路线的全流程乱标签对照均重做选择。\n\n### 独立一步结果\n\n收益=(CE_B(源SGD后)−CE_B(源SGD+反馈后))/CE_B(更新前)。下表单位%，正值为好，门槛0.01%；不是准确率百分点。0/50/200均为源监督快照，每一步独立开始，不是TQG连续轨迹。括号是正值种子数。\n\n']
    profiles=['OLD_H','OLD_HA','MEAN_H','MEAN_HA','LOCAL','KERNEL','KERNEL_JOINT','TEACHER_H','TEACHER_HA','TEACHER_LOCAL']
    lookup={(r['profile'],r['task']):r for r in a6['results']}
    text += [table(['方法','0→1','0→2','0→3'],[[p]+[stat(lookup[p,f'0_to_{d}']['step']['FULL'],100) for d in [1,2,3]] for p in profiles])]
    text += ['OLD是旧JOINT；MEAN是固定简单读取；KERNEL是源选参MEAN；KERNEL_JOINT是同参数匹配正则的参照，不是另挑的候选；TEACHER用源标签训练LR，再给目标软标签。\n\n### 两个候选的方向与标签归因\n\nBASE保留类比例与模型预测部分；BASE_M匹配完整反馈幅度；CPL/CRL保留BASE、仅替换标签项；完整等幅置乱另重复源端选择。BASE可能受源端选参间接影响，不是源标签完全无关的对照。\n\n']
    selected=[r for r in a6['results'] if r['profile'] in ['LOCAL','KERNEL']]
    text += [table(['任务','候选','方向余弦','方向对全流程置乱优势','单步对BASE优势%','单步对BASE_M优势%','综合通过'],
        [[r['task'],r['profile'],stat(r['gradient']['FULL']),stat(r['gradient']['DELTA_FULL_PERM_M_AVG']),stat(r['step']['DELTA_BASE'],100),stat(r['step']['DELTA_BASE_M'],100),r['joint_pass']] for r in selected])]
    text += [table(['任务','候选','单步对CPL优势%','单步对CRL优势%','单步对全幅置乱优势%','单步对全幅随机优势%'],
        [[r['task'],r['profile']]+[stat(r['step'][m],100) for m in ['DELTA_CPL_AVG','DELTA_CRL_AVG','DELTA_FULL_PERM_M_AVG','DELTA_FULL_RAND_M_AVG']] for r in selected])]
    text += ['综合要求至少2/3任务同时通过方向、独立一步、种子一致性和覆盖，并且后期快照不普遍反向。LOCAL/KERNEL均0/3。LOCAL0→3平均收益为正，但对BASE仅3/5改善、对BASE_M仅1/5；方向对置乱优势0.04546低于0.05，50快照3/5方向非正。\n\n']
    teacher=rows(feedback,'teacher_prediction_seeds.csv');teachers=[]
    for task in ['0_to_1','0_to_2','0_to_3']:
        for profile in ['TEACHER_H','TEACHER_HA','TEACHER_LOCAL']:
            true={int(r['seed']):float(r['nll']) for r in teacher if r['task']==task and r['profile']==profile and r['pool']=='B' and int(r['variant'])==0}
            null=defaultdict(list)
            for r in teacher:
                if r['task']==task and r['profile']==profile and r['pool']=='B' and int(r['variant'])>0:null[int(r['seed'])].append(float(r['nll']))
            delta=[statistics.mean(null[s])-true[s] for s in sorted(true)]
            teachers.append([task,profile,mean_sd(list(true.values())),mean_sd(delta),f'{sum(x>0 for x in delta)}/5'])
    text += ['### 教师直接预测（B池NLL）\n\n',table(['任务','教师','真实标签NLL','乱标签NLL−真实标签NLL','真标签更好种子'],teachers)]
    text += ['教师在0→1/2直接预测优于乱标签，说明并非没有源标签预测迁移；不等于有稳定梯度增量或TQG机制优势。局部教师此处固定所选bank做标签置乱，全流程重新选bank的反馈控制另行保存。方法间单步未额外强制等幅，不将幅度差归为机制优越。\n\n']
    text += ['## PROTEINS基础复验\n\n使用PROTEINS自己的五个预训练GIN，不是把Mutagenicity模型搬过来。fit199/query51（类别42/9），源/目标每折训练38或39张，源留出13或12张，目标A69/B70。骨干、预训练预算、样本量同时不同，不能把跨数据集差别唯一归因于样本少。\n\n']
    text += [table(['任务','签名','读取器','S→S NLL','S→T NLL','T→T NLL','源内可学','目标内可学'],
        [[r['task'],r['signature'],r['family'],f(r['S_S_NLL']),f(r['S_T_NLL']),f(r['T_T_NLL']),r['S_S_learnable'],r['T_T_learnable']] for r in protein_triangle])]
    text += ['源内/目标内主门槛均未过；支持分组最少2～7张，未达每组30张要求。0→2 H-MLP有平均改善但只有3/5种子同向，不能改写成完全无信息。没有可学前提，不能仅凭S→T与T→T差距宣布条件失稳。\n\n']
    text += ['## 结论、边界与下一步\n\n']
    text += [table(['条件','当前判断'],[
        ['签名任务充分性','有可读信息得到支持；完整任务充分性未证明'],
        ['跨域条件稳定性','普遍相容未获支持；跨域缺口存在，但原因未完全分离'],
        ['函数复杂度/读取器','当前读取方式有局限；调核改善源内预测，尚未解决反馈'],
        ['源监督支持','距离有诊断价值；源监督足以约束目标查询未证明'],
    ])]
    text += ['下一步应先明确可检验的迁移条件，区分源预测误差与目标Jacobian加权查询误差；如研究更多源标签，应使用更多独立监督并登记新的预算，而不是把骨干已见fit直接补进旧query。当前不继续加长训练、不按目标结果筛种子或堆模块。组合、连续200步及其基线、新PROTEINS适配和支持门控，均因A6条件未触发而未执行。\n\n']
    text += ['工程审计已核对读出、控制系数、数据权限和实际参数更新。A2-b/A6共10560次FULL/BASE普通模型重放一致；不称所有代码绝无错误。A4有1条JOINT观察器、A6有1条主读出和1条教师内折有限未收敛，均保留。开发A/B历史已反复观察，不能称盲测，final读取0。所有本次表格为既有产物再汇总，不是新增训练证据。\n\n']
    text += ['## 原始数据、复核与详细报告\n\n']
    for name,label in [(assumptions,'Mutagenicity条件诊断'),(proteins,'PROTEINS基础复验'),(kernel,'核学习曲线'),(local,'局部源bank'),(feedback,'反馈与教师')]:
        text += [f'- {label}：[报告]({name}/结果解读.md) · [逐种子结果](../../../experiments/{name}/results.csv) · [原始归档](../../../experiments/{name}/README.md)。\n']
    text += ['\n历史报告中的“下一步未执行”是当轮状态；后续A2-b/A4-a/A6现均已完成，以[完整研究结论](optimization_20261104/README.md)为准。下列SHA绑定本次直接读入的汇总输入；更底层逐图预测、标签访问、封存日志及模型参数由各归档artifact_manifest和audit追溯。\n\n']
    text += [table(['直接输入文件','SHA256'],[[f'[{p}](../../../{p})',f'`{sha}`'] for p,sha in sorted(SOURCES.items())])]
    document = ''.join(text)
    notes = {
        '实验一：签名任务充分性': '**设置速读：** 冻结骨干；H/HA×KRR/LR/MLP；四外折、三内折；每折训练147张，源留出49张、目标B评价271张；五模型种子。\n\n**“可学”通过标准（同时满足）：** ①真实标签模型较类别比例预测的平均NLL降低≥**0.02**；②较三次乱标签模型均值的平均NLL降低≥**0.02**；③至少**4/5个模型种子同时胜过这两个对照**。NLL是给真实类别的概率取负对数后的平均值，越低越好；0.02不是准确率提高2个百分点。表中3/3表示三个迁移任务都过筛选，不是准确率100%。',
        '实验二：跨域条件稳定性': '**设置速读：** 源训练与目标诊断训练各147张；在同一B271上比较S→T和T→T，重复五模型种子。\n\n**“迁移缺口候选”标准：** 目标内先通过实验一的可学性筛选；随后S→T的平均NLL比T→T高≥**0.02**，且至少**4/5种子**差值为正。缺口=S→T−T→T。这筛选的是迁移障碍，不是直接判定数学上的条件稳定性成立或不成立；正缺口越大，源规律迁过去损失越明显。',
        '实验三：核读取复杂度与有限监督': '**设置速读：** H/HA；训练图数**32/64/128/147**；核带宽乘数**0.5/1/2**；正则λ为**0.001/0.01/0.1/1**；保留固定与按样本数归一化两种约定。四外折、三内折、五模型种子；按训练域内折NLL选参。\n\n**如何判断：** 本组学习曲线用于诊断，没有“源内NLL变低就说明TQG通过”的独立标准。表内对比同轮固定参数参照与选参结果；源内改善之后，KERNEL还必须通过文末A6的方向、对照、单步和稳定性门槛。目标内训练仅诊断，不能拿其参数当适配候选。',
        '实验四：源监督支持与局部选图': '**设置速读：** 先在目标B上做近/远分层，再保持原JOINT读取器，比较源query中**64/128张**随机、H近邻、HA近邻与全**196张**。初始模型、A16、五种子；随机bank每个模型五个排列。\n\n**近远分层标准：** 远组迁移缺口比近组至少大**0.02**，至少**4/5种子**同向，且每折近/远组均≥**30张**。这是支持相关性，不是因果或数学支持证明。\n\n**局部选图标准：** 相比同数量、同读取核、同正则的随机bank，平均方向余弦提升≥**0.05**、相对梯度误差降低≥**10%**，两项各至少**4/5种子**同向，数值可比覆盖≥**80%**；至少**2/3任务**满足才算跨任务支持。单看某格余弦变大不算通过；两种误差聚合口径完整保留。',
        '最终验收：预测改善能否变成可靠目标更新？': '**设置速读：** 源监督**0/50/200**快照；每块fit16、A16，B271评价；学习率**0.0003**、反馈权重**0.1**。所有更新各自从同一快照独立开始。\n\n**完整A6门槛：** 每任务方向余弦>0，对完整等幅置乱与随机反馈的平均余弦优势各≥**0.05**；相对源SGD额外相对CE收益≥**1e−4（即0.01%）**，且优于BASE和两类完整等幅null。各项至少**4/5种子**同向、可比覆盖≥**80%**；50/200快照不能有≥3/5种子方向或单步收益非正的普遍反向；至少**2/3任务**共同通过。对教师/MEAN增量另列，不用它替代标签归因。',
        'PROTEINS基础复验': '**设置速读：** 独立的PROTEINS预训练模型；源query**51张（42/9两类）**；源/目标每折训练**38或39张**；目标B**70张**；五模型种子。\n\n**标准不降低：** 可学性沿用实验一的NLL改善≥**0.02**和至少**4/5种子同时胜过两对照**；支持分层仍要求每组≥**30张**。实际最小组仅**2～7张**，应记为证据不足，不应解释成“距离无用”。',
    }
    for heading, note in notes.items():
        marker = f'## {heading}\n\n'
        assert document.count(marker) == 1
        document = document.replace(marker, marker + note + '\n\n', 1)
    legend = '**表格加粗说明：** 加粗的是汇报需要关注的对照差异、负方向、样本不足、关键候选及口径敏感性，**不等于该数据通过检验，也不表示挑出的最优结果**。尤其H读取核＋128张H近邻虽有方向改善，仍须同时看误差与完整门槛；所有未加粗结果也完整保留。\n\n'
    document = document.replace('## 统一设置与术语\n\n', '## 统一设置与术语\n\n' + legend, 1)
    DEST.write_text(document.rstrip() + '\n')
    print(json.dumps(dict(document=str(DEST),source_files=len(SOURCES),triangle_rows=len(triangle),learning_curve_rows=len(learning),local_profiles=len({k[0] for k in grouped}),candidate_rows=len(selected))))


if __name__ == '__main__':
    main()
