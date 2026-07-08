# 生成式推荐广告排序推理性能优化

## 赛道概要背景

传统广告排序模型已难以满足个性化推荐需求，生成式广告排序模型凭借强大的序列建模与语义理解能力成为行业趋势。该类模型依托 Transformer 架构，能深度挖掘用户点击、转化等超长行为序列中的长距离依赖关系，精准捕捉用户兴趣演化规律，从而生成更具吸引力的个性化广告内容，提升广告点击率与用户体验。但在实际应用中，存在很多挑战，如模型参数规模大、注意力计算复杂、存在超长历史序列、量化影响推理精度等。

因此，本次赛事聚焦于如何提升生成式广告排序的推理性能，我们期待参赛选手能够从框架优化、算法创新、高性能计算等多个角度出发，提出突破现有技术瓶颈的创新方案。

## 详细说明

本次任务提供百度商业真实的用户脱敏行为数据、脱敏广告信息，选手需要在保证模型推理效果的前提下，极致优化推理性能。

## 数据集介绍

1. 用户行为数据：包括全局唯一的日志 ID 和用户 ID、广告曝光时间、广告点击时间等信息；
2. 广告内容：包括广告的文本描述、图片信息、广告主信息等；
3. 上下文信息：包括用户的地理位置、职业、性别、设备类型等；
4. 用户统计信息：包括用户的活跃度、兴趣标签、历史点击率等统计数据。

数据示例（按行字段，时间戳格式为 Unix Timestamp）：

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow" colspan="6">用户行为信息</th>
    <th class="tg-baqh">广告内容信息(N个字段)</th>
    <th class="tg-baqh">上下文(M个字段)</th>
    <th class="tg-baqh">用户统计信息(L个字段)</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">日志ID</td>
    <td class="tg-c3ow">用户ID</td>
    <td class="tg-c3ow">广告ID</td>
    <td class="tg-baqh">点击标签</td>
    <td class="tg-baqh">曝光时间戳</td>
    <td class="tg-baqh">用户点击时间戳</td>
    <td class="tg-baqh">文本描述, 图片ID, 广告主信息等</td>
    <td class="tg-baqh">地理位置, 职业, 性别, 设备类型等</td>
    <td class="tg-baqh">活跃度, 兴趣标签, 历史点击率等</td>
  </tr>
  <tr>
    <td class="tg-c3ow">412341</td>
    <td class="tg-c3ow">511</td>
    <td class="tg-c3ow">235</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">1758064791</td>
    <td class="tg-baqh">1758064799</td>
    <td class="tg-baqh">342, 54, 499</td>
    <td class="tg-baqh">6413, 6441, 57126</td>
    <td class="tg-baqh">8162, 505, 106</td>
  </tr>
  <tr>
    <td class="tg-c3ow">5541215</td>
    <td class="tg-c3ow">233</td>
    <td class="tg-c3ow">331</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">1757865600</td>
    <td class="tg-baqh">1757865610</td>
    <td class="tg-baqh">51, 8713, 6531</td>
    <td class="tg-baqh">82, 64100, 5331</td>
    <td class="tg-nrix">92, 55531, 25</td>
  </tr>
</tbody></table>

## 评估指标

1. 推理效率评估：参赛者提交 inference 脚本后，会通过统计 inference 脚本的运行时间，来计算在测试集上单条样本的平均推理时间。推理效率打分采用如下公示，如平均推理时间超过定义的时间限制，则本项和最终得分为 0；
$$
score_{latency} = \frac{latency_{base} - latency_{predict}}{latency_{base}}
$$

2. 策略效果评估：综合考虑 AUC 及 PCOC 指标，PCOC 需满足[0.85, 1.15]，AUC 需满足[0.65, 1] 方可进入榜单排序，否则本项和最终得分为 0，具体规则如下：得分由 pcoc 和 auc 组合而成：
$$
\frac{(auc_{predict} - 0.65) * 1000 + (0.15 - |proc - 1|) / 0.15 * 10}{360}
$$

- 指标说明：
  - AUC：ROC 曲线下的面积，越接近与 1 越好
  - PCOC：预估转化率 / 真实转化率，越接近于 1 越好

3. 计分规则：综合考虑推理性能和策略效果两个指标，计分规则如下所示；
$$
score_{all} = score_{latency} * 70 + score_{model} * 30
$$

## 警告 ⚠️

- 推理效率和策略效果任何一项得分为 0，整体得分为 0。
- 评估容器有整体运行时间限制（纯推理最长 5min），如果超出则无法计入成绩；（build_env.sh 等要在 20min 内）
- 任何作弊行为将会取消队伍成绩。

## 提交

参赛选手需要提交一个命名为【xxx】.zip 的压缩包，压缩包内需要包含以下内容:

1. 程序入口 infer.py 脚本，以及环境构建脚本 build_env.sh、requirements.txt。
2. 额外的 python 包环境，选手可以通过将 python 环境打包放在当前工作目录
3. 优化过的模型文件，如量化后的模型等

PS:

打包不要包含 xxx 文件夹 和 dataset 文件夹
权重若使用原版，无需修改权重参数且无需上传权重
若需要使用自定义权重，请自行完善和修改 infer.py 相关逻辑，系统测评后台默认调用赛事官方权重无需上传
若需要进行编译等其他复杂操作，请在 build_env.sh 中完成

## 参考资料

- 百度提出的推荐广告生成式排序模型： GRAB-百度推荐广告生成式排序模型技术实践 https://arxiv.org/pdf/2602.01865
- HSTU：Meta提出的用于长序列行为建模的高效模型 https://arxiv.org/abs/2402.17152
