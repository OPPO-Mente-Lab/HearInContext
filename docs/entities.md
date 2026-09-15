# 共用多实体评分

`evaluation/evaluate.py --mode entities` 接受NE、ContextASR等数据转换后的统一JSONL。`--mode benchmark`仍为默认，主基准输出不变。两种模式共用normalization.py和RapidFuzz编辑距离，无新增依赖。

## 准备AISHELL-1-NE

从 [SeACo仓库](https://github.com/R1ckShi/SeACo-Paraformer/tree/855b0fb7cadd57f111e192dcd08682ebf391693b/data/test) 获取 `text`（该版本test/uttid也含参考转写）及 `hotword.txt`，由用户单独取得AISHELL音频。仓库不附带这些第三方数据。

```bash
python evaluation/prepare_ne.py --reference-text text --hotwords hotword.txt \
  --condition hotwords --output ne.jsonl
python evaluation/evaluate.py --mode entities --manifest ne.jsonl \
  --hypotheses predictions.jsonl --output scores.json --observations rows.jsonl
```

官方版本有808条、400词；准备器不写死数量，保留词表原顺序。模型输入应给每条音频相同的完整词表，不按参考答案筛词。评分不调用模型。

转写格式：`{"example_id":"BAC009S0764W0179","hypothesis":"模型转写"}`。ID必须与manifest精确匹配。旧解码若带`aishell1::`前缀，应在数据适配时明确映射，评分器不猜测ID。

## 通用输入

```json
{"example_id":"demo","language":"zh","condition":"coarse","reference_text":"甲方联系乙方，再联系甲方","entity_list":["甲方","乙方"]}
```

ContextASR字段可映射为：`uniq_id`→`example_id`、`text`→`reference_text`、`entity_list`保留，语言转为`zh/en`，条件明确填写；多个条件同时评测时ID需包含条件。预测使用同样ID和`hypothesis`。

## 计数规则

- REF、HYP及所有实体独立应用同一语言规范化；按完整连续token序列匹配。
- 每个实体分别计算参考出现次数R和转写出现次数H；命中为min(R,H)，汇总后除以参考总出现次数。主基准每例单目标且出现一次时，就退化为每例命中统计。
- 不同的嵌套标签分别计数，例如“许玮甯”和“玮甯”；重复标注也保留其次数，不暗中去重。空规范化标签不匹配。
- `extra_entity_occurrences`为sum(max(H-R,0))，只是词面多出次数，不是对齐FP或官方Precision/F1。
- 参考没有实体时命中为0；整组实体分母为0时Recall输出null。空参考保留，非空转写计插入；整组参考单位数为0时错误率为null。
- 所有指标按组池化计数，不平均逐句百分比；输出`error_rate`和`entity_recall`为0—1比例。

NE四配置、两种条件已按此规则完成计数回归核验。参考实体为空时命中为0，分母为0时Recall为null。

本入口使用我们的统一口径，不等同于SeACo官方评分或外部基准的对齐式热词F1。样本级说话人字段可随输入保留供进一步审计；现有bootstrap入口仅适用于主基准parent_id分组，不直接用于NE说话人聚类。
