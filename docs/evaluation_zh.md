[English](evaluation.md) | [中文](evaluation_zh.md)

# 评测协议

以下命令均在代码仓库根目录执行。

## 输入格式

两个输入文件均为 UTF-8 编码的 JSONL，每行一个JSON对象。
各自的 `example_id` 必须唯一，两份文件的ID集合必须完全一致。

参考标注示例：

```json
{"example_id":"demo::C1::voice1","language":"en","condition":"C1","reference_text":"Please check the flour.","branch_word":"flour","parent_id":"demo","case_id":"demo_a","speaker":"voice1"}
```

模型转写示例：

```json
{"example_id":"demo::C1::voice1","hypothesis":"Please check the flour."}
```

- 语言：`zh`（中文）、`en`（英文）。
- 条件：`C0`、`C1`、`U`、`C3`、`USER_ONLY`、`FULL_HISTORY`。
- C3样本必须提供 `explicit_cue_depth`，取值为 `0`、`2` 或 `4`。
- 空转写是合法输入，保留并正常计分。

## 评测指标

- **CER/WER**：所有样本的编辑距离之和，除以规范化后参考文本的总计分单位数。
  中文按汉字和拉丁词计分，英文按词计分。
- **目标词召回率（Target Recall）**：模型转写中包含指定目标词的样本比例。
  匹配要求目标词的规范化token序列连续出现。
- **C3深度分组**：分别统计D0、D2和D4的上述指标。

Target Recall是存在性指标：转写同时包含目标与竞争词时，目标仍计命中，额外输出由CER/WER体现。它不是互斥候选选择准确率。

论文中的C3整体结果读取 `groups.zh_C3` 和 `groups.en_C3`，不要平均 `c3_depth` 中的百分比。深度标签来自不同case，分组差异不是同一样本的位置消融。

请使用与模型转写完全对应的数据版本。修订C3只改变上下文输入，不改变评分规则；ID相同不代表不同版本的转写可以混用。

参考文本（REF）、模型转写（HYP）和目标词标签独立执行同一套对应语言的规范化规则：

- 中文：原文含ASCII数字时先转为中文读法，再由本地实现进行字符与拉丁词切分。
- 英文：先统一字母缩写形式，再使用从 Transformers 4.57.6 导入的 Whisper EnglishTextNormalizer。

评测协议标识为 `hearincontext`。
输出JSON中的 `error_rate` 和 `target_recall` 为比例值，乘以100后为百分比。

拼写规范化会合并 `busing` 和 `bussing`，因此这组词规范化后的命中不能作为词面消歧证据。
字母缩写处理仍存在已知的词边界歧义；不会根据分数高低删除样本。

## 运行测试

```bash
python -m unittest discover -s tests
```

## 配对置信区间

统计入口与常规评分共用根目录依赖：

```bash
python evaluation/bootstrap.py \
  --manifest test.jsonl --base base.jsonl --tuned tuned.jsonl \
  --language zh --condition C1 \
  --replicates 10000 --seed 20260908 \
  --output outputs/paired_ci.json
```

两份转写必须覆盖同一manifest。按 `parent_id` 配对、有放回抽样，组内语义分支及音色一起保留；每次从总编辑数、参考单位数和目标命中数计算指标，不平均各组百分比。输出微调减基座的百分点差值和95% percentile置信区间：错误率差值负数更好，Recall差值正数更好。

这衡量给定checkpoint的样本不确定性，不代表重复训练种子的波动。

## 真实语音热词验证

论文使用 SeACo 发布的 AISHELL-1-NE 808条测试清单及400词表。NE与ContextASR等多实体输入共用 `--mode entities`，复用主基准的规范化和编辑距离；只改变标注单位，不另起规范化规则。输入、计数约定和NE准备命令见 [多实体评测](entities_zh.md)。这里不声称复刻SeACo官方R/P/F，也不直接比较其论文分数。
