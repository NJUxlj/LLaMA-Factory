# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
这个Python脚本用于评估模型生成的中文文本的质量。

它读取一个JSON文件，该文件包含模型预测的文本（"predict"）和对应的参考文本（"label"）。
然后，它使用jieba进行中文分词，并计算BLEU和ROUGE指标来衡量预测文本与参考文本的相似度和准确性。
最后，它计算所有样本的平均分数，并将结果保存到名为 "predictions_score.json" 的文件中。
'''

import json
import logging
import time

import fire        # 用于创建命令行接口。
from datasets import load_dataset


try:
    import jieba  # type: ignore
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
    from rouge_chinese import Rouge  # type: ignore

    jieba.setLogLevel(logging.CRITICAL)
    jieba.initialize()
except ImportError:
    print("Please install llamafactory with `pip install -e .[metrics]`.")
    raise


def compute_metrics(sample):
    hypothesis = list(jieba.cut(sample["predict"]))     # 使用jieba.cut对预测文本和参考文本进行分词
    reference = list(jieba.cut(sample["label"]))

    bleu_score = sentence_bleu(
        [list(sample["label"])],
        list(sample["predict"]),
        smoothing_function=SmoothingFunction().method3,   # 用了平滑函数来处理n-gram不存在的情况。
    )

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]

    metric_result = {}
    for k, v in result.items():
        metric_result[k] = round(v["f"] * 100, 4)

    metric_result["bleu-4"] = round(bleu_score * 100, 4)

    return metric_result


def main(filename: str):
    start_time = time.time()
    dataset = load_dataset("json", data_files=filename, split="train")
    # 用dataset.map将compute_metrics函数应用于数据集中的每个样本，并行处理（num_proc=8）。remove_columns参数用于删除原始的 "predict" 和 "label" 列。
    dataset = dataset.map(compute_metrics, num_proc=8, remove_columns=dataset.column_names)
    score_dict = dataset.to_dict()

    average_score = {}
    for task, scores in sorted(score_dict.items(), key=lambda x: x[0]):
        print(f"{task}: {sum(scores) / len(scores):.4f}")
        average_score[task] = sum(scores) / len(scores)

    with open("predictions_score.json", "w", encoding="utf-8") as f:
        json.dump(average_score, f, indent=4)

    print(f"\nDone in {time.time() - start_time:.3f}s.\nScore file saved to predictions_score.json")


if __name__ == "__main__":
    fire.Fire(main)  # 启动命令行界面
