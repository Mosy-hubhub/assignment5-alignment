"""CS336 Assignment 5, Section 3.4: 三种提示方式的 GSM8K 评估。

把本文件放在仓库的 cs336_alignment/evaluate_prompt.py 中。

先检查数据和模板，不需要 GPU 或第三方库：
    python -m cs336_alignment.evaluate_prompt --dry-run --limit 3

在 Linux GPU 环境中先试跑 10 题：
    uv run --extra gpu python -m cs336_alignment.evaluate_prompt --limit 10

完整评估：不传 --limit。可通过 --model 指定本地模型目录。
本脚本只评估，不更新模型参数。人工分析仍需自己完成。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 512
# 每条 prompt 返回N个独立采样结果
N = 1

PROMPT_FILES = {
    "question_only": "question_only.prompt",
    "r1_zero": "r1_zero.prompt",
    "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
}
CATEGORY_NAMES = {
    "category_1": "第1类:格式正确、答案正确",
    "category_2": "第2类:格式正确、答案错误",
    "category_3": "第3类:格式错误、答案错误",
    "other": "其他：格式错误但答案正确",
}


def parse_args() -> argparse.Namespace:
    """读取命令行配置；不传 --limit 时评估整个数据集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, help="绝对路径，或相对于仓库根目录的路径")
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPT_FILES), default=list(PROMPT_FILES))
    parser.add_argument("--output-dir", type=Path, help="结果的父目录；每次运行创建独立子目录")
    parser.add_argument("--dry-run", action="store_true", help="只读取数据、展示模板和生成参数")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size 和 --limit 必须是正整数。")
    if not 1 <= args.port <= 65535 or args.gpu < 0:
        parser.error("端口必须在 1~65535 之间,GPU 编号必须非负。")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization 必须在 0 和 1 之间。")
    if len(set(args.prompts)) != len(args.prompts):
        parser.error("--prompts 中不要重复填写同一种提示。")
    return args


def load_gsm8k(dataset_path: Path, limit: int | None = None) -> tuple[list[str], list[str]]:
    """逐行读取 JSONL;返回一一对应的题目和 #### 后面的最终答案。"""
    questions, ground_truths = [], []
    with dataset_path.open(encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            question, answer = example["question"], example["answer"]
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError(f"数据第 {line_number} 行的 question/answer 必须是字符串。")
            if "####" not in answer:
                raise ValueError(f"数据第 {line_number} 行没有标准答案分隔符 ####。")
            ground_truth = answer.rsplit("####", 1)[-1].strip()
            if not question.strip() or not ground_truth:
                raise ValueError(f"数据第 {line_number} 行的题目或最终答案为空。")
            questions.append(question)
            ground_truths.append(ground_truth)
            if limit is not None and len(questions) >= limit:
                break
    if not questions:
        raise ValueError(f"数据集没有可评估的题目：{dataset_path}")
    return questions, ground_truths


def load_prompt_templates(repo_root: Path, names: list[str]) -> dict[str, str]:
    """读取仓库已有的模板;three-shot 中的三个示例已经预先填好。"""
    templates = {}
    for name in names:
        path = repo_root / "cs336_alignment" / "prompts" / PROMPT_FILES[name]
        template = path.read_text(encoding="utf-8")
        if "{question}" not in template:
            raise ValueError(f"模板中缺少 {{question}} 占位符：{path}")
        templates[name] = template
    return templates


def build_prompts(template: str, questions: list[str]) -> list[str]:
    """用 .format() 填入待解题目；不把待解题的标准答案放进输入。"""
    return [template.format(question=question) for question in questions]


def make_sampling_params(prompt_name: str, seed: int) -> dict[str, Any]:
    """设置作业要求的采样参数；仅两种 r1_zero 提示使用答案结束标签。"""
    params = {"temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS, "n": N, "seed": seed}
    if prompt_name != "question_only":
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    return params


def load_reward_functions() -> dict[str, Callable]:
    """加载仓库提供的规则评分器；延迟导入，让 --dry-run 只依赖标准库。"""
    from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

    return {
        "question_only": question_only_reward_fn,
        "r1_zero": r1_zero_reward_fn,
        "r1_zero_three_shot": r1_zero_reward_fn,
    }


def start_vllm_server(args: argparse.Namespace) -> Any:
    """通过仓库的 VLLMServer 启动本机推理服务；不在这里训练模型。"""
    if sys.platform != "linux":
        raise RuntimeError("正式评估请在 Linux GPU 环境运行；本地可先使用 --dry-run。")
    from cs336_alignment.vllm_utils import VLLMServer

    server = VLLMServer(
        model_id=args.model,
        gpu=args.gpu,
        port=args.port,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        server.start()
    except BaseException:
        server.stop()
        raise
    return server


def generate_responses(
    server: Any,
    prompts: list[str],
    sampling_params: dict[str, Any],
    batch_size: int,
    raw_path: Path,
) -> list[dict[str, Any]]:
    """分批生成，每批立即保存原始输出，并打印进度。

    服务仍由仓库的 VLLMServer 管理。这里直接请求同一个 /v1/completions
    接口，是因为当前仓库的 generate_completions 没有透传 top_p；
    显式发送 top_p=1.0，避免依赖服务默认值。每道题只采样一个回答。
    """
    outputs = []
    with raw_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            payload = {
                "model": server.model_id,
                "prompt": batch,
                **sampling_params,
                "return_token_ids": True,
            }
            request = urllib.request.Request(
                f"{server.base_url}/v1/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=3600) as response:
                    body = json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                raise RuntimeError(f"vLLM 返回 HTTP {exc.code}：{detail}") from exc

            choices = sorted(body["choices"], key=lambda choice: choice["index"])
            if [choice["index"] for choice in choices] != list(range(len(batch))):
                raise ValueError("生成结果的数量或 index 不符合预期，无法与题目一一对应。")
            for local_index, choice in enumerate(choices):
                token_ids = choice.get("token_ids")
                output = {
                    "example_id": start + local_index,
                    "prompt": batch[local_index],
                    "response": choice["text"],
                    "finish_reason": choice.get("finish_reason"),
                    "num_generated_tokens": len(token_ids) if token_ids is not None else None,
                }
                outputs.append(output)
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
            f.flush()
            print(f"    已生成 {len(outputs)}/{len(prompts)} 个回答", flush=True)
    return outputs


def score_responses(
    prompt_name: str,
    questions: list[str],
    ground_truths: list[str],
    outputs: list[dict[str, Any]],
    reward_fn: Callable,
) -> list[dict[str, Any]]:
    """调用规则评分器，并按题目要求将每个回答分入三类；异常组合单独保留。"""
    if not len(questions) == len(ground_truths) == len(outputs):
        raise ValueError("题目、标准答案与模型输出数量不一致。")
    categories = {(1.0, 1.0): "category_1", (1.0, 0.0): "category_2", (0.0, 0.0): "category_3"}
    records = []
    for question, truth, output in zip(questions, ground_truths, outputs, strict=True):
        scores = reward_fn(response=output["response"], ground_truth=truth)
        scores = {key: float(scores[key]) for key in ("reward", "format_reward", "answer_reward")}
        if any(value not in (0.0, 1.0) for value in scores.values()):
            raise ValueError(f"评分器应返回二元奖励，实际为：{scores}")
        category = categories.get((scores["format_reward"], scores["answer_reward"]), "other")
        records.append({
            "prompt_name": prompt_name,
            "question": question,
            "ground_truth": truth,
            **output,
            **scores,
            "category": category,
        })
    return records


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    """统计准确率、格式通过率、三类数量，以及达到长度上限的比例。"""
    n = len(records)
    counts = Counter(record["category"] for record in records)
    return {
        "num_examples": n,
        "answer_accuracy": sum(record["answer_reward"] for record in records) / n,
        "format_accuracy": sum(record["format_reward"] for record in records) / n,
        "mean_reward": sum(record["reward"] for record in records) / n,
        "length_limit_rate": sum(record["finish_reason"] == "length" for record in records) / n,
        **{category: counts[category] for category in CATEGORY_NAMES},
    }


def make_review_samples(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """从第2、第3类各随机抽最多10条，保留人工判断字段；不自动判定解析错误。"""
    rng = random.Random(seed)
    review = []
    for category in ("category_2", "category_3"):
        candidates = [record for record in records if record["category"] == category]
        for record in rng.sample(candidates, min(10, len(candidates))):
            review.append({
                **record,
                "human_actually_correct": None,
                "human_parser_missed_correct_answer": None,
                "human_notes": "",
            })
    return review


def save_results(
    output_dir: Path,
    prompt_name: str,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    review_samples: list[dict[str, Any]],
    all_summaries: dict[str, dict[str, Any]],
) -> None:
    """保存逐题评分、待人工检查的样本，以及各提示的 JSON/CSV 汇总。"""
    for filename, rows in (
        (f"{prompt_name}.jsonl", records),
        (f"{prompt_name}_manual_review.jsonl", review_samples),
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    all_summaries[prompt_name] = summary
    (output_dir / "summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_name", *summary])
        writer.writeheader()
        for name, metrics in all_summaries.items():
            writer.writerow({"prompt_name": name, **metrics})


def main() -> None:
    """串起所有步骤；每步打印用途和实际结果，方便逐段理解程序。"""
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dataset_path = args.dataset or Path("data/gsm8k/test.jsonl")
    if not dataset_path.is_absolute():
        dataset_path = repo_root / dataset_path
    print("[1] parse_args：读取运行配置。", flush=True)
    print(f"    仓库：{repo_root}\n    模型：{args.model}\n    seed：{args.seed}", flush=True)

    questions, ground_truths = load_gsm8k(dataset_path, args.limit)
    print("\n[2] load_gsm8k：读取题目，提取 #### 后的标准答案。", flush=True)
    print(f"    数据：{dataset_path}\n    本次题数：{len(questions)}", flush=True)
    print(f"    第一题：{questions[0]}\n    标准答案：{ground_truths[0]}", flush=True)

    templates = load_prompt_templates(repo_root, args.prompts)
    print("\n[3] load_prompt_templates：读取仓库的提示模板。", flush=True)
    print(f"    已读取：{list(templates)}", flush=True)

    prepared = {}
    for name, template in templates.items():
        prompts = build_prompts(template, questions)
        params = make_sampling_params(name, args.seed)
        prepared[name] = (prompts, params)
        print(f"\n[4/{name}] build_prompts：填入题目，第一条完整输入如下：", flush=True)
        print(prompts[0], flush=True)
        print(f"    make_sampling_params：{params}", flush=True)

    if args.dry_run:
        print("\n[完成 dry-run] 数据和模板已检查；没有加载模型、生成回答或计算奖励。", flush=True)
        print("上面展示的标准答案只供评分使用，没有填进对应待解题的 prompt。", flush=True)
        return

    print("\n[5] load_reward_functions：导入仓库的规则评分器。", flush=True)
    reward_functions = load_reward_functions()
    for name in args.prompts:
        print(f"    {name} -> {reward_functions[name].__name__}", flush=True)

    output_parent = args.output_dir or Path("results/prompting_baselines")
    if not output_parent.is_absolute():
        output_parent = repo_root / output_parent
    output_dir = output_parent / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "model": args.model, "dataset": str(dataset_path), "num_examples": len(questions),
        "seed": args.seed, "gpu": args.gpu, "port": args.port,
        "batch_size": args.batch_size, "gpu_memory_utilization": args.gpu_memory_utilization,
        "templates": templates,
        "sampling_params": {name: params for name, (_, params) in prepared.items()},
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"    本次结果目录：{output_dir}", flush=True)

    print("\n[6] start_vllm_server：加载模型，启动本机 GPU 推理服务。", flush=True)
    server = start_vllm_server(args)
    print(f"    服务已就绪：{server.base_url}", flush=True)
    all_summaries = {}
    try:
        for name, (prompts, params) in prepared.items():
            start_time = time.monotonic()
            print(f"\n[7/{name}] generate_responses：分批生成，每批保存原始回答。", flush=True)
            outputs = generate_responses(
                server, prompts, params, args.batch_size, output_dir / f"{name}_raw.jsonl"
            )
            print(f"    第一条原始回答：\n{outputs[0]['response']}", flush=True)

            print(f"\n[8/{name}] score_responses：根据格式及标准答案评分。", flush=True)
            records = score_responses(name, questions, ground_truths, outputs, reward_functions[name])
            first = records[0]
            print(f"    第一条：format={first['format_reward']}, answer={first['answer_reward']}, "
                  f"reward={first['reward']}, {CATEGORY_NAMES[first['category']]}", flush=True)

            summary = summarize_results(records)
            summary["elapsed_seconds"] = round(time.monotonic() - start_time, 2)
            print(f"\n[9/{name}] summarize_results：汇总自动评分结果。", flush=True)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

            review = make_review_samples(records, args.seed)
            print(f"\n[10/{name}] make_review_samples：抽取第2、第3类供人工检查。", flush=True)
            for category in ("category_2", "category_3"):
                number = sum(row["category"] == category for row in review)
                print(f"    {CATEGORY_NAMES[category]}：抽到 {number} 条；该类共 {summary[category]} 条。",
                      flush=True)
            print("    若该类不足10条，将保存全部；少量题试跑不代替完整作业评估。", flush=True)

            save_results(output_dir, name, records, summary, review, all_summaries)
            print(f"\n[11/{name}] save_results：已保存评分、人工检查样本及汇总。", flush=True)
    finally:
        print("\n[12] server.stop：关闭本脚本启动的 vLLM 服务，释放资源。", flush=True)
        server.stop()

    print(f"\n评估完成，结果位于：{output_dir}", flush=True)
    print("查看 metrics.csv；逐题输出见各提示的 .jsonl。", flush=True)
    print("人工填写 *_manual_review.jsonl 的 human_* 字段，再据实际输出撰写3.4(a)(b)分析。",
          flush=True)


if __name__ == "__main__":
    main()
