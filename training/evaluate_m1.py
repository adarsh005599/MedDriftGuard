"""Evaluate the M1 adapter against the existing deterministic rewriter baseline.

This evaluation measures rewriter formatting and extraction behavior only. It does not
measure medical correctness or clinical safety.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.ledger.patient_ledger import PatientLedger
from src.rewriter.prompt_templates import REWRITER_SYSTEM_PROMPT
from src.rewriter.standalone_rewriter import StandaloneRewriter
from training.prepare_m1_data import validate_jsonl


def load_records(path: Path) -> list[dict[str, Any]]:
    validate_jsonl(path)
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def comparable_updates(updates: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (item["op"], item["fact_type"], item["value"].casefold())
        for item in updates
    }


class PromptedModelPredictor:
    def __init__(self, base_model: str, adapter_path: Path | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if adapter_path is not None:
            from peft import PeftModel

        self.torch = torch
        tokenizer_source = adapter_path if adapter_path is not None else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model
        self.model.eval()

    def predict(self, record: dict[str, Any]) -> dict[str, Any] | None:
        messages = [
            {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(record["input"], ensure_ascii=False),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(result, dict):
            return None
        try:
            validate_prediction(record, result)
        except (KeyError, TypeError, ValueError):
            return None
        return result


def validate_prediction(record: dict[str, Any], prediction: dict[str, Any]) -> None:
    """Check model output fields and ensure extracted facts came from this user turn."""
    required = {
        "standalone_query_en",
        "detected_script",
        "language",
        "topic_shift",
        "ledger_updates",
    }
    if set(prediction) != required:
        raise ValueError("prediction fields do not match the M1 output schema")
    output = prediction.copy()
    normalized_updates = []
    for update in output["ledger_updates"]:
        normalized = {
            key: update.get(key)
            for key in ("op", "fact_type", "value", "source_text")
        }
        normalized_updates.append(normalized)
    output["ledger_updates"] = normalized_updates
    validate_record({**record, "output": output})


def rule_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rewriter = StandaloneRewriter()
    ledgers: dict[str, PatientLedger] = defaultdict(PatientLedger)
    predictions = []
    for record in records:
        ledger = ledgers[record["conversation_id"]]
        result = rewriter.rewrite(
            user_message=record["input"]["user_msg"],
            ledger=ledger,
            last_turns=record["input"]["last_turns"],
            turn_id=record["turn_id"],
        )
        ledger.apply_updates(result.ledger_updates, turn_id=record["turn_id"])
        predictions.append(result.to_dict())
    return predictions


def summarize(records: list[dict[str, Any]], predictions: list[dict[str, Any] | None]) -> dict[str, Any]:
    total = len(records)
    valid = [prediction for prediction in predictions if prediction is not None]
    topic_matches = 0
    query_exact = 0
    update_precision_values = []
    update_recall_values = []
    by_script: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "valid": 0, "topic_correct": 0})

    for record, prediction in zip(records, predictions, strict=True):
        expected = record["output"]
        script = expected["detected_script"]
        bucket = by_script[script]
        bucket["total"] += 1
        if prediction is None:
            update_precision_values.append(0.0)
            update_recall_values.append(0.0)
            continue
        bucket["valid"] += 1
        if prediction.get("topic_shift") == expected["topic_shift"]:
            topic_matches += 1
            bucket["topic_correct"] += 1
        if prediction.get("standalone_query_en", "").strip().casefold() == expected[
            "standalone_query_en"
        ].strip().casefold():
            query_exact += 1

        expected_updates = comparable_updates(expected["ledger_updates"])
        predicted_updates = comparable_updates(prediction.get("ledger_updates", []))
        true_positive = len(expected_updates & predicted_updates)
        update_precision_values.append(
            true_positive / len(predicted_updates) if predicted_updates else float(not expected_updates)
        )
        update_recall_values.append(
            true_positive / len(expected_updates) if expected_updates else float(not predicted_updates)
        )

    return {
        "examples": total,
        "json_schema_valid_rate": round(len(valid) / total, 4) if total else 0.0,
        "topic_shift_accuracy": round(topic_matches / total, 4) if total else 0.0,
        "standalone_query_exact_match": round(query_exact / total, 4) if total else 0.0,
        "ledger_update_precision": round(sum(update_precision_values) / total, 4) if total else 0.0,
        "ledger_update_recall": round(sum(update_recall_values) / total, 4) if total else 0.0,
        "by_script": {
            script: {
                **counts,
                "valid_rate": round(counts["valid"] / counts["total"], 4),
                "topic_shift_accuracy": round(counts["topic_correct"] / counts["total"], 4),
            }
            for script, counts in sorted(by_script.items())
        },
    }


def evaluate(records: list[dict[str, Any]], predictor: PromptedModelPredictor) -> dict[str, Any]:
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["conversation_id"]].append((index, record))

    predictions: list[dict[str, Any] | None] = [None] * len(records)
    for rows in grouped.values():
        for index, record in sorted(rows, key=lambda row: row[1]["turn_id"]):
            predictions[index] = predictor.predict(record)
    return summarize(records, predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, default=Path("data/synthetic/m1_test.jsonl"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/train_m1/evaluation.json"))
    args = parser.parse_args()

    records = load_records(args.test_file)
    baseline = summarize(records, rule_predictions(records))
    prompted_baseline = evaluate(records, PromptedModelPredictor(args.base_model))
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    adapter = evaluate(records, PromptedModelPredictor(args.base_model, args.adapter_path))
    report = {
        "evaluation_scope": "synthetic rewriter behavior only; not medical correctness",
        "test_file": str(args.test_file),
        "base_model": args.base_model,
        "adapter_path": str(args.adapter_path),
        "rule_baseline": baseline,
        "prompted_base_model": prompted_baseline,
        "fine_tuned_adapter": adapter,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
