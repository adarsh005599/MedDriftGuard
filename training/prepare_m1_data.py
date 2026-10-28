"""Generate and validate strictly synthetic M1 rewriter examples.

This module uses templates only. It never accepts raw patient data and does not generate
medical advice, diagnoses, medication content, or dosing instructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPTS = ("en", "devanagari", "roman_hindi")
FACT_TYPES = {"symptom", "duration"}
OUTPUT_FIELDS = {
    "standalone_query_en",
    "detected_script",
    "language",
    "topic_shift",
    "ledger_updates",
}
TOPIC_SYMPTOMS = ("cough", "fever", "headache")
VARIANTS = 128

DOSE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|micrograms?|milligrams?|iu|units?|tablets?|capsules?)\b",
    re.IGNORECASE,
)
DIAGNOSIS_PATTERN = re.compile(
    r"\b(?:diagnos(?:is|ed)|you have [a-z]+ disease|confirmed (?:with|diagnosis))\b",
    re.IGNORECASE,
)
PII_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\b(?:\+?\d[\d ()-]{8,}\d)\b")

LANGUAGE_DATA: dict[str, dict[str, Any]] = {
    "en": {
        "language": "en",
        "symptoms": {
            "cough": ["cough", "a cough", "cough", "a cough"],
            "fever": ["fever", "a fever", "fever", "a fever"],
            "headache": ["headache", "a headache", "headache", "a headache"],
        },
        "reports": [
            "I have {symptom}.",
            "I am experiencing {symptom}.",
            "I noticed {symptom}.",
            "I have been feeling {symptom}.",
        ],
        "followups": [
            "What general information is available about this?",
            "Can you give general information about it?",
            "What information is available about this concern?",
            "Tell me general information about it.",
        ],
        "topic_questions": [
            "What general information is available about {symptom_topic}?",
            "Can you provide general information about {symptom_topic}?",
            "I want general information about {symptom_topic}.",
            "What information can you share about {symptom_topic}?",
        ],
        "topic_shift": [
            "Forget that. What general information is available about {symptom_topic}?",
            "Change of topic: can you share general information about {symptom_topic}?",
            "Leave that topic. I want general information about {symptom_topic}.",
            "Different question: what general information is available about {symptom_topic}?",
        ],
        "correction_initial": [
            "I have fever and cough.",
            "I have cough and fever.",
            "I am experiencing fever and cough.",
            "I noticed cough and fever.",
        ],
        "correction_update": [
            "I no longer have fever, but I still have cough.",
            "Fever is gone; I still have cough.",
            "I do not have fever now. I still have cough.",
            "No fever now, only cough remains.",
        ],
    },
    "roman_hindi": {
        "language": "hi",
        "symptoms": {
            "cough": ["khansi", "khaansi", "khansi", "khaansi"],
            "fever": ["bukhar", "bukhaar", "bukhar", "bukhaar"],
            "headache": ["sar dard", "sir dard", "sar dard", "sir dard"],
        },
        "reports": [
            "Mujhe {symptom} hai.",
            "Mujhe {symptom} ho rahi hai.",
            "Maine {symptom} notice ki hai.",
            "Mujhe abhi {symptom} hai.",
        ],
        "followups": [
            "Iske baare mein general information kya hai?",
            "Is par general information de sakte hain?",
            "Mujhe iski general information chahiye.",
            "Is concern ke baare mein general information batayein.",
        ],
        "topic_questions": [
            "{symptom_topic} ke baare mein general information kya hai?",
            "Kya aap {symptom_topic} par general information de sakte hain?",
            "Mujhe {symptom_topic} ki general information chahiye.",
            "{symptom_topic} ke baare mein kya jaankari hai?",
        ],
        "topic_shift": [
            "Chalo ye chhodo. {symptom_topic} par general information kya hai?",
            "Topic badalte hain: {symptom_topic} ki general information batayein.",
            "Ye topic chhodo. Mujhe {symptom_topic} par general information chahiye.",
            "Dusra sawal: {symptom_topic} ke baare mein general information kya hai?",
        ],
        "correction_initial": [
            "Mujhe bukhar aur khansi hai.",
            "Mujhe khansi aur bukhar hai.",
            "Bukhar hai aur khansi bhi hai.",
            "Mujhe bukhaar aur khaansi hai.",
        ],
        "correction_update": [
            "Ab bukhar nahi hai, lekin khansi abhi bhi hai.",
            "Bukhar nahi raha; khansi abhi hai.",
            "Mujhe ab bukhar nahi hai. Khansi abhi bhi hai.",
            "Bukhar nahi, sirf khansi abhi bhi hai.",
        ],
    },
    "devanagari": {
        "language": "hi",
        "symptoms": {
            "cough": ["खांसी", "खाँसी", "खांसी", "खाँसी"],
            "fever": ["बुखार", "बुखार", "बुखार", "बुखार"],
            "headache": ["सिरदर्द", "सिरदर्द", "सिर दर्द", "सिरदर्द"],
        },
        "reports": [
            "मुझे {symptom} है।",
            "मुझे {symptom} की शिकायत है।",
            "मुझे अभी {symptom} है।",
            "मैंने {symptom} महसूस की है।",
        ],
        "followups": [
            "इसके बारे में सामान्य जानकारी क्या है?",
            "क्या इस पर सामान्य जानकारी दे सकते हैं?",
            "मुझे इसकी सामान्य जानकारी चाहिए।",
            "इस चिंता के बारे में सामान्य जानकारी बताएं।",
        ],
        "topic_questions": [
            "{symptom_topic} के बारे में सामान्य जानकारी क्या है?",
            "क्या आप {symptom_topic} पर सामान्य जानकारी दे सकते हैं?",
            "मुझे {symptom_topic} की सामान्य जानकारी चाहिए।",
            "{symptom_topic} के बारे में क्या जानकारी है?",
        ],
        "topic_shift": [
            "यह छोड़िए। {symptom_topic} पर सामान्य जानकारी क्या है?",
            "विषय बदलते हैं: {symptom_topic} की सामान्य जानकारी बताएं।",
            "उस विषय को छोड़िए। मुझे {symptom_topic} पर सामान्य जानकारी चाहिए।",
            "दूसरा सवाल: {symptom_topic} के बारे में सामान्य जानकारी क्या है?",
        ],
        "correction_initial": [
            "मुझे बुखार और खांसी है।",
            "मुझे खांसी और बुखार है।",
            "बुखार है और खांसी भी है।",
            "मुझे बुखार और खाँसी है।",
        ],
        "correction_update": [
            "अब बुखार नहीं है, लेकिन खांसी अभी भी है।",
            "बुखार नहीं रहा; खांसी अभी है।",
            "मुझे अब बुखार नहीं है। खांसी अभी भी है।",
            "बुखार नहीं, सिर्फ खांसी अभी भी है।",
        ],
    },
}


def _update(op: str, value: str, source_text: str) -> dict[str, str]:
    return {
        "op": op,
        "fact_type": "symptom",
        "value": value,
        "source_text": source_text,
    }


def _source_span(text: str, candidates: list[str]) -> str:
    """Return the exact surface spelling found in a synthetic user message."""
    for candidate in sorted(candidates, key=len, reverse=True):
        match = re.search(re.escape(candidate), text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    raise ValueError(f"No source span from {candidates!r} occurs in generated message {text!r}")


def _record(
    conversation_id: str,
    group_id: str,
    turn_id: int,
    language_key: str,
    ledger: str,
    last_turns: list[dict[str, str]],
    user_msg: str,
    query: str,
    topic_shift: bool,
    updates: list[dict[str, str]],
) -> dict[str, Any]:
    language_data = LANGUAGE_DATA[language_key]
    return {
        "conversation_id": conversation_id,
        "split_group_id": group_id,
        "turn_id": turn_id,
        "input": {"ledger": ledger, "last_turns": last_turns, "user_msg": user_msg},
        "output": {
            "standalone_query_en": query,
            "detected_script": "en" if language_key == "en" else language_key,
            "language": language_data["language"],
            "topic_shift": topic_shift,
            "ledger_updates": updates,
        },
    }


def generate_examples(variants: int = VARIANTS) -> list[dict[str, Any]]:
    """Generate deterministic synthetic conversations across all configured scripts."""
    if variants < 1:
        raise ValueError("variants must be at least one")
    records: list[dict[str, Any]] = []

    for language_key, data in LANGUAGE_DATA.items():
        for topic in TOPIC_SYMPTOMS:
            for variant in range(variants):
                index = variant % len(data["reports"])
                symptom = data["symptoms"][topic][index]
                conversation_id = f"m1_followup_{topic}_{language_key}_v{variant:03d}"
                report = data["reports"][index].format(symptom=symptom)
                source_phrase = _source_span(report, data["symptoms"][topic])
                ledger = f"Symptom: {topic}"
                records.append(
                    _record(
                        conversation_id,
                        f"followup_{topic}_v{variant:03d}",
                        1,
                        language_key,
                        "No user facts recorded yet.",
                        [],
                        report,
                        f"general information about {topic}",
                        False,
                        [_update("add", topic, source_phrase)],
                    )
                )
                followup = data["followups"][index]
                records.append(
                    _record(
                        conversation_id,
                        f"followup_{topic}_v{variant:03d}",
                        2,
                        language_key,
                        ledger,
                        [{"role": "user", "content": report}],
                        followup,
                        f"general information about {topic}",
                        False,
                        [],
                    )
                )

        for variant in range(variants):
            index = variant % len(data["correction_initial"])
            conversation_id = f"m1_correction_{language_key}_v{variant:03d}"
            initial = data["correction_initial"][index]
            correction = data["correction_update"][index]
            initial_fever = _source_span(initial, data["symptoms"]["fever"])
            initial_cough = _source_span(initial, data["symptoms"]["cough"])
            correction_fever = _source_span(correction, data["symptoms"]["fever"])
            correction_cough = _source_span(correction, data["symptoms"]["cough"])
            group_id = f"correction_fever_cough_v{variant:03d}"
            records.append(
                _record(
                    conversation_id,
                    group_id,
                    1,
                    language_key,
                    "No user facts recorded yet.",
                    [],
                    initial,
                    "general information about fever and cough",
                    False,
                    [
                        _update("add", "fever", initial_fever),
                        _update("add", "cough", initial_cough),
                    ],
                )
            )
            records.append(
                _record(
                    conversation_id,
                    group_id,
                    2,
                    language_key,
                    "Symptom: fever; Symptom: cough",
                    [{"role": "user", "content": initial}],
                    correction,
                    "general information about cough",
                    False,
                    [
                        _update("retract", "fever", correction_fever),
                        _update("add", "cough", correction_cough),
                    ],
                )
            )

        for variant in range(variants):
            index = variant % len(data["topic_shift"])
            conversation_id = f"m1_topic_shift_{language_key}_v{variant:03d}"
            prior = data["reports"][index].format(symptom=data["symptoms"]["headache"][index])
            message = data["topic_shift"][index].format(
                symptom_topic=(
                    "cough"
                    if language_key == "en"
                    else ("khansi" if language_key == "roman_hindi" else "खांसी")
                )
            )
            group_id = f"topic_shift_headache_to_cough_v{variant:03d}"
            records.append(
                _record(
                    conversation_id,
                    group_id,
                    1,
                    language_key,
                    "No user facts recorded yet.",
                    [],
                    prior,
                    "general information about headache",
                    False,
                    [
                        _update(
                            "add",
                            "headache",
                            _source_span(prior, data["symptoms"]["headache"]),
                        )
                    ],
                )
            )
            records.append(
                _record(
                    conversation_id,
                    group_id,
                    2,
                    language_key,
                    "Symptom: headache",
                    [{"role": "user", "content": prior}],
                    message,
                    "general information about cough",
                    True,
                    [],
                )
            )

        for topic in TOPIC_SYMPTOMS:
            for variant in range(variants):
                index = variant % len(data["topic_questions"])
                conversation_id = f"m1_direct_question_{topic}_{language_key}_v{variant:03d}"
                message = data["topic_questions"][index].format(
                    symptom_topic=(
                        topic if language_key == "en" else data["symptoms"][topic][index]
                    )
                )
                records.append(
                    _record(
                        conversation_id,
                        f"direct_{topic}_v{variant:03d}",
                        1,
                        language_key,
                        "No user facts recorded yet.",
                        [],
                        message,
                        f"general information about {topic}",
                        False,
                        [],
                    )
                )

    validate_records(records)
    return records


def validate_record(record: Any) -> None:
    """Raise ValueError unless a record follows the safe M1 JSONL contract."""
    if not isinstance(record, dict) or set(record) != {
        "conversation_id",
        "split_group_id",
        "turn_id",
        "input",
        "output",
    }:
        raise ValueError("record must contain exactly the required top-level fields")
    if not isinstance(record["conversation_id"], str) or not record["conversation_id"]:
        raise ValueError("conversation_id must be non-empty")
    if not isinstance(record["split_group_id"], str) or not record["split_group_id"]:
        raise ValueError("split_group_id must be non-empty")
    if not isinstance(record["turn_id"], int) or record["turn_id"] < 1:
        raise ValueError("turn_id must be a positive integer")

    input_data = record["input"]
    if not isinstance(input_data, dict) or set(input_data) != {"ledger", "last_turns", "user_msg"}:
        raise ValueError("input must contain exactly ledger, last_turns, and user_msg")
    if not isinstance(input_data["ledger"], str) or not isinstance(input_data["user_msg"], str):
        raise ValueError("ledger and user_msg must be strings")
    if not input_data["user_msg"].strip():
        raise ValueError("user_msg must not be empty")
    if not isinstance(input_data["last_turns"], list) or len(input_data["last_turns"]) > 2:
        raise ValueError("last_turns must be a list of at most two messages")
    for turn in input_data["last_turns"]:
        if not isinstance(turn, dict) or set(turn) != {"role", "content"}:
            raise ValueError("each last_turn must contain role and content")
        if turn["role"] not in {"user", "assistant"} or not isinstance(turn["content"], str):
            raise ValueError("last_turn role/content is invalid")

    output = record["output"]
    if not isinstance(output, dict) or set(output) != OUTPUT_FIELDS:
        raise ValueError("output does not match the strict rewriter schema")
    if (
        not isinstance(output["standalone_query_en"], str)
        or not output["standalone_query_en"].strip()
    ):
        raise ValueError("standalone_query_en must be non-empty")
    if output["detected_script"] not in SCRIPTS:
        raise ValueError("detected_script must be en, devanagari, or roman_hindi")
    if output["language"] not in {"en", "hi"}:
        raise ValueError("language must be en or hi")
    if output["language"] != ("en" if output["detected_script"] == "en" else "hi"):
        raise ValueError("language does not correspond to detected_script")
    if not isinstance(output["topic_shift"], bool):
        raise ValueError("topic_shift must be boolean")
    if not isinstance(output["ledger_updates"], list):
        raise ValueError("ledger_updates must be a list")

    # Reject unsafe user input before checking cross-field consistency. This
    # makes the safety failure deterministic even when a deliberately corrupted
    # record also contains stale ``source_text`` from an earlier message.
    user_message = input_data["user_msg"]
    if DOSE_PATTERN.search(user_message):
        raise ValueError("training record contains a prohibited medication dose/unit")
    if DIAGNOSIS_PATTERN.search(user_message):
        raise ValueError("training record contains diagnosis language")
    if PII_PATTERN.search(user_message):
        raise ValueError("training record may contain an email or phone number")

    message = user_message
    for update in output["ledger_updates"]:
        if not isinstance(update, dict) or set(update) != {
            "op",
            "fact_type",
            "value",
            "source_text",
        }:
            raise ValueError("ledger update does not match strict schema")
        if update["op"] not in {"add", "retract", "correct"}:
            raise ValueError("ledger update operation is invalid")
        if update["fact_type"] not in FACT_TYPES:
            raise ValueError("unsupported ledger fact type")
        if not all(
            isinstance(update[field], str) and update[field].strip()
            for field in ("value", "source_text")
        ):
            raise ValueError("ledger update value/source_text must be non-empty strings")
        if update["source_text"].casefold() not in message.casefold():
            raise ValueError("source_text must be copied from the current user message")

    serialized = json.dumps(record, ensure_ascii=False)
    if DOSE_PATTERN.search(serialized):
        raise ValueError("training record contains a prohibited medication dose/unit")
    if DIAGNOSIS_PATTERN.search(serialized):
        raise ValueError("training record contains diagnosis language")
    if PII_PATTERN.search(serialized):
        raise ValueError("training record may contain an email or phone number")


def validate_records(records: list[Any]) -> None:
    for record in records:
        validate_record(record)


def _split_name(group_id: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    value = int(hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()[:16], 16)
    fraction = value / 0xFFFFFFFFFFFFFFFF
    if fraction < test_ratio:
        return "test"
    if fraction < test_ratio + val_ratio:
        return "val"
    return "train"


def split_by_conversation(
    records: list[dict[str, Any]], seed: int = 42, val_ratio: float = 0.15, test_ratio: float = 0.15
) -> dict[str, list[dict[str, Any]]]:
    """Assign complete conversations and all script variants to one deterministic split."""
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than one")
    validate_records(records)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_id = record["split_group_id"]
        groups[group_id].append(record)

    group_splits = {
        group_id: _split_name(group_id, seed, val_ratio, test_ratio) for group_id in groups
    }
    split_names = {split for split in group_splits.values()}
    # Ensure tiny generated corpora still have all three output partitions.
    for required in ("test", "val", "train"):
        if required not in split_names:
            candidates = sorted(
                groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
            )
            donor = next(
                (
                    name
                    for name in ("train", "val", "test")
                    if sum(s == name for s in group_splits.values()) > 1
                ),
                None,
            )
            if donor is None or not candidates:
                raise ValueError("need at least three split groups to create train/val/test")
            group_id = next(group for group in candidates if group_splits[group] == donor)
            group_splits[group_id] = required
            split_names = set(group_splits.values())

    result: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for group_id, group_records in groups.items():
        result[group_splits[group_id]].extend(group_records)
    for records_in_split in result.values():
        records_in_split.sort(key=lambda row: (row["conversation_id"], row["turn_id"]))
    return result


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            validate_record(record)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_dataset(
    output_dir: Path,
    variants: int = VARIANTS,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, int]:
    records = generate_examples(variants=variants)
    splits = split_by_conversation(records, seed, val_ratio, test_ratio)
    for split_name, split_records in splits.items():
        write_jsonl(output_dir / f"m1_{split_name}.jsonl", split_records)
    manifest = {
        "seed": seed,
        "variants_per_scenario": variants,
        "conversation_count": len({record["conversation_id"] for record in records}),
        "split_group_count": len({record["split_group_id"] for record in records}),
        "record_counts": {name: len(split_records) for name, split_records in splits.items()},
        "languages": sorted({record["output"]["detected_script"] for record in records}),
        "synthetic_only": True,
        "medical_advice_or_doses": False,
    }
    (output_dir / "m1_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def validate_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--variants", type=int, default=VARIANTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--validate", nargs="+", type=Path, help="Validate one or more JSONL files")
    args = parser.parse_args()

    if args.validate:
        for input_path in args.validate:
            print(f"{input_path}: {validate_jsonl(input_path)} valid records")
        return

    manifest = prepare_dataset(
        args.output_dir,
        variants=args.variants,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
