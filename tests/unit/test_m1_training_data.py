import copy
import json

import pytest

from training.prepare_m1_data import (
    generate_examples,
    prepare_dataset,
    split_by_conversation,
    validate_record,
    validate_jsonl,
    validate_records,
    write_jsonl,
)


def test_synthetic_generation_is_deterministic_and_covers_all_scripts():
    first = generate_examples(variants=2)
    second = generate_examples(variants=2)

    assert first == second
    assert {record["output"]["detected_script"] for record in first} == {
        "en",
        "devanagari",
        "roman_hindi",
    }
    assert all(record["conversation_id"].startswith("m1_") for record in first)


def test_generated_examples_follow_exact_schema_and_quote_user_facts():
    records = generate_examples(variants=2)

    validate_records(records)
    for record in records:
        message = record["input"]["user_msg"].casefold()
        for update in record["output"]["ledger_updates"]:
            assert update["source_text"].casefold() in message


@pytest.mark.parametrize(
    "bad_text",
    [
        "Take 5 mg daily.",
        "The diagnosis is asthma.",
        "Contact patient@example.com.",
        "Call +1 (555) 234-5678.",
    ],
)
def test_validator_rejects_prohibited_or_personal_content(bad_text):
    record = copy.deepcopy(generate_examples(variants=1)[0])
    record["input"]["user_msg"] = bad_text

    with pytest.raises(ValueError):
        validate_record(record)


def test_validator_rejects_unquoted_or_unsupported_patient_facts():
    record = copy.deepcopy(generate_examples(variants=1)[0])
    record["output"]["ledger_updates"][0]["source_text"] = "a different phrase"

    with pytest.raises(ValueError, match="copied from the current user message"):
        validate_record(record)

    record = copy.deepcopy(generate_examples(variants=1)[0])
    record["output"]["ledger_updates"][0]["fact_type"] = "diagnosis"
    with pytest.raises(ValueError, match="unsupported ledger fact type"):
        validate_record(record)


def test_conversation_groups_and_script_variants_never_leak_across_splits():
    records = generate_examples(variants=4)
    splits = split_by_conversation(records, seed=7)

    split_groups = {
        split_name: {record["split_group_id"] for record in split_records}
        for split_name, split_records in splits.items()
    }
    assert split_groups["train"].isdisjoint(split_groups["val"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["val"].isdisjoint(split_groups["test"])
    assert all(splits[name] for name in ("train", "val", "test"))

    group_scripts = {}
    for split_records in splits.values():
        for record in split_records:
            group_scripts.setdefault(record["split_group_id"], set()).add(
                record["output"]["detected_script"]
            )
    assert all(scripts == {"en", "devanagari", "roman_hindi"} for scripts in group_scripts.values())


def test_split_is_deterministic_and_all_turns_of_conversation_stay_together():
    records = generate_examples(variants=2)
    split_a = split_by_conversation(records, seed=19)
    split_b = split_by_conversation(records, seed=19)

    assert split_a == split_b
    conversation_splits = {}
    for split_name, rows in split_a.items():
        for row in rows:
            previous = conversation_splits.setdefault(row["conversation_id"], split_name)
            assert previous == split_name


def test_jsonl_writer_and_validator_round_trip(tmp_path):
    records = generate_examples(variants=1)
    jsonl_path = tmp_path / "m1_valid.jsonl"

    write_jsonl(jsonl_path, records)

    assert validate_jsonl(jsonl_path) == len(records)


def test_jsonl_validator_reports_malformed_and_unsafe_lines(tmp_path):
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"broken":\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"malformed.jsonl:1"):
        validate_jsonl(malformed)

    unsafe_record = copy.deepcopy(generate_examples(variants=1)[0])
    unsafe_record["input"]["user_msg"] = "I take 5 mg tablets."
    unsafe = tmp_path / "unsafe.jsonl"
    unsafe.write_text(json.dumps(unsafe_record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prohibited medication dose"):
        validate_jsonl(unsafe)


def test_prepare_dataset_writes_all_split_files_and_manifest(tmp_path):
    manifest = prepare_dataset(tmp_path / "synthetic", variants=2, seed=13)

    assert manifest["synthetic_only"] is True
    assert manifest["medical_advice_or_doses"] is False
    for split_name in ("train", "val", "test"):
        path = tmp_path / "synthetic" / f"m1_{split_name}.jsonl"
        assert path.exists()
        assert validate_jsonl(path) > 0
    assert (tmp_path / "synthetic" / "m1_manifest.json").exists()