"""Debug script for P3 acceptance test failures - iteration 3."""
import json
import logging
logging.disable(logging.CRITICAL)

from pathlib import Path
from src.ledger.patient_ledger import PatientLedger
from src.rewriter.standalone_rewriter import StandaloneRewriter

fixtures_path = Path("tests/fixtures/conversations_p3.json")
data = json.loads(fixtures_path.read_text(encoding="utf-8"))
conversations = data["conversations"]
rewriter = StandaloneRewriter()
failures = []

for conv in conversations:
    conv_id = conv["conv_id"]
    ledger = PatientLedger()
    history = []
    for turn in conv["turns"]:
        turn_id = turn["turn_id"]
        user_msg = turn["user_message"]
        res = rewriter.rewrite(
            user_message=user_msg,
            ledger=ledger,
            last_turns=history[-2:] if history else [],
            turn_id=turn_id,
        )
        ledger.apply_updates(res.ledger_updates, turn_id=turn_id)
        if turn["expected_topic_shift"] and not res.topic_shift:
            failures.append(f"[{conv_id} T{turn_id}] Expected topic_shift=True for: {user_msg!r}")
        for r in turn["expected_retractions"]:
            if r.lower() in [s.lower() for s in ledger.get_active_symptoms()]:
                failures.append(
                    f"[{conv_id} T{turn_id}] Retracted {r!r} still active! Active={ledger.get_active_symptoms()}"
                )
        q = res.standalone_query_en.lower()
        for mc in turn["query_must_contain"]:
            if mc.lower() not in q:
                failures.append(
                    f"[{conv_id} T{turn_id}] Query {res.standalone_query_en!r} MISSING {mc!r}"
                )
        for mn in turn["query_must_not_contain"]:
            if mn.lower() in q:
                failures.append(
                    f"[{conv_id} T{turn_id}] Query {res.standalone_query_en!r} has FORBIDDEN {mn!r}"
                )
        history.append({"role": "user", "content": user_msg})

print(f"Total failures: {len(failures)}")
for f in failures:
    print(f)
