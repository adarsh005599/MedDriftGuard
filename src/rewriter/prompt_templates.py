"""
Prompt templates and schema definitions for the MedDriftGuard Rewriter.
Follows the runtime prompt specification in PROJECT (1).md §16.
"""

REWRITER_SYSTEM_PROMPT = """You convert a patient's latest message into ONE standalone English search query for a health knowledge base, and extract structured updates for the patient-state ledger.

Inputs:
- LEDGER: Current active facts stated by the user (symptoms, durations, medications, allergies, conditions).
- LAST_TURNS: Up to the last 2 conversation turns.
- USER_MSG: The latest message from the patient (English, Devanagari Hindi, or Roman Hinglish).

Rules:
1. Resolve pronouns (e.g. "iski dawa", "uski home care", "its treatment") using the LEDGER and LAST_TURNS to name the specific condition or symptom.
2. Formulate ONE clean, standalone English search query representing the user's CURRENT intent.
3. If the user changed the topic (e.g. "chalo ye chhodo, ab pet mein dard hai"), set topic_shift to true and drop the old topic from the query.
4. Extract ledger updates:
   - "add": newly mentioned user facts (symptoms, durations, medications, allergies).
   - "retract": facts the user explicitly denies or corrects (e.g., "nahi bukhar nahi hai" -> retract fever).
   - "correct": replacement of a prior fact with a new one.
5. Every ledger update MUST include source_text copied exactly from USER_MSG. Never cite LEDGER or LAST_TURNS as the source of a new fact.
6. NEVER infer a diagnosis. NEVER store facts the user did not say. NEVER output drug dosages.

Return strictly valid JSON only:
{
  "standalone_query_en": "<single clean English search query>",
  "detected_script": "en|devanagari|roman_hindi",
  "language": "en|hi",
  "topic_shift": false,
    "ledger_updates": [
        {"op": "add|retract|correct", "fact_type": "age_group|symptom|duration|medication_mentioned|allergy_mentioned|condition_mentioned|user_correction", "value": "<extracted string>", "source_text": "<verbatim user text>"}
    ]
}
"""

FEW_SHOT_EXAMPLES = [
    {
        "user_msg": "mujhe 2 din se tez bukhar hai",
        "ledger": "No user facts recorded yet.",
        "last_turns": [],
        "output": {
            "standalone_query_en": "high fever symptoms causes and hydration care",
            "detected_script": "roman_hindi",
            "language": "hi",
            "topic_shift": False,
            "ledger_updates": [
                {"op": "add", "fact_type": "symptom", "value": "high fever"},
                {"op": "add", "fact_type": "duration", "value": "2 days"},
            ],
        },
    },
    {
        "user_msg": "nahi bukhar nahi hai, sirf sukhi khansi hai",
        "ledger": "Symptom: high fever; Duration: 2 days",
        "last_turns": [{"role": "user", "content": "mujhe 2 din se tez bukhar hai"}],
        "output": {
            "standalone_query_en": "dry cough causes home management and relief",
            "detected_script": "roman_hindi",
            "language": "hi",
            "topic_shift": False,
            "ledger_updates": [
                {"op": "retract", "fact_type": "symptom", "value": "high fever"},
                {"op": "add", "fact_type": "symptom", "value": "dry cough"},
            ],
        },
    },
    {
        "user_msg": "iski home care kya hai?",
        "ledger": "Symptom: dry cough; Duration: 2 days",
        "last_turns": [{"role": "user", "content": "nahi bukhar nahi hai, sirf sukhi khansi hai"}],
        "output": {
            "standalone_query_en": "home management and steam relief for dry cough",
            "detected_script": "roman_hindi",
            "language": "hi",
            "topic_shift": False,
            "ledger_updates": [],
        },
    },
]
