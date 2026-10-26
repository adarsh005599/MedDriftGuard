"""
Patient-State Ledger for MedDriftGuard.

Maintains a versioned, explicit record of facts stated by the user.
Decoupled completely from the medical knowledge base index.
Supports fact addition, retraction, correction, and superseding chains.
"""

from __future__ import annotations

from src.common.schemas import FactType, LedgerFact, LedgerUpdate


class PatientLedger:
    """
    In-memory versioned store of patient-reported facts.
    Strictly records user-stated information, never storing inferred diagnoses.
    """

    def __init__(self):
        self._facts: list[LedgerFact] = []
        self._fact_counter: int = 0

    @property
    def all_facts(self) -> list[LedgerFact]:
        """Return all facts including historical retracted and corrected facts."""
        return list(self._facts)

    def get_active_facts(self) -> list[LedgerFact]:
        """Return only currently active, unretracted user facts."""
        return [f for f in self._facts if f.status == "active"]

    def get_retracted_facts(self) -> list[LedgerFact]:
        """Return all facts that have been retracted or superseded."""
        return [f for f in self._facts if f.status in ("retracted", "corrected")]

    def get_active_symptoms(self) -> list[str]:
        """Convenience method to retrieve values of all active symptoms."""
        return [f.value for f in self._facts if f.status == "active" and f.fact_type == "symptom"]

    def find_fact(self, fact_type: FactType, value: str) -> LedgerFact | None:
        """Find the latest active fact matching fact_type and value."""
        val_clean = value.strip().lower()
        for f in reversed(self._facts):
            if (
                f.status == "active"
                and f.fact_type == fact_type
                and f.value.strip().lower() == val_clean
            ):
                return f
        return None

    def apply_update(self, update: LedgerUpdate, turn_id: int) -> LedgerFact:
        """
        Apply a single LedgerUpdate (add, retract, or correct).
        """
        self._fact_counter += 1
        new_fact_id = f"fact_{self._fact_counter:03d}"

        if update.op == "add":
            # Check if this exact fact is already active to avoid duplicates
            existing = self.find_fact(update.fact_type, update.value)
            if existing is not None:
                return existing

            new_fact = LedgerFact(
                fact_id=new_fact_id,
                fact_type=update.fact_type,
                value=update.value.strip(),
                turn_id=turn_id,
                status="active",
                source_text=update.source_text,
            )
            self._facts.append(new_fact)
            return new_fact

        elif update.op == "retract":
            # Find the existing active fact to retract
            target_fact = None
            if update.supersedes:
                for f in self._facts:
                    if f.fact_id == update.supersedes:
                        target_fact = f
                        break

            if target_fact is None:
                target_fact = self.find_fact(update.fact_type, update.value)

            if target_fact is None and update.fact_type == "symptom":
                value = update.value.strip().lower()
                target_fact = next(
                    (
                        fact
                        for fact in reversed(self._facts)
                        if fact.status == "active"
                        and fact.fact_type == "symptom"
                        and (
                            fact.value.strip().lower().endswith(f" {value}")
                            or value.endswith(f" {fact.value.strip().lower()}")
                        )
                    ),
                    None,
                )

            if target_fact is not None:
                target_fact.status = "retracted"
                target_fact_id = target_fact.fact_id
            else:
                target_fact_id = None

            # Record the retraction entry
            retraction_fact = LedgerFact(
                fact_id=new_fact_id,
                fact_type=update.fact_type,
                value=update.value.strip(),
                turn_id=turn_id,
                status="retracted",
                supersedes=target_fact_id,
                source_text=update.source_text,
            )
            self._facts.append(retraction_fact)
            return retraction_fact

        elif update.op == "correct":
            # Retract previous fact and link new active fact
            target_fact = None
            if update.supersedes:
                for f in self._facts:
                    if f.fact_id == update.supersedes:
                        target_fact = f
                        break

            if target_fact is None:
                # Find any active fact of same type to supersede
                for f in reversed(self._facts):
                    if f.status == "active" and f.fact_type == update.fact_type:
                        target_fact = f
                        break

            if target_fact is not None:
                target_fact.status = "corrected"
                target_fact_id = target_fact.fact_id
            else:
                target_fact_id = None

            corrected_fact = LedgerFact(
                fact_id=new_fact_id,
                fact_type=update.fact_type,
                value=update.value.strip(),
                turn_id=turn_id,
                status="active",
                supersedes=target_fact_id,
                source_text=update.source_text,
            )
            self._facts.append(corrected_fact)
            return corrected_fact

        raise ValueError(f"Unknown ledger update op: {update.op}")

    def apply_updates(self, updates: list[LedgerUpdate], turn_id: int) -> list[LedgerFact]:
        """Apply a batch of updates for a single conversational turn."""
        applied = []
        for upd in updates:
            applied.append(self.apply_update(upd, turn_id=turn_id))
        return applied

    def to_summary_string(self) -> str:
        """
        Produce a compact, readable summary of active user-stated facts.
        Used as grounding context for generators without exposing raw chat history.
        """
        active = self.get_active_facts()
        if not active:
            return "No user facts recorded yet."

        grouped: dict[str, list[str]] = {}
        for f in active:
            grouped.setdefault(f.fact_type, []).append(f.value)

        parts = []
        for f_type, vals in grouped.items():
            formatted_type = f_type.replace("_", " ").title()
            parts.append(f"{formatted_type}: {', '.join(vals)}")

        return "; ".join(parts)

    def reset(self) -> None:
        """Clear all facts (session reset)."""
        self._facts.clear()
        self._fact_counter = 0
