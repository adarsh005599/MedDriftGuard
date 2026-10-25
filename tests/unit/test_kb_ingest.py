"""
Unit tests for knowledge base ingestion pipeline (Phase 1).
Validates HTML cleaning, zero-dosage safety filtering, section chunking,
deterministic chunk IDs, and license verification.
"""

from pathlib import Path
import json
import pytest

from kb.cleaner import DocumentCleaner
from kb.safety_filter import contains_drug_dosage, strip_drug_dosages
from kb.chunker import TextChunker
from kb.ingest import parse_sources_md, run_ingestion
from src.common.schemas import RetrievedChunk


def test_cleaner_strips_unwanted_tags_and_ads():
    raw_html = """
    <html>
      <head><script>var tracker = 1;</script></head>
      <body>
        <nav><a href="/home">Home</a></nav>
        <div class="ad-banner">Buy vitamins now!</div>
        <main>
          <h1>Asthma Management</h1>
          <p>Asthma is a chronic lung disorder that causes airway inflammation.</p>
          <h2>When to See a Doctor</h2>
          <p>Seek immediate medical care if wheezing worsens suddenly.</p>
        </main>
        <footer><p>Copyright 2026</p></footer>
      </body>
    </html>
    """
    sections = DocumentCleaner.clean_html(raw_html)
    assert len(sections) == 2

    headings = [s.heading for s in sections]
    assert "Asthma Management" in headings
    assert "When to See a Doctor" in headings

    combined_text = " ".join(s.text for s in sections)
    assert "Buy vitamins now" not in combined_text
    assert "var tracker" not in combined_text
    assert "Copyright 2026" not in combined_text
    assert "airway inflammation" in combined_text


def test_safety_filter_detects_and_strips_dosages():
    # Test dosage detection
    assert contains_drug_dosage("Take 500 mg tablets every 4 hours.") is True
    assert contains_drug_dosage("Give 5 ml syrup every 8 hours for pain.") is True
    assert contains_drug_dosage("Drink plenty of water and rest in bed.") is False

    # Test stripping dosages
    unsafe_text = "For adult fever, take 500mg tablets every 6 hours. Drink 2 liters of water daily."
    cleaned, redactions = strip_drug_dosages(unsafe_text)

    assert redactions > 0
    assert "500mg" not in cleaned
    assert "Consult a doctor or pharmacist" in cleaned
    assert "Drink 2 liters of water daily" in cleaned


def test_chunker_deterministic_ids_and_schema_validation():
    chunker1 = TextChunker(target_tokens=50, min_tokens=10, overlap_pct=0.15)
    chunker2 = TextChunker(target_tokens=50, min_tokens=10, overlap_pct=0.15)

    sample_text = (
        "Hydration is essential during acute diarrhoeal illness to prevent electrolyte loss. "
        "Oral rehydration salts (ORS) should be mixed in safe clean drinking water and consumed "
        "frequently in small sips. If commercial packets are unavailable, clean boiled water with "
        "sugar and salt can provide temporary hydration while seeking medical guidance."
    )

    chunks1 = chunker1.chunk_section(
        source="MoHFW",
        url="https://mohfw.gov.in/dehydration",
        date_accessed="2026-10-25",
        topic="dehydration",
        section_name="Hydration Guidelines",
        section_text=sample_text,
        license_str="Government Open Data License - India (GODL)",
        language="en",
    )

    chunks2 = chunker2.chunk_section(
        source="MoHFW",
        url="https://mohfw.gov.in/dehydration",
        date_accessed="2026-10-25",
        topic="dehydration",
        section_name="Hydration Guidelines",
        section_text=sample_text,
        license_str="Government Open Data License - India (GODL)",
        language="en",
    )

    assert len(chunks1) > 0
    assert len(chunks1) == len(chunks2)
    # Confirm deterministic chunk IDs
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.chunk_id.startswith("moh_dehydration_")
        assert isinstance(c1, RetrievedChunk)


def test_chunker_deduplication():
    chunker = TextChunker(target_tokens=50, min_tokens=10)
    sample_text = "Fever is an elevation in body temperature above normal physiological range."

    chunks_first = chunker.chunk_section(
        source="MedlinePlus",
        url="https://medlineplus.gov/fever.html",
        date_accessed="2026-10-25",
        topic="fever",
        section_name="Overview",
        section_text=sample_text,
        license_str="Public Domain",
    )
    assert len(chunks_first) == 1

    # Second pass with exact same content should be deduplicated
    chunks_second = chunker.chunk_section(
        source="MedlinePlus",
        url="https://medlineplus.gov/fever.html",
        date_accessed="2026-10-25",
        topic="fever",
        section_name="Overview",
        section_text=sample_text,
        license_str="Public Domain",
    )
    assert len(chunks_second) == 0


def test_parse_sources_md_skips_unlicensed_sources(tmp_path):
    sources_content = """# Sources
| Source ID | Source Name | Base URL | License / Terms | Medical Scope | Ingestion Status |
|---|---|---|---|---|---|
| `medlineplus` | MedlinePlus | https://medlineplus.gov | Public Domain | Health Info | Approved |
| `unlicensed_blog` | Health Blog | https://blog.example.com | None | Blog | Unapproved |
| `missing_license` | Mystery Web | https://mystery.example.com | TBD | Tips | Unapproved |
"""
    tmp_file = tmp_path / "SOURCES.md"
    tmp_file.write_text(sources_content, encoding="utf-8")

    approved = parse_sources_md(tmp_file)
    assert "medlineplus" in approved
    assert "unlicensed_blog" not in approved
    assert "missing_license" not in approved


def test_generated_chunks_jsonl_validity():
    chunks_path = Path("kb/chunks.jsonl")
    assert chunks_path.exists(), "kb/chunks.jsonl must exist after ingestion"

    lines = chunks_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0, "kb/chunks.jsonl must not be empty"

    for line in lines:
        data = json.loads(line)
        chunk = RetrievedChunk(**data)
        assert chunk.chunk_id
        assert chunk.license
        assert chunk.text
        # Ensure zero prohibited dosages
        assert not contains_drug_dosage(chunk.text)
