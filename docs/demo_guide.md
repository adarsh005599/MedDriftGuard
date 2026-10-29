# Local demo guide

From the repository root on Windows:

```powershell
python -m pip install -e ".[dev,retrieval,ui]"
python -m pytest tests -q
python debug_p3.py
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
python -m streamlit run src\ui\streamlit_app.py
```

Suggested checks:

- Supported: `What home care is mentioned for a cough?`
- Roman Hinglish: `Mujhe khansi hai, ghar par kya kar sakta hoon?`
- Unsupported: `How does a warp-drive engine work?`
- Meaningless input: `ttt`

The UI should show real local citations for supported KB questions and a visible safe abstention for unsupported or meaningless input. The default site is DEMO/MOCK MODE for generation and claim scoring; retrieval is local.

Optional PDF/FAISS support is not needed for the default demo:

```powershell
python -m pip install -e ".[pdf]"
```

Only PDFs with reviewed provenance should be converted into trusted chunks. Do not put private patient documents or secrets in the repository.
