"""DocuTrust AI - PDF processing and study workspace (Streamlit).

Run:  streamlit run app.py
Garvit's retrieval system plugs in at the section marked "RETRIEVAL HOOK".
"""
import hashlib
import html
import io
import json
import os
import random
import re
import shutil
from collections import Counter
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

try:
    import pymupdf  # PyMuPDF
except ImportError:
    pymupdf = None

OCR_OK, OCR_ERR = False, ""
try:
    import pytesseract

    _win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if not shutil.which("tesseract") and os.path.exists(_win):
        pytesseract.pytesseract.tesseract_cmd = _win
    pytesseract.get_tesseract_version()
    OCR_OK = True
except Exception as _e:  # missing package or Tesseract binary
    OCR_ERR = str(_e)[:160]


def make_icon():
    im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((8, 8, 120, 120), radius=28, fill=(14, 18, 32, 255))
    d.rounded_rectangle((34, 24, 94, 104), radius=8, outline=(61, 217, 193, 255), width=7)
    d.line((50, 66, 60, 77, 80, 52), fill=(61, 217, 193, 255), width=8, joint="curve")
    return im


st.set_page_config(page_title="DocuTrust AI", page_icon=make_icon(), layout="wide",
                   initial_sidebar_state="expanded")
S = st.session_state

# ============================================================ RETRIEVAL HOOK
# Lazy per-session DocuTrust retrieval pipeline. The index is rebuilt only
# when the uploaded document set changes.
RETRIEVAL_CONNECTED = True


def _get_qa_pipeline():
    from src.backend.qa_pipeline import DocumentQAPipeline

    chunks = []
    signature = []
    for doc in S.docs.values():
        signature.append((doc.get("id"), doc.get("sha256"), len(doc.get("chunks", []))))
        for chunk in doc.get("chunks", []):
            if not str(chunk.get("text", "")).strip():
                continue
            chunks.append({
                "chunk_id": f"{doc['id']}-{chunk.get('id', 'chunk')}",
                "page_number": int(chunk.get("page", 1)),
                "text": chunk["text"],
                "doc": doc["name"],
            })

    if not chunks:
        raise RuntimeError("Upload and process a PDF before using the AI assistant.")

    signature = tuple(sorted(signature))
    pipeline = S.get("_docutrust_qa_pipeline")
    if pipeline is None:
        pipeline = DocumentQAPipeline()
        S["_docutrust_qa_pipeline"] = pipeline
        S["_docutrust_qa_signature"] = None

    if S.get("_docutrust_qa_signature") != signature:
        pipeline.index_document(chunks)
        S["_docutrust_qa_signature"] = signature

    return pipeline


def _evidence_context(evidence):
    return "\n\n".join(
        f"[Document: {item.get('doc', 'Uploaded PDF')} | Page {item['page_number']}]\n{item['text']}"
        for item in evidence
    )


def _parse_json_result(raw):
    raw = str(raw).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", raw, flags=re.S)
        if not match:
            raise ValueError("AI did not return valid JSON. Please generate again.")
        return json.loads(match.group(0))


def ai_generate(task, passages, options):
    """Generate document-grounded answers and study material with DocuTrust AI."""
    qa = _get_qa_pipeline()
    question = str(options.get("question", "")).strip()

    if task == "answer":
        if not question:
            raise ValueError("Enter a question about your uploaded documents.")
        result = qa.ask(question)
        doc_names = {}
        for doc_id, doc in S.docs.items():
            for chunk in doc.get("chunks", []):
                doc_names[f"{doc_id}-{chunk.get('id', 'chunk')}"] = doc["name"]
        source_lines = [
            f"- {doc_names.get(item['chunk_id'], 'Uploaded PDF')} · Page {item['page']}"
            for item in result["sources"]
        ]
        sources = "\n".join(dict.fromkeys(source_lines))
        verification = result.get("verification", {})
        footer = (
            f"\n\n---\n**Verification:** {result['status']} "
            f"· Citation page check: {'passed' if verification.get('citations_valid') else 'not passed'}"
        )
        if sources:
            footer += f"\n\n**Retrieved sources**\n{sources}"
        return result["answer"] + footer

    prompts = {
        "notes": (
            "Create clear, well-structured study notes from the evidence. Include a short summary, "
            "key points, important definitions and exam-focused takeaways where supported. "
            "Do not add facts outside the evidence. Add [Page N] citations to factual bullets. "
            "Exam-oriented mode: " + str(bool(options.get("exam", False)))
        ),
        "quiz": (
            "Create a multiple-choice quiz strictly from the evidence. Return ONLY valid JSON "
            "with a top-level key qs containing objects with keys q (question), opts (exactly "
            "four answer strings), ans (zero-based integer index of the correct option), and "
            "exp (brief explanation with [Page N]). "
            f"Create {int(options.get('n', 5))} questions at {options.get('level', 'Easy')} difficulty. "
            "Each ans must be the zero-based index of the correct option. No outside facts."
        ),
        "cards": (
            "Create flashcards strictly from the evidence. Return ONLY valid JSON with a top-level "
            "key cards containing objects with keys front (question or term), back (answer with "
            "[Page N]), and src (source page). "
            f"Create up to {int(options.get('n', 10))} useful cards. No outside facts."
        ),
    }
    if task not in prompts:
        raise ValueError(f"Unsupported AI task: {task}")

    query = question or prompts[task]
    evidence = qa.retrieve(query)
    if not evidence:
        raise RuntimeError("No relevant evidence was retrieved from the uploaded PDFs.")

    context = _evidence_context(evidence)
    raw = qa._ollama_chat([
        {
            "role": "system",
            "content": (
                "You are DocuTrust AI. Use ONLY the supplied PDF evidence. "
                "Never invent facts, standards, page numbers, or examples. "
                "If the evidence does not support an item, omit it. "
                "For factual statements, cite the supporting page as [Page N]."
            ),
        },
        {"role": "user", "content": f"TASK:\n{prompts[task]}\n\nPDF EVIDENCE:\n{context}"},
    ], num_predict=900 if task == "notes" else 1200)

    if task == "notes":
        return raw

    return _parse_json_result(raw)


# ============================================================ THEMES / TEXT
THEMES = {
    "Midnight Dark": dict(bg="#0a0f24", bg2="#1a1f52", surface="#131a35", surface2="#1d2850", text="#eef1ff", muted="#a6b2d8", accent="#3de0c8", accent2="#7c8cff", warn="#f0c36a", on="#03201a", border="#2d3b6e", side="#0c1230", dark=1),
    "Warm Dark": dict(bg="#1a1310", bg2="#3a2316", surface="#2a201a", surface2="#3b2c22", text="#f8eee3", muted="#c8b49f", accent="#f2b24f", accent2="#e58f6b", warn="#f0c36a", on="#2a1b04", border="#56412f", side="#150f0c", dark=1),
    "Dim Study": dict(bg="#1f2430", bg2="#2f3860", surface="#2a3140", surface2="#364058", text="#e7ebf3", muted="#a9b3c6", accent="#8db4ff", accent2="#b79bff", warn="#f0c36a", on="#0b1a38", border="#46526d", side="#1a1f2a", dark=1),
    "Sepia Reading": dict(bg="#f1e4cc", bg2="#fbf1da", surface="#fcf6e8", surface2="#eddcbc", text="#33281b", muted="#6a5841", accent="#7f4a86", accent2="#b8863b", warn="#8a5a0b", on="#ffffff", border="#d3be96", side="#ebd9b6", dark=0),
    "Clean Light": dict(bg="#f2f6fc", bg2="#dfe9fb", surface="#ffffff", surface2="#eaf0f9", text="#14202f", muted="#4f617a", accent="#0e7c86", accent2="#4f6bed", warn="#8a5a0b", on="#ffffff", border="#d0dcee", side="#ffffff", dark=0),
    "Forest Focus": dict(bg="#0c1914", bg2="#17382a", surface="#14261d", surface2="#1e3b2c", text="#e7f4ec", muted="#9cbdaa", accent="#6fd4a0", accent2="#5fb3c9", warn="#f0c36a", on="#052016", border="#2e5040", side="#0a1510", dark=1),
}
FONTS = {"Source Serif 4": "'Source Serif 4',Georgia,serif", "Figtree": "'Figtree',system-ui,sans-serif",
         "Atkinson Hyperlegible": "'Atkinson Hyperlegible',Verdana,sans-serif",
         "Noto Sans Devanagari": "'Noto Sans Devanagari','Figtree',sans-serif",
         "JetBrains Mono": "'JetBrains Mono',Consolas,monospace"}
PAGES = ["dashboard", "workspace", "library", "reader", "search", "ask", "study", "listen", "settings"]
ICONS = dict(dashboard="dashboard", workspace="upload_file", library="library_books", reader="menu_book",
             search="search", ask="forum", study="school", listen="headphones", settings="tune")

EN = dict(
    dashboard="Dashboard", workspace="PDF Workspace", library="Library", reader="Reader", search="Smart Search",
    ask="Ask AI", study="Study Tools", listen="Listen", settings="Settings",
    docs="Documents", pages="Pages processed", chunks="Chunks", ocr_pages="OCR pages", recent="Recent documents",
    status="Processing status", no_docs="No documents yet. Upload PDFs in PDF Workspace to get started.",
    upload="Upload PDFs", process="Process PDFs", theme="Theme", language="Language", font_size="Font size (px)",
    font_family="Font family", line_spacing="Line spacing", read_width="Reading width (px)",
    comfy="Comfortable reading mode", focus="Focus mode", exit_focus="Exit focus mode", fullscreen="Fullscreen",
    search_ph="Search keywords or a phrase", prev="Previous", next="Next", ask_ph="Ask about your documents",
    pending="AI answers pending: Garvit's retrieval system is not connected yet.",
    notes="Notes", quiz="Quiz", cards="Flashcards", generate="Generate", play="Play", pause="Pause",
    resume="Resume", stop="Stop", speed="Speed", known="Known", revise="Need revision", download="Download JSON",
    remove="Remove", open_reader="Open in Reader", shortcuts="Keyboard shortcuts", pomodoro="Pomodoro timer",
    score="Score", all="All", document="Document", page="Page", reading="Reading", appearance="Appearance",
    clear="Clear chat", passages="Relevant passages (keyword match)", ready="Ready", partial="Partly extracted",
    needs_ocr="Needs OCR", start="Start", reset="Reset", select_docs="Select documents", pages_range="Page range",
    new_quiz="New quiz", shuffle="Shuffle", rule_note="Built by rule-based extraction from your PDF text. Not AI-written.",
    ai_btn="Generate with AI (pending)", filter="Filter", sort="Sort", name="Name", ocr_settings="Extraction settings",
)
HI = dict(
    dashboard="डैशबोर्ड", workspace="पीडीएफ वर्कस्पेस", library="लाइब्रेरी", reader="रीडर", search="स्मार्ट खोज",
    ask="एआई से पूछें", study="अध्ययन टूल्स", listen="सुनें", settings="सेटिंग्स", docs="दस्तावेज़",
    pages="प्रोसेस हुए पेज", chunks="चंक्स", ocr_pages="ओसीआर पेज", recent="हाल के दस्तावेज़", status="प्रोसेसिंग स्थिति",
    no_docs="अभी कोई दस्तावेज़ नहीं है। शुरू करने के लिए पीडीएफ वर्कस्पेस में PDF अपलोड करें।",
    upload="PDF अपलोड करें", process="PDF प्रोसेस करें", theme="थीम", language="भाषा", font_size="फ़ॉन्ट आकार (px)",
    font_family="फ़ॉन्ट", line_spacing="पंक्ति अंतर", read_width="पढ़ने की चौड़ाई (px)", comfy="आरामदायक पढ़ने का मोड",
    focus="फ़ोकस मोड", exit_focus="फ़ोकस मोड बंद करें", fullscreen="फ़ुलस्क्रीन", search_ph="कीवर्ड या वाक्यांश खोजें",
    prev="पिछला", next="अगला", ask_ph="अपने दस्तावेज़ों के बारे में पूछें",
    pending="एआई उत्तर लंबित: Garvit का रिट्रीवल सिस्टम अभी जुड़ा नहीं है।", notes="नोट्स", quiz="क्विज़",
    cards="फ्लैशकार्ड", generate="बनाएँ", play="चलाएँ", pause="रोकें", resume="फिर चालू करें", stop="बंद करें",
    speed="गति", known="याद है", revise="दोहराना है", download="JSON डाउनलोड", remove="हटाएँ",
    open_reader="रीडर में खोलें", shortcuts="कीबोर्ड शॉर्टकट", pomodoro="पोमोडोरो टाइमर", score="स्कोर", all="सभी",
    document="दस्तावेज़", page="पेज", reading="पढ़ना", appearance="रूप-रंग", clear="चैट साफ़ करें",
    passages="संबंधित अंश (कीवर्ड मिलान)", ready="तैयार", partial="आंशिक रूप से निकाला गया", needs_ocr="ओसीआर ज़रूरी",
    start="शुरू", reset="रीसेट", select_docs="दस्तावेज़ चुनें", pages_range="पेज सीमा", new_quiz="नया क्विज़",
    shuffle="मिलाएँ", rule_note="यह आपके PDF टेक्स्ट से नियम-आधारित तरीके से बना है। एआई द्वारा लिखा नहीं गया।",
    ai_btn="एआई से बनाएँ (लंबित)", filter="फ़िल्टर", sort="क्रम", name="नाम", ocr_settings="एक्सट्रैक्शन सेटिंग्स",
)


EN.update(gallery="Theme gallery", use="Use", active="Active", greet_m="Good morning", greet_a="Good afternoon",
          greet_e="Good evening", hero_sub="{n} documents ready · {p} pages searchable")
HI.update(gallery="थीम गैलरी", use="उपयोग करें", active="चालू", greet_m="सुप्रभात", greet_a="नमस्कार",
          greet_e="शुभ संध्या", hero_sub="{n} दस्तावेज़ तैयार · {p} पेज खोजने योग्य")


def t(k):
    return (HI if S.get("lang") == "हिन्दी" else EN).get(k) or EN.get(k, k)


DEFAULTS = dict(theme="Midnight Dark", lang="English", font_size=17, font_family="Source Serif 4", line_height=1.7,
                read_width=720, comfy=False, focus=False, docs={}, nav="dashboard", chat=[], quiz=None, fc=None,
                fc_idx=0, fc_status={}, notes_out=None, ocr_on=True, ocr_lang="eng", dpi=200, chunk_size=900,
                overlap=120, rd_doc=None, rd_page=1, last_run=[])
for _k, _v in DEFAULTS.items():
    S.setdefault(_k, _v)


def pw(fn, label, key, default, options=None, **kw):
    """Persistent widget: its value survives navigating to other pages."""
    S.setdefault(key, default)
    if options is not None:
        if fn is st.multiselect:
            kw["default"] = [o for o in S[key] if o in options]
        elif fn is st.select_slider:  # select_slider uses value=, not index=
            if S[key] not in options:
                S[key] = options[0]
            kw["value"] = S[key]
        else:
            if S[key] not in options:
                S[key] = options[0]
            kw["index"] = options.index(S[key])
        args = (label, options)
    else:
        kw["value"] = S[key]
        args = (label,)
    return fn(*args, key="w_" + key, on_change=lambda: S.__setitem__(key, S["w_" + key]), **kw)


# ============================================================ STYLE
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700&family=Figtree:wght@400;500;600&family=Source+Serif+4:wght@400;600&family=Noto+Sans+Devanagari:wght@400;600&family=JetBrains+Mono&family=Atkinson+Hyperlegible:wght@400;700&display=swap');
html,body{background:var(--bg)!important}
.stApp{background:radial-gradient(1200px 620px at 88% -8%,color-mix(in srgb,var(--bg2) 85%,transparent),transparent 62%),radial-gradient(900px 500px at -5% 105%,color-mix(in srgb,var(--accent2) 10%,transparent),transparent 60%),var(--bg)!important;font-family:'Figtree','Noto Sans Devanagari',system-ui,sans-serif}
.stApp,.stApp p,.stApp li,.stApp label,[data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"] p{color:var(--text)}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:'Bricolage Grotesque','Noto Sans Devanagari',sans-serif;color:var(--text);letter-spacing:-.01em}
[data-testid="stCaptionContainer"],.stApp small{color:var(--muted)!important}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:linear-gradient(180deg,var(--side),color-mix(in srgb,var(--side) 82%,var(--accent2)))!important;border-right:1px solid var(--border);box-shadow:6px 0 24px -14px rgba(0,0,0,.45)}
.block-container{padding-top:2rem;padding-bottom:4rem;max-width:1180px}
div[data-baseweb="input"]>div,div[data-baseweb="select"]>div,div[data-baseweb="textarea"],textarea,input{background:var(--surface2)!important;color:var(--text)!important;border-color:var(--border)!important;transition:border-color .15s,box-shadow .15s}
div[data-baseweb="input"]>div:hover,div[data-baseweb="select"]>div:hover,div[data-baseweb="textarea"]:hover{border-color:color-mix(in srgb,var(--accent) 70%,var(--border))!important}
div[data-baseweb="input"]>div:focus-within,div[data-baseweb="select"]>div:focus-within,div[data-baseweb="textarea"]:focus-within{border-color:var(--accent)!important;box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 22%,transparent)}
div[data-baseweb="select"] *,div[data-baseweb="select"] svg{color:var(--text)!important;fill:var(--text)!important}
span[data-baseweb="tag"]{background:color-mix(in srgb,var(--accent) 24%,var(--surface2))!important;color:var(--text)!important;border:1px solid color-mix(in srgb,var(--accent) 45%,var(--border))}
span[data-baseweb="tag"] *{color:var(--text)!important}
div[data-baseweb="popover"]>div{border:1px solid var(--border);border-radius:12px;box-shadow:0 16px 40px -12px rgba(0,0,0,.55)}
div[data-baseweb="popover"] li:hover,div[data-baseweb="popover"] li[aria-selected="true"]{background:color-mix(in srgb,var(--accent) 20%,var(--surface2))!important}
[data-baseweb="slider"] [role="slider"]{background:var(--accent)!important;border:2px solid var(--surface)!important;box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 30%,transparent)!important}
[data-testid="stSliderThumbValue"],[data-testid="stSliderTickBarMin"],[data-testid="stSliderTickBarMax"]{color:var(--accent)!important}
[data-baseweb="checkbox"] span,[role="switch"]+div,[data-testid="stToggle"] label>div:first-child{border-color:var(--accent)}
hr{border-color:var(--border)!important}
div[data-baseweb="popover"] ul,div[data-baseweb="menu"]{background:var(--surface2)!important;color:var(--text)!important}
div[data-baseweb="popover"] li{color:var(--text)!important}
.stButton>button,.stDownloadButton>button,[data-testid^="stBaseButton"]{background:var(--surface2);color:var(--text);border:1px solid var(--border);border-radius:10px;transition:border-color .15s,transform .05s}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--accent);color:var(--text);background:color-mix(in srgb,var(--accent) 12%,var(--surface2));box-shadow:0 6px 18px -10px var(--accent)}
.stButton>button:active{transform:translateY(1px)}
 [data-testid="stPopover"] > button{background:var(--surface2)!important;color:var(--text)!important;border:1px solid var(--accent)!important;border-radius:10px!important;box-shadow:0 4px 14px -8px var(--accent)!important} [data-testid="stPopover"] > button:hover{background:color-mix(in srgb,var(--accent) 16%,var(--surface2))!important;border-color:var(--accent)!important} [data-testid="stPopover"] > button p,[data-testid="stPopover"] > button span,[data-testid="stPopover"] > button svg{color:var(--text)!important;fill:var(--text)!important}\n.stButton>button[kind="primary"],[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,var(--accent),color-mix(in srgb,var(--accent) 62%,var(--accent2)))!important;color:var(--on)!important;border-color:var(--accent)!important;font-weight:600;box-shadow:0 8px 20px -12px var(--accent)}
.stButton>button[kind="primary"]:hover,[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.07);box-shadow:0 10px 24px -10px var(--accent)}
.stButton>button[kind="primary"] p,[data-testid="stBaseButton-primary"] p{color:var(--on)!important}
button:focus-visible,input:focus-visible,textarea:focus-visible{outline:2px solid var(--accent)!important;outline-offset:2px}
[data-testid="stFileUploaderDropzone"]{background:var(--surface);border:1.5px dashed var(--border);border-radius:14px}
[data-testid="stFileUploaderDropzone"] *{color:var(--text)!important}
[data-testid="stFileUploaderDropzone"] button{color:var(--text)!important}
.stTabs [data-baseweb="tab"]{color:var(--muted)}
.stTabs [data-baseweb="tab"]:hover{color:var(--text)}
.stTabs [aria-selected="true"]{color:var(--accent)!important;font-weight:600}
.stTabs [data-baseweb="tab-highlight"]{background:var(--accent)!important}
.stTabs [data-baseweb="tab-border"]{background:var(--border)!important}
[data-testid="stExpander"] details,[data-testid="stForm"],[data-testid="stVerticalBlockBorderWrapper"]:has(>div>[data-testid="stVerticalBlock"]>.dt-flag){border-color:var(--border)!important;background:var(--surface);border-radius:14px}
[data-testid="stExpander"] summary p{color:var(--text)}
[data-testid="stCode"] pre,.stCode pre{background:var(--surface2)!important;border:1px solid var(--border);border-radius:10px}
[data-testid="stCode"] code{color:var(--text)!important}
[data-testid="stChatMessage"]{background:var(--surface);border:1px solid var(--border);border-radius:14px}
[data-testid="stChatInput"]>div,[data-testid="stBottom"]>div{background:var(--bg)!important}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:10px}
[data-testid="stSidebar"] [role="radiogroup"]{gap:2px}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.5rem .7rem;border-radius:10px;width:100%;cursor:pointer}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:var(--surface2)}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,color-mix(in srgb,var(--accent) 22%,var(--surface2)),var(--surface2));box-shadow:inset 3px 0 0 var(--accent);font-weight:600}
[data-testid="stSidebar"] [role="radiogroup"] label>div:first-child{display:none}
.dt-card{background:linear-gradient(160deg,color-mix(in srgb,var(--surface) 90%,var(--accent2)),color-mix(in srgb,var(--surface) 88%,transparent));backdrop-filter:blur(10px);border:1px solid var(--border);border-radius:14px;padding:1rem 1.15rem;box-shadow:0 10px 28px -16px rgba(0,0,0,.5);margin-bottom:.6rem;transition:border-color .15s,box-shadow .15s}
.dt-card:hover{border-color:color-mix(in srgb,var(--accent) 60%,var(--border));box-shadow:0 14px 30px -14px rgba(0,0,0,.55)}
.dt-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem;margin:.4rem 0 1.2rem}
.dt-stat{background:linear-gradient(160deg,var(--surface),color-mix(in srgb,var(--surface) 86%,var(--accent)));border:1px solid var(--border);border-top:3px solid var(--accent);border-radius:14px;padding:.9rem 1.1rem;box-shadow:0 10px 24px -16px rgba(0,0,0,.5)}
.dt-stat:nth-child(even){border-top-color:var(--accent2)}
.dt-stat b{display:block;font-family:'Bricolage Grotesque',sans-serif;font-size:1.9rem;line-height:1.1;color:var(--text)}
.dt-stat span{color:var(--muted);font-size:.88rem}
.dt-pill{display:inline-block;padding:.12rem .6rem;border-radius:99px;font-size:.78rem;font-weight:600;border:1px solid var(--border);background:var(--surface2);color:var(--text)}
.dt-ok{border-color:var(--accent);color:var(--accent);background:color-mix(in srgb,var(--accent) 16%,var(--surface2))}
.dt-warn{border-color:var(--warn);color:var(--warn);background:color-mix(in srgb,var(--warn) 14%,var(--surface2))}
.dt-pending{border:1px dashed var(--accent);background:color-mix(in srgb,var(--accent) 10%,transparent);border-radius:12px;padding:.7rem 1rem;color:var(--text);margin-bottom:1rem}
.dt-muted{color:var(--muted);font-size:.9rem}
.dt-hit{background:linear-gradient(160deg,var(--surface),color-mix(in srgb,var(--surface) 90%,var(--accent2)));border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:10px;padding:.8rem 1rem;margin:.5rem 0}
.dt-hit mark{background:color-mix(in srgb,var(--accent) 38%,transparent);color:var(--text);border-radius:3px;padding:0 2px}
.dt-logo{display:flex;align-items:center;gap:.65rem;margin:.2rem 0 1rem}
.dt-logo b{font-family:'Bricolage Grotesque',sans-serif;font-size:1.25rem;color:var(--text)}
.dt-logo small{display:block;color:var(--muted);font-size:.75rem;margin-top:-2px}
.dt-bar{height:8px;border-radius:99px;background:var(--surface2);overflow:hidden}
.dt-bar i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2))}
.reader{background:var(--surface);border:1px solid var(--border);border-top:3px solid var(--accent2);border-radius:14px;margin:0 auto;box-shadow:0 14px 34px -20px rgba(0,0,0,.55)}
.reader p{margin:0 0 1em}
@media (max-width:760px){.block-container{padding:1rem .8rem 4rem}.reader{padding:1rem!important}}
::selection{background:color-mix(in srgb,var(--accent) 40%,transparent)}
a{color:var(--accent)!important}
*{scrollbar-color:color-mix(in srgb,var(--accent) 45%,var(--surface2)) transparent}
.dt-logo svg{filter:drop-shadow(0 0 10px color-mix(in srgb,var(--accent) 45%,transparent))}
[data-testid="stAlertContainer"]{background:color-mix(in srgb,var(--accent) 10%,var(--surface))!important;border:1px solid color-mix(in srgb,var(--accent) 40%,var(--border));border-radius:12px;color:var(--text)!important}
[data-testid="stAlertContainer"] *{color:var(--text)!important}
[data-testid="stProgress"] [role="progressbar"]>div>div{background:linear-gradient(90deg,var(--accent),var(--accent2))!important}
.dt-hero{background:linear-gradient(120deg,color-mix(in srgb,var(--accent) 24%,var(--surface)),color-mix(in srgb,var(--accent2) 28%,var(--surface)));border:1px solid color-mix(in srgb,var(--accent) 38%,var(--border));border-radius:18px;padding:1.2rem 1.5rem;margin-bottom:1rem;box-shadow:0 18px 40px -22px var(--accent)}
.dt-hero h3{margin:0 0 .2rem;font-size:1.5rem}
.dt-hero span{color:var(--muted)}
.dt-swatch{border:2px solid var(--border);border-radius:14px;padding:.8rem 1rem;margin-bottom:.4rem;box-shadow:0 10px 24px -16px rgba(0,0,0,.5)}
.dt-swatch i{display:inline-block;width:18px;height:18px;border-radius:50%;margin-right:6px;border:1px solid rgba(128,128,128,.4)}
.dt-table{max-height:260px;overflow:auto;border:1px solid var(--border);border-radius:12px;background:var(--surface)}
.dt-table table{width:100%;border-collapse:collapse;font-size:.88rem}
.dt-table th{position:sticky;top:0;background:var(--surface2);text-align:left;padding:.45rem .7rem;color:var(--text)}
.dt-table td{padding:.4rem .7rem;border-top:1px solid var(--border);color:var(--text)}
.dt-table tr:hover td{background:color-mix(in srgb,var(--accent) 9%,transparent)}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

LOGO = ('<svg width="38" height="38" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<rect x="8" y="8" width="112" height="112" rx="28" fill="var(--surface2)" stroke="var(--border)"/>'
        '<rect x="36" y="25" width="56" height="78" rx="8" fill="none" stroke="var(--accent)" stroke-width="7"/>'
        '<path d="M50 66l10 11 20-25" fill="none" stroke="var(--accent)" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def inject_css():
    th = THEMES[S.theme]
    v = ";".join(f"--{k}:{x}" for k, x in th.items() if k != "dark")
    fam = FONTS[S.font_family]
    width = min(S.read_width, 680) if S.comfy else S.read_width
    lh = max(S.line_height, 1.85) if S.comfy else S.line_height
    pad = "2.2rem 2.6rem" if S.comfy else "1.4rem 1.8rem"
    focus = ('[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="stToolbar"]{display:none!important}'
             if S.focus else "")
    st.markdown(f"<style>:root{{{v};color-scheme:{'dark' if th['dark'] else 'light'}}}{CSS}"
                f".reader{{font-family:{fam};font-size:{S.font_size}px;line-height:{lh};max-width:{width}px;padding:{pad};"
                f"{'letter-spacing:.012em;' if S.comfy else ''}}}{focus}</style>", unsafe_allow_html=True)


def theme_vars():
    th = THEMES[S.theme]
    return ";".join(f"--{k}:{x}" for k, x in th.items() if k != "dark")


# ============================================================ PDF PROCESSING
def split_sentences(text):
    t = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", t) if s.strip()]


def chunk_text(text, size, overlap):
    sents, chunks, cur = split_sentences(text), [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > size:
            chunks.append(cur)
            cur = cur[-overlap:].split(" ", 1)[-1] + " " + s if overlap else s
        else:
            cur = f"{cur} {s}".strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def process_pdf(name, data, cfg, on_page):
    pdf = pymupdf.open(stream=data, filetype="pdf")
    pages, chunks = [], []
    for i, page in enumerate(pdf):
        text, method, err = page.get_text("text").strip(), "text", ""
        if len(text) < 30:
            if cfg["ocr_on"] and OCR_OK:
                try:
                    pix = page.get_pixmap(dpi=cfg["dpi"])
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr = pytesseract.image_to_string(img, lang=cfg["ocr_lang"]).strip()
                    text, method = (ocr, "ocr") if ocr else (text, "empty")
                except Exception as e:
                    method, err = "ocr_failed", str(e)[:120]
            else:
                method = "needs_ocr" if not text else "text"
        pages.append(dict(page=i + 1, method=method, chars=len(text), words=len(text.split()), text=text, error=err))
        for j, c in enumerate(chunk_text(text, cfg["chunk_size"], cfg["overlap"])):
            chunks.append(dict(id=f"p{i + 1}-c{j + 1}", page=i + 1, index=j + 1, method=method, text=c))
        on_page(i + 1, len(pdf))
    meta = {k: v for k, v in (pdf.metadata or {}).items() if v}
    n_pages = len(pdf)
    pdf.close()
    bad = sum(p["method"] in ("needs_ocr", "ocr_failed", "empty") for p in pages)
    status = "ready" if bad == 0 else ("needs_ocr" if bad == n_pages else "partial")
    sha = hashlib.sha256(data).hexdigest()
    return dict(id=sha[:12], name=name, size=len(data), sha256=sha, page_count=n_pages, metadata=meta, pages=pages,
                chunks=chunks, status=status, ocr_pages=sum(p["method"] == "ocr" for p in pages),
                processed_at=datetime.now().isoformat(timespec="seconds"))


def doc_json(d):
    return json.dumps({k: d[k] for k in ("name", "size", "sha256", "page_count", "metadata", "status", "ocr_pages",
                                         "processed_at", "pages", "chunks")}, ensure_ascii=False, indent=2)


def docs_list():
    return sorted(S.docs.values(), key=lambda d: d["processed_at"], reverse=True)


def status_pill(s):
    cls, label = {"ready": ("dt-ok", t("ready")), "partial": ("dt-warn", t("partial"))}.get(s, ("dt-warn", t("needs_ocr")))
    return f'<span class="dt-pill {cls}">{label}</span>'


def html_table(rows):
    cols = list(rows[0]) if rows else []
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<div class='dt-table'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def esc(s):
    return html.escape(s)


def go_reader(doc_id, page=1):
    S.rd_doc, S.rd_page, S.nav = doc_id, page, "reader"
    S.pop("w_rd_doc", None)
    S.pop("w_rd_page", None)


def need_docs():
    if not S.docs:
        st.info(t("no_docs"))
        return True
    return False


def page_units(scope):
    return [(d["name"], p["page"], p["text"]) for d, lo, hi in scope for p in d["pages"] if lo <= p["page"] <= hi and p["text"]]


def scope_picker(key):
    """Document + page-range picker. Returns [(doc, lo, hi)]."""
    ids = list(S.docs)
    sel = pw(st.multiselect, t("select_docs"), key + "_docs", ids, options=ids, format_func=lambda i: S.docs[i]["name"])
    if len(sel) == 1 and S.docs[sel[0]]["page_count"] > 1:
        n = S.docs[sel[0]]["page_count"]
        cur = S.get(key + "_rng", (1, n))
        S[key + "_rng"] = (max(1, min(cur[0], n)), max(1, min(cur[1], n)))
        if "w_" + key + "_rng" in S:
            S["w_" + key + "_rng"] = S[key + "_rng"]
        lo, hi = pw(st.slider, t("pages_range"), key + "_rng", (1, n), min_value=1, max_value=n)
    else:
        lo, hi = 1, 10 ** 6
    return [(S.docs[i], lo, hi) for i in sel]


# ============================================================ TEXT ANALYSIS (rule-based, no AI)
STOP = set("""the and for with that this from are was were been have has had not but you your they their them its into than
then which who whom whose can may will shall also such these those there here what when where while about over under
more most some any each other only very just like use used using via per our out one two""".split())
PRON = {"it", "this", "that", "there", "these", "those", "he", "she", "they", "we", "what", "which", "who", "i", "you"}
DEF_RE = re.compile(r"^(?:An?\s+|The\s+)?([A-Z][A-Za-z0-9\- ]{2,40}?)\s+(?:is defined as|is called|refers to|is known as|means|is|are)\s+(.{20,260})$")
FORM_RE = re.compile(r"^[^\n]{0,40}[A-Za-z\)\]]\s*=\s*[\w\(\[\-+*/^. ]{1,50}$")


def words(s):
    return re.findall(r"[A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\-']{2,}", s.lower())


def all_sentences(units):
    return [(s, d, p) for d, p, tx in units for s in split_sentences(tx) if 25 <= len(s) <= 400]


def extract_defs(units):
    out, seen = [], set()
    for s, d, p in all_sentences(units):
        m = DEF_RE.match(s)
        if m and m.group(1).split()[0].lower() not in PRON and len(m.group(1).split()) <= 5:
            term = m.group(1).strip()
            if term.lower() not in seen:
                seen.add(term.lower())
                out.append((term, m.group(2).strip().rstrip("."), d, p, s))
    return out


def extract_formulas(units):
    out = []
    for d, p, tx in units:
        for line in tx.splitlines():
            line = line.strip()
            if "=" in line and FORM_RE.match(line) and line not in [x[0] for x in out]:
                out.append((line, d, p))
    return out[:15]


def key_sentences(units, n):
    sents = all_sentences(units)
    freq = Counter(w for s, _, _ in sents for w in words(s) if w not in STOP and len(w) > 3)

    def score(i):
        ws = set(words(sents[i][0]))
        return -sum(freq[w] for w in ws) / (len(ws) ** 0.6 + 1)

    top = sorted(sorted(range(len(sents)), key=score)[:n])
    return [sents[i] for i in top]


def build_notes(units, exam):
    ks = key_sentences(units, 14)
    defs, forms = extract_defs(units), extract_formulas(units)
    md = ["# Study notes", f"_{t('rule_note')}_", ""]
    md += ["## Summary"] + [f"- {s} _({d}, p.{p})_" for s, d, p in ks[:5]]
    if ks[5:]:
        md += ["", "## Key points"] + [f"- {s} _({d}, p.{p})_" for s, d, p in ks[5:]]
    if defs:
        md += ["", "## Important definitions"] + [f"- **{a}**: {b} _({d}, p.{p})_" for a, b, d, p, _ in defs[:15]]
    if forms:
        md += ["", "## Formulas found"] + [f"- `{f}` _({d}, p.{p})_" for f, d, p in forms]
    if exam:
        md += ["", "## Exam practice prompts"]
        md += [f"- Define **{a}** and give one example." for a, *_ in defs[:6]]
        md += [f"- Explain, in your own words: {s[:110]}…" for s, _, _ in ks[:4]]
    return "\n".join(md)


def cloze_items(units, rng, limit=60):
    sents = all_sentences(units)
    vocab = Counter(w for s, _, _ in sents for w in words(s) if len(w) >= 6 and w not in STOP)
    rng.shuffle(sents)
    items = []
    for s, d, p in sents:
        ws = [w for w in words(s) if len(w) >= 6 and w not in STOP]
        if not ws:
            continue
        kw = max(ws, key=lambda w: vocab[w])
        same = [w for w in vocab if w != kw and abs(len(w) - len(kw)) <= 2 and w not in s.lower()]
        items.append(dict(sent=s, doc=d, page=p, kw=kw, pool=same, blank=re.sub(kw, "_____", s, count=1, flags=re.I)))
        if len(items) >= limit:
            break
    return items


def build_quiz(units, level, n, seed):
    rng, qs = random.Random(seed), []
    if level in ("Easy", "Medium"):
        defs = extract_defs(units)
        if len(defs) < 4:
            return [], "Not enough definition-style sentences (need at least 4) in the selected pages. Try Hard level or more pages."
        rng.shuffle(defs)
        for term, dfn, d, p, sent in defs:
            others = [x[0] for x in defs if x[0] != term]
            if level == "Medium":
                others.sort(key=lambda o: abs(len(o) - len(term)))
                others = others[:6]
            if len(others) < 3:
                continue
            opts = rng.sample(others, 3) + [term]
            rng.shuffle(opts)
            qs.append(dict(q=f"Which term does this describe?\n\n> {dfn}", opts=opts, ans=opts.index(term),
                           exp=f"From {d}, page {p}: {sent}"))
            if len(qs) >= n:
                break
    else:
        for it in cloze_items(units, rng):
            if len(it["pool"]) < 3:
                continue
            opts = rng.sample(it["pool"], 3) + [it["kw"]]
            rng.shuffle(opts)
            qs.append(dict(q=f"Fill in the blank:\n\n> {it['blank']}", opts=opts, ans=opts.index(it["kw"]),
                           exp=f"From {it['doc']}, page {it['page']}: {it['sent']}"))
            if len(qs) >= n:
                break
    if not qs:
        return [], "Could not build questions from the selected text. Select more pages."
    return qs, ""


def build_cards(units, n, seed):
    rng = random.Random(seed)
    cards = [dict(front=a, back=b, src=f"{d}, p.{p}") for a, b, d, p, _ in extract_defs(units)]
    for it in cloze_items(units, rng, n):
        if len(cards) >= n:
            break
        cards.append(dict(front=it["blank"], back=it["kw"], src=f"{it['doc']}, p.{it['page']}"))
    rng.shuffle(cards)
    for i, c in enumerate(cards[:n]):
        c["id"] = hashlib.md5((c["front"] + c["back"]).encode()).hexdigest()[:8]
    return cards[:n]


def keyword_retrieve(query, k=4):
    qw = [w for w in words(query) if w not in STOP]
    hits = []
    for d in S.docs.values():
        for c in d["chunks"]:
            low = c["text"].lower()
            sc = sum(low.count(w) for w in qw)
            if sc:
                hits.append((sc, d["name"], c["page"], c["text"]))
    hits.sort(key=lambda x: -x[0])
    return [dict(doc=h[1], page=h[2], text=h[3]) for h in hits[:k]]


def ai_button(task, key, passages_fn, options):
    if st.button(t("ai_btn") if not RETRIEVAL_CONNECTED else t("generate") + " (AI)", key=key,
                 disabled=not RETRIEVAL_CONNECTED, icon=":material/auto_awesome:"):
        try:
            return ai_generate(task, passages_fn(), options)
        except Exception as e:
            st.error(f"AI backend error: {e}")
    return None


# ============================================================ COMPONENTS (browser-side)
def pomodoro():
    th = theme_vars()
    components.html("""<style>:root{__TH__}body{margin:0;font-family:system-ui,sans-serif;color:var(--text);background:transparent}
.b{display:flex;gap:6px;margin-top:6px}button{flex:1;padding:6px;border-radius:8px;border:1px solid var(--border);background:var(--surface2);color:var(--text);cursor:pointer}
button.on{background:var(--accent);color:var(--on);border-color:var(--accent)}#t{font-size:34px;font-weight:700;text-align:center;font-variant-numeric:tabular-nums}
#m{display:flex;gap:6px}#m button{font-size:12px}#c{font-size:12px;text-align:center;color:var(--muted)}</style>
<div id="m"></div><div id="t">25:00</div><div id="c"></div>
<div class="b"><button id="go">__START__</button><button id="rs">__RESET__</button></div>
<script>
const M={Focus:25,Short:5,Long:15};let mode='Focus',left=25*60,tm=null,done=0;
const $=i=>document.getElementById(i);
function draw(){$('t').textContent=String(Math.floor(left/60)).padStart(2,'0')+':'+String(left%60).padStart(2,'0');$('c').textContent='Sessions: '+done;
 document.querySelectorAll('#m button').forEach(b=>b.className=b.textContent==mode?'on':'')}
Object.keys(M).forEach(k=>{const b=document.createElement('button');b.textContent=k;b.onclick=()=>{mode=k;left=M[k]*60;stop();draw()};$('m').appendChild(b)});
function stop(){clearInterval(tm);tm=null;$('go').textContent='__START__'}
function beep(){try{const a=new AudioContext(),o=a.createOscillator();o.connect(a.destination);o.frequency.value=880;o.start();setTimeout(()=>o.stop(),400)}catch(e){}}
$('go').onclick=()=>{if(tm){stop();return}$('go').textContent='__PAUSE__';tm=setInterval(()=>{left--;if(left<=0){beep();if(mode=='Focus')done++;stop();left=M[mode]*60}draw()},1000)};
$('rs').onclick=()=>{stop();left=M[mode]*60;draw()};draw();
</script>""".replace("__TH__", th).replace("__START__", t("start")).replace("__PAUSE__", t("pause")).replace("__RESET__", t("reset")),
        height=165)


def tts(text, lang, rate):
    th = theme_vars()
    components.html("""<style>:root{__TH__}body{margin:0;font-family:system-ui,sans-serif;color:var(--text)}
.r{display:flex;flex-wrap:wrap;gap:8px;align-items:center}button{padding:8px 16px;border-radius:10px;border:1px solid var(--border);background:var(--surface2);color:var(--text);cursor:pointer;font-size:14px}
button.p{background:var(--accent);color:var(--on);border-color:var(--accent);font-weight:600}input{accent-color:var(--accent)}#s{margin-top:10px;color:var(--muted);font-size:13px}</style>
<div class="r"><button class="p" id="a">__PLAY__</button><button id="b">__PAUSE__</button><button id="c">__RESUME__</button><button id="d">__STOP__</button>
<label style="font-size:13px">__SPEED__ <input id="sp" type="range" min="0.5" max="2" step="0.25" value="__RATE__"> <span id="sv">__RATE__x</span></label></div><div id="s">Ready</div>
<script>
const T=__TEXT__,L="__LANG__",synth=window.speechSynthesis;let ch=[],i=0,st='idle';const $=x=>document.getElementById(x);
const say=m=>$('s').textContent=m;
function split(t){const o=[];let c='';for(const s of t.split(/(?<=[.!?\\u0964\\n])\\s*/)){if((c+' '+s).length>170){if(c)o.push(c);c=s}else c+=' '+s}if(c.trim())o.push(c);return o.map(x=>x.trim()).filter(Boolean)}
function next(){if(st!=='playing')return;if(i>=ch.length){st='idle';say('Finished');return}
 const u=new SpeechSynthesisUtterance(ch[i]);u.lang=L;u.rate=parseFloat($('sp').value);
 const v=synth.getVoices().find(v=>v.lang.toLowerCase().startsWith(L.slice(0,2)));if(v)u.voice=v;
 u.onend=()=>{i++;next()};u.onerror=e=>{if(!['interrupted','canceled'].includes(e.error))say('Speech error: '+e.error)};
 synth.speak(u);say('Playing '+(i+1)+' / '+ch.length+(v?'':' (no '+L+' voice installed in this browser, using default)'))}
$('sp').oninput=()=>$('sv').textContent=$('sp').value+'x';
$('a').onclick=()=>{synth.cancel();ch=split(T);i=0;if(!ch.length){say('No text to read');return}st='playing';next()};
$('b').onclick=()=>{if(st==='playing'){synth.pause();st='paused';say('Paused')}};
$('c').onclick=()=>{if(st==='paused'){st='playing';synth.resume();say('Playing '+(i+1)+' / '+ch.length)}};
$('d').onclick=()=>{st='idle';synth.cancel();i=0;say('Stopped')};
window.addEventListener('pagehide',()=>synth.cancel());
</script>""".replace("__TH__", th).replace("__TEXT__", json.dumps(text).replace("</", "<\\/")).replace("__LANG__", lang)
        .replace("__RATE__", str(rate)).replace("__PLAY__", t("play")).replace("__PAUSE__", t("pause"))
        .replace("__RESUME__", t("resume")).replace("__STOP__", t("stop")).replace("__SPEED__", t("speed")), height=110)


def flashcard(card):
    th = theme_vars()
    components.html("""<style>:root{__TH__}body{margin:0;font-family:system-ui,'Noto Sans Devanagari',sans-serif}
.s{perspective:1000px;height:230px;cursor:pointer}.i{position:relative;height:100%;transition:transform .55s;transform-style:preserve-3d}
.s.f .i{transform:rotateY(180deg)}.c{position:absolute;inset:0;backface-visibility:hidden;border-radius:16px;border:1px solid var(--border);background:var(--surface);color:var(--text);
display:flex;flex-direction:column;align-items:center;justify-content:center;padding:22px;text-align:center;box-shadow:0 12px 30px -18px rgba(0,0,0,.6);overflow:auto}
.b{transform:rotateY(180deg);border-color:var(--accent)}small{color:var(--muted);margin-top:12px}.t{font-size:19px;line-height:1.45}@media(prefers-reduced-motion:reduce){.i{transition:none}}</style>
<div class="s" id="s" tabindex="0" role="button" aria-label="Flip card"><div class="i"><div class="c"><div class="t">__F__</div><small>Click to flip</small></div>
<div class="c b"><div class="t">__B__</div><small>__SRC__</small></div></div></div>
<script>const s=document.getElementById('s');s.onclick=()=>s.classList.toggle('f');s.onkeydown=e=>{if(e.key===' '||e.key==='Enter'){e.preventDefault();s.click()}}</script>"""
                    .replace("__TH__", th).replace("__F__", esc(card["front"])).replace("__B__", esc(card["back"]))
                    .replace("__SRC__", esc(card["src"])), height=250)


def fullscreen_button():
    th = theme_vars()
    components.html("""<style>:root{__TH__}body{margin:0}button{width:100%;padding:8px;border-radius:10px;border:1px solid var(--border);background:var(--surface2);color:var(--text);cursor:pointer;font-size:14px}</style>
<button id="f">&#x26F6; __L__</button><script>document.getElementById('f').onclick=()=>{const d=window.parent.document;
if(d.fullscreenElement)d.exitFullscreen();else d.documentElement.requestFullscreen().catch(()=>{})}</script>"""
                    .replace("__TH__", th).replace("__L__", t("fullscreen")), height=42)


def shortcuts_js():
    components.html("""<script>
const W=window.parent;if(!W.__dtKeys){W.__dtKeys=1;const P=W.document;
P.addEventListener('keydown',e=>{const tg=(e.target.tagName||'').toLowerCase();
 const typing=['input','textarea','select'].includes(tg)||e.target.isContentEditable;
 const click=s=>{const el=P.querySelector(s);if(el)el.click()};
 if(e.altKey&&/^[1-9]$/.test(e.key)){e.preventDefault();const l=P.querySelectorAll('[data-testid="stSidebar"] [role="radiogroup"] label');if(l[+e.key-1])l[+e.key-1].click();return}
 if(e.altKey&&e.key.toLowerCase()==='f'){e.preventDefault();click('.st-key-btn_focus button');return}
 if(e.altKey&&e.key==='Enter'){e.preventDefault();P.fullscreenElement?P.exitFullscreen():P.documentElement.requestFullscreen().catch(()=>{});return}
 if(e.altKey&&e.key==='ArrowRight'){e.preventDefault();click('.st-key-rd_next button');return}
 if(e.altKey&&e.key==='ArrowLeft'){e.preventDefault();click('.st-key-rd_prev button');return}
 if(!typing&&!e.ctrlKey&&!e.metaKey&&e.key==='/'){e.preventDefault();const i=P.querySelector('[data-testid="stMain"] input[type="text"],section.main input[type="text"]');if(i)i.focus()}
})}</script>""", height=0)


SHORTCUT_GUIDE = [("Alt + 1 … 9", "Go to Dashboard, Workspace, Library, Reader, Search, Ask AI, Study, Listen, Settings"),
                  ("/", "Focus the search box (ignored while typing)"), ("Alt + F", "Toggle focus mode"),
                  ("Alt + Enter", "Toggle fullscreen"), ("Alt + ← / →", "Previous / next page in Reader")]


def shortcut_table():
    st.markdown("".join(f"<div class='dt-muted' style='margin:.35rem 0'><span class='dt-pill'>{k}</span> &nbsp;{v}</div>"
                        for k, v in SHORTCUT_GUIDE), unsafe_allow_html=True)


# ============================================================ PAGES
def page_dashboard():
    ds = docs_list()
    pages_n = sum(d["page_count"] for d in ds)
    chunks_n = sum(len(d["chunks"]) for d in ds)
    ocr_n = sum(d["ocr_pages"] for d in ds)
    hr = datetime.now().hour
    greet = t("greet_m" if hr < 12 else "greet_a" if hr < 17 else "greet_e")
    st.markdown(f"<div class='dt-hero'><h3>{greet}</h3><span>{t('hero_sub').format(n=len(ds), p=pages_n)}</span></div>",
                unsafe_allow_html=True)
    st.markdown("<div class='dt-stats'>" + "".join(f"<div class='dt-stat'><b>{v}</b><span>{k}</span></div>" for k, v in
                [(t("docs"), len(ds)), (t("pages"), pages_n), (t("chunks"), chunks_n), (t("ocr_pages"), ocr_n)]) + "</div>",
                unsafe_allow_html=True)
    if need_docs():
        st.button(t("workspace"), key="dash_go", type="primary", icon=":material/upload_file:",
                  on_click=lambda: S.__setitem__("nav", "workspace"))
        return
    L, R = st.columns([3, 2])
    with L:
        st.subheader(t("recent"))
        for d in ds[:5]:
            st.markdown(f"<div class='dt-card'><b>{esc(d['name'])}</b> &nbsp;{status_pill(d['status'])}<br>"
                        f"<span class='dt-muted'>{d['page_count']} {t('page').lower()} · {len(d['chunks'])} {t('chunks').lower()} · "
                        f"{d['ocr_pages']} OCR · {d['processed_at'].replace('T', ' ')}</span></div>", unsafe_allow_html=True)
            st.button(t("open_reader"), key=f"dash_open_{d['id']}", on_click=go_reader, args=(d["id"],),
                      icon=":material/menu_book:")
    with R:
        st.subheader(t("status"))
        ready = sum(d["status"] == "ready" for d in ds)
        pct = int(100 * ready / len(ds))
        st.markdown(f"<div class='dt-card'>{ready}/{len(ds)} {t('ready').lower()}<div class='dt-bar' style='margin-top:.5rem'>"
                    f"<i style='width:{pct}%'></i></div></div>", unsafe_allow_html=True)
        st.subheader(t("pomodoro"))
        pomodoro()


def page_workspace():
    if pymupdf is None:
        st.error("PyMuPDF is not installed. Run:  pip install pymupdf")
        return
    with st.expander(t("ocr_settings")):
        c1, c2, c3 = st.columns(3)
        with c1:
            pw(st.toggle, "OCR for scanned pages", "ocr_on", True)
            pw(st.selectbox, "OCR language", "ocr_lang", "eng", options=["eng", "hin", "eng+hin"])
        with c2:
            pw(st.slider, "OCR render DPI", "dpi", 200, min_value=100, max_value=300, step=50)
        with c3:
            pw(st.slider, "Chunk size (chars)", "chunk_size", 900, min_value=300, max_value=2000, step=100)
            pw(st.slider, "Chunk overlap (chars)", "overlap", 120, min_value=0, max_value=400, step=20)
        if not OCR_OK:
            st.warning("OCR engine not available, so scanned pages will be flagged 'Needs OCR'. "
                       "Install Tesseract and `pip install pytesseract` (see the run guide). " + OCR_ERR)
        elif "hin" in S.ocr_lang:
            st.caption("Hindi OCR needs the 'hin' Tesseract language data installed.")
    files = st.file_uploader(t("upload"), type=["pdf"], accept_multiple_files=True, key="uploader")
    if st.button(t("process"), type="primary", disabled=not files, key="btn_process", icon=":material/play_arrow:"):
        cfg = dict(ocr_on=S.ocr_on, ocr_lang=S.ocr_lang, dpi=S.dpi, chunk_size=S.chunk_size, overlap=S.overlap)
        S.last_run = []
        bar = st.progress(0.0)
        for fi, f in enumerate(files):
            data = f.getvalue()
            sha = hashlib.sha256(data).hexdigest()[:12]
            if sha in S.docs and S.docs[sha].get("cfg") == cfg:
                st.info(f"{f.name}: already processed with these settings.")
                S.last_run.append(sha)
                continue
            try:
                def on_page(i, n, fi=fi, name=f.name):
                    bar.progress((fi + i / n) / len(files), text=f"{name}: page {i}/{n}")
                d = process_pdf(f.name, data, cfg, on_page)
                d["cfg"] = cfg
                S.docs[d["id"]] = d
                S.last_run.append(d["id"])
                if S.rd_doc is None:
                    S.rd_doc = d["id"]
            except Exception as e:
                st.error(f"Could not process {f.name}: {e}")
        bar.progress(1.0, text="Done")
    for did in [i for i in S.last_run if i in S.docs]:
        d = S.docs[did]
        with st.expander(f"{d['name']}  ({d['page_count']} pages, {len(d['chunks'])} chunks)", expanded=len(S.last_run) == 1):
            st.markdown(status_pill(d["status"]), unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                st.markdown("**Metadata**")
                meta = {"size_kb": round(d["size"] / 1024, 1), "pages": d["page_count"], "sha256": d["sha256"][:16] + "…",
                        "ocr_pages": d["ocr_pages"], **d["metadata"]}
                st.markdown(html_table([{"field": k, "value": v} for k, v in meta.items()]), unsafe_allow_html=True)
            with b:
                st.markdown("**Page-wise extraction method**")
                st.markdown(html_table([{k: p[k] for k in ("page", "method", "chars", "words")} for p in d["pages"]]),
                            unsafe_allow_html=True)
            st.markdown("**Chunk preview**")
            for c in d["chunks"][:5]:
                st.markdown(f"<div class='dt-hit'><span class='dt-muted'>{c['id']} · {c['method']}</span><br>{esc(c['text'][:420])}</div>",
                            unsafe_allow_html=True)
            st.download_button(t("download"), doc_json(d), file_name=f"{os.path.splitext(d['name'])[0]}.json",
                               mime="application/json", key=f"ws_dl_{did}", icon=":material/download:")
    if len(S.docs) > 1:
        st.download_button(t("download") + " (all)", json.dumps([json.loads(doc_json(d)) | {"name": d["name"]} for d in docs_list()],
                           ensure_ascii=False, indent=2), file_name="docutrust_all.json", mime="application/json", key="ws_dl_all")


def page_library():
    if need_docs():
        return
    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        q = pw(st.text_input, t("search_ph"), "lib_q", "")
    with c2:
        flt = pw(st.selectbox, t("filter"), "lib_f", "all", options=["all", "ready", "partial", "needs_ocr", "has_ocr"])
    with c3:
        srt = pw(st.selectbox, t("sort"), "lib_s", "recent", options=["recent", "name", "pages"])
    ds = docs_list()
    ds = [d for d in ds if q.lower() in d["name"].lower()]
    if flt == "has_ocr":
        ds = [d for d in ds if d["ocr_pages"]]
    elif flt != "all":
        ds = [d for d in ds if d["status"] == flt]
    if srt == "name":
        ds.sort(key=lambda d: d["name"].lower())
    elif srt == "pages":
        ds.sort(key=lambda d: -d["page_count"])
    if not ds:
        st.info("No documents match these filters.")
    cols = st.columns(2)
    for n, d in enumerate(ds):
        with cols[n % 2]:
            st.markdown(f"<div class='dt-card'><b>{esc(d['name'])}</b><br>{status_pill(d['status'])}<br>"
                        f"<span class='dt-muted'>{d['page_count']} pages · {len(d['chunks'])} chunks · {d['ocr_pages']} OCR · "
                        f"{round(d['size'] / 1024)} KB</span></div>", unsafe_allow_html=True)
            with st.expander(t("page") + " 1"):
                st.text((d["pages"][0]["text"] or "(no text)")[:900])
            b1, b2, b3 = st.columns(3)
            b1.button(t("open_reader"), key=f"lib_o_{d['id']}", on_click=go_reader, args=(d["id"],))
            b2.download_button("JSON", doc_json(d), file_name=f"{os.path.splitext(d['name'])[0]}.json", key=f"lib_d_{d['id']}")
            b3.button(t("remove"), key=f"lib_r_{d['id']}", on_click=lambda i=d["id"]: S.docs.pop(i, None))
    st.caption("Documents live in this browser session only; they are cleared when you close or refresh the tab.")


def page_reader():
    if need_docs():
        return
    ids = list(S.docs)
    if S.rd_doc not in S.docs:
        S.rd_doc = ids[0]
    top = st.columns([4, 1, 1, 1])
    with top[0]:
        pw(st.selectbox, t("document"), "rd_doc", ids[0], options=ids, format_func=lambda i: S.docs[i]["name"])
    d = S.docs[S.rd_doc]
    n = d["page_count"]
    S.rd_page = max(1, min(S.rd_page, n))
    if "w_rd_page" in S:
        S["w_rd_page"] = S.rd_page

    def step(delta):
        S.rd_page = max(1, min(n, S.rd_page + delta))
        S["w_rd_page"] = S.rd_page
    with top[1]:
        st.write("")
        st.button(t("prev"), key="rd_prev", on_click=step, args=(-1,), disabled=S.rd_page <= 1, icon=":material/chevron_left:")
    with top[2]:
        st.write("")
        st.button(t("next"), key="rd_next", on_click=step, args=(1,), disabled=S.rd_page >= n, icon=":material/chevron_right:")
    with top[3]:
        st.write("")
        fullscreen_button()
    if n > 1:
        pw(st.slider, t("page"), "rd_page", 1, min_value=1, max_value=n)
    p = d["pages"][S.rd_page - 1]
    st.caption(f"{t('page')} {p['page']}/{n} · {p['method']} · {p['words']} words")
    body = "".join(f"<p>{esc(x.strip()).replace(chr(10), ' ')}</p>" for x in re.split(r"\n\s*\n", p["text"]) if x.strip())
    st.markdown(f"<div class='reader'>{body or '<p class=dt-muted>No text on this page (scanned page? enable OCR in PDF Workspace).</p>'}</div>",
                unsafe_allow_html=True)


def snippet(text, terms, width=240):
    low = text.lower()
    pos = min([low.find(x) for x in terms if x in low] or [0])
    seg = text[max(0, pos - width // 3): max(0, pos - width // 3) + width].replace("\n", " ")
    out = esc(seg)
    for term in sorted(terms, key=len, reverse=True):
        out = re.sub(f"({re.escape(esc(term))})", r"<mark>\1</mark>", out, flags=re.I)
    return ("…" if pos > width // 3 else "") + out + "…"


def page_search():
    if need_docs():
        return
    st.caption("Keyword search across extracted text. Semantic search: pending Garvit's retrieval integration.")
    q = pw(st.text_input, t("search_ph"), "sq", "")
    c1, c2 = st.columns([3, 2])
    with c1:
        ids = list(S.docs)
        sel = pw(st.multiselect, t("select_docs"), "sq_docs", ids, options=ids, format_func=lambda i: S.docs[i]["name"])
    with c2:
        mode = pw(st.radio, "Match", "sq_mode", "Any keyword", options=["Any keyword", "All keywords", "Exact phrase"], horizontal=True)
    terms = [q.lower().strip()] if mode == "Exact phrase" else [w for w in re.findall(r"\w+", q.lower()) if len(w) > 1]
    if not q.strip() or not terms:
        st.info("Type a keyword or phrase to search all selected documents.")
        return
    hits = []
    for i in sel:
        d = S.docs[i]
        for p in d["pages"]:
            low = p["text"].lower()
            counts = [low.count(x) for x in terms]
            ok = all(counts) if mode != "Any keyword" else any(counts)
            if ok:
                hits.append((sum(counts), d, p))
    hits.sort(key=lambda h: -h[0])
    st.write(f"**{len(hits)}** matching pages")
    for n, (sc, d, p) in enumerate(hits[:30]):
        st.markdown(f"<div class='dt-hit'><b>{esc(d['name'])}</b> · {t('page')} {p['page']} · {sc} matches<br>"
                    f"{snippet(p['text'], terms)}</div>", unsafe_allow_html=True)
        st.button(t("open_reader"), key=f"sq_o_{n}", on_click=go_reader, args=(d["id"], p["page"]))


def page_ask():
    st.caption("Ask questions across your processed PDFs. Answers are grounded in retrieved passages and include page citations.")
    if S.chat:
        st.button(t("clear"), key="chat_clear", on_click=lambda: S.__setitem__("chat", []), icon=":material/delete:")
    for mi, m in enumerate(S.chat):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for pi, ps in enumerate(m.get("passages", [])):
                with st.expander(f"{ps['doc']} · {t('page')} {ps['page']}", expanded=pi == 0):
                    st.code(ps["text"], language=None, wrap_lines=True)
    q = st.chat_input(t("ask_ph"), key="chat_in", disabled=not S.docs)
    if not S.docs:
        st.info(t("no_docs"))
    if q:
        S.chat.append(dict(role="user", content=q))
        ps = keyword_retrieve(q)
        try:
            answer = ai_generate("answer", ps, {"question": q})
            S.chat.append(dict(role="assistant", content=answer, passages=ps))
        except Exception as e:
            S.chat.append(dict(role="assistant", content=f"AI backend error: {e}", passages=ps))
        st.rerun()


def page_study():
    if need_docs():
        return
    scope = scope_picker("st")
    units = page_units(scope)
    if not units:
        st.warning("The selected pages have no extracted text.")
        return
    tn, tq, tc = st.tabs([t("notes"), t("quiz"), t("cards")])
    with tn:
        exam = pw(st.toggle, "Exam-oriented notes", "exam", False)
        c1, c2 = st.columns(2)
        if c1.button(t("generate"), key="btn_notes", type="primary", icon=":material/edit_note:"):
            S.notes_out = build_notes(units, exam)
        ai = ai_button("notes", "btn_notes_ai", lambda: [dict(doc=d, page=p, text=x) for d, p, x in units], {"exam": exam})
        if ai:
            S.notes_out = str(ai)
        if S.notes_out:
            st.caption(t("rule_note"))
            st.markdown(S.notes_out)
            st.download_button("Download .md", S.notes_out, file_name="study_notes.md", key="notes_dl")
            with st.expander("Copy notes"):
                st.code(S.notes_out, language="markdown", wrap_lines=True)
    with tq:
        st.caption(t("rule_note"))
        c1, c2 = st.columns(2)
        with c1:
            lvl = pw(st.selectbox, "Difficulty", "qz_lvl", "Easy", options=["Easy", "Medium", "Hard"])
        with c2:
            cnt = pw(st.slider, "Questions", "qz_n", 5, min_value=3, max_value=15)
        if st.button(t("generate") if not S.quiz else t("new_quiz"), key="btn_quiz", type="primary", icon=":material/quiz:"):
            seed = random.randrange(10 ** 6)
            qs, err = build_quiz(units, lvl, cnt, seed)
            S.quiz = dict(qs=qs, seed=seed, done=False, answers=[], level=lvl) if qs else None
            if err:
                st.warning(err)
        ai_quiz = ai_button("quiz", "btn_quiz_ai", lambda: [dict(doc=d, page=p, text=x) for d, p, x in units], {"level": lvl, "n": cnt})
        if ai_quiz:
            try:
                qs = ai_quiz.get("qs", [])
                if not qs:
                    raise ValueError("The AI returned no quiz questions.")
                for item in qs:
                    if len(item.get("opts", [])) != 4 or not 0 <= int(item.get("ans", -1)) < 4:
                        raise ValueError("The AI quiz format was invalid.")
                seed = random.randrange(10 ** 6)
                S.quiz = dict(qs=qs, seed=seed, done=False, answers=[], level=lvl)
                st.success("AI quiz generated.")
            except Exception as e:
                st.error(f"Could not use AI quiz: {e}")
        qz = S.quiz
        if qz and not qz["done"]:
            with st.form(f"quiz_form_{qz['seed']}"):
                for i, q in enumerate(qz["qs"]):
                    st.markdown(f"**Q{i + 1}.** {q['q']}")
                    st.radio("Answer", q["opts"], index=None, key=f"qz_{qz['seed']}_{i}", label_visibility="collapsed")
                if st.form_submit_button("Submit answers", type="primary"):
                    qz["answers"] = [S.get(f"qz_{qz['seed']}_{i}") for i in range(len(qz["qs"]))]
                    qz["done"] = True
                    st.rerun()
        elif qz:
            ok = sum(a == q["opts"][q["ans"]] for a, q in zip(qz["answers"], qz["qs"]))
            pct = round(100 * ok / len(qz["qs"]))
            perf = "Excellent" if pct >= 80 else "Good" if pct >= 60 else "Keep practising" if pct >= 40 else "Needs revision"
            st.markdown(f"<div class='dt-stat'><b>{ok}/{len(qz['qs'])} · {pct}%</b><span>{t('score')} · {perf} · {qz['level']}</span></div>",
                        unsafe_allow_html=True)
            for i, (a, q) in enumerate(zip(qz["answers"], qz["qs"])):
                right = q["opts"][q["ans"]]
                with st.expander(f"Q{i + 1}: {'Correct' if a == right else 'Incorrect'}", expanded=a != right):
                    st.markdown(q["q"])
                    st.write(f"Your answer: {a or 'not answered'} | Correct answer: **{right}**")
                    st.caption(q["exp"])
    with tc:
        st.caption(t("rule_note"))
        n_cards = pw(st.slider, "Cards", "fc_n", 10, min_value=3, max_value=40)
        c1, c2 = st.columns(2)
        if c1.button(t("generate"), key="btn_cards", type="primary", icon=":material/style:"):
            S.fc = build_cards(units, n_cards, random.randrange(10 ** 6))
            S.fc_idx, S.fc_status = 0, {}
        if c2.button(t("shuffle"), key="btn_shuf", disabled=not S.fc, icon=":material/shuffle:"):
            random.shuffle(S.fc)
            S.fc_idx = 0
        ai_cards = ai_button("cards", "btn_cards_ai", lambda: [dict(doc=d, page=p, text=x) for d, p, x in units], {"n": n_cards})
        if ai_cards:
            try:
                cards = ai_cards.get("cards", [])
                if not cards:
                    raise ValueError("The AI returned no flashcards.")
                for card in cards:
                    card.setdefault("src", "PDF evidence")
                    card["id"] = hashlib.md5((card.get("front", "") + card.get("back", "")).encode()).hexdigest()[:8]
                S.fc = cards[:n_cards]
                S.fc_idx, S.fc_status = 0, {}
                st.success("AI flashcards generated.")
            except Exception as e:
                st.error(f"Could not use AI flashcards: {e}")
        if S.fc:
            flt = pw(st.radio, t("filter"), "fc_f", "all", options=["all", "revise", "unmarked"], horizontal=True,
                     format_func=lambda x: {"all": t("all"), "revise": t("revise"), "unmarked": "Unmarked"}[x])
            vis = [c for c in S.fc if flt == "all" or (flt == "revise" and S.fc_status.get(c["id"]) == "revise")
                   or (flt == "unmarked" and c["id"] not in S.fc_status)]
            known = sum(v == "known" for v in S.fc_status.values())
            st.caption(f"{t('known')}: {known} · {t('revise')}: {sum(v == 'revise' for v in S.fc_status.values())} · {len(S.fc)} total")
            if not vis:
                st.info("No cards in this filter.")
            else:
                S.fc_idx = max(0, min(S.fc_idx, len(vis) - 1))
                card = vis[S.fc_idx]
                st.write(f"{S.fc_idx + 1} / {len(vis)}  ·  " + S.fc_status.get(card["id"], "unmarked"))
                flashcard(card)

                def mark(val):
                    S.fc_status[card["id"]] = val
                    S.fc_idx = min(S.fc_idx + 1, len(vis) - 1)

                def move(d):
                    S.fc_idx = max(0, min(S.fc_idx + d, len(vis) - 1))
                b = st.columns(4)
                b[0].button(t("prev"), key="fc_prev", on_click=move, args=(-1,), disabled=S.fc_idx == 0)
                b[1].button(t("revise"), key="fc_rev", on_click=mark, args=("revise",))
                b[2].button(t("known"), key="fc_known", on_click=mark, args=("known",), type="primary")
                b[3].button(t("next"), key="fc_next", on_click=move, args=(1,), disabled=S.fc_idx >= len(vis) - 1)


def page_listen():
    if need_docs():
        return
    ids = list(S.docs)
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        did = pw(st.selectbox, t("document"), "ls_doc", ids[0], options=ids, format_func=lambda i: S.docs[i]["name"])
    d = S.docs[did]
    S.ls_page = max(1, min(int(S.get("ls_page", 1)), d["page_count"]))
    if "w_ls_page" in S:
        S["w_ls_page"] = S.ls_page
    with c2:
        pg = pw(st.number_input, t("page"), "ls_page", 1, min_value=1, max_value=d["page_count"], step=1)
    with c3:
        lang = pw(st.selectbox, t("language"), "ls_lang", "en-US", options=["en-US", "hi-IN", "en-IN"])
    rate = pw(st.select_slider, t("speed"), "ls_rate", 1.0, options=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    pg = max(1, min(int(pg), d["page_count"]))
    full = d["pages"][pg - 1]["text"]
    S[f"ls_text_{did}_{pg}"] = S.get(f"ls_text_{did}_{pg}", full)
    st.caption("Edit the box to read only the part you want, or leave it to read the whole page.")
    txt = st.text_area("Text to read", key=f"ls_text_{did}_{pg}", height=220)
    tts(txt, lang, rate)
    st.caption("Speech uses your browser's built-in voices. Hindi works if a Hindi voice is installed in your OS/browser.")


def set_theme(name):
    S.theme = name
    S["w_theme"] = name


def page_settings():
    a, b = st.columns(2)
    with a:
        st.subheader(t("appearance"))
        pw(st.selectbox, t("theme"), "theme", "Midnight Dark", options=list(THEMES))
        pw(st.selectbox, t("language"), "lang", "English", options=["English", "हिन्दी"])
        st.subheader(t("shortcuts"))
        shortcut_table()
    with b:
        st.subheader(t("reading"))
        pw(st.slider, t("font_size"), "font_size", 17, min_value=12, max_value=28)
        pw(st.selectbox, t("font_family"), "font_family", "Source Serif 4", options=list(FONTS))
        pw(st.slider, t("line_spacing"), "line_height", 1.7, min_value=1.2, max_value=2.4, step=0.1)
        pw(st.slider, t("read_width"), "read_width", 720, min_value=480, max_value=1100, step=20)
        pw(st.toggle, t("comfy"), "comfy", False)
    st.subheader(t("gallery"))
    gc = st.columns(3)
    for n, (name, th) in enumerate(THEMES.items()):
        with gc[n % 3]:
            dots = "".join(f"<i style='background:{th[k]}'></i>" for k in ("accent", "accent2", "surface2", "text"))
            st.markdown(f"<div class='dt-swatch' style='background:linear-gradient(135deg,{th['bg']},{th['bg2']});"
                        f"border-color:{th['accent'] if name == S.theme else th['border']}'>"
                        f"<b style='color:{th['text']}'>{name}</b><br>{dots}</div>", unsafe_allow_html=True)
            st.button(t("active") if name == S.theme else t("use"), key=f"th_{n}", on_click=set_theme, args=(name,),
                      disabled=name == S.theme, use_container_width=True)
    st.subheader("Preview")
    st.markdown("<div class='reader'><p>The quick brown fox jumps over the lazy dog. यह पढ़ने की सेटिंग्स का पूर्वावलोकन है।</p></div>",
                unsafe_allow_html=True)


PAGE_FN = dict(dashboard=page_dashboard, workspace=page_workspace, library=page_library, reader=page_reader,
               search=page_search, ask=page_ask, study=page_study, listen=page_listen, settings=page_settings)


# ============================================================ LAYOUT
def toggle_focus():
    S.focus = not S.focus
    if S.focus:
        S.nav = "reader"


def sidebar():
    with st.sidebar:
        st.markdown(f"<div class='dt-logo'>{LOGO}<div><b>DocuTrust AI</b><small>Read. Verify. Study.</small></div></div>",
                    unsafe_allow_html=True)
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed",
                 format_func=lambda p: f":material/{ICONS[p]}:  {t(p)}")
        if S.docs:
            st.markdown(f"<div class='dt-muted' style='margin:1rem 0 .3rem'>{t('recent')}</div>", unsafe_allow_html=True)
            for d in docs_list()[:3]:
                st.button(d["name"][:28], key=f"side_{d['id']}", on_click=go_reader, args=(d["id"],),
                          icon=":material/description:", use_container_width=True)
        with st.expander(t("pomodoro")):
            pomodoro()
        if not OCR_OK:
            st.caption("OCR engine not detected.")


def topbar():
    c1, c2, c3 = st.columns([6, 1.3, 1.3])
    c1.markdown(f"## {t(S.nav)}")
    with c2:
        if S.focus:
            st.button(t("exit_focus"), key="btn_focus", on_click=toggle_focus, type="primary", icon=":material/close_fullscreen:")
        else:
            st.button(t("focus"), key="btn_focus", on_click=toggle_focus, icon=":material/center_focus_strong:")
    with c3:
        with st.popover(t("shortcuts"), icon=":material/keyboard:"):
            shortcut_table()



# ============================================================ AUTHENTICATION
# Local email/password accounts are stored using salted PBKDF2 hashes.
# Google OAuth uses Streamlit's native OIDC support when configured in secrets.
import base64
import hmac
import sqlite3

AUTH_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docutrust_users.sqlite3")

def _auth_db():
    con = sqlite3.connect(AUTH_DB)
    con.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, name TEXT NOT NULL, salt BLOB NOT NULL, password_hash BLOB NOT NULL, created_at TEXT NOT NULL)")
    con.commit()
    return con

def _password_hash(password, salt=None):
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310000)
    return salt, digest

def _register_user(name, email, password):
    email = email.strip().lower()
    if not name.strip() or "@" not in email or "." not in email.split("@")[-1]:
        return False, "Please enter a valid name and email address."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."
    salt, digest = _password_hash(password)
    try:
        with _auth_db() as con:
            con.execute("INSERT INTO users(email,name,salt,password_hash,created_at) VALUES(?,?,?,?,?)",
                        (email, name.strip(), salt, digest, datetime.now().isoformat(timespec="seconds")))
        return True, "Account created. You can now sign in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists. Sign in instead."

def _verify_user(email, password):
    with _auth_db() as con:
        row = con.execute("SELECT name,salt,password_hash FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    if not row:
        return None
    _, candidate = _password_hash(password, row[1])
    return row[0] if hmac.compare_digest(candidate, row[2]) else None

def _google_oidc_configured():
    try:
        return bool(st.secrets.get("auth", {}).get("redirect_uri"))
    except Exception:
        return False

def auth_gate():
    """Return True when a local or Google-authenticated user is signed in."""
    google_user = getattr(st, "user", None)
    if google_user is not None and getattr(google_user, "is_logged_in", False):
        S["auth_user"] = {"name": getattr(google_user, "name", "Google user"),
                           "email": getattr(google_user, "email", "")}
        return True
    if S.get("auth_user"):
        return True

    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 15% 10%,#183c55 0,transparent 38%),radial-gradient(ellipse at 90% 85%,#30245e 0,transparent 42%),#080d1b}
    [data-testid="stHeader"],[data-testid="stToolbar"]{background:transparent}
    .auth-wrap{max-width:980px;margin:5vh auto 2vh;padding:2.2rem;border:1px solid #34435f;border-radius:28px;background:linear-gradient(145deg,rgba(19,31,56,.96),rgba(13,19,38,.96));box-shadow:0 30px 100px #0008}
    .auth-brand{display:flex;align-items:center;gap:15px;margin-bottom:1.5rem}
    .auth-logo{width:62px;height:62px;border-radius:19px;background:linear-gradient(135deg,#36e2c2,#7187ff);display:grid;place-items:center;color:#071624;font-size:32px;font-weight:900;box-shadow:0 0 35px #36e2c255}
    .auth-title{font-size:2.2rem;font-weight:850;letter-spacing:-1.4px;color:#f4f7ff;margin:0}
    .auth-sub{color:#aab8d5;font-size:1rem;margin-top:3px}
    .auth-tag{display:inline-block;border:1px solid #3b5673;border-radius:99px;padding:6px 12px;color:#70e8d1;font-size:.78rem;letter-spacing:1px}
    </style>
    <div class="auth-wrap"><div class="auth-brand"><div class="auth-logo">D✓</div><div><div class="auth-title">DocuTrust <span style="color:#58e1ca">AI</span></div><div class="auth-sub">Read smarter. Verify everything. Learn faster.</div></div></div><span class="auth-tag">YOUR DOCUMENT INTELLIGENCE WORKSPACE</span></div>
    """, unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("### Your PDFs. Your AI. Your knowledge.")
        st.markdown("Upload documents, ask grounded questions, explore cited answers and turn your files into study material.")
        st.markdown("<div style='padding:18px;border:1px solid #34435f;border-radius:18px;background:#111c33;color:#dbe8ff'>✨ <b>Document-grounded AI</b><br><br>🔎 <b>Page-level source references</b><br><br>📚 <b>Notes, quizzes & flashcards</b><br><br>⌨️ <b>Focused reading shortcuts</b></div>", unsafe_allow_html=True)
    with right:
        st.markdown("### Welcome back")
        mode = st.radio("Account", ["Sign in", "Create account"], horizontal=True, label_visibility="collapsed", key="auth_mode")
        if mode == "Create account":
            with st.form("auth_signup", clear_on_submit=False):
                name = st.text_input("Full name", placeholder="Your name")
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Create password", type="password", help="At least 8 characters")
                confirm = st.text_input("Confirm password", type="password")
                submit = st.form_submit_button("Create my account  →", type="primary", use_container_width=True)
            if submit:
                if password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, message = _register_user(name, email, password)
                    (st.success if ok else st.error)(message)
                    if ok:
                        st.session_state["auth_mode"] = "Sign in"
                        st.rerun()
        else:
            with st.form("auth_signin"):
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Sign in securely  →", type="primary", use_container_width=True)
            if submit:
                name = _verify_user(email, password)
                if name:
                    S["auth_user"] = {"name": name, "email": email.strip().lower(), "provider": "email"}
                    st.rerun()
                st.error("Email or password is incorrect.")
        st.divider()
        if _google_oidc_configured():
            if st.button("🌐  Continue with Google", use_container_width=True):
                st.login("google")
        else:
            st.button("🌐  Continue with Google", disabled=True, use_container_width=True,
                      help="Google OAuth is not configured yet. Add the credentials in .streamlit/secrets.toml.")
            st.caption("Google sign-in needs your Google OAuth client ID and secret before it can work.")
        st.caption("By continuing, you agree to use DocuTrust AI responsibly. Local email accounts are stored on this machine.")
    return False


if auth_gate():
    inject_css()
    sidebar()
    topbar()
    with st.sidebar:
        user = S.get("auth_user", {})
        st.caption(f"Signed in as {user.get('name', 'User')}")
        if st.button("Sign out", key="auth_signout", use_container_width=True):
            S.pop("auth_user", None)
            if getattr(getattr(st, "user", None), "is_logged_in", False):
                st.logout()
            st.rerun()
    try:
        PAGE_FN[S.nav]()
    except Exception as _err:
        st.error(f"Something went wrong on this page: {_err}")
    shortcuts_js()
try:
    PAGE_FN[S.nav]()
except Exception as _err:
    st.error(f"Something went wrong on this page: {_err}")
shortcuts_js()