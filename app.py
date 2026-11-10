"""DocuTrust AI - PDF processing and study workspace (Streamlit).

Run:  streamlit run app.py
Garvit's retrieval system plugs in at the section marked "RETRIEVAL HOOK".

v2: Welcome page (no sign-in wall), AI speed boost (cache + index pre-warm + fast mode),
    20+ background looks (solid, gradient, animated, custom) that restyle the whole app,
    and preferences that persist between runs.
"""
import hashlib
import html
import io
import json
import os
import random
import re
import shutil
import textwrap
from collections import Counter
from datetime import date, datetime, timedelta

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


# ------------------------------------------------------------ AI SPEED BOOST
# 1) identical answer/notes requests are served from a per-session cache,
# 2) the retrieval index is pre-warmed right after PDFs are processed,
# 3) "Fast AI mode" (Settings) trims the generation token budget.
def ai_generate(task, passages, options):
    """Cached wrapper: an identical answer/notes request returns instantly."""
    cache = S.setdefault("_ai_cache", {})
    if task in ("answer", "notes"):
        key = hashlib.md5(json.dumps([task, options, sorted(S.docs), bool(S.get("ai_fast", True))],
                                     sort_keys=True, default=str).encode()).hexdigest()
        if key in cache:
            return cache[key]
        out = _ai_generate_raw(task, passages, options)
        cache[key] = out
        if len(cache) > 60:
            cache.pop(next(iter(cache)))
        return out
    return _ai_generate_raw(task, passages, options)


def warm_ai_index():
    """Build the retrieval index now so the first question is fast."""
    if RETRIEVAL_CONNECTED and S.get("docs"):
        try:
            _get_qa_pipeline()
        except Exception:
            pass


def _ai_generate_raw(task, passages, options):
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
            "Write the notes in " + str(options.get("language", "English")) + ". "
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
    fast = bool(S.get("ai_fast", True))
    n_items = int(options.get("n", 5))
    if task == "notes":
        budget = 600 if fast else 900
    else:
        budget = min(1200, (110 if fast else 150) * n_items + 150)
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
    ], num_predict=budget)

    if task == "notes":
        return raw

    return _parse_json_result(raw)


# ============================================================ THEMES / BACKGROUNDS / TEXT
THEMES = {
    "Midnight Dark": dict(bg="#0a0f24", bg2="#1a1f52", surface="#131a35", surface2="#1d2850", text="#eef1ff", muted="#a6b2d8", accent="#3de0c8", accent2="#7c8cff", warn="#f0c36a", on="#03201a", border="#2d3b6e", side="#0c1230", dark=1),
    "Warm Dark": dict(bg="#1a1310", bg2="#3a2316", surface="#2a201a", surface2="#3b2c22", text="#f8eee3", muted="#c8b49f", accent="#f2b24f", accent2="#e58f6b", warn="#f0c36a", on="#2a1b04", border="#56412f", side="#150f0c", dark=1),
    "Dim Study": dict(bg="#1f2430", bg2="#2f3860", surface="#2a3140", surface2="#364058", text="#e7ebf3", muted="#a9b3c6", accent="#8db4ff", accent2="#b79bff", warn="#f0c36a", on="#0b1a38", border="#46526d", side="#1a1f2a", dark=1),
    "Sepia Reading": dict(bg="#f1e4cc", bg2="#fbf1da", surface="#fcf6e8", surface2="#eddcbc", text="#33281b", muted="#6a5841", accent="#7f4a86", accent2="#b8863b", warn="#8a5a0b", on="#ffffff", border="#d3be96", side="#ebd9b6", dark=0),
    "Clean Light": dict(bg="#f2f6fc", bg2="#dfe9fb", surface="#ffffff", surface2="#eaf0f9", text="#14202f", muted="#4f617a", accent="#0e7c86", accent2="#4f6bed", warn="#8a5a0b", on="#ffffff", border="#d0dcee", side="#ffffff", dark=0),
    "Forest Focus": dict(bg="#0c1914", bg2="#17382a", surface="#14261d", surface2="#1e3b2c", text="#e7f4ec", muted="#9cbdaa", accent="#6fd4a0", accent2="#5fb3c9", warn="#f0c36a", on="#052016", border="#2e5040", side="#0a1510", dark=1),
}

# Background looks. They repaint the WHOLE app (page, sidebar, cards, inputs, reader, popups).
# c = gradient colours (first two drive the page/surfaces), dark = dark palette, anim = animated gradient.
BG_OPTIONS = {
    "Theme default": None,
    "Deep Space": dict(c=["#05070f", "#141a3a"], dark=True),
    "AMOLED Black": dict(c=["#000000", "#0d0d14"], dark=True),
    "Ocean Depth": dict(c=["#04141f", "#0b3a52"], dark=True),
    "Sunset Plum": dict(c=["#1a0b1f", "#4a1942"], dark=True),
    "Emerald Night": dict(c=["#04140e", "#0f3d2b"], dark=True),
    "Crimson Noir": dict(c=["#14070a", "#3d0f1a"], dark=True),
    "Midnight Purple": dict(c=["#0f0720", "#2e1065"], dark=True),
    "Graphite": dict(c=["#17191c", "#2a2e33"], dark=True),
    "Cyber Teal": dict(c=["#021a1d", "#0a4a52"], dark=True),
    "Paper White": dict(c=["#fafafa", "#e9eef7"], dark=False),
    "Soft Cream": dict(c=["#fbf5e6", "#f0e2c0"], dark=False),
    "Sky Light": dict(c=["#eef6ff", "#cfe3ff"], dark=False),
    "Mint Fresh": dict(c=["#effaf4", "#cdeedd"], dark=False),
    "Rose Blush": dict(c=["#fff1f4", "#ffd6e0"], dark=False),
    "Lavender Mist": dict(c=["#f5f0ff", "#ddd0ff"], dark=False),
    "Peach Glow": dict(c=["#fff4ec", "#ffd9bf"], dark=False),
    "Aurora (animated)": dict(c=["#06121f", "#1b0f3a", "#0b3a52", "#2a0f3a"], dark=True, anim=True),
    "Sunset Flow (animated)": dict(c=["#2a0a1a", "#5c1a3a", "#7a2e1a", "#2a0a3a"], dark=True, anim=True),
    "Ocean Wave (animated)": dict(c=["#03111c", "#0a3d62", "#06283d", "#0b5563"], dark=True, anim=True),
    "Neon Pulse (animated)": dict(c=["#07040f", "#1a0b3d", "#0a2a3d", "#2a0b3d"], dark=True, anim=True),
    "Candy Sky (animated)": dict(c=["#fff1f8", "#e3f0ff", "#f3e8ff", "#e6fff5"], dark=False, anim=True),
    "Custom": None,
}
BG_NAMES = list(BG_OPTIONS)


def _is_light(hexcolor):
    h = str(hexcolor).lstrip("#")
    try:
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return False
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.6


def bg_spec():
    ch = S.get("bg_choice", "Theme default")
    if ch == "Custom":
        c1, c2 = S.get("bg_c1", "#0a0f24"), S.get("bg_c2", "#2a1b5e")
        anim = bool(S.get("bg_anim"))
        return dict(c=[c1, c2, c2, c1] if anim else [c1, c2], dark=not _is_light(c1), anim=anim)
    return BG_OPTIONS.get(ch)


def active_vars():
    """Final colour variables = chosen theme + background look (if any)."""
    th = dict(THEMES[S.theme])
    spec = bg_spec()
    if spec:
        b1, b2 = spec["c"][0], spec["c"][1]
        if bool(spec["dark"]) != bool(th["dark"]):  # keep accents readable on the new background
            ref = THEMES["Midnight Dark"] if spec["dark"] else THEMES["Clean Light"]
            th.update({k: ref[k] for k in ("accent", "accent2", "warn", "on")})
        th.update(bg=b1, bg2=b2, dark=int(bool(spec["dark"])))
        if spec["dark"]:
            th.update(surface=f"color-mix(in srgb,{b1} 84%,#ffffff)", surface2=f"color-mix(in srgb,{b1} 70%,#ffffff)",
                      border=f"color-mix(in srgb,{b1} 56%,#ffffff)", side=f"color-mix(in srgb,{b1} 90%,#000000)",
                      text="#eef1ff", muted="#aab4d2")
        else:
            th.update(surface=f"color-mix(in srgb,{b1} 25%,#ffffff)", surface2=f"color-mix(in srgb,{b2} 60%,#ffffff)",
                      border=f"color-mix(in srgb,{b2} 80%,#8d9bb5)", side=f"color-mix(in srgb,{b2} 55%,#ffffff)",
                      text="#14202f", muted="#4f617a")
    return th


def theme_vars():
    return ";".join(f"--{k}:{x}" for k, x in active_vars().items() if k != "dark")


FONTS = {"Source Serif 4": "'Source Serif 4',Georgia,serif", "Figtree": "'Figtree',system-ui,sans-serif",
         "Atkinson Hyperlegible": "'Atkinson Hyperlegible',Verdana,sans-serif",
         "Noto Sans Devanagari": "'Noto Sans Devanagari','Figtree',sans-serif",
         "JetBrains Mono": "'JetBrains Mono',Consolas,monospace"}

# Broad language menu for user-selected learning output. Actual translation quality
# depends on the connected local LLM and the language's model support.
TRANSLATION_LANGUAGES = [
    "English", "Hindi", "Hinglish", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati",
    "Kannada", "Malayalam", "Punjabi", "Urdu", "Odia", "Assamese", "Sanskrit", "Nepali",
    "Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Russian", "Ukrainian",
    "Polish", "Czech", "Slovak", "Romanian", "Hungarian", "Greek", "Turkish", "Arabic",
    "Hebrew", "Persian", "Pashto", "Chinese (Simplified)", "Chinese (Traditional)", "Japanese",
    "Korean", "Thai", "Vietnamese", "Indonesian", "Malay", "Filipino", "Burmese", "Khmer",
    "Lao", "Mongolian", "Kazakh", "Uzbek", "Swahili", "Amharic", "Somali", "Yoruba",
    "Igbo", "Hausa", "Zulu", "Afrikaans", "Xhosa", "Irish", "Welsh", "Scottish Gaelic",
    "Icelandic", "Norwegian", "Swedish", "Danish", "Finnish", "Estonian", "Latvian",
    "Lithuanian", "Bulgarian", "Serbian", "Croatian", "Bosnian", "Slovenian", "Albanian",
    "Macedonian", "Armenian", "Georgian", "Azerbaijani", "Kyrgyz", "Tajik", "Turkmen",
    "Kurdish", "Sindhi", "Kashmiri", "Dogri", "Maithili", "Konkani", "Manipuri", "Bodo",
    "Santali", "Tibetan", "Uyghur", "Javanese", "Sundanese", "Catalan", "Basque", "Galician",
    "Esperanto", "Latin", "Haitian Creole", "Luxembourgish", "Maltese", "Belarusian"
]
PAGES = ["dashboard", "workspace", "library", "reader", "search", "ask", "study", "listen",
         "learning_studio", "insights", "notebook", "planner", "settings"]
ICONS = dict(welcome="home", dashboard="dashboard", workspace="upload_file", library="library_books", reader="menu_book",
             search="search", ask="forum", study="school", listen="headphones",
             learning_studio="auto_awesome", insights="insights", notebook="bookmarks", planner="event_available", settings="tune")

EN = dict(
    dashboard="Dashboard", workspace="PDF Workspace", library="Library", reader="Reader", search="Smart Search",
    ask="Ask AI", study="Study Tools", listen="Listen", learning_studio="Learning Studio", settings="Settings",
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
    ask="एआई से पूछें", study="अध्ययन टूल्स", listen="सुनें", learning_studio="लर्निंग स्टूडियो",
    settings="सेटिंग्स", docs="दस्तावेज़",
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


EN.update(welcome="Welcome", insights="Insights", notebook="Notebook", planner="Study Planner")
HI.update(welcome="स्वागत", insights="इनसाइट्स", notebook="नोटबुक", planner="स्टडी प्लानर")
EN.update(gallery="Theme gallery", use="Use", active="Active", greet_m="Good morning", greet_a="Good afternoon",
          greet_e="Good evening", hero_sub="{n} documents ready · {p} pages searchable")
HI.update(gallery="थीम गैलरी", use="उपयोग करें", active="चालू", greet_m="सुप्रभात", greet_a="नमस्कार",
          greet_e="शुभ संध्या", hero_sub="{n} दस्तावेज़ तैयार · {p} पेज खोजने योग्य")


def t(k):
    return (HI if S.get("lang") == "हिन्दी" else EN).get(k) or EN.get(k, k)


DEFAULTS = dict(theme="Midnight Dark", lang="English", content_language="English", font_size=17, font_family="Source Serif 4", line_height=1.7,
                read_width=720, comfy=False, focus=False, docs={}, nav="dashboard", chat=[], quiz=None, fc=None,
                fc_idx=0, fc_status={}, notes_out=None, ocr_on=True, ocr_lang="eng", dpi=150, chunk_size=900,
                overlap=120, rd_doc=None, rd_page=1, last_run=[],
                bg_choice="Theme default", bg_c1="#0a0f24", bg_c2="#2a1b5e", bg_anim=False,
                ai_fast=True, user_name="", bookmarks=[], notes={}, tasks=[],
                planner=dict(goal=60, exam_on=False, exam_name="", exam="", log={}))
for _k, _v in DEFAULTS.items():
    S.setdefault(_k, _v)

# ------------------------------------------------------------ PERSISTENT PREFERENCES
PREFS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docutrust_prefs.json")
PREF_KEYS = ["theme", "lang", "content_language", "font_size", "font_family", "line_height", "read_width", "comfy",
             "bg_choice", "bg_c1", "bg_c2", "bg_anim", "ai_fast"]


def load_prefs():
    if S.get("_prefs_loaded"):
        return
    S["_prefs_loaded"] = True
    try:
        with open(PREFS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        for k in PREF_KEYS:
            if k in data:
                S[k] = data[k]
        if S.theme not in THEMES:
            S.theme = "Midnight Dark"
        if S.bg_choice not in BG_OPTIONS:
            S.bg_choice = "Theme default"
    except Exception:
        pass
    S["_prefs_saved"] = json.dumps({k: S.get(k) for k in PREF_KEYS}, sort_keys=True, default=str)


def save_prefs():
    snap = json.dumps({k: S.get(k) for k in PREF_KEYS}, sort_keys=True, default=str)
    if snap == S.get("_prefs_saved"):
        return
    try:
        with open(PREFS_FILE, "w", encoding="utf-8") as f:
            f.write(snap)
        S["_prefs_saved"] = snap
    except Exception:
        pass


load_prefs()


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


def sel(label, options, key, current, setter, **kw):
    """Always-in-sync selectbox: the stored choice wins, the widget just edits it."""
    S[key] = current
    return st.selectbox(label, options, key=key, on_change=lambda: setter(S[key]), **kw)


def set_theme(name):
    S.theme = name
    S["w_theme"] = name


def set_bg(name):
    S.bg_choice = name
    for k in ("w_bg_choice", "tb_bg", "wl_bg", "st_bg"):
        S[k] = name


def custom_bg_controls(prefix):
    """Two colour pickers + animate toggle for the Custom background."""
    c1, c2, c3 = st.columns([1, 1, 1])
    S[prefix + "_c1"], S[prefix + "_c2"], S[prefix + "_an"] = S.bg_c1, S.bg_c2, S.bg_anim
    with c1:
        st.color_picker("Colour 1", key=prefix + "_c1",
                        on_change=lambda: (S.__setitem__("bg_c1", S[prefix + "_c1"]), set_bg("Custom")))
    with c2:
        st.color_picker("Colour 2", key=prefix + "_c2",
                        on_change=lambda: (S.__setitem__("bg_c2", S[prefix + "_c2"]), set_bg("Custom")))
    with c3:
        st.write("")
        st.toggle("Animate", key=prefix + "_an",
                  on_change=lambda: (S.__setitem__("bg_anim", S[prefix + "_an"]), set_bg("Custom")))


# ============================================================ STYLE
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=Figtree:wght@400;500;600&family=Source+Serif+4:wght@400;600&family=Noto+Sans+Devanagari:wght@400;600&family=JetBrains+Mono&family=Atkinson+Hyperlegible:wght@400;700&display=swap');
html,body{background:var(--bg)!important}
.stApp{background:radial-gradient(1200px 620px at 88% -8%,color-mix(in srgb,var(--bg2) 85%,transparent),transparent 62%),radial-gradient(900px 500px at -5% 105%,color-mix(in srgb,var(--accent2) 10%,transparent),transparent 60%),var(--bg)!important;font-family:'Figtree','Noto Sans Devanagari',system-ui,sans-serif}
[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stBottom"],[data-testid="stBottomBlockContainer"]{background:transparent!important}
.stApp,.stApp p,.stApp li,.stApp label,[data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"] p{color:var(--text)}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:'Bricolage Grotesque','Noto Sans Devanagari',sans-serif;color:var(--text);letter-spacing:-.01em}
[data-testid="stCaptionContainer"],.stApp small{color:var(--muted)!important}
[data-testid="stHeader"]{background:transparent!important}
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
[data-testid="stPopoverBody"]{background:var(--surface)!important;border:1px solid var(--border)!important}
.stButton>button,.stDownloadButton>button,[data-testid^="stBaseButton"]{background:var(--surface2);color:var(--text);border:1px solid var(--border);border-radius:10px;transition:border-color .15s,transform .05s}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--accent);color:var(--text);background:color-mix(in srgb,var(--accent) 12%,var(--surface2));box-shadow:0 6px 18px -10px var(--accent)}
.stButton>button:active{transform:translateY(1px)}
[data-testid="stPopover"] > button{background:var(--surface2)!important;color:var(--text)!important;border:1px solid var(--accent)!important;border-radius:10px!important;box-shadow:0 4px 14px -8px var(--accent)!important}
[data-testid="stPopover"] > button:hover{background:color-mix(in srgb,var(--accent) 16%,var(--surface2))!important;border-color:var(--accent)!important}
[data-testid="stPopover"] > button p,[data-testid="stPopover"] > button span,[data-testid="stPopover"] > button svg{color:var(--text)!important;fill:var(--text)!important}
.stButton>button[kind="primary"],[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,var(--accent),color-mix(in srgb,var(--accent) 62%,var(--accent2)))!important;color:var(--on)!important;border-color:var(--accent)!important;font-weight:600;box-shadow:0 8px 20px -12px var(--accent)}
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
[data-testid="stChatInput"]>div,[data-testid="stBottom"]>div{background:transparent!important}
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
    th = active_vars()
    v = ";".join(f"--{k}:{x}" for k, x in th.items() if k != "dark")
    fam = FONTS[S.font_family]
    width = min(S.read_width, 680) if S.comfy else S.read_width
    lh = max(S.line_height, 1.85) if S.comfy else S.line_height
    pad = "2.2rem 2.6rem" if S.comfy else "1.4rem 1.8rem"
    focus = ('[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="stToolbar"]{display:none!important}'
             if S.focus else "")
    anim = ""
    spec = bg_spec()
    if spec and spec.get("anim"):
        grad = ",".join(spec["c"])
        anim = (f"html body .stApp{{background:linear-gradient(-45deg,{grad})!important;background-size:400% 400%!important;"
                "animation:dtflow 18s ease infinite!important}"
                "@keyframes dtflow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}")
    st.markdown(f"<style>:root{{{v};color-scheme:{'dark' if th['dark'] else 'light'}}}{CSS}{anim}"
                f".reader{{font-family:{fam};font-size:{S.font_size}px;line-height:{lh};max-width:{width}px;padding:{pad};"
                f"{'letter-spacing:.012em;' if S.comfy else ''}}}{focus}</style>", unsafe_allow_html=True)


# ============================================================ PDF PROCESSING
def split_sentences(text):
    t_ = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", t_) if s.strip()]


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
    sel_ = pw(st.multiselect, t("select_docs"), key + "_docs", ids, options=ids, format_func=lambda i: S.docs[i]["name"])
    if len(sel_) == 1 and S.docs[sel_[0]]["page_count"] > 1:
        n = S.docs[sel_[0]]["page_count"]
        cur = S.get(key + "_rng", (1, n))
        S[key + "_rng"] = (max(1, min(cur[0], n)), max(1, min(cur[1], n)))
        if "w_" + key + "_rng" in S:
            S["w_" + key + "_rng"] = S[key + "_rng"]
        lo, hi = pw(st.slider, t("pages_range"), key + "_rng", (1, n), min_value=1, max_value=n)
    else:
        lo, hi = 1, 10 ** 6
    return [(S.docs[i], lo, hi) for i in sel_]


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


SHORTCUT_GUIDE = [("Alt + 1 … 9", "Go to Dashboard, Workspace, Library, Reader, Search, Ask AI, Study, Listen, Learning Studio (more pages: use the sidebar)"),
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
    who = S.get("user_name", "")
    if who:
        greet += f", {esc(who)}"
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
            pw(st.slider, "OCR render DPI", "dpi", 150, min_value=100, max_value=300, step=50)
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
        with st.spinner("Warming up the AI index (makes your first question fast)..."):
            warm_ai_index()
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


def translate_learning_text(text, target_language):
    """Translate user-selected learning text through the existing local QA model."""
    text = str(text or "").strip()
    if not text:
        raise ValueError("There is no readable text to translate on this page.")
    if target_language == "Hinglish":
        style = "natural conversational Hinglish (Hindi written in Roman script, retaining useful English technical terms)"
    else:
        style = target_language
    qa = _get_qa_pipeline()
    result = qa._ollama_chat([
        {"role": "system", "content": (
            "You are a careful document translator. Translate faithfully into the requested language. "
            "Preserve headings, lists, names, numbers, formulas, code and technical terms. "
            "Do not summarize, add facts, or omit content. Return only the translation."
        )},
        {"role": "user", "content": f"Target language: {style}\n\nTEXT TO TRANSLATE:\n{text}"},
    ], num_predict=min(1800, max(400, len(text) // 2)))
    return str(result).strip()


def _page_reader_core():
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
    original_text = p.get("text", "")
    target_lang = pw(st.selectbox, "Learning / translation language", "rd_target_language",
                     S.get("content_language", "English"), options=TRANSLATION_LANGUAGES)
    view_mode = st.radio("Reader view", ["Original", "Translated", "Side by side"],
                         horizontal=True, key="rd_translation_view")
    if view_mode == "Original":
        body = "".join(f"<p>{esc(x.strip()).replace(chr(10), ' ')}</p>" for x in re.split(r"\n\s*\n", original_text) if x.strip())
        st.markdown(f"<div class='reader'>{body or '<p class=dt-muted>No text on this page (scanned page? enable OCR in PDF Workspace).</p>'}</div>", unsafe_allow_html=True)
    else:
        cache_key = f"rd_translation_{S.rd_doc}_{S.rd_page}_{target_lang}"
        if st.button(f"Translate this page to {target_lang}", key="rd_translate_page", type="primary"):
            if cache_key in S and S[cache_key]:
                pass  # instant: already translated
            else:
                try:
                    with st.spinner(f"Translating page into {target_lang}..."):
                        S[cache_key] = translate_learning_text(original_text, target_lang)
                except Exception as exc:
                    st.error(f"Translation failed: {exc}")
        translated = S.get(cache_key, "")
        if not translated:
            st.info("Select a language and click Translate this page. Translation uses the configured local AI model.")
        elif view_mode == "Translated":
            st.markdown(f"<div class='reader'>{''.join(f'<p>{esc(x)}</p>' for x in translated.splitlines() if x.strip())}</div>", unsafe_allow_html=True)
        else:
            left, right = st.columns(2)
            with left:
                st.markdown("**Original**")
                st.markdown(f"<div class='reader'>{''.join(f'<p>{esc(x.strip())}</p>' for x in original_text.splitlines() if x.strip())}</div>", unsafe_allow_html=True)
            with right:
                st.markdown(f"**{target_lang}**")
                st.markdown(f"<div class='reader'>{''.join(f'<p>{esc(x)}</p>' for x in translated.splitlines() if x.strip())}</div>", unsafe_allow_html=True)


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
        sel_ = pw(st.multiselect, t("select_docs"), "sq_docs", ids, options=ids, format_func=lambda i: S.docs[i]["name"])
    with c2:
        mode = pw(st.radio, "Match", "sq_mode", "Any keyword", options=["Any keyword", "All keywords", "Exact phrase"], horizontal=True)
    terms = [q.lower().strip()] if mode == "Exact phrase" else [w for w in re.findall(r"\w+", q.lower()) if len(w) > 1]
    if not q.strip() or not terms:
        st.info("Type a keyword or phrase to search all selected documents.")
        return
    hits = []
    for i in sel_:
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
    if S.docs and not S.chat:
        st.markdown("#### Try asking")
        c1, c2, c3 = st.columns(3)
        for col, label, question in [(c1, "Summarize documents", "Summarize the main points of my uploaded documents."),
                                     (c2, "Key findings", "What are the most important findings in these documents?"),
                                     (c3, "Important dates", "List the important dates and deadlines mentioned in my documents.")]:
            with col:
                if st.button(label, key="ask_suggest_" + label):
                    S["pending_question"] = question
                    st.rerun()
    if S.chat:
        st.button(t("clear"), key="chat_clear", on_click=lambda: S.__setitem__("chat", []), icon=":material/delete:")
    for mi, m in enumerate(S.chat):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for pi, ps in enumerate(m.get("passages", [])):
                with st.expander(f"{ps['doc']} · {t('page')} {ps['page']}", expanded=pi == 0):
                    st.code(ps["text"], language=None, wrap_lines=True)
    q = st.chat_input(t("ask_ph"), key="chat_in", disabled=not S.docs)
    q = S.pop("pending_question", None) or q
    if not S.docs:
        st.info(t("no_docs"))
    if q:
        S.chat.append(dict(role="user", content=q))
        ps = keyword_retrieve(q)
        try:
            with st.spinner("Searching your documents and generating an answer..."):
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
        lang = pw(st.selectbox, "Speech language", "ls_lang", "en-US", options=["en-US", "hi-IN", "en-IN", "bn-IN", "ta-IN", "te-IN", "mr-IN", "gu-IN", "kn-IN", "ml-IN", "pa-IN", "ur-IN", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-PT", "ru-RU", "ja-JP", "ko-KR", "zh-CN", "ar-SA"])
    rate = pw(st.select_slider, t("speed"), "ls_rate", 1.0, options=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    pg = max(1, min(int(pg), d["page_count"]))
    full = d["pages"][pg - 1]["text"]
    S[f"ls_text_{did}_{pg}"] = S.get(f"ls_text_{did}_{pg}", full)
    st.caption("Edit the box to read only the part you want, or leave it to read the whole page.")
    txt = st.text_area("Text to read", key=f"ls_text_{did}_{pg}", height=220)
    tts(txt, lang, rate)
    st.caption("Speech uses your browser's built-in voices. Hindi works if a Hindi voice is installed in your OS/browser.")


def bg_gallery():
    """Click-to-apply gallery of every background look."""
    names = [n for n in BG_NAMES if n not in ("Theme default", "Custom")]
    cols = st.columns(4)
    for i, name in enumerate(names):
        spec = BG_OPTIONS[name]
        grad = ",".join(spec["c"])
        anim = " ✨" if spec.get("anim") else ""
        with cols[i % 4]:
            st.markdown(f"<div style='height:54px;border-radius:12px;border:2px solid "
                        f"{'var(--accent)' if S.bg_choice == name else 'var(--border)'};"
                        f"background:linear-gradient(135deg,{grad})'></div>", unsafe_allow_html=True)
            st.button((name.replace(' (animated)', '') + anim) if S.bg_choice != name else "✓ " + name.replace(' (animated)', ''),
                      key=f"bgg_{i}", on_click=set_bg, args=(name,), use_container_width=True,
                      disabled=S.bg_choice == name)


def page_settings():
    a, b = st.columns(2)
    with a:
        settings_account()
        st.subheader(t("appearance"))
        pw(st.selectbox, t("theme"), "theme", "Midnight Dark", options=list(THEMES))
        sel("Background colour", BG_NAMES, "st_bg", S.bg_choice, set_bg)
        if S.bg_choice == "Custom":
            custom_bg_controls("st")
        pw(st.selectbox, "Website interface language", "lang", "English", options=["English", "हिन्दी"])
        pw(st.selectbox, "Default learning / translation language", "content_language", "English", options=TRANSLATION_LANGUAGES)
        st.caption("This preference is used by Reader and learning tools. Actual language quality depends on the configured AI model.")
        st.subheader("AI speed")
        S["st_fast"] = S.ai_fast
        st.toggle("Fast AI mode (shorter, quicker generations)", key="st_fast",
                  on_change=lambda: S.__setitem__("ai_fast", S["st_fast"]))
        st.caption("Answers and notes are also cached, so repeating the same request is instant.")
        st.button("Clear AI cache", key="clr_cache", on_click=lambda: S.__setitem__("_ai_cache", {}))
        st.subheader(t("shortcuts"))
        shortcut_table()
    with b:
        st.subheader(t("reading"))
        pw(st.slider, t("font_size"), "font_size", 17, min_value=12, max_value=28)
        pw(st.selectbox, t("font_family"), "font_family", "Source Serif 4", options=list(FONTS))
        pw(st.slider, t("line_spacing"), "line_height", 1.7, min_value=1.2, max_value=2.4, step=0.1)
        pw(st.slider, t("read_width"), "read_width", 720, min_value=480, max_value=1100, step=20)
        pw(st.toggle, t("comfy"), "comfy", False)
    st.subheader("Background gallery")
    bg_gallery()
    st.button("Reset background to theme default", key="bg_reset", on_click=set_bg, args=("Theme default",))
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


def _learning_notes_pdf(title, body):
    """Create a simple text-based PDF using the existing PyMuPDF dependency."""
    if pymupdf is None:
        raise RuntimeError("PDF export requires PyMuPDF. Install it with: pip install pymupdf")
    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842)
    y = 52
    page.insert_text((48, y), title[:90], fontsize=17, fontname="helv")
    y += 32
    for raw_line in str(body).splitlines():
        line = raw_line.strip()
        if not line:
            y += 9
            continue
        wrapped = textwrap.wrap(line, width=88) or [""]
        for part in wrapped:
            if y > 790:
                page = pdf.new_page(width=595, height=842)
                y = 52
            page.insert_text((48, y), part[:110], fontsize=9.5, fontname="helv")
            y += 15
    data = pdf.tobytes()
    pdf.close()
    return data


def _video_transcribe(video_bytes, suffix):
    """Transcribe a video with the optional open-source Whisper package."""
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Video transcription is not installed yet. In your project terminal run: "
            "pip install openai-whisper. FFmpeg must also be installed and available on PATH."
        ) from exc

    import tempfile
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(video_bytes)
            temp_path = temp.name
        model_name = S.get("ls_whisper_model", "base")
        model = whisper.load_model(model_name)
        result = model.transcribe(temp_path, fp16=False)
        return result.get("text", "").strip(), result.get("segments", [])
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _pdf_video_storyboard(doc, max_scenes=7):
    """Build a conservative, source-grounded storyboard from extracted PDF pages."""
    scenes = []
    for page_no, page in enumerate(doc.get("pages", []), start=1):
        raw = re.sub(r"\s+", " ", str(page.get("text", ""))).strip()
        if len(raw) < 80:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", raw)
        useful = [x.strip() for x in sentences if len(x.strip()) > 25]
        if not useful:
            useful = [raw]
        title = (useful[0][:72].strip(" .:-") or f"Key idea · Page {page_no}")
        bullets = useful[:3]
        narration = f"Page {page_no}. {title}. " + " ".join(bullets[:2])
        scenes.append({"title": title, "bullets": bullets, "narration": narration, "page": page_no})
        if len(scenes) >= max_scenes:
            break
    return scenes


def _pdf_video_slide(scene, index, total, animated=False, size=(1280, 720)):
    """Render a readable 16:9 slide; animated mode adds a distinct visual motif."""
    from PIL import ImageFont
    import textwrap as _tw
    im = Image.new("RGB", size, (10, 16, 38))
    draw = ImageDraw.Draw(im)
    try:
        title_font = ImageFont.truetype("segoeui.ttf", 42)
        body_font = ImageFont.truetype("segoeui.ttf", 27)
        small_font = ImageFont.truetype("segoeui.ttf", 20)
    except Exception:
        title_font, body_font, small_font = ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default()
    draw.rounded_rectangle((42, 38, 1238, 682), radius=28, fill=(19, 29, 62), outline=(61, 224, 200), width=3)
    draw.text((82, 68), "DOCUTRUST AI  ·  PDF VIDEO LESSON", font=small_font, fill=(61, 224, 200))
    draw.text((82, 120), f"{index + 1:02d} / {total:02d}", font=small_font, fill=(166, 178, 216))
    title_lines = _tw.wrap(scene["title"], width=42)[:2]
    y = 180
    for line in title_lines:
        draw.text((82, y), line, font=title_font, fill=(245, 248, 255))
        y += 58
    y += 24
    for bullet in scene["bullets"][:3]:
        wrapped = _tw.wrap(bullet, width=76)[:3]
        draw.ellipse((88, y + 8, 102, y + 22), fill=(61, 224, 200))
        for line in wrapped:
            draw.text((124, y), line, font=body_font, fill=(220, 229, 250))
            y += 39
        y += 20
    if animated:
        draw.rounded_rectangle((82, 590, 1195, 640), radius=18, fill=(35, 67, 119))
        draw.text((105, 601), "CONCEPT  →  EXPLANATION  →  REVIEW", font=small_font, fill=(255, 255, 255))
    else:
        draw.text((82, 610), f"Source: uploaded PDF · Page {scene['page']}", font=small_font, fill=(166, 178, 216))
    return im


def _render_pdf_video(scenes, voice_language="English", animated=False):
    # Render a narrated MP4 using cloud-compatible Edge TTS voices.
    import asyncio
    import uuid
    import edge_tts

    try:
        from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
        moviepy_v2 = True
    except ImportError:
        from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
        moviepy_v2 = False

    if not scenes:
        raise RuntimeError("No readable PDF text was found to build a video lesson.")

    prefixes_by_language = {
        "english": ("en-",), "hindi": ("hi-",), "hinglish": ("hi-",),
        "bengali": ("bn-",), "tamil": ("ta-",), "telugu": ("te-",),
        "marathi": ("mr-",), "gujarati": ("gu-",), "kannada": ("kn-",),
        "malayalam": ("ml-",), "punjabi": ("pa-",), "urdu": ("ur-",),
        "spanish": ("es-",), "french": ("fr-",), "german": ("de-",),
        "italian": ("it-",), "portuguese": ("pt-",), "dutch": ("nl-",),
        "russian": ("ru-",), "ukrainian": ("uk-",), "polish": ("pl-",),
        "czech": ("cs-",), "slovak": ("sk-",), "romanian": ("ro-",),
        "hungarian": ("hu-",), "greek": ("el-",), "turkish": ("tr-",),
        "arabic": ("ar-",), "hebrew": ("he-",), "persian": ("fa-",),
        "chinese (simplified)": ("zh-CN", "zh-SG"),
        "chinese (traditional)": ("zh-TW", "zh-HK"),
        "japanese": ("ja-",), "korean": ("ko-",), "thai": ("th-",),
        "vietnamese": ("vi-",), "indonesian": ("id-",), "malay": ("ms-",),
        "filipino": ("fil-",), "swahili": ("sw-",), "afrikaans": ("af-",),
        "irish": ("ga-",), "welsh": ("cy-",), "norwegian": ("nb-", "nn-"),
        "swedish": ("sv-",), "danish": ("da-",), "finnish": ("fi-",),
        "estonian": ("et-",), "latvian": ("lv-",), "lithuanian": ("lt-",),
        "bulgarian": ("bg-",), "serbian": ("sr-",), "croatian": ("hr-",),
        "bosnian": ("bs-",), "slovenian": ("sl-",), "albanian": ("sq-",),
        "macedonian": ("mk-",), "armenian": ("hy-",), "georgian": ("ka-",),
        "azerbaijani": ("az-",), "kazakh": ("kk-",), "uzbek": ("uz-",),
        "yoruba": ("yo-",), "igbo": ("ig-",), "hausa": ("ha-",),
        "zulu": ("zu-",), "xhosa": ("xh-",), "catalan": ("ca-",),
        "basque": ("eu-",), "galician": ("gl-",), "haitian creole": ("ht-",),
        "maltese": ("mt-",), "belarusian": ("be-",),
    }

    output_dir = os.path.join(os.getcwd(), "generated_videos", uuid.uuid4().hex[:10])
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "narration.mp3")
    mp4_path = os.path.join(output_dir, "lesson.mp4")
    audio = video = None
    clips = []

    try:
        script = "\n\n".join(str(scene.get("narration", "")).strip() for scene in scenes)
        if not script.strip():
            raise RuntimeError("The PDF storyboard contains no narration text.")

        requested = str(voice_language or "English").strip().lower()
        prefixes = prefixes_by_language.get(requested)
        if not prefixes:
            raise RuntimeError(
                f"No voice mapping is configured for {voice_language}. Choose another language."
            )

        voices = asyncio.run(edge_tts.list_voices())
        matching = [
            v for v in voices
            if any(str(v.get("Locale", "")).lower().startswith(p.lower()) for p in prefixes)
        ]
        if not matching:
            raise RuntimeError(
                f"No online narration voice is currently available for {voice_language}. "
                "Please choose another language."
            )
        selected = next(
            (v for v in matching if "Neural" in str(v.get("Name", ""))),
            matching[0],
        )
        asyncio.run(edge_tts.Communicate(
            text=script, voice=selected["ShortName"], rate="+0%", volume="+0%"
        ).save(audio_path))

        if not os.path.isfile(audio_path) or os.path.getsize(audio_path) < 1000:
            raise RuntimeError("Edge TTS did not create a valid narration audio file.")

        audio = AudioFileClip(audio_path)
        per_scene = max(0.5, audio.duration / len(scenes))
        for i, scene in enumerate(scenes):
            image = _pdf_video_slide(scene, i, len(scenes), animated=animated)
            image_path = os.path.join(output_dir, f"scene_{i:02d}.png")
            try:
                image.save(image_path)
            finally:
                image.close()

            clip = ImageClip(image_path)
            clip = clip.with_duration(per_scene) if moviepy_v2 else clip.set_duration(per_scene)
            if animated:
                if moviepy_v2:
                    clip = clip.resized(
                        lambda t_: 1.0 + 0.025 * min(1.0, t_ / max(per_scene, 0.1))
                    ).with_position("center")
                else:
                    clip = clip.resize(
                        lambda t_: 1.0 + 0.025 * min(1.0, t_ / max(per_scene, 0.1))
                    ).set_position("center")
            clips.append(clip)

        video = concatenate_videoclips(clips, method="compose")
        video = (
            video.with_audio(audio).with_duration(audio.duration)
            if moviepy_v2 else video.set_audio(audio).set_duration(audio.duration)
        )
        video.write_videofile(
            mp4_path, fps=24, codec="libx264", audio_codec="aac", logger="bar"
        )
        if not os.path.isfile(mp4_path) or os.path.getsize(mp4_path) == 0:
            raise RuntimeError("FFmpeg finished without creating a valid MP4.")
        with open(mp4_path, "rb") as handle:
            return handle.read()
    finally:
        if video is not None:
            try:
                video.close()
            except Exception:
                pass
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass
        if audio is not None:
            try:
                audio.close()
            except Exception:
                pass

def page_learning_studio():
    st.markdown(
        "<div class='dt-hero'><h3>🎓 AI Learning Studio</h3>"
        "<span>Turn your study PDFs and lecture videos into notes and visual learning material.</span></div>",
        unsafe_allow_html=True,
    )

    language = st.selectbox(
        "Explanation language", TRANSLATION_LANGUAGES,
        index=TRANSLATION_LANGUAGES.index(S.get("content_language", "English")) if S.get("content_language", "English") in TRANSLATION_LANGUAGES else 0,
        key="ls_language"
    )
    source_mode = st.radio(
        "Choose your source", ["📄 PDF Learning", "🎬 Video Learning"],
        horizontal=True, key="ls_source_mode",
    )

    if source_mode == "📄 PDF Learning":
        st.subheader("PDF → Concepts → Visual Learning")
        upload = st.file_uploader(
            "Upload a PDF to add it to your DocuTrust library",
            type=["pdf"], key="ls_pdf_upload",
        )
        if upload is not None:
            if pymupdf is None:
                st.error("PyMuPDF is not installed. Run: pip install pymupdf")
            elif st.button("Process PDF for Learning Studio", key="ls_process_pdf", type="primary"):
                cfg = dict(
                    ocr_on=S.ocr_on, ocr_lang=S.ocr_lang, dpi=S.dpi,
                    chunk_size=S.chunk_size, overlap=S.overlap,
                )
                try:
                    data = upload.getvalue()
                    doc = process_pdf(upload.name, data, cfg, lambda i, n: None)
                    doc["cfg"] = cfg
                    S.docs[doc["id"]] = doc
                    S["ls_selected_doc"] = doc["id"]
                    with st.spinner("Warming up the AI index..."):
                        warm_ai_index()
                    st.success(f"Processed {doc['name']} — {doc['page_count']} pages.")
                except Exception as exc:
                    st.error(f"Could not process PDF: {exc}")

        if not S.docs:
            st.info("Upload and process a PDF above, or process one in PDF Workspace first.")
        else:
            ids = list(S.docs)
            selected_id = st.selectbox(
                "Choose a processed document", ids,
                format_func=lambda did: S.docs[did]["name"],
                key="ls_selected_doc",
            )
            doc = S.docs[selected_id]
            st.caption(f"{doc['page_count']} pages · {len(doc['chunks'])} text chunks · {doc['status']}")

            c1, c2 = st.columns([2, 1])
            with c1:
                concept = st.text_input(
                    "Concept to understand (optional)",
                    placeholder="e.g. linked list insertion, deadlock, inheritance",
                    key="ls_pdf_concept",
                )
            with c2:
                exam_mode = st.toggle("Exam-focused notes", value=True, key="ls_exam_mode")

            tab_explain, tab_diagram, tab_animation, tab_notes, tab_video = st.tabs(
                ["🧠 Explanation", "🔷 Diagram", "🎞️ Animation", "📚 Notes PDF", "🎬 PDF → Video"]
            )

            with tab_explain:
                if st.button("Generate explanation from this PDF", key="ls_pdf_explain", type="primary"):
                    try:
                        prompt = concept.strip() or "Explain the main concepts in this document"
                        with st.spinner("Retrieving relevant PDF evidence and generating notes..."):
                            answer = ai_generate(
                                "answer", [], {"question": prompt + f". Explain in {language}."}
                            )
                        S["ls_pdf_answer"] = answer
                    except Exception as exc:
                        st.error(f"Could not generate explanation: {exc}")
                if S.get("ls_pdf_answer"):
                    st.markdown(S["ls_pdf_answer"])

            page_text = "\n".join(p.get("text", "") for p in doc.get("pages", []))
            tokens = [w for w in words(page_text) if w not in STOP and len(w) > 3]
            common = [w for w, _ in Counter(tokens).most_common(6)]

            with tab_diagram:
                st.markdown("**Document-derived keyword map**")
                if common:
                    nodes = "; ".join(f'"{esc(w)}"' for w in common)
                    edges = "\n".join(
                        f'"{esc(common[i])}" -> "{esc(common[i + 1])}";'
                        for i in range(len(common) - 1)
                    )
                    graph = "digraph Concepts { rankdir=LR; node [shape=box, style=rounded]; " + nodes + "; " + edges + " }"
                    st.graphviz_chart(graph)
                    st.caption("This is a keyword relationship preview, not a verified semantic concept map.")
                else:
                    st.info("No readable text was extracted from this PDF. Check OCR settings in PDF Workspace.")

            with tab_animation:
                st.markdown("**Animated concept-flow preview**")
                anim_terms = common[:3] if common else ["Source", "Concept", "Result"]
                labels = [html.escape(x.title()) for x in anim_terms]
                while len(labels) < 3:
                    labels.append(f"Step {len(labels) + 1}")
                components.html(
                    """<style>:root{__TH__}
                    body{margin:0;font-family:system-ui;color:var(--text)}
                    .track{display:flex;align-items:center;justify-content:space-around;gap:8px;
                    padding:24px 10px;background:var(--surface);border:1px solid var(--border);border-radius:14px}
                    .node{padding:14px 16px;border-radius:10px;background:var(--accent);color:var(--on);font-weight:650;
                    text-align:center;min-width:70px;animation:pulse 2.4s ease-in-out infinite}
                    .node:nth-child(3){animation-delay:.45s}.node:nth-child(5){animation-delay:.9s}
                    @keyframes pulse{0%,100%{transform:translateY(0);filter:brightness(.9)}
                    50%{transform:translateY(-8px);filter:brightness(1.25)}}
                    @media(prefers-reduced-motion:reduce){.node{animation:none}}
                    </style><div class="track"><div class="node">__A__</div><span>→</span>
                    <div class="node">__B__</div><span>→</span><div class="node">__C__</div></div>"""
                    .replace("__TH__", theme_vars())
                    .replace("__A__", labels[0]).replace("__B__", labels[1]).replace("__C__", labels[2]),
                    height=120,
                )
                st.caption("Motion preview uses extracted keywords. A full AI-generated step animation is a later integration.")

            with tab_notes:
                if st.button("Generate study notes", key="ls_pdf_notes", type="primary"):
                    try:
                        with st.spinner("Generating notes from retrieved PDF evidence..."):
                            notes = ai_generate("notes", [], {"exam": exam_mode, "language": language})
                        S["ls_pdf_notes_text"] = notes
                    except Exception as exc:
                        st.error(f"Could not generate notes: {exc}")
                if S.get("ls_pdf_notes_text"):
                    st.markdown(S["ls_pdf_notes_text"])
                    try:
                        notes_pdf = _learning_notes_pdf(
                            f"Study Notes - {doc['name']}", S["ls_pdf_notes_text"]
                        )
                        st.download_button(
                            "Download Study Notes PDF", notes_pdf,
                            file_name=f"{os.path.splitext(doc['name'])[0]}_study_notes.pdf",
                            mime="application/pdf", key="ls_pdf_notes_download",
                        )
                    except Exception as exc:
                        st.warning(f"PDF export unavailable: {exc}")

            with tab_video:
                st.markdown("### Create a narrated video lesson")
                st.caption(
                    "Creates a narrated slide-style MP4 from readable text in the selected PDF. "
                    "It is a visual summary, not a cinematic AI-generated video."
                )
                video_voice = st.selectbox(
                    "Narration voice language",
                    TRANSLATION_LANGUAGES,
                    index=(
                        TRANSLATION_LANGUAGES.index(language)
                        if language in TRANSLATION_LANGUAGES else 0
                    ),
                    key="ls_pdf_video_voice",
                    help=(
                        "Choose a narration language. Voice availability depends on "
                        "the speech provider; unsupported languages will be reported."
                    ),
                )
                video_motion = st.checkbox(
                    "Add gentle slide motion", value=True, key="ls_pdf_video_motion"
                )
                scenes = _pdf_video_storyboard(doc)
                if scenes:
                    st.caption(f"{len(scenes)} video scenes prepared from PDF pages.")
                    with st.expander("Preview video storyboard"):
                        for scene in scenes:
                            st.markdown(f"**Scene {scene['page']}: {scene['title']}**")
                            for bullet in scene["bullets"]:
                                st.markdown(f"- {bullet}")
                else:
                    st.warning(
                        "No readable PDF text was found. Check that the selected document "
                        "was processed successfully and OCR is enabled if it is scanned."
                    )

                if st.button(
                    "Generate narrated MP4",
                    key="ls_pdf_video_generate",
                    type="primary",
                    disabled=not bool(scenes),
                ):
                    try:
                        with st.spinner("Creating slides and narration. This can take a few minutes..."):
                            mp4_bytes = _render_pdf_video(
                                scenes, voice_language=video_voice, animated=video_motion
                            )
                        S["ls_pdf_video_bytes"] = mp4_bytes
                        S["ls_pdf_video_name"] = doc["name"]
                        st.success("Your narrated video lesson is ready.")
                    except Exception as exc:
                        st.error(f"Could not create the video: {exc}")
                        st.info(
                            "Video narration uses Edge TTS and MoviePy. Check your internet "
                            "connection and whether the selected language has an available voice."
                        )

                saved_video = S.get("ls_pdf_video_bytes")
                if saved_video and S.get("ls_pdf_video_name") == doc["name"]:
                    st.video(saved_video)
                    st.download_button(
                        "Download narrated video (MP4)",
                        data=saved_video,
                        file_name=f"{os.path.splitext(doc['name'])[0]}_video_lesson.mp4",
                        mime="video/mp4",
                        key="ls_pdf_video_download",
                    )

    else:
        st.subheader("Video → Transcript → Lecture Notes")
        video = st.file_uploader(
            "Upload a lecture video", type=["mp4", "mov", "mkv", "webm", "mpeg", "mpg"],
            key="ls_video_upload",
        )
        st.selectbox(
            "Whisper model size", ["tiny", "base", "small", "medium"],
            index=1, key="ls_whisper_model",
            help="Larger models may be more accurate but require more RAM and processing time.",
        )
        if video is not None:
            video_bytes = video.getvalue()
            st.video(video_bytes)
            st.caption(f"{video.name} · {len(video_bytes) / (1024 * 1024):.1f} MB")
            if st.button("Transcribe video", key="ls_transcribe_video", type="primary"):
                try:
                    with st.spinner("Transcribing video. This may take a while..."):
                        transcript, segments = _video_transcribe(
                            video_bytes, os.path.splitext(video.name)[1] or ".mp4"
                        )
                    if not transcript:
                        st.warning("No speech transcript was returned for this video.")
                    else:
                        S["ls_video_transcript"] = transcript
                        S["ls_video_segments"] = segments
                        S["ls_video_name"] = video.name
                        st.success("Video transcription completed.")
                except Exception as exc:
                    st.error(str(exc))

        transcript = S.get("ls_video_transcript", "")
        if transcript:
            st.markdown("### Transcript")
            edited = st.text_area(
                "Review or edit the transcript", value=transcript,
                height=260, key="ls_video_transcript_edit",
            )
            S["ls_video_transcript_final"] = edited
            st.download_button(
                "Download transcript (.txt)", edited,
                file_name=f"{os.path.splitext(S.get('ls_video_name', 'lecture'))[0]}_transcript.txt",
                mime="text/plain", key="ls_video_transcript_download",
            )
            st.markdown("### Lecture notes")
            if st.button("Prepare notes from transcript", key="ls_video_notes"):
                S["ls_video_notes_text"] = (
                    f"# Lecture Notes: {S.get('ls_video_name', 'Video')}\n\n"
                    f"## Language selected: {language}\n\n"
                    "## Transcript-based key content\n\n" + edited
                )
            if S.get("ls_video_notes_text"):
                st.markdown(S["ls_video_notes_text"])
                try:
                    video_pdf = _learning_notes_pdf(
                        f"Lecture Notes - {S.get('ls_video_name', 'Video')}",
                        S["ls_video_notes_text"],
                    )
                    st.download_button(
                        "Download Lecture Notes PDF", video_pdf,
                        file_name=f"{os.path.splitext(S.get('ls_video_name', 'lecture'))[0]}_lecture_notes.pdf",
                        mime="application/pdf", key="ls_video_notes_pdf",
                    )
                except Exception as exc:
                    st.warning(f"PDF export unavailable: {exc}")
        else:
            st.info(
                "Video transcription uses the optional open-source Whisper package. "
                "Install `openai-whisper` and FFmpeg to enable it."
            )

    st.divider()
    st.caption(
        "Source-grounding: PDF explanations and AI notes use DocuTrust's existing retrieval pipeline. "
        "Video notes are based on the generated transcript. Diagram and animation previews are illustrative."
    )



# ============================================================ NEW: INSIGHTS / NOTEBOOK / PLANNER
def _syllables(w):
    w = w.lower()
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def _read_label(score):
    return ("Very easy" if score >= 80 else "Easy" if score >= 70 else "Fairly easy" if score >= 60 else
            "Standard" if score >= 50 else "Difficult" if score >= 30 else "Very difficult")


def page_insights():
    if need_docs():
        return
    import pandas as pd
    ids = list(S.docs)
    did = pw(st.selectbox, t("document"), "in_doc", ids[0], options=ids, format_func=lambda i: S.docs[i]["name"])
    d = S.docs[did]
    full = "\n".join(p["text"] for p in d["pages"])
    total_words = sum(p["words"] for p in d["pages"])
    if not total_words:
        st.warning("This document has no extracted text yet. Enable OCR in PDF Workspace for scanned files.")
        return
    sents = split_sentences(full)
    eng = re.findall(r"[A-Za-z']+", full)
    uniq = len({w.lower() for w in words(full)})
    tiles = [("Reading time", f"{max(1, round(total_words / 200))} min"), ("Words", f"{total_words:,}"),
             ("Sentences", f"{len(sents):,}"), ("Avg sentence", f"{total_words / max(1, len(sents)):.1f} words"),
             ("Unique words", f"{uniq:,}")]
    if len(eng) > 30:
        syl = sum(_syllables(w) for w in eng)
        fl = 206.835 - 1.015 * (len(eng) / max(1, len(sents))) - 84.6 * (syl / len(eng))
        tiles.append(("Readability", f"{_read_label(fl)} ({fl:.0f})"))
    st.markdown("<div class='dt-stats'>" + "".join(f"<div class='dt-stat'><b>{v}</b><span>{k}</span></div>" for k, v in tiles) + "</div>",
                unsafe_allow_html=True)
    units = [(d["name"], p["page"], p["text"]) for p in d["pages"] if p["text"]]
    l, r = st.columns(2)
    kw = Counter(w for w in words(full) if w not in STOP and len(w) > 3).most_common(15)
    with l:
        st.subheader("Top keywords")
        if kw:
            st.bar_chart(pd.DataFrame({"mentions": [c for _, c in kw]}, index=[w for w, _ in kw]), color=active_vars()["accent"])
    with r:
        st.subheader("Words per page")
        st.area_chart(pd.DataFrame({"words": [p["words"] for p in d["pages"]]}, index=[p["page"] for p in d["pages"]]),
                      color=active_vars()["accent2"])
    st.subheader("Key sentences")
    for s_, dn, pg in key_sentences(units, 5):
        st.markdown(f"<div class='dt-hit'>{esc(s_)}<br><span class='dt-muted'>{t('page')} {pg}</span></div>", unsafe_allow_html=True)
    st.button(t("open_reader"), key="in_open", on_click=go_reader, args=(did,), icon=":material/menu_book:")
    defs, forms = extract_defs(units), extract_formulas(units)
    g1, g2 = st.tabs([f"Glossary ({len(defs)})", f"Formulas ({len(forms)})"])
    with g1:
        if defs:
            st.markdown(html_table([{"Term": a, "Meaning": b, "Page": p_} for a, b, _, p_, _ in defs]), unsafe_allow_html=True)
            st.download_button("Download glossary (.md)", "\n".join(f"- **{a}**: {b} (p.{p_})" for a, b, _, p_, _ in defs),
                               file_name="glossary.md", key="in_gl_dl")
        else:
            st.info("No definition-style sentences were found.")
    with g2:
        if forms:
            st.markdown(html_table([{"Formula": f, "Page": p_} for f, _, p_ in forms]), unsafe_allow_html=True)
        else:
            st.info("No formulas were found.")
    if len(ids) > 1:
        st.subheader("Compare with another document")
        other = st.selectbox("Compare with", [i for i in ids if i != did], format_func=lambda i: S.docs[i]["name"], key="in_cmp")
        o_text = "\n".join(p["text"] for p in S.docs[other]["pages"])
        a_set = {w for w, _ in Counter(w for w in words(full) if w not in STOP and len(w) > 3).most_common(60)}
        b_set = {w for w, _ in Counter(w for w in words(o_text) if w not in STOP and len(w) > 3).most_common(60)}
        shared = sorted(a_set & b_set)
        st.markdown(f"**{len(shared)} shared keywords:** " + (", ".join(shared) if shared else "none"))
        c1, c2 = st.columns(2)
        c1.caption("Only in this document: " + ", ".join(sorted(a_set - b_set)[:20]))
        c2.caption("Only in the other: " + ", ".join(sorted(b_set - a_set)[:20]))


def toggle_bookmark(did, pg):
    for b in list(S.bookmarks):
        if b["doc"] == did and b["page"] == pg:
            S.bookmarks.remove(b)
            return
    d = S.docs.get(did)
    S.bookmarks.append(dict(doc=did, name=d["name"] if d else did, page=pg))


def save_note(nkey):
    v = str(S.get("rd_note_" + nkey, "")).strip()
    if v:
        S.notes[nkey] = v
    else:
        S.notes.pop(nkey, None)


def reader_tools():
    """Bookmark + personal note for the page currently open in the Reader."""
    if not S.docs or S.rd_doc not in S.docs:
        return
    did, pg = S.rd_doc, S.rd_page
    marked = any(b["doc"] == did and b["page"] == pg for b in S.bookmarks)
    st.divider()
    c1, c2 = st.columns([1, 3])
    with c1:
        st.button("🔖 Remove bookmark" if marked else "🔖 Bookmark this page", key="rd_bm", on_click=toggle_bookmark,
                  args=(did, pg), type="secondary" if marked else "primary", use_container_width=True)
    nkey = f"{did}:{pg}"
    with c2:
        st.text_area("📝 Your note for this page", value=S.notes.get(nkey, ""), key="rd_note_" + nkey, height=90,
                     placeholder="Write anything you want to remember about this page...", on_change=save_note, args=(nkey,))


def page_reader():
    _page_reader_core()
    reader_tools()


def _doc_name(did):
    return S.docs[did]["name"] if did in S.docs else "(document not loaded)"


def page_notebook():
    tb, tn, tk = st.tabs([f"🔖 Bookmarks ({len(S.bookmarks)})", f"📝 Notes ({len(S.notes)})", "💾 Backup"])
    with tb:
        if not S.bookmarks:
            st.info("No bookmarks yet. Open a page in the Reader and press 'Bookmark this page'.")
        for i, b in enumerate(list(S.bookmarks)):
            snip = ""
            if b["doc"] in S.docs and b["page"] <= S.docs[b["doc"]]["page_count"]:
                snip = S.docs[b["doc"]]["pages"][b["page"] - 1]["text"][:160].replace("\n", " ")
            st.markdown(f"<div class='dt-card'><b>{esc(b['name'])}</b> · {t('page')} {b['page']}<br>"
                        f"<span class='dt-muted'>{esc(snip)}</span></div>", unsafe_allow_html=True)
            c1, c2, _ = st.columns([1, 1, 4])
            c1.button(t("open_reader"), key=f"nb_o_{i}", on_click=go_reader, args=(b["doc"], b["page"]), disabled=b["doc"] not in S.docs)
            c2.button(t("remove"), key=f"nb_r_{i}", on_click=toggle_bookmark, args=(b["doc"], b["page"]))
    with tn:
        if not S.notes:
            st.info("No notes yet. Write one under any page in the Reader.")
        else:
            md = ["# My notes", ""]
            for key, txt in sorted(S.notes.items()):
                did, pg = key.rsplit(":", 1)
                md += [f"## {_doc_name(did)} - page {pg}", txt, ""]
            st.download_button("Download all notes (.md)", "\n".join(md), file_name="my_notes.md", key="nb_md")
        for key, txt in sorted(S.notes.items()):
            did, pg = key.rsplit(":", 1)
            st.markdown(f"<div class='dt-card'><b>{esc(_doc_name(did))}</b> · {t('page')} {esc(pg)}<br>{esc(txt)}</div>", unsafe_allow_html=True)
            c1, c2, _ = st.columns([1, 1, 4])
            c1.button(t("open_reader"), key=f"nn_o_{key}", on_click=go_reader, args=(did, int(pg)), disabled=did not in S.docs)
            c2.button(t("remove"), key=f"nn_r_{key}", on_click=lambda k=key: (S.notes.pop(k, None), S.pop("rd_note_" + k, None)))
    with tk:
        st.caption("Notes, bookmarks, tasks and study log live in this browser session. Download a backup to keep them.")
        blob = json.dumps(dict(bookmarks=S.bookmarks, notes=S.notes, tasks=S.tasks, planner=S.planner), ensure_ascii=False, indent=2)
        st.download_button("Download backup (.json)", blob, file_name="docutrust_backup.json", mime="application/json", key="nb_bk")
        up = st.file_uploader("Restore from backup", type=["json"], key="nb_up")
        if up is not None and st.button("Restore now", key="nb_restore", type="primary"):
            try:
                data = json.loads(up.getvalue().decode("utf-8"))
                if isinstance(data.get("bookmarks"), list):
                    S.bookmarks = [b for b in data["bookmarks"] if isinstance(b, dict) and {"doc", "name", "page"} <= set(b)]
                if isinstance(data.get("notes"), dict):
                    S.notes = {str(k): str(v) for k, v in data["notes"].items()}
                if isinstance(data.get("tasks"), list):
                    S.tasks = [x for x in data["tasks"] if isinstance(x, dict) and "text" in x and "id" in x]
                if isinstance(data.get("planner"), dict):
                    for k in ("goal", "exam_on", "exam_name", "exam", "log"):
                        if k in data["planner"]:
                            S.planner[k] = data["planner"][k]
                st.success("Backup restored.")
            except Exception as exc:
                st.error(f"Could not restore: {exc}")


def log_minutes(m):
    k = date.today().isoformat()
    S.planner["log"][k] = S.planner["log"].get(k, 0) + int(m)


def study_streak():
    log, day = S.planner["log"], date.today()
    if not log.get(day.isoformat()):
        day -= timedelta(days=1)
    n = 0
    while log.get(day.isoformat(), 0) > 0:
        n += 1
        day -= timedelta(days=1)
    return n


def page_planner():
    import pandas as pd
    pl = S.planner
    today = date.today()
    done, goal = pl["log"].get(today.isoformat(), 0), max(1, int(pl["goal"]))
    week = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    tiles = [("Today", f"{done} min"), ("Daily goal", f"{goal} min"), ("Streak", f"{study_streak()} days"),
             ("This week", f"{sum(pl['log'].get(x.isoformat(), 0) for x in week)} min")]
    if pl.get("exam_on") and pl.get("exam"):
        try:
            left = (date.fromisoformat(pl["exam"]) - today).days
            tiles.append((pl.get("exam_name") or "Exam", f"{left} days" if left >= 0 else "done"))
        except ValueError:
            pass
    st.markdown("<div class='dt-stats'>" + "".join(f"<div class='dt-stat'><b>{v}</b><span>{k}</span></div>" for k, v in tiles) + "</div>",
                unsafe_allow_html=True)
    st.progress(min(1.0, done / goal), text="Daily goal reached 🎉" if done >= goal else f"{goal - done} min to today's goal")
    a, b = st.columns(2)
    with a:
        st.subheader("Log study time")
        c = st.columns(3)
        for col, m in zip(c, (15, 25, 45)):
            col.button(f"+{m} min", key=f"pl_add_{m}", on_click=log_minutes, args=(m,), use_container_width=True)
        st.caption("Finish a Pomodoro from the sidebar, then log it here.")
        st.button("Reset today", key="pl_reset", on_click=lambda: pl["log"].pop(today.isoformat(), None))
        S["pl_goal"] = int(pl["goal"])
        st.number_input("Daily goal (minutes)", min_value=5, max_value=600, step=5, key="pl_goal",
                        on_change=lambda: pl.__setitem__("goal", int(S["pl_goal"])))
    with b:
        st.subheader("Exam countdown")
        S["pl_exam_on"] = bool(pl["exam_on"])
        st.toggle("Track an exam", key="pl_exam_on", on_change=lambda: pl.__setitem__("exam_on", S["pl_exam_on"]))
        if pl["exam_on"]:
            S["pl_exam_name"] = pl["exam_name"]
            st.text_input("Exam name", key="pl_exam_name", placeholder="e.g. Semester exam",
                          on_change=lambda: pl.__setitem__("exam_name", S["pl_exam_name"].strip()))
            try:
                cur = date.fromisoformat(pl["exam"]) if pl["exam"] else today + timedelta(days=30)
            except ValueError:
                cur = today + timedelta(days=30)
            S["pl_exam_date"] = cur
            st.date_input("Exam date", key="pl_exam_date", on_change=lambda: pl.__setitem__("exam", S["pl_exam_date"].isoformat()))
            if not pl["exam"]:
                pl["exam"] = cur.isoformat()
    st.subheader("Last 7 days")
    st.bar_chart(pd.DataFrame({"minutes": [pl["log"].get(x.isoformat(), 0) for x in week]}, index=[x.strftime("%a %d") for x in week]),
                 color=active_vars()["accent"])
    st.subheader("Study tasks")
    with st.form("pl_add_task", clear_on_submit=True):
        txt = st.text_input("New task", placeholder="e.g. Revise chapter 3 and make flashcards")
        if st.form_submit_button("Add task", type="primary") and txt.strip():
            S.tasks.append(dict(id=hashlib.md5((txt + str(datetime.now().timestamp())).encode()).hexdigest()[:8], text=txt.strip(), done=False))
    for tk in S.tasks:
        st.checkbox(tk["text"], value=tk["done"], key="pl_t_" + tk["id"],
                    on_change=lambda tk=tk: tk.__setitem__("done", S["pl_t_" + tk["id"]]))
    if any(tk["done"] for tk in S.tasks):
        st.button("Clear completed", key="pl_clear", on_click=lambda: S.__setitem__("tasks", [x for x in S.tasks if not x["done"]]))
    elif not S.tasks:
        st.caption("No tasks yet. Add your first one above.")


PAGE_FN = dict(dashboard=page_dashboard, workspace=page_workspace, library=page_library, reader=page_reader,
               search=page_search, ask=page_ask, study=page_study, listen=page_listen,
               learning_studio=page_learning_studio, insights=page_insights, notebook=page_notebook, planner=page_planner,
               settings=page_settings)


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
        st.button("✨ Replay intro", key="replay_intro", use_container_width=True,
                  on_click=lambda: S.__setitem__("entered", False))


def topbar():
    c1, c2, c4 = st.columns([6, 1.3, 1.3])
    c1.markdown(f"## {t(S.nav)}")
    with c2:
        if S.focus:
            st.button(t("exit_focus"), key="btn_focus", on_click=toggle_focus, type="primary", icon=":material/close_fullscreen:")
        else:
            st.button(t("focus"), key="btn_focus", on_click=toggle_focus, icon=":material/center_focus_strong:")
    with c4:
        with st.popover(t("shortcuts"), icon=":material/keyboard:"):
            shortcut_table()


# ============================================================ ACCOUNT HELPERS (used from Settings)
# Optional local email accounts (salted PBKDF2) and Google OIDC are still available
# as "Continue with ..." options, but nobody is forced through a sign-in wall.
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


WELCOME_CSS = """<style>
.wl-orb{position:fixed;border-radius:50%;filter:blur(70px);opacity:.30;pointer-events:none;z-index:0;animation:wlorb 16s ease-in-out infinite}
.wl-o1{width:340px;height:340px;background:var(--accent);top:-90px;left:-70px}
.wl-o2{width:420px;height:420px;background:var(--accent2);bottom:-140px;right:-100px;animation-delay:-6s}
.wl-o3{width:240px;height:240px;background:color-mix(in srgb,var(--accent) 50%,var(--accent2));top:45%;left:42%;animation-delay:-11s;opacity:.2}
@keyframes wlorb{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(40px,-30px) scale(1.12)}66%{transform:translate(-30px,40px) scale(.92)}}
.wl-badge{display:inline-block;padding:.28rem .85rem;border:1px solid var(--accent);color:var(--accent);border-radius:99px;font-size:.78rem;letter-spacing:.14em;font-weight:700}
.wl-title{font-family:'Bricolage Grotesque',sans-serif;font-size:clamp(2.4rem,6vw,4.4rem);font-weight:800;line-height:1.04;margin:.7rem 0;
  background:linear-gradient(90deg,var(--accent),var(--accent2),var(--accent));background-size:200% auto;-webkit-background-clip:text;background-clip:text;
  color:transparent;animation:wlshine 6s linear infinite}
@keyframes wlshine{to{background-position:200% center}}
.wl-sub{color:var(--muted);font-size:1.12rem;max-width:680px;line-height:1.6}
.wl-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.8rem;margin:1.5rem 0 1rem}
.wl-grid .dt-card{animation:wlfloat 5s ease-in-out infinite}
.wl-grid .dt-card:nth-child(2){animation-delay:.6s}.wl-grid .dt-card:nth-child(3){animation-delay:1.2s}.wl-grid .dt-card:nth-child(4){animation-delay:1.8s}
@keyframes wlfloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
</style><div class="wl-orb wl-o1"></div><div class="wl-orb wl-o2"></div><div class="wl-orb wl-o3"></div>"""


def _goto(page):
    S["nav"] = page


def _enter():
    S["_fresh"] = True
    S["entered"] = True
    S["nav"] = "dashboard"


SPLASH_JS = """<script>
const P=window.parent,D=P.document,R=D.documentElement;
if(!P.__spOn){P.__spOn=1;
 const mv=e=>{
  if(!D.querySelector('.sp-stage')){D.removeEventListener('pointermove',mv);P.__spOn=0;
    ['--mx','--my','--px','--py'].forEach(k=>R.style.removeProperty(k));return}
  R.style.setProperty('--mx',e.clientX+'px');R.style.setProperty('--my',e.clientY+'px');
  R.style.setProperty('--px',((e.clientX/P.innerWidth)-.5)*2);R.style.setProperty('--py',((e.clientY/P.innerHeight)-.5)*2)};
 D.addEventListener('pointermove',mv)}
</script>"""


def splash():
    """Front page: logo + animated WELCOME. The Continue button fades in after ~3 seconds."""
    inject_css()
    rnd = random.Random(11)
    dots = "".join(
        f"<i class='sp-dot' style='left:{rnd.randint(1, 98)}%;top:{rnd.randint(30, 100)}%;"
        f"width:{rnd.randint(3, 9)}px;height:{rnd.randint(3, 9)}px;animation-delay:-{rnd.randint(0, 16)}s;"
        f"animation-duration:{rnd.randint(8, 18)}s'></i>" for _ in range(40))
    stars = "".join(
        f"<i class='sp-star' style='left:{rnd.randint(1, 99)}%;top:{rnd.randint(1, 60)}%;animation-delay:-{rnd.randint(0, 9)}s;"
        f"animation-duration:{rnd.randint(3, 7)}s'></i>" for _ in range(36))
    letters = "".join(f"<span style='--i:{i}'>{c}</span>" for i, c in enumerate("WELCOME"))
    hello = "".join(f"<span style='--k:{k}'>{g}</span>" for k, g in
                    enumerate(["नमस्ते", "Hello", "Hola", "こんにちは", "Bonjour", "مرحبا"]))
    logo = ("<svg width='104' height='104' viewBox='0 0 128 128' xmlns='http://www.w3.org/2000/svg' aria-hidden='true'>"
            "<rect x='8' y='8' width='112' height='112' rx='28' fill='var(--surface2)' stroke='var(--border)'/>"
            "<rect class='sp-draw' pathLength='1' x='36' y='25' width='56' height='78' rx='8' fill='none' stroke='var(--accent)' stroke-width='7'/>"
            "<path class='sp-draw d2' pathLength='1' d='M50 66l10 11 20-25' fill='none' stroke='var(--accent)' stroke-width='8' "
            "stroke-linecap='round' stroke-linejoin='round'/></svg>")
    orbit = "<div class='sp-orbit'>" + "".join(
        f"<div class='sp-arm' style='--n:{k}'><div class='sp-oi' style='--n:{k}'>{ic}</div></div>"
        for k, ic in enumerate(["📄", "💬", "🎓", "🎧", "🔎", "🎬"])) + "</div>"
    sp_html = ("""<style>
    [data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],[data-testid='stToolbar']{display:none!important}
    .sp-blob{position:fixed;border-radius:50%;pointer-events:none;z-index:0;width:75vmax;height:75vmax;opacity:.5;
      background:radial-gradient(circle,var(--c) 0%,transparent 62%);animation:spdrift 24s ease-in-out infinite}
    .sp-b1{--c:var(--accent);left:-28vmax;top:-34vmax}
    .sp-b2{--c:var(--accent2);right:-32vmax;bottom:-38vmax;animation-delay:-9s;animation-duration:29s}
    .sp-b3{--c:color-mix(in srgb,var(--accent) 50%,var(--accent2));left:32vw;top:16vh;width:46vmax;height:46vmax;opacity:.26;animation-delay:-15s}
    @keyframes spdrift{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(6vmax,4vmax) scale(1.12)}66%{transform:translate(-4vmax,7vmax) scale(.94)}}
    .sp-vignette{position:fixed;inset:0;z-index:0;pointer-events:none;background:radial-gradient(ellipse at center,transparent 35%,var(--bg) 92%)}
    .sp-floor{position:fixed;left:-50%;width:200%;bottom:0;height:50vh;z-index:0;pointer-events:none;opacity:.32;transform-origin:50% 100%;
      transform:perspective(520px) rotateX(64deg);
      background-image:linear-gradient(var(--accent) 1px,transparent 1px),linear-gradient(90deg,var(--accent) 1px,transparent 1px);background-size:64px 64px;
      -webkit-mask-image:linear-gradient(to top,#000 8%,transparent 90%);mask-image:linear-gradient(to top,#000 8%,transparent 90%);
      animation:spfloor 2.4s linear infinite}
    @keyframes spfloor{to{background-position:0 64px}}
    .sp-wave{position:fixed;left:50%;top:21%;width:0;height:0;border-radius:50%;border:2px solid var(--accent);opacity:0;z-index:0;pointer-events:none;
      animation:spwave 5s cubic-bezier(.2,.6,.3,1) infinite}
    .sp-wave.w2{animation-delay:1.7s;border-color:var(--accent2)}.sp-wave.w3{animation-delay:3.4s}
    @keyframes spwave{0%{width:0;height:0;margin:0;opacity:.85}100%{width:150vmax;height:150vmax;margin:-75vmax 0 0 -75vmax;opacity:0}}
    .sp-star{position:fixed;width:2px;height:2px;border-radius:50%;background:var(--text);z-index:0;pointer-events:none;opacity:0;animation:sptw 5s ease-in-out infinite}
    @keyframes sptw{0%,100%{opacity:0;transform:scale(.6)}50%{opacity:.85;transform:scale(1.8)}}
    .sp-dot{position:fixed;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent);opacity:0;z-index:0;pointer-events:none;animation:spdot 12s linear infinite}
    @keyframes spdot{0%{transform:translateY(40px) scale(.6);opacity:0}20%{opacity:.9}80%{opacity:.6}100%{transform:translateY(-160px) scale(1.2);opacity:0}}
    .sp-stage{position:relative;z-index:1;display:flex;flex-direction:column;align-items:center;text-align:center;margin-top:9vh}
    .sp-emblem{position:relative;width:172px;height:172px;display:grid;place-items:center;animation:spdrop 1.3s cubic-bezier(.2,.9,.2,1.1) both}
    @keyframes spdrop{from{opacity:0;transform:translateY(-70px) scale(.3) rotate(-200deg)}to{opacity:1;transform:none}}
    .sp-emblem::before{content:"";position:absolute;inset:-34px;border-radius:50%;background:radial-gradient(circle,color-mix(in srgb,var(--accent) 45%,transparent),transparent 68%);animation:sppulse 3.6s ease-in-out infinite}
    @keyframes sppulse{0%,100%{transform:scale(.9);opacity:.7}50%{transform:scale(1.12);opacity:1}}
    .sp-ring{position:absolute;inset:0;border-radius:50%;
      background:conic-gradient(from 0deg,var(--accent),transparent 30%,var(--accent2) 55%,transparent 80%,var(--accent));
      -webkit-mask:radial-gradient(circle,transparent 62%,#000 64%);mask:radial-gradient(circle,transparent 62%,#000 64%);animation:spspin 6s linear infinite}
    .sp-ring.r2{inset:14px;animation-duration:10s;animation-direction:reverse;opacity:.6}
    @keyframes spspin{to{transform:rotate(360deg)}}
    .sp-emblem svg{position:relative;filter:drop-shadow(0 0 26px var(--accent));animation:spbreath 3.6s ease-in-out infinite}
    @keyframes spbreath{0%,100%{transform:scale(1)}50%{transform:scale(1.07)}}
    .sp-title{display:flex;justify-content:center;perspective:700px;margin:1.5rem 0 .5rem;
      font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:clamp(3rem,11vw,8.2rem);line-height:1;letter-spacing:.06em;
      filter:drop-shadow(0 10px 34px color-mix(in srgb,var(--accent) 45%,transparent))}
    .sp-title span{display:inline-block;background:linear-gradient(180deg,var(--text) 10%,var(--accent) 60%,var(--accent2));
      -webkit-background-clip:text;background-clip:text;color:transparent;
      animation:splet .95s calc(var(--i)*.13s + .25s) cubic-bezier(.2,.9,.2,1.15) both,spletw 3.4s calc(var(--i)*.13s + 1.8s) ease-in-out infinite}
    @keyframes splet{from{opacity:0;transform:translateY(90px) rotateX(-85deg) scale(.5);filter:blur(16px)}to{opacity:1;transform:none;filter:blur(0)}}
    @keyframes spletw{0%,100%{transform:translateY(0);filter:brightness(1)}50%{transform:translateY(-12px);filter:brightness(1.5)}}
    .sp-line{height:3px;width:0;border-radius:99px;background:linear-gradient(90deg,transparent,var(--accent),var(--accent2),transparent);
      animation:spline 1.4s 1.3s cubic-bezier(.2,.8,.2,1) forwards}
    @keyframes spline{to{width:min(380px,72vw)}}
    .sp-hello{position:relative;height:2rem;width:min(380px,80vw);margin-top:1rem;color:var(--muted);font-size:1.15rem;letter-spacing:.3em;text-transform:uppercase}
    .sp-hello span{position:absolute;inset:0;opacity:0;animation:sphello 15s calc(2.2s + var(--k)*2.5s) infinite}
    @keyframes sphello{0%{opacity:0;transform:translateY(10px);filter:blur(5px)}3%,13%{opacity:1;transform:none;filter:none}17%,100%{opacity:0;transform:translateY(-10px);filter:blur(5px)}}
    .sp-load{width:min(240px,60vw);height:3px;border-radius:99px;background:color-mix(in srgb,var(--text) 14%,transparent);overflow:hidden;margin-top:.8rem;
      animation:spgone .5s 3s both}
    .sp-load i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--accent2));animation:spfill 3s linear both}
    @keyframes spfill{to{width:100%}}
    @keyframes spgone{to{opacity:0;height:0;margin-top:0}}
    .st-key-wl_continue{animation:spreveal 1s 3s both}
    @keyframes spreveal{0%{opacity:0;visibility:hidden;transform:translateY(18px) scale(.9)}1%{visibility:visible}100%{opacity:1;visibility:visible;transform:none}}
    .st-key-wl_continue button{position:relative;overflow:hidden;padding:.9rem 1.4rem;font-size:1.08rem;border-radius:99px;animation:spglow 2.6s ease-in-out 4s infinite}
    .st-key-wl_continue button::after{content:"";position:absolute;top:0;left:-60%;width:40%;height:100%;transform:skewX(-20deg);
      background:linear-gradient(100deg,transparent,rgba(255,255,255,.55),transparent);animation:spshine 3.4s 4.2s infinite}
    @keyframes spshine{0%{left:-60%}35%,100%{left:140%}}
    @keyframes spglow{0%,100%{box-shadow:0 0 0 0 color-mix(in srgb,var(--accent) 60%,transparent)}50%{box-shadow:0 0 0 16px transparent}}
    .sp-draw{stroke-dasharray:1;stroke-dashoffset:1;animation:spdraw 1.2s 1s ease-out forwards}
    .sp-draw.d2{animation-delay:2s;animation-duration:.6s}
    @keyframes spdraw{to{stroke-dashoffset:0}}
    .sp-flash{position:absolute;inset:0;border-radius:50%;border:3px solid var(--accent);opacity:0;animation:spflash 1.1s 2.6s ease-out}
    @keyframes spflash{0%{opacity:.9;transform:scale(.8)}100%{opacity:0;transform:scale(2.4)}}
    .sp-orbit{--R:min(250px,41vw);position:absolute;left:50%;top:50%;width:0;height:0;transform:scaleY(.34);pointer-events:none;animation:spfade 1.4s 1.6s both}
    .sp-orbit::before{content:"";position:absolute;left:calc(var(--R)*-1);top:calc(var(--R)*-1);width:calc(var(--R)*2);height:calc(var(--R)*2);
      border-radius:50%;border:1px dashed color-mix(in srgb,var(--accent) 40%,transparent)}
    .sp-arm{position:absolute;left:0;top:0;width:0;height:0;animation:spspin 24s linear infinite;animation-delay:calc(var(--n)*-4s)}
    .sp-oi{position:absolute;left:var(--R);top:0;width:46px;height:46px;margin:-23px 0 0 -23px;display:grid;place-items:center;font-size:22px;border-radius:50%;
      background:color-mix(in srgb,var(--surface) 80%,transparent);border:1px solid color-mix(in srgb,var(--accent) 55%,var(--border));
      box-shadow:0 0 18px color-mix(in srgb,var(--accent) 45%,transparent);animation:sporb 24s linear infinite;animation-delay:calc(var(--n)*-4s)}
    @keyframes sporb{from{transform:rotate(0) scaleY(2.941)}to{transform:rotate(-360deg) scaleY(2.941)}}
    @keyframes spfade{from{opacity:0}to{opacity:1}}
    .sp-spot{position:fixed;inset:0;z-index:0;pointer-events:none;
      background:radial-gradient(520px circle at var(--mx,50%) var(--my,35%),color-mix(in srgb,var(--accent) 30%,transparent),transparent 62%)}
    .sp-emblem{translate:calc(var(--px,0)*-16px) calc(var(--py,0)*-12px);transition:translate .3s ease-out}
    .sp-title{translate:calc(var(--px,0)*12px) calc(var(--py,0)*7px);transition:translate .3s ease-out}
    .sp-blob{translate:calc(var(--px,0)*28px) calc(var(--py,0)*20px);transition:translate .7s ease-out}
    @media (prefers-reduced-motion:reduce){.sp-draw{stroke-dashoffset:0!important}.sp-orbit,.sp-flash{display:none}.sp-emblem,.sp-title,.sp-blob{translate:none}}
    @media (prefers-reduced-motion:reduce){.sp-blob,.sp-floor,.sp-wave,.sp-star,.sp-dot,.sp-ring,.sp-emblem,.sp-emblem::before,.sp-emblem svg,.sp-title span,.sp-line,.sp-load,.sp-load i,.st-key-wl_continue,.st-key-wl_continue button,.st-key-wl_continue button::after{animation:none!important}
      .sp-line{width:min(380px,72vw)}.sp-load{display:none}.sp-hello span{animation:none!important}.sp-hello span:first-child{opacity:1}}
    </style><div class='sp-blob sp-b1'></div><div class='sp-blob sp-b2'></div><div class='sp-blob sp-b3'></div><div class='sp-vignette'></div><div class='sp-floor'></div><div class='sp-spot'></div>
    <div class='sp-wave'></div><div class='sp-wave w2'></div><div class='sp-wave w3'></div>""" + stars + dots +
                f"<div class='sp-stage'><div class='sp-emblem'><div class='sp-ring'></div><div class='sp-ring r2'></div>{logo}<div class='sp-flash'></div>{orbit}</div>"
                f"<div class='sp-title'>{letters}</div><div class='sp-line'></div><div class='sp-hello'>{hello}</div>"
                "<div class='sp-load'><i></i></div></div>")
    # one flush-left line: indented lines would be rendered as a code block by markdown
    st.markdown("".join(line.strip() for line in sp_html.splitlines()), unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.button("Continue to DocuTrust AI  →", key="wl_continue", type="primary", use_container_width=True,
                  on_click=_enter)
    components.html(SPLASH_JS, height=0)


def home_extras():
    """Everything DocuTrust AI can do, shown under the Dashboard."""
    feats = [("📄", "Smart PDF reading", "OCR for scanned pages, clean reader, 100+ translation languages", "workspace"),
             ("💬", "Cited AI answers", "Ask questions, every answer points to its page", "ask"),
             ("🔎", "Smart search", "Keyword, phrase and all-words search across documents", "search"),
             ("🎓", "Study tools", "Notes, quizzes and flashcards from your PDFs", "study"),
             ("🎬", "Learning Studio", "Narrated video lessons, diagrams, lecture transcripts", "learning_studio"),
             ("🎧", "Listen mode", "Hear any page read aloud in many languages", "listen"),
             ("📊", "Insights", "Reading time, readability, keywords, glossary, compare PDFs", "insights"),
             ("🔖", "Notebook", "Bookmarks and page notes with backup", "notebook"),
             ("🗓️", "Study planner", "Daily goal, streak, exam countdown and tasks", "planner"),
             ("🎨", "Your look", "6 themes, 21 backgrounds, fonts and reading comfort", "settings")]
    st.markdown("#### Everything you can do")
    for row in range(0, len(feats), 5):
        cols = st.columns(5)
        for col, (ico, name, desc, page) in zip(cols, feats[row:row + 5]):
            with col:
                st.markdown(f"<div class='dt-card'><div style='font-size:1.6rem'>{ico}</div><b>{name}</b><br>"
                            f"<span class='dt-muted'>{desc}</span></div>", unsafe_allow_html=True)
                st.button("Open", key="wl_go_" + page, use_container_width=True, on_click=_goto, args=(page,))


def settings_account():
    """Profile + optional account options (moved here from the old welcome page)."""
    st.subheader("Profile & account")
    S["st_name"] = S.get("user_name", "")
    st.text_input("Your name (shown on the Welcome page)", key="st_name", placeholder="Your name",
                  on_change=lambda: S.__setitem__("user_name", S["st_name"].strip()))
    if _google_oidc_configured():
        if st.button("🌐  Continue with Google", use_container_width=True, key="st_google"):
            st.login("google")
    else:
        st.button("🌐  Continue with Google", disabled=True, use_container_width=True, key="st_google_off",
                  help="Add Google OAuth credentials in .streamlit/secrets.toml to enable this.")
    with st.expander("✉️  Email account (optional)"):
        mode = st.radio("Account", ["Sign in", "Create account"], horizontal=True,
                        label_visibility="collapsed", key="st_mode")
        if mode == "Create account":
            with st.form("st_signup"):
                n = st.text_input("Full name")
                e = st.text_input("Email address")
                p = st.text_input("Password (min 8 chars)", type="password")
                c = st.text_input("Confirm password", type="password")
                go = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if go:
                if p != c:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = _register_user(n, e, p)
                    (st.success if ok else st.error)(msg)
        else:
            with st.form("st_signin"):
                e = st.text_input("Email address")
                p = st.text_input("Password", type="password")
                go = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if go:
                nm = _verify_user(e, p)
                if nm:
                    S["user_name"] = nm
                    st.success(f"Signed in as {nm}.")
                else:
                    st.error("Email or password is incorrect.")


# ============================================================ MAIN
_gu = getattr(st, "user", None)
if _gu is not None and getattr(_gu, "is_logged_in", False) and not S.get("user_name"):
    S["user_name"] = getattr(_gu, "name", "") or ""

if not S.get("entered"):
    splash()
else:
    inject_css()
    st.markdown(WELCOME_CSS, unsafe_allow_html=True)
    if S.pop("_fresh", False):  # one-time zoom-fade when entering the app from the intro
        st.markdown("<style>.block-container{animation:appin .9s cubic-bezier(.2,.8,.2,1) both}"
                    "@keyframes appin{from{opacity:0;transform:scale(1.05) translateY(16px);filter:blur(10px)}to{opacity:1;transform:none;filter:blur(0)}}"
                    "[data-testid='stSidebar']>div{animation:sbin .8s .1s cubic-bezier(.2,.8,.2,1) both}"
                    "@keyframes sbin{from{opacity:0;transform:translateX(-36px)}to{opacity:1;transform:none}}"
                    "@media (prefers-reduced-motion:reduce){.block-container,[data-testid='stSidebar']>div{animation:none!important}}</style>",
                    unsafe_allow_html=True)
    sidebar()
    topbar()
    try:
        PAGE_FN[S.nav]()
        if S.nav == "dashboard":
            home_extras()
    except Exception as _err:
        st.error(f"Something went wrong on this page: {_err}")
    shortcuts_js()
save_prefs()