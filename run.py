from __future__ import annotations
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "aya-expanse:8b"
OLLAMA_URL = "http://localhost:11434"

def ollama_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
        names = {m.get("name", "") for m in data.get("models", [])}
        return any(n == MODEL or n.startswith(MODEL + ":") for n in names)
    except Exception:
        return False

def main():
    if shutil.which("ollama") is None:
        raise SystemExit(
            "Ollama is not installed or not on PATH. Install Ollama from https://ollama.com/download, "
            f"then run: ollama pull {MODEL}"
        )

    if not ollama_ready():
        print("Starting/checking Ollama...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        for _ in range(12):
            time.sleep(2)
            if ollama_ready():
                break

    if not ollama_ready():
        raise SystemExit(
            "Ollama server is not responding. Open Ollama, then run this command again."
        )

    # Check model availability explicitly.
    with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as response:
        names = {m.get("name", "") for m in json.loads(response.read().decode()).get("models", [])}
    if not any(n == MODEL or n.startswith(MODEL + ":") for n in names):
        raise SystemExit(f"Model {MODEL} is missing. Run: ollama pull {MODEL}")

    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit("Dependencies missing. Run: python -m pip install -r requirements.txt")

    print("Starting DocuTrust AI...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")],
        cwd=ROOT,
        check=False
    )

if __name__ == "__main__":
    main()
