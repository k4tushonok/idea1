from pathlib import Path
from typing import Optional

TEMPLATES_DIR = Path(__file__).parent

def load_template(name: str) -> str:
    path = TEMPLATES_DIR / f"{name}.txt"
    if not path.exists():
        return f"{{{{missing_template:{name}}}}}"
    return path.read_text(encoding="utf-8")
