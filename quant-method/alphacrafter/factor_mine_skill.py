"""Load the upstream factor-mine research instructions as proposal context."""
import re
from pathlib import Path

from trader_lite import file_fingerprint

SKILL_FILES = ("SKILL.md", "references/op-types.md", "references/iter-note.md")
NOTE_FIELDS = ("op_type", "hypothesis", "change", "expected")


def load_factor_mine_skill(root: Path):
    root = root.resolve(strict=True)
    instructions, inputs = [], []
    for name in SKILL_FILES:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Skill file is outside its root or is not a file: {name}")
        fingerprint = file_fingerprint(path)
        if fingerprint["bytes"] > 32_768:
            raise ValueError(f"Skill document exceeds the supported context size: {name}")
        content = path.read_text(encoding="utf-8")
        if file_fingerprint(path) != fingerprint:
            raise ValueError(f"Skill document changed during read: {name}")
        if name == "SKILL.md":
            frontmatter, separator, body = content.removeprefix("---\n").partition("\n---\n")
            if not separator or not re.search(r"^name:\s*factor-mine\s*$", frontmatter, re.MULTILINE):
                raise ValueError("Expected the upstream factor-mine SKILL.md")
            content = re.sub(r"(?ms)^\x60{3}json qsh-form\n.*?^\x60{3}\s*$", "", body).strip()
        instructions.append(f"\n[{name}]\n{content}")
        inputs.append({"file": name, **fingerprint})
    return "\n".join(instructions), {"name": "factor-mine", "files": inputs}


def note_failure(note, op_type):
    if not isinstance(note, dict) or any(
        not isinstance(note.get(field), str) or not note[field].strip() for field in NOTE_FIELDS
    ):
        return "missing_or_invalid_iteration_note"
    if note["op_type"] != op_type:
        return "iteration_note_operation_mismatch"
    return None
