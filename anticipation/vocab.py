"""Offline token id -> text, read straight from the target GGUF.

The blind-spot map is the primary output and a table of bare token ids is not
readable, so decode without needing a server running. Byte-level BPE markers
are unescaped so that whitespace is visible rather than mangled.
"""

from __future__ import annotations

from pathlib import Path


class Vocab:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def __call__(self, token_id: int) -> str:
        return self.text(token_id)

    def text(self, token_id: int) -> str:
        if token_id is None or token_id < 0 or token_id >= len(self.tokens):
            return f"<{token_id}>"
        t = self.tokens[token_id]
        # byte-level BPE: U+0120 is a leading space, U+010A a newline
        return t.replace("Ġ", " ").replace("Ċ", "\n")

    def repr(self, token_id: int) -> str:
        """Quoted, so leading/trailing whitespace is visible in a table."""
        return f"{self.text(token_id)!r}"


def load(gguf_path: str | Path) -> Vocab | None:
    """Read the token list from a GGUF. Returns None if gguf-py is unavailable
    or the file has no tokenizer (a draft GGUF often does not)."""
    try:
        import gguf
    except ImportError:
        return None

    path = Path(gguf_path)
    if not path.exists():
        return None

    try:
        reader = gguf.GGUFReader(str(path))
        field = reader.get_field("tokenizer.ggml.tokens")
        if field is None:
            return None
        tokens = [
            str(bytes(field.parts[idx]), "utf-8", errors="replace")
            for idx in field.data
        ]
        return Vocab(tokens)
    except Exception:
        return None
