"""Offline token id -> text, read straight from the target GGUF.

GPT-2 style tokenizers store tokens in a byte-level encoding: every byte of the
original UTF-8 is mapped to a printable code point, so a curly apostrophe
(U+2019, bytes E2 80 99) is stored as the three characters "âĢĻ". Reading those
characters literally is what produces mojibake like "isnâĢĻt". The map is
reversed here, so the blind-spot map and the visualiser show real text.

One consequence of byte-level tokenization: a single token can hold a *partial*
UTF-8 sequence. Decoding token by token cannot always succeed, so `decode` and
`stream_text` accumulate bytes across tokens and emit characters only once they
are complete.
"""

from __future__ import annotations

from pathlib import Path


def _byte_decoder() -> dict[str, int]:
    """Reverse of GPT-2's bytes_to_unicode: printable code point -> byte."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs)}


_DECODER = _byte_decoder()


class Vocab:
    def __init__(self, tokens: list[str], byte_level: bool = True):
        self.tokens = tokens
        self.byte_level = byte_level

    def __len__(self) -> int:
        return len(self.tokens)

    def __call__(self, token_id: int) -> str:
        return self.text(token_id)

    def raw(self, token_id: int) -> str:
        if token_id is None or token_id < 0 or token_id >= len(self.tokens):
            return ""
        return self.tokens[token_id]

    def token_bytes(self, token_id: int) -> bytes:
        """The token's original bytes."""
        t = self.raw(token_id)
        if not self.byte_level:
            return t.encode("utf-8", errors="replace")
        try:
            return bytes(_DECODER[c] for c in t)
        except KeyError:
            # a special token such as <|im_end|> is stored literally
            return t.encode("utf-8", errors="replace")

    def text(self, token_id: int) -> str:
        """Decoded text for one token in isolation.

        A token holding a partial character decodes to a replacement character;
        use `decode` when a whole sequence is available.
        """
        if token_id is None or token_id < 0 or token_id >= len(self.tokens):
            return f"<{token_id}>"
        return self.token_bytes(token_id).decode("utf-8", errors="replace")

    def repr(self, token_id: int) -> str:
        """Quoted, so leading and trailing whitespace is visible in a table."""
        return f"{self.text(token_id)!r}"

    def decode(self, token_ids) -> str:
        """Correct text for a whole sequence, joining split characters."""
        buf = b"".join(self.token_bytes(t) for t in token_ids)
        return buf.decode("utf-8", errors="replace")

    def stream_text(self, token_ids) -> list[str]:
        """Per-token text that concatenates to `decode(token_ids)`.

        A token carrying an incomplete character yields "", and the token that
        completes it yields the whole character. This keeps a token-by-token
        display honest without ever rendering a half-character.
        """
        out, pending = [], b""
        for t in token_ids:
            pending += self.token_bytes(t)
            try:
                s = pending.decode("utf-8")
            except UnicodeDecodeError:
                # hold the incomplete tail back for the next token
                for cut in range(1, min(4, len(pending)) + 1):
                    try:
                        s = pending[:-cut].decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    out.append(s)
                    pending = pending[-cut:]
                    break
                else:
                    out.append("")
                continue
            out.append(s)
            pending = b""
        if pending:
            out[-1] += pending.decode("utf-8", errors="replace")
        return out


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

        model = reader.get_field("tokenizer.ggml.model")
        name = ""
        if model is not None:
            try:
                name = str(bytes(model.parts[model.data[0]]), "utf-8")
            except Exception:
                name = ""

        return Vocab(tokens, byte_level=(name == "gpt2"))
    except Exception:
        return None
