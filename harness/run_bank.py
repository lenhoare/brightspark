#!/usr/bin/env python3
"""Run a prompt bank against a traced llama-server and collect the log.

Launches the server itself so that the trace file, the manifest and the server
settings for a run are guaranteed to belong together -- a run directory is
self-describing, which matters once there are a dozen of them.

Study defaults worth keeping:
  --spec-draft-p-min 0   the draft's own confidence early-stop truncates blocks
                         before the target ever sees them, which confounds
                         divergence with the draft giving up. Leave it off for
                         study runs; the confidence value is logged either way.
  temperature 0          acceptance semantics differ under sampling.
  --parallel 1           required for task_id -> prompt_id mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anticipation import manifest as manifest_mod  # noqa: E402


def wait_for_health(port: int, proc: subprocess.Popen, timeout: float = 600.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy within {timeout}s")


def complete(port: int, prompt: str, args, timeout: float) -> dict:
    """Send one turn through the chat endpoint so the model's own template applies.

    This matters more than it looks: posting a bare string to /completion
    bypasses the chat template entirely, and an instruct model given a bare
    sentence continues the document instead of answering it. --jinja on the
    server only affects the chat endpoints.
    """
    payload = {
        "messages": (
            ([{"role": "system", "content": args.system}] if args.system else [])
            + [{"role": "user", "content": prompt}]
        ),
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
        "stream": False,
        # the template writes an empty <think></think> pair when this is false,
        # keeping the answer in one register instead of mixing reasoning and prose
        "chat_template_kwargs": {"enable_thinking": bool(args.thinking)},
    }
    if args.n_predict:
        payload["max_tokens"] = args.n_predict

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())

    choice = (d.get("choices") or [{}])[0]
    usage = d.get("usage") or {}

    return {
        "content": (choice.get("message") or {}).get("content", ""),
        "finish_reason": choice.get("finish_reason"),
        "n_tokens": usage.get("completion_tokens"),
        "timings": d.get("timings") or {},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=str(Path.home() / "llama.cpp/build/bin/llama-server"))
    ap.add_argument("--model", required=True, help="target GGUF")
    ap.add_argument("--model-draft", default=None,
                    help="draft GGUF; omit for self-contained methods such as draft-mtp")
    ap.add_argument("--spec-type", default="draft-dspark")
    ap.add_argument("--bank", required=True, help="prompt bank JSONL")
    ap.add_argument("--out", required=True, help="run directory to create")
    ap.add_argument("--n-draft", type=int, default=7)
    ap.add_argument("--p-min", type=float, default=0.0)
    ap.add_argument("--n-predict", type=int, default=0,
                    help="backstop on generated tokens; 0 lets the model stop on its own")
    ap.add_argument("--system", default="You are a helpful agent.",
                    help="system prompt; empty string sends none")
    ap.add_argument("--thinking", action="store_true",
                    help="let the model emit a <think> block (default: off, so the "
                         "answer stays in one register)")
    ap.add_argument("--limit", type=int, default=0, help="only send the first N prompts")
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--trace-ctx", type=int, default=32)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--keep-server-log", action="store_true", default=True)
    args = ap.parse_args()

    server = Path(args.server)
    if not server.exists():
        print(f"error: server binary not found at {server}", file=sys.stderr)
        return 1

    for p in (args.model, args.model_draft, args.bank):
        if p is None:
            continue
        if not Path(p).exists():
            print(f"error: not found: {p}", file=sys.stderr)
            return 1

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"error: run directory {out} exists and is not empty", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    bank = manifest_mod.read(args.bank)
    if args.limit:
        bank = bank[:args.limit]
    if not bank:
        print("error: prompt bank is empty", file=sys.stderr)
        return 1

    trace_path = out / "trace.jsonl"
    server_log = out / "server.log"

    cmd = [
        str(server),
        "-m", str(args.model),
    ]
    if args.model_draft:
        cmd += ["-md", str(args.model_draft)]
    cmd += [
        "--spec-type", args.spec_type,
        "--spec-draft-n-max", str(args.n_draft),
        "--spec-draft-p-min", str(args.p_min),
        "--spec-trace", str(trace_path),
        "--spec-trace-ctx", str(args.trace_ctx),
        "-ngl", "99", "-ngld", "99", "-fa", "on",
        "-c", str(args.n_ctx),
        "--parallel", "1",
        "--jinja",
        "--port", str(args.port),
        "--host", "127.0.0.1",
    ]

    (out / "run.json").write_text(json.dumps({
        "started": datetime.now(timezone.utc).isoformat(),
        "cmd": cmd,
        "bank": str(Path(args.bank).resolve()),
        "n_predict": args.n_predict,
        "args": vars(args),
    }, indent=2))

    print("launching:", " ".join(cmd), flush=True)

    log = open(server_log, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)

    results = []
    try:
        wait_for_health(args.port, proc)
        print(f"server healthy; sending {len(bank)} prompts sequentially", flush=True)

        for i, entry in enumerate(bank):
            pid = entry.get("prompt_id", f"prompt_{i:03d}")
            t0 = time.time()
            try:
                resp = complete(args.port, entry["prompt"], args, args.timeout)
            except Exception as e:  # keep going; a failed prompt breaks label alignment
                print(f"  [{i+1}/{len(bank)}] {pid}: FAILED ({e})", file=sys.stderr)
                results.append({"prompt_id": pid, "ok": False, "error": str(e)})
                continue

            dt = time.time() - t0
            timings = resp.get("timings", {})
            n_tok = resp.get("n_tokens") or timings.get("predicted_n") or 0
            why = resp.get("finish_reason") or "?"
            print(f"  [{i+1}/{len(bank)}] {pid}: {dt:.1f}s  {n_tok} tok  "
                  f"{timings.get('predicted_per_second', 0):.1f} tok/s  [{why}]", flush=True)

            results.append({
                "prompt_id": pid,
                "ok": True,
                "content": resp.get("content", ""),
                # why the take ended -- "stop" is the model choosing to stop,
                # "length" is our backstop cutting it off. Worth recording rather
                # than inferring from a token count after the fact.
                "finish_reason": resp.get("finish_reason"),
                "n_tokens": n_tok,
                "timings": timings,
                "wall_s": dt,
            })
    finally:
        print("stopping server", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        log.close()

    manifest_mod.write(out / "manifest.jsonl", bank)
    (out / "responses.json").write_text(json.dumps(results, indent=2))
    shutil.copy(args.bank, out / "bank.jsonl")

    n_failed = sum(1 for r in results if not r.get("ok"))
    if n_failed:
        print(f"\nWARNING: {n_failed} prompt(s) failed. The trace's task_id ordering "
              f"no longer lines up with the manifest, so labelling will refuse to "
              f"apply. Re-run before analysing.", file=sys.stderr)

    print(f"\nrun written to {out}")
    print(f"  trace:    {trace_path}")
    print(f"  next:     python3 -m anticipation.report {out}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
