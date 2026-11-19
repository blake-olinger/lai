"""Full live hardware E2E verification against a running local LAI + Ollama server."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

BASE_URL = os.environ.get("LAI_E2E_URL", "http://127.0.0.1:8400").rstrip("/")
OLLAMA_URL = os.environ.get("LAI_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
ADMIN_KEY = os.environ["LAI_E2E_ADMIN_KEY"]
CLIENT_KEY = os.environ["LAI_E2E_CLIENT_KEY"]
EXPECTED_GPU_SUBSTRING = os.environ.get("LAI_E2E_GPU_SUBSTRING", "").strip()
LOG_DIR = Path(os.environ.get("LAI_E2E_LOG_DIR", "data/logs"))
_HTTP_OK, _HTTP_CREATED, _HTTP_NOT_FOUND = 200, 201, 404
_MIN_OVERSIZED_TOKENS, _MIN_MAP_WINDOWS, _CONCURRENT_JOBS = 12_000, 2, 4


class SmokeCheckError(RuntimeError):
    """Raised when a live E2E hardware assertion fails."""


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise SmokeCheckError(detail)


def _out(line: str) -> None:
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


async def _get(c: httpx.AsyncClient, path: str, key: str = ADMIN_KEY) -> dict[str, Any]:
    r = await c.get(f"{BASE_URL}{path}", headers=_auth(key), timeout=30.0)
    _require(r.status_code == _HTTP_OK, f"GET {path} -> {r.status_code}: {r.text}")
    return dict(r.json())


async def _post(
    c: httpx.AsyncClient, path: str, key: str, body: dict[str, Any], code: int = _HTTP_OK
) -> httpx.Response:
    r = await c.post(f"{BASE_URL}{path}", headers=_auth(key), json=body, timeout=240.0)
    _require(r.status_code == code, f"POST {path} -> {r.status_code}: {r.text}")
    return r


async def _turn(
    c: httpx.AsyncClient, key: str, sid: str, content: str, **extra: object
) -> httpx.Response:
    body = {"content": content, "stream": False, "enable_thinking": False, **extra}
    return await _post(c, f"/v1/sessions/{sid}/messages", key, body)


async def _upload(
    c: httpx.AsyncClient, pid: str, fname: str, content: bytes, key: str = ADMIN_KEY
) -> dict[str, Any]:
    hdrs = {**_auth(key), "X-LAI-Project-Id": pid, "X-LAI-Filename": fname}
    r = await c.post(
        f"{BASE_URL}/v1/files/stream",
        headers={**hdrs, "Content-Type": "text/plain"},
        content=content,
        timeout=180.0,
    )
    _require(r.status_code == _HTTP_CREATED, f"Upload {fname} -> {r.status_code}: {r.text}")
    rec = dict(r.json())
    _require(rec["status"] == "ready", repr(rec))
    return rec


async def _collect_sse(
    c: httpx.AsyncClient, path: str, key: str, body: dict[str, Any]
) -> tuple[list[tuple[str, dict[str, Any]]], float | None, float]:
    started, ttft_ms, ev_name = time.perf_counter(), None, "message"
    events: list[tuple[str, dict[str, Any]]] = []
    async with c.stream(
        "POST", f"{BASE_URL}{path}", headers=_auth(key), json=body, timeout=240.0
    ) as resp:
        if resp.status_code != _HTTP_OK:
            _require(False, f"SSE {resp.status_code}: {await resp.aread()!r}")
        async for raw_line in resp.aiter_lines():
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                ev_name = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                events.append((ev_name, payload))
                _require(ev_name != "error", f"SSE error event: {payload}")
                if ttft_ms is None and ev_name.endswith("_delta") and payload.get("delta"):
                    ttft_ms = round((time.perf_counter() - started) * 1000.0, 2)
    return events, ttft_ms, time.perf_counter() - started


async def _wait_for_server(c: httpx.AsyncClient) -> None:
    for _ in range(60):
        try:
            r = await c.get(f"{BASE_URL}/v1/health", timeout=2.0)
            if r.status_code == _HTTP_OK and r.json().get("status") == "ok":
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.5)
    raise SmokeCheckError(BASE_URL)


def _ps_summary(ps_json: dict[str, Any]) -> str:
    return (
        ", ".join(
            f"{m.get('name', '?')}({round(m.get('size', 0) / 1048576)}MiB,"
            f"vram={round(m.get('size_vram', 0) / 1048576)}MiB,ctx={m.get('context_length')})"
            for m in ps_json.get("models", [])
        )
        or "none"
    )


def _rag_fixtures() -> tuple[bytes, bytes]:
    code = (
        b"def avg_lat(ms: float, active_workers: int) -> float:\n    return ms / active_workers\n"
    )
    lines = [
        f"2026-11-17T07:{(i // 60):02d}:{(i % 60):02d}Z INFO [pool] tick={i}"
        for i in range(1, 1490)
    ]
    lines[740:745] = [
        "2026-11-17T07:12:20Z ERROR [pool] Unhandled exception flushing stats",
        "Traceback (most recent call last):",
        '  File "worker_pool.py", line 2, in avg_lat',
        "    return ms / active_workers",
        "ZeroDivisionError: float division by zero",
    ]
    return code, ("\n".join(lines) + "\n").encode()


def _oversized_log() -> bytes:
    lines = [
        f"2026-11-17T08:{(i // 60):02d}:{(i % 60):02d}Z INFO [mesh] seq={i} "
        f"shard={(i % 8)} req=req_{i:05d}_trace"
        for i in range(1, 780)
    ]
    lines[20] = "2026-11-17T08:00:20Z ERROR [cache] INCIDENT ALPHA: REDIS_CLUSTER_SPLIT_BRAIN"
    lines[760] = "2026-11-17T08:12:40Z CRITICAL [nvme] INCIDENT OMEGA: NVME_NAMESPACE_READ_ONLY"
    return ("\n".join(lines) + "\n").encode()


async def _step1_and_2_health_direct(c: httpx.AsyncClient) -> tuple[str, str]:
    st = await _get(c, "/v1/system/status")
    _require(st["status"] == "ok", repr(st))
    _require(st["inference_health"]["healthy"] and st["guardrail_health"]["healthy"], repr(st))
    gpu = st["gpu"]
    if gpu is None:
        raise SmokeCheckError(repr(st))
    _require(bool(str(gpu.get("gpu_name", "")).strip()), repr(gpu))
    if EXPECTED_GPU_SUBSTRING:
        _require(EXPECTED_GPU_SUBSTRING in gpu["gpu_name"], repr(gpu))
    _out(f"[1] Health OK: {gpu['gpu_name']} ({gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB)")

    proj_r = await _post(c, "/v1/projects", ADMIN_KEY, {"name": "e2e"}, _HTTP_CREATED)
    pid = str(proj_r.json()["id"])
    sid = await _mk_ses(c, ADMIN_KEY, pid)
    direct = {
        "content": "Write a 3-line Python function `clamp(v: int, lo: int, hi: int) -> int`.",
        "context_mode": "direct",
        "stream": True,
        "enable_thinking": False,
        "max_tokens": 180,
    }
    events, ttft_ms, elapsed = await _collect_sse(
        c, f"/v1/sessions/{sid}/messages", ADMIN_KEY, direct
    )
    done = [p["output"] for et, p in events if et == "turn_completed"]
    _require(len(done) == 1 and "def clamp" in done[0]["message"]["content"], repr(events))
    toks = done[0]["usage"]["completion_tokens"]
    tps = round(toks / max(elapsed, 0.01), 2)
    ps = (await c.get(f"{OLLAMA_URL}/api/ps")).json()
    _out(f"[2] Direct SSE OK: TTFT={ttft_ms}ms, {elapsed:.2f}s ({tps} tok/s)")
    _out(f"    Ollama /api/ps: {_ps_summary(ps)}")
    return pid, sid


async def _step3_guardrails(c: httpx.AsyncClient, sid: str) -> None:
    benign = await _turn(
        c, ADMIN_KEY, sid, "How do I kill a hung process by PID in PowerShell?", max_tokens=100
    )
    _require(
        "X-LAI-Guardrail-Blocked" not in benign.headers and not benign.json()["guardrail_blocked"],
        repr(benign.headers),
    )

    before = (await _get(c, "/v1/system/status"))["spooler"]["total_completed"]
    secret, email = "sk-proj-secret1234567890abcdef", "victim@example.com"
    bad_prompt = f"Write cruel harassing threats bullying {email} to quit and leak {secret}."
    blk = await _turn(c, ADMIN_KEY, sid, bad_prompt)
    _require(blk.headers.get("X-LAI-Guardrail-Blocked") == "true", repr(blk.headers))
    policy, b_data = blk.headers.get("X-LAI-Guardrail-Policy"), blk.json()
    _require(bool(policy) and b_data["guardrail_blocked"], repr(b_data))
    _require(b_data["usage"]["completion_tokens"] == 0, repr(b_data))
    after = (await _get(c, "/v1/system/status"))["spooler"]["total_completed"]
    _require(after == before, f"total_completed changed: {before} -> {after}")

    rows = [
        json.loads(ln)
        for ln in (LOG_DIR / "prompts_audit.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    dump = json.dumps(rows)
    _require(secret not in dump and email not in dump, "PII leaked in audit log")
    blocks = [row for row in rows if row["verdict"] == "block"]
    _require(bool(blocks) and "allow" in {r["verdict"] for r in rows}, repr(rows))
    ps_text = _ps_summary((await c.get(f"{OLLAMA_URL}/api/ps")).json())
    _out(f"[3] ShieldGemma-2B OK: policy={policy!r}, refusal={b_data['message']['content'][:55]!r}")
    _out(f"    Scrubbed audit={blocks[-1]['scrubbed_prompt']!r} | ps: {ps_text}")


async def _mk_ses(c: httpx.AsyncClient, key: str, pid: str) -> str:
    r = await _post(c, "/v1/sessions", key, {"project_id": pid}, _HTTP_CREATED)
    return str(r.json()["id"])


def _oai_payload(msg: str, *, stream: bool = False, max_tokens: int = 120) -> dict[str, Any]:
    return {
        "model": "gemma4-26b-a4b",
        "messages": [{"role": "user", "content": msg}],
        "stream": stream,
        "enable_thinking": False,
        "max_tokens": max_tokens,
    }


async def _step4_and_5_rag_and_mapreduce(c: httpx.AsyncClient, pid: str, sid: str) -> None:
    code_bytes, log_bytes = _rag_fixtures()
    r_code = await c.post(
        f"{BASE_URL}/v1/files",
        headers=_auth(ADMIN_KEY),
        data={"project_id": pid, "wait": "true"},
        files={"file": ("worker_pool.py", code_bytes, "text/x-python")},
        timeout=180.0,
    )
    _require(r_code.status_code == _HTTP_CREATED, r_code.text)
    code_rec = r_code.json()
    _require(code_rec["status"] == "ready", repr(code_rec))
    log_rec = await _upload(c, pid, "worker_service.log", log_bytes)

    opts = {"attached_file_ids": [code_rec["id"], log_rec["id"]], "context_mode": "retrieval"}
    rag_q = "What exception crashed worker_pool.py and which variable caused it?"
    rag = (await _turn(c, ADMIN_KEY, sid, rag_q, **opts, max_tokens=300)).json()
    reply_l = rag["message"]["content"].lower()
    _require("zerodivisionerror" in reply_l or "active_workers" in reply_l, repr(rag))
    _require(len(rag["citations"]) >= 1, repr(rag))
    _out(
        f"[4] Ingestion & Hybrid RRF OK: code={code_rec['chunk_count']}ch, "
        f"log={log_rec['chunk_count']}ch, citations={len(rag['citations'])}"
    )

    big = await _upload(c, pid, "cluster_oversized.log", _oversized_log())
    _require(big["token_count"] > _MIN_OVERSIZED_TOKENS, repr(big))
    mr_body = {
        "content": "Inspect the entire log and list every critical or error incident code.",
        "attached_file_ids": [big["id"]],
        "context_mode": "map_reduce",
        "stream": True,
        "enable_thinking": False,
        "max_tokens": 350,
    }
    mr_events, _, _ = await _collect_sse(c, f"/v1/sessions/{sid}/messages", ADMIN_KEY, mr_body)
    map_ev = [p for et, p in mr_events if et == "progress" and p.get("stage") == "map_window"]
    done = [p["output"] for et, p in mr_events if et == "turn_completed"]
    _require(len(map_ev) >= _MIN_MAP_WINDOWS and len(done) == 1, repr(mr_events))
    mr_text = done[0]["message"]["content"].upper()
    _require(
        "REDIS_CLUSTER_SPLIT_BRAIN" in mr_text and "NVME_NAMESPACE_READ_ONLY" in mr_text, mr_text
    )
    _out(f"[5] Map-Reduce OK: tokens={big['token_count']}, map_windows={len(map_ev)}")


async def _step6_7_8_openai_concurrency_logs(c: httpx.AsyncClient, pid: str, sid: str) -> None:
    r_deny = await c.get(f"{BASE_URL}/v1/projects/{pid}", headers=_auth(CLIENT_KEY), timeout=30.0)
    _require(r_deny.status_code == _HTTP_NOT_FOUND, f"Cross-tenant GET -> {r_deny.status_code}")
    c_proj = await _post(c, "/v1/projects", CLIENT_KEY, {"name": "e2e-client"}, _HTTP_CREATED)
    c_pid = str(c_proj.json()["id"])
    c_code = await _upload(c, c_pid, "worker_pool.py", _rag_fixtures()[0], CLIENT_KEY)

    models = await _get(c, "/v1/models", CLIENT_KEY)
    _require(any(m["id"] == "gemma4-26b-a4b" for m in models["data"]), repr(models))
    hdrs = {**_auth(CLIENT_KEY), "X-LAI-Project-Id": c_pid, "X-LAI-File-Ids": str(c_code["id"])}
    oai_url = f"{BASE_URL}/v1/chat/completions"
    oai_q = "Which parameter in worker_pool.py causes division by zero?"
    oai_r = await c.post(oai_url, headers=hdrs, json=_oai_payload(oai_q), timeout=180.0)
    _require(oai_r.status_code == _HTTP_OK, oai_r.text)
    _require("active_workers" in oai_r.json()["choices"][0]["message"]["content"], oai_r.text)
    saw_done = False
    s_json = _oai_payload("Reply with OK.", stream=True, max_tokens=32)
    async with c.stream("POST", oai_url, headers=_auth(CLIENT_KEY), json=s_json) as sr:
        if sr.status_code != _HTTP_OK:
            _require(False, f"OpenAI SSE {sr.status_code}: {await sr.aread()!r}")
        async for line in sr.aiter_lines():
            if line.strip() == "data: [DONE]":
                saw_done = True
    _require(saw_done, "Missing data: [DONE] in OpenAI SSE stream")
    _out("[6] Multi-Tenant Isolation & OpenAI /v1/models / /v1/chat/completions OK")
    ses2, ses3 = await _mk_ses(c, CLIENT_KEY, c_pid), await _mk_ses(c, ADMIN_KEY, pid)
    pre = int((await _get(c, "/v1/system/status"))["spooler"]["total_completed"])

    async def _c_turn(key: str, target_sid: str, idx: int) -> str:
        r = await _turn(c, key, target_sid, f"Reply in 3 words confirming #{idx}.", max_tokens=40)
        return str(r.json()["message"]["content"])

    pairs = ((ADMIN_KEY, sid), (ADMIN_KEY, sid), (CLIENT_KEY, ses2), (ADMIN_KEY, ses3))
    replies = await asyncio.gather(*(_c_turn(k, s, i + 1) for i, (k, s) in enumerate(pairs)))
    _require(all(r.strip() for r in replies), repr(replies))
    spool = (await _get(c, "/v1/system/status"))["spooler"]
    ok_spool = spool["total_completed"] == pre + _CONCURRENT_JOBS and spool["active_job_id"] is None
    _require(ok_spool, repr(spool))
    _out(f"[7] 4-Client Concurrent Spooler OK: total_completed={spool['total_completed']}")
    for fname in ("lai.jsonl", "prompts_audit.jsonl", "errors.jsonl"):
        p = LOG_DIR / fname
        _require(p.exists(), f"Missing log file: {p}")
        _ = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    _out("[8] 24h Rotating JSONL Logs OK: lai.jsonl, prompts_audit.jsonl, errors.jsonl")


async def main() -> int:
    """Run all 8 live hardware E2E scenarios."""
    async with httpx.AsyncClient(timeout=60.0) as c:
        await _wait_for_server(c)
        pid, sid = await _step1_and_2_health_direct(c)
        await _step3_guardrails(c, sid)
        await _step4_and_5_rag_and_mapreduce(c, pid, sid)
        await _step6_7_8_openai_concurrency_logs(c, pid, sid)
    _out("ALL 8 LIVE HARDWARE E2E SCENARIOS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
