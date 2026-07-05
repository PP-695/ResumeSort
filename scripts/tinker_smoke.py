"""Tinker connectivity smoke test.

Run: python scripts/tinker_smoke.py

Checks, in order:
  1. Lists currently available Tinker models (live models.json from the docs site).
  2. PRIMARY: chat-completes via the documented OpenAI-compatible endpoint with a
     base-model name and asserts a JSON round-trip (this is the path the app uses).
  3. (Optional, --native) Samples via the native tinker SDK with a locally rendered
     chat template. Requires the `tinker` and `transformers` packages.

Requires TINKER_API_KEY in the environment (or .env).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from resumesort.llm import DEFAULT_MODEL, TINKER_OAI_BASE_URL, TinkerLLM, parse_json_object

MODELS_JSON_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"


def list_models() -> None:
    print(f"== Current Tinker models ({MODELS_JSON_URL}) ==")
    try:
        request = urllib.request.Request(MODELS_JSON_URL, headers={"User-Agent": "grifter-filter-smoke/1.0"})
        with urllib.request.urlopen(request, timeout=15) as resp:
            models = json.load(resp)
    except Exception as exc:
        print(f"  could not fetch models.json: {exc}")
        return
    for model in models:
        note = f"  [{model['note']}]" if model.get("note") else ""
        print(
            f"  {model['tinker_id']:60s} {model.get('type', ''):18s}"
            f" sample={model.get('sample', '?'):8s}{note}"
        )
    print(f"  total: {len(models)}")


def oai_json_roundtrip(model: str) -> bool:
    print(f"\n== OpenAI-compatible endpoint ({TINKER_OAI_BASE_URL}) model={model} ==")
    llm = TinkerLLM(base_model=model, enabled=True)
    start = time.time()
    text = llm.complete('Return exactly this JSON object: {"status": "ok"}', max_tokens=400, temperature=0.0)
    elapsed = time.time() - start
    if llm.status.enabled is False:
        print(f"  FAIL: {llm.status.reason}")
        return False
    print(f"  latency: {elapsed:.1f}s")
    print(f"  response: {text[:200]!r}")
    ok = parse_json_object(text).get("status") == "ok"
    print(f"  JSON round-trip: {'PASS' if ok else 'FAIL'}")
    return ok


def native_sample(model: str) -> None:
    print(f"\n== Native SDK sample from {model} (informational) ==")
    try:
        import tinker
    except ImportError:
        print("  skipped: tinker package not installed (dev-only dependency)")
        return
    try:
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(base_model=model)
        tokenizer = sampling_client.get_tokenizer()
        messages = [
            {"role": "system", "content": "You respond with JSON only."},
            {"role": "user", "content": 'Return exactly this JSON object: {"status": "ok"}'},
        ]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, enable_thinking=False
            )
        except (TypeError, ValueError):
            rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if hasattr(rendered, "get") and "input_ids" in rendered:
            rendered = rendered["input_ids"]
        if rendered and isinstance(rendered[0], (list, tuple)):
            rendered = rendered[0]
        prompt = tinker.types.ModelInput.from_ints(list(rendered))
        params = tinker.types.SamplingParams(max_tokens=400, temperature=0.0)
        result = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
        sequences = getattr(result, "sequences", None) or getattr(result, "samples", [])
        text = tokenizer.decode(sequences[0].tokens, skip_special_tokens=True).strip()
        print(f"  raw response: {text[:200]!r}")
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {str(exc)[:200]}")


CANNED_RESUME = """Jordan Smith
jordan@example.com | github.com/jordansmith | GPA: 3.7/4.0
Software engineer with 4 years of experience

Skills: Python, Rust, FastAPI, Communication

Projects
Built a real-time chat backend with FastAPI and Redis pub/sub, serving 50k daily messages
Proficient in Python, Rust, and MongoDB
"""


def prompt_quality_check(model: str) -> bool:
    """Eyeball harness: extraction exclusions/atomicity + judge rubric behavior."""
    from resumesort.llm import TinkerLLM
    from resumesort.schemas import EvidenceItem

    llm = TinkerLLM(base_model=model, enabled=True)

    print("\n== Extraction quality ==")
    tagged = llm.extract_claims_tagged(CANNED_RESUME, fallback_claims=[], max_claims=8)
    ok = True
    if not tagged:
        print(f"  FAIL: no claims extracted (parse_failures={llm.parse_failures}, error={llm.status.reason!r})")
        ok = False
    for item in tagged:
        print(f"  [{item['kind']:10s}] {item['claim']}")
    joined = " ".join(item["claim"].lower() for item in tagged)
    for junk in ("gpa", "github.com", "years of experience", "@"):
        if junk in joined:
            print(f"  FAIL: junk claim leaked ({junk})")
            ok = False
    compound = [i for i in tagged if i["claim"].count(",") >= 2 and "proficient" in i["claim"].lower()]
    if compound:
        print(f"  FAIL: compound claim not split: {compound[0]['claim']}")
        ok = False

    print("\n== Judge quality ==")
    evidence = [
        EvidenceItem("readme", "chat-backend", "https://github.com/x/chat-backend",
                     "Real-time chat server built with FastAPI. Uses Redis pub/sub for fanout. Includes locust load tests.",
                     {"languages": {"Python": 12000}, "stars": 4}),
        EvidenceItem("languages", "dotfiles", "https://github.com/x/dotfiles", "Shell, Lua",
                     {"languages": {"Shell": 900}, "stars": 0}),
    ]
    verdict = llm.judge_claim("Built a real-time chat backend with FastAPI and Redis", evidence)
    if verdict:
        print(f"  verdict={verdict.verdict} conf={verdict.confidence} src={verdict.evidence_source}")
        print(f"  explanation: {verdict.explanation[:100]}")
        if verdict.verdict != "SUPPORTED" or verdict.evidence_source != evidence[0].path_or_url:
            print("  WARN: expected SUPPORTED citing chat-backend")
        if verdict.confidence == 0.5:
            print("  WARN: default-looking confidence (rubric may be ignored)")
    else:
        print("  FAIL: judge returned None")
        ok = False

    status = llm.status
    print(f"\n  calls={llm.api_calls} ok={llm.api_successes} parse_failures={llm.parse_failures} "
          f"truncations={llm.truncations} json_mode={llm._json_mode}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="also test the native tinker SDK path")
    parser.add_argument("--prompts", action="store_true", help="run the prompt-quality harness")
    args = parser.parse_args()

    if not os.getenv("TINKER_API_KEY"):
        print("TINKER_API_KEY is not set; aborting.")
        return 1

    model = os.getenv("TINKER_MODEL", os.getenv("TINKER_BASE_MODEL", DEFAULT_MODEL))
    if args.prompts:
        return 0 if prompt_quality_check(model) else 1

    list_models()
    ok = oai_json_roundtrip(model)
    if args.native:
        native_sample(model)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
