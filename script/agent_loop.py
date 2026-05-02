"""Agent loop: multi-model feedback cycle for coding tasks.

Pipeline:
  prompt
    → Accord debate (plan)
    → [loop]
        → Claude executes (writes code)
        → Gemini reviews (critiques diff)
        → Accord debates (code + review)
        → if consensus → OpenCode/DeepSeek V3 runs tests
            → tests pass → DONE
            → tests fail → feed failures as delta → loop again
        → else loop with conflict delta

Usage:
    python -m orchestrator.agent_loop "add X to warp/crates/..." --repo ~/Git/warp
    python -m orchestrator.agent_loop "add X" --repo ~/Git/warp --max-iter 3
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

import httpx

# Load OPENROUTER_API_KEY from Accord's .env if not already in environment
_ACCORD_ENV = Path.home() / "Git/accord/orchestrator/.env"
if "OPENROUTER_API_KEY" not in os.environ and _ACCORD_ENV.exists():
    for _line in _ACCORD_ENV.read_text().splitlines():
        if _line.startswith("OPENROUTER_API_KEY="):
            os.environ["OPENROUTER_API_KEY"] = _line.split("=", 1)[1].strip()
            break

ACCORD_URL = "http://127.0.0.1:7878"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_V3_MODEL = "deepseek/deepseek-chat"   # DeepSeek V3 on OpenRouter
CONFIDENCE_THRESHOLD = 0.85
DEFAULT_MAX_ITER = 5


# ── Accord API ────────────────────────────────────────────────────────────────

async def accord_create_session(client: httpx.AsyncClient, title: str) -> str:
    resp = await client.post(f"{ACCORD_URL}/api/sessions", json={"title": title})
    resp.raise_for_status()
    return resp.json()["id"]


async def accord_run(client: httpx.AsyncClient, session_id: str, prompt: str) -> dict:
    """Run one turn in a session. Returns the SynthAnswer dict."""
    resp = await client.post(
        f"{ACCORD_URL}/api/sessions/{session_id}/run",
        json={"user_prompt": prompt},
        timeout=300.0,
    )
    resp.raise_for_status()
    return resp.json()["assistant_message"]["synthesis"]


# ── CLI agents ────────────────────────────────────────────────────────────────

def _run_cli(args: list[str], *, cwd: Path, input_text: str | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0 and not result.stdout.strip():
        raise RuntimeError(f"{args[0]} failed: {result.stderr[:500]}")
    return result.stdout.strip()


def claude_execute(prompt: str, repo: Path) -> str:
    """Run claude Code CLI in the repo. Returns stdout (plan/summary)."""
    print("  → claude executing...")
    return _run_cli(
        ["claude", "--dangerously-skip-permissions", "-p", prompt, "--output-format", "text"],
        cwd=repo,
    )


def gemini_review(prompt: str, repo: Path) -> str:
    """Run gemini CLI to review. Returns critique text."""
    print("  → gemini reviewing...")
    return _run_cli(["gemini", "-p", prompt], cwd=repo)


def git_diff(repo: Path) -> str:
    """Return staged + unstaged diff against HEAD."""
    return _run_cli(["git", "diff", "HEAD"], cwd=repo)


def run_tests(repo: Path) -> tuple[bool, str]:
    """Run cargo nextest. Returns (passed, output)."""
    print("  → opencode: running tests (deepseek v3)...")
    result = subprocess.run(
        ["cargo", "nextest", "run", "--no-fail-fast", "--workspace",
         "--exclude", "command-signatures-v2"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    return passed, output


async def deepseek_analyze_failures(test_output: str, diff: str, task: str) -> str:
    """Call DeepSeek V3 via OpenRouter to analyze test failures.
    Returns a concrete fix description for Claude to apply."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set — needed for OpenCode/DeepSeek test analysis")

    prompt = (
        f"You are a Rust expert analyzing test failures.\n\n"
        f"Original task: {task}\n\n"
        f"Code changes (git diff):\n```\n{diff[:4000]}\n```\n\n"
        f"Test output:\n```\n{test_output[-4000:]}\n```\n\n"
        "List the exact fixes needed to make the tests pass. "
        "Be specific: file paths, function names, what to change. "
        "No preamble — just the fix list."
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/rlins/accord",
                "X-Title": "Accord agent-loop",
            },
            json={
                "model": DEEPSEEK_V3_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
            },
        )

    if resp.status_code == 429:
        raise RuntimeError("DeepSeek V3 rate limited (429). Retry later.")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── Synthesis helpers ─────────────────────────────────────────────────────────

def synthesis_to_text(synth: dict) -> str:
    parts: list[str] = []
    for s in synth.get("sections", []):
        if s.get("type") == "h3":
            parts.append(f"### {s['text']}")
        elif s.get("type") == "p":
            parts.append(s["text"])
        elif s.get("type") == "ul":
            parts.extend(f"- {it}" for it in s.get("items", []))
    return "\n".join(parts)


def is_done(synth: dict) -> bool:
    return (
        synth.get("confidence", 0) >= CONFIDENCE_THRESHOLD
        and len(synth.get("conflict_points", [])) == 0
    )


def extract_delta(synth: dict) -> str:
    conflicts = synth.get("conflict_points", [])
    if not conflicts:
        return ""
    lines = ["Resolve these conflicts in the implementation:"]
    for c in conflicts:
        lines.append(f"- Issue: {c['claim']}")
        lines.append(f"  Resolution: {c['resolution']}")
    return "\n".join(lines)


# ── Main loop ─────────────────────────────────────────────────────────────────

async def agent_loop(user_prompt: str, repo: Path, max_iter: int) -> None:
    repo = repo.expanduser().resolve()
    if not repo.is_dir():
        print(f"[error] repo not found: {repo}", file=sys.stderr)
        sys.exit(1)

    async with httpx.AsyncClient() as client:
        # Verify Accord is running
        try:
            await client.get(f"{ACCORD_URL}/api/health", timeout=5.0)
        except Exception:
            print(f"[error] Accord not reachable at {ACCORD_URL}. Run: cd ~/Git/accord && ./orchestrator/run.sh", file=sys.stderr)
            sys.exit(1)

        session_id = await accord_create_session(client, f"agent-loop: {user_prompt[:60]}")
        print(f"[session] {session_id}")

        # ── Phase 1: Accord debates the task → plan ───────────────────────
        print("\n[1/1] Accord debate → plan")
        plan_synth = await accord_run(client, session_id, user_prompt)
        plan_text = synthesis_to_text(plan_synth)
        print(f"      confidence={plan_synth['confidence']:.2f}  conflicts={len(plan_synth.get('conflict_points', []))}")

        last_synth: dict = {}

        for iteration in range(1, max_iter + 1):
            print(f"\n[loop {iteration}/{max_iter}]")

            # ── Claude executes ───────────────────────────────────────────
            if iteration == 1:
                claude_prompt = (
                    f"Task: {user_prompt}\n\n"
                    f"Agreed plan:\n{plan_text}\n\n"
                    "Implement the plan. Make the code changes directly in the repo."
                )
            else:
                delta = extract_delta(last_synth)
                claude_prompt = (
                    f"The previous implementation had issues. Fix them:\n\n{delta}\n\n"
                    "Apply the fixes directly in the repo."
                )

            claude_output = claude_execute(claude_prompt, repo)
            diff = git_diff(repo)

            if not diff:
                print("  [warn] no file changes detected after claude run")
                diff = claude_output  # fall back to text output for review

            # ── Gemini reviews ────────────────────────────────────────────
            gemini_prompt = (
                f"Review this implementation for the task: {user_prompt}\n\n"
                f"Git diff:\n```\n{diff[:8000]}\n```\n\n"
                "Be specific. Point out bugs, missing cases, or design issues."
            )
            review = gemini_review(gemini_prompt, repo)

            # ── Accord debates code + review ──────────────────────────────
            print("  → accord debating...")
            accord_prompt = (
                f"Claude implemented the task. Gemini reviewed it.\n\n"
                f"Implementation diff:\n```\n{diff[:4000]}\n```\n\n"
                f"Gemini review:\n{review}\n\n"
                "Does the implementation satisfy the original task? "
                "What conflicts remain? What is the consensus?"
            )
            last_synth = await accord_run(client, session_id, accord_prompt)

            conf = last_synth.get("confidence", 0)
            n_conflicts = len(last_synth.get("conflict_points", []))
            print(f"  confidence={conf:.2f}  conflicts={n_conflicts}")

            if is_done(last_synth):
                # ── OpenCode: run tests via DeepSeek V3 ──────────────────
                passed, test_output = run_tests(repo)
                if passed:
                    print(f"\n✓ Done in {iteration} iteration(s). Consensus + tests green.")
                    print(f"\nFinal synthesis:\n{synthesis_to_text(last_synth)}")
                    return

                print(f"  [tests failed] deepseek v3 analyzing failures...")
                fix_instructions = await deepseek_analyze_failures(test_output, diff, user_prompt)
                print(f"  DeepSeek fix plan:\n    {fix_instructions[:200]}...")

                # Inject test failures as the next iteration's delta
                last_synth = {
                    "confidence": 0.0,
                    "conflict_points": [
                        {
                            "claim": "Tests are failing after implementation",
                            "models": ["opencode/deepseek-v3"],
                            "resolution": fix_instructions,
                        }
                    ],
                    "sections": [],
                }
                # Continue loop — Claude will pick up the fix instructions as delta

        print(f"\n✗ Max iterations ({max_iter}) reached without full consensus.")
        print(f"Last confidence: {last_synth.get('confidence', 0):.2f}")
        remaining = last_synth.get("conflict_points", [])
        if remaining:
            print("Remaining conflicts:")
            for c in remaining:
                print(f"  - {c['claim']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent coding loop via Accord")
    parser.add_argument("prompt", help="The coding task to execute")
    parser.add_argument("--repo", default="~/Git/warp", help="Path to the target repo (default: ~/Git/warp)")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help=f"Max loop iterations (default: {DEFAULT_MAX_ITER})")
    args = parser.parse_args()

    asyncio.run(agent_loop(args.prompt, Path(args.repo), args.max_iter))


if __name__ == "__main__":
    main()
