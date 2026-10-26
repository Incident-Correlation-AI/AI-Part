#!/usr/bin/env python3
"""Deterministic pre-push secret / forbidden-path scan.

There is no external secret scanner on the audit host.  This script is the
documented, reproducible substitute.  It never prints a matched value: only
the path, line, and pattern id.

Checks
------
1. **Forbidden paths** — local secrets, datasets, model binaries, ZIPs,
   prompt archives and virtualenv/cache output must never be staged.
2. **Private key blocks** — PEM/OpenSSH private keys.
3. **Provider token shapes** — AWS, GitHub, Slack, Google, OpenAI, Stripe,
   npm and generic Bearer tokens.
4. **Credential assignments** — ``password = ...`` style values that are not
   obvious placeholders or environment-variable references.
5. **High-entropy tokens** — long mixed-case alphanumeric strings that do not
   look like a known hex hash.  Reported redacted.

Usage::

    python scripts/secret_scan.py --staged
    python scripts/secret_scan.py --paths deploy src scripts tests docs .dockerignore pyproject.toml

Exit code 0 means no findings.  Exit code 1 means at least one finding.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORBIDDEN_PATH_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"(^|/)\.env$", "environment file with runtime secrets"),
    (r"(^|/)\.env\.(?!example$)[^/]+$", "environment file with runtime secrets"),
    (r"(^|/)(id_rsa|id_ed25519|id_ecdsa)(\.|$)", "private SSH key"),
    (r"\.(pem|key|p12|pfx)$", "private key or certificate bundle"),
    (r"\.(joblib|pkl|pickle|pt|pth|onnx|h5|safetensors)$", "model binary"),
    (r"\.(zip|tar|tgz|tar\.gz|7z|rar)$", "archive"),
    (r"(^|/)trained_model_data_from_kaggle(/|$)", "dataset/run archive directory"),
    (r"(^|/)artifacts(/|$)", "local build output / model artifacts"),
    (r"(^|/)\.venv(/|$)", "virtual environment"),
    (r"(^|/)(reports|data)(/|$)", "generated data or reports"),
    (r"(^|/)opencode_prompt\.md$", "prompt archive (include in audit bundle, not source)"),
)

SECRET_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("private_key_block", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ("aws_access_key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("aws_secret_key", r"(?i)\baws_secret_access_key\b\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{30,}"),
    ("github_token", r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ("github_pat", r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    ("slack_token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    ("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ("openai_key", r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    ("stripe_key", r"\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    ("npm_token", r"\bnpm_[A-Za-z0-9]{30,}\b"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ("bearer_token", r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    (
        "credential_assignment",
        r"(?i)\b(password|passwd|secret|api[_-]?key|apikey|access[_-]?token|"
        r"refresh[_-]?token|session[_-]?token|client[_-]?secret|private[_-]?key)"
        r"\b\s*[:=]\s*['\"]?([A-Za-z0-9._~+/=-]{12,})['\"]?",
    ),
)

#: Values that are clearly not real credentials.
PLACEHOLDER_VALUES = frozenset({
    "changeme", "change-me", "example", "placeholder", "your-key-here",
    "your_key_here", "replace-me", "replace_me", "none", "null", "redacted",
    "dummy", "test", "testing", "not-a-secret", "not_a_secret", "xxx",
    "***", "****", "********",
})

_HEX_RE = re.compile(r"^[0-9a-fA-F]{16,}$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-+/=.]{32,128}")
_PATH_OR_URL_RE = re.compile(
    r"^(?:[A-Za-z][A-Za-z0-9+.-]*://|//|\.{0,2}/)"  # scheme, protocol-relative, relative/absolute path
    r"|(?:/|^)[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*"
    r"\.(?:md|txt|py|json|ya?ml|csv|jsonl|zip|tar|gz|joblib|pkl|log|sqlite3?|toml|ini|cfg)$"
)
_CODE_REFERENCE_PREFIXES = (
    "os.environ", "os.getenv", "environ", "getenv", "self.", "cls.",
    "config.", "settings.", "process.env", "import.meta", "none", "null",
    "true", "false", "getpass", "secrets.", "uuid.", "base64.",
)
_SKIP_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".woff",
    ".woff2", ".ttf", ".eot", ".mp4", ".mov", ".mp3", ".sqlite3", ".db",
})


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    pattern: str
    message: str


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().strip("'\"").lower()
    if lowered in PLACEHOLDER_VALUES:
        return True
    if lowered.startswith("$") or lowered.startswith("${"):
        return True
    if lowered.startswith("<") and lowered.endswith(">"):
        return True
    if set(lowered) <= {"*", "x", "-", "_", "."}:
        return True
    if len(lowered) < 12:
        return True
    return False


def _is_code_reference(value: str) -> bool:
    """True for expressions that read a secret rather than embed one."""
    lowered = value.strip().strip("'\"").lower()
    if not lowered:
        return True
    if any(lowered.startswith(prefix) for prefix in _CODE_REFERENCE_PREFIXES):
        return True
    if any(char in value for char in "()[]{}"):
        return True
    return False


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {}
    for char in value:
        counts[char] = counts.get(char, 0) + 1
    total = len(value)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _looks_like_hash(token: str) -> bool:
    return bool(_HEX_RE.fullmatch(token))


def _looks_like_path_or_url(token: str) -> bool:
    """Documentation paths and repository URLs are not credentials."""
    return bool(_PATH_OR_URL_RE.search(token))


def _forbidden_path(path: str) -> Optional[str]:
    normalized = path.replace("\\", "/").lstrip("./")
    for pattern, reason in FORBIDDEN_PATH_PATTERNS:
        if re.search(pattern, normalized):
            return reason
    return None


def _scan_text(path: str, text: str) -> List[Finding]:
    findings: List[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for pattern_id, regex in SECRET_PATTERNS:
            for match in re.finditer(regex, line):
                if pattern_id == "credential_assignment":
                    value = match.group(2)
                    if _is_placeholder(value) or _is_code_reference(value):
                        continue
                findings.append(
                    Finding(
                        path=path,
                        line=number,
                        pattern=pattern_id,
                        message="credential-shaped value (value redacted)",
                    )
                )
        for token in _TOKEN_RE.findall(line):
            candidate = token.split("=", 1)[-1]
            if _looks_like_hash(candidate) or _looks_like_path_or_url(candidate):
                continue
            has_lower = any(c.islower() for c in candidate)
            has_upper = any(c.isupper() for c in candidate)
            has_digit = any(c.isdigit() for c in candidate)
            if not (has_lower and has_upper and has_digit):
                continue
            if _entropy(candidate) < 4.2:
                continue
            findings.append(
                Finding(
                    path=path,
                    line=number,
                    pattern="high_entropy_token",
                    message="high-entropy mixed-case token (value redacted)",
                )
            )
    return findings


def _read_file(path: str) -> Optional[str]:
    extension = os.path.splitext(path)[1].lower()
    if extension in _SKIP_EXTENSIONS:
        return None
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _staged_files() -> List[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _staged_content(path: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "show", f":{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    raw = result.stdout
    if b"\x00" in raw[:4096]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _walk(paths: Iterable[str]) -> List[str]:
    files: List[str] = []
    for raw in paths:
        candidate = raw if os.path.isabs(raw) else os.path.join(REPO_ROOT, raw)
        if os.path.isfile(candidate):
            files.append(os.path.relpath(candidate, REPO_ROOT))
        elif os.path.isdir(candidate):
            for root, directories, names in os.walk(candidate):
                directories[:] = [
                    name for name in directories
                    if name not in {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}
                ]
                for name in names:
                    files.append(os.path.relpath(os.path.join(root, name), REPO_ROOT))
    return sorted(set(files))


def scan(paths: Sequence[str], staged: bool) -> List[Finding]:
    findings: List[Finding] = []
    files = _staged_files() if staged else _walk(paths)
    for path in files:
        reason = _forbidden_path(path)
        if reason:
            findings.append(
                Finding(path, 0, "forbidden_path", f"must not be committed: {reason}")
            )
            continue
        text = _staged_content(path) if staged else _read_file(path)
        if text is None:
            continue
        findings.extend(_scan_text(path, text))
    return findings


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="scan the git index instead of the working tree")
    parser.add_argument("--paths", nargs="*", default=[], help="files or directories to scan")
    args = parser.parse_args(argv)

    if not args.staged and not args.paths:
        parser.error("provide --staged or at least one --paths entry")

    findings = scan(args.paths, args.staged)
    mode = "staged index" if args.staged else "working tree"
    print(f"secret scan ({mode}): {len(findings)} finding(s)")
    for finding in findings:
        location = f"{finding.path}:{finding.line}" if finding.line else finding.path
        print(f"  {location}  [{finding.pattern}] {finding.message}")
    if findings:
        print("SECRET SCAN FAILED: remove or redact the listed items before pushing.")
        return 1
    print("secret scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
