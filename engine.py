"""Storage-independent learning, feedback and retrieval."""

import json
import re
from collections import Counter
from datetime import datetime, timezone

REDACTIONS = (
    (r"\[@id:.*?@\]|\[CQ:[^\]]*\]|\[(?:Image|File|Audio|Video):[^\]]*\]", "[附件或提及]"),
    (r"https?://\S+|[\w.+-]+@[\w.-]+\.\w+|\d{5,}", "[已隐去]"),
)


def scrub(text, names=()):
    text = re.sub(r"\s+", " ", text or "").strip()
    if text.startswith(("/", "[", "<")):
        return ""
    for pattern, replacement in REDACTIONS:
        text = re.sub(pattern, replacement, text, flags=re.I)
    for name in sorted(set(names), key=len, reverse=True):
        if len(name) >= 2:
            text = text.replace(name, "某群友")
    return text[:180]


def new_state():
    return {
        "enabled": False, "cursor": 0, "revision": 0, "entries": {},
        "history": [], "recent": [], "last_run": None,
    }


def checkpoint(state):
    state["history"] = (state["history"] + [{
        "revision": state["revision"],
        "entries": json.loads(json.dumps(state["entries"], ensure_ascii=False)),
    }])[-5:]
    state["revision"] += 1


def build_samples(rows):
    names = {name for row in rows for name in (row["sender_name"], row["sender_nickname"]) if name}
    by_platform_id = {}
    platform_counts = Counter(row["message_id"] for row in rows)
    samples = {}
    speakers = {}
    for row in rows:
        row_id = row["id"]
        text = scrub(row["content_text"], names)
        if not text:
            continue
        speaker = speakers.setdefault(row["sender_id"], len(speakers) + 1)
        sample = {"id": row_id, "speaker": speaker, "text": text}
        try:
            reference = str(json.loads(row["ext_data"] or "{}").get("ref_msg_id") or "")
        except (ValueError, AttributeError):
            reference = ""
        if reference in by_platform_id and reference != "0":
            sample["reply_to"] = by_platform_id[reference]
        samples[row_id] = sample
        platform_id = row["message_id"]
        if platform_id and platform_counts[platform_id] == 1:
            by_platform_id[platform_id] = row_id
    return samples


def parse_candidates(raw, samples):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].removesuffix("```").strip()
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list")
    candidates = {}
    for item in data[:18]:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        fields = ("term", "scene", "meaning", "example")
        if kind not in ("term", "style") or any(not isinstance(item.get(field), str) for field in fields):
            continue
        term, scene, meaning, example = (scrub(item[field]) for field in fields)
        if not (2 <= len(term) <= 28 and scene and meaning):
            continue
        source = item.get("evidence")
        if not isinstance(source, list):
            continue
        evidence = sorted({key for key in source[:20] if type(key) is int and key in samples})
        if kind == "term":
            evidence = [key for key in evidence if term.casefold() in samples[key]["text"].casefold()]
        elif not any(samples[key].get("reply_to") in evidence for key in evidence):
            continue
        if not evidence:
            continue
        # Multiple messages from one participant in one batch are one signal.
        speakers = {samples[key]["speaker"] for key in evidence}
        key = f"{kind}/{term.casefold()}/{scene.casefold()}"
        candidates[key] = {
            "kind": kind, "term": term, "scene": scene, "meaning": meaning, "example": example,
            "evidence": evidence, "signals": len(speakers),
        }
    return candidates


def absorb(state, candidates, cursor, auto_accept, threshold):
    checkpoint(state)
    entries = state["entries"]
    for key, incoming in candidates.items():
        entry = entries.get(key)
        if entry is None:
            if len(entries) >= 200:
                continue
            entry = entries[key] = {**incoming, "status": "candidate", "score": 0, "confirmed": False}
        else:
            fresh = set(incoming["evidence"]) - set(entry["evidence"])
            if not fresh:
                continue
            entry["signals"] += min(len(fresh), incoming["signals"])
            entry["evidence"] = sorted(set(entry["evidence"]) | fresh)[-40:]
            if not entry["confirmed"] and incoming["meaning"] != entry["meaning"]:
                entry["alternative"] = incoming["meaning"]
                entry["status"] = "candidate" if entry["status"] != "disabled" else "disabled"
        if (auto_accept and entry["kind"] == "term" and entry["status"] == "candidate"
                and not entry.get("alternative") and entry["signals"] >= threshold):
            entry["status"] = "active"
    state["cursor"] = max(cursor, state["cursor"])
    state["last_run"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


def keywords(text):
    text = text.casefold()
    words = set(re.findall(r"[a-z0-9]{2,}", text))
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        words.update(run[i:i + 2] for i in range(len(run) - 1))
    return words


def rank(state, current, context):
    current = current.casefold()
    context = context.casefold()
    current_words = keywords(current)
    context_words = keywords(context)
    available = []
    for key, entry in state["entries"].items():
        if entry["status"] != "active":
            continue
        term = entry["term"].casefold()
        cues = keywords(entry["scene"])
        overlap = len(cues & current_words)
        nearby = len(cues & context_words)
        recent = state["recent"].count(key)
        if term in current:
            relevance = 8 + 2 * overlap + nearby
        elif entry["kind"] == "term" and term in context and overlap:
            relevance = 3 + overlap
        elif entry["kind"] == "style" and overlap and recent < 2:
            relevance = 2 + 2 * overlap + nearby
        else:
            continue
        priority = relevance + max(-3, min(3, entry["score"])) - (recent if entry["kind"] == "style" else 0)
        if entry["kind"] == "style":
            priority += entry.get("engagement", 0) / 4
        available.append((priority, key))
    chosen = []
    terms = set()
    for _, key in sorted(available, reverse=True):
        term = state["entries"][key]["term"].casefold()
        if term not in terms:
            terms.add(term)
            chosen.append(key)
    return chosen[:3]


def markdown(state):
    lines = ["# 群聊词库", "", "以下用法来自当前群聊，理解时结合语境。"]
    for key in sorted(state["entries"]):
        entry = state["entries"][key]
        if entry["status"] != "active":
            continue
        lines.extend(["", f"## {entry['term']}", "", f"- 场景：{entry['scene']}",
                      f"- 含义或接法：{entry['meaning']}"])
        if entry["example"]:
            lines.append(f"- 例句：{entry['example']}")
    return "\n".join(lines) + "\n"
