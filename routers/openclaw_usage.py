"""
OpenClaw Usage API Router
Parses OpenClaw session JSONL files to expose usage/cost data for frontend dashboards.
Data format mirrors the OpenClaw Control UI "Export JSON" output.
"""

import json
import glob
import os
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

from fastapi import APIRouter, Query

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENCLAW_AGENTS_DIR = os.getenv(
    "OPENCLAW_AGENTS_DIR",
    os.path.expanduser("~/.openclaw/agents"),
)

router = APIRouter(prefix="/openclaw/usage", tags=["openclaw-usage"])


# ---------------------------------------------------------------------------
# Helpers – JSONL parsing
# ---------------------------------------------------------------------------


def _discover_session_files(agent_id: str = "main") -> list[str]:
    """Find all .jsonl session transcript files for an agent."""
    pattern = os.path.join(OPENCLAW_AGENTS_DIR, agent_id, "sessions", "*.jsonl")
    return sorted(glob.glob(pattern))


def _parse_session_file(
    path: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
):
    """
    Parse a single session JSONL file.
    Returns (session_meta, assistant_entries) where each assistant entry
    contains usage, cost, model, provider, timestamp, tool info, and duration.
    """
    session_meta = {}
    entries = []

    with open(path, "r") as f:
        # Keep track of the last user message timestamp to compute latency
        last_user_ts: Optional[float] = None

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = record.get("type")

            if rtype == "session":
                session_meta = {
                    "sessionId": record.get("id"),
                    "sessionFile": os.path.basename(path),
                    "startTimestamp": record.get("timestamp"),
                }
                continue

            if rtype != "message":
                continue

            msg = record.get("message", {})
            role = msg.get("role")
            ts_str = record.get("timestamp") or msg.get("timestamp")

            # Parse timestamp
            ts_epoch_ms: Optional[int] = None
            if isinstance(ts_str, str):
                try:
                    ts_epoch_ms = int(
                        datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() * 1000
                    )
                except Exception:
                    pass
            elif isinstance(ts_str, (int, float)):
                ts_epoch_ms = int(ts_str) if ts_str > 1e12 else int(ts_str * 1000)

            # Filter by time range
            if ts_epoch_ms:
                if start_ms and ts_epoch_ms < start_ms:
                    continue
                if end_ms and ts_epoch_ms > end_ms:
                    continue

            if role == "user":
                last_user_ts = ts_epoch_ms
                continue

            if role != "assistant":
                continue

            usage = msg.get("usage")
            if not usage:
                continue

            # Extract tool calls from content
            tool_names = []
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "toolCall":
                        name = block.get("name")
                        if name:
                            tool_names.append(name)

            # Compute latency (time between user message and assistant response)
            duration_ms: Optional[int] = None
            if last_user_ts and ts_epoch_ms:
                duration_ms = ts_epoch_ms - last_user_ts

            entries.append(
                {
                    "usage": usage,
                    "provider": msg.get("provider"),
                    "model": msg.get("model"),
                    "timestamp": ts_epoch_ms,
                    "stopReason": msg.get("stopReason"),
                    "toolNames": tool_names,
                    "durationMs": duration_ms,
                }
            )
            last_user_ts = None  # Reset after pairing

    return session_meta, entries


def _date_from_ms(epoch_ms: int) -> str:
    """Convert epoch ms to YYYY-MM-DD string (UTC)."""
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _build_session_cost_summary(
    session_meta: dict,
    entries: list,
) -> dict:
    """
    Build a SessionCostSummary-compatible dict from parsed entries.
    Matches the OpenClaw Export JSON schema.
    """
    # Totals
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    total_tokens = 0
    total_cost = 0.0
    input_cost = 0.0
    output_cost = 0.0
    cache_read_cost = 0.0
    cache_write_cost = 0.0
    missing_cost = 0

    # Daily buckets
    daily_usage: dict[str, dict] = defaultdict(
        lambda: {"date": "", "tokens": 0, "cost": 0.0}
    )
    daily_messages: dict[str, dict] = defaultdict(
        lambda: {
            "date": "",
            "total": 0,
            "user": 0,
            "assistant": 0,
            "toolCalls": 0,
            "toolResults": 0,
            "errors": 0,
        }
    )
    daily_latency_buckets: dict[str, list] = defaultdict(list)
    daily_model_usage: dict[str, dict] = defaultdict(
        lambda: {"date": "", "provider": "", "model": "", "tokens": 0, "cost": 0.0, "count": 0}
    )

    # Aggregate tool usage
    tool_counter: dict[str, int] = defaultdict(int)
    total_tool_calls = 0

    # Model usage
    model_usage_map: dict[str, dict] = {}

    # Latency stats
    latencies: list[int] = []

    # Activity tracking
    first_activity: Optional[int] = None
    last_activity: Optional[int] = None
    activity_dates: set[str] = set()

    for entry in entries:
        usage = entry["usage"]
        cost_obj = usage.get("cost", {})
        ts = entry.get("timestamp")

        inp = usage.get("input", 0)
        out = usage.get("output", 0)
        cr = usage.get("cacheRead", 0)
        cw = usage.get("cacheWrite", 0)
        tok = usage.get("totalTokens", 0) or (inp + out + cr + cw)

        total_input += inp
        total_output += out
        total_cache_read += cr
        total_cache_write += cw
        total_tokens += tok

        if cost_obj:
            c_total = cost_obj.get("total", 0) or 0
            total_cost += c_total
            input_cost += cost_obj.get("input", 0) or 0
            output_cost += cost_obj.get("output", 0) or 0
            cache_read_cost += cost_obj.get("cacheRead", 0) or 0
            cache_write_cost += cost_obj.get("cacheWrite", 0) or 0
        else:
            missing_cost += 1

        # Tools
        for tn in entry.get("toolNames", []):
            tool_counter[tn] += 1
            total_tool_calls += 1

        # Daily
        if ts:
            date_str = _date_from_ms(ts)
            activity_dates.add(date_str)
            if first_activity is None or ts < first_activity:
                first_activity = ts
            if last_activity is None or ts > last_activity:
                last_activity = ts

            # Daily usage
            d = daily_usage[date_str]
            d["date"] = date_str
            d["tokens"] += tok
            d["cost"] += cost_obj.get("total", 0) or 0

            # Daily messages (each entry = 1 assistant response; assume 1 user msg per response)
            dm = daily_messages[date_str]
            dm["date"] = date_str
            dm["total"] += 2  # user + assistant
            dm["user"] += 1
            dm["assistant"] += 1
            dm["toolCalls"] += len(entry.get("toolNames", []))

            # Daily latency
            dur = entry.get("durationMs")
            if dur and dur > 0:
                daily_latency_buckets[date_str].append(dur)
                latencies.append(dur)

            # Daily model usage
            model_key = f"{date_str}|{entry.get('provider', '')}|{entry.get('model', '')}"
            dmu = daily_model_usage[model_key]
            dmu["date"] = date_str
            dmu["provider"] = entry.get("provider", "")
            dmu["model"] = entry.get("model", "")
            dmu["tokens"] += tok
            dmu["cost"] += cost_obj.get("total", 0) or 0
            dmu["count"] += 1

        # Model usage aggregate
        mkey = f"{entry.get('provider', '')}|{entry.get('model', '')}"
        if mkey not in model_usage_map:
            model_usage_map[mkey] = {
                "provider": entry.get("provider"),
                "model": entry.get("model"),
                "count": 0,
                "totals": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "totalTokens": 0,
                    "totalCost": 0,
                    "inputCost": 0,
                    "outputCost": 0,
                    "cacheReadCost": 0,
                    "cacheWriteCost": 0,
                    "missingCostEntries": 0,
                },
            }
        mu = model_usage_map[mkey]
        mu["count"] += 1
        mt = mu["totals"]
        mt["input"] += inp
        mt["output"] += out
        mt["cacheRead"] += cr
        mt["cacheWrite"] += cw
        mt["totalTokens"] += tok
        mt["totalCost"] += cost_obj.get("total", 0) or 0
        mt["inputCost"] += cost_obj.get("input", 0) or 0
        mt["outputCost"] += cost_obj.get("output", 0) or 0
        mt["cacheReadCost"] += cost_obj.get("cacheRead", 0) or 0
        mt["cacheWriteCost"] += cost_obj.get("cacheWrite", 0) or 0

    # Build latency stats helper
    def _latency_stats(vals: list[int]) -> dict:
        if not vals:
            return {"count": 0, "avgMs": 0, "p95Ms": 0, "minMs": 0, "maxMs": 0}
        vals_sorted = sorted(vals)
        p95_idx = max(0, int(len(vals_sorted) * 0.95) - 1)
        return {
            "count": len(vals_sorted),
            "avgMs": round(sum(vals_sorted) / len(vals_sorted)),
            "p95Ms": vals_sorted[p95_idx],
            "minMs": vals_sorted[0],
            "maxMs": vals_sorted[-1],
        }

    # Build daily latency list
    daily_latency_list = []
    for date_str in sorted(daily_latency_buckets.keys()):
        stats = _latency_stats(daily_latency_buckets[date_str])
        stats["date"] = date_str
        daily_latency_list.append(stats)

    # Assemble summary
    summary: dict = {
        "sessionId": session_meta.get("sessionId"),
        "sessionFile": session_meta.get("sessionFile"),
        "firstActivity": first_activity,
        "lastActivity": last_activity,
        "durationMs": (last_activity - first_activity) if first_activity and last_activity else None,
        "activityDates": sorted(activity_dates),
        # Token totals
        "input": total_input,
        "output": total_output,
        "cacheRead": total_cache_read,
        "cacheWrite": total_cache_write,
        "totalTokens": total_tokens,
        # Cost totals
        "totalCost": round(total_cost, 6),
        "inputCost": round(input_cost, 6),
        "outputCost": round(output_cost, 6),
        "cacheReadCost": round(cache_read_cost, 6),
        "cacheWriteCost": round(cache_write_cost, 6),
        "missingCostEntries": missing_cost,
        # Daily breakdowns
        "dailyBreakdown": sorted(daily_usage.values(), key=lambda d: d["date"]),
        "dailyMessageCounts": sorted(daily_messages.values(), key=lambda d: d["date"]),
        "dailyLatency": daily_latency_list,
        "dailyModelUsage": sorted(daily_model_usage.values(), key=lambda d: d["date"]),
        # Message counts
        "messageCounts": {
            "total": sum(d["total"] for d in daily_messages.values()),
            "user": sum(d["user"] for d in daily_messages.values()),
            "assistant": sum(d["assistant"] for d in daily_messages.values()),
            "toolCalls": total_tool_calls,
            "toolResults": total_tool_calls,  # approximate: 1 result per call
            "errors": 0,
        },
        # Tool usage
        "toolUsage": {
            "totalCalls": total_tool_calls,
            "uniqueTools": len(tool_counter),
            "tools": sorted(
                [{"name": k, "count": v} for k, v in tool_counter.items()],
                key=lambda t: -t["count"],
            ),
        },
        # Model usage
        "modelUsage": list(model_usage_map.values()),
        # Latency
        "latency": _latency_stats(latencies),
    }

    return summary


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/analytics")
async def get_usage_analytics(
    agent_id: str = Query("main", description="Agent ID to query"),
    days: Optional[int] = Query(None, description="Limit to last N days"),
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Full Usage Analytics dashboard endpoint.
    Powers: stat cards, Cost & Tokens tab, Messages tab, Performance tab,
    Tool Usage table, and Model Usage table.

    Response shape:
    {
      "stats": { "totalCost", "totalTokens", "totalMessages", "avgLatencyMs" },
      "costAndTokens": [{ "date", "tokens", "cost" }, ...],
      "messages": [{ "date", "total", "user", "assistant", "toolCalls" }, ...],
      "performance": [{ "date", "avgMs", "p95Ms", "minMs", "maxMs" }, ...],
      "toolUsage": { "totalCalls", "uniqueTools", "tools": [{ "name", "count" }] },
      "modelUsage": [{ "model", "provider", "calls", "tokens", "cost" }]
    }
    """
    from datetime import timedelta

    start_ms = None
    end_ms = None

    if start:
        start_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    if end:
        end_ms = int(
            datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000
        ) + 86400_000
    if days and not start_ms:
        now = datetime.now(timezone.utc)
        start_ms = int((now - timedelta(days=days)).timestamp() * 1000)

    session_files = _discover_session_files(agent_id)
    if not session_files:
        return {
            "stats": {"totalCost": 0, "totalTokens": 0, "totalMessages": 0, "avgLatencyMs": 0},
            "costAndTokens": [],
            "messages": [],
            "performance": [],
            "toolUsage": {"totalCalls": 0, "uniqueTools": 0, "tools": []},
            "modelUsage": [],
        }

    all_entries = []
    for sf in session_files:
        _, entries = _parse_session_file(sf, start_ms, end_ms)
        all_entries.extend(entries)

    if not all_entries:
        return {
            "stats": {"totalCost": 0, "totalTokens": 0, "totalMessages": 0, "avgLatencyMs": 0},
            "costAndTokens": [],
            "messages": [],
            "performance": [],
            "toolUsage": {"totalCalls": 0, "uniqueTools": 0, "tools": []},
            "modelUsage": [],
        }

    # ---- Aggregate ----
    total_cost = 0.0
    total_tokens = 0
    total_messages = 0
    latencies: list[int] = []

    # Daily buckets
    daily_cost: dict[str, dict] = defaultdict(lambda: {"date": "", "tokens": 0, "cost": 0.0})
    daily_msgs: dict[str, dict] = defaultdict(
        lambda: {"date": "", "total": 0, "user": 0, "assistant": 0, "toolCalls": 0}
    )
    daily_perf: dict[str, list] = defaultdict(list)

    # Tool usage
    tool_counter: dict[str, int] = defaultdict(int)
    total_tool_calls = 0

    # Model usage
    model_map: dict[str, dict] = {}

    for entry in all_entries:
        usage = entry["usage"]
        cost_val = usage.get("cost", {}).get("total", 0) or 0
        tok = usage.get("totalTokens", 0) or 0

        total_cost += cost_val
        total_tokens += tok
        total_messages += 1

        dur = entry.get("durationMs")
        if dur and dur > 0:
            latencies.append(dur)

        ts = entry.get("timestamp")
        if ts:
            d = _date_from_ms(ts)

            dc = daily_cost[d]
            dc["date"] = d
            dc["tokens"] += tok
            dc["cost"] += cost_val

            dm = daily_msgs[d]
            dm["date"] = d
            dm["total"] += 2  # user + assistant pair
            dm["user"] += 1
            dm["assistant"] += 1
            dm["toolCalls"] += len(entry.get("toolNames", []))

            if dur and dur > 0:
                daily_perf[d].append(dur)

        # Tools
        for tn in entry.get("toolNames", []):
            tool_counter[tn] += 1
            total_tool_calls += 1

        # Models
        mkey = f"{entry.get('provider', '')}|{entry.get('model', '')}"
        if mkey not in model_map:
            model_map[mkey] = {
                "model": entry.get("model", ""),
                "provider": entry.get("provider", ""),
                "calls": 0,
                "tokens": 0,
                "cost": 0.0,
            }
        mm = model_map[mkey]
        mm["calls"] += 1
        mm["tokens"] += tok
        mm["cost"] += cost_val

    # ---- Build sorted daily arrays with zero-fill ----
    # Determine date range to fill
    range_days = days or 5  # default 5 days if not specified
    now = datetime.now(timezone.utc)
    date_range = []
    for i in range(range_days):
        date_range.append((now - timedelta(days=range_days - 1 - i)).strftime("%Y-%m-%d"))

    cost_and_tokens = []
    messages_list = []
    performance_list = []

    for d in date_range:
        # Cost & Tokens
        if d in daily_cost:
            dc = daily_cost[d]
            cost_and_tokens.append({"date": d, "tokens": dc["tokens"], "cost": round(dc["cost"], 6)})
        else:
            cost_and_tokens.append({"date": d, "tokens": 0, "cost": 0})

        # Messages
        if d in daily_msgs:
            dm = daily_msgs[d]
            messages_list.append({
                "date": d,
                "total": dm["total"],
                "user": dm["user"],
                "assistant": dm["assistant"],
                "toolCalls": dm["toolCalls"],
            })
        else:
            messages_list.append({"date": d, "total": 0, "user": 0, "assistant": 0, "toolCalls": 0})

        # Performance
        vals = daily_perf.get(d, [])
        if vals:
            vals_sorted = sorted(vals)
            p95_idx = max(0, int(len(vals_sorted) * 0.95) - 1)
            performance_list.append({
                "date": d,
                "avgMs": round(sum(vals_sorted) / len(vals_sorted)),
                "p95Ms": vals_sorted[p95_idx],
                "minMs": vals_sorted[0],
                "maxMs": vals_sorted[-1],
            })
        else:
            performance_list.append({"date": d, "avgMs": 0, "p95Ms": 0, "minMs": 0, "maxMs": 0})

    # Avg latency for top card
    avg_latency = round(sum(latencies) / len(latencies)) if latencies else 0

    return {
        "stats": {
            "totalCost": round(total_cost, 6),
            "totalTokens": total_tokens,
            "totalMessages": total_messages,
            "avgLatencyMs": avg_latency,
        },
        "costAndTokens": cost_and_tokens,
        "messages": messages_list,
        "performance": performance_list,
        "toolUsage": {
            "totalCalls": total_tool_calls,
            "uniqueTools": len(tool_counter),
            "tools": sorted(
                [{"name": k, "count": v} for k, v in tool_counter.items()],
                key=lambda t: -t["count"],
            ),
        },
        "modelUsage": sorted(
            [
                {
                    "model": m["model"],
                    "provider": m["provider"],
                    "calls": m["calls"],
                    "tokens": m["tokens"],
                    "cost": round(m["cost"], 6),
                }
                for m in model_map.values()
            ],
            key=lambda m: -m["cost"],
        ),
    }


@router.get("/summary/card")
async def get_usage_summary_card(
    agent_id: str = Query("main", description="Agent ID to query"),
    days: int = Query(5, description="Number of days to include (default 5)"),
):
    """
    Lightweight usage summary card — returns only what's needed for the
    "Usage Summary" widget: headline stats + daily cost bars.

    Response shape:
    {
      "days": 5,
      "totalCost": 33.78,
      "totalMessages": 353,
      "totalTokens": 20000000,
      "dailyCost": [
        {"date": "2026-03-13", "cost": 0.12, "tokens": 50000, "messages": 10},
        ...
      ]
    }
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    start_ms = int((now - timedelta(days=days)).timestamp() * 1000)

    session_files = _discover_session_files(agent_id)
    if not session_files:
        return {"days": days, "totalCost": 0, "totalMessages": 0, "totalTokens": 0, "dailyCost": []}

    all_entries = []
    for sf in session_files:
        _, entries = _parse_session_file(sf, start_ms=start_ms)
        all_entries.extend(entries)

    if not all_entries:
        return {"days": days, "totalCost": 0, "totalMessages": 0, "totalTokens": 0, "dailyCost": []}

    # Bucket by date
    daily: dict[str, dict] = defaultdict(lambda: {"date": "", "cost": 0.0, "tokens": 0, "messages": 0})
    total_cost = 0.0
    total_tokens = 0
    total_messages = 0

    for entry in all_entries:
        cost_val = (entry["usage"].get("cost", {}).get("total", 0) or 0)
        tok = entry["usage"].get("totalTokens", 0) or 0
        total_cost += cost_val
        total_tokens += tok
        total_messages += 1  # each entry = 1 assistant response ≈ 1 exchange

        ts = entry.get("timestamp")
        if ts:
            date_str = _date_from_ms(ts)
            d = daily[date_str]
            d["date"] = date_str
            d["cost"] += cost_val
            d["tokens"] += tok
            d["messages"] += 1

    # Fill in missing dates with zeros so the chart has no gaps
    daily_list = []
    for i in range(days):
        date_str = (now - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        if date_str in daily:
            d = daily[date_str]
            daily_list.append({
                "date": d["date"],
                "cost": round(d["cost"], 6),
                "tokens": d["tokens"],
                "messages": d["messages"],
            })
        else:
            daily_list.append({"date": date_str, "cost": 0, "tokens": 0, "messages": 0})

    return {
        "days": days,
        "totalCost": round(total_cost, 6),
        "totalMessages": total_messages,
        "totalTokens": total_tokens,
        "dailyCost": daily_list,
    }


@router.get("/summary")
async def get_usage_summary(
    agent_id: str = Query("main", description="Agent ID to query"),
    days: Optional[int] = Query(None, description="Limit to last N days"),
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Aggregated usage summary across all sessions.
    Returns the same schema as the OpenClaw Control UI "Export JSON" button.

    This is your main endpoint — it has everything:
    totals, daily breakdowns, message counts, tool usage, model usage, and latency.
    """
    start_ms = None
    end_ms = None

    if start:
        start_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    if end:
        end_ms = int(
            datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000
        ) + 86400_000  # include the end date fully

    if days and not start_ms:
        now = datetime.now(timezone.utc)
        start_ms = int((now.timestamp() - days * 86400) * 1000)

    session_files = _discover_session_files(agent_id)
    if not session_files:
        return {"error": "No session files found", "agentId": agent_id}

    # Aggregate across all sessions
    all_entries = []
    session_summaries = []

    for sf in session_files:
        meta, entries = _parse_session_file(sf, start_ms, end_ms)
        if entries:
            summary = _build_session_cost_summary(meta, entries)
            session_summaries.append(summary)
            all_entries.extend(entries)

    if not all_entries:
        return {"error": "No usage data found in the given range", "agentId": agent_id}

    # Build a combined summary from all entries
    combined_meta = {
        "sessionId": "all",
        "sessionFile": f"{len(session_files)} files",
    }
    combined = _build_session_cost_summary(combined_meta, all_entries)
    combined["sessionCount"] = len(session_summaries)
    combined["sessions"] = session_summaries

    return combined


@router.get("/sessions")
async def get_session_list(
    agent_id: str = Query("main", description="Agent ID to query"),
):
    """
    List all discovered sessions with basic metadata.
    Use a session ID from this list to query /sessions/{session_id} for details.
    """
    session_files = _discover_session_files(agent_id)
    sessions = []

    for sf in session_files:
        meta, entries = _parse_session_file(sf)
        if not meta:
            continue

        total_cost = sum(
            (e["usage"].get("cost", {}).get("total", 0) or 0) for e in entries
        )
        total_tokens = sum(
            (e["usage"].get("totalTokens", 0) or 0) for e in entries
        )
        timestamps = [e["timestamp"] for e in entries if e.get("timestamp")]

        sessions.append(
            {
                "sessionId": meta.get("sessionId"),
                "sessionFile": meta.get("sessionFile"),
                "messageCount": len(entries),
                "totalTokens": total_tokens,
                "totalCost": round(total_cost, 6),
                "firstActivity": min(timestamps) if timestamps else None,
                "lastActivity": max(timestamps) if timestamps else None,
            }
        )

    return {"agentId": agent_id, "count": len(sessions), "sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session_usage(
    session_id: str,
    agent_id: str = Query("main", description="Agent ID to query"),
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Detailed usage for a specific session.
    Returns the full SessionCostSummary matching the OpenClaw Export JSON format.
    """
    start_ms = None
    end_ms = None

    if start:
        start_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    if end:
        end_ms = int(
            datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000
        ) + 86400_000

    session_files = _discover_session_files(agent_id)

    for sf in session_files:
        if session_id in os.path.basename(sf):
            meta, entries = _parse_session_file(sf, start_ms, end_ms)
            if not entries:
                return {"error": "No usage data found for this session in the given range"}
            return _build_session_cost_summary(meta, entries)

    return {"error": f"Session {session_id} not found", "agentId": agent_id}
