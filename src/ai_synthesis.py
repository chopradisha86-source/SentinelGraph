"""
src/ai_synthesis.py

Takes a cluster of related error logs (e.g. the output of
clusterer.cluster_error_logs() for a single cluster_id) and asks a Gemini
model, via the google-genai SDK, to synthesize them into a structured SRE
incident / post-mortem report.

Auth: set GEMINI_API_KEY (or GOOGLE_API_KEY) env var, or pass api_key
explicitly to generate_incident_report().
"""

import logging
import os
from typing import List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"

REQUIRED_OUTPUT_TEMPLATE = """Incident Overview

* Primary Failing Service:
* Impact Severity:
* Root Cause Hypothesis:

Timeline of Events
(Bullet points of failure sequence based on timestamps)

Recommended Remediation
(Step-by-step mitigation actions)"""

SYSTEM_INSTRUCTION = f"""You are an experienced Site Reliability Engineer (SRE) writing a concise
incident summary for an on-call channel. You will be given a cluster of
related error log entries (same or similar error type, possibly spanning
multiple services and traces).

Your job is to synthesize these logs into an incident report that a
human on-call engineer can act on immediately.

You MUST follow this exact output structure, with these exact section
headers, in this exact order. Do not add extra sections, do not add a
title above "Incident Overview", and do not wrap the output in code
fences or add any preamble/postamble text outside the structure:

{REQUIRED_OUTPUT_TEMPLATE}

Formatting rules:
- Under "Incident Overview", fill in each of the three bullet points
  ("Primary Failing Service", "Impact Severity", "Root Cause Hypothesis")
  with a single, specific line of text after the colon.
- "Impact Severity" must be one of: Low, Medium, High, Critical, plus a
  short (<15 word) justification.
- Under "Timeline of Events", produce a chronologically ordered bulleted
  list, one bullet per distinct event/burst, each starting with the
  event's timestamp in the logs, describing what failed and where.
- Under "Recommended Remediation", produce a numbered, step-by-step list
  of concrete mitigation actions an SRE could take right now, ordered
  from most immediate/urgent to longer-term follow-up.
- Base every claim strictly on the provided log data. If the root cause
  is uncertain, say so explicitly rather than inventing a cause.
- Be concise. No filler, no restating these instructions."""


def _format_logs_for_prompt(clustered_logs: List[dict]) -> str:
    """Render a list of log dicts into a compact, model-readable block,
    sorted chronologically."""
    sortable = sorted(clustered_logs, key=lambda e: e.get("timestamp", ""))

    lines = []
    for entry in sortable:
        lines.append(
            "- timestamp={timestamp} | service={service} | status={status} | "
            "trace_id={trace_id} | job_id={job_id} | error_msg=\"{error_msg}\"".format(
                timestamp=entry.get("timestamp", "unknown"),
                service=entry.get("service", "unknown"),
                status=entry.get("status", "unknown"),
                trace_id=entry.get("trace_id", "unknown"),
                job_id=entry.get("job_id", "unknown"),
                error_msg=entry.get("error_msg", ""),
            )
        )
    return "\n".join(lines)


def generate_incident_report(
    clustered_logs: List[dict],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    """
    Generate a structured SRE incident report from a cluster of related
    error logs using the google-genai SDK.

    Parameters
    ----------
    clustered_logs : list of log dicts, each expected to have keys like
        job_id, service, status, error_msg, trace_id, timestamp
        (this is exactly the shape returned by clusterer.cluster_error_logs()
        for a single cluster).
    api_key : Gemini API key. Falls back to GEMINI_API_KEY / GOOGLE_API_KEY
        env vars if not provided.
    model : Gemini model name to use.
    temperature : sampling temperature; kept low by default so the report
        stays factual and consistently formatted.

    Returns
    -------
    str: the model's response text, following the required
    "Incident Overview / Timeline of Events / Recommended Remediation"
    structure.
    """
    if not clustered_logs:
        raise ValueError("clustered_logs must contain at least one log entry.")

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not resolved_key:
        raise ValueError(
            "No API key provided. Pass api_key=... or set the GEMINI_API_KEY "
            "(or GOOGLE_API_KEY) environment variable."
        )

    client = genai.Client(api_key=resolved_key)

    log_block = _format_logs_for_prompt(clustered_logs)
    services_involved = sorted({e.get("service", "unknown") for e in clustered_logs})

    user_prompt = f"""Cluster contains {len(clustered_logs)} related log entries across
services: {", ".join(services_involved)}.

Log entries (chronological):
{log_block}

Write the incident report now, following the required structure exactly."""

    logger.info("Requesting incident report from %s for %d logs.", model, len(clustered_logs))

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=temperature,
        ),
    )

    return response.text
