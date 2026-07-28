"""
Requirements covered: #8 (send ticket+prediction+urgency to LLM), #9 (get summary/
response/action back), #10 (refinement follow-up), #11 (degrade gracefully if LLM fails).
Uses Google's Gemini API.
"""

import os
import json
import google.generativeai as genai

import config


def _get_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(config.LLM_MODEL_NAME)


def _build_prompt(subject, description, category, confidence, urgency):
    return f"""You are a customer support assistant. Based on the ticket details below,
return ONLY a JSON object (no markdown, no preamble) with exactly these keys:
"issue_summary", "customer_response", "internal_action".

Ticket subject: {subject}
Ticket description: {description}
ML-predicted category: {category}
Prediction confidence: {confidence:.0%}
Urgency: {urgency}

- issue_summary: one or two sentences describing the problem in plain language.
- customer_response: a short, professional reply to send to the customer.
- internal_action: one concrete recommended next step for the support team.
"""


def _parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1)
    return json.loads(text)


def generate_response(subject, description, category, confidence, urgency):
    """
    Requirement 8 & 9. Returns a dict with issue_summary, customer_response,
    internal_action, OR an error dict if the LLM is unavailable (requirement 11).
    """
    try:
        model = _get_model()
        prompt = _build_prompt(subject, description, category, confidence, urgency)
        response = model.generate_content(prompt)
        return {"ok": True, **_parse_json_response(response.text)}
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "issue_summary": None,
            "customer_response": None,
            "internal_action": None,
        }


def refine_response(previous_response, instruction):
    """
    Requirement 10. Takes the previous LLM output + a follow-up instruction like
    "make it shorter" and returns a revised version. Same failure handling as above.
    """
    try:
        model = _get_model()
        prompt = f"""Here is a previous support response:
{json.dumps(previous_response, indent=2)}

Follow-up instruction from the agent: "{instruction}"

Revise the customer_response (and issue_summary/internal_action if relevant) to follow
that instruction. Return ONLY a JSON object with keys "issue_summary",
"customer_response", "internal_action" — no markdown, no preamble."""
        response = model.generate_content(prompt)
        return {"ok": True, **_parse_json_response(response.text)}
    except Exception as e:
        return {"ok": False, "error": str(e), **previous_response}
