import json
import os
from typing import Dict, List, Any
import re
from src.common.config import settings
from src.common.logger import logger


class QueryGenerator:
    """Expands media description into targeted queries adapted to each platform's search conventions."""

    def __init__(self):
        self._genai_client = None
        self._init_client()

    def _init_client(self):
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=api_key)
            except Exception as e:
                logger.debug("query_gen.client_init_failed", error=str(e))

    def generate_queries(self, description: str) -> Dict[str, List[str]]:
        """Generates platform search queries using LLM if available, falling back to dynamic token extraction."""
        logger.info("llm.generate_queries", desc_len=len(description))

        # 1. Attempt dynamic LLM query generation (1 call, ~150 tokens)
        if self._genai_client:
            try:
                prompt = (
                    f"You are an OSINT investigator tracing the earliest origin of this media.\n"
                    f"Media Description: \"{description}\"\n\n"
                    f"Generate 3-4 natural, effective search queries for each platform to find where this specific media first appeared or went viral.\n"
                    f"Rules:\n"
                    f"- Strictly use details, people, and actions described in the text.\n"
                    f"- Do NOT invent words or assumptions not present in the description.\n"
                    f"- Output strictly valid JSON matching this exact structure:\n"
                    f'{{"web": ["q1", "q2", "q3"], "reddit": ["q1", "q2", "q3"], "x": ["q1", "q2", "q3"], "youtube": ["q1", "q2", "q3"], "reverse_image": ["q1", "q2", "q3"]}}\n'
                    f"Do not include markdown or explanations."
                )
                models_to_try = ["gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-flash-latest"]
                resp = None
                for m_name in models_to_try:
                    try:
                        resp = self._genai_client.models.generate_content(
                            model=m_name,
                            contents=prompt
                        )
                        if resp and resp.text:
                            break
                    except Exception as me:
                        logger.debug("llm.query_gen_model_failed", model=m_name, error=str(me))

                if not resp or not resp.text:
                    raise RuntimeError("All Gemini query models failed")
                text = resp.text.strip()
                if "```json" in text:
                    text = text.split("```json")[-1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[-1].split("```")[0].strip()

                queries = json.loads(text)
                if all(k in queries for k in ["web", "reddit", "x", "youtube"]):
                    logger.info("llm.queries_generated_via_llm", web_count=len(queries["web"]))
                    return queries
            except Exception as e:
                logger.warn("llm.query_gen_llm_fallback", error=str(e))

        # 2. Dynamic Fallback: Extract proper nouns and salient tokens (NO hardcoded nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', description)
        stop_proper = {
            "in this", "the", "at least", "several", "consistent", "this outdoor", "scene",
            "an", "a", "he", "she", "it", "they", "there", "here", "his", "her", "their",
            "this", "that", "these", "those", "image", "photo", "picture", "video",
            "background", "foreground", "visible", "features", "showing", "holding", "wearing"
        }
        filtered_proper = [
            p for p in proper_nouns
            if p.lower() not in stop_proper and len(p) > 2
        ]

        words = re.findall(r'\b[a-zA-Z]{4,}\b', description.lower())
        stop_words = {
            "this", "that", "with", "from", "were", "what", "which", "there", "image", "video",
            "showing", "scene", "consistent", "featuring", "visible", "large", "outdoor",
            "holding", "front", "least", "other", "several", "depicted", "wearing", "adorned",
            "resembling", "depicts", "features", "plain", "identifiable", "elaborate", "traditional"
        }
        salient_keywords = [w for w in words if w not in stop_words]

        p_term = " ".join(filtered_proper[:2]) if filtered_proper else " ".join(salient_keywords[:2])
        s_term = " ".join(salient_keywords[:3]) if salient_keywords else p_term

        return {
            "web": [
                f"{p_term} {s_term}".strip(),
                f"{p_term} origin",
                f"{p_term} viral image",
                f"{s_term} first post"
            ],
            "reddit": [
                f"{p_term} {s_term}".strip(),
                f"{p_term} image",
                f"{p_term} discussion",
                f"{s_term}"
            ],
            "x": [
                f"{p_term} {s_term}".strip(),
                f"{p_term} viral",
                f"{p_term} photo",
                f"{s_term}"
            ],
            "youtube": [
                f"{p_term} {s_term}".strip(),
                f"{p_term} video",
                f"{p_term} viral",
                f"{s_term}"
            ],
            "reverse_image": [
                f"{p_term} {s_term}".strip(),
                f"{p_term} photo",
                f"{s_term} picture"
            ],
            "telegram": [
                f"{p_term} {s_term}".strip(),
                f"{p_term} alert"
            ]
        }

    def generate_event_specific_queries(self, event_profile: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generates targeted search queries anchored around a verified historical incident
        and canonical event entities dynamically without hardcoded nouns.
        """
        entities = event_profile.get("key_entities", [])
        primary = " ".join(entities[:3]) if entities else event_profile.get("event_name", "")
        date_str = event_profile.get("event_date") or ""
        event_name = event_profile.get("event_name", primary)

        # Extract 4-digit year for natural search matching (avoid hyphenated dates)
        year_match = re.search(r'\b(20[0-2][0-9])\b', date_str)
        year_str = year_match.group(1) if year_match else ""

        temporal_anchor = f"{primary} {year_str}".strip()

        logger.info(
            "llm.generate_event_queries",
            event_name=event_name,
            anchor=temporal_anchor
        )

        return {
            "web": [
                f"{temporal_anchor} news report",
                f"{event_name} {year_str}".strip(),
                f"{primary} breaking article",
                f"{temporal_anchor} eyewitness footage"
            ],
            "reddit": [
                f"{temporal_anchor}",
                f"{event_name} {year_str}".strip(),
                f"{primary} video",
                f"{primary} discussion"
            ],
            "x": [
                f"{temporal_anchor}",
                f"{event_name} {year_str}".strip(),
                f"{primary} footage",
                f"{primary} first video"
            ],
            "youtube": [
                f"{temporal_anchor} video",
                f"{event_name} {year_str}".strip(),
                f"{primary} raw video",
                f"{primary} ground footage"
            ],
            "reverse_image": [
                f"{temporal_anchor}",
                f"{primary} photo",
                f"{event_name} original picture",
                f"{primary} wire photo"
            ],
            "telegram": [
                f"{temporal_anchor}",
                f"{event_name} alert"
            ]
        }
