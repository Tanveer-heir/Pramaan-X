"""
Event Context Extraction Engine (§2.4).
Extracts verified real-world incident entities, dates, and event profiles
from high-confidence reverse visual search matches.
"""

from typing import List, Dict, Any, Optional
import re
from urllib.parse import urlparse, unquote
from collections import Counter
from src.common.logger import logger


class EventContextExtractor:
    """
    Extracts ground-truth event context from reverse visual search headlines and articles.
    Enables deep contextual pivoting from generic visual description to concrete historical incidents.
    """

    KNOWN_EVENT_PATTERNS = [
        {
            "id": "kisan_andolan_red_fort_2021",
            "name": "Kisan Andolan Red Fort Flag Hoisting",
            "date": "2021-01-26",
            "keywords": ["red fort", "nishan sahib", "kisan", "farmer", "republic day", "tractor rally", "lal qila"],
            "summary": "26 January 2021 Republic Day tractor rally during Kisan Andolan where Nishan Sahib flag was hoisted at Red Fort Delhi"
        },
        {
            "id": "caa_protest_shaheen_bagh_2020",
            "name": "Shaheen Bagh Anti-CAA Protest",
            "date": "2020-01-15",
            "keywords": ["shaheen bagh", "caa", "nrc", "jamia"],
            "summary": "Anti-CAA demonstration and sit-in protest at Shaheen Bagh New Delhi"
        },
        {
            "id": "delhi_riots_2020",
            "name": "North East Delhi Riots",
            "date": "2020-02-24",
            "keywords": ["jafrabad", "maujpur", "northeast delhi", "delhi riots"],
            "summary": "North East Delhi communal clashes and unrest in February 2020"
        }
    ]

    @classmethod
    def extract_context(
        cls,
        visual_hits: List[Dict[str, Any]],
        initial_description: str = "",
        min_visual_similarity: float = 0.60
    ) -> Dict[str, Any]:
        """
        Analyzes discovered visual search matches and articles to extract
        the verified historical event, location, date, and entities.
        """
        logger.info("context_extractor.start", num_hits=len(visual_hits))

        # 1. Aggregate textual evidence from visual match URLs, titles, and snippets
        text_corpus = []
        for hit in visual_hits:
            title = hit.get("title", "")
            snippet = hit.get("text", "")
            url = hit.get("post_url") or hit.get("url", "")
            
            # Clean URL slugs (e.g. report-nishan-sahib-flag-hoisted-at-red-fort-delhi)
            parsed_url = urlparse(url)
            slug = unquote(parsed_url.path).replace("-", " ").replace("_", " ").replace("/", " ")
            
            combined = f"{title} {snippet} {slug}"
            text_corpus.append(combined.lower())

        corpus_all = " ".join(text_corpus) + " " + initial_description.lower()

        # 2. Check for matching known major forensic event patterns (with false-positive protection)
        best_event = None
        best_score = 0
        desc_lower = initial_description.lower()
        for pattern in cls.KNOWN_EVENT_PATTERNS:
            kw_hits = [kw for kw in pattern["keywords"] if kw in corpus_all]
            in_desc_count = sum(1 for kw in pattern["keywords"] if kw in desc_lower)
            # Require at least 3 keywords across corpus OR 2 directly in user's media description
            if (len(kw_hits) >= 3 or in_desc_count >= 2) and len(kw_hits) > best_score:
                best_score = len(kw_hits)
                best_event = pattern

        # 3. Extract and normalize dates to ISO YYYY-MM-DD
        detected_date = None
        months = {
            "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
            "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
            "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
            "nov": 11, "november": 11, "dec": 12, "december": 12
        }

        m_iso = re.search(r'\b(20[0-2][0-9])[/-](0[1-9]|1[0-2])[/-](0[1-9]|[12][0-9]|3[01])\b', corpus_all)
        if m_iso:
            detected_date = f"{m_iso.group(1)}-{m_iso.group(2)}-{m_iso.group(3)}"
        else:
            m_text_a = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+([0-3]?[0-9]),?\s+(20[0-2][0-9])\b', corpus_all, re.I)
            if m_text_a:
                mo = months[m_text_a.group(1)[:3].lower()]
                day = int(m_text_a.group(2))
                yr = int(m_text_a.group(3))
                detected_date = f"{yr}-{mo:02d}-{day:02d}"
            else:
                m_text_b = re.search(r'\b([0-3]?[0-9])\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(20[0-2][0-9])\b', corpus_all, re.I)
                if m_text_b:
                    day = int(m_text_b.group(1))
                    mo = months[m_text_b.group(2)[:3].lower()]
                    yr = int(m_text_b.group(3))
                    detected_date = f"{yr}-{mo:02d}-{day:02d}"

        # 4. Extract prominent entities & multi-word capitalized phrases
        proper_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', " ".join(h.get("title", "") for h in visual_hits) + " " + initial_description)
        stop_entities = {
            "Report", "Video", "Breaking", "Watch", "News", "India", "Full", "Live", "Exclusive",
            "The", "An", "A", "He", "She", "It", "They", "There", "Here", "His", "Her", "Their",
            "This", "That", "Image", "Photo", "Picture", "Scene", "Delhi", "Source", "Gemini", "Flash"
        }
        filtered_entities = [e for e in proper_entities if e not in stop_entities and len(e) > 2]
        entity_counts = Counter(filtered_entities)
        top_entities = [e for e, count in entity_counts.most_common(6)]

        # 5. Build canonical Event Profile
        if best_event:
            event_date = best_event["date"] or detected_date
            event_name = best_event["name"]
            event_summary = best_event["summary"]
            matched_kws = [k.title() for k in best_event.get("keywords", []) if k in corpus_all]
            entities = list(dict.fromkeys(top_entities + matched_kws))
            confidence = min(0.96, 0.70 + (best_score * 0.05))
            event_detected = True
        else:
            # Dynamic extraction from verbatim headlines or prominent entities
            m_quote = re.search(r'"([^"]{15,120})"', initial_description)
            if m_quote:
                event_name = m_quote.group(1).strip()
                event_summary = event_name
                event_detected = True
                confidence = 0.85
            elif top_entities:
                event_name = " ".join(top_entities[:3])
                event_summary = initial_description[:150] if initial_description else event_name
                event_detected = True
                confidence = 0.75
            else:
                event_name = "Media Attribution Analysis"
                event_summary = initial_description[:120] if initial_description else "Media verification"
                event_detected = False
                confidence = 0.50

            event_date = detected_date
            entities = top_entities

        profile = {
            "event_detected": event_detected,
            "event_name": event_name,
            "event_date": event_date,
            "key_entities": entities[:6],
            "event_summary": event_summary,
            "confidence": round(confidence, 2),
            "evidence_sources_count": len(visual_hits)
        }

        logger.info(
            "context_extractor.complete",
            event=event_name,
            date=event_date,
            confidence=profile["confidence"]
        )
        return profile
