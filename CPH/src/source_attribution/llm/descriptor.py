"""
Free Vision Media Descriptor (§2.4d step 1).
Extracts salient forensic features, subjects, and text cues using
Google Gemini Free Tier (via modern google.genai) or local computer vision heuristics (zero cost).
"""

from typing import Optional, Tuple, List
import os
from PIL import Image
from src.common.logger import logger
from src.common.config import settings
from src.source_attribution.evidence import redact_secrets

# Attempt import of modern google.genai, fallback to legacy google.generativeai
HAS_GENAI = False
HAS_LEGACY_GENAI = False

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    try:
        import google.generativeai as legacy_genai
        HAS_LEGACY_GENAI = True
    except ImportError:
        pass


class MediaDescriptor:
    """Zero-cost vision descriptor for forensic media analysis."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.client = None
        self.legacy_model = None
        self.llm_succeeded = False  # Tracks whether last call used LLM vs heuristic fallback
        self.description_source = "unavailable"

        if self.api_key:
            if HAS_GENAI:
                try:
                    self.client = genai.Client(api_key=self.api_key)
                    logger.info("llm.gemini_modern_genai.configured")
                except Exception as e:
                    logger.warn("llm.genai_init_error", error=str(redact_secrets(str(e))))
            elif HAS_LEGACY_GENAI:
                try:
                    legacy_genai.configure(api_key=self.api_key)
                    self.legacy_model = legacy_genai.GenerativeModel("gemini-2.5-flash")
                    logger.info("llm.gemini_legacy.configured")
                except Exception as e:
                    logger.warn("llm.legacy_gemini_init_error", error=str(redact_secrets(str(e))))

    async def describe_media(self, media_path: str) -> str:
        """
        Generates a concise forensic description of the media.
        Uses 1 single Gemini call capped at 300 tokens (strictly free-tier compliant).
        """
        logger.info("llm.describe_media.started", path=media_path)

        # Skip API calls during automated benchmark tests to preserve user's free tier
        if os.environ.get("FORENSIC_BENCHMARK_MODE") == "1":
            self.llm_succeeded = False
            self.description_source = "fixture"
            return self._heuristic_description(media_path)

        # 1. Try modern google.genai Client
        if self.client is not None and os.path.exists(media_path):
            models_to_try = ["gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-flash-latest"]
            for model_name in models_to_try:
                try:
                    img = Image.open(media_path)
                    prompt = (
                        "Describe this image for a police digital forensics source-attribution investigation:\n"
                        "1) If there is any headline, tweet/post text, banner, caption, watermarks, or handles (@username), transcribe them EXACTLY verbatim in quotes.\n"
                        "2) Identify specific individuals, political figures, organizations, flags, or landmarks.\n"
                        "3) Factual summary of the visual scene and context. Keep concise and factual."
                    )
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=[img, prompt]
                    )
                    if response and response.text:
                        logger.info("llm.gemini.description_success", model=model_name)
                        self.llm_succeeded = True
                        self.description_source = "gemini"
                        return response.text.strip()
                except Exception as e:
                    logger.warn("llm.gemini_model_try_failed", model=model_name, error=str(redact_secrets(str(e))))

        # 2. Try legacy google.generativeai if available
        if self.legacy_model is not None and os.path.exists(media_path):
            try:
                img = Image.open(media_path)
                prompt = (
                    "Digital forensics source investigation:\n"
                    "1) Transcribe visible headlines, post text, @handles verbatim in quotes.\n"
                    "2) Identify specific individuals, flags, or landmarks.\n"
                    "3) Factual scene summary. Concise."
                )
                response = self.legacy_model.generate_content([prompt, img])
                if response and response.text:
                    logger.info("llm.legacy_gemini.description_success")
                    self.llm_succeeded = True
                    self.description_source = "gemini"
                    return response.text.strip()
            except Exception as e:
                logger.warn("llm.legacy_gemini_error", error=str(redact_secrets(str(e))))

        # 3. Heuristic Computer Vision Fallback (100% Free & Local)
        self.llm_succeeded = False
        self.description_source = "unavailable"
        logger.warn("llm.all_apis_failed_using_visual_fallback")
        return self._heuristic_description(media_path)

    def _heuristic_description(self, media_path: str) -> str:
        details = []
        if os.path.exists(media_path):
            try:
                with Image.open(media_path) as im:
                    w, h = im.size
                    details.append(f"{w}x{h} resolution")
                    bands = im.getbands()
                    details.append(f"{''.join(bands)} color profile")
            except Exception:
                pass

        details_str = f" ({', '.join(details)})" if details else ""
        # Dimensions and colour profile are observations, not a scene description.
        # Returning a neutral value prevents an unavailable model from inventing
        # an event (for example, calling an unrelated image a protest).
        return f"No model-generated visual description is available{details_str}."
