# File: openai_compatible_engine.py
#
# Engine adapter for any OpenAI-compatible chat-completions endpoint
# (OpenAI, Gemini's OpenAI-compatible API, OpenRouter, a local vLLM server, ...).
# It exposes the same two calls as anthropic_engine.py, so the agent is unchanged.
# Select it with QSA_ENGINE=openai (see env_template.txt).

import base64
import os
import sys
from mimetypes import guess_type

from dotenv import load_dotenv
from openai import APIError, OpenAI

load_dotenv(override=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # leave unset for api.openai.com

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is not set (needed for QSA_ENGINE=openai).")
    print("Set it in .env, together with OPENAI_BASE_URL for non-OpenAI endpoints.")
    sys.exit(1)

# Text model for reasoning; vision model for plot analysis (must accept images)
DEFAULT_MODEL = os.environ.get("QSA_TEXT_MODEL", "gpt-4o")
VISION_MODEL = os.environ.get("QSA_VISION_MODEL", DEFAULT_MODEL)

_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def _complete(model, messages, max_tokens, temperature):
    """Streamed chat completion (some endpoints only accept streaming).

    Retries with the newer token parameter, or without temperature, for models that reject them.
    """
    params = {"max_tokens": max_tokens, "temperature": temperature}
    for _ in range(3):
        try:
            stream = _client.chat.completions.create(model=model, messages=messages, stream=True, **params)
            return "".join(
                chunk.choices[0].delta.content or ""
                for chunk in stream
                if chunk.choices and chunk.choices[0].delta
            )
        except APIError as e:  # also raised mid-stream, e.g. "`temperature` is deprecated for this model"
            response = getattr(e, "response", None)
            detail = (str(e) + str(getattr(e, "body", "")) + (response.text if response is not None else "")).lower()
            if "max_tokens" in detail and "max_tokens" in params:
                params["max_completion_tokens"] = params.pop("max_tokens")
            elif "temperature" in detail and "temperature" in params:
                params.pop("temperature")
            else:
                raise
    raise RuntimeError("chat completion failed after parameter fallbacks")


def call_llm(
    user_prompt: str,
    system_message: str = "You are a helpful assistant to atomic physicist.",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> str:
    """Sends a system message and a user message; returns the text reply."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    return _complete(model, messages, max_tokens, temperature)


def call_vision(image_path: str, additional_context: str = "Describe this image.") -> str:
    """Sends a saved plot together with the conversation context to the vision model."""
    with open(image_path, "rb") as img_file:
        encoded_data = base64.standard_b64encode(img_file.read()).decode("utf-8")
    media_type, _ = guess_type(image_path)
    if not media_type:
        media_type = "image/jpeg"  # default to jpeg if unknown

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded_data}"}},
                {"type": "text", "text": additional_context},
            ],
        }
    ]
    return _complete(VISION_MODEL, messages, 1024, 0.7)
