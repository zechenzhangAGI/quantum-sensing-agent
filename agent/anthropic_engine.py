# File: anthropic_engine.py

import anthropic
import base64
from mimetypes import guess_type
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='C:\\Users\\NVAFM_6th_fl_2\\quantum-sensing-agent 0614\\quantum-sensing-agent\\agent\\.env',override="True")

# Get Anthropic API key with better error handling
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "YOUR_ANTHROPIC_API_KEY_HERE":
    print("ERROR: Anthropic API key not found or not set!")
    print("Please create a .env file in the project root with your API key:")
    print("1. Copy env_template.txt to .env")
    print("2. Replace YOUR_ANTHROPIC_API_KEY_HERE with your actual Anthropic API key")
    print("3. The .env file is already in .gitignore so it won't be committed")
    sys.exit(1)

# Model configuration - can be overridden via environment variables
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "claude-3-5-sonnet-20241022")
sonnet_new = os.environ.get("SONNET_NEW", "claude-3-7-sonnet-20250219")

def call_llm(
    user_prompt: str,
    system_message: str = "You are a helpful assistant to atomic physicist.",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> str:
    """
    Calls Anthropic's API with a 'system' message and a user message.

    Args:
      user_prompt: The text from the user (the question, command, etc.).
      system_message: The 'system' or 'instruction' message (like "You are an advanced AI...").
      model: Which Claude model to use.
      max_tokens: Token limit for the response.
      temperature: Sampling temperature, 0.0 for deterministic.

    Returns:
      The LLM's text completion/response as a Python string.
    """

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # The method name `client.messages.create(...)` and the parameter structure 
    # may differ depending on your library version. 
    # The snippet you provided looks like a newer/beta interface.
    # We'll mirror your snippet as closely as possible:

    system_content_blocks = [
        {
            "type": "text",
            "text": system_message, # The original system_message string
            "cache_control": {"type": "ephemeral"}
        }
    ]

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        # The snippet shows a "system" parameter for the system message:
        system=system_content_blocks,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    )

    # The returned object presumably has a 'content' attribute for the text:
    return response.content[0].text


def call_vision(image_path: str, additional_context: str = "Describe this image.") -> str:
    """
    Reads an image, encodes it in base64, determines its media type,
    and calls Anthropic's vision interface with an additional text prompt that includes conversation context.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
    encoded_data = base64.standard_b64encode(image_data).decode('utf-8')
    media_type, _ = guess_type(image_path)
    if not media_type:
        media_type = "image/jpeg"  # default to jpeg if unknown

    # Build a composite message that includes both the image and the additional context.
    message = client.messages.create(
        model=sonnet_new,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": additional_context
                    }
                ],
            }
        ],
    )
    return message.content[0].text
