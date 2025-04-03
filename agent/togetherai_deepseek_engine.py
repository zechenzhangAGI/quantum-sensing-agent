# File: togetherai_engine.py

from together import Together
import base64
from mimetypes import guess_type

# Replace with your actual Together AI API key
TOGETHER_AI_API_KEY = "9590faae3ecdd6dd4936f410d33f4ed6e6ef08202f9b92109a66cbe34ed2bb4d"


# Choose a default model, e.g., "deepseek-3" or "deepseek-2.5"
DEFAULT_MODEL = "deepseek-ai/DeekSeek-V3"

def call_llm(
    user_prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str:
    """
    Calls Together AI's DeepSeek API with a 'system' message and a user message.

    Args:
      user_prompt: The text from the user (the question, command, etc.).
      system_message: The 'system' or 'instruction' message (like "You are an advanced AI...").
      model: Which DeepSeek model to use.
      max_tokens: Token limit for the response.
      temperature: Sampling temperature, 0.0 for deterministic.

    Returns:
      The LLM's text completion/response as a Python string.
    """

    client = Together(api_key=TOGETHER_AI_API_KEY)

    # The method name `client.messages.create(...)` and the parameter structure 
    # may differ depending on your library version. 
    # We'll mirror your snippet as closely as possible:

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        # The snippet shows a "system" parameter for the system message:
        system=system_message,
        messages=[
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    # The returned object presumably has a 'content' attribute for the text:
    return response['content']


def call_vision(image_path: str, additional_context: str = "Describe this image.") -> str:
    """
    Reads an image, encodes it in base64, determines its media type,
    and calls Together AI's vision interface with an additional text prompt that includes conversation context.
    """
    client = Together(api_key=TOGETHER_AI_API_KEY)
    
    # Open and encode image to base64
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
    encoded_data = base64.b64encode(image_data).decode('utf-8')
    
    media_type, _ = guess_type(image_path)
    if not media_type:
        media_type = "image/jpeg"  # default to jpeg if unknown

    # Build a composite message that includes both the image and the additional context.
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
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
    return response['content']
