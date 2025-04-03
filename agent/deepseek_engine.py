#deepseek API key: sk-420e50c5cfd347a38f3cc53bb41acde0

import openai
import base64
from mimetypes import guess_type

# Replace with your actual DeepSeek API key
DEEPSEEK_API_KEY = "sk-420e50c5cfd347a38f3cc53bb41acde0"

# Configure the OpenAI SDK to use DeepSeek's API endpoint and your API key
openai.api_key = DEEPSEEK_API_KEY
openai.api_base = "https://api.deepseek.com/v1"

# Choose a default model, e.g., "deepseek-chat"
DEFAULT_MODEL = "deepseek-chat"

def call_llm(
    user_prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.7
) -> str:
    """
    Calls DeepSeek's chat API with a system message and a user prompt.

    Args:
      user_prompt: The text input from the user.
      system_message: The system/instruction message.
      model: Which DeepSeek model to use.
      max_tokens: Maximum tokens for the response.
      temperature: Sampling temperature (0.0 for deterministic responses).

    Returns:
      The response text from DeepSeek.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False
    )
    
    # Return the content of the first message in the choices list
    return response.choices[0].message.content


def call_vision(image_path: str, additional_context: str = "Describe this image.") -> str:
    """
    Reads an image file, encodes it in base64, and sends it along with additional text context to DeepSeek.
    
    Note: If DeepSeek provides a separate vision API, refer to its documentation.
          This example embeds a snippet of the base64-encoded image within the prompt.
    
    Args:
      image_path: Path to the image file.
      additional_context: Additional text context for image analysis.
    
    Returns:
      The response text from DeepSeek.
    """
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
    encoded_data = base64.b64encode(image_data).decode('utf-8')
    
    media_type, _ = guess_type(image_path)
    if not media_type:
        media_type = "image/jpeg"  # default to JPEG if unknown

    # Construct a prompt that includes a snippet of the base64-encoded image.
    # (Using only a portion for brevity; adjust according to your needs.)
    prompt = f"{additional_context}\nImage Data (base64 snippet): {encoded_data[:64000]}..."
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes images."},
        {"role": "user", "content": prompt}
    ]
    
    response = openai.ChatCompletion.create(
        model=DEFAULT_MODEL,
        messages=messages,
        max_tokens=8000,
        temperature=0.7,
        stream=False
    )
    
    return response.choices[0].message.content
