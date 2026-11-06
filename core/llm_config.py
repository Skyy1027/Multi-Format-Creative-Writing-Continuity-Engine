"""Gemini client configuration with model fallback cascade."""

import os
import time
from typing import List, Type

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

load_dotenv()

# Fallback cascade: start with high-quota flash models, fall back to pro if needed
MODEL_CASCADE: List[str] = [
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-2.5-pro",
]

MAX_RETRIES_PER_MODEL = 4
INITIAL_DELAY = 15          # seconds
MAX_WAIT_PER_MODEL = 240    # 4 minutes max wait on any single model


def get_llm(model_name: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_output_tokens=8192,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def invoke_with_model_cascade(
    schema_class: Type[BaseModel],
    messages: List[BaseMessage],
) -> BaseModel:
    """Invoke structured output with exponential backoff and model fallback."""
    last_exception = None

    for model_idx, model_name in enumerate(MODEL_CASCADE):
        print(f"  [Model: {model_name}] Starting...")
        llm = get_llm(model_name)
        structured_llm = llm.with_structured_output(schema_class)
        total_waited = 0

        for attempt in range(MAX_RETRIES_PER_MODEL + 1):
            try:
                result = structured_llm.invoke(messages)
                if model_idx > 0:
                    print(f"  [Model: {model_name}] Success (after fallback)")
                return result

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                is_parse_error = isinstance(e, OutputParserException)
                is_model_error = "404" in error_str or "NOT_FOUND" in error_str

                # Deprecated or missing model in cascade - jump to next
                if is_model_error:
                    print(f"  [Model: {model_name}] Model unavailable (404). Skipping...")
                    last_exception = e
                    break

                if not is_rate_limit and not is_parse_error:
                    raise

                if is_parse_error:
                    print(f"  [Model: {model_name}] Output parsing error, retrying...")

                last_exception = e

                if attempt >= MAX_RETRIES_PER_MODEL:
                    print(f"  [Model: {model_name}] Retries exhausted.")
                    break

                delay = INITIAL_DELAY * (2 ** attempt)

                if total_waited + delay > MAX_WAIT_PER_MODEL:
                    print(
                        f"  [Model: {model_name}] Wait budget exceeded "
                        f"({total_waited}s / {MAX_WAIT_PER_MODEL}s). Switching model..."
                    )
                    break

                print(
                    f"  [Rate limit: {model_name}] "
                    f"Backoff {delay}s ({attempt + 1}/{MAX_RETRIES_PER_MODEL})..."
                )
                time.sleep(delay)
                total_waited += delay

        if model_idx < len(MODEL_CASCADE) - 1:
            print(f"  → Falling back from {model_name} to {MODEL_CASCADE[model_idx + 1]}")
        else:
            print(f"  [Model: {model_name}] Last model exhausted.")

    raise RuntimeError(
        f"All models in cascade exhausted: {MODEL_CASCADE}. "
        f"Last error: {last_exception}."
    )

