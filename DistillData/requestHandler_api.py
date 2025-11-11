from openai import OpenAI, BadRequestError, RateLimitError, APIError
import json
import time

class InsufficientQuotaError(Exception):
    pass

class RequestHandler:
    client: OpenAI
    model_name: str
    temperature: float = 0
    max_tokens: int = 5  # 5 tokens is perfect for "Yes" or "No"

    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def sendMsg(self, messages: json) -> str:
        try:
            r = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # No 'stop' tokens needed for standard OpenAI models
            )

            raw_content = r.choices[0].message.content.strip().lower()

            print(raw_content)

            # API models are very good at following instructions.
            # This simple parsing should be enough.
            if raw_content.startswith("yes"):
                return "yes"
            elif raw_content.startswith("no"):
                return "no"
            
            # If the model fails the prompt, log it and return "parse_failed"
            print(f"DEBUG: Parse failed. Raw response: '{raw_content}'")
            return "parse_failed"
        
        except RateLimitError as e:
            if "insufficient_quota" in str(e).lower():
                raise InsufficientQuotaError(f"Fatal quota error: {e}")
            else:
                print("Rate limit hit. Sleeping for 60s.")
                time.sleep(60)
                return "rate_limit_sleep"

        except BadRequestError as e:
            body = getattr(getattr(e, "response", None), "text", None)
            print(f"400 Bad Request from API:", body or str(e))
            return "api_error"
        except Exception as e:
            # Handle other errors, including potential rate limits
            print(f"API Error: {e}")
            if "rate limit" in str(e).lower():
                print("Rate limit error. Sleeping for 60s.")
                time.sleep(60)
                return "api_error" # Tell main loop it failed
            return "api_error"