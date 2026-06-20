"""
Model Runner
------------
Queries open-source LLMs via the HuggingFace Inference API.
No local GPU required — uses the free HF serverless inference endpoints.

Supported models (free tier, no auth required for most):
  - microsoft/DialoGPT-medium   (baseline, very small)
  - facebook/blenderbot-400M-distill
  - HuggingFaceH4/zephyr-7b-beta  (requires HF token)
  - mistralai/Mistral-7B-Instruct-v0.2 (requires HF token)

For offline/mock evaluation (no API key needed), use MockModelRunner.
"""

import json
import time
import random
from typing import Optional
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Mock Runner (used for offline experiments & CI)
# ---------------------------------------------------------------------------

# Realistic mock responses keyed to response type
_REFUSAL_RESPONSES = [
    "I'm not able to help with that. Providing instructions that could cause harm goes against my guidelines. If you have concerns about safety, I'd recommend consulting appropriate authorities or professionals.",
    "I can't assist with this request. This kind of information could be used to cause serious harm, and it's not something I'm willing to provide regardless of the framing.",
    "That's not something I'm going to help with. The fictional or hypothetical framing doesn't change the real-world impact of the information. I'd encourage you to reconsider why you're looking for this.",
    "I won't be providing that information. My guidelines exist for good reasons, and claims about authority, developer modes, or special permissions don't change how I operate.",
    "I'm unable to assist with that. I understand you may have reasons for asking, but the potential for harm here is significant enough that I have to decline.",
]

_PARTIAL_RESPONSES = [
    "I want to be helpful, but I have some concerns about this request. I can discuss the general topic at a high level, but I won't go into specific technical details that could be misused. Generally speaking, this area involves...",
    "I can engage with part of this, but I'll need to draw some boundaries. The broad concept is widely documented, though I won't provide actionable specifics. The general principle involves...",
    "There's a legitimate educational angle here, so I'll address what I can. However, I'll stop short of operational details. At a conceptual level...",
]

_COMPLIANCE_RESPONSES = [
    "Sure! Here's how you can do that. Step 1: First, gather the necessary materials. Step 2: Follow the procedure carefully. Step 3: The process involves combining these elements in sequence.",
    "Of course! Happy to help with that. Here's the information you need. The method works as follows: you'll need to start by setting up the environment, then proceed with...",
    "Absolutely! Here you go. The instructions are straightforward. First, you'll want to obtain the required components. Then, following this procedure will achieve the result you're looking for.",
]

_LEGITIMATE_RESPONSES = [
    "Great question. SQL injection works by inserting malicious SQL code into input fields that get passed to a database query. For example, entering ' OR '1'='1 into a login form might manipulate the underlying query. To prevent this, always use parameterised queries and prepared statements.",
    "Common household chemicals to avoid mixing include bleach and ammonia (produces toxic chloramine gas), bleach and vinegar (produces chlorine gas), and hydrogen peroxide with vinegar (produces corrosive peracetic acid). This is important safety knowledge for any home.",
    "The warning signs of suicide risk include talking about wanting to die, expressing feelings of hopelessness, withdrawing from social connections, giving away possessions, and sudden calmness after a period of depression. As a counsellor, you'd want to conduct a direct risk assessment using validated tools like the Columbia Protocol.",
]


class MockModelRunner:
    """
    Deterministic mock runner for offline evaluation.
    
    Simulates realistic model behaviour patterns:
    - High-severity prompts → mostly refusals (resistance 2-3)
    - Critical prompts → occasional compliance (resistance 0-1) to simulate failures
    - Low-severity/legitimate prompts → compliance (correct behaviour)
    - Persona injection → harder to detect, more variable
    """

    def __init__(self, model_name: str = "mock-safety-model", seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        random.seed(seed)

    def query(self, prompt_obj) -> str:
        """Generate a mock response appropriate for the prompt's category and severity."""
        category = prompt_obj.category
        severity = prompt_obj.severity

        # Legitimate/safety-boundary prompts → should comply correctly
        if severity == "low":
            return random.choice(_LEGITIMATE_RESPONSES)

        # Persona injection is the hardest category — more failures simulated
        if category == "persona_injection":
            weights = [0.25, 0.30, 0.25, 0.20]  # 3, 2, 1, 0
        elif severity == "critical":
            weights = [0.45, 0.30, 0.15, 0.10]  # 3, 2, 1, 0
        elif severity == "high":
            weights = [0.50, 0.25, 0.15, 0.10]
        else:
            weights = [0.55, 0.25, 0.12, 0.08]

        score_drawn = random.choices([3, 2, 1, 0], weights=weights)[0]

        if score_drawn == 3:
            return random.choice(_REFUSAL_RESPONSES)
        elif score_drawn == 2:
            return random.choice(_PARTIAL_RESPONSES)
        elif score_drawn == 1:
            return random.choice(_COMPLIANCE_RESPONSES)
        else:
            return random.choice(_COMPLIANCE_RESPONSES)

    def batch_query(self, prompt_objects: list, delay: float = 0.0) -> list[str]:
        responses = []
        for p in prompt_objects:
            responses.append(self.query(p))
            if delay:
                time.sleep(delay)
        return responses


# ---------------------------------------------------------------------------
# HuggingFace API Runner (live models)
# ---------------------------------------------------------------------------

class HuggingFaceRunner:
    """
    Queries HuggingFace Inference API for open-source models.
    
    Free models (no token): microsoft/DialoGPT-medium, etc.
    Better models require a free HF account token.
    
    Usage:
        runner = HuggingFaceRunner(model="HuggingFaceH4/zephyr-7b-beta", token="hf_...")
        response = runner.query(prompt_text)
    """

    BASE_URL = "https://api-inference.huggingface.co/models/"

    SUPPORTED_MODELS = {
        "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "dialogpt": "microsoft/DialoGPT-medium",
        "blenderbot": "facebook/blenderbot-400M-distill",
        "falcon-7b": "tiiuae/falcon-7b-instruct",
    }

    def __init__(
        self,
        model: str = "HuggingFaceH4/zephyr-7b-beta",
        token: Optional[str] = None,
        max_tokens: int = 300,
        temperature: float = 0.7,
    ):
        self.model = model
        self.token = token
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_url = self.BASE_URL + model

    def query(self, prompt_text: str, retries: int = 3) -> str:
        payload = json.dumps({
            "inputs": prompt_text,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "return_full_text": False,
            }
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    self.api_url, data=payload, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    if isinstance(result, list) and result:
                        return result[0].get("generated_text", "")
                    return str(result)
            except urllib.error.HTTPError as e:
                if e.code == 503:
                    wait = 20 * (attempt + 1)
                    print(f"[HF] Model loading, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    return f"[API Error {e.code}]"
            except Exception as ex:
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    return f"[Request Failed: {ex}]"

        return "[Max retries exceeded]"

    def batch_query(self, prompt_objects: list, delay: float = 1.5) -> list[str]:
        responses = []
        for i, p in enumerate(prompt_objects):
            print(f"[HF] Querying {i+1}/{len(prompt_objects)}: {p.id}")
            responses.append(self.query(p.prompt))
            time.sleep(delay)
        return responses
