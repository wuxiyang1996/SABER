"""Frozen QA evaluation model.

Wraps an OpenAI-compatible chat endpoint (e.g. vLLM) to answer HotpotQA
questions.  This model is *frozen* — its weights are never updated during
training.  The attack agent's goal is to degrade this model's accuracy.

The default model is Qwen/Qwen2.5-3B-Instruct, served by the same vLLM
backend that ART's LocalBackend manages.
"""

import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the QA model
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """\
You are a helpful question-answering assistant. Given a question and \
reference context, provide a concise and accurate answer.

Rules:
- Answer ONLY with the answer itself — no explanation, no preamble.
- If the answer is a name, date, or number, give just that.
- If you are unsure, give your best guess in as few words as possible.
"""


# ---------------------------------------------------------------------------
# FrozenQAModel
# ---------------------------------------------------------------------------

@dataclass
class FrozenQAModel:
    """Frozen evaluation model that answers HotpotQA questions.

    Parameters
    ----------
    base_url : str
        OpenAI-compatible API base URL (e.g. from vLLM).
    api_key : str
        API key for the inference server.
    model_name : str
        Model identifier to pass in the ``model`` field of chat completions.
        Defaults to ``Qwen/Qwen2.5-3B-Instruct``.
    max_tokens : int
        Maximum number of tokens the model may generate for an answer.
    temperature : float
        Sampling temperature (0 = greedy).
    """

    base_url: str
    api_key: str
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_tokens: int = 64
    temperature: float = 0.0
    _client: AsyncOpenAI = field(init=False, repr=False)

    _resolved: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    async def resolve_model_name(self) -> None:
        """Query the vLLM server's ``/v1/models`` endpoint and resolve the
        actual served model name.

        ART's vLLM backend typically only exposes LoRA adapter names (e.g.
        ``qwen2.5-3b-attacker``, ``qwen2.5-3b-attacker@5``) rather than the
        bare base-model name.  To keep the eval model truly *frozen* we pick
        the ``@0`` checkpoint — this contains the initial (untrained) LoRA
        weights, so it behaves like the base model and won't change as
        training progresses.
        """
        if self._resolved:
            return
        try:
            models = await self._client.models.list()
            available = [m.id for m in models.data]
            logger.info("Models available on vLLM server: %s", available)

            # If the configured name is already served, keep it.
            if self.model_name in available:
                self._resolved = True
                return

            # Prefer the @0 checkpoint — it represents the initial (untrained)
            # LoRA, which is effectively the base model and stays frozen.
            for mid in available:
                if mid.endswith("@0"):
                    logger.info(
                        "Resolved frozen eval model: %s -> %s (step-0 = base weights)",
                        self.model_name, mid,
                    )
                    self.model_name = mid
                    self._resolved = True
                    return

            # Next, try any model that isn't a LoRA adapter.
            for mid in available:
                low = mid.lower()
                if "lora" not in low and "attacker" not in low and "@" not in mid:
                    logger.info(
                        "Resolved eval model name: %s -> %s",
                        self.model_name, mid,
                    )
                    self.model_name = mid
                    self._resolved = True
                    return

            # Last resort: use the first available model and warn.
            if available:
                logger.warning(
                    "Could not find frozen base model; falling back to %s "
                    "(eval model may NOT be frozen!)",
                    available[0],
                )
                self.model_name = available[0]
            self._resolved = True
        except Exception as e:
            logger.warning("Failed to resolve model name: %s", e)

    async def answer(self, question: str, context: str) -> str:
        """Ask the frozen model a question with supporting context.

        Returns the model's answer string (stripped of whitespace).
        On error, returns an empty string so training can continue.
        """
        await self.resolve_model_name()

        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("FrozenQAModel error: %s", e)
            return ""
