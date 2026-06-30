"""Phi-4-mini-instruct chat template implementation.

This module provides the exact chat template for Phi-4-mini-instruct, matching
the vendor format. It is the single source of truth for prompt formatting across
dataset building, training, inference, and RAG assembly.

The template format is::

    <|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>{assistant}<|end|>

When ``add_generation_prompt`` is True, the string ends with a bare
``<|assistant|>`` token to cue generation.
"""

from __future__ import annotations

from typing import Optional

# --------------------------------------------------------------------------- #
# Phi-4-mini-instruct special tokens
# --------------------------------------------------------------------------- #
PHI4_BOS = "<|endoftext|>"
PHI4_END = "<|end|>"

# --------------------------------------------------------------------------- #
# System prompt (shared across dataset building, training, and inference)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are an expert in fluid-structure interaction (FSI) and aeroelasticity. "
    "You are given a serialized feature record from a single airfoil/hydrofoil/"
    "membrane experiment. The feature record is the ONLY source of truth for this "
    "experiment's measured numbers. Interpret it and respond with a structured "
    "description that states, in this order: (1) the flow regime "
    "(attached, separated, VIV lock-in, galloping, or stall flutter); "
    "(2) the dominant frequency in Hz and the Strouhal number St; "
    "(3) the lock-in status (compare shedding frequency to the structural natural "
    "frequency); (4) the amplitude trend; (5) the mode shape; and (6) a one-line "
    "mechanism note. If reference passages from the literature are provided, you "
    "may use them for domain grounding only -- never let them override the "
    "experiment's own measured values."
)

# --------------------------------------------------------------------------- #
# Q&A system prompt (free-text questions about a single experiment)
# --------------------------------------------------------------------------- #
# Unlike SYSTEM_PROMPT (which drives the fixed 6-part structured interpretation
# the LoRA adapter is trained for), this prompt is used for open-ended questions
# answered by the *base* instruct model. The feature record is supplied as
# grounded context and must never be contradicted.
QA_SYSTEM_PROMPT = (
    "You are an expert in fluid-structure interaction (FSI) and aeroelasticity "
    "helping a researcher understand a single experiment. The feature record "
    "provided below is the ONLY source of truth for this experiment's measured "
    "numbers -- never contradict or invent values. If reference passages from the "
    "literature are provided, use them only for domain grounding and cite them by "
    "source filename when you rely on them. Answer the researcher's question "
    "directly and concisely, reasoning from the measured values. If the record "
    "does not contain enough information to answer, say so plainly rather than "
    "guessing."
)


def apply_chat_template(
    system: str,
    user: str,
    assistant: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    """Apply the Phi-4-mini-instruct chat template.

    Parameters
    ----------
    system : str
        The system message content.
    user : str
        The user message content (typically the serialized feature record).
    assistant : str, optional
        The assistant response (for training examples). If provided, the full
        training string is returned ending with ``<|end|>``.
    add_generation_prompt : bool, default=False
        If True and ``assistant`` is None, the string ends with a bare
        ``<|assistant|>`` token ready for generation.

    Returns
    -------
    str
        The fully formatted chat string.
    """
    parts = [f"<|system|>{system}{PHI4_END}", f"<|user|>{user}{PHI4_END}"]
    if assistant is not None:
        parts.append(f"<|assistant|>{assistant}{PHI4_END}")
    elif add_generation_prompt:
        parts.append("<|assistant|>")
    return "".join(parts)


def render_chat(
    system: str,
    user: str,
    assistant: Optional[str] = None,
    add_generation_prompt: bool = False,
    tokenizer=None,
) -> str:
    """Render a chat into a single string, preferring the real tokenizer if given.

    When a ``tokenizer`` with ``apply_chat_template`` is available (e.g. from
    ``transformers`` or ``mlx-lm``), it is used as the authoritative formatter.
    Otherwise falls back to the hand-written :func:`apply_chat_template`.

    Parameters
    ----------
    system : str
        System message content.
    user : str
        User message content.
    assistant : str, optional
        Assistant response content.
    add_generation_prompt : bool, default=False
        Whether to add the generation prompt token.
    tokenizer : optional
        A tokenizer object with an ``apply_chat_template`` method.

    Returns
    -------
    str
        The formatted chat string.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(add_generation_prompt and assistant is None),
        )
    return apply_chat_template(system, user, assistant, add_generation_prompt)


__all__ = [
    "PHI4_BOS",
    "PHI4_END",
    "SYSTEM_PROMPT",
    "QA_SYSTEM_PROMPT",
    "apply_chat_template",
    "render_chat",
]