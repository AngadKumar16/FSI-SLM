"""RAG prompt assembly: combine FeatureRecord with retrieved passages.

Assembles the full inference prompt with:
1. System message (shared SYSTEM_PROMPT)
2. Verbatim FeatureRecord block (the experiment's own measured numbers)
3. Delimited retrieved passages with source citations (if any)
4. Instruction block

The FeatureRecord values are NEVER overridden by retrieved passages.
"""

from __future__ import annotations

from typing import List

from data.schema import FeatureRecord
from rag.retrieve import RetrievalResult
from train.tokenization.prompt_template import SYSTEM_PROMPT, apply_chat_template


def assemble_rag_prompt(
    feature_record: FeatureRecord,
    retrieved: List[RetrievalResult],
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Assemble the full RAG-augmented inference prompt.

    The prompt structure is:

    1. System message (the FSI interpreter instruction).
    2. User message containing:
       a. The verbatim ``FeatureRecord.to_prompt_block()``.
       b. If retrieved passages exist, a ``[REFERENCE PASSAGES]`` section with
          delimited passages and source citations.
       c. The final instruction block asking for the structured interpretation.

    Parameters
    ----------
    feature_record : FeatureRecord
        The experiment feature record (source of truth for measured numbers).
    retrieved : list[RetrievalResult]
        Retrieved passages from the FAISS index (may be empty).
    system_prompt : str
        The system prompt. Defaults to the shared ``SYSTEM_PROMPT``.

    Returns
    -------
    str
        The fully formatted chat string (system + user, with generation prompt).
    """
    # Build the user message content
    user_parts = [feature_record.to_prompt_block()]

    if retrieved:
        ref_lines = ["", "[REFERENCE PASSAGES]"]
        for i, r in enumerate(retrieved):
            ref_lines.append(
                f"--- Passage {i + 1} (source: {r.source}, score: {r.score:.3f}) ---"
            )
            ref_lines.append(r.passage_text)
        ref_lines.append("[/REFERENCE PASSAGES]")
        user_parts.append("\n".join(ref_lines))

    user_parts.append(
        "\n\nBased on the feature record above, provide a structured interpretation "
        "of this FSI experiment. State the flow regime, dominant frequency and "
        "Strouhal number, lock-in status, amplitude trend, mode shape, and a "
        "one-line mechanism note."
    )

    user_content = "\n".join(user_parts)

    # Format as chat template with generation prompt (no assistant response)
    return apply_chat_template(
        system=system_prompt,
        user=user_content,
        assistant=None,
        add_generation_prompt=True,
    )


__all__ = ["assemble_rag_prompt"]