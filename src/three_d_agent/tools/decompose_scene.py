"""decompose_scene: LLM-driven user request -> objects + task + constraints."""
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from three_d_agent.agent.json_utils import parse_llm_json
from three_d_agent.agent.prompts import DECOMPOSE_PROMPT, SYSTEM_PROMPT


def decompose_scene(
    user_input: str,
    image_path: Path | None,
    llm: BaseChatModel,
) -> dict[str, Any]:
    image_note = (
        f"An image is attached at {image_path}. Use it to ground object descriptions."
        if image_path
        else "No image attached."
    )
    user_msg = DECOMPOSE_PROMPT.format(user_input=user_input, image_note=image_note)
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                           HumanMessage(content=user_msg)])
    text = response.content if isinstance(response.content, str) else str(response.content)
    result = parse_llm_json(text)
    for key in ("objects", "task", "constraints"):
        if key not in result:
            raise ValueError(f"Missing key {key!r} in LLM response: {result}")
    return result
