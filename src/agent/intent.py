import json
import re
import anthropic

from src.models.schemas import Intent, IntentResult, ChatMessage


_CLASSIFIER_PROMPT = """

"""

def classify_intent(
    client: anthropic.Anthropic,
    user_message: str,
    history: list[ChatMessage]
) -> IntentResult:
    """
    
    """
    history_snippet = ""
    if history:
        recent = history[-3:]
        history_snippet = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in recent
        )
        history_snippet = f"\nRecent conversation:\n{history_snippet}\n"

    response = client.messages.create(
        model="",
        max_tokens=128,
        system=_CLASSIFIER_PROMPT,
        messages=[{
            "role": "user",
            "content": f"{history_snippet} User message: {user_message}"
        }]
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if the model adds them
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
        return IntentResult(
            intent=Intent(data["intent"]),
            confidence=float(data.get("confidence", 0.8)),
            extracted_query=data.get("extracted_query", user_message),
            product_id=data.get("product_id")
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return IntentResult(
            intent=Intent.GENERAL,
            confidence=0.,
            extracted_query=user_message,
            product_id=None
        )
