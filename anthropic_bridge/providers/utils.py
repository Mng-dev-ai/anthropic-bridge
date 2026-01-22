def map_reasoning_effort(
    budget_tokens: int | None, model_id: str | None = None
) -> str | None:
    if not budget_tokens:
        return None

    if budget_tokens >= 32000:
        if _model_supports_xhigh(model_id):
            return "xhigh"
        return "high"
    elif budget_tokens >= 15000:
        return "high"
    elif budget_tokens >= 10000:
        return "medium"
    elif budget_tokens >= 1:
        return "low"

    return None


def _model_supports_xhigh(model_id: str | None) -> bool:
    if not model_id:
        return False

    lower = model_id.lower()
    if "openai/" in lower:
        lower = lower.split("openai/", 1)[1]

    # OpenAI docs: xhigh is supported for gpt-5.1-codex-max and for models
    # after gpt-5.1-codex-max (e.g., gpt-5.2 and gpt-5.2-codex variants).
    return lower.startswith(("gpt-5.1-codex-max", "gpt-5.2"))
