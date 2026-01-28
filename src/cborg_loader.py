# cborg_loader.py

import os
from typing import Optional, Any
from langchain_openai import ChatOpenAI

def init_cborg_chat_model(
    model: str = "openai/gpt-4o",
    temperature: Optional[float] = 0.0,
    api_key: Optional[str] = None,
    base_url: str = "https://api.cborg.lbl.gov/v1",
    **kwargs: Any
):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key or os.environ.get("CBORG_API_KEY"),
        **kwargs
    )