"""Utility functions for the retrieval graph.

This module contains utility functions for handling messages, documents,
and other common operations in project.

Functions:
    get_message_text: Extract text content from various message formats.
    format_docs: Convert documents to an xml-formatted string.
"""
import os
from urllib.parse import urljoin

import requests
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import AnyMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message.

    This function extracts the text content from various message formats.

    Args:
        msg (AnyMessage): The message object to extract text from.

    Returns:
        str: The extracted text content of the message.

    Examples:
        >>> from langchain_core.messages import HumanMessage
        >>> get_message_text(HumanMessage(content="Hello"))
        'Hello'
        >>> get_message_text(HumanMessage(content={"text": "World"}))
        'World'
        >>> get_message_text(HumanMessage(content=[{"text": "Hello"}, " ", {"text": "World"}]))
        'Hello World'
    """
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: list[Document] | None) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def _custom_model_urls(path: str) -> list[str]:
    """Build candidate URLs for the custom OpenAI-compatible endpoint.

    If the configured HTTPS endpoint fails with an SSL handshake error, callers
    can retry the same path over HTTP as a compatibility fallback.
    """
    base_url = os.environ["CUSTOM_MODEL_URL"].rstrip("/")
    candidates = [base_url]
    if base_url.startswith("https://"):
        candidates.append("http://" + base_url.removeprefix("https://"))
    elif base_url.startswith("http://"):
        candidates.append("https://" + base_url.removeprefix("http://"))

    urls: list[str] = []
    for candidate in dict.fromkeys(candidates):
        urls.append(urljoin(candidate.rstrip("/") + "/", path))
    return urls


class CustomChatModel(BaseChatModel):
    model_name: str = "claude-opus-4-6"

    def __init__(self, model_name: str | None = None):
        super().__init__()
        if model_name is not None:
            object.__setattr__(self, "model_name", model_name)

    @property
    def _llm_type(self) -> str:
        return "custom-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        payload_messages = []
        for m in messages:
            role = getattr(m, "type", "user")
            if role == "human":
                role = "user"
            payload_messages.append({"role": role, "content": m.content})

        last_exc: Exception | None = None
        resp = None
        for url in _custom_model_urls("chat/completions"):
            try:
                resp = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {os.environ['CUSTOM_MODEL_API_KEY']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_name,
                        "messages": payload_messages,
                        **({"stop": stop} if stop else {}),
                        **kwargs,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                break
            except requests.exceptions.SSLError as exc:
                last_exc = exc
                continue

        if resp is None:
            assert last_exc is not None
            raise last_exc

        data = resp.json()

        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices:
                choice0 = choices[0]
                if isinstance(choice0, dict):
                    message = choice0.get("message") or {}
                    answer = message.get("content")
                    if answer is not None:
                        return ChatResult(
                            generations=[
                                ChatGeneration(message=AIMessage(content=answer))
                            ]
                        )
            if "output" in data:
                return ChatResult(
                    generations=[
                        ChatGeneration(message=AIMessage(content=data["output"]))
                    ]
                )

        raise ValueError(f"Unexpected chat response format: {data!r}")


def load_chat_model(fully_specified_name: str):
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name

    if provider == "custom":
        return CustomChatModel(model_name=model)

    # If we have a custom OpenAI-compatible endpoint configured, prefer it for
    # provider-less model names so older saved configs keep working.
    if not provider and os.getenv("CUSTOM_MODEL_URL"):
        return CustomChatModel(model_name=model)

    from langchain.chat_models import init_chat_model
    return init_chat_model(model, model_provider=provider)

