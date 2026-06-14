"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

import json
from datetime import datetime, timezone
from typing import cast

import requests
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text, load_chat_model


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


_IMAGE_REQUEST_KEYWORDS = (
    "图片",
    "照片",
    "出图",
    "生成图",
    "生成图片",
    "生成一张",
    "画一张",
    "画个",
    "绘制",
    "图像",
    "photorealistic",
    "image",
    "picture",
    "draw",
    "photo",
    "portrait",
    "illustration",
)


def _is_image_request(text: str) -> bool:
    """Heuristically detect whether the user is asking for an image."""
    normalized = text.strip().lower()
    return any(keyword in text for keyword in _IMAGE_REQUEST_KEYWORDS) or any(
        keyword in normalized for keyword in ("photo", "portrait", "illustration")
    )


def _build_image_prompt(user_text: str) -> str:
    """Build a ready-to-use prompt for an image generation model."""
    base = user_text.strip()
    return (
        "我不能直接在这里输出图片文件，但我已经把你的需求整理成可直接用于出图模型的提示词：\n\n"
        f"{base}\n\n"
        "推荐增强词：写实摄影、高清细节、自然光、真实毛发纹理、浅景深、"
        "35mm镜头、ultra realistic、photorealistic、high detail、natural colors。\n"
        "负面提示词：模糊、低清晰度、畸形、多余肢体、文字、水印、噪点。\n\n"
        "如果你愿意，我还可以继续帮你把它改成 Midjourney / SDXL / Flux 更适配的版本。"
    )


async def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.
    """
    messages = state.messages
    human_input = get_message_text(messages[-1])
    if _is_image_request(human_input):
        return {"queries": [human_input]}

    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        return {"queries": [human_input]}

    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                configuration.query_system_prompt
                + "\n\nReturn ONLY valid JSON in the form: {{\"query\": \"...\"}}",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.query_model)

    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "queries": "\n- ".join(state.queries),
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    raw_content = get_message_text(response)

    query_text = raw_content.strip()
    if query_text.startswith("{"):
        try:
            parsed = json.loads(query_text)
            if isinstance(parsed, dict) and isinstance(parsed.get("query"), str):
                query_text = parsed["query"].strip()
        except json.JSONDecodeError:
            pass

    if not query_text:
        query_text = human_input

    return {"queries": [query_text]}


async def retrieve(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state."""
    if state.messages and _is_image_request(get_message_text(state.messages[-1])):
        return {"retrieved_docs": []}

    try:
        with retrieval.make_retriever(config) as retriever:
            response = await retriever.ainvoke(state.queries[-1], config)
            return {"retrieved_docs": response}
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        # If the embedding backend is temporarily unavailable, keep the chat flow alive
        # and continue without retrieved context instead of crashing the whole run.
        print(f"[retrieval] falling back to empty context because retrieval failed: {exc}")
        return {"retrieved_docs": []}


async def respond(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our agent."""
    last_user_text = get_message_text(state.messages[-1]) if state.messages else ""
    if _is_image_request(last_user_text):
        return {"messages": [AIMessage(content=_build_image_prompt(last_user_text))]}

    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.response_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.response_model)

    retrieved_docs = format_docs(state.retrieved_docs)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    return {"messages": [response]}


builder = StateGraph(State, input_schema=InputState, context_schema=Configuration)

builder.add_node(generate_query)  # type: ignore[arg-type]
builder.add_node(retrieve)  # type: ignore[arg-type]
builder.add_node(respond)  # type: ignore[arg-type]
builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "respond")

graph = builder.compile(
    interrupt_before=[],
    interrupt_after=[],
)
graph.name = "RetrievalGraph"
