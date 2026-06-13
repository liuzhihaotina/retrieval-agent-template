"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.

The retrievers support filtering results by user_id to ensure data isolation between users.
"""

import os
from contextlib import contextmanager
from typing import Generator
from urllib.parse import urljoin

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from retrieval_graph.configuration import Configuration, IndexConfiguration

## Encoder constructors

def make_text_encoder(model: str):
    provider, model = model.split("/", maxsplit=1)

    if provider == "custom":
        # Backward-compatible fallback for the old placeholder value.
        if model == "embedding":
            model = "text-embedding-3-small"
        return CustomEmbeddings(model_name=model)

    if provider == "openai":
        return OpenAIEmbeddings(model=model)


## Retriever constructors


@contextmanager
def make_elastic_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    connection_options = {}
    if configuration.retriever_provider == "elastic-local":
        connection_options = {
            "es_user": os.environ["ELASTICSEARCH_USER"],
            "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
        }

    else:
        if os.getenv("ELASTICSEARCH_API_KEY"):
            connection_options = {
                "es_api_key": os.environ["ELASTICSEARCH_API_KEY"]
            }
        else:
            connection_options = {
                "es_user": os.environ["ELASTICSEARCH_USER"],
                "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
            }

    vstore = ElasticsearchStore(
        **connection_options,  # type: ignore
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )

    search_kwargs = configuration.search_kwargs

    search_filter = search_kwargs.setdefault("filter", [])
    search_filter.append({"term": {"metadata.user_id": configuration.user_id}})
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_pinecone_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_pinecone import PineconeVectorStore

    search_kwargs = configuration.search_kwargs

    search_filter = search_kwargs.setdefault("filter", {})
    search_filter.update({"user_id": configuration.user_id})
    vstore = PineconeVectorStore.from_existing_index(
        os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
    )
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_mongodb_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="langgraph_retrieval_agent.default",
        embedding=embedding_model,
    )
    search_kwargs = configuration.search_kwargs
    pre_filter = search_kwargs.setdefault("pre_filter", {})
    pre_filter["user_id"] = {"$eq": configuration.user_id}
    yield vstore.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = IndexConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    user_id = configuration.user_id
    if not user_id:
        raise ValueError("Please provide a valid user_id in the configuration.")
    match configuration.retriever_provider:
        case "elastic" | "elastic-local":
            with make_elastic_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "mongodb":
            with make_mongodb_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(Configuration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )


class CustomEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.base_url = base_url or os.environ["CUSTOM_MODEL_URL"]
        self.api_key = api_key or os.environ["CUSTOM_MODEL_API_KEY"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        url = urljoin(self.base_url.rstrip("/") + "/", "embeddings")
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model_name, "input": text},
            timeout=60,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            if resp.status_code == 404:
                raise RuntimeError(
                    "Embedding request returned 404. "
                    f"Check that '{self.model_name}' exists on {url!r} and that the service supports the /embeddings endpoint."
                ) from exc
            raise
        data = resp.json()

        if isinstance(data, dict):
            if "data" in data and data["data"]:
                embedding = data["data"][0].get("embedding")
                if embedding is not None:
                    return embedding
            if "embedding" in data:
                return data["embedding"]

        raise ValueError(f"Unexpected embedding response format: {data!r}")
