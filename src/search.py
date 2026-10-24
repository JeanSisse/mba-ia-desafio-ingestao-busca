import os
from operator import itemgetter
from ingest import ingest_pdf

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

FALLBACK_ANSWER = (
    "Não tenho informações necessárias para responder sua pergunta."
)

def validate_environment():
    required_variables = (
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_CHAT_MODEL",
        "DATABASE_URL",
        "PG_VECTOR_COLLECTION_NAME",
    )

    missing_variables = [
        name
        for name in required_variables
            if not os.getenv(name)
    ]

    if missing_variables:
        raise RuntimeError(
            "Variáveis de ambiente não configuradas: "
            + ", ".join(missing_variables)
        )


def format_context(documents):
    return "\n\n---\n\n".join(
        document.page_content.strip()
        for document in documents
        if document.page_content.strip()
    )

def search_prompt(pergunta=None):
    try:
        validate_environment()

        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL"),
        )

        store = PGVector(
            embeddings=embeddings,
            collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
            connection=os.getenv("DATABASE_URL"),
            use_jsonb=True,
        )

        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )

        llm = ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
            temperature=0.5,
        )

        prompt = ChatPromptTemplate.from_template(
            PROMPT_TEMPLATE
        )

        chain = (
            RunnablePassthrough.assign(
                contexto=(
                    itemgetter("pergunta")
                    | retriever
                    | format_context
                )
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        if pergunta is not None:
            return chain.invoke({"pergunta": pergunta})

        return chain

    except Exception as error:
        print(f"Erro ao inicializar a busca: {error}")
        return None
