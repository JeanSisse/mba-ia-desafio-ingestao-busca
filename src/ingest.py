import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
PGVECTOR_URL = os.getenv("DATABASE_URL")
PGVECTOR_COLLECTION = os.getenv("PG_VECTOR_COLLECTION_NAME")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

def validate_environment():
    required_variables = {
        "PDF_PATH": PDF_PATH,
        "OPENAI_EMBEDDING_MODEL": OPENAI_EMBEDDING_MODEL,
        "PGVECTOR_URL": PGVECTOR_URL,
        "PGVECTOR_COLLECTION": PGVECTOR_COLLECTION,
    }

    missing_variables = [
        name
        for name, value in required_variables.items()
        if not value
    ]

    if missing_variables:
        raise RuntimeError(
            "Variáveis de ambiente não configuradas: "
            + ", ".join(missing_variables)
        )

def ingest_pdf():
    pdf_path = Path(__file__).resolve().parent.parent / PDF_PATH

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Arquivo PDF não encontrado: {pdf_path.resolve()}"
        )

    documents = PyPDFLoader(str(pdf_path)).load()
    if not documents:
        print("Nenhum conteúdo foi encontrado no PDF.")
        return

    splits = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            add_start_index=False
        ).split_documents(documents)
    if not splits:
        raise SystemExit(0)

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    store = PGVector(
        embeddings=embeddings,
        collection_name=PGVECTOR_COLLECTION,
        connection=PGVECTOR_URL,
        use_jsonb=True,
    )
    store.add_documents(documents=enriched, ids=ids)

def ingest_pdf_with_validation():
    validate_environment()
    ingest_pdf()


if __name__ == "__main__":
    ingest_pdf_with_validation()