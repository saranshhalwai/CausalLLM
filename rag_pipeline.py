import os
import argparse
import sys
from glob import glob
from dotenv import load_dotenv
import graph_rag
import graph_rag

# Load environment variables
load_dotenv()

# Select LLM
# Try Groq first, then Gemini (Google), then fail.
llm = None
embeddings = None


def init_components():
    global llm, embeddings
    from langchain_huggingface import HuggingFaceEmbeddings

    # 1. Embeddings
    print("Initializing Embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    graph_rag.set_embeddings(embeddings)

    # 2. LLM
    # Priority: Ollama > Groq > Gemini
    if os.getenv("OLLAMA_MODEL"):
        print(f"Using Ollama LLM ({os.getenv('OLLAMA_MODEL')})...")
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)
    elif os.getenv("GROQ_API_KEY"):
        print("Using Groq LLM...")
        from langchain_groq import ChatGroq

        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    elif os.getenv("GOOGLE_API_KEY"):
        print("Using Google Gemini LLM...")
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    else:
        print(
            "Error: No API Key found. Please set OLLAMA_MODEL, GROQ_API_KEY, or GOOGLE_API_KEY in .env"
        )
        sys.exit(1)

    # 3. Load Graph
    graph_rag.load_graph()


def ingest_data(input_pattern="PS1/text_outputs/*.txt", persist_dir="./chroma_db"):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    init_components()

    print(f"Loading text files from {input_pattern}...")
    docs = []
    # Read text files
    files = glob(input_pattern)
    if not files:
        print(f"No text files found matching {input_pattern}")
        return

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
                # Basic metadata
                meta = {"source": os.path.basename(fpath)}
                docs.append(Document(page_content=text, metadata=meta))
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    print(f"Loaded {len(docs)} documents.")

    # Text Splitting (Vector)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    print(f"Ingestion complete. Database saved to {persist_dir}")

    # Graph Ingestion
    print("Starting Graph Extraction...")
    # For graph extraction, we use the raw docs (or large chunks).
    # Using raw docs might be too big for LLM context window?
    # Recursive splitter chunks (1000 chars) are safer.
    graph_rag.extract_and_store_graph(splits, llm)
    print("Graph Ingestion complete.")


def query_rag(query_text, persist_dir="./chroma_db"):
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate

    init_components()

    if not os.path.exists(persist_dir):
        print(f"Vector DB not found at {persist_dir}. Please run with --ingest first.")
        return

    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Prompt
    # We want "evidence-based" answers suitable for Causal Analysis
    template = """You are a Causal Commonsense Analysis assistant. 
    Answer the question strictly based on the provided context and graph relationships.
    If the context does not contain the answer, say "I cannot find the answer in the provided documents."
    
    Prioritize explaining the CAUSE and EFFECT relationships found in the text.
    Cite the source document names if possible.

    Context:
    {context}
    
    Graph Insights:
    {graph_context}

    Question: {question}
    
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = get_rag_chain(retriever, prompt)

    print(f"Querying: {query_text}")
    print("-" * 40)
    result = chain.invoke(query_text)
    print(result["answer"])
    print("-" * 40)


def format_docs(docs):
    return "\n\n".join(
        doc.page_content + f"\n(Source: {doc.metadata.get('source', 'unknown')})"
        for doc in docs
    )


def get_rag_chain(retriever, prompt=None):
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    if prompt is None:
        # Default Prompt
        template = """You are a Causal Commonsense Analysis assistant. 
        Answer the question strictly based on the provided context and graph relationships.
        If the context does not contain the answer, say "I cannot find the answer in the provided documents."
        
        Prioritize explaining the CAUSE and EFFECT relationships found in the text.
        Cite the source document names if possible.

        Context:
        {context}
        
        Graph Insights:
        {graph_context}

        Question: {question}
        
        Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "graph_context": lambda x: graph_rag.get_graph_context(x),
        }
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple RAG for Corporate Reports")
    parser.add_argument(
        "--ingest", action="store_true", help="Ingest PDFs from PS1/text_outputs"
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="PS1/text_outputs/*.txt",
        help="Glob pattern for input text files",
    )
    parser.add_argument("--query", type=str, help="Query string to ask the system")

    args = parser.parse_args()

    if args.ingest:
        ingest_data(input_pattern=args.input_pattern)
    elif args.query:
        query_rag(args.query)
    else:
        parser.print_help()
