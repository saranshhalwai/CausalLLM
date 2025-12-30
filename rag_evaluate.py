import argparse
import sys
import os
import json
import rag_pipeline
import rag_pipeline
from rag_pipeline import query_rag, init_components, get_rag_chain, format_docs

# Access llm via rag_pipeline.llm after init
# For simplicity, we will import the chain components and run them here, or refactor rag_pipeline.
# Let's refactor rag_pipeline slightly to return answer provided we can import it.
# Actually, I'll just use the logic from rag_pipeline here since I can import init_components and build the chain.

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def evaluate_metrics(question, answer, context_str):
    """
    Asks the LLM to rate the answer on specific causal metrics.
    """
    eval_template = """You are an expert evaluator for a Causal Analysis RAG system.
    
    Metric 1: Evidence citation. (0 or 1). Did the answer cite a source document?
    Metric 2: Causal Logic. (1-5). How well did the answer explain the cause-and-effect relationship?
    Metric 3: Conciseness. (1-5).
    
    Context provided to system:
    {context}
    
    User Question: {question}
    System Answer: {answer}
    
    Output JSON only:
    {{
        "evidence_score": <0 or 1>,
        "causal_score": <1-5>,
        "conciseness_score": <1-5>,
        "explanation": "<brief explanation>"
    }}
    """
    
    prompt = ChatPromptTemplate.from_template(eval_template)
    chain = prompt | rag_pipeline.llm | StrOutputParser()
    
    try:
        res = chain.invoke({"question": question, "answer": answer, "context": context_str})
        # Clean up json
        res = res.strip().replace("```json", "").replace("```", "")
        return json.loads(res)
    except Exception as e:
        return {"error": str(e)}

def run_evaluation(persist_dir="./chroma_db"):
    init_components()
    
    if not os.path.exists(persist_dir):
        print("No DB found.")
        return

    # Rebuild retrieval chain
    # from rag_pipeline import embeddings # Already imported via init_components implicitly or we can just access it
    from rag_pipeline import embeddings
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    

    # Sample Causal Queries
    queries = [
        "How did supply chain constraints affect Apple's product availability in 2023?",
        "What factors contributed to Nvidia's data center revenue growth?",
        "How did inflation impacting Johnson & Johnson's operational costs?",
        "What was the effect of foreign currency exchange rates on Apple's net sales?"
    ]
    
    print(f"Running evaluation on {len(queries)} queries...")
    results = []
    
    # Get the chain
    # We can use the default prompt or a custom one. The task is to "Reuse the LLM chain".
    # rag_pipeline uses a specific prompt. rag_evaluate used a slightly different one locally ("gen_template").
    # The user request says "Reuse the LLM chain in rag_pipeline.py ... instead of remaking the same chain".
    # So I should use the chain AS IS from rag_pipeline, meaning I should use the prompt from rag_pipeline.
    # The local gen_template in rag_evaluate was: "Answer the question based on context. Explain causes and effects. Cite sources."
    # The rag_pipeline template is more detailed: "You are a Causal Commonsense Analysis assistant... Prioritize explaining CAUSE and EFFECT..."
    # Using the rag_pipeline one is better and aligns with "Reuse the LLM chain".
    
    chain = get_rag_chain(retriever)

    for q in queries:
        print(f"Query: {q}")
        
        # Invoke chain
        # Chain returns {"context": [docs], "question": q, "answer": str}
        res = chain.invoke(q)
        
        docs = res["context"]
        answer = res["answer"]
        
        # We need formatted context string for evaluation metric
        context_str = format_docs(docs)
        
        print(f"Answer: {answer[:100]}...")
        
        # Evaluate
        eval_res = evaluate_metrics(q, answer, context_str)
        print(f"Metrics: {eval_res}\n")
        
        results.append({
            "query": q,
            "answer": answer,
            "metrics": eval_res
        })

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Evaluation complete. Results saved to evaluation_results.json")

if __name__ == "__main__":
    run_evaluation()
