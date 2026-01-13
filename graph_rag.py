import os
import networkx as nx
import pickle
import numpy as np
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from tqdm import tqdm

# Global embeddings model
embedding_model = None
# Cache for node embeddings: {node_id: embedding_vector}
node_embeddings = {}

# Global graph instance
G = nx.DiGraph()

def init_graph_transformer(llm):
    return LLMGraphTransformer(llm=llm, allowed_nodes=['events'], allowed_relationships=['causes', 'correlates'])

def set_embeddings(model):
    global embedding_model, node_embeddings
    embedding_model = model
    # Clear cache when model changes
    node_embeddings = {}

def extract_and_store_graph(docs, llm, persist_path="graph_data.pkl"):
    global G, node_embeddings
    # Clear embedding cache as graph is changing
    node_embeddings = {}
    print(f"Extracting Graph Data from {len(docs)} documents (incremental)...")
    
    llm_transformer = init_graph_transformer(llm)
    
    try:
        # Iterate over docs individually to allow progress tracking and partial stopping
        for i, doc in enumerate(tqdm(docs, desc="Extracting Graph")):
            # Convert single document
            # convert_to_graph_documents expects a list
            graph_results = llm_transformer.convert_to_graph_documents([doc])
            
            for graph_doc in graph_results:
                for node in graph_doc.nodes:
                    if not G.has_node(node.id):
                        G.add_node(node.id, type=node.type)
                
                for edge in graph_doc.relationships:
                    G.add_edge(edge.source.id, edge.target.id, relation=edge.type)
            
            # Periodic save (every 50 docs) to avoid data loss on crash/exit
            if (i + 1) % 50 == 0:
                 with open(persist_path, "wb") as f:
                    pickle.dump(G, f)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
    except Exception as e:
        print(f"\nError encountered: {e}. Saving current progress...")
        
    # Final save
    print(f"Saving graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to {persist_path}...")
    with open(persist_path, "wb") as f:
        pickle.dump(G, f)
        
def load_graph(persist_path="graph_data.pkl"):
    global G
    if os.path.exists(persist_path):
        print(f"Loading graph from {persist_path}...")
        with open(persist_path, "rb") as f:
            G = pickle.load(f)
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        # Clear embedding cache on load
        node_embeddings = {}
    else:
        print("No existing graph found. Please ingest data first.")

def get_graph_context(query, depth=1):
    """
    Simple retrieval: find nodes in the graph that appear in the query, 
    and return their neighbors (1-hop or 2-hop).
    """
    global G, embedding_model, node_embeddings
    
    relevant_subgraph = []
    found_nodes = []
    
    # 1. Vector Similarity Search (if embeddings available)
    if embedding_model:
        # Compute embeddings for all nodes if not cached
        nodes_to_embed = [n for n in G.nodes() if n not in node_embeddings]
        if nodes_to_embed:
            print(f"Computing embeddings for {len(nodes_to_embed)} new nodes...")
            try:
                embeddings = embedding_model.embed_documents(nodes_to_embed)
                for node, emb in zip(nodes_to_embed, embeddings):
                    node_embeddings[node] = np.array(emb)
            except Exception as e:
                print(f"Error computing node embeddings: {e}")
        
        # Embed query
        try:
            query_emb = np.array(embedding_model.embed_query(query))
            
            # Compute Cosine Similarity
            # simple dot product if normalized, but let's do full cosine
            results = []
            for node, node_emb in node_embeddings.items():
                # Cosine similarity: (A . B) / (||A|| * ||B||)
                score = np.dot(query_emb, node_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(node_emb))
                if score > 0.4: # Threshold
                     results.append((node, score))
            
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            found_nodes = [r[0] for r in results[:5]] # Top 5
            print(f"Vector search found: {results[:5]}")
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            found_nodes = []

    # 2. Fallback / Augment: Exact String Matching (if vector search didn't find much or model missing)
    if not found_nodes:
        query_lower = query.lower()
        for node in G.nodes():
            if node.lower() in query_lower:
                found_nodes.append(node)
        if found_nodes:
             print(f"Fallback string match found: {found_nodes}")

            
    if not found_nodes:
        return "No direct graph matches found."
        
    print(f"Graph Entities Found: {found_nodes}")
    
    visited = set()
    
    for start_node in found_nodes:
        # Get neighbors
        # For directed graph, we might care about successors or predecessors or both.
        # Let's look at successors (outgoing edges) -> 'causes'
        # And predecessors (incoming edges) -> 'caused by'
        
        # Outgoing
        if start_node in G:
             for neighbor in G.successors(start_node):
                edge_data = G.get_edge_data(start_node, neighbor)
                relation = edge_data.get("relation", "related_to")
                triplet = f"({start_node}) --[{relation}]--> ({neighbor})"
                if triplet not in visited:
                    relevant_subgraph.append(triplet)
                    visited.add(triplet)

        # Incoming
        if start_node in G:
            for neighbor in G.predecessors(start_node):
                edge_data = G.get_edge_data(neighbor, start_node)
                relation = edge_data.get("relation", "related_to")
                triplet = f"({neighbor}) --[{relation}]--> ({start_node})"
                if triplet not in visited:
                    relevant_subgraph.append(triplet)
                    visited.add(triplet)
                    
    if not relevant_subgraph:
         return f"Entities {found_nodes} found in graph but have no connections."
         
    return "\n".join(relevant_subgraph)
