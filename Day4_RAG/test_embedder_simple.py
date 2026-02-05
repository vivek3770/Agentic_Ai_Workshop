
import sys
import os

# Add parent directory to path to allow importing from RAG package if running from within RAG folder, 
# or ensure we can import RAG if running from parent. 
# Better: just append current cwd.
sys.path.append(os.getcwd())

try:
    from RAG.embedder import get_embedding
except ImportError:
    # Fallback if running directly inside RAG folder
    sys.path.append(os.path.dirname(os.getcwd()))
    from embedder import get_embedding

def test_single_embedding():
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    chunk_text = f"Q: {question}\nA: {answer}"
    
    print(f"Testing embedding for chunk:\n---\n{chunk_text}\n---")
    
    try:
        vector = get_embedding(chunk_text)
        print("\n[SUCCESS] Embedding generated.")
        print(f"Vector dimensions: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")
        
        if len(vector) == 768:
            print("\n[CHECK] Dimension 768 matches text-embedding-004 default.")
        else:
            print(f"\n[WARN] Dimension {len(vector)} differs from expected 768.")
            
    except Exception as e:
        print(f"\n[ERROR] Failed to generate embedding: {e}")

if __name__ == "__main__":
    test_single_embedding()