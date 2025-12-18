# RAG Chatbot with Local LLM - No API Key Required!
# This chatbot uses RAG with a completely local model
from transformers.integrations import accelerate

# Step 1: Install required packages
print("Installing packages... (this may take 2-3 minutes)")
# !pip
# install - q
# sentence - transformers
# faiss - cpu
# transformers
# torch
# accelerate

# Step 2: Import libraries
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Step 3: Initialize embedding model
print("\n📦 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded!")

# Step 4: Initialize local LLM (using a small efficient model)
print("\n📦 Loading language model (this may take 3-5 minutes)...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small but capable model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
print("✅ Language model loaded!")

# Step 5: Sample knowledge base (replace with your own documents)
documents = [
    "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed. It uses algorithms to find patterns.",
    "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes that process information.",
    "Deep learning uses multiple layers of neural networks to progressively extract higher-level features from raw input data. It excels at image and speech recognition.",
    "Natural language processing (NLP) helps computers understand, interpret and generate human language. It powers chatbots, translation, and text analysis.",
    "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with language generation to produce more accurate and grounded responses.",
    "Transformers are a type of neural network architecture introduced in 2017. They use self-attention mechanisms and revolutionized NLP tasks.",
    "GPT models are autoregressive language models based on the transformer architecture. They generate text by predicting the next token in a sequence.",
    "Vector databases store embeddings and enable semantic search over documents. They allow finding similar content based on meaning rather than keywords.",
    "Fine-tuning adapts pre-trained models to specific tasks with smaller datasets. It's more efficient than training from scratch.",
    "Embeddings are vector representations of text that capture semantic meaning. Similar texts have similar embeddings in vector space.",
    "FAISS is a library for efficient similarity search. It can quickly find nearest neighbors in high-dimensional vector spaces.",
    "LangChain is a framework for developing applications with large language models. It provides tools for chaining prompts and managing context.",
    "Prompt engineering is the practice of designing effective inputs for language models. Good prompts can dramatically improve output quality.",
    "Hugging Face is a platform for sharing and using machine learning models. It hosts thousands of pre-trained models for various tasks."
]

# Step 6: Create embeddings and build FAISS index
print("\n📚 Creating embeddings for knowledge base...")
doc_embeddings = embedding_model.encode(documents)
dimension = doc_embeddings.shape[1]

# Build FAISS index for fast similarity search
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings).astype('float32'))

print(f"✅ Indexed {len(documents)} documents")


# Step 7: Retrieval function
def retrieve_relevant_docs(query, top_k=3):
    """Retrieve top_k most relevant documents for the query"""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)

    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs, distances[0]


# Step 8: Generate response with local LLM
def generate_response(prompt, max_length=300):
    """Generate response using local LLM"""

    # Format prompt for TinyLlama chat format
    formatted_prompt = f"""<|system|>
You are a helpful AI assistant. Answer questions based on the provided context. If you cannot answer from the context, say so honestly.</s>
<|user|>
{prompt}</s>
<|assistant|>
"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()

    return response


# Step 9: RAG Chat function
def rag_chat(user_query):
    """Main RAG pipeline: retrieve + generate"""

    # Retrieve relevant documents
    print(f"\n🔍 Searching knowledge base...")
    relevant_docs, distances = retrieve_relevant_docs(user_query, top_k=3)

    print(f"✅ Retrieved {len(relevant_docs)} relevant documents\n")

    # Create context from retrieved documents
    context = "\n".join([f"{i + 1}. {doc}" for i, doc in enumerate(relevant_docs)])

    # Create prompt with context
    prompt = f"""Context information:
{context}

Question: {user_query}

Please answer the question based on the context above. Be concise and accurate."""

    # Generate response using local LLM
    print("🤖 Generating response...\n")
    response = generate_response(prompt)

    # Display results
    print("=" * 70)
    print("📚 RETRIEVED CONTEXT:")
    print("=" * 70)
    for i, doc in enumerate(relevant_docs, 1):
        print(f"{i}. {doc}")
        print(f"   (Relevance score: {distances[i - 1]:.4f})")

    print("\n" + "=" * 70)
    print("🤖 AI RESPONSE:")
    print("=" * 70)
    print(response)
    print("=" * 70)

    return response


# Step 10: Interactive chat loop
print("\n" + "=" * 70)
print("✅ RAG CHATBOT IS READY - NO API KEY NEEDED!")
print("=" * 70)
print("Type 'quit' to exit")
print("Type 'help' for example questions\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ['quit', 'exit', 'q']:
        print("👋 Goodbye!")
        break

    if user_input.lower() == 'help':
        print("\n💡 Example questions to try:")
        print("  - What is RAG?")
        print("  - Explain neural networks")
        print("  - What's the difference between machine learning and deep learning?")
        print("  - How do transformers work?")
        print("  - What are embeddings?\n")
        continue

    if user_input:
        try:
            rag_chat(user_input)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again with a different question.\n")
        print("\n")