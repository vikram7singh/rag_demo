# Import necessary libraries
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load environment variables
load_dotenv()

# Load documents from a directory (you can change this path as needed)
documents = SimpleDirectoryReader("data").load_data()

# # Print the number of documents
# print(f"Number of documents: {len(documents)}")

# # Sample the first few documents (e.g., up to 3)
# print("\nSample of documents:")
# for i, doc in enumerate(documents[:3], 1):
#     print(f"\nDocument {i}:")
#     print(f"Content: {doc.text[:200]}...")  # Print first 200 characters

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine
query_engine = index.as_query_engine()

# Example query
response = query_engine.query("What years does the strategic plan cover?")

print(response)