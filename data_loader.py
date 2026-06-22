from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Chunking: Process of breaking it down into smaller pieces 
# & then embedding those smaller pieces 
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

# Split text into chunks with some overlap to preserve context
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    # Read the PDF & extract its contents into document/page objects 
    docs = PDFReader().load_data(file=path)
    # Extract text from each document/page while skipping empty pages 
    texts = [d.text for d in docs if getattr(d, "text", None)]
    # List to store all generated text chunks 
    chunks = [] 
    # Looping through every page's text 
    for t in texts: 
        # Split a page into text chunks & add them to the list 
        chunks.extend(splitter.split_text(t))
    # Return all text chunks ready for embedding 
    return chunks 

# Fx to convert text chunks into numerical vector embeddings for semantic search
def embed_texts(texts: list[str]) -> list[list[float]]:
    # Request embeddings from OpenAI fo all provided text chunks 
    response = client.embeddings.create(
        # Embedding model used to convert text into vectors 
        model=EMBED_MODEL, 
        input=texts, # list of text chunks to embed 
    )
    # Extract & return only the embedding vectors from the API response 
    return [item.embedding for item in response.data]
