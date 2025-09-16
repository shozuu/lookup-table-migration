import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini with safety settings
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY"),
    client_options={"api_endpoint": "generativelanguage.googleapis.com"}
)

def main():
    try:
        # Test text for embedding
        test_text = "This is a test sentence for generating embeddings using Gemini API."
        
        logger.info("Sending embedding request to Gemini API...")
        
        # Generate embedding using the text-embedding-004 model
        result = genai.embed_content(
            model="models/text-embedding-004",
            contents="The quick brown fox jumps over the lazy dog."     
        )
        print(result.embeddings)

        result = genai.embed_content(
            model="models/text-embedding-004",
            content=test_text,
            task_type="retrieval_document" 
        )
        
        embedding = result['embedding']
        
        print("API key is working for embeddings!")
        print(f"Text: '{test_text}'")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 10 embedding values: {embedding[:10]}")
        
        # Test with multiple texts
        logger.info("Testing batch embedding...")
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science."
        ]
        
        batch_result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document"
        )
        
        print(f"\nBatch embedding successful!")
        print(f"Number of embeddings: {len(batch_result['embedding'])}")
        for i, text in enumerate(texts):
            print(f"Text {i+1}: '{text}' -> Embedding shape: {len(batch_result['embedding'][i])}")
        
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()