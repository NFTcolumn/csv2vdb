import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

class CSV2VectorDB:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the converter with a specified embedding model."""
        self.chroma_client = chromadb.PersistentClient(path="./vector_db")
        self.embedding_model = SentenceTransformer(model_name)
        
    def create_collection(self, name):
        """Create or get a ChromaDB collection."""
        return self.chroma_client.get_or_create_collection(
            name=name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

    def process_csv(self, csv_path, collection_name, text_columns, metadata_columns=None):
        """
        Process a CSV file and store it in the vector database.
        
        Args:
            csv_path (str): Path to the CSV file
            collection_name (str): Name for the ChromaDB collection
            text_columns (list): Columns to be embedded
            metadata_columns (list, optional): Columns to be stored as metadata
        """
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Create or get collection
        collection = self.create_collection(collection_name)
        
        # Prepare data for embedding
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            # Combine text columns for embedding
            text_content = " ".join(str(row[col]) for col in text_columns)
            documents.append(text_content)
            
            # Prepare metadata
            metadata = {}
            if metadata_columns:
                for col in metadata_columns:
                    metadata[col] = str(row[col])
            metadatas.append(metadata)
            
            # Create unique ID
            ids.append(f"{collection_name}_{idx}")
        
        # Add data to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(documents)

    def process_large_csv(self, file_path, collection_name, text_columns, metadata_columns=None, batch_size=1000):
        """
        Process a large CSV file in batches to manage memory efficiently.
        
        Args:
            file_path (str): Path to the large CSV file
            collection_name (str): Name for the ChromaDB collection
            text_columns (list): Columns to be embedded
            metadata_columns (list, optional): Columns to be stored as metadata
            batch_size (int): Number of rows to process in each batch
        """
        total_processed = 0
        
        # Process the CSV in chunks
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=batch_size)):
            # Save chunk to temporary file
            temp_file = f"temp_chunk_{chunk_num}.csv"
            chunk.to_csv(temp_file, index=False)
            
            try:
                # Process the chunk
                processed = self.process_csv(
                    csv_path=temp_file,
                    collection_name=collection_name,
                    text_columns=text_columns,
                    metadata_columns=metadata_columns
                )
                total_processed += processed
                print(f"Processed batch {chunk_num + 1}: {processed} records")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return total_processed

    def query(self, collection_name, query_text, n_results=5):
        """
        Perform semantic search on a collection.
        
        Args:
            collection_name (str): Name of the collection to search
            query_text (str): The search query
            n_results (int): Number of results to return
            
        Returns:
            dict: Search results including documents, metadata, and distances
        """
        collection = self.chroma_client.get_collection(name=collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

def main():
    # Example usage
    converter = CSV2VectorDB()
    
    # Process a CSV file
    csv_path = "example.csv"  # Path to your CSV file
    collection_name = "my_collection"  # Name for your collection
    text_columns = ["text"]  # Columns to be embedded
    metadata_columns = ["category", "date"]  # Optional metadata columns
    
    # For small files, use process_csv
    num_processed = converter.process_csv(
        csv_path=csv_path,
        collection_name=collection_name,
        text_columns=text_columns,
        metadata_columns=metadata_columns
    )
    print(f"Successfully processed {num_processed} records")
    
    # Example of processing a large file in batches
    # num_processed = converter.process_large_csv(
    #     file_path="large_file.csv",
    #     collection_name="large_collection",
    #     text_columns=["text"],
    #     metadata_columns=["category", "date"],
    #     batch_size=1000
    # )
    # print(f"Successfully processed {num_processed} records from large file")
    
    # Example of querying the database
    results = converter.query(
        collection_name=collection_name,
        query_text="machine learning",
        n_results=2
    )
    print("\nSearch Results:")
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        print(f"\nDocument: {doc}")
        print(f"Metadata: {metadata}")

if __name__ == "__main__":
    main()
