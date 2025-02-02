# CSV to Vector Database Converter

This tool converts CSV files into a vector database using ChromaDB and sentence-transformers. It's designed to help prepare data for use with LLM agents by creating embeddings from text data.

## Features

- Convert CSV files to ChromaDB vector database
- Support for multiple text columns for embedding
- Optional metadata columns support
- Persistent storage of embeddings
- Uses the efficient all-MiniLM-L6-v2 model for embeddings

## Installation

It's recommended to use a virtual environment to avoid conflicts with other Python packages:

1. Clone this repository
2. Create and activate a virtual environment:
```bash
# Create a virtual environment
python -m venv venv

# Activate it on macOS/Linux
source venv/bin/activate

# Activate it on Windows
.\venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Why Use a Virtual Environment?

Virtual environments are essential because they:
- Isolate project dependencies from your system Python
- Prevent conflicts between different project requirements
- Make it easier to manage package versions
- Enable clean installation and removal of packages
- Make your project more reproducible across different systems

## Usage

1. Prepare your CSV file with text columns you want to embed and any additional metadata columns.

2. Use the converter in your Python code:

```python
from csv2vec import CSV2VectorDB

# Initialize the converter
converter = CSV2VectorDB()

# Process a CSV file
num_processed = converter.process_csv(
    csv_path="your_file.csv",
    collection_name="your_collection",
    text_columns=["text_column1", "text_column2"],  # Columns to be embedded
    metadata_columns=["category", "date"]  # Optional metadata columns
)

print(f"Successfully processed {num_processed} records")
```

## Example

An example CSV file (`example.csv`) is included to demonstrate the expected format:

```csv
text,category,date
"This is a sample text...",technical,2024-01-01
```

To run the example:

```bash
python csv2vec.py
```

## Vector Database Structure

The converter creates a `vector_db` directory in your project folder, which contains:
- Embeddings for your text data
- Associated metadata
- Index information

You can use this vector database with your LLM agent for semantic search and other operations.

## Customization

You can customize the embedding model by passing a different model name to the constructor:

```python
converter = CSV2VectorDB(model_name="your-preferred-model")
```

## Working with Large CSV Files

When dealing with large CSV files (200k+ lines, 800MB+), here's how to efficiently use this tool to reduce API costs:

1. **Preprocessing Large Files**
   ```python
   from csv2vec import CSV2VectorDB
   import pandas as pd
   
   # Process in batches to manage memory
   def process_large_csv(file_path, batch_size=1000):
       converter = CSV2VectorDB()
       for chunk in pd.read_csv(file_path, chunksize=batch_size):
           chunk.to_csv('temp_chunk.csv', index=False)
           converter.process_csv(
               csv_path='temp_chunk.csv',
               collection_name='your_collection',
               text_columns=['your_text_column'],
               metadata_columns=['your_metadata_columns']
           )
   ```

2. **Using the Vector Database with Your LLM Agent**
   ```python
   import chromadb
   
   def semantic_search(query, n_results=5):
       # Connect to your existing database
       client = chromadb.PersistentClient(path="./vector_db")
       collection = client.get_collection("your_collection")
       
       # Perform semantic search
       results = collection.query(
           query_texts=[query],
           n_results=n_results
       )
       return results
   
   # Example usage in your LLM agent
   def process_user_query(user_query):
       # First, search the vector database
       relevant_docs = semantic_search(user_query)
       
       # Use these relevant documents with your LLM API
       context = "\n".join(relevant_docs['documents'][0])
       
       # Now you can send a much more focused query to your LLM
       llm_prompt = f"""
       Context: {context}
       
       User Question: {user_query}
       """
       # Send to your LLM API with the focused context
   ```

## Cost Optimization Tips

1. **Efficient Context Management**
   - Instead of sending entire CSV files to your LLM, use the vector database to find the most relevant content
   - This significantly reduces token usage by only sending relevant context

2. **Batch Processing**
   - Process large CSV files in smaller chunks
   - Allows for better memory management and error recovery

3. **Smart Querying**
   - Use metadata filtering to narrow down searches
   - Adjust n_results based on your needs to control context size

4. **Database Maintenance**
   - Regularly update embeddings for dynamic content
   - Consider archiving old or irrelevant data to maintain performance

## Notes

- The vector database is persistent and will be stored in the `vector_db` directory
- Text from multiple columns will be concatenated before embedding
- All metadata values are converted to strings for storage
- For large files, consider processing during off-peak hours
- Monitor your database size and performance metrics
