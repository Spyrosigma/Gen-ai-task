# PDF Chatbot with RAG

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Streamlit for the user interface. It allows users to upload PDF documents, process them, and then chat with an AI assistant about the content of those documents.

## Features

*   **PDF Upload:** Upload multiple PDF documents through a drag-and-drop interface or file selector.
*   **Document Processing:** Uses LlamaParse for efficient text extraction and chunking from PDFs.
*   **Vector Storage:** Stores document chunks and their vector embeddings in a Weaviate vector database with multi-tenancy support (based on User ID).
*   **RAG Implementation:** Retrieves relevant document chunks based on user queries and feeds them as context to a Large Language Model (LLM).
*   **LLM Integration:** Uses Groq's API (specifically Llama 3.3 70B) for generating responses based on the retrieved context and user query.
*   **Chat Interface:** Provides a user-friendly chat interface built with Streamlit to interact with the documents.
*   **Automatic Summarization:** Generates a brief summary of the uploaded documents after processing.
*   **User-Specific Data:** Associates uploaded documents and chat history with a specific User ID for basic multi-tenancy.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Big_Pdf_RAG
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create a `requirements.txt` file based on the imports in the Python scripts if one doesn't exist.)*

4.  **Set up environment variables:**
    *   Create a `.env` file in the project root directory.
    *   Add the following variables with your credentials and desired paths:
        ```dotenv
        # File Paths (Adjust if needed, defaults might work)
        LOCAL_FILE_INPUT_DIR=./input_docs/
        LOCAL_FILE_OUTPUT_DIR=./output_docs/

        # API Keys
        LLAMAPARSE_API_KEY="your_llamaparse_api_key"
        WEAVIATE_REST_URL="your_weaviate_cluster_url"
        WEAVIATE_API_KEY="your_weaviate_api_key"
        GROQ_API_KEY="your_groq_api_key"

        # Weaviate Configuration
        WEAVIATE_COLLECTION_NAME="PdfRagCollection" # Or your preferred name

        # RAG Configuration
        TOP_K=3 # Number of relevant chunks to retrieve
        ```
    *   Replace placeholder values with your actual API keys and Weaviate URL.

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your web browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Enter a User ID:** In the sidebar, provide a User ID. This ID will be used to store and retrieve your documents in Weaviate. If left empty, it defaults to "default".

4.  **Upload PDFs:** Go to the "Upload" tab. Drag and drop your PDF files or use the file browser.

5.  **Process Documents:** Once files are uploaded, click the "Process Documents for RAG" button. The application will:
    *   Parse the PDFs using LlamaParse.
    *   Chunk the extracted text.
    *   Generate vector embeddings.
    *   Upload the data to your Weaviate collection under the specified User ID.
    *   Generate and display an initial summary based on the first chunk.

6.  **Chat with Documents:** Go to the "Chat" tab. Ask questions about the content of your uploaded documents. The chatbot will retrieve relevant information and generate an answer.

## Configuration

*   **Environment Variables (`.env`):** All external service credentials (LlamaParse, Weaviate, Groq) and configuration parameters (file paths, Weaviate collection name, `TOP_K`) are managed through the `.env` file. See the Setup section for details.
*   **`config.py`:** Loads the environment variables for use within the application.
*   **`llm_provider.py`:** Configures the LLM (Groq Llama 3.3 70B) and defines system prompts for summarization and querying.
*   **`ingestion/doc_processor.py`:** Handles the LlamaParse configuration and document processing workflow.
*   **`ingestion/weaviate_client.py`:** Manages interaction with the Weaviate vector database, including data upload and querying.

## Technologies Used

*   **Python:** Core programming language.
*   **Streamlit:** Web application framework for the UI.
*   **LlamaParse:** Document parsing and text extraction.
*   **Weaviate:** Vector database for storing and retrieving document chunks.
*   **Groq (Llama 3.3 70B):** Large Language Model for generation.
*   **`llama-index-core` (implicitly via LlamaParse):** Core data structures.
*   **`python-dotenv`:** Environment variable management.
*   **`nest_asyncio`:** Handling asyncio loops within Streamlit/other environments.

## Future Enhancements

*   More robust error handling during processing.
*   Support for more document types (e.g., images, web pages).
*   Advanced RAG strategies (e.g., query rewriting, reranking).
*   More sophisticated multi-tenancy and user management.
*   Option to delete specific documents or clear data for a user.
*   Displaying source document snippets alongside answers.
