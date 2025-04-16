import streamlit as st
import tempfile
import os
from pathlib import Path
import uuid
import time
import asyncio
import random
from typing import Optional , List, Dict

from ingestion.doc_processor import process_llama_documents
from ingestion.weaviate_client import QueryManager
from llm_provider import LLMProvider
import threading

from config import (
    LOCAL_FILE_INPUT_DIR,
    WEAVIATE_COLLECTION_NAME,
    WEAVIATE_API_KEY,
    WEAVIATE_REST_URL,
    TOP_K
)


llm = LLMProvider()


st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>

    .main {
        background-color: #1E1E2E;
        color: #E0E0E0;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        top : 0;
        margin-bottom: 0.2rem;
        color: white;
        text-align: center;
        padding: 1rem 0;
    }
    .block-container {
        padding-bottom: 1rem !important;
    }
    
    .chat-container {
        display: flex;
        flex-direction: column;
        height: calc(55vh - 10rem);
        padding-bottom: 1rem;
        overflow-y: auto;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 0.75rem;
        max-width: 85%;
    }
    .chat-message.user {
        background-color: #3a3c4e;
        margin-left: auto;
        border-bottom-right-radius: 0.2rem;
    }
    .chat-message.bot {
        background-color: #2D4263;
        margin-right: auto;
        border-bottom-left-radius: 0.2rem;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
    }
    .user-avatar {
        background-color: #C84B31;
    }
    .bot-avatar {
        background-color: #4F90CD;
    }
    .message-content {
        color: #ECDBBA;
    }
    
    .input-container {
        position: fixed;
        bottom: 0;
        left: 17rem; 
        right: 2rem;
        background-color: #1E1E2E;
        padding: 1rem 1rem 2rem 1rem;
        z-index: 1000;
    }
    .stTextInput > div > div > input {
        border-radius: 2rem;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        background-color: #2A2D3E;
        color: white;
        
    }
    .stTextInput > div {
        flex: 1;
    }
    
    .css-1d391kg, .css-1544g2n.e1fqkh3o4 {
        background-color: #141423;
    }

    .pdf-item {
        padding: 0.75rem;
        border-radius: 0.5rem;
        background-color: #2A2D3E;
        margin-bottom: 0.75rem;
        border-left: 3px solid #4F90CD;
    }
    .pdf-title {
        color: #ECDBBA;
        font-weight: bold;
    }
    .pdf-size {
        color: #A9A9A9;
        font-size: 0.8rem;
    }
    .drop-area {
        border: 2px dashed #4F90CD;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        background-color: rgba(79, 144, 205, 0.1);
    }
    .drop-text {
        color: #A9A9A9;
    }
    
    .processing-step {
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid #4F90CD;
        background-color: rgba(79, 144, 205, 0.1);
        border-radius: 0.3rem;
    }
    
    .stButton button {
        border-radius: 0.5rem;
        font-weight: bold;
        background-color: #4F90CD;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #3A7DB5;
    }
    
    .welcome-container {
        text-align: center;
        padding: 3rem;
        background-color: rgba(79, 144, 205, 0.05);
        border-radius: 1rem;
        margin: 2rem auto;
        max-width: 800px;
    }
    

    .css-1rs6os.edgvbvh3 {
        visibility: hidden;
    }
    .viewerBadge_container__1QSob {
        display: none;
    }
    

    .robot-icon {
        font-size: 2rem;
        margin-left: 0.5rem;
        vertical-align: middle;
    }

    /* Override Streamlit defaults */
    .stProgress > div > div > div > div {
        background-color: #4F90CD;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #1E1E2E;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        color: white;
        font-weight: bold;
    }
    
    div[data-testid="stVerticalBlock"] {
        padding-bottom: 1rem;
    }
    
    .chat-tab-input-container {
        position: fixed;
        bottom: 0;
        left: 17rem;  /* Sidebar width + padding */
        right: 2rem;
        padding: 1rem 1rem 2rem 1rem;
        z-index: 1000;
    }
</style>
""",
    unsafe_allow_html=True,
)



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = True

if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat"

if "processing_status" not in st.session_state:
    st.session_state.processing_status = (
        "idle"  # Can be 'idle', 'processing', 'completed', 'error'
    )


def ensure_directory_exists(directory_path):
    """Ensure the specified directory exists, create it if not"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


def save_uploaded_pdf(uploaded_file):
    """Save the uploaded PDF to INPUT_DIRECTORY and return the file path"""
    # Ensure input directory exists
    ensure_directory_exists(LOCAL_FILE_INPUT_DIR)

    # Create a unique filename
    filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    filepath = os.path.join(LOCAL_FILE_INPUT_DIR, filename)

    # Save the file
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return filepath


def format_file_size(size_bytes):
    """Format file size in a human-readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024 or unit == "GB":
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024


def clear_uploaded_documents():
    """Clear all uploaded documents and reset processing state"""
    st.session_state.uploaded_pdfs = []
    st.session_state.processing_complete = True
    st.session_state.processing_status = "idle"


def display_chat_message(message, is_user=False):
    """Display a chat message with the appropriate styling"""
    if is_user:
        st.markdown(
            f"""
        <div class="chat-message user">
            <div class="avatar user-avatar">👤</div>
            <div class="message-content">{message}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="chat-message bot">
            <div class="avatar bot-avatar">🤖</div>
            <div class="message-content">{message}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


async def process_query(query, tenant, is_summary : bool = False, text : Optional[str] = None):
    """Process a user query and return a response (placeholder for RAG query)"""

    
    if is_summary:
        content = f'''
                <Context Starts>:
                {"\n\n".join(text)}
                </Context Ends>
                '''
        
        res = llm.get_summary(text=content)

    else:
        user_context = ""
        past_conv = 3
        if len(st.session_state.chat_history) > 0:
            user_context = "\n".join(
                [
                    f"{msg['role']}: {msg['content']}"
                    for msg in st.session_state.chat_history[-2 * past_conv:-1]
                ]
            )
        context_texts = []
        with QueryManager(
            wcd_api_key=WEAVIATE_API_KEY, wcd_url=WEAVIATE_REST_URL
        ) as query_manager:
            contexts = query_manager.query_by_text(
                collection_name=WEAVIATE_COLLECTION_NAME, query_text=query, tenant=tenant, limit=int(TOP_K)
            )
            consolidated_results = []
            context_texts = []
            if contexts:
                for obj in contexts:
                    text = obj.properties.get("text", "")
                    filename = obj.properties.get("filename", "")
                    distance = (
                        obj.metadata.distance
                        if hasattr(obj.metadata, "distance")
                        else "N/A"
                    )
                    context_texts.append(text)
                    consolidated_results.append((text, filename, distance))

        print("------Context texts-------:\n", context_texts)
        print("\n------user_context-------:\n", user_context)
        
        
        content = f'''
                    Query: {query}
                    ----------\n--------
                    {f"Previous Conversation:\n{user_context}\n\n" if user_context else ""}
                    ----------\n--------
                    <Context Starts>:
                    {"\n\n".join(context_texts)}
                    </Context Ends>
                    '''
        
        res = llm.query(query=content)
    
    return res


async def main():

    with st.sidebar:
        st.markdown(
            "<h2 style='color:#ECDBBA;'>PDF Documents</h2>", unsafe_allow_html=True
        )
        st.text("Enter your User_Id, All your PDFs will be saved in Weaviate using this User_Id. If you don't enter a User_Id, it will be saved as 'default' ")
        
        user_id = st.text_input("User Id", value="default", placeholder="Enter your User Id, ex: SpyroSigma")
        if not user_id:
            user_id = "default"

        if st.session_state.uploaded_pdfs:
            for i, pdf in enumerate(st.session_state.uploaded_pdfs):
                st.markdown(
                    f"""
                <div class="pdf-item">
                    <div class="pdf-title">{pdf['name']}</div>
                    <div class="pdf-size">{format_file_size(pdf['size'])}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        
        else:
            st.info("No documents uploaded yet")

    st.markdown(
        "<h1 class='main-header'>PDF Chatbot <span class='robot-icon'>🤖</span></h1>",
        unsafe_allow_html=True,
    )

    # Define tabs with improved styling
    tab1, tab2 = st.tabs(["Upload", "Chat"])

    st.markdown(
        """
    <style>
        /* Improved tab styling */
        .stTabs [data-baseweb="tab"] {
            padding: 12px 24px;
            margin: 0 4px 0 0;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        
        .stTabs [data-baseweb="tab-list"] {
            margin-bottom: 20px;
            border-bottom: 1px solid #343654;
            padding-bottom: 4px;
        }
        
        
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Upload PDFs Tab
    with tab1:
        st.markdown("<h3>Upload Your Documents</h3>", unsafe_allow_html=True)

        # Drop area for files
        st.markdown(
            """
        <div class="drop-area">
            <div class="drop-text">Drag and drop files here</div>
            <div class="drop-text">Limit 200MB per file • PDF</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        # Display current upload status
        if st.session_state.uploaded_pdfs:
            st.info(f"{len(st.session_state.uploaded_pdfs)} document(s) uploaded")

        # Process uploaded files
        if uploaded_files:
            new_files_added = False
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [
                    pdf["name"] for pdf in st.session_state.uploaded_pdfs
                ]:
                    new_files_added = True
                    filepath = save_uploaded_pdf(uploaded_file)

                    # Add to session state
                    st.session_state.uploaded_pdfs.append(
                        {
                            "name": uploaded_file.name,
                            "path": filepath,
                            "size": uploaded_file.size,
                        }
                    )

            if new_files_added:
                st.session_state.processing_complete = False

        # Process button - only show if there are documents to process
        if st.session_state.uploaded_pdfs and not st.session_state.processing_complete:
            col1, col2 = st.columns([3, 1])

            with col1:
                if st.button(
                    "Process Documents for RAG",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.processing_status = "processing"
                    with st.spinner("Processing your documents..."):
                        try:
                            progress_placeholder = st.empty()
                            step_status = st.empty()
                            substep_status = st.empty()
                            
                            # Use a list to store results between threads
                            shared_results = [None]
                            processing_done = [False]
                            
                            async def process_documents():
                                return await process_llama_documents(
                                    user_id=user_id,
                                    collection_name=WEAVIATE_COLLECTION_NAME,
                                )
                            
                            def run_processing():
                                try:
                                    # Use the shared_results list to store the results
                                    shared_results[0] = asyncio.run(process_documents())
                                    print("Processing Results ----> \n", shared_results[0])
                                    processing_done[0] = True
                                except Exception as e:
                                    st.error(f"Processing error: {str(e)}")
                                    processing_done[0] = True
                            
                            processing_thread = threading.Thread(target=run_processing)
                            processing_thread.start()
                            
                            steps = ["Parsing PDFs", "Chunking documents", "Generating vectors", "Building index"]
                            substeps = [
                                ["Extracting text", "Analyzing document structure", "Processing metadata"],
                                ["Creating text chunks", "Optimizing chunk size", "Handling overlaps"],
                                ["Computing embeddings", "Optimizing vector dimensions", "Normalizing vectors"],
                                ["Creating vector index", "Building search structures", "Optimizing retrieval"]
                            ]
                            
                            progress_bar = progress_placeholder.progress(0)
                            step_idx = 0
                            substep_idx = 0
                            progress = 0
                            
                            # Keep showing animation until processing is done
                            while not processing_done[0]:
                                current_step = steps[step_idx % len(steps)]
                                current_substeps = substeps[step_idx % len(steps)]
                                current_substep = current_substeps[substep_idx % len(current_substeps)]
                                
                                step_status.markdown(
                                    f"<div class='processing-step'><b>Processing: {current_step}...</b></div>",
                                    unsafe_allow_html=True,
                                )
                                substep_status.markdown(f"- {current_substep}...")
                                
                                # Update progress (cycle between 0-90% until real processing completes)
                                progress += 0.01
                                if progress > 0.9:  # Cap at 90% until real processing completes
                                    progress = 0.1
                                    step_idx += 1
                                    substep_idx = 0
                                else:
                                    substep_idx = (substep_idx + 1) % len(current_substeps)
                                
                                progress_bar.progress(progress)
                                time.sleep(0.2)
                            
                            # Show 100% completion
                            progress_bar.progress(1.0)
                            step_status.markdown(
                                "<div class='processing-step'><b>Processing complete!</b></div>",
                                unsafe_allow_html=True,
                            )
                            substep_status.empty()

                            # Process results using shared_results instead of results
                            if shared_results[0]:
                                st.session_state.processing_complete = True
                                st.session_state.processing_status = "completed"
                                st.success(
                                    "✅ Documents processed successfully! You can now chat with your documents."
                                )

                                with st.spinner("Generating summary..."):
                                    summary = await process_query(
                                        query="Summarize", 
                                        is_summary=True, 
                                        tenant=user_id, 
                                        text=shared_results[0][1]  # Access the second element of the tuple
                                    )
                                    st.markdown(summary)
                            
                            else:
                                st.session_state.processing_status = "error"
                                st.error(
                                    "❌ Error processing documents. Please try again."
                                )
                        except Exception as e:
                            st.session_state.processing_status = "error"
                            st.error(
                                f"❌ An error occurred during processing: {str(e)}"
                            )

            with col2:
                if st.button("Clear All", use_container_width=True):
                    clear_uploaded_documents()
                    st.rerun()

        elif st.session_state.processing_complete and st.session_state.uploaded_pdfs:
            st.success(
                "✅ Your documents have been processed! Go to the Chat tab to ask questions."
            )

            col1, col2 = st.columns([3, 1])

            with col1:
                if st.button("Upload More Documents", use_container_width=True):
                    st.session_state.processing_complete = False
                    st.rerun()

            with col2:
                if st.button("Clear All", use_container_width=True):
                    clear_uploaded_documents()
                    st.rerun()

        # Show processing status
        if st.session_state.processing_status == "processing":
            st.info("⏳ Processing documents... Please wait.")
        elif st.session_state.processing_status == "completed":
            st.success("✅ Summary Generated successfully! ")
        elif st.session_state.processing_status == "error":
            st.error("❌ Processing failed. Please try again.")

    # Chat Tab
    with tab2:
        st.markdown(
            '<div class="chat-container" id="chat-container">', unsafe_allow_html=True
        )

        # Display chat history
        if not st.session_state.chat_history:
            # Initial greeting
            display_chat_message(
                "Hello! I'm your PDF assistant. You can chat with me about your previously uploaded documents or upload new ones in the Upload tab."
            )
        else:
            # Display all messages
            for message in st.session_state.chat_history:
                display_chat_message(message["content"], message["role"] == "user")

        st.markdown("</div>", unsafe_allow_html=True)

        # Input area for chat
        st.markdown('<div class="chat-tab-input-container">', unsafe_allow_html=True)
        
        # Create a callback to handle query submission
        def handle_submit():
            if st.session_state.query_input.strip():
                query = st.session_state.query_input
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": query})
                # Clear the input
                st.session_state.query_input = ""
                # Set a flag to process the query
                st.session_state.process_query = True
                st.session_state.current_query = query

        # Create form for query input
        with st.form(key="query_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                query_input = st.text_input(
                    "Ask me about your documents:",
                    key="query_input",
                    placeholder="Type your question here...",
                    label_visibility="collapsed"
                )
            with col2:
                submit_button = st.form_submit_button(
                    "Send", 
                    on_click=handle_submit,
                    use_container_width=True
                )
        
        # Process the query if the flag is set
        if "process_query" in st.session_state and st.session_state.process_query:
            with st.spinner("Thinking..."):
                response = asyncio.run(process_query(query=st.session_state.current_query, tenant=user_id))
                
            # Add AI response to chat history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )
            
            # Reset the flag
            st.session_state.process_query = False
            st.rerun()  # Update display with new messages

        st.markdown("</div>", unsafe_allow_html=True)

        # Auto-scroll to bottom script
        st.markdown(
            """
        <script>
            function scrollToBottom() {
                const chatContainer = document.getElementById('chat-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }
            setTimeout(scrollToBottom, 500);
        </script>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
