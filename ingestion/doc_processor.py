import nest_asyncio
import os
import logging
from typing import Optional
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
import uuid, shutil

from config import (
    LLAMAPARSE_API_KEY,
    LOCAL_FILE_INPUT_DIR,
    LOCAL_FILE_OUTPUT_DIR,
    WEAVIATE_API_KEY,
    WEAVIATE_REST_URL,
)
from ingestion.weaviate_client import DataManager
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


nest_asyncio.apply()

from ingestion.script_llamaparse import run_llama_script


def llama_parse(
    input_directory: str, output_directory: str, api_key: Optional[str] = None
) -> bool:
    """
    Process documents from the input directory and save extracted content to the output directory.

    Args:
        input_directory: Path to the directory containing input documents
        output_directory: Path where processed documents will be saved
        api_key: LlamaParse API key (optional, will use default if not provided)

    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            logger.info(f"Created output directory: {output_directory}")

        # Check if input directory exists
        if not os.path.exists(input_directory):
            logger.error(f"Input directory does not exist: {input_directory}")
            return False

        # Initialize LlamaParse
        parser = LlamaParse(
            api_key=LLAMAPARSE_API_KEY,
            result_type="markdown",
            verbose=True,
        )

        # Configure file extractors
        file_extractor = {
            ".pdf": parser,
            ".txt": parser,
            ".docx": parser,
            ".pptx": parser,
        }

        # Load documents
        logger.info(f"Loading documents from {input_directory}")
        documents = SimpleDirectoryReader(
            input_dir=input_directory, file_extractor=file_extractor, recursive=True
        ).load_data(show_progress=True)

        logger.info(f"Processing {len(documents)} documents")

        # Process and save each document
        for i, doc in enumerate(documents):
            try:
                document_data = doc.model_dump_json()
                output_path = f"{output_directory}/docs-{uuid.uuid4()}.json"

                with open(output_path, "w") as f:
                    f.write(str(document_data))

                logger.info(f"Saved document to {output_path}")

            except Exception as e:
                logger.error(f"Error processing document {i}: {str(e)}")

        logger.info("Document processing completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        return False


async def process_llama_documents(user_id: str, collection_name: str) -> str:
    """
    Process documents using LlamaParse.

    Returns:
        str: Status message about the processing result
    """
    try:
        input_dir = LOCAL_FILE_INPUT_DIR
        output_dir = LOCAL_FILE_OUTPUT_DIR

        results = llama_parse(input_directory=input_dir, output_directory=output_dir)
        await run_llama_script(output_dir)

        # Collect all data objects for the session
        json_dir_path = output_dir
        print(f"Scanning for JSON files in: {json_dir_path}")
        file_count = 0
        object_count = 0
        data_objects = []
        # Iterate over all JSON files in the directory
        for filename in os.listdir(json_dir_path):
            if filename.endswith(".json"):
                file_count += 1
                json_file_path = os.path.join(json_dir_path, filename)
                try:
                    with open(json_file_path, "r") as json_file:
                        data_objects += json.load(json_file)

                    object_count += len(data_objects)

                except json.JSONDecodeError as je:
                    logger.error(f"JSON parse error in file {filename}: {je}")
                    continue
                except Exception as e:
                    logger.error(
                        f"Error processing file {filename}: {e}", exc_info=True
                    )
                    continue


        print(
            f"Uploading {len(data_objects)} objects to Weaviate collection '{collection_name}' for tenant '{user_id}'"
        )

        # print("\n----------Data Objects----------\n:", data_objects)
        chunk = data_objects[0] if data_objects else None
        first_chunk = chunk.get("text")
        # print(f"First chunk of data object: {first_chunk}")
        with DataManager(
            wcd_url=WEAVIATE_REST_URL, wcd_api_key=WEAVIATE_API_KEY
        ) as weaviate_uploader:
            res = weaviate_uploader.upload_objects(
                collection_name=collection_name,
                data_objects=data_objects,
                tenant=user_id,
            )
           

        if not results:
            raise Exception("LlamaParse processing returned no results or failed")

        if not res:
            raise Exception("Weaviate upload failed or returned no results")

        print(
            f"Documents processed successfully and Uploaded to Weaviate collection '{collection_name}' for tenant '{user_id}'"
        )
        return [True, first_chunk]

    except Exception as e:
        error_msg = f"Error processing documents with LlamaParse: {str(e)}"
        print(error_msg)
        return False

    finally:
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
            print(f"Cleaned up user-specific input directory: {input_dir}")
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Cleaned up user-specific output directory: {output_dir}")
