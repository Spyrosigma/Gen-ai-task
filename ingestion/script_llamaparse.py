import json
import os
import asyncio
from typing import List, Dict


async def process_json_file(file_path: str) -> None:
    """
    Process a single JSON file to extract filename and text fields
    and save the processed data back to the same file
    
    Args:
        file_path: Path to the JSON file to process
    """
    try:
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if the data is a dictionary (single document) or list (multiple documents)
        if isinstance(data, dict):
            data = [data]  # Convert to list for consistent processing
        
        # Create a list to store processed items
        processed_items = []
        
        for item in data:
            processed_item = {}
            
            # Extract filename from metadata if it exists
            if 'metadata' in item and 'file_name' in item['metadata']:
                processed_item['filename'] = item['metadata']['file_name']
            elif 'filename' in item:
                processed_item['filename'] = item['filename']
            
            # Extract text content
            if 'text' in item:
                processed_item['text'] = item['text']
            
            # Only add items that have both text and filename
            if 'text' in processed_item and 'filename' in processed_item:
                processed_items.append(processed_item)
        
        # Write back to the same file
        if processed_items:
            with open(file_path, 'w') as f:
                json.dump(processed_items, f, indent=2)
            
            print(f"Successfully processed and updated {file_path}")
        else:
            print(f"No valid data found in {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def find_json_files(directory: str) -> List[str]:
    """
    Find all JSON files in the given directory.
    
    Args:
        directory: Directory to search for JSON files
        
    Returns:
        List of paths to JSON files
    """
    json_files = []
    for file in os.listdir(directory):
        if file.endswith('.json'):
            json_files.append(os.path.join(directory, file))
    return json_files

async def run_llama_script(input_dir: str):
    """
    Asynchronously process all JSON files in a directory and save results back to the same files.
    
    Args:
        input_dir: Directory containing JSON files to process
    """
    json_files = find_json_files(input_dir)
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    # Process each file, saving back to the original file
    for file_path in json_files:
        await process_json_file(file_path)
    
    print(f"All files processed. Results saved back to original files.")

