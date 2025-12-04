"""
Utility functions for handling file operations, particularly for shared document storage.
"""
import os
import base64
from typing import Optional
from pathlib import Path

from .config import settings

def read_shared_file(filename: str) -> str:
    """
    Read a file from the shared PDF directory and return its base64-encoded content.
    
    Args:
        filename: The name of the file (not a path) to read from the shared directory
        
    Returns:
        Base64-encoded string of the file content
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If there are permission issues
    """
    # Ensure the filename doesn't contain path traversal
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        raise ValueError("Filename cannot contain path components")
        
    full_path = os.path.join(settings.shared_pdf_directory, safe_name)
    
    # Verify the path is within the shared directory
    full_path = os.path.abspath(full_path)
    shared_dir = os.path.abspath(settings.shared_pdf_directory)
    if not full_path.startswith(shared_dir):
        raise ValueError("Invalid file path")
    
    # Ensure the shared directory exists
    os.makedirs(shared_dir, exist_ok=True)
    
    # Read and encode the file
    with open(full_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def is_shared_file_reference(file_ref: str) -> bool:
    """Check if a file reference is a shared file reference (starts with 'shared:')"""
    return isinstance(file_ref, str) and file_ref.startswith('shared:')
