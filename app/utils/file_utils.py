import os
from typing import Dict, Any, Optional

# MIME type mapping based on extension
MIME_TYPE_MAP = {
    # Document types
    'pdf': 'application/pdf',
    'doc': 'application/msword',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'txt': 'text/plain',
    'md': 'text/markdown',
    # Image types
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'gif': 'image/gif',
    # Other types
    'html': 'text/html',
    'css': 'text/css',
    'js': 'application/javascript',
    'json': 'application/json',
    'xml': 'application/xml',
    'zip': 'application/zip',
}

def get_file_extension(filename: str) -> str:
    """
    Get the extension of a file
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension without the dot
    """
    return os.path.splitext(filename)[1].lower().replace(".", "")

def get_mime_type(file_content: bytes, filename: str = None) -> str:
    """
    Determine MIME type based on file extension
    
    Args:
        file_content: Raw file content as bytes (not used in this version)
        filename: Name of the file (used to get extension)
        
    Returns:
        MIME type
    """
    if filename:
        extension = get_file_extension(filename)
        return MIME_TYPE_MAP.get(extension, 'application/octet-stream')
    return 'application/octet-stream'

def is_supported_file_type(filename: str) -> bool:
    """
    Check if file type is supported
    
    Args:
        filename: Name of the file
        
    Returns:
        True if supported, False otherwise
    """
    extension = get_file_extension(filename)
    supported_extensions = ['pdf', 'docx', 'doc', 'txt', 'md']
    return extension in supported_extensions

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing special characters
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove special characters that might cause issues in S3
    sanitized = "".join(c for c in filename if c.isalnum() or c in "._- ")
    return sanitized