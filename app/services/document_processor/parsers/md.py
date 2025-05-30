# app/services/document_processor/parsers/md.py
import logging

logger = logging.getLogger("md_parser")

class MDParser:
    async def parse(self, content: bytes) -> str:
        """Extract text from Markdown file"""
        try:
            logger.info("Parsing Markdown file")
            
            # Decode bytes to string
            # Tryin utf-8 first, then fall back to latin-1 if that fails
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')
            
            logger.info(f"Successfully extracted {len(text)} characters from Markdown")
            return text
        except Exception as e:
            logger.error(f"Error parsing Markdown: {str(e)}")
            raise ValueError(f"Failed to parse Markdown: {str(e)}")