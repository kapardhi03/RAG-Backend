class TxtParser:
    async def parse(self, file_content: bytes) -> str:
        """
        Parse a TXT file and extract its text content
        
        Args:
            file_content: Binary content of the TXT file
            
        Returns:
            Extracted text from the TXT file
        """
        try:
            # Try UTF-8 first
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Try Latin-1 if UTF-8 fails
                text = file_content.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Failed to parse DOCX: {type(e).__name__}: {str(e)}")
        
        return text.strip()