import io
import PyPDF2

class PDFParser:
    async def parse(self, file_content: bytes) -> str:
        """
        Parse a PDF file and extract its text content
        
        Args:
            file_content: Binary content of the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        pdf_file = io.BytesIO(file_content)
        
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            return text.strip()
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {type(e).__name__}: {str(e)}")
        
    # Uncomment the following methods if you want to enable OCR and image extraction functionality
    # async def _extract_images_and_ocr(self, file_content: bytes) -> str:
    #     """Extract images from PDF and perform OCR"""
    #     if not self.enable_ocr:
    #         return ""
        
    #     try:
    #         # Open PDF with PyMuPDF
    #         pdf_document = fitz.open(stream=file_content, filetype="pdf")
    #         ocr_text = ""
            
    #         for page_num in range(len(pdf_document)):
    #             page = pdf_document.load_page(page_num)
                
    #             # Get images from the page
    #             image_list = page.get_images()
                
    #             if image_list:
    #                 ocr_text += f"--- Images from Page {page_num + 1} ---\n"
                    
    #                 for img_index, img in enumerate(image_list):
    #                     try:
    #                         # Extract image
    #                         xref = img[0]
    #                         pix = fitz.Pixmap(pdf_document, xref)
                            
    #                         # Convert to PIL Image
    #                         if pix.n - pix.alpha < 4:  # GRAY or RGB
    #                             img_data = pix.tobytes("ppm")
    #                             pil_image = Image.open(io.BytesIO(img_data))
    #                         else:  # CMYK
    #                             pix1 = fitz.Pixmap(fitz.csRGB, pix)
    #                             img_data = pix1.tobytes("ppm")
    #                             pil_image = Image.open(io.BytesIO(img_data))
    #                             pix1 = None
                            
    #                         # Perform OCR
    #                         image_text = pytesseract.image_to_string(
    #                             pil_image, 
    #                             config=self.tesseract_config
    #                         ).strip()
                            
    #                         if image_text:
    #                             ocr_text += f"Image {img_index + 1} text:\n{image_text}\n\n"
                            
    #                         pix = None
                            
    #                     except Exception as img_error:
    #                         print(f"Error processing image {img_index + 1} on page {page_num + 1}: {img_error}")
    #                         continue
            
    #         pdf_document.close()
    #         return ocr_text
            
    #     except Exception as e:
    #         print(f"Error during image extraction: {e}")
    #         return ""
    
    # async def parse_with_metadata(self, file_content: bytes) -> Dict[str, Any]:
    #     """
    #     Parse PDF and return both content and metadata
        
    #     Args:
    #         file_content: Binary content of the PDF file
            
    #     Returns:
    #         Dictionary containing text content, metadata, and processing info
    #     """
    #     try:
    #         # Extract text content
    #         text_content = await self.parse(file_content)
            
    #         # Extract metadata using PyPDF2
    #         pdf_file = io.BytesIO(file_content)
    #         reader = PyPDF2.PdfReader(pdf_file)
            
    #         metadata = {
    #             'num_pages': len(reader.pages),
    #             'text_content': text_content,
    #             'has_images': False,
    #             'ocr_enabled': self.enable_ocr
    #         }
            
    #         # Add PDF metadata if available
    #         if reader.metadata:
    #             pdf_info = {
    #                 'title': reader.metadata.get('/Title', ''),
    #                 'author': reader.metadata.get('/Author', ''),
    #                 'subject': reader.metadata.get('/Subject', ''),
    #                 'creator': reader.metadata.get('/Creator', ''),
    #                 'producer': reader.metadata.get('/Producer', ''),
    #                 'creation_date': reader.metadata.get('/CreationDate', ''),
    #                 'modification_date': reader.metadata.get('/ModDate', '')
    #             }
    #             metadata['pdf_info'] = pdf_info
            
    #         # Check for images if OCR is enabled
    #         if self.enable_ocr:
    #             pdf_document = fitz.open(stream=file_content, filetype="pdf")
    #             total_images = 0
    #             for page_num in range(len(pdf_document)):
    #                 page = pdf_document.load_page(page_num)
    #                 total_images += len(page.get_images())
                
    #             metadata['has_images'] = total_images > 0
    #             metadata['total_images'] = total_images
    #             pdf_document.close()
            
    #         return metadata
            
    #     except Exception as e:
    #         raise ValueError(f"Failed to parse PDF with metadata: {type(e).__name__}: {str(e)}")
