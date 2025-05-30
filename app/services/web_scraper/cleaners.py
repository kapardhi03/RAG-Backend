from bs4 import BeautifulSoup
import re

class HTMLCleaner:
    async def clean_html(self, html_content: str) -> str:
        """
        Clean HTML content by removing unwanted elements and focusing on main content
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned HTML content
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted css elements
        for element in soup(["script", "style", "iframe", "nav", "footer", "aside"]):
            element.decompose()
        
        # Extract title and put it at the beginning of the content
        title_element = soup.title
        title_text = title_element.string if title_element else ""
        
        # Try to find main content container
        main_tags = soup.find_all(['main', 'article', 'div', 'section'], 
                                class_=lambda c: c and any(x in str(c).lower() for x in ['content', 'main', 'article', 'post']))
        
        if main_tags:
            main_tag = main_tags[0]
            
            new_html = f"<html><head><title>{title_text}</title></head><body>"
            new_html += f"<h1>{title_text}</h1>"
            new_html += str(main_tag)
            new_html += "</body></html>"
            
            return new_html
        else:
            return str(soup)
    
    async def clean_markdown(self, markdown_text: str) -> str:
        """
        Clean markdown text output from html2text
        
        Args:
            markdown_text: Markdown text from html2text
            
        Returns:
            Cleaned text content
        """
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', markdown_text)
        
        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Remove common navigation text patterns
        navigation_patterns = ['previous', 'next', 'menu', 'skip to content', 'search', 'share']
        for pattern in navigation_patterns:
            text = re.sub(fr'\b{pattern}\b', '', text, flags=re.IGNORECASE)
        
        # Clean up list markers for better readability
        text = re.sub(r'(\* |\+ |- )', '• ', text)
        
        return text.strip()