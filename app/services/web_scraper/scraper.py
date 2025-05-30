# app/services/web_scraper/scraper.py
import httpx
import html2text
import logging
from app.services.web_scraper.cleaners import HTMLCleaner
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        self.html_cleaner = HTMLCleaner()
    
    async def scrape(self, url: str) -> str:
        """Scrape content from a URL"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ""
    
    async def clean_content(self, html_content: str) -> str:
        """Clean HTML content and convert to plain text"""
        try:
            # Use BeautifulSoup to clean content
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'meta', 'noscript']):
                script.extract()
            
            # Convert to plain text
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.ignore_tables = False
            
            text = h.handle(str(soup))
            
            # Remove excessive newlines
            text = '\n'.join(line for line in text.splitlines() if line.strip())
            
            return text
        except Exception as e:
            logger.error(f"Error cleaning content: {e}")
            return ""

def extract_all_urls(base_url: str) -> list[str]:
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        raw_links = {urljoin(base_url, a.get("href")) for a in soup.find_all("a", href=True)}
        
        # Filter out non-HTML links
        html_links = []
        for link in raw_links:
            parsed = urlparse(link)
            path = parsed.path.lower()
            if not path.endswith((".jpg", ".jpeg", ".png", ".gif", ".svg", ".pdf", ".docx", ".mp4", ".zip", ".js", ".css")):
                html_links.append(link)

        return list(set(html_links))
    
    except Exception as e:
        raise RuntimeError(f"Error extracting URLs: {str(e)}") 