"""
URL processing utilities.
"""

import re
import logging
from typing import List, Set, Optional
from urllib.parse import urljoin, urlparse, urlunparse

logger = logging.getLogger(__name__)

class URLValidator:
    """Validates and filters URLs for processing"""
    
    # File extensions to exclude
    EXCLUDED_EXTENSIONS = {
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.rar', '.7z', '.tar', '.gz',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
        '.css', '.js', '.json', '.xml', '.rss',
        '.exe', '.dmg', '.pkg', '.deb', '.rpm'
    }
    
    # URL patterns to exclude
    EXCLUDED_PATTERNS = [
        r'/api/',
        r'/admin/',
        r'/wp-admin/',
        r'/login',
        r'/logout',
        r'/register',
        r'/signin',
        r'/signup',
        r'\.php$',
        r'/feed/',
        r'/rss/',
        r'/sitemap',
        r'/robots\.txt',
        r'/favicon\.ico',
        r'mailto:',
        r'tel:',
        r'javascript:',
        r'#'
    ]
    
    @classmethod
    def is_valid_url(cls, url: str, base_domain: Optional[str] = None) -> bool:
        """
        Check if URL is valid for processing.
        
        Args:
            url: URL to validate
            base_domain: Base domain to restrict to (optional)
            
        Returns:
            True if URL is valid for processing
        """
        try:
            # Basic URL validation
            if not url or not isinstance(url, str):
                return False
            
            # Must be HTTP/HTTPS
            if not url.startswith(('http://', 'https://')):
                return False
            
            # Parse URL
            parsed = urlparse(url)
            
            # Check domain restriction
            if base_domain and parsed.netloc != base_domain:
                return False
            
            # Check for excluded file extensions
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in cls.EXCLUDED_EXTENSIONS):
                return False
            
            # Check for excluded patterns
            if any(re.search(pattern, url, re.IGNORECASE) for pattern in cls.EXCLUDED_PATTERNS):
                return False
            
            # Check URL length (avoid extremely long URLs)
            if len(url) > 2000:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating URL {url}: {e}")
            return False
    
    @classmethod
    def clean_url(cls, url: str) -> str:
        """
        Clean and normalize URL.
        
        Args:
            url: URL to clean
            
        Returns:
            Cleaned URL
        """
        try:
            # Parse URL
            parsed = urlparse(url)
            
            # Remove fragment
            cleaned = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                ''  # Remove fragment
            ))
            
            # Remove trailing slash for consistency (except root)
            if cleaned.endswith('/') and len(parsed.path) > 1:
                cleaned = cleaned[:-1]
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Error cleaning URL {url}: {e}")
            return url
    
    @classmethod
    def filter_urls(cls, urls: List[str], base_domain: Optional[str] = None, max_urls: int = 1000) -> List[str]:
        """
        Filter and clean a list of URLs.
        
        Args:
            urls: List of URLs to filter
            base_domain: Base domain to restrict to
            max_urls: Maximum number of URLs to return
            
        Returns:
            Filtered and cleaned list of URLs
        """
        cleaned_urls = set()
        
        for url in urls:
            try:
                # Clean URL
                cleaned_url = cls.clean_url(url)
                
                # Validate URL
                if cls.is_valid_url(cleaned_url, base_domain):
                    cleaned_urls.add(cleaned_url)
                    
                    # Stop if we reach the limit
                    if len(cleaned_urls) >= max_urls:
                        break
                        
            except Exception as e:
                logger.warning(f"Error processing URL {url}: {e}")
                continue
        
        # Sort for consistent ordering
        return sorted(list(cleaned_urls))


async def extract_all_urls(
    base_url: str, 
    max_urls: int = 500,
    include_crawling: bool = True,
    same_domain_only: bool = True
) -> List[str]:
    """
    Comprehensive URL extraction function that combines multiple methods.
    
    This is the main function that should be used for URL extraction.
    
    Args:
        base_url: Base URL to start extraction from
        max_urls: Maximum number of URLs to return
        include_crawling: Whether to include crawling in addition to sitemap discovery
        same_domain_only: Whether to restrict to same domain only
        
    Returns:
        List of discovered URLs
    """
    from app.services.web_scraper.scraper import AdvancedWebScraper
    
    logger.info(f"Starting comprehensive URL extraction for: {base_url}")
    
    try:
        # Parse base domain for filtering
        base_domain = urlparse(base_url).netloc if same_domain_only else None
        
        scraper = AdvancedWebScraper()
        discovered_urls = set()
        
        # Step 1: Discover from sitemaps and robots.txt
        logger.info("Step 1: Discovering URLs from sitemaps...")
        sitemap_urls = await scraper.discover_sitemap_urls(base_url)
        discovered_urls.update(sitemap_urls)
        logger.info(f"Found {len(sitemap_urls)} URLs from sitemaps")
        
        # Step 2: Smart crawling (if enabled and needed)
        if include_crawling and len(discovered_urls) < max_urls // 2:
            logger.info("Step 2: Performing smart crawl...")
            crawl_limit = min(50, max_urls - len(discovered_urls))
            
            crawl_results = await scraper.smart_crawl(
                start_url=base_url,
                max_pages=crawl_limit,
                same_domain_only=same_domain_only
            )
            
            discovered_urls.update(crawl_results.keys())
            logger.info(f"Smart crawl found {len(crawl_results)} additional URLs")
        
        # Step 3: Filter and clean URLs
        logger.info("Step 3: Filtering and cleaning URLs...")
        filtered_urls = URLValidator.filter_urls(
            list(discovered_urls),
            base_domain=base_domain,
            max_urls=max_urls
        )
        
        logger.info(f"URL extraction completed. Returning {len(filtered_urls)} URLs")
        return filtered_urls
        
    except Exception as e:
        logger.error(f"Failed to extract URLs: {e}")
        raise Exception(f"URL extraction failed: {str(e)}")
    
    finally:
        # Cleanup scraper
        if 'scraper' in locals():
            try:
                scraper.cleanup()
            except:
                pass