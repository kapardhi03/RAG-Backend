"""
Optimized URL processing utilities with concurrent processing and timeouts.
"""

import re
import logging
import asyncio
import aiohttp
from typing import List, Set, Optional
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
import time

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
        """Check if URL is valid for processing"""
        try:
            if not url or not isinstance(url, str) or len(url) > 2000:
                return False
            
            if not url.startswith(('http://', 'https://')):
                return False
            
            parsed = urlparse(url)
            
            if base_domain and parsed.netloc != base_domain:
                return False
            
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in cls.EXCLUDED_EXTENSIONS):
                return False
            
            if any(re.search(pattern, url, re.IGNORECASE) for pattern in cls.EXCLUDED_PATTERNS):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating URL {url}: {e}")
            return False
    
    @classmethod
    def clean_url(cls, url: str) -> str:
        """Clean and normalize URL"""
        try:
            parsed = urlparse(url)
            cleaned = urlunparse((
                parsed.scheme, parsed.netloc, parsed.path,
                parsed.params, parsed.query, ''
            ))
            
            if cleaned.endswith('/') and len(parsed.path) > 1:
                cleaned = cleaned[:-1]
            
            return cleaned
        except Exception as e:
            logger.warning(f"Error cleaning URL {url}: {e}")
            return url
    
    @classmethod
    def filter_urls(cls, urls: List[str], base_domain: Optional[str] = None, max_urls: int = 1000) -> List[str]:
        """Filter and clean URLs efficiently"""
        cleaned_urls = set()
        
        for url in urls:
            if len(cleaned_urls) >= max_urls:
                break
                
            try:
                cleaned_url = cls.clean_url(url)
                if cls.is_valid_url(cleaned_url, base_domain):
                    cleaned_urls.add(cleaned_url)
            except Exception:
                continue
        
        return sorted(list(cleaned_urls))

class FastURLExtractor:
    """Fast URL extraction with concurrent processing and timeouts"""
    
    def __init__(self):
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch URL content with timeout and error handling"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.debug(f"Fetched {url}: {len(content)} chars")
                    return content
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {url}")
            return None
        except Exception as e:
            logger.warning(f"Error fetching {url}: {e}")
            return None
    
    async def extract_sitemap_urls(self, sitemap_content: str) -> List[str]:
        """Extract URLs from sitemap XML"""
        urls = []
        try:
            soup = BeautifulSoup(sitemap_content, 'xml')
            
            # Look for <loc> tags
            for loc in soup.find_all('loc'):
                if loc.text:
                    urls.append(loc.text.strip())
            
            # If no <loc> tags, try <url> tags
            if not urls:
                for url in soup.find_all('url'):
                    loc = url.find('loc')
                    if loc and loc.text:
                        urls.append(loc.text.strip())
                        
        except Exception as e:
            logger.error(f"Failed to parse sitemap: {e}")
        
        return urls
    
    async def extract_robots_urls(self, robots_content: str, base_url: str) -> List[str]:
        """Extract sitemap URLs from robots.txt"""
        urls = []
        try:
            for line in robots_content.splitlines():
                line = line.strip()
                if line.lower().startswith('sitemap:'):
                    sitemap_url = line.split(':', 1)[1].strip()
                    if sitemap_url.startswith('/'):
                        sitemap_url = urljoin(base_url, sitemap_url)
                    urls.append(sitemap_url)
        except Exception as e:
            logger.error(f"Failed to parse robots.txt: {e}")
        
        return urls
    
    async def discover_sitemap_urls(self, base_url: str) -> List[str]:
        """Discover URLs from sitemaps with concurrent processing"""
        logger.info(f"Discovering sitemap URLs for: {base_url}")
        
        # Common sitemap locations
        sitemap_urls = [
            urljoin(base_url, '/sitemap.xml'),
            urljoin(base_url, '/sitemap_index.xml'),
            urljoin(base_url, '/robots.txt')
        ]
        
        discovered_urls = set()
        
        # Fetch all sitemaps concurrently
        tasks = [self.fetch_url_content(url) for url in sitemap_urls]
        contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, content in enumerate(contents):
            if isinstance(content, Exception) or not content:
                continue
            
            try:
                if 'sitemap' in sitemap_urls[i]:
                    urls = await self.extract_sitemap_urls(content)
                    discovered_urls.update(urls)
                    logger.info(f"Found {len(urls)} URLs in {sitemap_urls[i]}")
                else:  # robots.txt
                    robots_sitemaps = await self.extract_robots_urls(content, base_url)
                    
                    # Fetch sitemap URLs from robots.txt
                    if robots_sitemaps:
                        sitemap_tasks = [self.fetch_url_content(url) for url in robots_sitemaps[:5]]  # Limit to 5
                        sitemap_contents = await asyncio.gather(*sitemap_tasks, return_exceptions=True)
                        
                        for sitemap_content in sitemap_contents:
                            if isinstance(sitemap_content, str):
                                sitemap_urls_found = await self.extract_sitemap_urls(sitemap_content)
                                discovered_urls.update(sitemap_urls_found)
                                
            except Exception as e:
                logger.warning(f"Error processing {sitemap_urls[i]}: {e}")
        
        logger.info(f"Total discovered URLs: {len(discovered_urls)}")
        return list(discovered_urls)
    
    async def extract_page_links(self, url: str, base_domain: str) -> List[str]:
        """Extract links from a single page"""
        content = await self.fetch_url_content(url)
        if not content:
            return []
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                
                if (absolute_url.startswith(('http://', 'https://')) and 
                    urlparse(absolute_url).netloc == base_domain and 
                    '#' not in absolute_url):
                    links.append(absolute_url)
            
            return links
            
        except Exception as e:
            logger.warning(f"Error extracting links from {url}: {e}")
            return []
    
    async def light_crawl(self, start_url: str, max_pages: int = 10) -> List[str]:
        """Lightweight crawl with concurrent processing"""
        logger.info(f"Starting light crawl from: {start_url}")
        
        base_domain = urlparse(start_url).netloc
        discovered_urls = {start_url}
        to_crawl = {start_url}
        crawled = set()
        
        # Process pages in batches
        batch_size = 5
        
        while to_crawl and len(crawled) < max_pages:
            # Get next batch
            current_batch = list(to_crawl)[:batch_size]
            to_crawl -= set(current_batch)
            
            # Crawl batch concurrently
            tasks = [self.extract_page_links(url, base_domain) for url in current_batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, links in enumerate(results):
                crawled.add(current_batch[i])
                
                if isinstance(links, list):
                    new_links = set(links) - discovered_urls - crawled
                    discovered_urls.update(new_links)
                    to_crawl.update(list(new_links)[:5])  # Limit new URLs per page
        
        logger.info(f"Light crawl found {len(discovered_urls)} URLs")
        return list(discovered_urls)

async def extract_all_urls(
    base_url: str, 
    max_urls: int = 100,  # Reduced default
    include_crawling: bool = True,
    same_domain_only: bool = True,
    timeout: int = 60,  # Add timeout parameter
    limit: int = 500,
) -> List[str]:
    """
    Fast URL extraction with timeouts and concurrent processing.
    
    Args:
        base_url: Base URL to start extraction from
        max_urls: Maximum number of URLs to return (reduced default)
        include_crawling: Whether to include light crawling
        same_domain_only: Whether to restrict to same domain only
        timeout: Maximum time in seconds for the entire operation
        
    Returns:
        List of discovered URLs
    """
    start_time = time.time()
    logger.info(f"Starting fast URL extraction for: {base_url} (max: {max_urls}, timeout: {timeout}s)")
    
    try:
        # Set up timeout for the entire operation
        async def extract_with_timeout():
            base_domain = urlparse(base_url).netloc if same_domain_only else None
            
            async with FastURLExtractor() as extractor:
                discovered_urls = set()
                
                # Step 1: Fast sitemap discovery
                logger.info("Step 1: Discovering sitemap URLs...")
                try:
                    sitemap_urls = await extractor.discover_sitemap_urls(base_url)
                    discovered_urls.update(sitemap_urls)
                    logger.info(f"Sitemap discovery found {len(sitemap_urls)} URLs")
                except Exception as e:
                    logger.warning(f"Sitemap discovery failed: {e}")
                
                # Step 2: Light crawling if needed and enabled
                if include_crawling and len(discovered_urls) < max_urls // 2:
                    logger.info("Step 2: Light crawling...")
                    try:
                        crawl_limit = min(10, max_urls - len(discovered_urls))  # Reduced crawl limit
                        crawl_urls = await extractor.light_crawl(base_url, crawl_limit)
                        discovered_urls.update(crawl_urls)
                        logger.info(f"Light crawl found {len(crawl_urls)} additional URLs")
                    except Exception as e:
                        logger.warning(f"Light crawl failed: {e}")
                
                # Step 3: Filter and validate
                logger.info("Step 3: Filtering and validating URLs...")
                filtered_urls = URLValidator.filter_urls(
                    list(discovered_urls),
                    base_domain=base_domain,
                    max_urls=max_urls
                )
                
                return filtered_urls[:limit]
        
        # Run with timeout
        result = await asyncio.wait_for(extract_with_timeout(), timeout=timeout)
        
        elapsed = time.time() - start_time
        logger.info(f"URL extraction completed in {elapsed:.2f}s. Found {len(result)} URLs")
        
        return result
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.warning(f"URL extraction timed out after {elapsed:.2f}s")
        return []
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"URL extraction failed after {elapsed:.2f}s: {e}")
        return []

# Quick extraction function for simple use cases
async def extract_sitemap_urls_fast(base_url: str, max_urls: int = 50, timeout: int = 30) -> List[str]:
    """
    Very fast sitemap-only extraction for quick results.
    
    Args:
        base_url: Base URL to extract from
        max_urls: Maximum URLs to return
        timeout: Timeout in seconds
        
    Returns:
        List of URLs from sitemaps only
    """
    logger.info(f"Fast sitemap extraction for: {base_url}")
    
    try:
        async with FastURLExtractor() as extractor:
            async def fast_extract():
                urls = await extractor.discover_sitemap_urls(base_url)
                base_domain = urlparse(base_url).netloc
                return URLValidator.filter_urls(urls, base_domain, max_urls)
            
            return await asyncio.wait_for(fast_extract(), timeout=timeout)
            
    except asyncio.TimeoutError:
        logger.warning(f"Fast extraction timed out for {base_url}")
        return []
    except Exception as e:
        logger.error(f"Fast extraction failed for {base_url}: {e}")
        return []