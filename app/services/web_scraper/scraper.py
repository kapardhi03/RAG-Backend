import asyncio
import logging
import random
import time
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urljoin, urlparse, parse_qs
import json

import aiohttp
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import fake-useragent with fallback
try:
    from fake_useragent import UserAgent
    HAS_USER_AGENT = True
except ImportError:
    HAS_USER_AGENT = False
    import warnings
    warnings.warn("fake-useragent not installed. Using fallback user agents.")

# Try to import Selenium components (optional)
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.common.exceptions import TimeoutException
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    warnings.warn("Selenium not installed. Advanced scraping features will be limited.")

from app.core.config import settings

logger = logging.getLogger(__name__)

class FallbackUserAgent:
    """Fallback user agent handler when fake-useragent is not available"""
    
    def __init__(self):
        self.agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        self._index = 0
    
    @property
    def random(self):
        import random
        return random.choice(self.agents)
    
    @property
    def chrome(self):
        return self.agents[0]
    
    @property
    def firefox(self):
        return self.agents[3]

class AdvancedWebScraper:
    """Advanced web scraper with anti-blocking capabilities"""
    
    def __init__(self):
        self.session = self._create_session()
        self.driver = None
        self.proxy_list = getattr(settings, 'PROXY_LIST', []) if hasattr(settings, 'USE_PROXY') and settings.USE_PROXY else []
        self.current_proxy_index = 0
        
        # Initialize user agent handler
        if HAS_USER_AGENT:
            try:
                self.user_agent = UserAgent()
                logger.info("Initialized UserAgent successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize UserAgent: {e}")
                self.user_agent = FallbackUserAgent()
        else:
            self.user_agent = FallbackUserAgent()
            logger.info("Using fallback user agent")
        
        # Rate limiting
        self.request_delay = (1, 3)  # Random delay between requests
        self.last_request_time = 0
        
        logger.info("Advanced web scraper initialized")
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
        session = requests.Session()
        
        # Retry strategy
        max_retries = getattr(settings, 'MAX_RETRIES', 3)
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_random_headers(self) -> Dict[str, str]:
        """Generate random headers to avoid detection"""
        return {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
    
    def _get_proxy(self) -> Optional[Dict[str, str]]:
        """Get next proxy from the list"""
        if not self.proxy_list:
            return None
        
        proxy = self.proxy_list[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_list)
        
        return {
            'http': proxy,
            'https': proxy
        }
    
    async def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay[0]:
            delay = random.uniform(*self.request_delay)
            await asyncio.sleep(delay)
        
        self.last_request_time = time.time()
    
    async def scrape_with_requests(self, url: str) -> Optional[str]:
        """Scrape using requests library with anti-blocking measures"""
        try:
            await self._rate_limit()
            
            headers = self._get_random_headers()
            proxies = self._get_proxy() if hasattr(settings, 'USE_PROXY') and settings.USE_PROXY else None
            
            timeout = getattr(settings, 'SCRAPING_TIMEOUT', 30)
            
            async with aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
                connector=aiohttp.TCPConnector(ssl=False) if proxies else None
            ) as session:
                
                async with session.get(
                    url,
                    proxy=proxies.get('http') if proxies else None
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Successfully scraped {url} with requests")
                        return content
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
                        
        except Exception as e:
            logger.error(f"Requests scraping failed for {url}: {e}")
            return None
    
    def _setup_selenium_driver(self, headless: bool = True) -> Optional[webdriver.Chrome]:
        """Setup Selenium WebDriver with anti-detection measures"""
        if not HAS_SELENIUM:
            logger.error("Selenium not available")
            return None
        
        try:
            chrome_options = ChromeOptions()
            
            if headless:
                chrome_options.add_argument('--headless')
            
            # Anti-detection arguments
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-plugins')
            chrome_options.add_argument('--disable-images')
            
            # Random user agent
            chrome_options.add_argument(f'--user-agent={self.user_agent.chrome}')
            
            # Proxy support
            if hasattr(settings, 'USE_PROXY') and settings.USE_PROXY and self.proxy_list:
                proxy = self.proxy_list[self.current_proxy_index]
                chrome_options.add_argument(f'--proxy-server={proxy}')
            
            # Random window size
            window_sizes = ['1920,1080', '1366,768', '1440,900', '1280,1024']
            chrome_options.add_argument(f'--window-size={random.choice(window_sizes)}')
            
            driver = webdriver.Chrome(options=chrome_options)
            
            # Execute script to remove webdriver property
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            return driver
            
        except Exception as e:
            logger.error(f"Failed to setup Selenium driver: {e}")
            return None
    
    async def scrape_with_selenium(self, url: str, wait_for_element: str = None) -> Optional[str]:
        """Scrape using Selenium for JavaScript-heavy sites"""
        if not HAS_SELENIUM:
            logger.warning("Selenium not available, falling back to requests")
            return await self.scrape_with_requests(url)
        
        driver = None
        try:
            await self._rate_limit()
            
            driver = self._setup_selenium_driver()
            if not driver:
                return None
            
            driver.get(url)
            
            # Wait for specific element if provided
            if wait_for_element:
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element))
                    )
                except TimeoutException:
                    logger.warning(f"Element {wait_for_element} not found within timeout")
            
            # Random scroll to simulate human behavior
            total_height = driver.execute_script("return document.body.scrollHeight")
            for i in range(1, 4):
                driver.execute_script(f"window.scrollTo(0, {total_height * i / 4});")
                await asyncio.sleep(random.uniform(0.5, 1.5))
            
            content = driver.page_source
            logger.info(f"Successfully scraped {url} with Selenium")
            
            return content
            
        except Exception as e:
            logger.error(f"Selenium scraping failed for {url}: {e}")
            return None
        finally:
            if driver:
                driver.quit()
    
    async def scrape_with_fallback(self, url: str, prefer_selenium: bool = False) -> Optional[str]:
        """Scrape with automatic fallback between methods"""
        methods = ['selenium', 'requests'] if prefer_selenium else ['requests', 'selenium']
        
        for method in methods:
            try:
                if method == 'requests':
                    content = await self.scrape_with_requests(url)
                else:
                    content = await self.scrape_with_selenium(url)
                
                if content and self._validate_content(content):
                    return content
                    
            except Exception as e:
                logger.warning(f"Method {method} failed for {url}: {e}")
                continue
        
        logger.error(f"All scraping methods failed for {url}")
        return None
    
    def _validate_content(self, content: str) -> bool:
        """Validate that the scraped content is meaningful"""
        if not content or len(content) < 100:
            return False
        
        # Check for common blocking indicators
        blocking_indicators = [
            'access denied',
            'blocked',
            'captcha',
            'bot detection',
            'rate limit',
            'too many requests',
            'forbidden'
        ]
        
        content_lower = content.lower()
        for indicator in blocking_indicators:
            if indicator in content_lower:
                return False
        
        return True
    
    async def extract_clean_text(self, html_content: str, url: str = None) -> str:
        """Extract and clean text from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
                element.decompose()
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, type(soup))):
                comment.extract()
            
            # Extract title
            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            
            # Try to find main content
            main_content = None
            content_selectors = [
                'main',
                'article',
                '[role="main"]',
                '.content',
                '.main-content',
                '#content',
                '#main'
            ]
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.body or soup
            
            # Extract text
            text = main_content.get_text(separator='\n', strip=True)
            
            # Clean up text
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = '\n'.join(lines)
            
            # Add title if available
            if title:
                cleaned_text = f"{title}\n\n{cleaned_text}"
            
            # Add URL as source
            if url:
                cleaned_text = f"Source: {url}\n\n{cleaned_text}"
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Failed to extract clean text: {e}")
            return ""
    
    async def scrape_multiple_urls(
        self, 
        urls: List[str], 
        max_concurrent: int = 5,
        prefer_selenium: bool = False
    ) -> Dict[str, str]:
        """Scrape multiple URLs concurrently with rate limiting"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}
        
        async def scrape_single(url: str):
            async with semaphore:
                try:
                    html_content = await self.scrape_with_fallback(url, prefer_selenium)
                    if html_content:
                        text_content = await self.extract_clean_text(html_content, url)
                        results[url] = text_content
                    else:
                        results[url] = ""
                except Exception as e:
                    logger.error(f"Failed to scrape {url}: {e}")
                    results[url] = ""
        
        # Create tasks for all URLs
        tasks = [scrape_single(url) for url in urls]
        
        # Execute with progress logging
        for i, task in enumerate(asyncio.as_completed(tasks)):
            await task
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{len(urls)} URLs")
        
        logger.info(f"Scraping completed: {len([r for r in results.values() if r])} successful out of {len(urls)}")
        
        return results
    
    async def discover_sitemap_urls(self, base_url: str) -> List[str]:
        """Discover URLs from sitemap.xml and robots.txt"""
        urls = set()
        
        # Try common sitemap locations
        sitemap_urls = [
            urljoin(base_url, '/sitemap.xml'),
            urljoin(base_url, '/sitemap_index.xml'),
            urljoin(base_url, '/robots.txt')
        ]
        
        for sitemap_url in sitemap_urls:
            try:
                content = await self.scrape_with_requests(sitemap_url)
                if content:
                    if 'sitemap' in sitemap_url:
                        urls.update(self._extract_sitemap_urls(content))
                    else:  # robots.txt
                        urls.update(self._extract_robots_urls(content, base_url))
            except Exception as e:
                logger.warning(f"Failed to fetch {sitemap_url}: {e}")
        
        return list(urls)
    
    def _extract_sitemap_urls(self, sitemap_content: str) -> List[str]:
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
    
    def _extract_robots_urls(self, robots_content: str, base_url: str) -> List[str]:
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
    
    async def smart_crawl(
        self, 
        start_url: str, 
        max_pages: int = 50,
        same_domain_only: bool = True
    ) -> Dict[str, str]:
        """Intelligent crawling with link discovery"""
        
        crawled_urls = set()
        to_crawl = {start_url}
        results = {}
        
        base_domain = urlparse(start_url).netloc
        
        while to_crawl and len(crawled_urls) < max_pages:
            current_url = to_crawl.pop()
            
            if current_url in crawled_urls:
                continue
            
            # Skip if different domain and same_domain_only is True
            if same_domain_only and urlparse(current_url).netloc != base_domain:
                continue
            
            try:
                html_content = await self.scrape_with_fallback(current_url)
                if html_content:
                    text_content = await self.extract_clean_text(html_content, current_url)
                    results[current_url] = text_content
                    
                    # Extract links for further crawling
                    new_urls = self._extract_links(html_content, current_url)
                    to_crawl.update(new_urls)
                
                crawled_urls.add(current_url)
                logger.info(f"Crawled {len(crawled_urls)}/{max_pages}: {current_url}")
                
            except Exception as e:
                logger.error(f"Failed to crawl {current_url}: {e}")
                crawled_urls.add(current_url)
        
        return results
    
    def _extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract all links from HTML content"""
        urls = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, href)
                
                # Filter out non-HTTP URLs and anchors
                if absolute_url.startswith(('http://', 'https://')) and '#' not in absolute_url:
                    urls.append(absolute_url)
                    
        except Exception as e:
            logger.error(f"Failed to extract links: {e}")
        
        return urls
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'session'):
                self.session.close()
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Cleanup resources on deletion"""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup


# Simple function for backward compatibility
async def extract_all_urls(base_url: str) -> List[str]:
    """
    Simple function to extract URLs from a website.
    This is for backward compatibility with existing code.
    """
    scraper = AdvancedWebScraper()
    try:
        # Try sitemap first
        urls = await scraper.discover_sitemap_urls(base_url)
        
        # If not many URLs found, try crawling
        if len(urls) < 10:
            crawl_results = await scraper.smart_crawl(base_url, max_pages=20)
            urls.extend(crawl_results.keys())
        
        # Remove duplicates and return
        return list(set(urls))
    
    except Exception as e:
        logger.error(f"Error extracting URLs: {e}")
        raise
    finally:
        scraper.cleanup()