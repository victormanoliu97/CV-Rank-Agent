"""WebCrawl tool bound to the job_parser node."""
from crawl4ai import *

async def web_crawl(url: str) -> str:
    """Tool: crawl a URL and return its raw text content."""
    async with AsyncWebCrawler() as crawler:
        content = await crawler.arun(url=url)
        return content.markdown