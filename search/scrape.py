from crawl4ai import AsyncWebCrawler
import asyncio
import sys

# Set ProactorEventLoopPolicy on Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


async def get_pages_content(urls):
    async with AsyncWebCrawler(verbose=True) as crawler:
        results = await crawler.arun_many(
            urls=urls,
            remove_overlay_elements=True,
            excluded_tags=['nav', 'footer', 'aside']
        )

        contents = []

        for result in results:
            if result.markdown_v2 != None:
                contents.append({
                    "content":result.markdown_v2.raw_markdown,
                    "source": result.url
                })

    return contents
