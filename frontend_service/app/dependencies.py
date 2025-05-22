import aiohttp

async def get_http_client():
    async with aiohttp.ClientSession() as session:
        yield session