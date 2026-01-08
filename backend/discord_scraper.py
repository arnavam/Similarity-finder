"""
Discord Scraper Module for Code Similarity Checker.
Scrapes URLs from Discord channels for specific #tags.
Supports incremental scraping (only fetches new messages since last scrape).
"""

import asyncio
import os
import re
from typing import List, Optional, Tuple

# Discord.py for bot interactions
try:
    import discord

    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    print("⚠️ discord.py not installed. Discord scraping disabled.")

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")

# URL regex pattern
URL_PATTERN = re.compile(
    r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*", re.IGNORECASE
)


def is_discord_configured() -> bool:
    """Check if Discord bot is properly configured."""
    configured = DISCORD_AVAILABLE and bool(DISCORD_BOT_TOKEN)
    if not configured:
        if not DISCORD_AVAILABLE:
            print("⚠️ Discord: discord.py library not available")
        elif not DISCORD_BOT_TOKEN:
            print("⚠️ Discord: DISCORD_BOT_TOKEN not set in environment")
    return configured


async def _scrape_channel_messages(
    channel_id: str, tag: str, after_message_id: Optional[str] = None, limit: int = 500
) -> Tuple[List[str], Optional[str]]:
    """
    Internal async function to scrape messages from a Discord channel.

    Args:
        channel_id: Discord channel ID
        tag: Tag to search for (e.g., "#nlp-classification")
        after_message_id: If set, only fetch messages after this ID (incremental)
        limit: Maximum number of messages to fetch

    Returns:
        (urls, last_message_id) - list of URLs found and ID of newest message
    """
    if not is_discord_configured():
        print("❌ Discord not configured")
        return [], None

    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)
    urls = []
    newest_message_id = None

    @client.event
    async def on_ready():
        nonlocal urls, newest_message_id

        try:
            channel = client.get_channel(int(channel_id))
            if not channel:
                channel = await client.fetch_channel(int(channel_id))

            if not channel:
                print(f"❌ Channel {channel_id} not found")
                await client.close()
                return

            # Prepare fetch parameters
            after = None
            if after_message_id:
                try:
                    after = discord.Object(id=int(after_message_id))
                except:
                    pass

            # Fetch messages
            messages = []
            async for message in channel.history(
                limit=limit, after=after, oldest_first=False
            ):
                messages.append(message)

            if messages:
                # Track newest message for incremental scraping
                newest_message_id = str(messages[0].id)

            # Search for tag and extract URLs
            tag_lower = tag.lower().lstrip("#")

            for message in messages:
                content = message.content

                # Check if message contains the tag
                # Match patterns like: #tag, # tag, or just the tag name at start
                tag_patterns = [f"#{tag_lower}", f"# {tag_lower}", tag_lower]

                content_lower = content.lower()
                # ===== Legacy API (backwards compatible) =====

                # Check if any tag pattern is in the message
                has_tag = any(pattern in content_lower for pattern in tag_patterns)

                if has_tag:
                    # Extract all URLs from this message
                    found_urls = URL_PATTERN.findall(content)
                    urls.extend(found_urls)

            print(f"✅ Scraped {len(messages)} messages, found {len(urls)} URLs for #{tag_lower}")

        except Exception as e:
            print(f"❌ Error scraping channel: {e}")
        finally:
            await client.close()

    # Run the client
    try:
        await client.start(DISCORD_BOT_TOKEN)
    except Exception as e:
        print(f"❌ Discord client error: {e}")
        if not client.is_closed():
            await client.close()

    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls, newest_message_id


def scrape_tag_from_channel(
    channel_id: str, tag: str, after_message_id: Optional[str] = None
) -> Tuple[List[str], Optional[str]]:
    """
    Synchronous wrapper for Discord channel scraping.
    Uses a separate thread with its own event loop to avoid conflicts with FastAPI's uvloop.

    Args:
        channel_id: Discord channel ID
        tag: Tag to search for (e.g., "nlp-classification" or "#nlp-classification")
        after_message_id: For incremental scraping

    Returns:
        (urls, last_message_id)
    """
    import concurrent.futures

    # Normalize tag
    tag = tag.lstrip("#")

    def run_in_thread():
        """Run the async scraper in a new thread with its own event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                _scrape_channel_messages(channel_id, tag, after_message_id)
            )
        finally:
            loop.close()

    # Run in a separate thread to avoid event loop conflicts
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_thread)
        return future.result(timeout=60)  # 60 second timeout


def get_or_scrape_urls(
    channel_id: str, tag: str, force_refresh: bool = False
) -> List[str]:
    """
    Main entry point. Uses cache with incremental updates.

    1. Check cache for channel_id + tag
    2. If cached and not force_refresh, fetch only NEW messages
    3. Merge new URLs with cached URLs
    4. Update cache

    Args:
        channel_id: Discord channel ID
        tag: Tag name (with or without #)
        force_refresh: If True, re-scrape everything ignoring cache

    Returns:
        List of URLs
    """
    # Import here to avoid circular imports
    from auth import get_cached_discord_urls, update_discord_cache

    tag = tag.lstrip("#")

    # Check cache
    cached = get_cached_discord_urls(channel_id, tag)

    if cached and not force_refresh:
        # Incremental scrape - only get new messages
        after_id = cached.get("last_message_id")
        new_urls, new_last_id = scrape_tag_from_channel(channel_id, tag, after_id)

        if new_urls:
            # Merge: new URLs first, then cached (preserving order, no duplicates)
            all_urls = new_urls + [
                u for u in cached.get("urls", []) if u not in new_urls
            ]
            last_id = new_last_id or after_id
            update_discord_cache(channel_id, tag, all_urls, last_id)
            return all_urls
        else:
            # No new messages, return cached
            cached_urls = cached.get("urls", [])
            print(f"📋 Returning {len(cached_urls)} cached URLs for #{tag}")
            return cached_urls
    else:
        # Full scrape
        urls, last_id = scrape_tag_from_channel(channel_id, tag)
        if urls and last_id:
            update_discord_cache(channel_id, tag, urls, last_id)
        return urls
