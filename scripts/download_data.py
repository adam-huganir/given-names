"""
Python port of the PHP script to generate a CSV of given names and their diminutives
using the English Wiktionary.
"""

import argparse
import asyncio
import csv
import html
import json
import os
import re
import sys
from pathlib import Path
import logging
from typing import Any

import httpx
import unicodedata
from tqdm.asyncio import tqdm

from given_names.util import multidict

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
                    level=os.getenv("LOG_LEVEL", "INFO").upper())
logger.info("Starting script...")

logging.getLogger("httpx").setLevel(logging.WARNING)

MEDIAWIKI_API_URL = "https://en.wiktionary.org/w/api.php"
USER_AGENT = (
    f"bb-names-diminutives-script/0.1 (https://github.com/your-repo; your-email) httpx/{httpx.__version__} "
    f"python/{sys.version_info.major}.{sys.version_info.minor}"
)
OUTPUT_DIR = Path("./gen")
REQUEST_DELAY_SECONDS = 3  # Be nice to Wiktionary
MAX_TITLES_PER_REQUEST = 30  # MediaWiki API limit is often 50, but let's be conservative
CONCURRENT_CONNECTIONS = asyncio.Semaphore(4)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a CSV file of formal given names and their common diminutives."
    )
    parser.add_argument(
        "-g",
        "--gender",
        required=True,
        choices=["male", "female", "m", "f"],
        help='Specify the gender ("male" or "female" only) for which to generate diminutives.',
    )
    return parser.parse_args()


async def get_category_members(client: httpx.AsyncClient, gender: str) -> list[str]:
    """Fetches all page titles from the relevant Wiktionary category."""
    logger.info(f"Fetching category members for '{gender}'...")
    category = f"Category:English diminutives of {gender} given names"
    titles = []
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": "max",  # Fetch max allowed per request
        "cmprop": "title",
        "formatversion": 2,  # Use modern format
    }
    cmcontinue = None

    # Use tqdm for progress indication, no total since we don't know the number of pages beforehand
    pbar = tqdm(desc="Fetching category pages", unit="page")
    while True:
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        else:
            params.pop("cmcontinue", None)

        try:
            async with CONCURRENT_CONNECTIONS:
                response = await client.get(MEDIAWIKI_API_URL, params=params)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()

            if "error" in data:
                logger.info(f"API Error: {data['error']['info']}")
                break

            batch_titles = [
                page["title"] for page in data.get("query", {}).get("categorymembers", [])
            ]
            titles.extend(batch_titles)
            # logger.info(f"  Fetched {len(batch_titles)} titles...") # Replaced by tqdm
            pbar.update(1)  # Update progress bar by one page (batch)

            if "continue" in data:
                cmcontinue = data["continue"]["cmcontinue"]
                # logger.info("  Continuing...") # Replaced by tqdm
                await asyncio.sleep(REQUEST_DELAY_SECONDS)
            else:
                break  # No more pages

        except httpx.RequestError as exc:
            logger.info(f"HTTP Request failed: {exc}")
            break
        except json.JSONDecodeError:
            logger.info("Failed to decode JSON response.")
            break

    pbar.close()
    logger.info(f"Finished fetching category members. Total titles: {len(titles)}")
    return titles


async def fetch_page_contents(client: httpx.AsyncClient, titles: list[str], gender: str) -> dict[str, str]:
    """Fetches the latest revision content for a list of page titles."""
    if (filepath := OUTPUT_DIR / f"raw_mediawiki_{gender}_diminutives.json").exists():
        logger.info("Output file already exists, skipping fetch.")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"Fetching content for {len(titles)} pages...")
    contents = {}
    total_batches = (len(titles) + MAX_TITLES_PER_REQUEST - 1) // MAX_TITLES_PER_REQUEST
    # Split titles into chunks to avoid overly long URLs / large requests
    # Wrap the loop with tqdm for progress
    async for i in tqdm(range(0, len(titles), MAX_TITLES_PER_REQUEST), total=total_batches,
                        desc="Fetching page content", unit="batch"):
        chunk = titles[i: i + MAX_TITLES_PER_REQUEST]
        # logger.info(f"  Fetching batch {i // MAX_TITLES_PER_REQUEST + 1} ({len(chunk)} titles)...") # Replaced by tqdm
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "titles": "|".join(chunk),
            "rvprop": "content",
            "rvslots": "main",  # Request content from the main slot
            "formatversion": 2,
        }
        url = httpx.URL(MEDIAWIKI_API_URL, params=params)
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.info(f"API Error: {data['error']['info']}")
                continue  # Skip this batch

            pages = data.get("query", {}).get("pages", [])
            for page in pages:
                if page.get("missing") or "revisions" not in page:
                    logger.info(f"  Warning: Title '{page['title']}' missing or has no revision.")
                    continue
                # Get content from the first revision's main slot
                content = page["revisions"][0]["slots"]["main"]["content"]
                # Normalize Unicode (NFC form) and decode HTML entities
                normalized_content = unicodedata.normalize("NFC", html.unescape(content))
                contents[page["title"]] = normalized_content

        except httpx.RequestError as exc:
            logger.info(f"HTTP Request failed for batch: {exc}")
        except json.JSONDecodeError:
            logger.info("Failed to decode JSON response for batch.")
        except KeyError as e:
            logger.info(f"Unexpected API response structure: missing key {e}")

        # Delay between batches
        if i + MAX_TITLES_PER_REQUEST < len(titles):
            await asyncio.sleep(REQUEST_DELAY_SECONDS)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(contents, f, ensure_ascii=False, indent=2)
    logger.info(f"Finished fetching content. Retrieved content for {len(contents)} pages.")
    return contents

def handle_directives(directive: str , params: multidict[str, Any]):
    match directive:
        case "l":
            return

def parse_wiktionary_directives(content: str) -> multidict[str, Any]:
    directives = multidict()
    for d in re.finditer(r"\{\{(?P<directive>[^}]+)}}", content):
        directive, *params = re.split(r"(?<=[^\\])\|", d["directive"])

        assert directive[0] not in ["'\""], f"Didn't expect to handle quotes, fix this: {d.string!r}"
        keyed_params = multidict()
        for param in params:
            key, _, value = param.partition("=")
            keyed_params[key.strip()] = value.strip()
        directives[directive.strip()] = keyed_params
    return directives

def parse_wikitext(title: str, content: str, gender: str) -> dict[str, list[str]]:
    """
    Parses the wikitext of a diminutive page to find the formal names it relates to.

    This is the most complex and brittle part, highly dependent on Wiktionary templates
    and formatting conventions, which change over time.
    This implementation is a starting point based on common patterns like:
    * {{diminutive of|FormalName|lang=en}}
    * ==English== sections with definitions
    * Lists mentioning formal names

    Returns: A dictionary mapping formal names found to the diminutive (the page title).
             Example: {"William": ["Bill"], "Wilhelmina": ["Bill"]}
    """
    # TODO: Implement robust wikitext parsing.
    # This will likely involve regex to find relevant sections (e.g., ==English==)
    # and templates (e.g., {{diminutive of|...}}, {{given name|...}}).
    # It's complex because wikitext structure is not strictly defined.

    formal_names = []

    # Example: Find {{diminutive of|FormalName|lang=en}} variants
    # Regex needs refinement to handle variations in template parameters and whitespace.
    _pattern_2010 = r"\{\{\s*(?:diminutive of|diminutive|dim)\s*\|(?:lang=en\s*\|)?\s*([^|}]+)"
    pattern_2025 = r"\{\{(?P<type>given\s+name)\|(?P<lang>[^\}\|]+)\|(?P<gender>[^\|]+?)\|(dimof=)(?P<name>[^\}\|]+?)\}\}"
    matches = re.finditer(pattern_2025, content, re.IGNORECASE)
    for name in matches:
        # check some assumptions for the new pattern matching
        assert name["type"] == "given name"
        lang = name.group("lang").strip()
        assert lang
        gender_name = name.group("gender").strip()
        assert gender_name
        name = name.group("name").strip()
        assert name
        if "," in name:
            names = name.split(",")
        else:
            names = [name]
        for n in names:
            # Basic cleanup - remove potential wiki links [[...]] or formatting ''...''
            cleaned_name = re.sub(r"\[\[([^]|]+)(?:\|[^]]+)?]]", r"\1", n).strip()
            cleaned_name = re.sub(r"''", "", cleaned_name).strip()
            if cleaned_name and cleaned_name != "...":  # Avoid placeholder/empty names
                formal_names.append((cleaned_name, gender_name, lang))
            else:
                raise ValueError(
                    f"Invalid name '{n}' found in page '{title}' with content '{content}'."
                )

    # Placeholder for more advanced parsing (e.g., analyzing ==English== sections)
    # This would require more sophisticated regex or potentially a wikitext parser library
    # if the simple template matching is insufficient.

    # We map each found formal name back to the current diminutive (page title)
    if title == "Abby":
        breakpoint()
    results = {
        "female": {
            formal: [title] for formal, gender_name, lang in set(formal_names) if formal and gender_name == "female"
        },
        "male": {
            formal: [title] for formal, gender_name, lang in set(formal_names) if formal and gender_name == "male"
        }
    }
    other = (set(results) - {gender}).pop()

    if not results[gender]:
        logger.info(
            f"  No diminutives found in page '{title}' for {gender} given names. {len(results[other])} {other} names found."
        )
    return results[gender]


async def main():
    args = parse_arguments()
    match args.gender:
        case "male" | "m":
            gender = "male"
        case "female" | "f":
            gender = "female"
        case _:
            raise ValueError("Invalid gender specified.")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use httpx with appropriate User-Agent
    headers = {"User-Agent": USER_AGENT.format(httpx=httpx, sys=__import__("sys"))}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=30.0, ) as client:
        # 1. Get category members
        titles = await get_category_members(client, gender)
        if not titles:
            logger.info("No titles found in category, exiting.")
            return

        # Optional: Save raw titles list
        cat_filename = OUTPUT_DIR / f"English_diminutives_of_{gender}_given_names.txt"
        logger.info(f"Saving raw title list to {cat_filename}...")
        with open(cat_filename, "w", encoding="utf-8") as f:
            for title in titles:
                f.write(title + "\n")

        # 2. Fetch content for each title
        contents = await fetch_page_contents(client, titles, gender)
        if not contents:
            logger.info("No page content fetched, exiting.")
            return

        # 3. Parse content and build the map
        logger.info("Parsing wikitext for diminutives...")
        # Structure: { "Formal Name": ["Diminutive1", "Diminutive2", ...], ... }
        formal_to_diminutives = {}
        total_pages = len(contents)
        diminutive_map = {}
        count = 0

        logger.info(f"Parsing wikitext for {total_pages} pages...")
        # Wrap the loop with tqdm
        for title, content in tqdm(contents.items(), total=total_pages, desc="Parsing wikitext", unit="page"):
            # count += 1
            # if count % 50 == 0 or count == total_pages:
            #     logger.info(f"  Processed {count}/{total_pages} pages...") # Replaced by tqdm

            parsed_data = parse_wikitext(title, content, gender)
            # Merge results: if a formal name already exists, append the new diminutive
            for formal_name, diminutives_list in parsed_data.items():
                if formal_name not in diminutive_map:
                    diminutive_map[formal_name] = []
                diminutive_map[formal_name].extend(diminutives_list)

        logger.info(f"Finished parsing. Found mappings for {len(diminutive_map)} formal names.")

        # 4. Write to CSV
        csv_filename = OUTPUT_DIR / f"{gender}_diminutives.csv"
        logger.info(f"Writing results to {csv_filename}...")
        try:
            with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Formal Name", "Diminutive(s)"])
                # Sort by formal name for consistent output
                for formal_name in sorted(diminutive_map.keys()):
                    # Join multiple diminutives with a semicolon
                    diminutives_str = "; ".join(
                        sorted(list(set(diminutive_map[formal_name]))))  # Ensure unique diminutives
                    writer.writerow([formal_name, diminutives_str])

            logger.info(f"Successfully wrote {len(diminutive_map)} formal names to {csv_filename}")
        except IOError as e:
            logger.info(f"Error writing CSV file: {e}")


if __name__ == "__main__":
    asyncio.run(main())
