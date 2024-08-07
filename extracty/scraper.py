import logging

from bs4 import BeautifulSoup
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

from pydantic import HttpUrl
import extract_html_dark_webpages

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class WebScraper:
    def __init__(self, url: HttpUrl):
        self.url = url

    def __clean_html_content(
        self,
        html_content: str,
        wanted_tags: list[str],
        unwanted_tags: list[str] = ["script", "style", "nav", "ul", "li", "menu", "header", "footer", "aside", "ad", "form", "input", "button"],
    ) -> str:
        """
        Cleans the HTML content by removing unwanted tags, extracting text from wanted tags,
        and removing unnecessary lines.

        Args:
            html_content (str): The HTML content to be cleaned.
            wanted_tags: The list of tags from which to extract text.
            unwanted_tags (list[str], optional): The list of unwanted tags to be removed.
                Defaults to ["script", "style"].

        Returns:
            str: The cleaned content.

        """
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()

        text_parts = []
        for tag in wanted_tags:
            elements = soup.find_all(tag)
            for element in elements:
                if tag == "a":
                    href = element.get("href")
                    text_parts.append(
                        f"{element.get_text()} ({href})" if href else element.get_text()
                    )
                else:
                    text_parts.append(element.get_text())

        content = " ".join(text_parts)
        lines = content.split("\n")
        stripped_lines = [line.strip() for line in lines]
        non_empty_lines = [line for line in stripped_lines if line]
        seen = set()
        deduped_lines = [
            line for line in non_empty_lines if not (line in seen or seen.add(line))
        ]
        cleaned_content = " ".join(deduped_lines)

        return cleaned_content

    def scraping_with_langchain(
        self, wanted_tags: list[str] = ["title", "meta", "span", "div", "a"]
    ):
        """
        Scrapes the content of a web page using Requests.

        Args:
            wanted_tags (list[str], optional): List of HTML tags to extract from the page. Defaults to ["h1", "h2", "h3", "span", "p"].

        Returns:
            str: The cleaned HTML content of the page.

        Raises:
            Exception: If any error occurs during the scraping process.
        """
        # ["h1", "h2", "h3", "span", "p", "div", "a", "title", "meta"]
        # ["h2", "div", "span", "tbody", "em"]
        try:
            # loader = AsyncHtmlLoader([self.url])
            # docs = loader.load() # loader is an instance of AsyncHtmlLoader, docs is List[Document]
            # # docs[0] is a Document, docs[0].page_content is a String
            # cleaned_content = self.__clean_html_content(
            #     docs[0].page_content, wanted_tags
            # )
            # # we can change docs[0].page_content input to a String from a JSON file saved after extract_dark_webpages
            # # unless we simply return a String from extract_dark_webpages, then we can import it into scraper.py
            # # without saving anything to a JSON file
            # # something like: extract_dark_webpages.run(url)
            # return cleaned_content
            
            content = extract_html_dark_webpages.run(self.url)
            cleaned_content = self.__clean_html_content(
                content, wanted_tags
            )
            
            return cleaned_content
        
        except Exception as e:
            logging.error(f"Scraping Error: {e}")
            raise

    async def ascraping_with_playwright(
        self,
        wanted_tags: list[str] = ["h1", "h2", "h3", "span", "p"],
    ):
        """
        Scrapes the content of a web page using Playwright.

        Args:
            wanted_tags (list[str], optional): List of HTML tags to extract from the page. Defaults to ["h1", "h2", "h3", "span", "p"].

        Returns:
            str: The cleaned HTML content of the page.

        Raises:
            TimeoutError: If a timeout occurs during the scraping process.
            Exception: If any other error occurs during the scraping process.
        """
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                page.set_default_navigation_timeout(60000)
                await page.goto(self.url)
                page_source = await page.content()
                cleaned_content = self.__clean_html_content(page_source, wanted_tags)
                return cleaned_content
        except PlaywrightTimeoutError as e:
            logging.error(f"Playwright Timeout Error: {e}")
            raise
        except Exception as e:
            logging.error(f"Scraping Error: {e}")
            raise
