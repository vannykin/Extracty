import asyncio

from pydantic import BaseModel, Field, create_model
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from groq import Groq as Client

from extracty import WebScraper

from typing import Type, TypeVar

T = TypeVar("T", bound=BaseModel)
DataT = TypeVar("DataT")


class BaseExtractor(BaseModel):
    """
    Base schema for extractors.

    Pydantic schemas will be used with the help of instructor to make structured prompts and it will make the extraction process easier.

    Attributes:
        name (str): The name of the extracted thing.
        data (list[dict[str, str]]): The important data to be extracted.
    """

    name: str = Field(
        ...,
        description="Only the general name of extracted thing",
        examples=["latest_stock_details", "trending_news"],
    )
    data: DataT = Field(
        ...,
        description="The important data to be extracted, if the data is huge then it should be a list of dictionaries",
        examples=[
            {"name": "stock_name", "value": "Apple Inc."},
            {"name": "stock_price", "value": "$150.00"},
        ],
    )


class LLMExtractor:
    def __init__(
        self,
        query: str,
        url: str,
        client: Client,
        # fields: dict[str, Type] | None = None,
    ):
        """
        Initializes an instance of the LLMExtractor class.

        Args:
            query (str): The query string used for extraction.
            url (str): The URL of the webpage to extract data from.
            api_key (str): The OpenAI api key for accessing the extraction service.
            gpt_model (str): The GPT model to use for extraction, defaults to "gpt-4".
            fields: dict[str, Type] | None: A dictionary containing the field names and their corresponding types, defaults to None.

        """ 
        self.query = query
        self.url = url
        self.client = client
        # self.fields = fields

    def __get_content(self) -> str:
        """
        Retrieves the content of a web page using a WebScraper object.

        Returns:
            str: The content of the web page.

        Raises:
            TimeoutError: If the scraping process times out or the page takes too long to load.
            Exception: If any other exception occurs during the scraping process.
        """
        scraper = WebScraper(self.url)
        try:
            # content = await scraper.ascraping_with_playwright()
            content = scraper.scraping_with_langchain()
            return content
        except PlaywrightTimeoutError as pte:
            raise TimeoutError(
                "The scraping process timed out. Or the page took too long to load. Please try again later."
            )
        except Exception as e:
            raise e

    def __create_pydantic_model(self, fields: dict[str, Type]) -> Type[T]:
        """
        Create a Pydantic model dynamically based on fields provided.

        Args:
            fields (Dict[str, Type]): A dictionary containing the field names and their corresponding types.

        Returns:
            Type[T]: The dynamically created Pydantic model.

        """
        data_model = create_model(
            "DataModel",
            **{
                field_name: (field_type, Field(...))
                for field_name, field_type in fields.items()
            },
        )

        dynamic_model = create_model(
            "CustomExtractor",
            name=(str, Field(..., description="Name of the item")),
            data=(list[data_model], Field(..., description="The dynamic data fields")),
        )
        return dynamic_model

    def __generate_prompt(self, content: str) -> list[dict]:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful extractor that extract and structure data.",
            },
            {
                "role": "user",
                "content": f"Can you {self.query} from {content}",
            },
        ]

        return messages

    def __call_openai(
        self, prompt: list[dict], client: Client) -> str:
        # c = instructor.patch(client)
        # response = c.chat.completions.create(
        #     model=gpt_model,
        #     messages=prompt,
        #     response_model=pydantic_schema,
        #     temperature=0.125,
        # )
        # return response
        
        response = client.chat.completions.create(
            messages=prompt,
            model="llama3-70b-8192",# Or "Llama-3-8B" depending on your preference
        )

        # generated_text = response["choices"][0].message.content
        generated_text = response.choices[0].message.content
        
        return generated_text

    def __async_run_content(self) -> str:
        """
        Runs the __get_content method asynchronously and returns the content.

        Returns:
            str: The content obtained from the __get_content method.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        content = loop.run_until_complete(self.__get_content())
        loop.close()
        return content

    def extract(self) -> str:
        """
        Extracts data from a web page using the OpenAI API.

        Returns:
            dict: The extracted data.

        Raises:
            TimeoutError: If the scraping process times out or the page takes too long to load.
            Exception: If any other exception occurs during the scraping process.
        """
        # content = self.__async_run_content()
        content = self.__get_content()

        # pydantic_schema = (
        #     self.__create_pydantic_model(fields=self.fields)
        #     if self.fields
        #     else BaseExtractor
        # )

        prompt = self.__generate_prompt(content)

        response = self.__call_openai(
            prompt=prompt,
            # pydantic_schema=pydantic_schema,
            client=self.client,
        )

        # TODO: implement more logic to handle response and create a structured output
        return response