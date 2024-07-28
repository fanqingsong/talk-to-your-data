import pprint

import requests
from typing import List
from langchain_core.embeddings import Embeddings


class CustomAPIEmbeddings(Embeddings):
    def __init__(self, model_name: str, api_url: str):
        self.model_name = model_name
        self.api_url = api_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            headers={'Authorization': 'Bearer your_token_here'},
            json={
                "model": self.model_name,
                "input": texts,
            },
        )

        pprint.pprint(response.json())

        data_list = response.json()['data']  # Adjust this based on the response format of your API
        ans = [one['embedding'] for one in data_list]

        return ans

    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]
        #
        # response = requests.post(
        #     self.api_url,
        #     headers={'Authorization': 'Bearer your_token_here'},
        #     json={
        #         "model": self.model_name,
        #         "input": text,
        #     },
        # )
        #
        # ret = response.json()  # Adjust this based on the response format of your API
        # pprint.pprint(ret)
        #
        # return ret['data'][0]['embedding']


if __name__ == '__main__':
    embeddings = CustomAPIEmbeddings(
        model_name="Xenova/text-embedding-ada-002",
        api_url="http://192.168.0.108:1234/v1/embeddings",
        # api_key="sss"
    )

    query = "What is the cultural heritage of India?"
    query_embedding = embeddings.embed_query(query)
    pprint.pprint(query_embedding)
