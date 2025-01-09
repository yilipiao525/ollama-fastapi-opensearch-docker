from opensearchpy import OpenSearch, RequestsHttpConnection
import requests

class OpenSearchClient:
    def __init__(self, host: str = "opensearch", port: int = 9200, use_ssl: bool = False):
        """
        Basic client for interacting with an OpenSearch cluster.
        
        :param host: Hostname of the OpenSearch service (Docker container name or IP).
        :param port: Port on which OpenSearch is running.
        :param use_ssl: Whether to use SSL for connections (False for local dev).
        """
        self.host = host
        self.port = port
        self.use_ssl = use_ssl

        # Construct the connection URL
        scheme = "https" if use_ssl else "http"
        self.client = OpenSearch(
            hosts=[{"host": self.host, "port": self.port}],
            scheme=scheme,
            connection_class=RequestsHttpConnection,
            verify_certs=use_ssl
        )

    def ping(self) -> bool:
        """Check if the OpenSearch server is alive."""
        return self.client.ping()

    def create_index(self, index_name: str, body: dict) -> None:
        """
        Create an index if it doesn't already exist.
        
        :param index_name: Name of the index to create
        :param body: Index settings and mappings
        """
        if not self.client.indices.exists(index=index_name):
            response = self.client.indices.create(index=index_name, body=body)
            print(f"Created index {index_name}: {response}")
        else:
            print(f"Index {index_name} already exists.")

    def index_document(self, index_name: str, doc_id: str, document: dict) -> None:
        """
        Index a single document into an index.
        
        :param index_name: The index to add the document
        :param doc_id: Document ID
        :param document: The document body (dict) to index
        """
        response = self.client.index(index=index_name, id=doc_id, body=document)
        print(f"Indexed doc {doc_id} to {index_name}: {response['result']}")

    def search_documents(self, index_name: str, query: dict, size: int = 10) -> dict:
        """
        Search documents within an index using a query body.
        
        :param index_name: The index to search
        :param query: A dict specifying the query
        :param size: Number of results to return
        :return: Raw search response (dict)
        """
        response = self.client.search(index=index_name, body=query, size=size)
        return response

    def delete_index(self, index_name: str) -> None:
        """Delete an existing index."""
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
            print(f"Deleted index {index_name}")
        else:
            print(f"Index {index_name} does not exist.")

    def get_index_info(self, index_name):
        """Get information about an index"""
        return self.client.indices.get(index=index_name)

    def get_index_stats(self, index_name):
        """Get statistics about an index"""
        return self.client.indices.stats(index=index_name)

    def list_indices(self):
        """List all indices"""
        return list(self.client.indices.get_alias().keys())
