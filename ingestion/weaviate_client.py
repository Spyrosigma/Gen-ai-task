import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure
from weaviate.classes.query import Filter, MetadataQuery
from typing import List, Dict, Any, Optional, Union

from config import WEAVIATE_API_KEY, WEAVIATE_REST_URL

class WeaviateClient:
    """Base client class for Weaviate operations.

    Attributes:
        wcd_url: Weaviate Cloud URL
        wcd_api_key: Weaviate Cloud REST API key

    """

    def __init__(self, wcd_url: str = None, wcd_api_key: str = None):
        """Initialize the Weaviate client.

        Args:
            wcd_url: Weaviate Cloud URL
            wcd_api_key: Weaviate Cloud REST API key
        """

        self.wcd_url = wcd_url
        self.wcd_api_key = wcd_api_key

        if not self.wcd_url or not self.wcd_api_key:
            raise KeyError("WEAVIATE_REST_URL and WEAVIATE_API_KEY are required")

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.wcd_url,
            auth_credentials=Auth.api_key(self.wcd_api_key),
            skip_init_checks=True,
        )

    def close(self):
        """Close the Weaviate client connection."""
        if self.client:
            self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic client closure."""
        self.close()


class CollectionManager(WeaviateClient):
    """Class for managing Weaviate collections."""

    def create_collection(
        self,
        collection_name: str,
        enable_multi_tenancy: bool = True,
        vectorizer_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
    ) -> str:
        """Create a collection with optional multi-tenancy and vectorizer.

        Args:
            collection_name: Name of the collection to create
            enable_multi_tenancy: Whether to enable multi-tenancy
            vectorizer_model: Vectorizer model name

        Returns:
            Status message
        """
        try:
            # Configure multi-tenancy if enabled
            multi_tenancy_config = None
            if enable_multi_tenancy:
                multi_tenancy_config = Configure.multi_tenancy(
                    enabled=True,
                    auto_tenant_creation=True,
                    auto_tenant_activation=True,
                )

            # Configure vectorizer
            vectorizer_config = [
                Configure.NamedVectors.text2vec_weaviate(
                    name="text",
                    source_properties=["text"],
                    model=vectorizer_model,
                )
            ]

        #     properties = [
        #     weaviate.classes.config.Property(name="text", data_type=weaviate.classes.config.DataType.TEXT)
        # ]

            # Create collection
            response = self.client.collections.create(
                name=collection_name,
                # properties=properties,  # Add the properties here
                multi_tenancy_config=multi_tenancy_config,
                vectorizer_config=vectorizer_config,
            )

            return response.name
            # return f"Collection '{collection_name}' created successfully with multi-tenancy and Snowflake vectorizer."

        except Exception as e:
            return f"Error creating collection: {e}"

    def list_collections(self, simple: bool = False) -> List[Dict]:
        """List all collections.

        Args:
            simple: Whether to return simplified collection information

        Returns:
            List of collection information
        """
        collections = self.client.collections.list_all(simple=simple)
        print(f"Collections: {collections}")
        return collections

    def get_collection(self, collection_name: str):
        """Get a collection by name.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection object
        """
        return self.client.collections.get(collection_name)

    def delete_collection(self, collection_name: str) -> str:
        """Delete a collection by name.

        Args:
            collection_name: Name of the collection

        Returns:
            Status message
        """
        try:
            response = self.client.collections.delete(collection_name)
            return response
            # return f"Collection '{collection_name}' deleted successfully."
        except Exception as e:
            return f"Error deleting collection: {e}"


class TenantManager(CollectionManager):
    """Class for managing tenants within collections."""

    def create_tenants(self, collection_name: str, tenant_list: List[str]) -> str:
        """Create multiple tenants in a collection.

        Args:
            collection_name: Name of the collection
            tenant_list: List of tenant names to create

        Returns:
            Status message
        """
        try:
            collection = self.get_collection(collection_name)
            for tenant in tenant_list:
                collection.tenants.create(tenant)
            return (
                f"Created {len(tenant_list)} tenants in collection '{collection_name}'"
            )
        except Exception as e:
            return f"Error creating tenants: {e}"

    def list_tenants(self, collection_name: str) -> List[Dict]:
        """List all tenants in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            List of tenant information
        """
        collection = self.get_collection(collection_name)
        return collection.tenants.get()


# Ingestion pipeline
class DataManager(TenantManager):
    """Class for managing data operations within collections and tenants."""

    def upload_objects(
        self, collection_name: str, data_objects: List[Dict], tenant: str
    ) -> str:
        """Upload data objects to a collection with specified tenant.

        Args:
            collection_name: Name of the collection
            data_objects: List of data objects to upload
            tenant: Tenant name

        Returns:
            Status message
        """
        try:
            # Get collection with specific tenant
            tenant_collection = self.get_collection(collection_name).with_tenant(tenant)

            # Batch import data objects to the tenant
            failed_objects = []
            with tenant_collection.batch.dynamic() as batch:
                for data_object in data_objects:
                    batch.add_object(properties=data_object)

                # Check for errors
                if batch.number_errors > 0:
                    failed_objects = tenant_collection.batch.failed_objects
                    return f"Partial import: {batch.number_errors} objects failed out of {len(data_objects)}"

            return f"Successfully imported {len(data_objects)} objects to tenant '{tenant}'"

        except Exception as e:
            return f"Error uploading objects: {e}"

    def delete_objects(
        self, collection_name: str, tenant: str, object_ids: List[str]
    ) -> str:
        """Delete objects by ID from a collection with specified tenant.

        Args:
            collection_name: Name of the collection
            tenant: Tenant name
            object_ids: List of object IDs to delete

        Returns:
            Status message
        """
        try:
            # Get collection with specific tenant
            collection = self.get_collection(collection_name).with_tenant(tenant)
            response = collection.data.delete_many(
                where=Filter.by_property("filename").contains_any(object_ids),
                verbose=True
            )

            return (
                f"Successfully deleted {len(object_ids)} objects from tenant '{tenant}'. \nRespone:--\n{response}"
            )

        except Exception as e:
            return f"Error deleting objects: {e}"


# Querying pipeline
class QueryManager(DataManager):
    """Class for querying data from collections and tenants."""

    def query_by_text(
        self,
        collection_name: str,
        tenant: str,
        query_text: str,
        filters: Optional[Filter] = None,
        limit: int = 5,
    ) -> List[Dict]:
        """Query objects by text similarity.

        Args:
            collection_name: Name of the collection
            tenant: Tenant name
            query_text: Text to search for
            filters: Optional filters
            limit: Maximum number of results

        Returns:
            List of matching objects
        """
        try:
            # Get collection with specific tenant
            collection = self.get_collection(collection_name)
            tenant_collection = collection.with_tenant(tenant)

            # Execute query
            response = tenant_collection.query.near_text(
                query=query_text,
                filters=filters,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
            )

            return response.objects

        except Exception as e:
            print(f"Error querying collection: {e}")
            return []

    def query_docs(
        self,
        collection_name: str,
        tenant: str,
        property_name: str,
        property_values: Dict[str,str],
    ) -> List[Dict]:
        """Query objects by property filters.

        Args:
            collection_name: Name of the collection
            tenant: Tenant name
            property_name: Property name to filter on
            property_values: List of property values to match

        Returns:
            List of strings matching the property values
        """
        # Create filter for multiple property values
        filters = Filter.any_of(
            [
                Filter.by_property(property_name).equal(value)
                for value in property_values.keys()
            ]
        )

        # Get collection with specific tenant
        collection = self.get_collection(collection_name)
        tenant_collection = collection.with_tenant(tenant)

        # Execute query
        response = tenant_collection.query.fetch_objects(filters=filters)
        
        # Create a dictionary to store results by filename
        results_by_file = {}
        
        # Process each object and organize by filename
        for obj in response.objects:
            filename = property_values[obj.properties["filename"]]
            text = obj.properties["text"]
            
            if filename not in results_by_file:
                results_by_file[filename] = {
                    'text': [],  # List to store text chunks
                }
            
            # Simply append text chunks without page ordering
            results_by_file[filename]['text'].append(text)
        
        # Return a dictionary of combined texts by filename
        return {filename: ' '.join(results_by_file[filename]['text']) for filename in results_by_file}


# if __name__ == '__main__':
#     with CollectionManager(wcd_api_key=WEAVIATE_API_KEY, wcd_url=WEAVIATE_REST_URL) as cm:
#         # Example usage
#         collection_name = "test_collection"
#         print(cm.create_collection(collection_name))
#         print(cm.list_collections())
#         # cm.delete_collection(collection_name)