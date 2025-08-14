import weaviate
from weaviate.classes.init import Auth
from contextlib import contextmanager


@contextmanager
def client():
    client = None
    try:
        client = weaviate.connect_to_local(
            host="localhost", port=8080,
            auth_credentials=Auth.api_key("123456")
        )
        yield client
    finally:
        if client is not None:
            client.close()
