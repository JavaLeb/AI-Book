from weaviate_client import client

with client() as client:
    # 读取单个集合.
    articles = client.collections.get("Article")
    articles_config = articles.config.get()
    print(articles_config)

    # 读取所有集合.
    response = client.collections.list_all(simple=False)
    print(response)
