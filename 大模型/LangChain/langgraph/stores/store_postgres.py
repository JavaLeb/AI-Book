from langgraph.store.postgres import PostgresStore

DB_URI = 'postgresql://postgres:root@localhost:5432/postgres'

with PostgresStore.from_conn_string(DB_URI) as store:
    # ⚠️ 首次使用必须执行，创建 store 相关表
    store.setup()
    namespace = ("users", "user-001")
    # 写入
    store.put(namespace, "preferences", {"language": "zh-CN", "level": "入门"})
    # 读取
    items = store.get(namespace, "preferences")
    print('items=', items)
    # 搜索（精确匹配命名空间前缀）
    items = store.search(("users",))
    for i in items:
        print('key=', i.key, 'value=', i.value)
