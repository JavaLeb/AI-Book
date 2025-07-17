from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 查看数据库列表.
list_databases = client.list_databases()
print('list_databases=', list_databases)

db_name = "demo"
# 删除数据库.
if db_name in list_databases:
    drop_database = client.drop_database(
        db_name=db_name
    )
    print('drop_database=', drop_database)
# 创建数据库.
create_database = client.create_database(db_name=db_name,
                                         properties={
                                             "database.replica.number": 3
                                         })
print('create_database=', create_database)

# 增加数据库属性.
client.alter_database_properties(
    db_name=db_name,
    properties={"database.max.collections": 10})
# 修改数据库已有属性.
client.alter_database_properties(
    db_name=db_name,
    properties={"database.replica.number": 2})

# 查看数据库属性.
describe_database = client.describe_database(db_name=db_name)
print('describe_database=', describe_database)
# 使用数据库.
client.use_database(db_name=db_name)

client.close()
