import vanna
from vanna.remote import VannaDefault

# 需要先注册.
vn = VannaDefault(model='chinook', api_key=vanna.get_api_key('liuerbao@163.com'))

# 使用在线提供的.
vn.connect_to_sqlite('https://vanna.ai/Chinook.sqlite')
# # 使用本地.
# vn.connect_to_sqlite('Chinook.sqlite')

vn.ask("What are the top 10 albums by sales?")

from vanna.flask import VannaFlaskApp
VannaFlaskApp(vn).run()