from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path='data/document_loader.txt',
    csv_args={
        'delimiter': '|',
        'fieldnames': ['id', 'name', 'age']
    }
)

# 每一行是Document对象.
documents = loader.load()


for document in documents:
    print('type=', type(document))
    print('document=', document)
print('--------------------------------------------')
for document in loader.lazy_load():
    print('document=', document)
