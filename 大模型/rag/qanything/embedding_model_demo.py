from BCEmbedding import EmbeddingModel

# list of sentences
sentences = [
    '首先，在充分了解我校《招生章程》基础上，根据自己的特长、志向、高考成绩、身体条件等因素综合考虑，选择适合自己的专业（类）；'
    '其次，我校专业录取没有级差，专业（类）分档按“分数优先（含国家政策加分），遵循志愿”的方式进行录取；'
    '第三，在“院校+专业组”改革省份，建议考生填报志愿时选择服从调剂，以降低平行志愿退档、滑档风险。'
    ,
    '1.学分制。2.多类型人才培养。3.拔尖创新人才培养。'
]

# init embedding model
# 使用huggingface.
# model = EmbeddingModel(model_name_or_path="maidalun1020/bce-embedding-base_v1")
# 使用本地bce-embedding-base_v1.
model_path = r"E:\LLM\bce-embedding-base_v1"
model = EmbeddingModel(model_name_or_path=model_path)
# extract embeddings
embeddings = model.encode(sentences)
print('shape=', embeddings.shape)
print('embeddings', embeddings)

from transformers import AutoModel, AutoTokenizer

# init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

device = 'cuda'  # if no GPU, set "cpu"
model.to(device)

# get inputs
inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

# get embeddings
outputs = model(**inputs_on_device, return_dict=True)
embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize
print('shape=', embeddings.shape)
print('embeddings=', embeddings)

from sentence_transformers import SentenceTransformer

# init embedding model
model = SentenceTransformer(model_path)
# extract embeddings
embeddings = model.encode(sentences, normalize_embeddings=True)
print('shape=', embeddings.shape)
print('embeddings=', embeddings)
