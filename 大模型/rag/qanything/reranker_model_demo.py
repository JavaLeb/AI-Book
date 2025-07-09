from BCEmbedding import RerankerModel

# your query and corresponding passages
query = '填报安徽大学志愿应注意哪些问题？'
passages = [
    '首先，在充分了解我校《招生章程》基础上，根据自己的特长、志向、高考成绩、身体条件等因素综合考虑，选择适合自己的专业（类）；'
    '其次，我校专业录取没有级差，专业（类）分档按“分数优先（含国家政策加分），遵循志愿”的方式进行录取；'
    '第三，在“院校+专业组”改革省份，建议考生填报志愿时选择服从调剂，以降低平行志愿退档、滑档风险。'
    ,
    '1.学分制。2.多类型人才培养。3.拔尖创新人才培养。'
]

# construct sentence pairs
sentence_pairs = [(query, passage) for passage in passages]

# init reranker model
# 使用Huggingface.
# model = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")
# 使用本地RerankerModel.
model_path = r"E:\LLM\bce-reranker-base_v1"
model = RerankerModel(model_name_or_path=model_path)

# method 0: calculate scores of sentence pairs
scores = model.compute_score(sentence_pairs)
print('scores=', scores)

# method 1: rerank passages
rerank_results = model.rerank(query, passages)
print('rerank_results=', rerank_results)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = 'cuda'  # if no GPU, set "cpu"
model.to(device)

# get inputs
inputs = tokenizer(sentence_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

# calculate scores
scores = model(**inputs_on_device, return_dict=True).logits.view(-1, ).float()
scores = torch.sigmoid(scores)
print('scores=', scores)

from sentence_transformers import CrossEncoder

sentence_pairs = [[query, passage] for passage in passages]
# init reranker model
model = CrossEncoder(model_path, max_length=512)
# calculate scores of sentence pairs
scores = model.predict(sentence_pairs)
print('scores=', scores)
