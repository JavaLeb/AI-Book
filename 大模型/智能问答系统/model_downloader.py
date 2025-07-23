from modelscope import snapshot_download

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('ZhipuAI/GLM-4-9B-0414', cache_dir=r'E:\LLM')

# # chatglm3-6b模型下载
# model_dir = snapshot_download('ZhipuAI/chatglm3-6b', cache_dir=r'E:\LLM')


# chatglm2-6b模型下载
model_dir = snapshot_download('ZhipuAI/chatglm2-6b', cache_dir=r'E:\LLM')
