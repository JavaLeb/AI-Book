from langchain.rate_limiters import InMemoryRateLimiter
from langchain.chat_models import init_chat_model
import time

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 1 request every 10s
    check_every_n_seconds=0.1,  # Check every 100ms whether allowed to make a request
    max_bucket_size=1,  # Controls the maximum burst size.
)

model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    rate_limiter=rate_limiter
)

for i in range(4):
    tic = time.time()
    response = model.invoke("Say hello briefly.")
    toc = time.time()
    print(f"Call {i + 1}: 耗时 {toc - tic:.2f}s | 回复: {response.content[:40]!r}")
