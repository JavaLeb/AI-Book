import json
import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from langgraph.store.base import BaseStore, Item


class FileStore(BaseStore):
    """
    一个基于 JSON 文件的自定义 LangGraph Store 实现
    """

    def __init__(self, path: str = "store_data.json"):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    # ---------------------------- 内部工具 ----------------------------

    def _load(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict):
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _ns_key(self, namespace: tuple[str, ...], key: str) -> str:
        return json.dumps(namespace) + "::" + key

    # ---------------------------- 同步方法 ----------------------------

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Any = None,
    ) -> None:
        data = self._load()
        full_key = self._ns_key(namespace, key)
        data[full_key] = {
            "namespace": namespace,
            "key": key,
            "value": deepcopy(value),
        }
        self._save(data)

    def get(self, namespace: tuple[str, ...], key: str) -> Optional[Item]:
        data = self._load()
        item = data.get(self._ns_key(namespace, key))
        if not item:
            return None
        return Item(**item)

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        data = self._load()
        data.pop(self._ns_key(namespace, key), None)
        self._save(data)

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[dict] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Item]:
        if query is not None:
            raise NotImplementedError(
                "当前 FileStore 不支持语义搜索，请使用支持 vector search 的后端"
            )

        data = self._load()
        results: list[Item] = []

        for item in data.values():
            if tuple(item["namespace"])[:len(namespace_prefix)] == namespace_prefix:
                results.append(Item(**item))

        return results[offset : offset + limit]

    def list_namespaces(
        self,
        *,
        prefix: Optional[tuple[str, ...]] = None,
        suffix: Optional[tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        data = self._load()
        namespaces = set()

        for item in data.values():
            ns = tuple(item["namespace"])
            if prefix and ns[:len(prefix)] != prefix:
                continue
            if suffix and ns[-len(suffix):] != suffix:
                continue
            if max_depth and len(ns) > max_depth:
                continue
            namespaces.add(ns)

        return list(namespaces)[offset : offset + limit]

    # ---------------------------- 异步方法（必须） ----------------------------

    async def aput(self, *args, **kwargs):
        return await asyncio.to_thread(self.put, *args, **kwargs)

    async def aget(self, *args, **kwargs):
        return await asyncio.to_thread(self.get, *args, **kwargs)

    async def adelete(self, *args, **kwargs):
        return await asyncio.to_thread(self.delete, *args, **kwargs)

    async def asearch(self, *args, **kwargs):
        return await asyncio.to_thread(self.search, *args, **kwargs)

    async def alist_namespaces(self, *args, **kwargs):
        return await asyncio.to_thread(self.list_namespaces, *args, **kwargs)


import pytest
from langgraph.store.memory import InMemoryStore



@pytest.fixture
async def store():
    yield FileStore("test_store.json")


@pytest.fixture
def reference():
    return InMemoryStore()


async def test_put_and_get(store, reference):
    ns = ("test", "ns")
    for s in [store, reference]:
        await s.aput(ns, "k1", {"val": 1})
        item = await s.aget(ns, "k1")
        assert item is not None
        assert item.value == {"val": 1}


async def test_delete(store, reference):
    ns = ("test", "ns")
    for s in [store, reference]:
        await s.aput(ns, "k1", {"val": 1})
        await s.adelete(ns, "k1")
        assert await s.aget(ns, "k1") is None


async def test_search_prefix(store, reference):
    for s in [store, reference]:
        await s.aput(("user", "memories"), "m1", {"text": "likes pizza"})
        results = await s.asearch(("user",))
        assert any(r.key == "m1" for r in results)