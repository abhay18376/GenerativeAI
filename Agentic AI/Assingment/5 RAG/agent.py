from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from datetime import datetime
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.cohere import Cohere

# Simple ReAct-style tool-using agent for competitive analysis.
@dataclass
class CompetitiveAnalysisAgent:
    index: VectorStoreIndex
    cohere_api_key: str
    history: List[Tuple[str, str]] = field(default_factory=list)  # (user, assistant)
    reasoning_log_path: str = "logs/reasoning.jsonl"

    def _log(self, record: Dict[str, Any]) -> None:
        record["ts"] = datetime.utcnow().isoformat()
        with open(self.reasoning_log_path, "a", encoding="utf-8") as f:
            f.write(json_dumps(record) + "\n")

    def _classify_intent(self, query: str) -> str:
        q = query.lower()
        if "compare" in q or "vs" in q or "versus" in q:
            return "comparison"
        if "strength" in q or "weakness" in q or "swot" in q:
            return "swot"
        if "marketing" in q:
            return "marketing"
        if "product" in q:
            return "product"
        if "financial" in q or "arr" in q or "revenue" in q:
            return "financial"
        return "overview"

    def _pick_k(self, intent: str) -> int:
        # Simple adaptive behavior for retrieval depth
        return {
            "overview": 2,
            "marketing": 3,
            "product": 3,
            "financial": 3,
            "swot": 4,
            "comparison": 6
        }.get(intent, 3)

    def _retrieve(self, query: str, k: int) -> List[str]:
        retriever = self.index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        texts = [n.get_text() for n in nodes]
        return texts

    def _synthesize(self, query: str, contexts: List[str]) -> str:
        # Use Cohere LLM via LlamaIndex wrapper
        llm = Cohere(api_key=self.cohere_api_key, model="command-a-03-2025")
        prompt = (
            "You are a competitive analysis assistant. "
            "Given the user's query and retrieved context snippets, produce a concise, well-structured answer. "
            "Cite competitors explicitly and keep to business-relevant insights.\n\n"
            f"USER QUERY:\n{query}\n\n"
            "CONTEXT SNIPPETS:\n" + "\n\n---\n".join(contexts) + "\n\n"
            "RESPONSE:\n"
        )
        resp = llm.complete(prompt)
        return str(resp)

    def reason_and_act(self, query: str) -> str:
        intent = self._classify_intent(query)
        k = self._pick_k(intent)
        plan = [
            {"step": "classify_intent", "intent": intent},
            {"step": "retrieve", "k": k},
            {"step": "synthesize", "model": "command-a-03-2025"}
        ]
        # Log plan
        self._log({"event": "plan", "query": query, "plan": plan})

        # Act: retrieve
        contexts = self._retrieve(query, k=k)
        self._log({"event": "retrieval", "retrieved": len(contexts)})

        # Act: synthesize
        answer = self._synthesize(query, contexts)
        self._log({"event": "answer", "answer_preview": answer[:200]})

        # Update history
        self.history.append((query, answer))
        return answer

    def get_history(self, n: int = 10) -> List[Tuple[str, str]]:
        return self.history[-n:]

# Small helper because json might be missing if the user runs this in a very limited env
def json_dumps(obj: Any) -> str:
    try:
        import json as _json
        return _json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)