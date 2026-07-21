"""
Hybrid research graph (Corrective/Adaptive RAG pattern, with web-result caching).

This is the piece that guarantees the "check the vector DB first, only go to
the web if needed" behavior deterministically — rather than leaving it up to
an agent's discretion whether to call a tool. It also means live web search
results don't just get thrown away: they're written back into a dedicated
"web cache" vector index, so a similar future question can be answered from
that cache instead of hitting the web again.

Flow:
    retrieve (society docs)
        -> grade
            --sufficient--> generate
            --insufficient--> check_web_cache
                -> grade_cache
                    --sufficient--> generate            (reused, no API call)
                    --insufficient--> live_web_search
                        -> store_web_cache -> generate  (cached for next time)

The society-document index and the web-cache index are kept strictly
separate (see vectorstore.py) so this graph is the only thing that ever
reads/writes the cache — it never leaks into `/ask` (docs-only Q&A).
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from backend import config, prompts
from backend.vectorstore import VectorStore
from backend.tools import web_search as web_search_tool
from backend.agent import Agent

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if the retrieved excerpts are sufficient to answer "
        "the question, otherwise 'no'."
    )

class ResearchState(TypedDict):
   question: str
   doc_context: str
   cache_context: str
   web_context: str
   needs_web: bool
   needs_live_search: bool
   used_cache: bool
   used_live_search: bool
   answer: str

class HybridResearchGraph:
    def __init__(self):
        self._llm=None
        self._agent = None
        self._graph = None

    def _get_llm(self):
        if self._llm is None:
            self._agent = Agent()
            self._llm = self._agent.get_llm()
        return self._llm 
    
    def _grade(self, question: str, context: str) -> bool:
        if not context.strip():
            return False

        llm=self._get_llm()
        grader = llm.with_structured_output(GradeDocuments)
        messages = prompts.GRADE_PROMPT.format_messages(question=question, context=context)
        result = grader.invoke(messages)
        return result.binary_score.strip().lower() == "yes"

    def retrieve_node(self, state: ResearchState)->ResearchState:
        vectorstore: VectorStore = VectorStore()
        retriever = vectorstore.get_retriever()
        docs = retriever.invoke(state["question"]) if retriever else []
        context = vectorstore.format_docs(docs) if docs else ""
        return {**state, "doc_context": context}  

    def grade_node(self, state: ResearchState) -> ResearchState:
        sufficient = self._grade(state["question"], state["doc_context"])
        return { **state, "needs_web": not sufficient}

    def route_after_grade(self, state: ResearchState) -> str:
        return "generate" if not state["needs_web"] else "check_web_cache"

    def check_web_cache_node(self, state: ResearchState) -> ResearchState:
        if not config.ENABLE_WEB_CACHE:
            return {
                **state, 
                "cache_context": "", 
                "needs_live_search": True
            }

        vectorstore: VectorStore = VectorStore()
        retriever = vectorstore.get_web_cache_retriever()
        docs = retriever.invoke(state["question"]) if retriever else[]
        context = vectorstore.format_docs(docs) if docs else ""
        return {**state, "cache_context": context}

    def grade_cache_node(self, state: ResearchState) -> ResearchState:
        sufficient = self._grade(state["question"], state["cache_context"])
        return {
                **state,
                "needs_live_search": not sufficient,
                "used_cache": sufficient,
                "web_context": state["cache_context"] if sufficient else "",
        }

    def route_after_cache_grade(self, state: ResearchState) -> str:
        return "generate" if not state["needs_live_search"] else "live_web_search"

    def live_web_search_node(self, state: ResearchState) -> ResearchState:
        result = web_search_tool.invoke(state["question"])
        return {**state, "web_context": result, "used_live_search": True}

    def store_web_cache_node(self, state: ResearchState) -> ResearchState:
        if config.ENABLE_WEB_CACHE and state["web_context"].strip():
            vectorstore: VectorStore = VectorStore()
            vectorstore.ingest_web_cache(state["question"], state["web_context"])

        return state

    def generate_node(self, state: ResearchState) -> ResearchState:
        llm = self._get_llm()
        messages = prompts.HYBRID_ANSWER_PROMPT.format_messages(
                question=state["question"],
                doc_context=state["doc_context"] or "(no relevant content found in society documents)",
                web_context=state.get("web_context") or "(not used — society documents were sufficient)",
            )
        response = llm.invoke(messages)
        return {**state, "answer": response.content} 

    def get_research_graph(self):
        if self._graph is None:
            g = StateGraph(ResearchState)
        
            g.add_node("retrieve", self.retrieve_node)
            g.add_node("grade", self.grade_node)
            g.add_node("check_web_cache", self.check_web_cache_node)
            g.add_node("grade_cache", self.grade_cache_node)
            g.add_node("live_web_search", self.live_web_search_node)
            g.add_node("store_web_cache", self.store_web_cache_node)
            g.add_node("generate", self.generate_node)

            g.set_entry_point("retrieve")
            g.add_edge("retrieve", "grade")
            g.add_conditional_edges(
                "grade", 
                self.route_after_grade, 
                {
                    "generate": "generate", 
                    "check_web_cache": "check_web_cache"
                }
            )

            g.add_edge("check_web_cache", "grade_cache")
            g.add_conditional_edges(
                "grade_cache",
                self.route_after_cache_grade,
                {
                    "generate": "generate", 
                    "live_web_search": "live_web_search"
                },
            )

            g.add_edge("live_web_search", "store_web_cache")
            g.add_edge("store_web_cache", "generate")
            g.add_edge("generate", END)

        self._graph = g.compile()
        return self._graph
        
    def run_research(self, question: str) -> dict:
        """
        Runs the hybrid research graph and returns a dict with:
            answer            - the final synthesized answer
            used_cache        - whether a cached prior web result was reused
            used_live_search  - whether a fresh live web search was needed
            doc_context       - raw retrieved society-document excerpts
            web_context       - raw web/cache content actually used, if any
        """
        graph = self.get_research_graph()
        initial_state: ResearchState = {
            "question": question,
            "doc_context": "",
            "cache_context": "",
            "web_context": "",
            "needs_web": False,
            "needs_live_search": False,
            "used_cache": False,
            "used_live_search": False,
            "answer": "",
        }
        result = graph.invoke(initial_state)
        return {
            "answer": result["answer"],
            "used_cache": result["used_cache"],
            "used_live_search": result["used_live_search"],
            "doc_context": result["doc_context"],
            "web_context": result["web_context"],
        }

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        if not history:
            return "(no earlier turns in this conversation)"

        lines = []
        for turn in history:
            role = "User" if turn.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {turn.get('context', '')}")

        return "\n".join(lines)

    def run_research_stream(self, question: str, history: list[dict] | None = None):
        history = history or []

        state: ResearchState = {
            "question": question,
            "doc_context": "",
            "cache_context": "",
            "web_context": "",
            "needs_web": False,
            "needs_live_search": False,
            "used_cache": False,
            "used_live_search": False,
            "answer": "",
        }

        state = self.retrieve_node(state)
        state = self.grade_node(state)

        if state["needs_web"]:
            state = self.check_web_cache_node(state)
            state = self.grade_cache_node(state)

            if state["needs_live_search"]:
                state = self.live_web_search_node(state)
                state = self.store_web_cache_node(state)

        llm = self._get_llm()
        messages = prompts.HYBRID_ANSWER_PROMPT.format_messages(
            question=state["question"],
            doc_context=state["doc_context"] or "(no relevant content found in society documents)",
            web_context=state.get("web_context") or "(not used — society documents were sufficient)",
            chat_history=self._format_history(history),
        )

        full_answer = ""
        for chunk in llm.stream(messages):
            piece = chunk.content or ""
            if piece:
                full_answer += piece
                yield {
                    "type": "token", 
                    "content":piece
                }

        yield {
                "type": "done",
                "answer": full_answer,
                "used_cache": state["used_cache"],
                "used_live_search": state["used_live_search"],
                "doc_context": state["doc_context"],
                "web_context": state["web_context"],
            }
