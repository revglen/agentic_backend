from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from backend import config, prompts
from backend.vectorstore import VectorStore
from backend.tools import ALL_TOOLS, web_search as web_search_tool


class Agent:
    def __init__(self):
        self._llm = None
        self._agent = None

    def get_llm(self):
        if self._llm is None:
            self._llm = ChatGroq(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                api_key=config.GROQ_API_KEY,
            )

        return self._llm

    def get_agent(self):
        if self._agent is None:
            self._agent = create_react_agent(self.get_llm(), ALL_TOOLS)

        return self._agent

    def _run(self, messages) -> str:
        agent = self.get_agent()
        result = agent.invoke({"messages": messages})
        final_message = result["messages"][-1]  # was result["message"] -- KeyError
        return final_message.content

    def _stream(self, messages):
        llm = self.get_llm()
        full_answer = ""
        for chunk in llm.stream(messages):
            piece = chunk.content or ""
            if piece:
                full_answer += piece
                yield {"type": "token", "content": piece}

        yield {"type": "done", "answer": full_answer}

    def _retrieve_by_labels(self, labels: list[str] | None, k: int = 20) -> str:
        if not labels:
            return ""

        vectorstore: VectorStore = VectorStore()
        blocks = []
        for label in labels:
            retriever = vectorstore.get_retriever(k=k, label_filter=label)
            docs = retriever.invoke(label) if retriever else []
            if docs:
                blocks.append(
                    f"### Uploaded document -- label: {label}\n{vectorstore.format_docs(docs)}"
                )

        return "\n\n".join(blocks)

    def _doc_qa_messages(self, question: str):
        vectorstore: VectorStore = VectorStore()
        retriever = vectorstore.get_retriever()
        docs = retriever.invoke(question) if retriever else []
        context = (
            vectorstore.format_docs(docs)
            if docs
            else "(no documents uploaded yet, or no matching content found)"
        )
        messages = prompts.DOC_QA_PROMPT.format_messages(context=context, question=question)
        return messages, context

    def run_doc_qa(self, question: str) -> str:
        messages, _ = self._doc_qa_messages(question)
        return self._run(messages)

    def run_doc_qa_stream(self, question: str):
        messages, context = self._doc_qa_messages(question)
        for event in self._stream(messages):
            if event["type"] == "done":
                event["doc_context"] = context
            yield event

    def _compare_messages(
        self,
        proposals_text: str,
        use_web_research: bool,
        proposal_labels: list[str] | None = None,
    ):
        uploaded_context = self._retrieve_by_labels(proposal_labels)
        typed_text = (proposals_text or "").strip()
        combined_proposals = "\n\n".join(part for part in [uploaded_context, typed_text] if part)
        if not combined_proposals:
            combined_proposals = typed_text  # keep old behavior if both are empty

        if use_web_research:
            web_context = (
                web_search_tool.invoke(combined_proposals) or "(no relevant web results found)"
            )
        else:
            web_context = "(not requested)"

        messages = prompts.PROPOSAL_COMPARE_PROMPT.format_messages(
            proposals=combined_proposals, web_context=web_context
        )
        return messages, web_context, combined_proposals

    def run_compare_proposals(
        self,
        proposals_text: str,
        use_web_research: bool = True,
        proposal_labels: list[str] | None = None,
    ) -> str:
        messages, _, _ = self._compare_messages(proposals_text, use_web_research, proposal_labels)
        # Both branches now return -- previously the use_web_research=True
        # branch fell off the end of the function and returned None.
        return self._run(messages)

    def run_compare_proposals_stream(
        self,
        proposals_text: str,
        use_web_research: bool = True,
        proposal_labels: list[str] | None = None,
    ):
        messages, web_context, combined_proposals = self._compare_messages(
            proposals_text, use_web_research, proposal_labels
        )
        for event in self._stream(messages):
            if event["type"] == "done":
                event["web_context"] = web_context
                event["used_web_research"] = use_web_research
                event["proposal_labels_used"] = proposal_labels or []
            yield event

    def _compliance_messages(self, project_details: str, doc_labels: list[str] | None = None):
        uploaded_context = self._retrieve_by_labels(doc_labels)
        combined_details = (project_details or "").strip()
        if uploaded_context:
            combined_details = f"{combined_details}\n\nSupporting documents on file:\n{uploaded_context}"

        # Keep the web search query focused on the short, user-typed facts
        # rather than the (potentially large) uploaded document text.
        search_results = (
            web_search_tool.invoke(f"RERA MHADA municipal approval status: {project_details}")
            or "(no relevant web results found)"
        )
        messages = prompts.COMPLIANCE_PROMPT.format_messages(
            project_details=combined_details, search_results=search_results
        )
        return messages, search_results, combined_details

    def run_compliance_check(self, project_details: str, doc_labels: list[str] | None = None) -> str:
        messages, _, _ = self._compliance_messages(project_details, doc_labels)
        return self._run(messages)

    def run_compliance_check_stream(self, project_details: str, doc_labels: list[str] | None = None):
        messages, search_results, combined_details = self._compliance_messages(
            project_details, doc_labels
        )
        for event in self._stream(messages):
            if event["type"] == "done":
                event["search_results"] = search_results
                event["doc_labels_used"] = doc_labels or []
            yield event