"""Deep RAG implementation with LangGraph for multi-step reasoning."""
import os
import subprocess
import sys
from typing import Dict, List, Optional

# Fix OpenMP conflict on macOS (must be before any imports that use OpenMP)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure langgraph is available
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("Installing langgraph...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langgraph"])
    from langgraph.graph import StateGraph, END
    print("✓ langgraph installed successfully")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .graph_nodes import (
    RAGState, Plan, Step, PastStep, RetrievalDecision, PolicyDecision,
    get_past_context_str, format_docs
)

# Try to import rich, fallback to print if not available
try:
    from rich.console import Console
    from rich import print as rprint
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None
    rprint = print
from .config import get_config
from .rag_chain import create_llm
from .retrieval import HybridRetriever

# Try to import Tavily for web search (initialize to None to avoid scoping issues)
TavilySearchResults = None
TavilyClient = None
HAS_TAVILY_LANGCHAIN = False
HAS_TAVILY_DIRECT = False

try:
    from langchain_tavily import TavilySearchResults
    HAS_TAVILY_LANGCHAIN = True
    HAS_TAVILY_DIRECT = False
except ImportError:
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        HAS_TAVILY_LANGCHAIN = True
        HAS_TAVILY_DIRECT = False
    except ImportError:
        HAS_TAVILY_LANGCHAIN = False
        # Fallback to direct tavily-python
        try:
            from tavily.client import TavilyClient
            HAS_TAVILY_DIRECT = True
        except ImportError:
            try:
                import tavily
                TavilyClient = tavily.TavilyClient
                HAS_TAVILY_DIRECT = True
            except ImportError:
                HAS_TAVILY_DIRECT = False


class DeepRAGSystem:
    """Deep RAG system with LangGraph orchestration."""
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        vector_store: Optional[FAISS] = None,
        documents: Optional[List[Document]] = None,
        embedding_function = None
    ):
        """
        Initialize Deep RAG system.
        
        Args:
            config: Configuration dictionary
            vector_store: FAISS vector store
            documents: List of documents with metadata
            embedding_function: Embedding function
        """
        self.config = config or get_config()
        self.vector_store = vector_store
        self.documents = documents
        self.embedding_function = embedding_function
        
        # Create LLMs
        self.reasoning_llm = create_llm(self.config)
        
        # Initialize web search tool
        self._setup_web_search()
        
        # Initialize agents
        self._setup_agents()
        
        # Initialize retrieval
        self._setup_retrieval()
        
        # Build graph
        self.graph = self._build_graph()
        self.compiled_graph = None
    
    def _setup_web_search(self):
        """Set up web search tool using Tavily."""
        self.web_search_tool = None
        tavily_api_key = self.config.get("tavily_api_key")
        
        if not tavily_api_key:
            if HAS_RICH:
                console.print("  ⚠ Tavily API key not found. Web search will be disabled.")
            else:
                print("  ⚠ Tavily API key not found. Web search will be disabled.")
            return
        
        try:
            if HAS_TAVILY_LANGCHAIN and TavilySearchResults is not None:
                # Use LangChain Tavily integration
                self.web_search_tool = TavilySearchResults(
                    k=self.config.get('web_search_results', 5),
                    api_key=tavily_api_key
                )
            elif HAS_TAVILY_DIRECT and TavilyClient is not None:
                # Use direct Tavily client
                self.tavily_client = TavilyClient(api_key=tavily_api_key)
                self.web_search_tool = "direct"  # Mark as direct client
            else:
                # Try to install langchain-tavily
                import subprocess
                import sys
                try:
                    print("Installing langchain-tavily...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-tavily"])
                    from langchain_tavily import TavilySearchResults as NewTavilySearchResults
                    self.web_search_tool = NewTavilySearchResults(
                        k=self.config.get('web_search_results', 5),
                        api_key=tavily_api_key
                    )
                    print("✓ langchain-tavily installed successfully")
                except Exception as install_error:
                    if HAS_RICH:
                        console.print(f"  ⚠ Could not install Tavily: {install_error}. Web search will be disabled.")
                    else:
                        print(f"  ⚠ Could not install Tavily: {install_error}. Web search will be disabled.")
        except Exception as e:
            if HAS_RICH:
                console.print(f"  ⚠ Error setting up Tavily: {e}. Web search will be disabled.")
            else:
                print(f"  ⚠ Error setting up Tavily: {e}. Web search will be disabled.")
            self.web_search_tool = None
    
    def _setup_agents(self):
        """Set up all LLM agents."""
        # Planner Agent
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research planner specializing in energy sector analysis. 
            Break down complex questions into a structured, multi-step research plan. 
            For each step, decide whether to search documents (search_documents) or the web (search_web).
            Use search_documents for information likely in the provided documents.
            Use search_web for current events, recent data, or external information.
            
            Available tools:
            - search_documents: Search the provided energy sector documents
            - search_web: Search the internet for current information
            
            Create a plan with 3-5 steps that will comprehensively answer the question."""),
            ("human", "Question: {question}")
        ])
        self.planner_agent = planner_prompt | self.reasoning_llm.with_structured_output(Plan)
        
        # Query Rewriter Agent
        query_rewriter_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a search query optimization expert specializing in energy sector and technical documents.
            Rewrite and expand the sub-question into an optimal search query that will retrieve the most relevant information.
            Consider the keywords and past context when crafting the query."""),
            ("human", """Sub-question: {sub_question}
            Keywords: {keywords}
            Past context: {past_context}
            
            Provide an optimized search query:""")
        ])
        self.query_rewriter_agent = query_rewriter_prompt | self.reasoning_llm | StrOutputParser()
        
        # Retrieval Supervisor Agent
        retrieval_supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a retrieval strategy expert. Decide which retrieval strategy to use:
            - vector_search: For semantic/synonym matching
            - keyword_search: For exact term matching
            - hybrid_search: For complex queries needing both
            
            Choose the best strategy for the given query."""),
            ("human", "Query: {sub_question}\n\nWhich retrieval strategy should be used?")
        ])
        self.retrieval_supervisor_agent = retrieval_supervisor_prompt | self.reasoning_llm.with_structured_output(RetrievalDecision)
        
        # Reflection Agent
        reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research analyst. Summarize the key findings from the retrieved context
            for the given sub-question. Be concise but comprehensive."""),
            ("human", "Sub-question: {sub_question}\n\nContext: {context}\n\nSummary:")
        ])
        self.reflection_agent = reflection_prompt | self.reasoning_llm | StrOutputParser()
        
        # Distiller Agent
        distiller_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a context distillation expert. Extract and synthesize only the most relevant
            information from the context that directly answers the question. Remove redundancy."""),
            ("human", "Question: {question}\n\nContext: {context}\n\nDistilled context:")
        ])
        self.distiller_agent = distiller_prompt | self.reasoning_llm | StrOutputParser()
        
        # Policy Agent
        policy_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a master strategist specializing in energy sector research and analysis.
            Decide whether to continue searching for more information or stop and generate the final answer.
            
            IMPORTANT: If we have completed all planned steps, you MUST stop.
            
            Continue if:
            - Critical information is still missing AND we haven't completed all steps
            - The question requires more depth AND we haven't completed all steps
            - Additional perspectives are needed AND we haven't completed all steps
            
            Stop if:
            - All planned steps have been completed (CRITICAL)
            - Sufficient information has been gathered
            - All sub-questions have been addressed
            - Further search is unlikely to add value
            - Current step >= max_steps"""),
            ("human", """Original question: {original_question}
            Research history: {research_history}
            Current step: {current_step} of {max_steps}
            Total plan steps: {total_steps}
            
            Should we continue or stop? Remember: If current_step >= total_steps, you MUST stop.""")
        ])
        self.policy_agent = policy_prompt | self.reasoning_llm.with_structured_output(PolicyDecision)
    
    def _setup_retrieval(self):
        """Set up retrieval components."""
        if self.vector_store and self.documents and self.embedding_function:
            self.hybrid_retriever = HybridRetriever(
                self.documents,
                self.vector_store,
                self.embedding_function
            )
        else:
            self.hybrid_retriever = None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph."""
        graph = StateGraph(RAGState)
        
        # Add nodes
        graph.add_node("plan", self.plan_node)
        graph.add_node("retrieve_documents", self.retrieval_node)
        graph.add_node("retrieve_web", self.web_search_node)
        graph.add_node("rerank", self.rerank_node)
        graph.add_node("compress", self.compression_node)
        graph.add_node("reflect", self.reflection_node)
        graph.add_node("generate_final_answer", self.final_answer_node)
        graph.add_node("choose_next_tool", lambda state: state)
        
        # Set entry point
        graph.set_entry_point("plan")
        
        # Add edges
        graph.add_edge("plan", "choose_next_tool")
        
        # Conditional routing
        graph.add_conditional_edges(
            "choose_next_tool",
            self.route_by_tool,
            {
                "search_documents": "retrieve_documents",
                "search_web": "retrieve_web",
                "finalize": "generate_final_answer"
            }
        )
        
        # Flow after retrieval
        graph.add_edge("retrieve_documents", "rerank")
        graph.add_edge("retrieve_web", "rerank")
        graph.add_edge("rerank", "compress")
        graph.add_edge("compress", "reflect")
        
        # After reflection, check if we should continue
        graph.add_conditional_edges(
            "reflect",
            self.should_continue_node,
            {
                "continue": "choose_next_tool",
                "stop": "generate_final_answer"
            }
        )
        
        graph.add_edge("generate_final_answer", END)
        
        return graph
    
    def plan_node(self, state: RAGState) -> Dict:
        """Generate a plan for answering the question."""
        if HAS_RICH:
            console.print("--- 🧠: Generating Plan ---")
        else:
            print("--- 🧠: Generating Plan ---")
        plan = self.planner_agent.invoke({"question": state["original_question"]})
        if HAS_RICH:
            rprint(plan)
        else:
            print(plan)
        return {"plan": plan, "current_step_index": 0, "past_steps": []}
    
    def retrieval_node(self, state: RAGState) -> Dict:
        """Retrieve documents from vector store."""
        current_step_index = state["current_step_index"]
        if not state["plan"] or current_step_index >= len(state["plan"].steps):
            return {"retrieved_docs": []}
        
        current_step = state["plan"].steps[current_step_index]
        msg = f"--- 🔍: Retrieving Documents (Step {current_step_index + 1}: {current_step.sub_question}) ---"
        if HAS_RICH:
            console.print(msg)
        else:
            print(msg)
        
        past_context = get_past_context_str(state['past_steps'])
        rewritten_query = self.query_rewriter_agent.invoke({
            "sub_question": current_step.sub_question,
            "keywords": ", ".join(current_step.keywords),
            "past_context": past_context
        })
        if HAS_RICH:
            console.print(f"  Rewritten Query: {rewritten_query}")
        else:
            print(f"  Rewritten Query: {rewritten_query}")
        
        # Get retrieval decision
        retrieval_decision = self.retrieval_supervisor_agent.invoke({"sub_question": rewritten_query})
        if HAS_RICH:
            console.print(f"  Strategy: {retrieval_decision.strategy}")
        else:
            print(f"  Strategy: {retrieval_decision.strategy}")
        
        # Retrieve using chosen strategy
        if self.hybrid_retriever:
            if retrieval_decision.strategy == 'vector_search':
                retrieved_docs = self.hybrid_retriever.vector_search(
                    rewritten_query,
                    section_filter=current_step.document_section,
                    k=self.config.get('top_k_retrieval', 10)
                )
            elif retrieval_decision.strategy == 'keyword_search':
                retrieved_docs = self.hybrid_retriever.bm25_search(
                    rewritten_query,
                    k=self.config.get('top_k_retrieval', 10)
                )
            else:  # hybrid_search
                retrieved_docs = self.hybrid_retriever.hybrid_search(
                    rewritten_query,
                    section_filter=current_step.document_section,
                    k=self.config.get('top_k_retrieval', 10)
                )
        else:
            # Fallback to simple vector search
            retrieved_docs = self.vector_store.similarity_search(
                rewritten_query,
                k=self.config.get('top_k_retrieval', 10)
            )
        
        return {"retrieved_docs": retrieved_docs}
    
    def web_search_node(self, state: RAGState) -> Dict:
        """Search the web using Tavily API."""
        current_step_index = state["current_step_index"]
        if not state["plan"] or current_step_index >= len(state["plan"].steps):
            return {"retrieved_docs": []}
        
        current_step = state["plan"].steps[current_step_index]
        msg = f"--- 🌐: Searching Web (Step {current_step_index + 1}: {current_step.sub_question}) ---"
        if HAS_RICH:
            console.print(msg)
        else:
            print(msg)
        
        # Check if web search is available
        if not self.web_search_tool:
            if HAS_RICH:
                console.print("  ⚠ Web search not available (Tavily API key not configured or tool not initialized)")
            else:
                print("  ⚠ Web search not available (Tavily API key not configured or tool not initialized)")
            return {"retrieved_docs": []}
        
        # Rewrite query for web search
        past_context = get_past_context_str(state['past_steps'])
        rewritten_query = self.query_rewriter_agent.invoke({
            "sub_question": current_step.sub_question,
            "keywords": ", ".join(current_step.keywords),
            "past_context": past_context
        })
        
        if HAS_RICH:
            console.print(f"  Web Search Query: {rewritten_query}")
        else:
            print(f"  Web Search Query: {rewritten_query}")
        
        try:
            # Perform web search
            if isinstance(self.web_search_tool, str) and self.web_search_tool == "direct":
                # Use direct Tavily client
                response = self.tavily_client.search(
                    query=rewritten_query,
                    max_results=self.config.get('web_search_results', 5)
                )
                # Convert to Document format
                retrieved_docs = [
                    Document(
                        page_content=result.get("content", ""),
                        metadata={
                            "source": result.get("url", "unknown"),
                            "title": result.get("title", ""),
                            "score": result.get("score", 0.0)
                        }
                    )
                    for result in response.get("results", [])
                ]
            else:
                # Use LangChain Tavily tool
                results = self.web_search_tool.invoke({"query": rewritten_query})
                # Convert to Document format
                retrieved_docs = [
                    Document(
                        page_content=res.get("content", res.get("snippet", "")),
                        metadata={
                            "source": res.get("url", "unknown"),
                            "title": res.get("title", ""),
                            "score": res.get("score", 0.0)
                        }
                    )
                    for res in results
                ]
            
            if HAS_RICH:
                console.print(f"  ✓ Found {len(retrieved_docs)} web results")
            else:
                print(f"  ✓ Found {len(retrieved_docs)} web results")
            
            return {"retrieved_docs": retrieved_docs}
            
        except Exception as e:
            if HAS_RICH:
                console.print(f"  ⚠ Error during web search: {e}")
            else:
                print(f"  ⚠ Error during web search: {e}")
            return {"retrieved_docs": []}
    
    def rerank_node(self, state: RAGState) -> Dict:
        """Rerank retrieved documents."""
        if HAS_RICH:
            console.print("--- 🎯: Reranking Documents ---")
        else:
            print("--- 🎯: Reranking Documents ---")
        # TODO: Implement cross-encoder reranking
        # For now, just return the documents as-is
        reranked_docs = state["retrieved_docs"][:self.config.get('top_n_rerank', 3)]
        if HAS_RICH:
            console.print(f"  Selected top {len(reranked_docs)} documents.")
        else:
            print(f"  Selected top {len(reranked_docs)} documents.")
        return {"reranked_docs": reranked_docs}
    
    def compression_node(self, state: RAGState) -> Dict:
        """Compress and distill context."""
        if HAS_RICH:
            console.print("--- ✂️: Distilling Context ---")
        else:
            print("--- ✂️: Distilling Context ---")
        current_step_index = state["current_step_index"]
        if not state["plan"] or current_step_index >= len(state["plan"].steps):
            return {"synthesized_context": ""}
        
        current_step = state["plan"].steps[current_step_index]
        context = format_docs(state["reranked_docs"])
        synthesized_context = self.distiller_agent.invoke({
            "question": current_step.sub_question,
            "context": context
        })
        if HAS_RICH:
            console.print(f"  Distilled context: {synthesized_context[:200]}...")
        else:
            print(f"  Distilled context: {synthesized_context[:200]}...")
        return {"synthesized_context": synthesized_context}
    
    def reflection_node(self, state: RAGState) -> Dict:
        """Reflect on findings and update research history."""
        if HAS_RICH:
            console.print("--- 🤔: Reflecting on Findings ---")
        else:
            print("--- 🤔: Reflecting on Findings ---")
        current_step_index = state["current_step_index"]
        plan = state.get("plan")
        
        if not plan or current_step_index >= len(plan.steps):
            return {"past_steps": state["past_steps"], "current_step_index": current_step_index}
        
        current_step = plan.steps[current_step_index]
        summary = self.reflection_agent.invoke({
            "sub_question": current_step.sub_question,
            "context": state['synthesized_context']
        })
        if HAS_RICH:
            console.print(f"  Summary: {summary[:200]}...")
        else:
            print(f"  Summary: {summary[:200]}...")
        
        new_past_step: PastStep = {
            "step_index": current_step_index + 1,
            "sub_question": current_step.sub_question,
            "retrieved_docs": state['reranked_docs'],
            "summary": summary
        }
        
        research_history = get_past_context_str(state["past_steps"] + [new_past_step])
        
        # Increment step index AFTER reflection
        new_step_index = current_step_index + 1
        
        return {
            "past_steps": state["past_steps"] + [new_past_step],
            "current_step_index": new_step_index,
            "research_history": research_history
        }
    
    def final_answer_node(self, state: RAGState) -> Dict:
        """Generate final answer."""
        if HAS_RICH:
            console.print("--- ✅: Generating Final Answer ---")
        else:
            print("--- ✅: Generating Final Answer ---")
        
        final_context = ""
        for i, step in enumerate(state['past_steps']):
            final_context += f"\n--- Findings from Research Step {i+1} ---\n"
            for doc in step['retrieved_docs']:
                source = doc.metadata.get('section') or doc.metadata.get('source', 'Unknown')
                final_context += f"Source: {source}\nContent: {doc.page_content}\n\n"
        
        final_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert energy sector analyst. Synthesize the research findings
            into a comprehensive, multi-paragraph answer for the user's original question.
            Your answer must be grounded in the provided context. Include citations where appropriate."""),
            ("human", "Original Question: {question}\n\nResearch History and Context:\n{context}")
        ])
        
        final_answer_chain = final_answer_prompt | self.reasoning_llm | StrOutputParser()
        final_answer = final_answer_chain.invoke({
            "question": state['original_question'],
            "context": final_context
        })
        
        return {"final_answer": final_answer}
    
    def route_by_tool(self, state: RAGState) -> str:
        """Route to the appropriate tool based on current step."""
        current_step_index = state.get("current_step_index", 0)
        
        if not state.get("plan") or current_step_index >= len(state["plan"].steps):
            return "finalize"
        
        current_step = state["plan"].steps[current_step_index]
        tool = current_step.tool
        
        if tool == "search_10k" or tool == "search_documents":
            return "search_documents"
        elif tool == "search_web":
            return "search_web"
        else:
            return "finalize"
    
    def should_continue_node(self, state: RAGState) -> str:
        """Decide whether to continue or stop."""
        current_step = state.get("current_step_index", 0)
        max_steps = state.get("max_steps", self.config.get("max_reasoning_iterations", 7))
        plan = state.get("plan")
        
        # First check: Maximum steps reached
        if current_step >= max_steps:
            if HAS_RICH:
                console.print(f"--- ⏹️: Maximum steps reached ({current_step}/{max_steps}) ---")
            else:
                print(f"--- ⏹️: Maximum steps reached ({current_step}/{max_steps}) ---")
            return "stop"
        
        # Second check: No plan available
        if not plan:
            if HAS_RICH:
                console.print("--- ⏹️: No plan available, stopping ---")
            else:
                print("--- ⏹️: No plan available, stopping ---")
            return "stop"
        
        # Third check: All plan steps completed (CRITICAL - must stop here)
        # Note: current_step_index is 0-indexed, so if it equals len(steps), we've done all steps
        total_plan_steps = len(plan.steps)
        if current_step >= total_plan_steps:
            if HAS_RICH:
                console.print(f"--- ⏹️: All plan steps completed ({current_step}/{total_plan_steps}) - STOPPING ---")
            else:
                print(f"--- ⏹️: All plan steps completed ({current_step}/{total_plan_steps}) - STOPPING ---")
            return "stop"
        
        # Ask policy agent only if we haven't completed all steps
        # But make it clear in the prompt that we're close to the end
        try:
            decision = self.policy_agent.invoke({
                "original_question": state["original_question"],
                "research_history": state.get("research_history", ""),
                "current_step": current_step,
                "max_steps": min(max_steps, total_plan_steps),
                "total_steps": total_plan_steps
            })
            
            if decision.decision == "stop":
                msg = f"--- ⏹️: Policy decision to stop: {decision.reasoning} ---"
                if HAS_RICH:
                    console.print(msg)
                else:
                    print(msg)
                return "stop"
            else:
                # CRITICAL: Double-check we haven't exceeded plan steps
                # This is a safety net in case policy agent makes a mistake
                if current_step >= total_plan_steps:
                    if HAS_RICH:
                        console.print(f"--- ⏹️: SAFETY STOP - All steps done ({current_step}/{total_plan_steps}), overriding policy ---")
                    else:
                        print(f"--- ⏹️: SAFETY STOP - All steps done ({current_step}/{total_plan_steps}), overriding policy ---")
                    return "stop"
                
                msg = f"--- ➡️: Policy decision to continue: {decision.reasoning} ---"
                if HAS_RICH:
                    console.print(msg)
                else:
                    print(msg)
                return "continue"
        except Exception as e:
            # If policy agent fails, stop if we've completed all steps
            if current_step >= total_plan_steps:
                if HAS_RICH:
                    console.print(f"--- ⏹️: Policy error, but all steps done ({current_step}/{total_plan_steps}): {e} ---")
                else:
                    print(f"--- ⏹️: Policy error, but all steps done ({current_step}/{total_plan_steps}): {e} ---")
                return "stop"
            # If policy fails but we haven't done all steps, continue (safer than stopping)
            if HAS_RICH:
                console.print(f"--- ⚠️: Policy error, continuing: {e} ---")
            else:
                print(f"--- ⚠️: Policy error, continuing: {e} ---")
            return "continue"
    
    def compile(self):
        """Compile the graph."""
        if self.compiled_graph is None:
            self.compiled_graph = self.graph.compile()
        return self.compiled_graph
    
    def answer(self, question: str) -> str:
        """
        Answer a question using the deep RAG system.
        
        Args:
            question: The question to answer
            
        Returns:
            The answer string
        """
        if self.compiled_graph is None:
            self.compile()
        
        graph_input: RAGState = {
            "original_question": question,
            "question": question,
            "plan": None,
            "past_steps": [],
            "current_step_index": 0,
            "retrieved_docs": [],
            "web_results": [],
            "reranked_docs": [],
            "compressed_context": "",
            "synthesized_context": "",
            "research_history": "",
            "final_answer": "",
            "current_step": 0,
            "max_steps": self.config.get("max_reasoning_iterations", 7)
        }
        
        final_state = None
        # Increase recursion limit to handle complex multi-step plans
        # Each step can involve multiple graph nodes (retrieve -> rerank -> compress -> reflect)
        # Estimate: 4 nodes per step (retrieve, rerank, compress, reflect) + plan + final answer
        # We'll start with a high limit and adjust based on plan
        recursion_limit = 200  # High default limit
        
        try:
            for chunk in self.compiled_graph.stream(
                graph_input,
                stream_config={"recursion_limit": recursion_limit},
                stream_mode="values"
            ):
                final_state = chunk
                # Safety check: if we have a final answer, we can break early
                if final_state and final_state.get("final_answer"):
                    break
        except Exception as e:
            if "recursion limit" in str(e).lower():
                # If we hit recursion limit, try to get final answer from last state
                if final_state:
                    if final_state.get("final_answer"):
                        print("⚠ Recursion limit reached, but final answer available")
                        return final_state.get("final_answer")
                    # Try to generate final answer from available context
                    if final_state.get("past_steps"):
                        print("⚠ Recursion limit reached, generating final answer from available context")
                        try:
                            return self._generate_final_from_context(final_state)
                        except:
                            pass
                raise RuntimeError(
                    f"Recursion limit reached ({recursion_limit}). The graph may be in an infinite loop. "
                    f"Last state step: {final_state.get('current_step_index') if final_state else 'unknown'}"
                ) from e
            raise
        
        if final_state:
            answer = final_state.get("final_answer", "")
            if answer:
                return answer
            else:
                # If no final answer but we have context, try to generate one
                if final_state.get("past_steps"):
                    return self._generate_final_from_context(final_state)
                return "Error: No final answer generated."
        else:
            return "Error: No final state generated."
    
    def _generate_final_from_context(self, state: RAGState) -> str:
        """Generate final answer from available context when recursion limit is hit."""
        final_context = ""
        for i, step in enumerate(state.get('past_steps', [])):
            final_context += f"\n--- Findings from Research Step {i+1} ---\n"
            for doc in step.get('retrieved_docs', []):
                source = doc.metadata.get('section') or doc.metadata.get('source', 'Unknown')
                final_context += f"Source: {source}\nContent: {doc.page_content}\n\n"
        
        if not final_context:
            return "Error: No context available to generate answer."
        
        final_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert energy sector analyst. Synthesize the research findings
            into a comprehensive answer for the user's original question.
            Your answer must be grounded in the provided context."""),
            ("human", "Original Question: {question}\n\nResearch Context:\n{context}")
        ])
        
        final_answer_chain = final_answer_prompt | self.reasoning_llm | StrOutputParser()
        return final_answer_chain.invoke({
            "question": state.get('original_question', ''),
            "context": final_context
        })

