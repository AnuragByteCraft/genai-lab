import os
import ast
import sqlite3
import subprocess
import networkx as nx
from typing import List, TypedDict, Annotated, Literal
from pathlib import Path

# --- LangChain / LangGraph Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

# --- LSP Imports ---
from multilspy import SyncLanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger

# suppress the warning on parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda"

# Configuration
WORKSPACE_DIR = os.getcwd()
print("Workspace Dir = ", WORKSPACE_DIR)
VECTOR_DB_DIR = os.path.join(WORKSPACE_DIR, ".vibe_cache/chroma")
# MODEL_NAME = "google/gemma-3-12b"
MODEL_NAME = "openai/gpt-oss-20b"
BASE_URL = "http://192.168.68.122:1234/v1"
API_KEY = "lm-studio"


# ==============================================================================
# LAYER 1: THE CONTEXT ENGINE
# ==============================================================================

class CodeContextEngine:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.graph = nx.DiGraph()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-large",
            model_kwargs={"device": device}  # Use "cuda" or remove for CPU
        )
        self.vector_store = Chroma(
            collection_name="codebase_index",
            embedding_function=self.embeddings,
            persist_directory=VECTOR_DB_DIR
        )

    def scan_codebase(self):
        print("🔍 Scanning codebase for Context Building...")
        docs = []
        for file_path in self.root_dir.rglob("*.py"):
            if any(excl in str(file_path) for excl in ("venv", ".git", "__pycache__")):
                continue
            relative_path = file_path.relative_to(self.root_dir)
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            self._analyze_imports(str(relative_path), content)
            docs.append(Document(page_content=content, metadata={"source": str(relative_path)}))

        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(docs)
            self.vector_store.add_documents(splits)
            print(f"✅ Indexed {len(splits)} code chunks.")
        print(f"✅ Built Dependency Graph with {self.graph.number_of_nodes()} files.")

    def _analyze_imports(self, file_name: str, content: str):
        self.graph.add_node(file_name)
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.graph.add_edge(file_name, alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.graph.add_edge(file_name, node.module)
        except SyntaxError:
            pass

    def get_context(self, query: str, current_file: str = None) -> str:
        rag_results = self.vector_store.similarity_search(query, k=3)
        context_str = "--- Semantic Search Results ---\n"
        for doc in rag_results:
            context_str += f"File: {doc.metadata['source']}\nContent:\n{doc.page_content[:500]}...\n\n"

        if current_file and current_file in self.graph:
            neighbors = set(self.graph.successors(current_file)) | set(self.graph.predecessors(current_file))
            context_str += f"--- Related Files (Dependency Graph) ---\n"
            context_str += f"Current File: {current_file} is connected to: {', '.join(neighbors)}\n"
        return context_str


context_engine = CodeContextEngine(WORKSPACE_DIR)


# ==============================================================================
# LAYER 2: THE PERCEPTION LAYER (LSP Client — SAFE & SYNC)
# ==============================================================================

class LSPHandler:
    def __init__(self):
        self.config = MultilspyConfig.from_dict({"code_language": "python"})
        self.logger = MultilspyLogger()
        # We create the server instance on demand per request
        # (multilspy handles process reuse internally)
        pass

    def _run_with_lsp(self, func, *args, **kwargs):
        """
        Helper: Runs an LSP operation safely in a context-managed block.
        """
        server = SyncLanguageServer.create(self.config, self.logger, WORKSPACE_DIR)
        try:
            with server.start_server():
                return func(server, *args, **kwargs)
        except Exception as e:
            return f"LSP Error: {e}"

    def get_definition(self, file_path: str, line: int, column: int):
        def _impl(srv, fp, l, c):
            result = srv.request_definition(fp, l, c)
            if result:
                loc = result[0]
                uri = loc["uri"]
                if uri.startswith("file://"):
                    def_path = uri[7:]
                else:
                    def_path = uri
                start_line = loc["range"]["start"]["line"] + 1
                start_char = loc["range"]["start"]["character"] + 1
                return f"Defined at {def_path}:{start_line}:{start_char}"
            return "No definition found."
        return self._run_with_lsp(_impl, file_path, line, column)

    def get_hover(self, file_path: str, line: int, column: int):
        def _impl(srv, fp, l, c):
            result = srv.request_hover(fp, l, c)
            if result and "contents" in result:
                contents = result["contents"]
                if isinstance(contents, list):
                    rendered = []
                    for item in contents:
                        if isinstance(item, dict):
                            rendered.append(item.get("value", str(item)))
                        else:
                            rendered.append(str(item))
                    return "\n".join(rendered)
                elif isinstance(contents, dict):
                    return contents.get("value", str(contents))
                else:
                    return str(contents)
            return "No hover info."
        return self._run_with_lsp(_impl, file_path, line, column)

    def get_references(self, file_path: str, line: int, column: int):
        def _impl(srv, fp, l, c):
            refs = srv.request_references(fp, l, c)
            if refs:
                locations = []
                for ref in refs:
                    uri = ref["uri"]
                    if uri.startswith("file://"):
                        path = uri[7:]
                    else:
                        path = uri
                    line_num = ref["range"]["start"]["line"] + 1
                    locations.append(f"{path}:{line_num}")
                return locations
            return []
        return self._run_with_lsp(_impl, file_path, line, column)

    def get_document_symbols(self, file_path: str):
        def _impl(srv, fp):
            symbols = srv.request_document_symbols(fp)
            if not symbols:
                return []
            names = []
            for sym in symbols:
                # Safely extract name
                if isinstance(sym, dict):
                    name = sym.get("name")
                    if name:
                        names.append(name)
                # Some LSPs return flat lists like ["func_name", ...]
                elif isinstance(sym, str):
                    names.append(sym)
            return names

        return self._run_with_lsp(_impl, file_path)


# Initialize LSP handler (will start on first use)
lsp_handler = LSPHandler()

# ==============================================================================
# LAYER 3: ACTION LAYER (Tools)
# ==============================================================================

@tool
def read_file(file_path: str) -> str:
    """
    Read a file given its file path and return a string.
    :param file_path: path to file
    :return: content of file
    """
    path = Path(WORKSPACE_DIR) / file_path
    if not path.exists():
        return "Error: File not found."
    return path.read_text(encoding="utf-8")


@tool
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file given its file path and content to be written and return a string.
    :param file_path: path to file to be written
    :param content: content to be written to the specified file
    :return: success message or error message
    """
    print("📝 Writing to file:", file_path)
    try:
        path = Path(WORKSPACE_DIR) / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"SUCCESS: Wrote {len(content)} chars to {file_path}"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def list_files(directory: str = ".") -> str:
    """
    List all files in a directory.
    :param directory: directory to list
    :return: string that contains all files in the directory
    """
    path = Path(WORKSPACE_DIR) / directory
    if not path.exists():
        return "Directory not found."
    files = []
    for f in path.rglob("*"):
        if f.is_file() and not any(excl in str(f) for excl in ("venv", ".git", "__pycache__")):
            files.append(str(f.relative_to(WORKSPACE_DIR)))
    return "\n".join(sorted(files))


@tool
def run_linter(file_path: str) -> str:
    """
    Run Linter against a file.
    :param file_path: path to file
    :return: linter result or error message
    """
    try:
        result = subprocess.run(
            ["ruff", "check", file_path],
            capture_output=True,
            text=True,
            cwd=WORKSPACE_DIR
        )
        if result.returncode == 0:
            return "✅ Linting passed."
        return f"⚠️ Linting failed:\n{result.stdout}"
    except FileNotFoundError:
        return "❌ Ruff not installed. Run: pip install ruff"


@tool
def search_codebase(query: str, current_file: str = None) -> str:
    """
    Search codebase for a query.
    :param query: query to search in a context
    :param current_file: file to search in
    :return: search result or error message
    """
    return context_engine.get_context(query, current_file)


@tool
def lsp_hover(file_path: str, line: int, column: int) -> str:
    """
    Get hover documentation for symbol at location.
    :param file_path: path to file
    :param line: line number
    :param column: column number
    :return: hover result or error message
    """
    return lsp_handler.get_hover(file_path, line, column)


@tool
def lsp_definition(file_path: str, line: int, column: int) -> str:
    """
    Get definition location for symbol at location.
    :param file_path: path to file
    :param line: line number
    :param column: column number
    :return: definition result or error message
    """
    return lsp_handler.get_definition(file_path, line, column)


tools = [
    read_file, write_file, list_files, run_linter, search_codebase,
    lsp_hover, lsp_definition
]


# ==============================================================================
# LAYER 4: THE BRAIN (LangGraph Agent)
# ==============================================================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_file: str
    error_context: str
    plan: str
    iterations: int


def planner_node(state: AgentState):
    print("🧠 Planner: Creating plan...")
    llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL_NAME, temperature=0.0)
    system_prompt = (
        "You are an experienced planner whose responsibility is to create step by step plans for others to execute. "
        "Analyze the user request and create a concise, step-by-step implementation plan. "
        "DO NOT write code. DO NOT claim files are created. DO NOT use tools. "
        "You will be providing your plan to downstream agents with clear instructional steps.\n"
    )

    system_prompt = """
    You are an experienced planning agent. 
Your ONLY job is to produce a concise, step-by-step plan for another agent to execute.

STRICT RULES:
- Produce ONLY a plan.
- DO NOT perform the task.
- DO NOT describe outcomes of the task.
- DO NOT say files were created, modified, or saved.
- DO NOT claim that you performed any work.
- DO NOT use tools.
- DO NOT provide code.
- DO NOT provide solutions, only instructions.
- Refer to the target files ONLY as items the executor will operate on.

Your output must be a numbered list of steps that an executor can follow to complete the user request.

If the user asks you to “add”, “update”, “save”, etc., you MUST convert this into instructions for the executor, not actions you perform yourself.

    """
    """
            "1. Create api.py with FastAPI app and /hello endpoint.\n"
        "2. Create test_api.py with httpx test for /hello.\n"
        "3. Lint both files."

    """

    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])

    print("Response from Planner")
    print(response)

    return {"plan": response.content, "messages": [response], "iterations": 0}


def executor_node(state: AgentState):
    print("🤖 Executor: Acting on plan...")
    llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL_NAME, temperature=0.0).bind_tools(tools)
    system_prompt = (
        f"Plan: {state.get('plan', 'No plan')}\n"
        "You are a Senior Python Developer. Use tools to implement the plan. "
        "Always read before writing. After writing, run the linter. "
        "Use the tools read_file to read file or write_file to write file."
        "Use lsp_hover or lsp_definition if you need to understand symbols."
    )
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])

    print("Response from Executor")
    print(response)

    return {"messages": [response], "iterations": state["iterations"] + 1}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    print(f"Last message: {last_message}")
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("returning tool calls")
        return "tools"
    print("returning END")
    return END


conn = sqlite3.connect("agentic_ide.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)


def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges("executor", should_continue)
    workflow.add_edge("tools", "executor")
    app = workflow.compile(checkpointer=checkpointer)

    return app


def backend_main():
    # Clean start: ensure LSP and context are ready
    print("\n🚀 Initializing IDE Backend...\n")
    context_engine = CodeContextEngine(WORKSPACE_DIR)
    context_engine.scan_codebase()

    # Start LSP once (kept alive)
    # lsp_handler.start()

    try:
        # === DEMO 1: LSP Features ===
        demo_file = "demo_app.py"
        demo_path = Path(WORKSPACE_DIR) / demo_file

        print(f"\n🧪 DEMO 1: Testing LSP on {demo_file} (creating if missing)...\n")
        # Ensure file exists
        if not demo_path.exists():
            write_file.invoke({"file_path": demo_file,
                               "content": "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n\nprint(greet('World'))"})

        # Optional: Give LSP a moment to index (usually not needed, but safe)
        import time

        time.sleep(0.2)

        print("🔍 Hover on 'greet' at line 4, col 7:")
        print("   →", lsp_handler.get_hover(demo_file, 3, 6))  # 0-based line

        print("🔍 get references:")
        print("   →", lsp_handler.get_references(demo_file, 3, 6))  # 0-based line

        print("\n📍 Definition of 'greet':")
        print("   →", lsp_handler.get_definition(demo_file, 3, 6))

        print("\n🔠 Document symbols:")
        print("   →", lsp_handler.get_document_symbols(demo_file))

        # === DEMO 2: Agent-Driven Code Generation ===
        print("\n\n🧪 DEMO 2: Agent creating FastAPI endpoint + test...\n")
        user_request = "Create a FastAPI app in 'api.py' with a /hello endpoint returning {'message': 'Hello World'}. Also create a test in 'test_api.py' using httpx."
        user_request = (
            "Create a file 'api.py' that defines a FastAPI app with a GET /hello endpoint returning {'message': 'Hello World'}. "
            "Also create 'test_api.py' that uses httpx to test this endpoint. "
            "After writing each file, run the linter on it."
        )
        initial_state = {
            "messages": [HumanMessage(content=user_request)],
            "current_file": None,
            "error_context": None,
            "plan": None,
            "iterations": 0
        }
        config = {"configurable": {"thread_id": "demo_fastapi"}}

        app = create_graph()

        for event in app.stream(initial_state, config=config):
            for key, value in event.items():
                print(f"\n🔹 Node '{key}' executed.")
                if "messages" in value:
                    msg = value["messages"][-1]
                    if isinstance(msg, AIMessage):
                        if msg.content:
                            print(f"   💬 {msg.content[:200]}...")
                        if msg.tool_calls:
                            call = msg.tool_calls[0]
                            print(f"   🛠️  Tool: {call['name']}({call['args']})")

        # === Final Check ===
        print("\n✅ Demo complete. Check 'api.py' and 'test_api.py'.")

    finally:
        # Always clean up LSP
        # lsp_handler.stop()
        pass


# ==============================================================================
# MAIN: Demo with Full IDE Feature Exercise
# ==============================================================================

if __name__ == "__main__":
    backend_main()
