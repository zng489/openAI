## DataOps & Delta Share Helper – MCP Server (`poc_mcp_tiny.py`)

### Overview

This module implements a local **RAG (Retrieval-Augmented Generation) MCP server** specialized in:

- **Data Ops processes (L1, L2, L3)**  
- **Delta Share** (configuration, troubleshooting, permissions)  
- **Data governance and ITOPS runbooks**

The server exposes a single MCP tool, `perguntar_runbook`, which:

- Receives a **question in Portuguese** about the supported domains.
- Retrieves the most relevant document chunks from a local FAISS index.
- Builds a constrained prompt in Portuguese.
- Uses a local GPT4All model (TinyLlama) to generate an answer **only based on the retrieved context**.

This server is designed to run locally and to be consumed by MCP‑compatible clients (e.g. Claude Desktop).

---

### Files and Models Required

All of the following must be present in the **same directory** as `poc_mcp_tiny.py`:

- `embeddings.npy`  
  - `numpy.ndarray` of shape `(N, D)` with precomputed embeddings for each text chunk.  
  - Stored as `float32`.
- `chunks.pkl`  
  - Pickled Python `List[str]` containing the text chunks (same order as `embeddings.npy`).
- `bge-small-en-v1.5/`  
  - Directory containing the `SentenceTransformer` embedding model.
- `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`  
  - Quantized TinyLlama chat model used by GPT4All.

If any of these files or directories are missing, the script logs an error (paths included) and exits with a non‑zero status.

---

### Runtime Architecture

**Key components:**

- **Embeddings & Chunks**
  - `embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)`
  - `chunks: List[str] = pickle.load(open(CHUNKS_PATH, "rb"))`
  - Both are kept in memory for fast retrieval.

- **Vector Index (FAISS)**
  - `faiss.IndexFlatL2` built over the embeddings.
  - Used to retrieve the `K_RETRIEVE` most similar chunks to a query.

- **Embedding Model**
  - `SentenceTransformer(MODEL_EMBEDDING)` (BGE small).
  - Encodes incoming questions into vectors compatible with the FAISS index.

- **LLM (GPT4All)**
  - `GPT4All(model_name=MODEL_LLM_PATH, device="gpu")`
  - Can be switched to `"cpu"` if needed.
  - Used inside a `chat_session()` context for each question.

- **MCP Server**
  - `FastMCP(name="DataOps & Delta Share Helper")`
  - Exposes the `perguntar_runbook` tool.

Logging is configured to **stderr only**, and warnings are suppressed to keep the MCP output channel clean.

---

### Core Functions

- `buscar_contexto(pergunta: str, k: int = K_RETRIEVE) -> List[str>`
  - Encodes the user question with the embedding model.
  - Queries the FAISS index for the `k` nearest neighbors.
  - Returns the corresponding text chunks from `chunks`.
  - Logs how many relevant snippets were found; returns `[]` on error.

- `montar_prompt(pergunta: str, contextos: List[str]) -> str`
  - Builds a **Portuguese system + user prompt** for the TinyLlama chat model.
  - Includes:
    - System instructions: the assistant is specialized in runbooks, Data Ops, Delta Share, governance, ITOPS.
    - A rule to answer **only based on the provided context**, or state that the information was not found.
    - A numbered list of retrieved chunks (`─── Trecho 1 ───`, etc.).
    - The user question.

---

### MCP Tool: `perguntar_runbook`

```text
@mcp.tool()
def perguntar_runbook(pergunta: str) -> str:
    ...
```

- **Purpose**
  - Main entry point for MCP clients.
  - Answers questions related to:
    - Data Ops processes (L1/L2/L3)
    - Delta Share (config, troubleshooting, permissions)
    - Data governance and ITOPS runbooks
    - Any topic explicitly present in the indexed documents

- **Input**
  - `pergunta: str` — question in natural language (Portuguese).

- **Behavior**
  - Logs the incoming question (first 100 characters).
  - Calls `buscar_contexto(pergunta)` to get relevant chunks.
  - If no chunks are found, returns:
    - `"Nenhum trecho relevante encontrado nos documentos carregados."`
  - Otherwise:
    - Builds a prompt with `montar_prompt`.
    - Opens `with llm.chat_session():` and calls `llm.generate(...)` with:
      - `max_tokens = MAX_TOKENS` (default 1000)
      - `temp = TEMPERATURE` (default 0.18)
      - `repeat_penalty = REPEAT_PENALTY` (default 1.12)
    - Performs basic cleanup on the raw model output (removing tags such as `<|assistant|>`, `</s>`, `[/INST]`).
    - Returns the cleaned text.

- **Error Handling**
  - On exception during generation, logs stack trace and returns:
    - `"Erro ao gerar resposta: {mensagem}"`.

---

### Running the Server

From the project root (or directly inside the `mcp` folder), run:

```bash
python mcp/poc_mcp_tiny.py
```

On startup the script will:

- Initialize embeddings, chunks, FAISS index, embedding model and LLM.
- Log readiness messages such as:
  - `"Servidor RAG pronto para receber perguntas!"`
  - `"Iniciando servidor MCP 'DataOps & Delta Share Helper'..."`.
- Block on `mcp.run()`, waiting for connections from an MCP‑compatible client.

---

### Intended Usage (Client Side)

From an MCP client (e.g. Claude Desktop) configured to talk to this server:

- Call the tool **`perguntar_runbook`** with a `pergunta` string, for example:

```text
perguntar_runbook("Como devo agir em um incidente de falha no Delta Share para o domínio BEES Data?")
```

The server will:

- Retrieve the most relevant snippets from the indexed runbooks/documents.
- Generate a clear, professional answer in **Brazilian Portuguese**.
- Explicitly state when the information is not available in the documents.

This tool is **not** intended for generic questions (weather, news, finance, etc.); it should be used only for topics covered by the loaded runbooks and Data Ops documentation.

