# Documentação do Script RAG - main_mcp copy.ipynb

## Visão Geral

Este notebook implementa um pipeline completo de **RAG (Retrieval-Augmented Generation)** — um sistema que combina busca semântica em documentos com geração de respostas usando um modelo de linguagem local. O objetivo é permitir que um LLM responda perguntas com base no conteúdo de arquivos PDF, sem necessidade de internet.

---

## Dependências

Antes de executar o script, instale as bibliotecas necessárias:

```bash
pip install PyMuPDF
pip install sentence-transformers
pip install faiss-cpu   # ou faiss-gpu se tiver GPU NVIDIA
pip install gpt4all
```

**Bibliotecas utilizadas:**
- **PyMuPDF (fitz)** — Extração de texto de arquivos PDF
- **sentence-transformers** — Criação de embeddings (vetores semânticos)
- **numpy** — Manipulação de arrays numéricos
- **faiss** — Índice vetorial para busca por similaridade
- **gpt4all** — Interface com modelos LLM locais (TinyLlama)

---

## Arquitetura do Pipeline

O fluxo do script segue estas etapas:

```
PDF → Extração de Texto → Divisão em Chunks → Embeddings → Índice FAISS
                                                                    ↓
Pergunta do Usuário → Embedding da Pergunta → Busca no FAISS → Chunks Relevantes
                                                                    ↓
                                            Prompt (Contexto + Pergunta) → LLM → Resposta
```

---

## Componentes Principais

### 1. Extração de Texto do PDF

**Função:** `extrair_texto_do_pdf(caminho_pdf)`

- Abre o arquivo PDF com PyMuPDF
- Percorre todas as páginas e extrai o texto
- Retorna uma string com o conteúdo completo

**Uso:** Passe o caminho absoluto ou relativo do arquivo PDF.

---

### 2. Divisão em Chunks

**Função:** `dividir_em_chunks(texto, tamanho_chunk=500, sobreposicao=50)`

- Divide o texto longo em blocos menores para processamento
- **tamanho_chunk:** Número de caracteres por bloco (padrão: 500)
- **sobreposicao:** Caracteres compartilhados entre blocos consecutivos (padrão: 50)

A sobreposição ajuda a evitar que informações importantes sejam cortadas entre chunks.

---

### 3. Criação de Embeddings

**Função:** `criar_embeddings(chunks, nome_modelo)`

- Usa o modelo **BGE-small-en-v1.5** para converter cada chunk em um vetor numérico (embedding)
- Os embeddings permitem busca semântica: textos com significado parecido têm vetores próximos
- Cada vetor tem 384 dimensões

**Modelo usado:** `./bge-small-en-v1.5` (local) — bom equilíbrio entre performance e qualidade.

---

### 4. Salvamento dos Dados

Os embeddings e chunks são salvos em disco:

- `embeddings.npy` — Array NumPy com os vetores
- `chunks.pkl` — Lista de chunks em formato pickle

---

### 5. Índice FAISS

- **FAISS** (Facebook AI Similarity Search) cria um índice para busca rápida por similaridade
- Usa distância L2 (euclidiana) entre vetores
- Permite encontrar os K chunks mais semelhantes a uma pergunta em pouco tempo

---

### 6. Busca de Contexto

**Função:** `buscar_contexto(pergunta, k=3)`

- Converte a pergunta em embedding
- Busca os **k** chunks mais similares no índice FAISS
- Retorna uma lista com os trechos de texto relevantes

---

### 7. Montagem do Prompt

**Função:** `montar_prompt(pergunta, contextos)`

- Combina o contexto recuperado com a pergunta
- Usa o formato de chat do modelo (tags `<|system|>`, `<|user|>`, `<|assistant|>`)
- Instrui o LLM a responder apenas com base no contexto e em português

---

### 8. Geração da Resposta

**Função:** `responder_rag(pergunta)`

- Integra todo o pipeline: busca → prompt → LLM
- Usa **GPT4All** com o modelo **TinyLlama** (1.1B parâmetros, quantizado Q4_K_M)
- Configuração: `max_tokens=800`, `temp=0.2`

---

## Configuração

### Arquivo PDF

Altere a variável `CAMINHO_DO_PDF` na célula principal:

```python
CAMINHO_DO_PDF = r"C:\caminho\para\seu\arquivo.pdf"
```

### Modelo de Embedding

```python
MODELO_EMBEDDING = './bge-small-en-v1.5'
```

### Modelo LLM

```python
llm = GPT4All(
    r"./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    device="gpu"   # ou "cpu"
)
```

---

## Ordem de Execução

1. **Células 0–5:** Instalação de dependências, imports e configuração do PyTorch
2. **Célula 6:** Extração do PDF, divisão em chunks, criação de embeddings
3. **Célula 7:** Salvamento de embeddings e chunks
4. **Célula 8:** Carregamento dos dados e criação do índice FAISS
5. **Células 9–10:** Definição das funções de busca e montagem do prompt
6. **Células 11–12:** Inicialização do LLM e função RAG completa
7. **Célula 13:** Exemplo de pergunta e resposta

---

## Exemplo de Uso

```python
pergunta = "What is the Data Ops L2 process?"
resposta = responder_rag(pergunta)
print(resposta)
```

---

## Notas Técnicas

- **GPU:** O script usa GPU (CUDA/ROCm) quando disponível; avisos sobre "Flash Efficient attention" em AMD podem aparecer
- **Memória:** O modelo TinyLlama e os embeddings consomem RAM/VRAM; em máquinas fracas, use `device="cpu"`
- **Primeira execução:** O modelo BGE será baixado automaticamente na primeira vez
- **Formato de chunks:** Os chunks são salvos em pickle; evite modificar a estrutura sem reprocessar

---

## Próximos Passos Sugeridos

- Integrar um banco vetorial persistente (ex.: ChromaDB) em vez de pickle/NumPy
- Ajustar `tamanho_chunk` e `sobreposicao` conforme o tipo de documento
- Experimentar modelos maiores (ex.: Llama 3) para respostas mais complexas
- Adicionar interface (Streamlit/Gradio) para perguntas interativas




































# =============================================================================
# MCP Server - Assistente RAG para Runbooks / Data Ops / Delta Share
# =============================================================================

# No topo do arquivo, antes de qualquer outro import torch-related
import torch
if not hasattr(torch.distributed, 'is_initialized'):
    torch.distributed.is_initialized = lambda: False

# ⭐ CRITICAL: Redirect ALL output to stderr
import sys
import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

# ⭐ Configure logging to stderr ONLY
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Now import other modules
from fastmcp import FastMCP
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
from typing import List

# ────────────────────────────────────────────────
# CONFIGURAÇÕES
# ────────────────────────────────────────────────

# ⭐ Use absolute paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_PATH = os.path.join(SCRIPT_DIR, "embeddings.npy")
CHUNKS_PATH     = os.path.join(SCRIPT_DIR, "chunks.pkl")
MODEL_EMBEDDING = os.path.join(SCRIPT_DIR, "bge-small-en-v1.5")
MODEL_LLM_PATH  = os.path.join(SCRIPT_DIR, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Parâmetros de busca e geração
K_RETRIEVE      = 4
MAX_TOKENS      = 1000
TEMPERATURE     = 0.18
REPEAT_PENALTY  = 1.12

# ────────────────────────────────────────────────
# CARREGAMENTO DOS COMPONENTES
# ────────────────────────────────────────────────

# ⭐ Replace all print() with logger
logger.info("Inicializando RAG local... (pode demorar na primeira execução)")

try:
    # 1. Embeddings + chunks
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    logger.info(f"✓ Embeddings carregados: {embeddings.shape}")
    
    with open(CHUNKS_PATH, "rb") as f:
        chunks: List[str] = pickle.load(f)
    logger.info(f"✓ {len(chunks)} chunks carregados")

    # 2. Índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logger.info(f"✓ Índice FAISS: {index.ntotal} vetores × {dimension} dimensões")

    # 3. Modelos
    embedding_model = SentenceTransformer(MODEL_EMBEDDING)
    logger.info("✓ Embedding model carregado")
    
    llm = GPT4All(
        model_name=MODEL_LLM_PATH,
        device="gpu",  # mude para "cpu" se necessário
    )
    logger.info("✓ LLM carregado")
    
    logger.info("=" * 60)
    logger.info("Servidor RAG pronto para receber perguntas!")
    logger.info("=" * 60)

except FileNotFoundError as e:
    logger.error(f"❌ Arquivo não encontrado: {e}")
    logger.error(f"Diretório de trabalho: {SCRIPT_DIR}")
    logger.error("Certifique-se de que os seguintes arquivos existem:")
    logger.error(f"  - {EMBEDDINGS_PATH}")
    logger.error(f"  - {CHUNKS_PATH}")
    logger.error(f"  - {MODEL_EMBEDDING}")
    logger.error(f"  - {MODEL_LLM_PATH}")
    sys.exit(1)
except Exception as e:
    logger.error(f"❌ Erro ao inicializar: {e}", exc_info=True)
    sys.exit(1)

# ────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ────────────────────────────────────────────────

def buscar_contexto(pergunta: str, k: int = K_RETRIEVE) -> List[str]:
    """Busca os k trechos mais similares à pergunta"""
    try:
        emb_query = embedding_model.encode([pergunta]).astype(np.float32)
        distances, indices = index.search(emb_query, k)
        resultados = []
        for idx in indices[0]:
            if idx < len(chunks):
                resultados.append(chunks[idx])
        logger.info(f"Busca: encontrados {len(resultados)} trechos relevantes")
        return resultados
    except Exception as e:
        logger.error(f"Erro na busca: {e}")
        return []


def montar_prompt(pergunta: str, contextos: List[str]) -> str:
    """Monta o prompt no formato que o modelo espera"""
    if not contextos:
        contexto_texto = "(nenhum trecho relevante encontrado)"
    else:
        contexto_texto = "\n\n".join(
            f"─── Trecho {i+1} ───\n{ctx.strip()}\n" 
            for i, ctx in enumerate(contextos)
        )

    return f"""<|system|>
Você é um assistente técnico especializado em runbooks internos, processos Data Ops, Delta Share, governança de dados e procedimentos ITOPS.
Responda SOMENTE com base no contexto fornecido abaixo.
Se a informação não estiver no contexto ou não for clara, responda apenas:
"Não encontrei essa informação nos documentos disponíveis no momento."

Use linguagem clara, objetiva, profissional e em português do Brasil.
</s>

<|user|>
CONTEXTO DISPONÍVEL:
{contexto_texto}

PERGUNTA:
{pergunta}
</s>

<|assistant|>"""


# ────────────────────────────────────────────────
# SERVIDOR MCP
# ────────────────────────────────────────────────

mcp = FastMCP(
    name="DataOps & Delta Share Helper",
)


@mcp.tool()
def perguntar_runbook(pergunta: str) -> str:
    """
    Ferramenta principal para responder dúvidas sobre:
    • Processos Data Ops (L1, L2, L3...)
    • Delta Share (configuração, troubleshooting, permissões...)
    • Governança de dados
    • Runbooks e procedimentos ITOPS
    • Qualquer tema presente nos documentos carregados

    NÃO usar para perguntas genéricas (clima, notícias, cotação dólar, etc).
    """
    logger.info(f"Recebida pergunta: {pergunta[:100]}...")
    
    contextos = buscar_contexto(pergunta)

    if not contextos:
        return "Nenhum trecho relevante encontrado nos documentos carregados."

    prompt = montar_prompt(pergunta, contextos)

    try:
        logger.info("Gerando resposta com LLM...")
        with llm.chat_session():
            resposta = llm.generate(
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temp=TEMPERATURE,
                repeat_penalty=REPEAT_PENALTY,
            )

        # Limpeza básica da resposta
        resposta = resposta.split("<|assistant|>")[-1].strip()
        resposta = resposta.split("</s>")[0].strip()
        resposta = resposta.split("[/INST]")[-1].strip()

        logger.info("✓ Resposta gerada com sucesso")
        return resposta.strip()

    except Exception as e:
        logger.error(f"Erro ao gerar resposta: {e}", exc_info=True)
        return f"Erro ao gerar resposta: {str(e)}"


# ────────────────────────────────────────────────
# INÍCIO DO SERVIDOR
# ────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("🚀 Iniciando servidor MCP 'DataOps & Delta Share Helper'...")
    logger.info("Aguardando conexões via Claude Desktop...")
    mcp.run()
