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
