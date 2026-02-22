# Openclaw - Tutorial Documentation OpenClaw

## Visão Geral

Esta é uma documentação e tutorial de instalação e configuração do OpenClaw 2026.1.30.

Comando para ver versão:
```bash
openclaw --version
```

---

## Configuração do OpenClaw

- Dados armazenados em formato Delta
- Não há validação nesta etapa
- Schema pode variar

---

## Configuração de LM Studio

### Ambiente

```bash
conda activate wh
```

### Servidor

```bash
lms server start          # Inicia o servidor (porta 1234)
lms server stop           # Para o servidor
lms server status         # Verifica status do servidor
```

### Modelos

```bash
lms load                  # Carrega um modelo (seleção interativa)
lms load gpt_4o_mini --context-length 16384
lms unload                # Descarrega um modelo
lms unload gpt_4o_mini    # Descarrega modelo específico
lms unload --all          # Descarrega todos os modelos
lms ps                    # Lista modelos carregados na memória
lms ls                    # Lista modelos disponíveis em disco
```

### Outros comandos úteis

```bash
lms --help
lms chat                  # Chat interativo com o modelo
lms get                   # Buscar e baixar modelos
lms import                # Importar arquivo de modelo
```

### API HTTP (curl)

Listar modelos disponíveis:
```bash
curl http://localhost:1234/v1/models
```

Testar chat completion:
```bash
curl http://localhost:1234/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"gpt_4o_mini\", \"messages\": [{\"role\": \"user\", \"content\": \"oi\"}], \"max_tokens\": 100}"
```

### Fluxo recomendado

1. `conda activate wh` — Ativar ambiente
2. `lms server start` — Iniciar servidor
3. `lms load` — Carregar modelo
4. `lms server status` — Confirmar que está rodando

---

## Comandos de OpenClaw

### Gateway

```bash
openclaw gateway              # Inicia o gateway (modo interativo)
openclaw gateway restart      # Reinicia o gateway (Scheduled Task)
```

O gateway fica disponível em `http://127.0.0.1:18789` e o canvas em `http://127.0.0.1:18789/__openclaw__/canvas/`.

### Daemon / Serviço

```bash
openclaw daemon restart       # Reinicia o daemon (Scheduled Task)
```

### Diagnóstico e correção

```bash
openclaw doctor               # Verifica a instalação e configuração
openclaw doctor --fix         # Corrige problemas (remove chaves inválidas do openclaw.json)
openclaw security audit --deep
```

### Observações

- Em caso de erro "Unrecognized key" no `openclaw.json`, execute `openclaw doctor --fix`
- O backup da configuração é salvo em `openclaw.json.bak`
