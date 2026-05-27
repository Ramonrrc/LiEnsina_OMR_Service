# LiEnsina OMR Service

Microservico FastAPI interno para leitura de cartao resposta estilo ENEM com visao computacional tradicional.

O fluxo correto e:

```text
React -> NestJS -> FastAPI OMR -> NestJS -> PostgreSQL
```

O OMR nao acessa banco, usuarios, turmas ou regras pedagogicas. Ele retorna sugestoes tecnicas de leitura; autorizacao, nota final, ownership e auditoria ficam no back-end.

## Endpoints

- `GET /health`: healthcheck interno.
- `POST /v1/omr/process`: processa uma imagem/PDF.
- `POST /v1/omr/process-batch`: processa paginas de um PDF/lote.

Em producao, os endpoints de processamento exigem `X-LiEnsina-OMR-Token` igual a `OMR_INTERNAL_TOKEN`. A documentacao automatica do FastAPI fica desativada em producao.

## Seguranca

- O container roda como usuario nao-root.
- Em producao, nao publique a porta 8000 no host; use somente a rede interna do Docker Compose.
- O back-end aplica timeout, circuit breaker e rate limit antes de chamar o OMR.
- O OMR valida magic bytes e aceita apenas JPEG, PNG, WEBP ou PDF.
- SVG e GIF sao rejeitados.
- O tamanho maximo e controlado por `OMR_MAX_UPLOAD_MB`.

## Rodando localmente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set OMR_INTERNAL_TOKEN=replace_with_64_plus_random_chars_internal_omr_token
uvicorn app.main:app --reload --port 8000
```

## Docker local

Para diagnostico local isolado:

```bash
docker build -t liensina-omr-service .
docker run --rm -e OMR_ENV=development -e OMR_INTERNAL_TOKEN=replace_with_64_plus_random_chars_internal_omr_token -p 127.0.0.1:18080:8000 liensina-omr-service
```

Para producao, use o `docker-compose.yml` da raiz, que usa `expose: 8000` e nao publica porta no host.

## Variaveis

- `OMR_ENV=development|production`
- `OMR_INTERNAL_TOKEN`: token interno forte.
- `OMR_MAX_UPLOAD_MB=16`
- `OMR_REQUEST_CONCURRENCY=1`
- `OMR_PAGE_CONCURRENCY=1`
- `OMR_OPENCV_THREADS=1`
- `OMR_PDF_RENDER_SCALE=2.6`
- `OMR_REQUEST_TIMEOUT_SECONDS=120`
- `OMR_MAX_PDF_PAGES=6`
- `OMR_MIN_CONFIDENCE_FOR_AUTO_APPROVAL=0.88`

## Testes de seguranca

Instale as dependencias de teste fora da imagem de producao:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

Os testes cobrem token interno obrigatorio, magic bytes, rejeicao de payload extra, limite de paginas por PDF, timeout e validacao de QR inconsistente.

## Benchmark OMR local

```bash
python scripts/benchmark_omr.py --file caminho/do/cartao.pdf --questions 100 --iterations 1 --concurrency 1
```

Na maquina local testada, o PDF `cartoes_resposta_preenchidos_fotos_celular.pdf` com 20 paginas e gabarito de 100 questoes ficou melhor com `OMR_PAGE_CONCURRENCY=1`.
