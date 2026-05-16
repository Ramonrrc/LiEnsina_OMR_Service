# LiEnsina OMR Service

Microservico FastAPI independente para leitura de cartao resposta estilo ENEM usando visao computacional tradicional.

Responsabilidades deste servico:

- receber imagem ou PDF do cartao resposta
- validar qualidade basica da foto
- detectar QR Code
- alinhar perspectiva usando marcadores
- detectar bolhas preenchidas com OpenCV
- comparar com gabarito enviado pelo NestJS
- retornar sugestao estruturada de correcao

Ele nao acessa PostgreSQL, Prisma, usuarios, turmas ou regras pedagogicas. O fluxo correto e:

```text
React -> NestJS -> FastAPI OMR -> NestJS -> PostgreSQL
```

## Endpoints

### `GET /health`

Retorna status do servico.

### `POST /v1/omr/process`

Multipart form:

- `image`: arquivo da foto do cartao ou PDF com o cartao na primeira pagina
- `payload`: JSON string com metadados e gabarito

Payload exemplo:

```json
{
  "examId": "14952a03-c6ee-4af0-9f02-598dab657a52",
  "versionId": "14952a03-c6ee-4af0-9f02-598dab657a52:v1",
  "answerCardId": "answer-card-2026-0001",
  "studentId": "student-uuid",
  "classId": "class-uuid",
  "templateVersion": "liensina-omr-v1",
  "answerKey": [
    { "questionNumber": 1, "questionId": "q1", "correctOption": "A" },
    { "questionNumber": 2, "questionId": "q2", "correctOption": "C" }
  ]
}
```

Resposta exemplo:

```json
{
  "examId": "14952a03-c6ee-4af0-9f02-598dab657a52",
  "versionId": "14952a03-c6ee-4af0-9f02-598dab657a52:v1",
  "answerCardId": "answer-card-2026-0001",
  "studentId": "student-uuid",
  "classId": "class-uuid",
  "templateVersion": "liensina-omr-v1",
  "suggestedScore": 7.5,
  "correctCount": 15,
  "wrongCount": 4,
  "blankCount": 0,
  "multipleCount": 1,
  "totalQuestions": 20,
  "confidence": 0.82,
  "requiresReview": true,
  "shouldRetakeImage": false,
  "failures": ["MULTIPLE_MARKS_DETECTED"],
  "quality": {
    "width": 1280,
    "height": 1920,
    "brightness": 132.4,
    "contrast": 48.2,
    "blur": 214.7,
    "warnings": [],
    "confidence": 0.91
  },
  "qrCode": {
    "found": true,
    "raw": "{\"examId\":\"...\"}",
    "parsed": { "examId": "14952a03-c6ee-4af0-9f02-598dab657a52" },
    "warnings": []
  },
  "detectedAnswers": [
    {
      "questionNumber": 1,
      "questionId": "q1",
      "detectedOption": "A",
      "correctOption": "A",
      "isCorrect": true,
      "status": "ok",
      "confidence": 0.94,
      "markedOptions": ["A"],
      "optionScores": [
        { "option": "A", "fillRatio": 0.51 },
        { "option": "B", "fillRatio": 0.13 }
      ]
    }
  ],
  "metadata": { "engine": "opencv-threshold-v1", "autoApprovalEligible": false }
}
```

## Estrategia de OMR

1. Se o arquivo for PDF, renderiza a primeira pagina para imagem.
2. Converte a imagem para tons de cinza.
3. Calcula qualidade: resolucao, brilho, contraste e blur.
4. Busca QR Code com `pyzbar`.
5. Detecta quatro marcadores pretos do cartao.
6. Aplica transformacao de perspectiva para um canvas canonico `1100x1550`.
7. Calcula as coordenadas esperadas das bolhas conforme a quantidade de questoes.
8. Para cada bolha, aplica Otsu threshold invertido e mede a razao de pixels escuros dentro de uma mascara circular.
9. Classifica cada linha como `ok`, `blank`, `multiple` ou `low_confidence`.
10. Compara com o gabarito recebido do NestJS e retorna sugestao.

## Rodando localmente

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Docker

```bash
docker build -t liensina-omr-service .
docker run --rm -p 8000:8000 liensina-omr-service
```

## Modelagem futura Prisma

O backend atual do projeto salva colecoes JSON em PostgreSQL. Caso migre para Prisma relacional, use `docs/prisma-model.example.prisma` como referencia.
