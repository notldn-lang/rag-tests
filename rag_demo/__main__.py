### Utility libraries
import argparse
import os
import time
from dotenv import load_dotenv

### Google GenAI для эмбеддингов и для ответов
from google import genai

### PostgreSQL adapter for Python
import psycopg

### Voyage AI для reranking (ключ в VOYAGE_API_KEY в .env)
import voyageai

### PyPDF for text extraction
from PyPDF2 import PdfReader

### Constants
load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Изменили модель эмбеддингов на бесплатную от Google
EMBEDDING_MODEL = "gemini-embedding-001"
ANSWER_MODEL = "gemini-3.8-flash"

# Инициализируем клиент (ключ берём из GEMINI_API_KEY в .env, используется и для
# эмбеддингов, и для ответов)
google_client = genai.Client()

# Клиент Voyage AI для reranking (ключ берём из VOYAGE_API_KEY в .env)
voyage_client = voyageai.Client()
RERANK_MODEL = "rerank-2-lite"

### Argument parser
parser = argparse.ArgumentParser(description="RAG Demo with Google Embeddings")
parser.add_argument(
    "--skip-embedding-step",
    action="store_true",
    help="Skip the embedding step and use the existing embeddings if this flag is provided.",
)
args = parser.parse_args()

### Useful functions
def get_embedding(text):
    """Получение эмбеддинга от Google (text-embedding-004)"""
    response = google_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return response.embeddings[0].values

def get_answer(question, context):
    prompt = f"""
Answer the question using only the following context:

{context}

Question: {question}
"""
    response = google_client.models.generate_content(
        model=ANSWER_MODEL,
        contents=prompt,
    )
    return response.text, prompt


### PostgreSQL database url and connection
# ВНИМАНИЕ: Убедитесь, что колонка в таблице chunks поддерживает размерность 3072 (для gemini-embedding-001)
database_url = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:6432/rag_demo"
)
db = psycopg.Connection.connect(database_url)


# Напоминание: это простая нарезка для примера. В проде используйте нормальные чанкеры.
def split_string_by_length(input_string, length, overlap=0):
    step = length - overlap
    return [input_string[i : i + length] for i in range(0, len(input_string), step)]


# Loop through chunks from the pdf and create embeddings in the database
if not args.skip_embedding_step:
    print("Cleaning database...")
    db.execute("TRUNCATE TABLE chunks")

    tic = time.perf_counter()
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)

        # Пропускаем скрытые файлы (например, .DS_Store)
        if filename.startswith('.'):
            continue

        reader = PdfReader(file_path)
        content = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content += text

        for chunk in split_string_by_length(content, CHUNK_SIZE, CHUNK_OVERLAP):
            print(f"Creating embedding for chunk: {chunk[0:20]}...")
            
            vector = get_embedding(chunk)
            
            db.execute(
                "INSERT INTO chunks (embedding, chunk) VALUES (%s, %s)",
                [str(vector), chunk],
            )
            
            # Небольшая пауза (0.5 сек), чтобы не упереться в бесплатные лимиты RPM (Requests Per Minute) от Google
            time.sleep(0.5)

        print(f"\nTotal index time: {time.perf_counter() - tic}ms")
        db.commit()

question = input("\nEnter question: ")

# Create embedding from question using Google
question_embedding = get_embedding(question)

# Hybrid search: fuse vector similarity (semantic) with Postgres full-text search
# (keyword, via ts_rank_cd) using Reciprocal Rank Fusion, since the two scores
# aren't on comparable scales — only their rank order is.
RRF_K = 60
CANDIDATE_LIMIT = 20

result = db.execute(
    """
    WITH vector_search AS (
        SELECT id, chunk, row_number() OVER (ORDER BY embedding <=> %(qvec)s::vector) AS rank
        FROM chunks
        ORDER BY embedding <=> %(qvec)s::vector
        LIMIT %(limit)s
    ),
    text_search AS (
        SELECT id, chunk,
               row_number() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %(question)s)) DESC) AS rank
        FROM chunks
        WHERE tsv @@ plainto_tsquery('english', %(question)s)
        LIMIT %(limit)s
    )
    SELECT
        COALESCE(v.id, t.id) AS id,
        COALESCE(v.chunk, t.chunk) AS chunk,
        COALESCE(1.0 / (%(rrf_k)s + v.rank), 0.0) + COALESCE(1.0 / (%(rrf_k)s + t.rank), 0.0) AS score
    FROM vector_search v
    FULL OUTER JOIN text_search t ON v.id = t.id
    ORDER BY score DESC
    LIMIT %(limit)s
    """,
    {
        "qvec": str(question_embedding),
        "question": question,
        "limit": CANDIDATE_LIMIT,
        "rrf_k": RRF_K,
    },
)

rows = list(result)

print("RRF candidate scores: ", [row[2] for row in rows])

# Rerank the fused candidates with Voyage AI and keep only the top 5 by
# true relevance to the question, rather than trusting RRF order directly.
candidate_chunks = [row[1] for row in rows]
reranking = voyage_client.rerank(
    question, candidate_chunks, model=RERANK_MODEL, top_k=5
)

print("rerank scores: ", [r.relevance_score for r in reranking.results])
context = "\n\n".join([r.document for r in reranking.results])

answer, prompt = get_answer(question, context)

print(f"\nUsing {len(reranking.results)} chunks in answer. Answer:\n")
print(answer)

view_prompt = input("\nWould you like to see the raw prompt? [Y/N] ")
if view_prompt == "Y":
    print("\n" + prompt)