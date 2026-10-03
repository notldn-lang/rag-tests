# rag-demo

A bare bones RAG application for educational purposes.

DISCLAIMER: There are several concepts in this repository that can be implemented in much better ways.  The point of this repository is to remove unfamiliar terms and abstractions as much as possible to demonstrate the essential concepts of a RAG application.

You should get acquainted first with RAG and [when to use it and when not to.](https://www.anthropic.com/news/contextual-retrieval)
Embeddings and answers are generated with the [Google Gemini API](https://ai.google.dev/gemini-api/docs) (`gemini-embedding-001` and `gemini-3.8-flash`). Retrieval is a hybrid of pgvector similarity and Postgres full-text search, fused with Reciprocal Rank Fusion, then reranked with [Voyage AI](https://docs.voyageai.com/docs/reranker) (`rerank-2-lite`).

## Prerequisites

- **Python 3.12 or higher**: Ensure you have Python 3.12 or a later version installed.
- **Poetry**: Install Poetry on your machine for dependency management.
- **Docker**: Ensure Docker is installed in your WSL environment.
- **SSH Key**: Configure your SSH key with your GitHub account if you haven't already.

Links:
- [Updating Python to 3.12 in WSL](https://stackoverflow.com/questions/78284506/how-to-update-python-to-the-latest-version-3-12-2-in-wsl2)
- [Installing Poetry with Official installer](https://python-poetry.org/docs/#installing-with-the-official-installer)
- [Generating a new SSH key and adding it to the ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)


## Sample output
```
Cleaning database...
Creating embedding for chunk: ize
human priors to ...
Creating embedding for chunk: odifications, ensuri...
Creating embedding for chunk: u et al., 2023; Qian...
Creating embedding for chunk: ned agents and meta-...
Creating embedding for chunk: n Hendrycks, Collin ...

Total index time: 11.419859999790788ms

Enter question: what are the main contributions from the godel agents paper?
scores:  [46.2205144035501, 45.83206449938195, 44.91440161297333, 44.8048227792767, 44.57226661278396]

Using 5 chunks in answer. Answer:

Compiling declarative language model calls into self-improving pipelines
```

## Setup:

This setup works for my WSL installation of Ubuntu with root access.

1. **Clone the repository**:
```bash
git clone git@github.com:smferro54/rag-workshop-from-scratch.git
cd rag-workshop-from-scratch/
```

2. **Start a pgvector docker container**:

```bash
docker run -p 6432:5432  --name pgvector -e POSTGRES_PASSWORD=postgres -d pgvector/pgvector:pg17
```

Note: The host port for the docker container is 6432 instead of the normal 5432 to avoid port collisions

3. **Setup the database: Ensure you are in the directory where the repository was cloned.**
```bash
psql -h localhost -p 6432 -U postgres -c "CREATE DATABASE rag_demo;"
psql -h localhost -p 6432 -U postgres rag_demo < schema.sql
```
Note: All psql commands will prompt for a password, which is **"postgres"**.

*Troubleshoot*: 
```bash
sudo apt update
sudo apt install postgresql-client-common
sudo apt install postgresql-client
```

4. **Create the .env file:**
```bash
touch .env
```
Note: You can use the example_env.txt as a reference for the required environment variables (`GEMINI_API_KEY` and `VOYAGE_API_KEY`).

5. **Export your API key environment variables:**
```
source .env
```

6. **Install dependencies with Poetry:**
```
poetry install
```
*Troubleshoot*: 
```
sudo apt update
curl -fsSL https://pyenv.run | bash
pyenv install 3.12.0
pyenv local 3.12.0
```

### Running

`poetry run python -m rag_demo`

You can skip the embedding step if you already have a database and want to experiment with different models. 
`poetry run python -m rag_demo --skip-embedding-step`

### Check chunks table first 5 rows
```
psql -h localhost -p 6432 -U postgres rag_demo -c "SELECT * FROM chunks LIMIT 5;"
```
