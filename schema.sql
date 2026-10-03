-- Drop the chunks table if it exists
DROP TABLE IF EXISTS chunks;

-- Drop the vector extension if it exists
DROP EXTENSION IF EXISTS vector;

-- Create the vector extension
CREATE EXTENSION vector;

-- Create the chunks table
-- 3072 dims matches Google's gemini-embedding-001 default output dimension
CREATE TABLE chunks (
    id bigserial PRIMARY KEY,
    embedding vector(3072),
    chunk TEXT,
    -- Full-text search vector for hybrid (keyword + semantic) retrieval
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk)) STORED
);

CREATE INDEX chunks_tsv_idx ON chunks USING GIN (tsv);