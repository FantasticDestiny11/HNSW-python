```sql
CREATE DATABASE demo;
CREATE EXTENSION vector;
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));
INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;

CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
```