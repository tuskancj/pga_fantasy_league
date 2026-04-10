# pga-vanna Lambda

Text-to-SQL Lambda using Vanna AI + ChromaDB + Claude Sonnet.

## What it does
Receives a natural language question, uses Vanna's RAG framework to find 
relevant SQL examples from ChromaDB, sends to Claude Sonnet, returns SQL.

## Architecture
- Runtime: Container image (Python 3.12)
- LLM: Claude Sonnet via Anthropic API (key in Secrets Manager: pga-db-sync-secret → anthropic_key)
- Vector store: ChromaDB in-memory at /tmp/chromadb (rebuilt on cold start)
- Training: 12 verified Q&A examples + DDL schema + documentation (hardcoded in lambda_function.py)

## Deploy
1. cd pga-vanna
2. docker build --platform linux/amd64 --provenance=false -t pga-vanna .
3. docker tag pga-vanna:latest 462913860224.dkr.ecr.us-east-1.amazonaws.com/pga-vanna:latest
4. docker push [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/pga-vanna:latest
5. aws lambda update-function-code --function-name pga-vanna --image-uri [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/pga-vanna:latest --region us-east-1

## ECR Login (if needed)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 462913860224.dkr.ecr.us-east-1.amazonaws.com

## Adding new training examples
Add to TRAINING_EXAMPLES list in lambda_function.py, rebuild and redeploy.
Only add verified SQL that has been tested against the database.

## Lambda config
- Memory: 1024MB
- Timeout: 120s
- Role: pga-dfs-api-role-yndg6zyv
- ENV: SECRET_NAME=pga-db-sync-secret