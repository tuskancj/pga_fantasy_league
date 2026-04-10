FROM public.ecr.aws/lambda/python:3.12
ENV HOME=/tmp
RUN pip install 'vanna[anthropic]' 'chromadb==0.5.23' boto3 psycopg2-binary --no-cache-dir
COPY lambda_function.py /var/task/
CMD ["lambda_function.lambda_handler"]
