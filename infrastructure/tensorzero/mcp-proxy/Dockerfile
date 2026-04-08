FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir aiohttp
COPY proxy.py .
CMD ["python", "proxy.py"]
