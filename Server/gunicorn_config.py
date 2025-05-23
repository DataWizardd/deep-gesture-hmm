import multiprocessing
import os

# Gunicorn 설정
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5

# 로그 설정
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"

# 워커 설정
max_requests = 1000
max_requests_jitter = 50

# 프로세스 이름
proc_name = "signwave_app"
