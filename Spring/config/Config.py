import time

def getNowTime():
    return int(round(time.time()))

class Config:
    def __init__(self):

        self.namespace = "hipster"

        self.SLO = 200
        self.max_pod = 8
        self.min_pod = 1

        self.k8s_config = '/your_k8s_admin.conf'
        self.k8s_yaml = '/your_benchmark.yaml'

        self.duration = 1 * 20 * 60  # 20 min
        self.start = getNowTime()
        self.end = self.start + self.duration

        self.prom_range_url = "http://your_prometheus_port/api/v1/query_range"
        self.prom_no_range_url = "http://your_prometheus_port/api/v1/query"
        
        self.step = 5
