import time
import numpy as np
import schedule
from bayes_opt import BayesianOptimization
from util.KubernetesClient import KubernetesClient
from util.PrometheusClient import PrometheusClient
from config.Config import Config
import threading


class MicroScaler:
    def __init__(self, config: Config):
        self.prom_url = config.prom_no_range_url
        self.config_path = config.k8s_config
        self.namespace = config.namespace

        # service power小于p_min的服务需要收缩
        self.p_min = 0.7
        # service power大于p_max的服务需要收缩
        self.p_max = 0.833
        self.n_iter = 3

        # 超过latency_max的服务即为abnormal
        self.latency_max = config.SLO
        self.pod_max = config.max_pod

        self.duration = config.duration
        self.k8s_util = KubernetesClient(config)
        self.prom_util = PrometheusClient(config)

        self.mss = self.k8s_util.get_svcs()
        self.so = set()
        self.si = set()

    # 运营成本
    def price(self, pod_count):
        return pod_count

    # 获取service power = p50 / p90
    def p_value(self, svc):
        # 设置prometheus的TSDB时间窗口
        begin = int(round((time.time() - 30)))
        end = int(round(time.time()))
        self.prom_util.set_time_range(begin, end)

        p90 = self.prom_util.p90(svc)
        p50 = self.prom_util.p50(svc)
        if p90 == 0:
            return np.NaN
        else:
            return float(p50) / float(p90)

    # 检查所有的SLO异常服务，并调用service power来筛选，需要in和out的服务分别存储在si和so
    def detector(self):
        
        # 获取所有服务的Pod数量
        svc_count_dic = self.k8s_util.get_svcs_counts()
        [print(svc, svc_count_dic[svc]) for svc in svc_count_dic.keys() if svc_count_dic[svc] != 1]
        
        # 根据SLO检查异常服务
        svcs = self.k8s_util.get_svcs()
        ab_svcs = []
        for svc in svcs:
            
            begin = int(round((time.time() - 30)))
            end = int(round(time.time()))
            self.prom_util.set_time_range(begin, end)
            
            t = self.prom_util.p90(svc)
            # print(svc, t)
            if t > self.latency_max:
                ab_svcs.append(svc)

        self.service_power(ab_svcs)

    # 根据service power筛选异常服务
    def service_power(self, ab_svcs):
        for ab_svc in ab_svcs:
            p = self.p_value(ab_svc)
            if np.isnan(p):
                continue
            elif p > self.p_max:
                self.si.add(ab_svc)
            # elif p < self.p_min:
            else:
                self.so.add(ab_svc)

    # 伸缩
    def auto_scale(self):

        # 扩张svc out中的svc
        for svc in self.so:
            origin_pod_count = self.k8s_util.get_svc_count(svc)
            if origin_pod_count == self.pod_max:
                continue
            index = self.mss.index(svc)

            # 指定需要优化的参数和范围：x，从now到max
            pbounds = {'x': (origin_pod_count, self.pod_max), 'index': [index, index]}
            # 创建线性进行BO，BO的奖励函数是scale返回的score，BO的优化参数是pbounds
            t = threading.Thread(target=self.BO, args=(self.scale, pbounds))
            t.setDaemon(True)
            t.start()

        # 收缩svc in中的svc
        for svc in self.si:
            origin_pod_count = self.k8s_util.get_svc_count(svc)
            index = self.mss.index(svc)
            if origin_pod_count == 1:
                continue
            pbounds = {'x': (1, origin_pod_count), 'index': [index, index]}
            t = threading.Thread(target=self.BO, args=(self.scale, pbounds))
            t.setDaemon(True)
            t.start()
        self.so.clear()
        self.si.clear()

    # 将mss中第index个svc调整为x个
    def scale(self, x, index):
        svc = self.mss[int(index)]
        self.k8s_util.patch_scale(svc, int(x))
        print('{} is scaled to {}'.format(svc, int(x)))
        
        # 保证所有的Pod都可用
        while True:
            if self.k8s_util.all_avaliable():
                break

        time.sleep(30)

        svcs_counts = self.k8s_util.get_svcs_counts()
        for svc in svcs_counts.keys():

            begin = int(round((time.time() - 30)))
            end = int(round(time.time()))
            self.prom_util.set_time_range(begin, end)

            P90 = self.prom_util.p90(svc)

            score = 0
            if P90 > self.latency_max:
                # 奖励函数是运营成本和p90的平衡，
                score = -P90 * self.price(svcs_counts[svc]) - P90 * 10
            else:
                score = -P90 * self.price(svcs_counts[svc]) + P90 * 10
            return score

    # BO优化
    def BO(self, f, pbounds):
        optimizer = BayesianOptimization(
            f=f,
            pbounds=pbounds,
            random_state=1,
        )
        # 默认内核函数
        gp_param = {'kernel': None}
        optimizer.maximize(
            **gp_param,
            # 优化迭代次数
            n_iter=self.n_iter
        )

    def auto_task(self):
        print("microscaler go...")
        self.detector()
        schedule.clear()

        # 获取异常服务
        schedule.every(30).seconds.do(self.detector)
        
        # 伸缩
        schedule.every(2).minutes.do(self.auto_scale)

        time_start = time.time()  # 开始计时
        while True:
            time_c = time.time() - time_start
            if time_c > self.duration:
                # 超过指定运行时间，退出
                schedule.clear()
                break
            schedule.run_pending()

        print("microscaler stop...")

    def start(self):
        self.auto_task()
