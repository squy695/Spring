from cmath import nan
import time
import numpy as np
import schedule
from util.KubernetesClient import KubernetesClient
from util.PrometheusClient import PrometheusClient
from util.Fuzzy import FuzzyController
from util.Macro import Macro
from config.Config import Config
import math


class Spring:
    def __init__(self, config: Config):
        self.config = config
        self.k8s_util = KubernetesClient(config)
        self.prom_util = PrometheusClient(config)
        self.max_pod = config.max_pod
        self.min_pod = config.min_pod
        self.mss = self.k8s_util.get_svcs_without_state()
        if "loadgenerator" in self.mss:
            self.mss.remove("loadgenerator")

        # ZR
        self.SLO_target = 200

        # 采样间隔
        self.scale_interval = 10
        self.save_money_times = 1
        
        self.p = {svc: 0 for svc in self.mss}

        self.fuzzy_default = FuzzyController()

        self.controller_map = {}
        # 一个是p控制器，一个是d控制器
        for svc in self.mss:
            self.controller_map[svc] = [FuzzyController(), FuzzyController()]
        
        # 知识库
        self.knowledge = {svc: {} for svc in self.mss}
        self.no_learn = True

        self.no_macro = False
        self.macro = Macro()

    def sswr(self, x):
        if x < math.floor(x)+0.5:
            x = math.floor(x)
        else:
            x = math.ceil(x)
        return max(x, 1)

    # 记录所有服务的归一化后的p和d，并且存储p，以便后续的p的计算
    def get_p_d(self, p90):
        p = {}
        d = {}
        for svc in self.mss:
            key = svc+"&p90"
            # 采集到的p是nan，说明p=0
            if np.isnan(p90[key]):
                p90[key] = self.SLO_target
            # 初始以600作为p的max，当出现超过600的之后，p=1，就直接上最大count=8了
            p[svc] = max(1, (p90[key] - self.SLO_target) / self.SLO_target)

            d[svc] = (p[svc] - self.p[svc]) / 2
            # 存储当前时延p
            self.p[svc] = p[svc]

        return p, d

    def abnormal(self, p, d):
        return (
            p > self.fuzzy_default.alpha
            or d > self.fuzzy_default.alpha
        )

    # [-0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8] · y = (after - before) / before
    # 返回y[7]
    def analyse(self, before, after):
        x = (after - before) / before

        v = [0, 0, 0, 0, 0, 0, 0]
        if x <= -0.8:
            v[0] = 1
        elif -0.8 < x and x < -0.5:
            v[0] = -(x + 0.5)/0.3
            v[1] = 1 - v[0]
        elif x == -0.5:
            v[1] = 1
        elif -0.5 < x and x < -0.2:
            v[1]= -(x + 0.2)/0.3
            v[2]= 1 - v[1]
        elif x == -0.2:
            v[2] = 1
        elif -0.2 < x and x < 0:
            v[2] = -x / 0.2
            v[3] = 1 - v[2]
        elif x == 0:
            v[3] = 1
        elif 0 < x and x < 0.2:
            v[4] = x / 0.2
            v[3] = 1 - v[4]
        elif x == 0.2:
            v[4] = 1
        elif 0.2 < x and x < 0.5:
            v[4] = (0.5 - x)/0.3
            v[5] = 1- v[4]
        elif x == 0.5:
            v[5] = 1
        elif 0.5 < x and x < 0.8:
            v[5] = (0.8 - x)/0.3
            v[6] = 1- v[5]
        elif x >= 0.8:
            v[6] = 1

        return v

    def set_time(self):
        begin = int(round((time.time() - 30)))
        end = int(round(time.time()))
        self.prom_util.set_time_range(begin, end)


    # 周期性地缩减count
    def save_money(self):

        self.set_time()
    
        svcs_count = self.k8s_util.get_svcs_counts()
        for svc in self.mss:
            self.k8s_util.patch_scale(svc, math.ceil(0.7 * svcs_count[svc]))
        
        print(f'save money {self.save_money_times} times')
        self.save_money_times += 1

    def scale(self):

        self.set_time()

        # 获得所有服务的p90的比例、微分
        svcs_p90 = self.prom_util.get_svc_latency()

        # 集群空闲会导致key error
        if 'adservice&p90' not in svcs_p90.keys():
            return

        svcs_qps = self.prom_util.get_svc_qps()
        svcs_count = self.k8s_util.get_svcs_counts()
        svcs_p, svcs_d = self.get_p_d(svcs_p90)

        abnormal_svc=[]
        true_error_svcs = self.macro.true_error_svcs()

        # 在无异常子图时save money
        if len(true_error_svcs) == 0:
            self.save_money()

        for svc in self.mss:

            # 异常服务
            if self.abnormal(svcs_p[svc], svcs_d[svc]):
                # 这里加入异常服务的判断
                if self.no_macro:
                    abnormal_svc.append(svc)
                else:
                    if svc in true_error_svcs:
                        abnormal_svc.append(svc)

            # 正常服务，记录经验，qps与count的关系，qps以5个为一阶
            else:
                #if self.no_learn:
                #    continue
                if (svcs_qps[svc]//5)*5 in self.knowledge[svc].keys():
                    self.knowledge[svc][(svcs_qps[svc]//5)*5]=(self.knowledge[svc][(svcs_qps[svc]//5)*5]+svcs_count[svc])/2
                else:
                    self.knowledge[svc][(svcs_qps[svc]//5)*5]=svcs_count[svc]


        # 具体的scale操作
        for svc in abnormal_svc:
            # 根据对应的控制器，求得对应的增减比例
            controller = self.controller_map[svc]
            p90_svc = svcs_p90[svc+'&p90']
            qps_svc = svcs_qps[svc+'&qps']
            count_svc = svcs_count[svc]
            p_svc = svcs_p[svc]
            d_svc = svcs_d[svc]

            # 模糊输入
            p_X = controller[0].fuzzize(p_svc)
            d_X = controller[1].fuzzize(d_svc)

            # 推理，得到模糊输出
            Y_svc = controller[0].inference([p_X, d_X])

            # 模糊输出清晰化
            U_svc = self.fuzzy_default.unfuzzize(Y_svc)

            new_count_svc = (1 + U_svc) * count_svc
            new_count_svc = min(max(self.config.min_pod, new_count_svc), self.config.max_pod)
            # 向上取整
            new_count_svc = math.ceil(new_count_svc)

            # 先不急着生效，先看看当前情况有没有过往经验
            qps_svc=(qps_svc//5)*5
            if qps_svc in self.knowledge[svc].keys():
                
                # new_count_svc=self.knowledge[svc][qps_svc]
                knowledge_count_svc=self.knowledge[svc][qps_svc]
                # 如果要学习的话，顺便学习一下，往这个经验靠近
                if self.no_learn:
                    break
                # 实际给出的隶属度
                y_pred = Y_svc
                # 根据经验的隶属度
                y_true = self.analyse(count_svc, knowledge_count_svc)
                controller[0].learn(y_pred, y_true, p_svc)
                controller[1].learn(y_pred, y_true, d_svc)
                
            self.k8s_util.patch_scale(svc, new_count_svc)

    def start(self):
        print("Spring go...")
        self.scale()
        schedule.every(self.scale_interval).seconds.do(self.scale)

        time_start = time.time()  # 开始计时
        while True:
            time_c = time.time() - time_start
            if time_c > self.config.duration:
                schedule.clear()
                break
            schedule.run_pending()

        print("Spring stop...")
