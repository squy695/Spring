from cmath import nan
import time
import numpy as np
from util.KubernetesClient import KubernetesClient
from util.PrometheusClient import PrometheusClient
from util.Fuzzy import FuzzyController
from config.Config import Config


class Node:

    def __init__(self, name):
        self.name = name

class Edge:
    def __init__(self, father, son, p90):
        self.father = father
        self.son = son
        self.p90 = p90


# 根据当前的拓扑网络，分辨出真正需要扩展的服务和不需要扩展的服务
class Macro:
    def __init__(self) -> None:
        self.config = Config()
        self.k8s_util = KubernetesClient(self.config)
        self.prom_util = PrometheusClient(self.config)

        self.fuzzy_default = FuzzyController()
        self.SLO_target = 300
        self.max_pod = self.config.max_pod
        self.min_pod = self.config.min_pod
        self.mss = self.k8s_util.get_svcs_without_state()
        if "loadgenerator" in self.mss:
            self.mss.remove("loadgenerator")


    def is_abnormal_edge(self, time):
        return time >= self.SLO_target

    # 返回所有异常节点中，只在尾部的异常节点 [name]
    def get_end_error_node(self, error_node, error_edge):
        end = []
        for name, node in error_node.items():
            end.append(name)
        for name, edges in error_edge.items():
            for edge in edges:
                if (
                    edge.father in error_node.keys()
                    and edge.son in error_node.keys()
                    and edge.father in end
                ):
                    end.remove(edge.father)

        return end

    # 返回统计过后的每个点的in qps之和和out qps之和，只需要统计异常点的
    def get_in_out_count_all(self, svc_in_conut, svc_out_count, error_node_name):
        in_count, out_count = {}, {}

        for name in error_node_name:
            in_sum, out_sum = 0, 0
            
            # 没有入边
            if name not in svc_in_conut.keys():
                in_sum = 0
            else:
                for father, qps in svc_in_conut[name].items():
                    in_sum += qps

            # 没有入边
            if name not in svc_out_count.keys():
                out_sum = 0
            else:
                for son, qps in svc_out_count[name].items():
                    out_sum += qps
                    
            in_count[name] = in_sum
            out_count[name] = out_sum

        return in_count, out_count

    def set_time_range(self):
        begin = int(round((time.time() - 60)))
        end = int(round(time.time()))
        self.prom_util.set_time_range(begin, end)

    # 返回真正需要扩展的服务的name
    # 1. 构建异常子图
    # 2. 尾部的异常服务是true
    # 3. p90 / p50 > 10的服务是true
    # 4. detention > 50的服务是true
    def true_error_svcs(self):

        true_error_node_name = []

        # 异常节点 name:Node
        error_node = {}
        # 异常子图 name:[Edges（所有边）]
        error_edge = {}

        self.set_time_range()
        # 单行，map，name_name
        p90_call = self.prom_util.get_call_latency()
        if p90_call == {}:
            return []
        # 多行，df
        # name_name
        p90_call_range = self.prom_util.get_call_p90_latency_range()

        for key, value in p90_call.items():
            if value == nan:
                p90_call[key] = 0
            father = key[: key.find("_")]
            son = key[key.find("_") + 1 :]

            if self.is_abnormal_edge(value):
                error_node[son] = Node(son)
                error_edge[son] = []

            # 异常节点的所有出边入边都要保存并赋权
            #if father in error_node.keys() or son in error_node.keys():
            if son in error_node.keys():
                error_edge[son].append(Edge(father, son, p90_call_range[key].values))

        # 1 尾部异常服务
        true_error_node_name += (self.get_end_error_node(error_node, error_edge))

        # 2 请求滞留率
        detention = {}
        svcs_in_count = self.prom_util.get_svc_qps_range()
        svcs_out_count = self.prom_util.get_svc_qps_range_source()

        for name in error_node.keys():
            # 节点的滞留率
            if name not in true_error_node_name:
                detention[name] = max(0, (svcs_in_count[name].mean() - svcs_out_count[name].mean()))
                if detention[name] > 10:
                    true_error_node_name.append(name)

        # 3 p90/p50
        svc_p_all = self.prom_util.get_svc_latency()

        p90_p50 = {}
        for name in error_node.keys():
            p90_p50[name] = svc_p_all[name+'&p90'] / svc_p_all[name+'&p50']
            if p90_p50[name] > 5 and name not in true_error_node_name:
                true_error_node_name.append(name)
        
        return true_error_node_name
