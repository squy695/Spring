import math


class FuzzyController:
    def __init__(self):
        self.nl = -3 / 4
        self.nm = -2 / 4
        self.ns = -1 / 4
        self.zr = 0
        self.ps = 1 / 4
        self.pm = 2 / 4
        self.pl = 3 / 4
        self.alpha = 0.1
        self.beta = 0.11
        self.study_rate = 0.01

        self.bias_neg = 0
        self.bias_pos = 0.4
        self.weight = [
            -0.8 + self.bias_neg,
            -0.5 + self.bias_neg,
            -0.2 + self.bias_neg,
            0,
            0.2 + self.bias_pos,
            0.5 + self.bias_pos,
            0.8 + self.bias_pos,
        ]

    def params(self):
        print(self.nl, self.nm, self.ns, self.zr, self.ps, self.pm, self.pl)

    # 归一化隶属向量v
    def one(self, v):
        all = sum(v)
        return [i / all for i in v]

    # 给两个坐标a,b，返回x在a和b确定的函数下的y值
    def line(self, a, b, x):
        return (a[1] - b[1]) * (x - b[0]) / (a[0] - b[0]) + b[1]

    # 将[-1,1]的输入转换成隶属度
    def fuzzize(self, x):
        v = [0 for i in range(7)]

        if x <= self.nl * (1 - self.alpha):
            v[0] = 1
        elif self.nl * (1 - self.alpha) < x and x <= self.nm * (1 + self.alpha):
            v[0] = self.line([self.nl, 1], [self.nm * (1 + self.alpha), 0], x)
            v[1] = self.line([self.nl * (1 - self.alpha), 0], [self.nm, 1], x)
        elif self.nm * (1 + self.alpha) < x and x <= self.nm * (1 - self.alpha):
            v[1] = 1
        elif self.nm * (1 - self.alpha) < x and x <= self.ns * (1 + self.alpha):
            v[1] = self.line([self.nm, 1], [self.ns * (1 + self.alpha), 0], x)
            v[2] = self.line([self.nm * (1 - self.alpha), 0], [self.ns, 1], x)
        elif self.ns * (1 + self.alpha) < x and x <= self.ns * (1 - self.alpha):
            v[2] = 1
        elif self.ns * (1 - self.alpha) < x and x <= -self.alpha:
            v[2] = self.line([self.ns, 1], [-self.alpha, 0], x)
            v[3] = self.line([self.ns * (1 - self.alpha), 0], [self.zr, 1], x)
        elif -self.alpha < x and x <= self.alpha:
            v[3] = 1
        elif self.alpha < x and x <= self.ps * (1 - self.alpha):
            v[3] = self.line([self.zr, 1], [self.ps * (1 - self.alpha), 0], x)
            v[4] = self.line([self.alpha, 0], [self.ps, 1], x)
        elif self.ps * (1 - self.alpha) < x and x <= self.ps * (1 + self.alpha):
            v[4] = 1
        elif self.ps * (1 + self.alpha) < x and x <= self.pm * (1 - self.alpha):
            v[4] = self.line([self.ps, 1], [self.pm * (1 - self.alpha), 0], x)
            v[5] = self.line([self.ps * (1 + self.alpha), 0], [self.pm, 1], x)
        elif self.pm * (1 - self.alpha) < x and x <= self.pm * (1 + self.alpha):
            v[5] = 1
        elif self.pm * (1 + self.alpha) < x and x <= self.pl * (1 - self.alpha):
            v[5] = self.line([self.pm, 1], [self.pl * (1 - self.alpha), 0], x)
            v[6] = self.line([self.pm * (1 + self.alpha), 0], [self.pl, 1], x)
        
        elif self.pl * (1 - self.alpha) < x:
            v[6] = max(
                1,
                x / (self.pl - self.pm * (1 + self.alpha))
                - (self.pm * (1 + self.alpha)) / (self.pl - self.pm * (1 + self.alpha)),
            )
            #return v

        return self.one(v)

    # 推理，X=[x1,x2]，Y=[1,2,3,4,5,6,7]
    def inference(self, X):
        Y=[0, 0, 0, 0, 0, 0, 0]

        # 如果P非常大，超过了PL，也就是X[0][6]>=1，那么直接输出最大的Y
        if X[0][6] >= 1:
            Y[6]=X[0][6]
            return Y

        for i,x0 in enumerate(X[0]):
            if x0 == 0 :
                continue
            for j,x1 in enumerate(X[1]):
                if x1 == 0:
                    continue
                Level=math.ceil((i+j)/2)
                Y[Level]=(x0+x1)/2

        return self.one(Y)

    # 加权面积反模糊化
    def unfuzzize(self, Y):
        U = 0
        for i in range(7):
            U += Y[i] * self.weight[i]
        return U

    # 指出一条线和偏差，返回该线的斜率
    def line_adjust(self, a, b, dy, x):
        # 对a[0]的斜率
        da = (2 * dy * (a[1] - b[1]) * (b[0] - x)) / (7 * (a[0] - b[0]) * (a[0] - b[0]))
        # 对b[0]的斜率
        db = (2 * dy * (a[1] - b[1]) * (x - a[0])) / (7 * (a[0] - b[0]) * (a[0] - b[0]))

        return da, db

    # 根据理想隶属度和实际隶属度，调整level，pred_y必须归一化
    def learn(self, pred_y, true_y, x):

        dy = [pred_y[i] - true_y[i] for i in range(7)]

        # 正负都有偏差，说明经验是错误的
        if sum(dy[0:3]) !=0 and sum(dy[4:]) !=0:
            return

        # n 三个偏差
        if dy[0] !=0 and dy[1] !=0 and dy[2] !=0:
            # 如果是+ -，则集体向左
            if dy[0] > 0 and dy[2] < 0:
                time = 1
                nl = self.nl - self.study_rate * time
                # 判断新的调整方案是否符合顺序
                count = 0
                # 先更新nl
                while not (
                    nl >= -1
                    and nl <= self.nm * (1 + self.beta)
                ):
                    if count == 5:
                        time = 0
                    count += 1
                    time /= 2

                    nl = self.nl - self.study_rate * time
                # 符合顺序，更新参数
                self.nl = nl

                # 再更新nm
                time = 1
                nm = self.nm - self.study_rate * time
                while not (
                    nm >= self.nl * (1 - self.beta)
                    and nm <= self.ns * (1 + self.beta)
                ):
                    if count == 5:
                        time = 0
                    count += 1
                    time /= 2

                    nm = self.nm - self.study_rate * time
                # 符合顺序，更新参数
                self.nm = nm

                # ns
                time = 1
                ns = self.ns - self.study_rate * time
                while not (
                    ns >= self.nm * (1 - self.beta)
                    and ns <= -self.beta
                ):
                    if count == 5:
                        time = 0
                    count += 1
                    time /= 2

                    ns = self.ns - self.study_rate * time
                # 符合顺序，更新参数
                self.ns = ns

            # 如果是- +，则集体向右
            elif dy[0] < 0 and dy[2] > 0:
                time = 1
                ns = self.ns + self.study_rate * time
                # 判断新的调整方案是否符合顺序
                count = 0
                # ns
                while not (
                    ns >= self.nm * (1 - self.beta)
                    and ns <= -self.beta
                ):
                    if count == 5:
                        time = 0
                    count += 1
                    time /= 2

                    ns = self.ns + self.study_rate * time
                # 符合顺序，更新参数
                self.ns = ns

                time = 1
                nm = self.nm + self.study_rate * time
                # 判断新的调整方案是否符合顺序
                count = 0
                # nm
                while not (
                    nm >= self.nl * (1 - self.beta)
                    and nm <= self.ns * (1 + self.beta)
                ):
                    if count == 5:
                        time = 0
                    count += 1
                    time /= 2

                    nm = self.nm + self.study_rate * time
                # 符合顺序，更新参数
                self.nm = nm

                time = 1
                nl = self.nl + self.study_rate * time
                # 判断新的调整方案是否符合顺序
                count = 0
                # nl
                while not (
                    nl >= -1
                    and nl <= self.nm * (1 + self.beta)
                ):
                    if count == 5:
                        time = 0
                    count += 1
                    time /= 2

                    nl = self.nl + self.study_rate * time
                # 符合顺序，更新参数
                self.nl = nl

        # p 三个偏差
        elif dy[4] !=0 and dy[5] !=0 and dy[6] !=0:
            # 如果是+ -，则集体向左
            if dy[4] > 0 and dy[6] < 0:
                time = 1
                ps = self.ps - self.study_rate * time
                # 判断新的调整方案是否符合顺序
                # ps
                count = 0
                while not (
                    ps <= self.pm * (1 - self.beta)
                    and ps >= self.beta
                ):
                    if count == 5:
                        time = 0
                    count += 1
                    time /= 2

                    ps = self.ps - self.study_rate * time
                # 符合顺序，更新参数
                self.ps = ps

                time = 1
                pm = self.pm - self.study_rate * time
                # 判断新的调整方案是否符合顺序
                # pm
                count = 0
                while not (
                    pm <= self.pl * (1 - self.beta)
                    and pm >= self.ps * (1 + self.beta)
                ):
                    if count == 5:
                        time = 0
                    count += 1
                    time /= 2

                    pm = self.pm - self.study_rate * time
                # 符合顺序，更新参数
                self.pm = pm

                time = 1
                pl = self.pl - self.study_rate * time
                # 判断新的调整方案是否符合顺序
                # pl
                count = 0
                while not (
                    pl <= 1
                    and pl >= self.pm * (1 + self.beta)
                ):
                    if count == 5:
                        time = 0
                    count += 1
                    time /= 2

                    pl = self.pl - self.study_rate * time
                # 符合顺序，更新参数
                self.pl = pl

            # 如果是- +，则集体向右
            elif dy[4] < 0 and dy[6] > 0:
                time = 1
                pl = self.pl + self.study_rate * time
                pm = self.pm + self.study_rate * time
                ps = self.ps + self.study_rate * time
                # 判断新的调整方案是否符合顺序
                count = 0
                while not (
                    pl <= 1
                    and pl >= pm * (1 + self.beta)
                    and pm <= pl * (1 - self.beta)
                    and pm >= ps * (1 + self.beta)
                    and ps <= pm * (1 - self.beta)
                    and ps >= self.beta
                ):
                    if count == 5:
                        time = 0
                    count += 1
                    time /= 2
                    
                    pl = self.pl + self.study_rate * time
                    pm = self.pm + self.study_rate * time
                    ps = self.ps + self.study_rate * time
                # 符合顺序，更新参数
                self.pl, self.pm, self.ps = pl, pm, ps

        # 根据dy中，两个连续的非零误差，来判断是哪两条线。单独的一个非零误差是不可能存在的
        elif dy[0] != 0 and dy[1] != 0:
            da, db = self.line_adjust(
                [self.nl, 1], [self.nm * (1 + self.alpha), 0], dy[0], x
            )
            _da, _db = da, db

            nl = self.nl - self.study_rate * da - self.study_rate * db
            nm = self.nm - self.study_rate * da - self.study_rate * db
            count = 0
            # 判断新的调整方案是否符合顺序 nl
            while not (
                nl >= -1
                and nl <= self.nm * (1 + self.beta)
            ):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1

                nl = self.nl - self.study_rate * da - self.study_rate * db
            # 符合顺序，更新参数
            self.nl = nl

            count = 0
            # 判断新的调整方案是否符合顺序 nm
            while not (
                nm >= self.nl * (1 - self.beta)
                and nm <= self.ns * (1 + self.beta)
            ):
                _da, _db = 0.5 * _da, 0.5 * _db
                if count == 5:
                    _da, _db = 0, 0
                count += 1

                nm = self.nm - self.study_rate * _da - self.study_rate * _db
            # 符合顺序，更新参数
            self.nm = nm

            da, db = self.line_adjust(
                [self.nl * (1 - self.alpha), 0], [self.nm, 1], dy[1], x
            )
            _da, _db = da, db

            nl = self.nl - self.study_rate * da - self.study_rate * db
            nm = self.nm - self.study_rate * da - self.study_rate * db
            count = 0
            # 判断新的调整方案是否符合顺序 nl
            while not (
                nl >= -1
                and nl <= self.nm * (1 + self.beta)
            ):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1

                nl = self.nl - self.study_rate * da - self.study_rate * db
            # 符合顺序，更新参数
            self.nl = nl

            count = 0
            # 判断新的调整方案是否符合顺序 nm
            while not (
                nm >= self.nl * (1 - self.beta)
                and nm <= self.ns * (1 + self.beta)
            ):
                _da, _db = 0.5 * _da, 0.5 * _db
                if count == 5:
                    _da, _db = 0, 0
                count += 1

                nm = self.nm - self.study_rate * _da - self.study_rate * _db
            # 符合顺序，更新参数
            self.nm = nm

        elif dy[1] != 0 and dy[2] != 0:
            da, db = self.line_adjust(
                [self.nm, 1], [self.ns * (1 + self.alpha), 0], dy[1], x
            )
            _da, _db = da, db

            nm = self.nm - self.study_rate * da - self.study_rate * db
            ns = self.ns - self.study_rate * da - self.study_rate * db
            count = 0
            # 判断新的调整方案是否符合顺序 nm
            while not (
                nm >= self.nl * (1 - self.beta)
                and nm <= self.ns * (1 + self.beta)
            ):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1
                nm = self.nm - self.study_rate * da - self.study_rate * db
            # 符合顺序，更新参数
            self.nm = nm

            count = 0
            # 判断新的调整方案是否符合顺序 ns
            while not (
                ns >= self.nm * (1 - self.beta)
                and ns <= -self.beta
            ):
                _da, _db = 0.5 * _da, 0.5 * _db
                if count == 5:
                    _da, _db = 0, 0
                count += 1
                ns = self.ns - self.study_rate * _da - self.study_rate * _db
            # 符合顺序，更新参数
            self.ns = ns

            da, db = self.line_adjust(
                [self.nm * (1 - self.alpha), 0], [self.ns, 1], dy[2], x
            )
            _da, _db = da, db

            nm = self.nm - self.study_rate * da - self.study_rate * db
            ns = self.ns - self.study_rate * da - self.study_rate * db
            count = 0
            # 判断新的调整方案是否符合顺序 nm
            while not (
                nm >= self.nl * (1 - self.beta)
                and nm <= self.ns * (1 + self.beta)
            ):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1
                nm = self.nm - self.study_rate * da - self.study_rate * db
            # 符合顺序，更新参数
            self.nm = nm

            count = 0
            # 判断新的调整方案是否符合顺序 ns
            while not (
                ns >= self.nm * (1 - self.beta)
                and ns <= -self.beta
            ):
                _da, _db = 0.5 * _da, 0.5 * _db
                if count == 5:
                    _da, _db = 0, 0
                count += 1
                ns = self.ns - self.study_rate * _da - self.study_rate * _db
            # 符合顺序，更新参数
            self.ns = ns

        elif dy[2] != 0 and dy[3] != 0:
            da, db = self.line_adjust([self.ns, 1], [-self.alpha, 0], dy[2], x)
            
            ns = self.ns - self.study_rate * da
            count = 0
            # 判断新的调整方案是否符合顺序
            while not (self.nm * (1 - self.beta) <= ns and ns < -self.beta):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1
                ns = self.ns - self.study_rate * da
            # 符合顺序，更新参数
            self.ns = ns

            da, db = self.line_adjust([self.ns * (1 - self.alpha), 0], [0, 1], dy[3], x)
            
            ns = self.ns - self.study_rate * da
            count = 0
            # 判断新的调整方案是否符合顺序
            while not (self.nm * (1 - self.beta) <= ns and ns < -self.beta):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1
                ns = self.ns - self.study_rate * da
            # 符合顺序，更新参数
            self.ns = ns

        elif dy[3] != 0 and dy[4] != 0:
            da, db = self.line_adjust([0, 1], [self.ps * (1 - self.alpha), 0], dy[3], x)
            ps = self.ps - self.study_rate * db
            count = 0
            # 判断新的调整方案是否符合顺序
            while not (self.beta <= ps and ps <= self.pm * (1 - self.beta)):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1
                ps = self.ps - self.study_rate * db
            # 符合顺序，更新参数
            self.ps = ps

            da, db = self.line_adjust([self.alpha, 0], [self.ps, 1], dy[4], x)
            ps = self.ps - self.study_rate * db
            count = 0
            # 判断新的调整方案是否符合顺序
            while not (self.beta <= ps and ps <= self.pm * (1 - self.beta)):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1
                ps = self.ps - self.study_rate * db
            # 符合顺序，更新参数
            self.ps = ps

        elif dy[4] != 0 and dy[5] != 0:
            da, db = self.line_adjust(
                [self.ps, 1], [self.pm * (1 - self.alpha), 0], dy[4], x
            )
            _da, _db = da, db

            ps = self.ps - self.study_rate * da - self.study_rate * db
            pm = self.pm - self.study_rate * da - self.study_rate * db
            count = 0
            # 判断新的调整方案是否符合顺序 ps
            while not (
                ps >= self.beta
                and ps <= self.pm * (1 - self.beta)
            ):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1
                ps = self.ps - self.study_rate * da - self.study_rate * db
            # 符合顺序，更新参数
            self.ps = ps

            count = 0
            # 判断新的调整方案是否符合顺序 pm
            while not (
                pm >= self.ps * (1 + self.beta)
                and pm <= self.pl * (1 - self.beta)
            ):
                _da, _db = 0.5 * _da, 0.5 * _db
                if count == 5:
                    _da, _db = 0, 0
                count += 1
                pm = self.pm - self.study_rate * _da - self.study_rate * _db
            # 符合顺序，更新参数
            self.pm = pm

            da, db = self.line_adjust(
                [self.ps * (1 + self.alpha), 0], [self.pm, 1], dy[5], x
            )
            _da, _db = da, db

            ps = self.ps - self.study_rate * da - self.study_rate * db
            pm = self.pm - self.study_rate * da - self.study_rate * db
            count = 0
            # 判断新的调整方案是否符合顺序 ps
            while not (
                ps >= self.beta
                and ps <= self.pm * (1 - self.beta)
            ):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1
                ps = self.ps - self.study_rate * da - self.study_rate * db
            # 符合顺序，更新参数
            self.ps = ps

            count = 0
            # 判断新的调整方案是否符合顺序 pm
            while not (
                pm >= self.ps * (1 + self.beta)
                and pm <= self.pl * (1 - self.beta)
            ):
                _da, _db = 0.5 * _da, 0.5 * _db
                if count == 5:
                    _da, _db = 0, 0
                count += 1
                pm = self.pm - self.study_rate * _da - self.study_rate * _db
            # 符合顺序，更新参数
            self.pm = pm

        elif dy[5] != 0 and dy[6] != 0:
            da, db = self.line_adjust(
                [self.pm, 1], [self.pl * (1 - self.alpha), 0], dy[5], x
            )
            _da, _db = da, db
            pm = self.pm - self.study_rate * da - self.study_rate * db
            pl = self.pl - self.study_rate * da - self.study_rate * db
            
            count = 0
            # 判断新的调整方案是否符合顺序 pm
            while not (
                pm >= self.ps * (1 + self.beta)
                and pm <= self.pl * (1 - self.beta)
            ):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1
                pm = self.pm - self.study_rate * da - self.study_rate * db
            # 符合顺序，更新参数
            self.pm = pm

            count = 0
            # 判断新的调整方案是否符合顺序 pl
            while not (
                pl >= self.pm * (1 + self.beta)
                and pl <= 1
            ):
                _da, _db = 0.5 * _da, 0.5 * _db
                if count == 5:
                    _da, _db = 0, 0
                count += 1
                pl = self.pl - self.study_rate * _da - self.study_rate * _db
            # 符合顺序，更新参数
            self.pl = pl

            da, db = self.line_adjust(
                [self.pm * (1 + self.alpha), 0], [self.pl, 1], dy[6], x
            )
            _da, _db = da, db

            pm = self.pm - self.study_rate * da
            pl = self.pl - self.study_rate * db
            count = 0
            # 判断新的调整方案是否符合顺序 pm
            while not (
                pm >= self.ps * (1 + self.beta)
                and pm <= self.pl * (1 - self.beta)
            ):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1
                pm = self.pm - self.study_rate * da - self.study_rate * db
            # 符合顺序，更新参数
            self.pm = pm

            count = 0
            # 判断新的调整方案是否符合顺序 pl
            while not (
                pl >= self.pm * (1 + self.beta)
                and pl <= 1
            ):
                _da, _db = 0.5 * _da, 0.5 * _db
                if count == 5:
                    _da, _db = 0, 0
                count += 1
                pl = self.pl - self.study_rate * _da - self.study_rate * _db
            # 符合顺序，更新参数
            self.pl = pl

        # 是溢出
        elif dy[6] != 0 and sum(dy[0:6]) == 0:
            da, db = self.line_adjust(
                [self.pl, 1], [self.pm * (1 + self.alpha), 0], dy[6], x
            )
            _da, _db = da, db
            pm = self.pm - self.study_rate * da - self.study_rate * db
            pl = self.pl - self.study_rate * da - self.study_rate * db
            
            count = 0
            # 判断新的调整方案是否符合顺序 pm
            while not (
                pm >= self.ps * (1 + self.beta)
                and pm <= self.pl * (1 - self.beta)
            ):
                da, db = 0.5 * da, 0.5 * db
                if count == 5:
                    da, db = 0, 0
                count += 1
                pm = self.pm - self.study_rate * da - self.study_rate * db
            # 符合顺序，更新参数
            self.pm = pm

            count = 0
            # 判断新的调整方案是否符合顺序 pl
            while not (
                pl >= self.pm * (1 + self.beta)
                and pl <= 1
            ):
                _da, _db = 0.5 * _da, 0.5 * _db
                if count == 5:
                    _da, _db = 0, 0
                count += 1
                pl = self.pl - self.study_rate * _da - self.study_rate * _db
            # 符合顺序，更新参数
            self.pl = pl


'''
fuzzy = FuzzyController()

x=0.6
e=0.6

#print(fuzzy.fuzzize(x))
print(fuzzy.inference([fuzzy.fuzzize(x), fuzzy.fuzzize(e)]))
print(fuzzy.unfuzzize(fuzzy.inference([fuzzy.fuzzize(x), fuzzy.fuzzize(e)])))

for i in range(100):
    # nl
    #fuzzy.learn(fuzzy.fuzzize(x), [0.2, 0.8, 0, 0, 0, 0, 0], x)
    # nm
    #fuzzy.learn(fuzzy.fuzzize(x), [0, 0.2, 0.8, 0, 0, 0, 0], x)
    # ns
    #fuzzy.learn(fuzzy.fuzzize(x), [0, 0, 0.2, 0.8, 0, 0, 0], x)
    # ps
    #fuzzy.learn(fuzzy.fuzzize(x), [0, 0, 0, 0.6, 0.4, 0, 0], x)
    # pm
    #fuzzy.learn(fuzzy.fuzzize(x), [0, 0, 0, 0, 0.2, 0.8, 0], x)
    # pl
    #fuzzy.learn(fuzzy.fuzzize(x), [0, 0, 0, 0, 0, 0.8, 0.2], x)

    # n 三个
    #fuzzy.learn(fuzzy.fuzzize(x), [0, 0, 1, 0, 0, 0, 0], x)
    # p 三个
    fuzzy.learn(fuzzy.fuzzize(x), [0, 0, 0, 0, 1, 0, 0], x)

    fuzzy.params()

#print(fuzzy.fuzzize(x))
print(fuzzy.inference([fuzzy.fuzzize(x), fuzzy.fuzzize(e)]))
print(fuzzy.unfuzzize(fuzzy.inference([fuzzy.fuzzize(x), fuzzy.fuzzize(e)])))

'''