import os
import numpy as np
import dataclasses
import functools

import pyximport

pyximport.install(language_level=3, setup_args=dict(include_dirs=np.get_include()))
import _C_GGA

np.set_printoptions(linewidth=400)


def _print(*args, **kwargs):
    if os.getenv("DEBUG") is not None:
        print(*args, **kwargs)


@dataclasses.dataclass
class Problem:
    n: int  # num of PCBs
    m: int  # num of machines
    p: int  # num of type of components
    C: np.ndarray  # (m,)
    A: np.ndarray  # (p, n)
    S: float
    s: float

    lambda1: float
    lambda2: float
    lambda3: float

    @functools.cached_property
    def m_eye(self) -> np.ndarray:  # (m, m)
        return np.eye(self.m, dtype=np.int32)

    @functools.cached_property
    def SIM(self) -> np.ndarray:
        A = self.A
        COM = (A[..., None] * A[:, None]).sum(axis=0)
        TOTAL = (A[..., None] + A[:, None] - A[..., None] * A[:, None]).sum(axis=0)
        return COM.astype(np.float32) / TOTAL.astype(np.float32)

    @functools.cached_property
    def SIZE(self) -> np.ndarray:
        return self.A.sum(axis=0)


@dataclasses.dataclass
class Solution:
    problem: Problem

    # x[i, j] == 1 if the i-th PCB is assembled on machine j
    x: np.ndarray  # (n, m)

    # order[i] is the index of the i-th PCB in the overall sequence
    # e.g. if order = [0, 2, 1], PCB 0 on machine 0, PCB 1, 2 on machine 1
    # then PCB 2 is assembled *before* PCB 1
    order: np.ndarray  # (n,)

    @classmethod
    def decode(cls, pr: Problem, gene: np.ndarray) -> "Solution":
        m, n = pr.m, pr.n
        assert gene.shape == (2 * n, )
        x_code = gene[:n]
        x_code = np.trunc(x_code * m).astype(np.int).clip(0, m - 1)
        x = pr.m_eye[x_code]

        order_code = gene[n:]
        order = order_code.argsort().astype(np.int32)
        return Solution(pr, x, order)

    def encode(self) -> np.ndarray:
        pr = self.problem
        gene_x = (self.x.argmax(axis=1).astype(np.float32) + 0.5) / pr.m
        gene_order = np.linspace(0, 1, num=pr.n)[self.order]
        return np.concatenate([gene_x, gene_order], axis=0)

    def summary(self) -> str:
        print("===> A")
        for j in range(self.problem.n):
            print(f"PCB {j}: ", end="")
            print(" ".join(map(str, np.nonzero(self.problem.A[:, j])[0])))
        print("===> C")
        print(self.problem.C)
        print("===> x")
        for t in range(self.problem.m):
            print(f"machine {t}: ", end="")
            order = self.order[self.x[:, t] != 0]
            sorted = np.nonzero(self.x[:, t])[0][order.argsort()]
            print(" ".join(map(str, sorted)))
        print("===> order")
        print(self.order)
        print("===> f")
        for i in range(1, 4):
            print(f"f{i} ", getattr(self, f"f{i}"))

    @functools.cached_property
    def L(self) -> np.ndarray:  # (m, p, n)
        """
        L[t, i, k] == 1 if component i is used at instant k on machine t.
        """
        return _C_GGA.calc_L(self.x, self.problem.A, self.order)

    def _L_purepy(self):
        "Naive pure Python implementation for reference."
        pr = self.problem
        result = np.zeros((pr.m, pr.p, pr.n), dtype=np.int32)
        x = self.x
        A = pr.A
        for t in range(pr.m):
            for j in range(pr.n):
                if x[j, t] == 0: continue
                for i in range(pr.p):
                    if A[i, j] == 1:
                        result[t, i, self.order[j]] = 1
        return result

    @functools.cached_property
    def sw(self) -> np.ndarray:  # (m, n)
        "sw[t, k] is times of loading/unloading at instant k on machine t."
        pr = self.problem
        return _C_GGA.calc_sw(pr.m, pr.n, pr.p, pr.C, self.L)

    def _sw_purepy(self):
        "Naive pure Python implementation for reference."
        pr = self.problem
        L = self.L

        sw = np.zeros((pr.m, pr.n))
        for t in range(pr.m):
            C = pr.C[t]

            slots = np.zeros((pr.p, ))
            next = np.argmax(L[t], axis=-1)
            next[L[t].sum(axis=-1) == 0] = pr.n + 1
            cur_load = 0
            for instant in range(pr.n):
                used = False
                for i in range(pr.p):
                    if L[t, i, instant] == 0: continue
                    used = True
                    if slots[i] == 0:
                        cur_load += 1
                        slots[i] = 1
                        _print(f"[t={t} instant={instant}] LOAD {i}")
                        sw[t, instant] += 1

                    loc = instant + 1
                    while loc < pr.n and L[t, i, loc] == 0:
                        loc += 1
                    next[i] = loc

                if not used: continue
                while cur_load > C:
                    selected = None
                    maxv = 0
                    for i in range(pr.n):
                        if slots[i] == 0: continue
                        if L[t, i, instant] == 1: continue
                        if maxv < next[i]:
                            maxv = next[i]
                            selected = i
                    slots[selected] = 0
                    _print(f"[t={t} instant={instant}] UNLOAD {selected}")
                    cur_load -= 1
                    sw[t, instant] += 1

                while cur_load < C:
                    selected = None
                    minv = pr.n + 1
                    for i in range(pr.n):
                        if slots[i] == 1: continue
                        if minv > next[i]:
                            minv = next[i]
                            selected = i
                    if selected is None:
                        break
                    slots[selected] = 1
                    _print(f"[t={t} instant={instant}] LOAD {selected}")
                    cur_load += 1
                    sw[t, instant] += 1

        return sw

    @functools.cached_property
    def f1(self) -> float:
        SIM = self.problem.SIM
        x = self.x

        fs = x.T @ SIM @ x
        _print(SIM)
        return -np.diag(fs).min()

    @functools.cached_property
    def f2(self) -> float:
        fl = (self.problem.SIZE.reshape(1, -1) @ self.x).astype(np.float32)
        fl_max, fl_min = fl.max(), fl.min()
        _print(fl)
        return (fl_max - fl_min) / fl_max

    @functools.cached_property
    def f3(self) -> float:
        pr = self.problem
        TSs = pr.S * self.x.sum(axis=0) + pr.s * self.sw.sum(axis=1)
        return TSs.max().astype(np.float32)

    @functools.cached_property
    def f(self) -> float:
        pr = self.problem
        return pr.lambda1 * self.f1 + pr.lambda2 * self.f2 + pr.lambda3 * self.f3


if __name__ == '__main__':
    np.random.seed(43)
    A = (np.random.rand(10, 10) > 0.5).astype(np.int32)
    pr = Problem(
        n=10,
        m=3,
        p=10,
        C=np.array([A.sum(axis=0).max() + 2] * 3, dtype=np.int32),
        A=A,
        S=3,
        s=1,
        lambda1=0.2,
        lambda2=0.2,
        lambda3=0.6,
    )
    x = np.zeros((10, 3), dtype=np.int32)
    for i in range(10):
        x[i, np.random.randint(0, 3)] = 1
    order = np.array(list(range(10)), dtype=np.int32)
    sol = Solution(
        problem=pr,
        x=x,
        order=np.random.permutation(order),
    )
    sol.summary()
