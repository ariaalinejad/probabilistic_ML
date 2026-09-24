import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import spams
import scipy.sparse as sp


# The BlindEDAA class runs the EDAA algorithm to estimate
# E=endmember spectra (shape (L,p))
# A=abundances (shape (p, N))
# Where L=number of bands, N=number of pixels and p is the number of endmembers


# init params:
# M: how many times you restart the optimization. More=better chance of good solution, but slower.
# T: how many "outer rounds" of updates per try.
# K1: how many update steps for A (abundances) per outer round
# K2: how many update steps for B per outer round.
# AA_init: if true, start from SPAMS AA solution (better start)
# FISTA_steps: how hard SPAMS AA init runs.
# l2_fit: choose whether you score solutions with L2 error or L1 error.
class BlindEDAA:
    def __init__(
        self,
        T=15,  # 50
        K1=5,  # 20
        K2=5,  # 20
        M=5,  # 100
        AA_init=True,
        FISTA_steps=1,
        l2_fit=False,
    ):
        self.T = T
        self.K1 = K1
        self.K2 = K2
        self.M = M
        self.AA_init = AA_init
        self.FISTA_steps = FISTA_steps
        self.l2_fit = l2_fit

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # GPU if available, else CPU
        self.endmembers = []

    def solve(
        self,
        Y,  # (L (bands), N (pixels)) hyperspectral matrix
        p,  # number of endmembers to estimate
        seed=0,
        **kwargs,
    ):
        best_E = None
        best_A = None
        min_max_corrcoef = 10.0

        L, N = Y.shape

        # TODO AA init (3 FISTA steps)
        # logger.debug(f"AA init ({self.AA_init})")
        print(f"[INFO] AA init ({self.AA_init})")

        # ---------- 1) Initialization using SPAMS Archetypal Analysis ----------
        # This provides a good starting guess for A and B (better than random).
        # A_init: (p, N)  abundance-like matrix
        # B_init: (N, p)  coefficients used to build endmembers as E = Y @ B
        _, A_init, B_init = spams.archetypalAnalysis(
            np.asfortranarray(Y, dtype=np.float64),
            p=p,
            Z0=None,
            returnAB=True,
            robust=False,
            epsilon=1e-3,
            randominit=False,
            numThreads=-1,
            stepsAS=0,
            stepsFISTA=self.FISTA_steps,
            computeXtX=True,
        )

        # SPAMS may return sparse matrices; convert to dense numpy arrays
        A_init = sp.csc_matrix.toarray(A_init)
        B_init = sp.csc_matrix.toarray(B_init)

        # logger.debug(f"A init shape => {A_init.shape}")
        # logger.debug(f"B init shape => {B_init.shape}")
        print(f"[INFO] A init shape => {A_init.shape}")
        print(f"[INFO] B init shape => {B_init.shape}")

        # Convert Y to torch tensor for fast math (CPU/GPU)
        Y = torch.Tensor(Y)  # Now all later math uses torch (faster + can use GPU).

        def residual(a, b):  # L2 reconstruction error
            return 0.5 * ((Y - (Y @ b) @ a) ** 2).sum()

        def residual_l1(a, b):  # L1 reconstruction error
            return (Y - (Y @ b) @ a).abs().sum()

        def loss(a, b):
            return residual(a, b)

        def grad_A(a, b):
            YB = Y @ b
            ret = -YB.t() @ (Y - YB @ a)
            return ret

        def grad_B(a, b):
            return -Y.t() @ ((Y - Y @ b @ a) @ a.t())

        def update(a, b):
            return F.softmax(torch.log(a) + b, dim=0)

        def computeLA(a, b):
            YB = Y @ b
            S = torch.linalg.svdvals(YB)
            return S[0] * S[0]

        max_correl = lambda e: np.max(np.corrcoef(e.T) - np.eye(p))

        results = {}

        tic = time.time()

        for m in tqdm(range(self.M)):
            torch.manual_seed(m + seed)
            generator = np.random.RandomState(m + seed)

            with torch.no_grad():

                # Matrix initialization
                # B = F.softmax(0.1 * torch.rand((N, p)), dim=0)
                # A = (1 / p) * torch.ones((p, N))
                if self.AA_init:
                    B = torch.Tensor(np.copy(B_init))
                    A = torch.Tensor(np.copy(A_init))
                else:
                    B = F.softmax(0.1 * torch.rand((N, p)), dim=0)
                    A = (1 / p) * torch.ones((p, N))

                # Send matrices on GPU
                Y = Y.to(self.device)
                A = A.to(self.device)
                B = B.to(self.device)

                # Random Step size factor
                factA = 2 ** generator.randint(-3, 4)

                # Compute step sizes
                self.etaA = factA / computeLA(A, B)
                self.etaB = self.etaA * ((p / N) ** 0.5)

                for ii in range(self.T):
                    for kk in range(self.K1):
                        A = update(A, -self.etaA * grad_A(A, B))

                    for kk in range(self.K2):
                        B = update(B, -self.etaB * grad_B(A, B))

                # fit_m = residual_l1(A, B).item()
                if self.l2_fit:
                    fit_m = loss(A, B).item()
                else:
                    fit_m = residual_l1(A, B).item()
                E = (Y @ B).cpu().numpy()
                A = A.cpu().numpy()
                Xmap = B.t().cpu().numpy()
                Rm = max_correl(E)
                # Store results
                results[m] = {
                    "Rm": Rm,
                    "Em": E,
                    "Am": A,
                    "Bm": Xmap,
                    "fit_m": fit_m,
                    "factA": factA,
                }

        min_fit_l1 = np.min([v["fit_m"] for k, v in results.items()])

        def fit_l1_cutoff(idx, tol=0.05):
            val = results[idx]["fit_m"]
            return (abs(val - min_fit_l1) / abs(val)) < tol

        sorted_indices = sorted(
            filter(fit_l1_cutoff, results),
            key=lambda x: results[x]["Rm"],
        )

        # sorted_indices = sorted(
        #     filter(fit_l1_cutoff, results),
        #     key=lambda x: results[x]["fit_m"],
        # )

        best_result_idx = sorted_indices[0]
        best_result = results[best_result_idx]

        best_E = best_result["Em"]
        best_A = best_result["Am"]
        self.Xmap = best_result["Bm"]

        toc = time.time()
        elapsed_time = round(toc - tic, 2)
        # logger.info(f"{self} took {elapsed_time}s")
        print(f"[INFO] {self.__class__.__name__} finished in {elapsed_time} seconds")

        return best_E, best_A

    def transform(self, Y, E, K=None):
        '''
        Estimate simplex-constrained abundances A for pixels Y (L, N) given a
        fixed set of endmembers E (L, p), using the same non-negative mirror
        descent update used for A inside solve() -- E is held fixed here.
        '''
        if K is None:
            K = self.T * self.K1

        Y = torch.Tensor(Y).to(self.device)
        E = torch.Tensor(E).to(self.device)
        p = E.shape[1]
        N = Y.shape[1]

        with torch.no_grad():
            A = (1 / p) * torch.ones((p, N), device=self.device)

            S = torch.linalg.svdvals(E)
            eta = 1.0 / (S[0] * S[0])

            def grad_A(a):
                return -E.t() @ (Y - E @ a)

            def update(a, g):
                return F.softmax(torch.log(a) + g, dim=0)

            for _ in range(K):
                A = update(A, -eta * grad_A(A))

        return A.cpu().numpy()

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg


if __name__ == "__main__":
    pass
