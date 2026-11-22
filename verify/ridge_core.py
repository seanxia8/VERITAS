"""Ridge with 5-fold run-grouped CV: verbatim copy of the k1_readout.py implementation
(sealed at b755b23), importable without running K1."""
import numpy as np

ALPHAS = np.logspace(-6, 4, 21)


def ridge_path(X, Y, alphas):
    """Coefficients for all alphas via one SVD (X, Y already centred)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y
    return [Vt.T @ ((s / (s * s + a))[:, None] * UtY) for a in alphas]


def fit_predict(Xtr, Ytr, Xte, groups, alphas=ALPHAS):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    ug = np.unique(groups)
    fold_of_group = {g: i % 5 for i, g in enumerate(ug)}
    fold = np.array([fold_of_group[g] for g in groups])
    sse = np.zeros(len(alphas))
    for f in range(5):
        tr, va = fold != f, fold == f
        xm, ym = Ztr[tr].mean(0), Ytr[tr].mean(0)
        for i, B in enumerate(ridge_path(Ztr[tr] - xm, Ytr[tr] - ym, alphas)):
            sse[i] += (((Ztr[va] - xm) @ B + ym - Ytr[va]) ** 2).sum()
    best = int(np.argmin(sse))
    xm, ym = Ztr.mean(0), Ytr.mean(0)
    B = ridge_path(Ztr - xm, Ytr - ym, [alphas[best]])[0]
    return (Zte - xm) @ B + ym, float(alphas[best])


