"""조수 하루의 3차원 실험실 EP.03 — 필터를 미끄러뜨리는 망 (CNN)

질문: 왼쪽에서만 본 고양이를 오른쪽으로 옮겨도 알아볼까?
  - 16x16 회색 그림(0~255) 고양이·개·여우 얼굴을 코드로 만든다.
  - 학습 그림은 얼굴을 왼쪽(dx 0~2)에만 두고, 시험 그림은 오른쪽(dx 4~5)으로 옮긴다.
  - 같은 데이터로 일반 망(256-32-3)과 CNN(3x3 필터 4개 → 3x3 필터 8개 → 최댓값 → 3)을 학습한다.
저장소 맨 위 폴더에서:  python3 tools/experiment.py   → data/cnn_run.json
"""
import numpy as np, json

SEED = 7
g = np.random.default_rng(SEED)
CLS = ["cat", "dog", "fox"]


# ---------------------------------------------------------------- 그림 만들기
def face(kind, r):
    """10x10 얼굴 하나를 16x16 바탕 왼쪽 위에 그린다(0~255)."""
    f = np.zeros((16, 16))
    yy, xx = np.mgrid[0:16, 0:16]
    tone = r.uniform(112, 140)
    if kind == "cat":                                   # 둥근 머리 + 뾰족한 귀 + 수염
        f[((xx - 5) / 4.6) ** 2 + ((yy - 6.2) / 3.6) ** 2 <= 1] = tone
        e = r.integers(0, 2)
        for cx in (2, 8):                               # 위로 솟은 세모 귀
            for k in range(3 + e):
                f[3 - k + (1 - e), cx - (2 + e - k) // 2: cx + (2 + e - k) // 2 + 1] = tone + 20
            f[1 + (1 - e), cx] = 215
        for cx in (3, 7):
            f[5:7, cx] = 250; f[6, cx] = 25             # 눈
        f[7, 4:7] = 175                                 # 코·입
        for y in (7, 8):                                # 머리 밖으로 뻗은 수염
            f[y, 0] = f[y, 10] = 250
    elif kind == "dog":                                 # 둥근 머리 + 처진 귀 + 큰 코
        f[((xx - 5) / 3.8) ** 2 + ((yy - 5.6) / 4.0) ** 2 <= 1] = tone
        L = 4 + r.integers(0, 2)
        for cx in (0, 1, 9, 10):
            f[2:2 + L, cx] = 70                         # 양옆으로 처진 귀
        for cx in (3, 7):
            f[4:6, cx] = 250; f[5, cx] = 25
        f[7:9, 4:7] = 30; f[7, 5] = 90                  # 큰 검은 코
    else:                                               # fox: 위가 넓고 아래가 좁은 얼굴
        for y in range(2, 11):
            w = max(0, int(round(5 - (y - 2) * 0.55)))
            f[y, 5 - w:5 + w + 1] = tone
            if 5 <= y <= 8:
                f[y, 5 - w:5 - w + 2] = f[y, 5 + w - 1:5 + w + 1] = 205   # 흰 뺨
        e = r.integers(0, 2)
        for cx in (1, 9):                               # 크고 높은 귀
            for k in range(3 + e):
                f[2 + e - k, cx - (2 + e - k) // 2: cx + (2 + e - k) // 2 + 1] = tone - 15
        for cx in (3, 7):
            f[4, cx] = 250; f[4:6, cx] = 250; f[5, cx] = 25
        f[10, 5] = 25                                   # 작은 코
    return f


def make(n, dxs, r):
    X, y, pos = [], [], []
    for i in range(n):
        c = i % 3
        f = face(CLS[c], r)
        dx, dy = int(r.choice(dxs)), int(r.integers(0, 4))
        img = np.roll(np.roll(f, dy, 0), dx, 1) * r.uniform(.85, 1.1)
        img = np.clip(img + r.normal(0, 14, (16, 16)), 0, 255)
        X.append(img); y.append(c); pos.append(dx)
    return np.array(X, dtype=np.float32), np.array(y), np.array(pos)


Xtr, ytr, _ = make(1800, [0, 1, 2], g)          # 학습: 왼쪽
Xsame, ysame, _ = make(900, [0, 1, 2], g)       # 시험 1: 같은 왼쪽, 처음 보는 그림
Xmove, ymove, _ = make(900, [4, 5], g)          # 시험 2: 오른쪽으로 옮긴 그림
norm = lambda X: X / 255.0


# ---------------------------------------------------------------- 망 두 개
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def patches(F):
    """F (N,H,W,C) → 3x3 이웃 묶음 (N,H,W,9C), 바깥은 0으로 채움."""
    N, H, W, C = F.shape
    P = np.pad(F, ((0, 0), (1, 1), (1, 1), (0, 0)))
    return np.concatenate([P[:, u:u + H, v:v + W, :] for u in range(3) for v in range(3)], -1)


def unpatch(dP, C):
    N, H, W, _ = dP.shape
    dF = np.zeros((N, H + 2, W + 2, C), np.float32)
    k = 0
    for u in range(3):
        for v in range(3):
            dF[:, u:u + H, v:v + W, :] += dP[..., k * C:(k + 1) * C]; k += 1
    return dF[:, 1:-1, 1:-1, :]


class CNN:
    def __init__(s, r):
        s.K1 = r.normal(0, .5, (9, 4)).astype(np.float32); s.b1 = np.zeros(4, np.float32)
        s.K2 = r.normal(0, .25, (36, 8)).astype(np.float32); s.b2 = np.zeros(8, np.float32)
        s.V = r.normal(0, .4, (8, 3)).astype(np.float32); s.c = np.zeros(3, np.float32)
        s.params = ["K1", "b1", "K2", "b2", "V", "c"]

    def forward(s, X):
        I = norm(X)[..., None]
        s.P1 = patches(I); s.Z1 = s.P1 @ s.K1 + s.b1; F1 = np.maximum(s.Z1, 0)            # 16x16x4
        N = F1.shape[0]; G = F1.reshape(N, 8, 2, 8, 2, 4)
        s.F1, s.M1 = F1, G.max((2, 4))                                                     # 8x8x4 (2x2 최댓값)
        s.arg1 = (G == s.M1[:, :, None, :, None, :])
        s.P2 = patches(s.M1); s.Z2 = s.P2 @ s.K2 + s.b2; F2 = np.maximum(s.Z2, 0)         # 8x8x8
        s.F2 = F2; s.h = F2.reshape(N, 64, 8).max(1)                                       # 그림 전체 최댓값
        s.idx = F2.reshape(N, 64, 8).argmax(1)
        return softmax(s.h @ s.V + s.c)

    def backward(s, p, y):
        N = len(y); d = p.copy(); d[np.arange(N), y] -= 1; d /= N
        gr = {"V": s.h.T @ d, "c": d.sum(0)}
        dh = d @ s.V.T
        dF2 = np.zeros((N, 64, 8), np.float32); np.put_along_axis(dF2, s.idx[:, None, :], dh[:, None, :], 1)
        dZ2 = dF2.reshape(N, 8, 8, 8) * (s.Z2 > 0)
        gr["K2"] = s.P2.reshape(-1, 36).T @ dZ2.reshape(-1, 8); gr["b2"] = dZ2.sum((0, 1, 2))
        dM1 = unpatch(dZ2 @ s.K2.T, 4)
        dF1 = (s.arg1 * dM1[:, :, None, :, None, :]).reshape(N, 16, 16, 4)
        dZ1 = dF1 * (s.Z1 > 0)
        gr["K1"] = s.P1.reshape(-1, 9).T @ dZ1.reshape(-1, 4); gr["b1"] = dZ1.sum((0, 1, 2))
        return gr


class MLP:
    def __init__(s, r):
        s.A = r.normal(0, .08, (256, 32)).astype(np.float32); s.a = np.zeros(32, np.float32)
        s.V = r.normal(0, .25, (32, 3)).astype(np.float32); s.c = np.zeros(3, np.float32)
        s.params = ["A", "a", "V", "c"]

    def forward(s, X):
        s.x = norm(X).reshape(len(X), 256); s.z = s.x @ s.A + s.a; s.h = np.maximum(s.z, 0)
        return softmax(s.h @ s.V + s.c)

    def backward(s, p, y):
        N = len(y); d = p.copy(); d[np.arange(N), y] -= 1; d /= N
        dz = (d @ s.V.T) * (s.z > 0)
        return {"V": s.h.T @ d, "c": d.sum(0), "A": s.x.T @ dz, "a": dz.sum(0)}


def acc(m, X, y):
    return float((m.forward(X).argmax(1) == y).mean())


def train(m, epochs=40, bs=60, lr=.01):
    mom = {k: np.zeros_like(getattr(m, k)) for k in m.params}
    log = {"loss": [], "same": [], "move": [], "filters": []}
    for ep in range(epochs):
        order = g.permutation(len(ytr)); L = 0
        for i in range(0, len(order), bs):
            b = order[i:i + bs]; p = m.forward(Xtr[b])
            L += -np.log(p[np.arange(len(b)), ytr[b]] + 1e-9).sum()
            gr = m.backward(p, ytr[b])
            for k in m.params:                          # 모멘텀 0.9 경사하강법
                mom[k] = .9 * mom[k] - lr * gr[k]; setattr(m, k, getattr(m, k) + mom[k])
        log["loss"].append(round(float(L) / len(ytr), 4))
        log["same"].append(round(acc(m, Xsame, ysame), 4))
        log["move"].append(round(acc(m, Xmove, ymove), 4))
        if isinstance(m, CNN): log["filters"].append(m.K1.T.astype(float).round(3).tolist())
    return log


cnn, mlp = CNN(g), MLP(g)
init_K1 = cnn.K1.T.astype(float).round(3).tolist()      # 학습 전 필터 (무작위)
logC = train(cnn, lr=.05)
logM = train(mlp, lr=.05)

# ---------------------------------------------------------------- 결과
r2 = np.random.default_rng(SEED + 100)
hero = face("cat", r2)                                   # 무대에 올릴 고양이 한 마리 (잡음 없이)
slide = np.array([np.roll(np.roll(hero, 3, 0), dx, 1) for dx in range(6)], np.float32)
pc, pm = cnn.forward(slide).astype(float), mlp.forward(slide).astype(float)
samples = {k: np.roll(np.roll(face(k, r2), 3, 0), 1, 1) for k in CLS}
S = np.array([samples[k] for k in CLS] + [slide[4]], np.float32)

nC1, nC = 9 * 4 + 4, 9 * 4 + 4 + 36 * 8 + 8 + 8 * 3 + 3
nM = 256 * 32 + 32 + 32 * 3 + 3
madds = {"cnn": 16 * 16 * 9 * 4 + 8 * 8 * 36 * 8 + 8 * 3, "mlp": 256 * 32 + 32 * 3}
out = {
    "seed": SEED, "classes": CLS,
    "hero": hero.round().astype(int).tolist(),
    "samples": S.round().astype(int).tolist(),
    "train_examples": Xtr[:9].round().astype(int).tolist(), "train_labels": ytr[:9].tolist(),
    "move_examples": Xmove[:9].round().astype(int).tolist(), "move_labels": ymove[:9].tolist(),
    "slide": {"dx": list(range(6)), "cnn_cat": pc[:, 0].round(4).tolist(), "mlp_cat": pm[:, 0].round(4).tolist(),
              "cnn_p": pc.round(4).tolist(), "mlp_p": pm.round(4).tolist()},
    "cnn": {"log": logC, "init_K1": init_K1, "K1": cnn.K1.T.astype(float).round(4).tolist(), "b1": cnn.b1.astype(float).round(4).tolist(),
            "K2": cnn.K2.T.astype(float).round(4).tolist(), "b2": cnn.b2.astype(float).round(4).tolist(),
            "V": cnn.V.astype(float).round(4).tolist(), "c": cnn.c.astype(float).round(4).tolist(), "params": nC},
    "mlp": {"log": {k: v for k, v in logM.items() if v}, "params": nM},
    "madds": madds,
    "cover_train": (Xtr > 90).mean(0).round(3).astype(float).tolist(),   # 학습 그림에서 그 픽셀이 밝았던 비율
    "cover_move": (Xmove > 90).mean(0).round(3).astype(float).tolist(),
    "final": {"cnn_same": logC["same"][-1], "cnn_move": logC["move"][-1],
              "mlp_same": logM["same"][-1], "mlp_move": logM["move"][-1]},
}
print("CNN  same %.4f  move %.4f  params %d" % (logC["same"][-1], logC["move"][-1], nC))
print("MLP  same %.4f  move %.4f  params %d" % (logM["same"][-1], logM["move"][-1], nM))
print("hero cat p(cat) by dx  CNN", pc[:, 0].round(3), " MLP", pm[:, 0].round(3))
print("multiply-adds", madds)
json.dump(out, open("data/cnn_run.json", "w"), separators=(",", ":"))
