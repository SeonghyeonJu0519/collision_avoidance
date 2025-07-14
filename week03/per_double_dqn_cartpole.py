"""
Prioritized Replay + Double-DQN on CartPole-v1
==============================================
저장 파일
    • returns_per.npy         : 에피소드별 리턴
    • per_ddqn_cartpole.pth   : 학습된 Q-Network 가중치
"""

import random, math, collections, numpy as np, gymnasium as gym
import torch, torch.nn as nn

# ─── 하이퍼파라미터 ─────────────────────────────────────────
ENV_ID          = "CartPole-v1"
BUFFER_CAP      = 20_000
BATCH_SIZE      = 64
GAMMA           = 0.99
LR              = 1e-3
EPISODES        = 600
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 0.995
TAU             = 0.005          # soft-update
ALPHA           = 0.6            # PER 우선순위 지수
BETA_START      = 0.4            # IS-weights 시작값
BETA_FRAMES     = EPISODES * 200 # 총 step 수 대략
PRIOR_EPS       = 1e-6           # 0 방지

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─── 네트워크 정의 ───────────────────────────────────────────
class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
    def forward(self, x): return self.fc(x)

# ─── Prioritized ReplayBuffer ───────────────────────────────
class PERBuffer:
    def __init__(self, cap=BUFFER_CAP):
        self.ptr, self.size, self.cap = 0, 0, cap
        self.s  = np.zeros((cap, 4),  np.float32)
        self.a  = np.zeros((cap, 1),  np.int64)
        self.r  = np.zeros((cap, 1),  np.float32)
        self.s2 = np.zeros((cap, 4),  np.float32)
        self.d  = np.zeros((cap, 1),  np.float32)
        self.p  = np.ones((cap,),     np.float32)  # priority

    def push(self, s,a,r,s2,d):
        idx = self.ptr
        self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx] = s, a, r, s2, d
        self.p[idx] = self.p.max()           # 새 샘플은 최대 우선순위
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch, beta):
        probs = self.p[:self.size] ** ALPHA
        probs /= probs.sum()
        idxs  = np.random.choice(self.size, batch, p=probs)
        # IS-weight
        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max()
        to_t = lambda x, dtype: torch.tensor(x[idxs]).to(device, dtype)
        return (*map(lambda arr: to_t(arr, torch.float32), (self.s,self.a,self.r,self.s2,self.d)),
                torch.tensor(idxs), torch.tensor(weights, dtype=torch.float32, device=device))

    def update_prior(self, idxs, td_errors):
        self.p[idxs.cpu().numpy()] = (td_errors.abs().cpu().numpy() + PRIOR_EPS)

# ─── 학습 준비 ───────────────────────────────────────────────
env = gym.make(ENV_ID)
obs_dim, act_dim = env.observation_space.shape[0], env.action_space.n
policy, target = QNet(obs_dim, act_dim).to(device), QNet(obs_dim, act_dim).to(device)
target.load_state_dict(policy.state_dict())
opt = torch.optim.Adam(policy.parameters(), lr=LR)
buffer = PERBuffer()
returns, eps, beta, frame = [], EPS_START, BETA_START, 0

# ─── 메인 루프 ───────────────────────────────────────────────
for ep in range(EPISODES):
    s,_ = env.reset()
    done, ep_ret = False, 0
    while not done:
        a = env.action_space.sample() if random.random() < eps else \
            policy(torch.tensor(s).to(device)).argmax().item()

        s2,r,term,trunc,_ = env.step(a)
        buffer.push(s,a,r,s2, term or trunc)
        s, done, ep_ret = s2, term or trunc, ep_ret + r
        frame += 1

        # 학습(버퍼 warm-up 후)
        if buffer.size >= 1_000:
            beta = min(1.0, beta + (1.0 - BETA_START) / BETA_FRAMES)
            S,A,R,S2,D, idxs, w = buffer.sample(BATCH_SIZE, beta)
            q = policy(S).gather(1, A.long()).squeeze()

            with torch.no_grad():
                a_star = policy(S2).argmax(1, keepdim=True)
                q_next = target(S2).gather(1, a_star).squeeze()
                tgt    = R.squeeze() + GAMMA * q_next * (1 - D.squeeze())

            td = tgt - q
            loss = (w * td.pow(2)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            buffer.update_prior(idxs, td.detach())

            # soft-update
            for t,w_ in zip(target.parameters(), policy.parameters()):
                t.data.copy_(t.data * (1-TAU) + w_.data * TAU)

    # ε 감쇠 & 로그
    eps = max(EPS_END, eps * EPS_DECAY)
    returns.append(ep_ret)
    if (ep+1) % 10 == 0:
        print(f"Ep {ep+1:3d} | Ret {ep_ret:3.0f} | ε={eps:.3f} | β={beta:.2f}")

# ─── 결과 저장 ───────────────────────────────────────────────
np.save("returns_per.npy", returns)
torch.save(policy.state_dict(), "per_ddqn_cartpole.pth")
print("최근 100 평균:", np.mean(returns[-100:]))
