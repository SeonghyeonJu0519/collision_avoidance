"""
A2C + Generalized Advantage Estimation (GAE-λ) — CartPole-v1
저장:  returns_a2c_gae.npy, actor_a2c_gae.pth, critic_a2c_gae.pth
"""

import gymnasium as gym, torch, torch.nn as nn, numpy as np, collections

GAMMA = 0.99; LR_A = 3e-4; LR_C = 1e-3; EPISODES = 600; λ = 0.95
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 네트워크 정의 ──────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self): 
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return nn.functional.softmax(self.fc3(x), -1)

class Critic(nn.Module):
    def __init__(self): 
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)

actor, critic = Actor().to(device), Critic().to(device)
opt_a, opt_c  = torch.optim.Adam(actor.parameters(), LR_A), torch.optim.Adam(critic.parameters(), LR_C)

# ── 학습 루프 ──────────────────────────────────────────────
env = gym.make("CartPole-v1"); R_hist=[]
for ep in range(EPISODES):
    s,_ = env.reset(); done=False; buf=collections.deque(); ep_ret=0
    while not done:
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        dist = torch.distributions.Categorical(actor(s_t))
        a    = dist.sample()
        logp = dist.log_prob(a)
        v    = critic(s_t)
        s2,r,term,trunc,_ = env.step(a.item())
        buf.append((logp, v, r)); s, done = s2, term or trunc; ep_ret+=r

    # rollout 끝난 뒤
    adv, gae, R_list = [], 0, []
    next_v = 0
    for logp, v, r in reversed(buf):
        delta = r + GAMMA * next_v - v
        gae = delta + GAMMA * λ * gae
        adv.insert(0, gae)
        R_list.insert(0, gae + v)
        next_v = v.item()
    
    loss_a = -(torch.stack([l for l, _, _ in buf]) * torch.tensor(adv, device=device)).mean()
    loss_c = nn.functional.mse_loss(torch.stack([v for _, v, _ in buf]),
                                   torch.tensor(R_list, device=device))
    opt_a.zero_grad(); loss_a.backward(); opt_a.step()
    opt_c.zero_grad(); loss_c.backward(); opt_c.step()

    R_hist.append(ep_ret)
    if (ep+1)%10==0: print(f"Ep {ep+1:3d} | Ret {ep_ret:3.0f}")

np.save("returns_a2c_gae.npy", R_hist)
torch.save(actor.state_dict(), "actor_a2c_gae.pth")
torch.save(critic.state_dict(), "critic_a2c_gae.pth")
