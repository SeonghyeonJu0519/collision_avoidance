"""
A2C + GAE on Pendulum-v1  (continuous [-1,1])
저장: returns_pendulum.npy, actor_pendulum.pth
"""
import gymnasium as gym, torch, torch.nn as nn, numpy as np, math, collections

GAMMA = 0.99; LR_A = 3e-4; LR_C = 1e-3; EPISODES = 600; λ = 0.95
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCont(nn.Module):                 # μ, logσ 출력
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(3,256), nn.Tanh(),
                                nn.Linear(256,256), nn.Tanh())
        self.mu_head     = nn.Linear(256,1)
        self.log_std     = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        h  = self.fc(x)
        mu = torch.tanh(self.mu_head(h))    # 액션 범위 [-1,1]
        std= self.log_std.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = nn.functional.tanh(self.fc1(x))
        x = nn.functional.tanh(self.fc2(x))
        return self.fc3(x).squeeze(-1)

actor, critic = ActorCont().to(device), Critic().to(device)
opt_a, opt_c = torch.optim.Adam(actor.parameters(), LR_A), torch.optim.Adam(critic.parameters(), LR_C)

# ── 학습 루프 ──────────────────────────────────────────────
env = gym.make("Pendulum-v1"); R_hist=[]
for ep in range(EPISODES):
    s,_ = env.reset(); done=False; buf=collections.deque(); ep_ret=0
    while not done:
        s_t = torch.tensor(s, dtype=torch.float32, device=device)
        dist = actor(s_t)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        v = critic(s_t)
        action = a.clamp(-2, 2)  # Pendulum 액션 범위
        s2,r,term,trunc,_ = env.step(action.cpu().numpy())
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
    if (ep+1)%10==0: print(f"Ep {ep+1:3d} | Ret {ep_ret:6.1f}")

np.save("returns_pendulum.npy", R_hist)
torch.save(actor.state_dict(), "actor_pendulum.pth")
torch.save(critic.state_dict(), "critic_pendulum.pth")
