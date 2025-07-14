# dqn_cartpole.py  (week02 폴더)
import gymnasium as gym, random, numpy as np, collections, torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt

ENV_ID = "CartPole-v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, cap=20_000):
        self.buf = collections.deque(maxlen=cap)
    def push(self, *exp): self.buf.append(exp)
    def sample(self, batch):
        batch = random.sample(self.buf, batch)
        s,a,r,s2,d = map(np.array, zip(*batch))
        return map(lambda x: torch.tensor(x, dtype=torch.float32, device=device), (s,a,r,s2,d))
    def __len__(self): return len(self.buf)

env = gym.make(ENV_ID)
obs_dim, act_dim = env.observation_space.shape[0], env.action_space.n
policy_net = QNet(obs_dim, act_dim).to(device)
target_net = QNet(obs_dim, act_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())

opt   = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
gamma = 0.99
buffer= ReplayBuffer()
batch_size = 64

eps, eps_min, eps_decay = 1.0, 0.05, 0.995
episodes = 600
returns  = []

for ep in range(episodes):
    s,_ = env.reset(seed=None)
    done, ep_ret = False, 0
    while not done:
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = policy_net(torch.tensor(s).float().to(device)).argmax().item()
        s2, r, term, trunc, _ = env.step(a)
        buffer.push(s, a, r, s2, term or trunc)
        s, done, ep_ret = s2, term or trunc, ep_ret + r

        if len(buffer) >= 1_000:
            S,A,R,S2,D = buffer.sample(batch_size)
            q   = policy_net(S).gather(1, A.long().unsqueeze(1)).squeeze()
            with torch.no_grad():
                q_next = target_net(S2).max(1)[0]
                target = R + gamma * q_next * (1 - D)
            loss = nn.functional.mse_loss(q, target)
            opt.zero_grad(); loss.backward(); opt.step()

        # soft update
        tau = 0.005
        for tgt, src in zip(target_net.parameters(), policy_net.parameters()):
            tgt.data.copy_(tgt.data * (1 - tau) + src.data * tau)

    eps = max(eps_min, eps * eps_decay)
    returns.append(ep_ret)
    if (ep+1) % 10 == 0:
        print(f"Ep {ep+1:4d} | Return: {ep_ret:3.0f} | ε={eps:.3f}")

print("평균 리턴(최근100):", np.mean(returns[-100:]))
np.save("returns_dqn.npy", returns)
torch.save(policy_net.state_dict(), "dqn_cartpole.pth")

returns = np.load("returns_dqn.npy")
plt.plot(returns, alpha=.3)
plt.plot(np.convolve(returns, np.ones(100)/100, 'valid'))
plt.title("CartPole DQN Return")
plt.savefig("return_curve_dqn.png")

with torch.no_grad():
    states = torch.randn(1000, obs_dim).to(device)
    qs = policy_net(states).cpu().numpy().flatten()
plt.hist(qs, bins=40); plt.title("Q-value distribution"); plt.savefig("q_hist.png")

