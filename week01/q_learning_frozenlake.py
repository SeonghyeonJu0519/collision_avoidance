import gymnasium as gym, numpy as np, time
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=False)
nS, nA = env.observation_space.n, env.action_space.n
Q = np.zeros((nS, nA))

def act(s, eps):
    return env.action_space.sample() if np.random.rand() < eps else np.argmax(Q[s])

gamma, alpha = 0.95, 0.1
eps_start, eps_final, eps_decay = 1.0, 0.05, 0.999
episodes, success = 3000, []

for ep in range(episodes):
    s,_ = env.reset()
    done = False
    eps  = max(eps_final, eps_start * (eps_decay ** ep))
    while not done:
        a      = act(s, eps)
        s2,r,t,tr,_ = env.step(a)
        Q[s,a] += alpha * (r + gamma * Q[s2].max() - Q[s,a])
        s       = s2
        done    = t or tr
    success.append(r)

print("마지막 100회 승률 :", np.mean(success[-100:]))

np.save("Q_table.npy", Q)  # CP-1
wins = np.convolve(success, np.ones(100)/100, "valid")
plt.plot(wins)
plt.title("FrozenLake Q-learning")
plt.xlabel("episodes")
plt.ylabel("success rate")
plt.savefig("learning_curve.png")
