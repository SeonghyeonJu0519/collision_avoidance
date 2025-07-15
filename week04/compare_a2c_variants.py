import numpy as np, matplotlib.pyplot as plt, glob
def smooth(x, k=50): return np.convolve(x, np.ones(k)/k, 'valid')

curves = {'DQN':'returns_dqn.npy',
          'DDQN':'returns_ddqn.npy',
          'PER-DDQN':'returns_per.npy',
          'A2C-δ':'returns_a2c.npy',
          'A2C-GAE':'returns_a2c_gae.npy'}
for label,f in curves.items():
    if not glob.glob(f): continue
    data = np.load(f); plt.plot(smooth(data), label=label)
plt.legend(); plt.xlabel("Episode"); plt.ylabel("Return(100-avg)")
plt.savefig("return_compare4.png"); print("saved → return_compare4.png")
