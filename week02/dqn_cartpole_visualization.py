import numpy as np, matplotlib.pyplot as plt
returns = np.load("returns.npy")
plt.plot(returns, alpha=.3)
plt.plot(np.convolve(returns, np.ones(100)/100, 'valid'))
plt.title("CartPole DQN Return")
plt.savefig("return_curve.png")

