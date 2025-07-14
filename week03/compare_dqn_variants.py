"""
returns_*.npy (DQN, DDQN, PER) 파일을 한 번에 비교 그래프로 그려줍니다.
"""

import numpy as np, matplotlib.pyplot as plt, glob, os

files = sorted(glob.glob("returns_*.npy"))
styles= dict(dqn="--", ddqn="-.", per="-" )

for f in files:
    name = os.path.splitext(os.path.basename(f))[0].split("_")[1]  # dqn / ddqn / per
    data = np.load(f)
    smth = np.convolve(data, np.ones(100)/100, 'valid')
    plt.plot(smth, styles.get(name,"-"), label=name.upper())

plt.title("Return (100-avg) Comparison"); plt.xlabel("Episode"); plt.ylabel("Return")
plt.legend(); plt.grid(); plt.savefig("return_comparison.png")
print("saved → return_comparison.png")
