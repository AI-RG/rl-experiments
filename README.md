# A2C

This code can run reinforcement learning experiments: in particular, it is designed to investigate the potential of Self-Organized Criticality (SOC) as a method to speed learning. Practically speaking, criticality is encouraged by the addition of another loss term
https://latex.codecogs.com/gif.latex?L_%7BSOC%7D_%28s%29%20%3D%20ReLU%28%7C%5Cbar%7Bs%7D%7C%20-%20m_&plus;%29%5E2%20&plus;%20ReLU%28m_-%20-%20%7C%5Cbar%7Bs%7D%7C%29%5E2
which penalizes the time-averaged state hidden state s (element-wise)

This repository is modified from a version of the A2C algorithm in OpenAI's collection of baselines.


- Original paper: https://arxiv.org/abs/1602.01783
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- `python -m baselines.a2c.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
