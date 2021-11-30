"""
NF
https://pyro.ai/examples/normalizing_flows_i.html#Background
https://akosiorek.github.io/ml/2018/04/03/norm_flows.html
https://blog.evjang.com/2018/01/nf1.html
NODE
https://jontysinai.github.io/jekyll/update/2019/01/18/understanding-neural-odes.html
"""

import torch


# import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import pyro
import os
smoke_test = ('CI' in os.environ)
