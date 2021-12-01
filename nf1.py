"""

NF Math Proof
http://homepage.divms.uiowa.edu/~rdecook/stat2020/notes/Transformations_rv_continuous_pt2.pdf (p5,6)
https://59clc.files.wordpress.com/2011/01/real-and-complex-analysis.pdf  (p 150)
https://drive.google.com/file/d/1ig4oertmWfwT7QtcURl4QY5gCZ69EPIj/view?usp=sharing
https://www.hugendubel.de/de/ebook_pdf/vladimir_i_bogachev-measure_theory-11429515-produkt-details.html (3.7 p194)
https://drive.google.com/file/d/1K8Bk14WuuA8kpYrkXQIntmdBC65alWdl/view?usp=sharing
NF Overview
https://akosiorek.github.io/ml/2018/04/03/norm_flows.html
https://pyro.ai/examples/normalizing_flows_i.html#Background
https://blog.evjang.com/2018/01/nf1.html
NODE
https://jontysinai.github.io/jekyll/update/2019/01/18/understanding-neural-odes.html
"""

import os

import matplotlib.pyplot as plt
import pyro.distributions as dist
import pyro.distributions.transforms as T
import seaborn as sns
import torch
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

smoke_test = ('CI' in os.environ)
dist_x = dist.Normal(torch.zeros(1), torch.ones(1))
exp_transform = T.ExpTransform()
dist_y = dist.TransformedDistribution(dist_x, [exp_transform])
# Direct transformation
# 1
plt.subplot(1, 2, 1)
plt.hist(dist_x.sample([1000]).numpy(), bins=50)
plt.title('Standard Normal')
plt.subplot(1, 2, 2)
plt.hist(dist_y.sample([1000]).numpy(), bins=50)
plt.title('Standard Log-Normal')
plt.show()

dist_x = dist.Normal(torch.zeros(1), torch.ones(1))
affine_transform = T.AffineTransform(loc=3, scale=0.5)
exp_transform = T.ExpTransform()
dist_y = dist.TransformedDistribution(dist_x, [affine_transform, exp_transform])
# 2
plt.subplot(1, 2, 1)
plt.hist(dist_x.sample([1000]).numpy(), bins=50)
plt.title('Standard Normal')
plt.subplot(1, 2, 2)
plt.hist(dist_y.sample([1000]).numpy(), bins=50)
plt.title('Log-Normal')
plt.show()

# Learnables

# 3 - univariate learning
n_samples = 1000
X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
X = StandardScaler().fit_transform(X)

plt.title(r'Samples from $p(x_1,x_2)$')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.show()

plt.subplot(1, 2, 1)
sns.distplot(X[:, 0], hist=False, kde=True,
             bins=None,
             hist_kws={'edgecolor': 'black'},
             kde_kws={'linewidth': 2})
plt.title(r'$p(x_1)$')
plt.subplot(1, 2, 2)
sns.distplot(X[:, 1], hist=False, kde=True,
             bins=None,
             hist_kws={'edgecolor': 'black'},
             kde_kws={'linewidth': 2})
plt.title(r'$p(x_2)$')
plt.show()
#learning part
base_dist = dist.Normal(torch.zeros(2), torch.ones(2))
spline_transform = T.Spline(2, count_bins=16)
flow_dist = dist.TransformedDistribution(base_dist, [spline_transform])

# %%time
steps = 1 if smoke_test else 1001
dataset = torch.tensor(X, dtype=torch.float)
optimizer = torch.optim.Adam(spline_transform.parameters(), lr=1e-2)
for step in range(steps):
    optimizer.zero_grad()
    loss = -flow_dist.log_prob(dataset).mean()
    loss.backward()
    optimizer.step()
    flow_dist.clear_cache()

    if step % 200 == 0:
        print('step: {}, loss: {}'.format(step, loss.item()))

# Plot after training
X_flow = flow_dist.sample(torch.Size([1000,])).detach().numpy()
plt.title(r'Joint Distribution')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.scatter(X[:,0], X[:,1], label='data', alpha=0.5)
plt.scatter(X_flow[:,0], X_flow[:,1], color='firebrick', label='flow', alpha=0.5)
plt.legend()
plt.show()

plt.subplot(1, 2, 1)
sns.distplot(X[:,0], hist=False, kde=True,
             bins=None,
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2},
             label='data')
sns.distplot(X_flow[:,0], hist=False, kde=True,
             bins=None, color='firebrick',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2},
             label='flow')
plt.title(r'$p(x_1)$')
plt.subplot(1, 2, 2)
sns.distplot(X[:,1], hist=False, kde=True,
             bins=None,
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2},
             label='data')
sns.distplot(X_flow[:,1], hist=False, kde=True,
             bins=None, color='firebrick',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2},
             label='flow')
plt.title(r'$p(x_2)$')
plt.show()
