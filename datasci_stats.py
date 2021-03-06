# -*- coding: utf-8 -*-
"""DataSci_Stats.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e7cwQDNW_YWNQsPAIU564lIR9QWvlHRi
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
import math

np.random.seed(1234)

long_breaks=stats.poisson.rvs(loc=10,mu=60,size=3000)
pd.Series(long_breaks).hist()

short_breaks=stats.poisson.rvs(loc=10,mu=15,size=6000)
pd.Series(short_breaks).hist()

breaks=np.concatenate((long_breaks,short_breaks))

pd.Series(breaks).hist()

breaks.mean()

"""<h2>To find point estimate</h2>"""

point_estimates=[]

for x in range(500):
  sample=np.random.choice(a=breaks,size=100)

point_estimates.append(sample.mean())

pd.DataFrame(point_estimates).hist()

def confidenceinterval():
  """This function is used to calculate confidence interval"""
  sample_sz=100
  sample=np.random.choice(a= breaks,size = sample_sz)
  sample_mean=sample.mean()
  sample_stdev=sample.std()
  sigma=sample_stdev/math.sqrt(sample_sz)

  return stats.t.interval(alpha=0.95,df=sample_sz-1,loc=sample_mean,scale=sigma)

t_int=0
for i in range(10000):
  interval=confidenceinterval()
  if 39.99>interval[0] and 39.99 <= interval[1]:
    t_int +=1

print(t_int/10000)



