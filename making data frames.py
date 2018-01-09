# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 01:43:16 2017

@author: Berend
"""

import pandas as pd
import numpy as np

inputfile = 'economic growth scenarios.xlsx'

inputdf = pd.read_excel(inputfile, sheetname='elec')


for i in inputdf.columns:
    working_df = inputdf[i]

p1 = np.linspace(2012,2020,9)
p2 = np.linspace(2021,2030,10)
p3 = np.linspace(2031,2040,10)
p4 = np.linspace(2041,2050,10)

timeaxis = np.array()
for i in [p1, p2, p3, p4]:
    timeaxis.append(i)
