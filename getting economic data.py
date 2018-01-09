# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 08:55:28 2017

@author: Berend
"""


import datetime
import calendar
import pandas as pd

period_null = datetime.date(2013, 1, 1)

def add_months(sourcedate,months):
    month = sourcedate.month - 1 + months
    year = int(sourcedate.year + month / 12 )
    month = month % 12 + 1
    day = min(sourcedate.day,calendar.monthrange(year, month)[1])
    return datetime.date(year,month,day)


inputfile = 'economic growth scenarios.xlsx'
economic_growth = pd.read_excel(inputfile, sheet='ec_growth', skiprows=1, index_col=0, header=0)

print(economic_growth['PACES'])

step = 0
while step < 36:
        cur_df = economic_growth['PACES']
        period_now = add_months(period_null, step)
        step += 1
        index_new = int(period_now.strftime('%Y'))
        if (i/12).is_integer():
            growth = cur_df[index_new]
            base = (1 + growth) * base

print(period_now)
