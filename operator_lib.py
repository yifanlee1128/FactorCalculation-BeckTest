#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Marcials
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import math
from numpy import *

# 触顶回降，触底反弹
def bounce(factor,date_length,direction,buy_quantile,sell_quantile,**kwargs):
    factor = factor
    rolling_window=date_length
    sell_point = factor.rolling(rolling_window).quantile(sell_quantile,interpolation='midpoint')
    buy_point = factor.rolling(rolling_window).quantile(buy_quantile,interpolation='midpoint')
    buy_position=factor-buy_point
    buy_position[buy_position>=0]=0
    buy_position[buy_position<0]=1
    sell_position=factor-sell_point
    sell_position[sell_position<=0]=0
    sell_position[sell_position>0]=-1
    position=sell_position+buy_position
    position[position.isnull()]=0

    return position

#为了fix交易次数过多问题，要连续n个数据都大于buy_point才买入，连续n个数据都小于sell_point才卖出
def stable_quantile(factor,date_length,direction,buy_quantile,sell_quantile,**kwargs):
    factor = factor
    rolling_window=date_length
    sell_point=factor.expanding().quantile(axis=0,quantile=sell_quantile,index=factor.index)
    buy_point=factor.expanding().quantile(axis=0,quantile=buy_quantile,index=factor.index)

    #得到持仓信息
    buy_position=factor-buy_point
    buy_position[buy_position<=0]=0
    buy_position[buy_position>0]=1
    sell_position=factor-sell_point
    sell_position[sell_position<=0]=0
    sell_position[sell_position>0]=-1
    position=buy_position+sell_position
    for i in range(1,rolling_window,1):
        position=position+buy_position.shift(i)
        position=position+sell_position.shift(i)
    position=position/rolling_window
    position[(position<1) & (position>-1)]=0
    position[position.isnull()]=0

    return position

# 布林带
def Bolinger_Bands(factor,date_length,direction,**kwargs):
    factor = factor
    rolling_window=date_length
    mean=factor.rolling(rolling_window,min_periods=1).mean()
    std=factor.rolling(rolling_window,min_periods=1).std()
    force = mean+std
    support = mean-std
    position = factor
    for i in range(len(factor)-1):
        if ((factor.iloc[i,0]>force.iloc[i,0])&(factor.iloc[i+1,0]<force.iloc[i+1,0])):
            position.iloc[i+1,0]=-1
        elif ((factor.iloc[i,0]<support.iloc[i,0])&(factor.iloc[i+1,0]>support.iloc[i+1,0])):
            position.iloc[i+1,0]=1
        else:
            position.iloc[i+1,0]=0
    position[position.isnull()]=0

    return position

# 捕捉波动率大的点，是一波行情的开端,但是上涨还是下跌用factor和其均值判断
def Volatility(factor,date_length,direction,date_length2,**kwargs):
    factor = factor
    rolling_window=date_length
    std_window=date_length2
    std=factor.rolling(rolling_window,min_periods=1).std()
    std_mean=std.rolling(std_window,min_periods=1).mean()
    std_std=std.rolling(std_window,min_periods=1).std()
    position=std-std_mean-std_std
    position[position>0]=1
    position[position<=0]=0
    mean=factor.rolling(rolling_window,min_periods=1).mean()
    trend=factor-mean
    trend[trend>0]=1
    trend[trend<0]=-1
    position = position*trend
    position[position.isnull()]=0
    return position


# 双均线
def double_average(factor,date_length,direction,date_length2,**kwargs):
    factor = factor
    rolling_window=date_length
    rolling_window_short=date_length2
    mean=factor.rolling(rolling_window,min_periods=1).mean()
    mean_short=factor.rolling(rolling_window_short,min_periods=1).mean()
    trend=mean_short-mean
    trend[trend>0]=1
    trend[trend<0]=-1
    position = (trend-trend.shift(1))/2
    position[(position<1) & (position>-1)]=0
    position[position.isnull()]=0

    return position

# 同时满足几个条件才下单
def multi_condition(position1,position2,position3):
    position = (position1+position2+position3)/3
    position[(position<1) & (position>-1)]=0
    position[position.isnull()]=0

    return position
