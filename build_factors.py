#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:40:28 2018
Revised lastly on Sep 14 2018

@author: LI YIFAN
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

'''
量价因子，formula by definition
'''
"""
def get_ACD(date_length, adj_close, adj_high, adj_low):
    last_close = adj_close.shift(1)
    min1 = adj_low.where(adj_low < last_close, last_close)
    max1 = adj_high.where(adj_high > last_close, last_close)

    dif = adj_close - min1.where(adj_close > last_close, max1)
    dif_if = dif.where(adj_close != last_close, 0)
    dif_if = dif_if / adj_close
    
    ACD = dif_if.tail(date_length).sum()
    return ACD


def get_AR(date_length, adj_open, adj_high, adj_low):
    H1D = adj_high.tail(date_length).sum()
    L1D = adj_low.tail(date_length).sum()
    O1D = adj_open.tail(date_length).sum()
    AR = (H1D - O1D) / (O1D - L1D)
    return AR

def get_OBV(date_length, adj_close, adj_volume):    
    pct_change = adj_close.pct_change()
    OBV_single = adj_volume.where(pct_change > 0, -adj_volume)
    OBV_single = OBV_single.where(pct_change != 0, 0)
    
    OBV = OBV_single.tail(date_length).sum()
    adj_volume_mean = adj_volume.tail(date_length * 5).mean()
    
    OBV = OBV / adj_volume_mean
    return OBV

def get_WMS(date_length, adj_close, adj_high, adj_low):
    highest = adj_high.tail(date_length * 20).max()
    lowest = adj_low.tail(date_length * 20).min()
    
    close_price = adj_close.iloc[-1]
    WMS = (highest - close_price) / (highest - lowest)
    
    return WMS

def get_RSI(date_length, adj_close):
    price_change = adj_close - adj_close.shift(1)
    price_change_reverse = price_change*(-1)
    U = price_change.where(price_change > 0, 0)
    D = price_change_reverse.where(price_change_reverse > 0, 0)
    RS_nominator = U.tail(date_length * 20).mean()
    RS_denominator = D.tail(date_length * 20).mean()
    RS1 = RS_nominator / RS_denominator
    RSI1 = RS1 / (1 + RS1) * 100
    return RSI1

def get_PSY(date_length, adj_close):
    adj_return = adj_close.pct_change()
    rising = adj_return > 0
    rising_period = rising.tail(date_length * 20).sum()
    PSY = rising_period / (date_length * 20) * 100
    return PSY
    
def get_CR(date_length, adj_close, adj_high, adj_low):
    mid = (adj_close + adj_high + adj_low) / 3
    upper = adj_high - mid.shift(1)
    upper = upper.where(upper > 0, 0)
    upper_sum = upper.tail(date_length).sum()
    
    lower = mid.shift(1) - adj_low
    lower = lower.where(lower >0 , 0)
    lower_sum = lower.tail(date_length).sum()
    
    result = upper_sum / lower_sum
    
    return result
    
def get_Aroon(date_length, adj_high, adj_low):
#    highPrice = high.rolling(window = date_length, center = False, min_periods = int(date_length * mobs_rate)).max()
    high = adj_high.tail(date_length)
    high.index = np.arange(date_length)
    low = adj_low.tail(date_length)
    low.index = np.arange(date_length)
    
    highIndex = high.idxmax()
    lowIndex = low.idxmin()
    highDay = (date_length - 1 - highIndex)*1.0 
    lowDay = (date_length - 1 - lowIndex)*1.0 
    
    Aroon = highDay - lowDay
    return Aroon
    
def get_PVT(date_length, adj_close, adj_volume):
    pct_change = adj_close.pct_change(periods = date_length).iloc[-1]
    adj_volume = adj_volume.tail(date_length).mean()
    adj_volume_mean = adj_volume.tail(5 * date_length).mean()
    
    PVT = pct_change * adj_volume / (adj_volume_mean)
    return PVT
    
def get_ADTM(date_length, adj_open, adj_high, adj_low):
    DTM1 = adj_high - adj_open
    DTM2 = adj_open - adj_open.shift(1)
    DTM_max = DTM1.where(DTM1 > DTM2, DTM2)
    DTM0 = pd.DataFrame(0, index = DTM1.index, columns = DTM2.columns)
    DTM = DTM0.where(DTM2 <= 0, DTM_max)
    STM = DTM.tail(date_length).sum()
    
    DBM1 = adj_open - adj_low
    DBM2 = adj_open.shift(1) - adj_open
    DBM_max = DBM1.where(DBM1 > DBM2, DBM2)
    DBM0 = pd.DataFrame(0, index = DTM1.index, columns = DTM2.columns)
    DBM = DBM0.where(DTM2 >= 0, DBM_max)
    SBM = DBM.tail(date_length).sum()

    nominator = STM-SBM
    denominator = STM.where(STM > SBM, SBM)
    
    ADTM = nominator / denominator
    return ADTM
  
    
def get_AD(date_length, adj_close, adj_high, adj_low, adj_volume):
    adj_volume_rollmean = adj_volume.tail(5 * date_length).mean()
    single = ((adj_close - adj_low) - (adj_high - adj_close)) / (adj_high - adj_low)  * adj_volume / adj_volume_rollmean

    AD = single.tail(date_length).sum()
    return AD    
    
def get_ATR(date_length, adj_close, adj_high, adj_low):
    max1_1 = adj_high - adj_low
    max1_2 = (adj_close.shift(1) - adj_high).abs()
    max2 = (adj_close.shift(1) - adj_low).abs()
    
    max1 = max1_1.where(max1_1 > max1_2, max1_2)
    maxx = max1.where(max1 > max2, max2)
    
    ATR = maxx.tail(date_length).sum()
    return ATR

def getAvgDev(data):
    return (data - data.mean()).abs().mean()
#    return np.sum(np.absolute(data - np.mean(data))) / len(data)

def get_CCI(date_length, adj_close, adj_high, adj_low):
    mid = (adj_close + adj_high + adj_low) / 3
    nominator = (mid.iloc[-1] - mid.tail(date_length).mean())
    denominator = mid.tail(date_length).apply(getAvgDev)
    
    CCI = nominator / denominator
    return CCI
    
def get_ILLIQ(date_length, adj_close, amount):
    rtn = adj_close.pct_change()
    rtn = rtn.abs()
        
    ILLIQ = (rtn / amount) * 10**8
    ILLIQ = ILLIQ.tail(date_length).mean()
    
    return ILLIQ
    
def get_TO(date_length, turnover):
    TO = turnover.tail(date_length).sum()
    
    return TO
"""

"""尚未完成的包括：KDJ，RSTR504,JDQS20,CmraCNE5,Cmra,Hbeta，Hsigma，Hurst"""

def daily_return(close):
    return close.pct_change()

def get_MA(date_length, adj_close):
    MA=adj_close.rolling(window=date_length,center=False).mean()
    return MA

def get_EMA(date_length, adj_close):
    if date_length==1:
        return adj_close
    else:
        alpha = 2/(date_length+1)
        EMA=alpha*adj_close+(1-alpha)*get_EMA(date_length-1,adj_close.shift(1))
        return EMA

def get_BBI(adj_close):
    BBI=(get_MA(3,adj_close)+get_MA(6,adj_close)+get_MA(12,adj_close)+get_MA(24,adj_close))/4
    return BBI

def get_TEMA(date_length,adj_close):
    N=date_length
    TEMA=3*get_EMA(N,adj_close)-3*get_EMA(N,get_EMA(N,adj_close))+get_EMA(N,get_EMA(N,get_EMA(N,adj_close)))
    return TEMA

def get_BollDown(date_length,adj_close):
    MA=get_MA(date_length,adj_close)
    BollDown=MA-adj_close.rolling(window=20,center=False).std()
    return BollDown

def get_BollUp(date_length,adj_close):
    MA=get_MA(date_length,adj_close)
    BollUp=MA+adj_close.rolling(window=20,center=False).std()
    return BollUp

"""
KDJ指标未完成
def get_KDJ_K(adj_close,adj_low,adj_high,date_length=9):
    '''
    temp1=adj_low.rolling(window=date_length,center=False).min()
    lowest=temp1.where(temp1.notnull(),adj_low)
    temp2=adj_high.rolling(window=date_length,center=False).max()
    highest=temp2.where(temp2.notnull(),adj_high)
    RSV=(adj_close-lowest)/(highest-lowest)
    fac1=pd.DataFrame([2/3]*len(adj_close.index),index=adj_close.index).cumprod()*50
    temp3=pd.DataFrame([1]+[2/3]*9,index=adj_close.index,columns='A').cumprod()
    temp3.sort_values(inplace=True,by=['A'],ascending=True)
    temp3.index=adj_close.index
    RSV=RSV.mul(temp3.squeeze(),axis=0)
    fac2=RSV.cumsum()*1/3
    KDJ_K=fac1+fac2
    '''
    lowest = adj_low.rolling(window=date_length, center=False,min_periods=1).min()
    highest = adj_high.rolling(window=date_length, center=False,min_periods=1).max()
    lowest.iloc[:9-date_length,:]=np.nan
    highest.iloc[:9-date_length,:]=np.nan
    RSV = (adj_close - lowest) / (highest - lowest)
    if date_length==1:
        temp=(2*50+RSV)/3
        return temp
    else:
        temp=get_KDJ_K(adj_close.shift(1),adj_low.shift(1),adj_high.shift(1),date_length-1)
        return temp.where(temp.notnull(),50)*2/3+RSV/3



def get_KDJ_D(adj_close,adj_low,adj_high,date_length=9):
    '''
    KDJ_K=get_KDJ_K(adj_close,adj_low,adj_high,date_length)
    fac1=pd.DataFrame([2/3]*len(adj_close.index),index=adj_close.index).cumprod()*50
    temp3=pd.DataFrame([1]+[2/3]*9,index=adj_close.index,columns='A').cumprod()
    temp3.sort_values(inplace=True,by=['A'],ascending=True)
    temp3.index=adj_close.index
    KDJ_K=KDJ_K.mul(temp3.squeeze(),axis=0)
    fac2=KDJ_K.cumsum()*1/3
    KDJ_D=fac1+fac2
    return KDJ_D
    '''
    KDJ_K = get_KDJ_K(adj_close, adj_low, adj_high, date_length)
    if date_length==1:
        temp=(2*50+KDJ_K)/3
        return temp
    else:
        temp=get_KDJ_D(adj_close.shift(1),adj_low.shift(1),adj_high.shift(1),date_length-1)
        return temp.where(temp.notnull(),50)*2/3+KDJ_K/3


def get_KDJ_J(adj_close,adj_low,adj_high):
    KDJ_J=3*get_KDJ_K(adj_close,adj_low,adj_high)-2*get_KDJ_D(adj_close,adj_low,adj_high)
    return KDJ_J

"""
def get_UpRVI(date_length,adj_close):
    SD=adj_close.rolling(window=10,center=False).std()
    temp=(adj_close-adj_close.shift(1)).where(SD.notnull(),np.nan)
    USD=SD.mask(temp<0 ,0)
    UpRVI=get_EMA(2*date_length-1,USD)
    return UpRVI

def get_DownRVI(date_length,adj_close):
    SD=adj_close.rolling(window=10,center=False).std()
    temp=(adj_close-adj_close.shift(1)).where(SD.notnull(),np.nan)
    DSD=SD.mask(temp>0 ,0)
    DownRVI=get_EMA(2*date_length-1,DSD)
    return DownRVI

def get_RVI(date_length,adj_close):
    RVI=100*get_UpRVI(date_length,adj_close)/(get_UpRVI(date_length,adj_close)+get_DownRVI(date_length,adj_close))
    return RVI

def get_DBCD(adj_close,date_lengthN=5,date_lengthM=16,date_legnthT=17):
    BIAS=(adj_close/get_MA(date_lengthN,adj_close)-1)*100
    DIF=BIAS.diff(date_lengthM)
    DBCD=get_EMA(date_legnthT,DIF)
    return DBCD

def get_Illiquidity(adj_close,deal_amount):
    R=adj_close.pct_change()
    Illiquidity=R.rolling(window=20,center=False).sum()/deal_amount.rolling(window=20,center=False).sum()*1e9
    return Illiquidity

def get_Mfi(adj_volume,adj_close,adj_low,adj_high,date_length=14):
    TYP=(adj_close+adj_high+adj_low)/3
    MF=TYP*adj_volume
    temp=TYP-TYP.shift(1)
    temp_1 = MF.where(temp > 0, 0)
    temp_2 = MF.where(temp < 0, 0)
    MR=temp_1.rolling(window=date_length,center=False).sum()/temp_2.rolling(window=date_length,center=False).sum()
    MFI=100-100/(1+MR)
    return MFI

def get_CR(adj_close,adj_low,adj_high,date_length=20):
    TYP=(adj_low+adj_high+adj_close)/3
    temp_long=adj_high-TYP.shift(1)
    temp_long1=temp_long.mask(temp_long<0,0)
    temp_short=TYP.shift(1)-adj_low
    temp_short1=temp_short.mask(temp_short<0,0)
    CR=100*temp_long1.rolling(window=date_length,center=False).sum()/temp_short1.rolling(window=20,center=False).sum()
    return CR

def get_MassIndex(adj_high,adj_low):
    EMAHL=get_EMA(9,adj_high-adj_low)
    EMA_Ratio=EMAHL/get_EMA(9,EMAHL)
    MassIndex=EMA_Ratio.rolling(window=25,center=False).sum()
    return MassIndex

def get_BullPower(adj_close,adj_high,date_length=13):
    BullPower = adj_high - get_EMA(date_length, adj_close)
    return BullPower

def get_BearPower(adj_close,adj_low,date_length=13):
    BearPower = adj_low - get_EMA(date_length, adj_close)
    return BearPower

def get_Elder(adj_close,adj_low,adj_high,date_length=13):
    BullPower=get_BullPower(adj_close,adj_high,date_length)
    BearPower=get_BearPower(adj_close,adj_low,date_length)
    Elder=(BullPower-BearPower)/adj_close
    return Elder

def get_SwingIndex(adj_close,adj_open,adj_low,adj_high):
    A = (adj_close - adj_close.shift(1)).abs()
    B = (adj_low - adj_close.shift(1)).abs()
    C = (adj_high - adj_low.shift(1)).abs()
    D = (adj_close.shift(1) - adj_open.shift(1)).abs()
    E = adj_close - adj_close.shift(1)
    F = adj_close - adj_open
    G = adj_close.shift(1) - adj_open.shift(1)
    X = E + F / 2 + G
    K = B.mask(A > B, A)
    R1 = A + B / 2 + D / 4
    R2 = A / 2 + B + D / 4
    R3 = C + D / 4
    MAX = A.mask(B > A, B)
    MAX = MAX.mask(C > MAX, C)
    R = R1
    R = R.mask(MAX == B, R2)
    R = R.mask(MAX == C, R3)
    SI=16*X/R*K
    return SI

def get_ASI(adj_close,adj_open,adj_low,adj_high):
    SI=get_SwingIndex(adj_close,adj_open,adj_low,adj_high)
    ASI=SI.rolling(window=20,center=False).sum()
    return ASI

def get_AD(adj_close,adj_high,adj_low,adj_volume,date_length=1):
    """按文档结果除以1e6，防止数值过大"""
    MFV=((adj_close-adj_low)-(adj_high-adj_close))/(adj_high-adj_low)*adj_volume
    MFV=MFV.mask(adj_high==adj_low,adj_close.pct_change()*adj_volume)
    MFV.iloc[0, :] = MFV.iloc[0, :].where(MFV.iloc[0, :].notnull(), 0)
    temp=MFV.cumsum()
    AD=get_MA(date_length,temp)/1e6
    return AD

def get_ChaikinOscillator(adj_close,adj_low,adj_high,adj_volume):
    AD=get_AD(adj_close,adj_low,adj_high,adj_volume)
    ChaikinOscillator=get_EMA(3,AD)-get_EMA(10,AD)
    return ChaikinOscillator

def get_ChaikinVolatility(adj_high,adj_low):
    HLEMA=get_EMA(10,adj_high-adj_low)
    ChaikinVolatility=100*HLEMA.diff(10)/HLEMA.shift(10)
    return ChaikinVolatility

def get_EMV(date_length,adj_high,adj_low,adj_volume):
    temp=((adj_high+adj_low)/2-(adj_high.shift(1)+adj_low.shift(1))/2)*(adj_high-adj_low)/adj_volume
    EMV=get_EMA(date_length,temp)
    return EMV

def get_PlusDM(adj_high):
    "参照网上资料定义"
    plusdm=adj_high-adj_high.shift(1)
    plusdm=plusdm.mask(plusdm<0,0)
    return plusdm

def get_MinusDM(adj_low):
    "参照网上资料定义"
    minusdm=adj_low.shift(1)-adj_low
    minusdm=minusdm.mask(minusdm<0,0)
    return minusdm

def get_PlusDI(adj_close,adj_high,adj_low,date_length=14):
    TH=adj_high.mask(adj_close.shift(1)>adj_high,adj_close.shift(1))
    TL=adj_low.mask(adj_close.shift(1)<adj_low,adj_close.shift(1))
    TR=TH-TL
    plusdi=get_EMA(date_length,get_PlusDM(adj_high))/get_EMA(date_length,TR)
    return plusdi

def get_MinusDI(adj_close,adj_high,adj_low,date_length=14):
    TH=adj_high.mask(adj_close.shift(1)>adj_high,adj_close.shift(1))
    TL=adj_low.mask(adj_close.shift(1)<adj_low,adj_close.shift(1))
    TR=TH-TL
    minusdi=get_EMA(date_length,get_MinusDM(adj_low))/get_EMA(date_length,TR)
    return minusdi

def get_ADX(adj_close,adj_high,adj_low,date_length=14):
    "DX参照网上资料定义"
    plusdi=get_PlusDI(adj_close,adj_high,adj_low,date_length)
    minusdi=get_MinusDI(adj_close,adj_high,adj_low,date_length)
    DX=(plusdi-minusdi).abs()/(plusdi+minusdi)
    ADX=get_EMA(date_length,DX)
    return ADX

def get_ADXR(adj_close,adj_high,adj_low,date_length=14,delta_date=14):
    "date_length用于ADX计算公式中的指数平滑,是间接参数，delta_date是本函数的直接参数"
    adx1=get_ADX(adj_close,adj_high,adj_low,date_length)
    adx2=adx1.shift(delta_date)
    ADXR=(adx1+adx2)/2
    return ADXR

def get_DIFF(adj_close):
    DIFF=get_EMA(12,adj_close)-get_EMA(26,adj_close)
    return DIFF

def get_DEA(adj_close,date_length=9):
    DEA=get_EMA(date_length,get_DIFF(adj_close))
    return DEA

def get_MACD(adj_close,date_length=9):
    MACD=2*(get_DIFF(adj_close)-get_DEA(adj_close,date_length))
    return MACD

def get_MTM(adj_close,delta_date=10):
    MTM=adj_close-adj_close.shift(delta_date)
    return MTM

def get_MTMMA(adj_close,delta_date=10,date_length=10):
    MTMMA=get_MA(date_length,get_MTM(adj_close,delta_date))
    return MTMMA

def get_UOS(adj_close,adj_high,adj_low,date_length_M=7,date_length_N=14,date_length_O=28):
    TH = adj_high.mask(adj_close.shift(1) > adj_high, adj_close.shift(1))
    TL = adj_low.mask(adj_close.shift(1) < adj_low, adj_close.shift(1))
    TR = TH - TL
    XR=adj_close-TL
    XRM=XR.rolling(window=date_length_M,center=False).sum()/TR.rolling(window=date_length_M,center=False).sum()
    XRN=XR.rolling(window=date_length_N,center=False).sum()/TR.rolling(window=date_length_N,center=False).sum()
    XRO = XR.rolling(window=date_length_O, center=False).sum() / TR.rolling(window=date_length_O, center=False).sum()
    UOS=100*(XRM*date_length_N*date_length_O+XRN*date_length_M*date_length_O+XRO*date_length_M*date_length_N)/(
            date_length_N*date_length_M+date_length_N*date_length_O+date_length_M*date_length_O)
    return UOS

def get_ULcer(date_length,adj_close):
    R=(adj_close-adj_close.rolling(window=date_length,center=False).max())/adj_close.rolling(window=date_length,center=False).max()
    Ulcer=np.square(R)
    Ulcer=Ulcer.rolling(window=date_length,center=False).sum()/date_length
    Ulcer=np.sqrt(Ulcer)
    return Ulcer

def get_DHILO(adj_high,adj_low):
    """三个月处理为60个交易日"""
    temp1=np.log(adj_high)
    temp2=np.log(adj_low)
    DHILO=(temp1-temp2).rolling(window=60,center=False).median()
    return DHILO



def get_ARC(date_length,adj_close):
    def get_EMA_ARC(date_length, adj_close):
        if date_length == 1:
            return adj_close
        else:
            EMA = 1/date_length * adj_close + (1 - 1/date_length) * get_EMA_ARC(date_length - 1, adj_close.shift(1))
            return EMA
    RC=adj_close/adj_close.shift(date_length)
    ARC=get_EMA_ARC(date_length,RC)
    return ARC

def get_APBMA(date_length,adj_close):
    APBMA=get_MA(date_length,(adj_close-get_MA(date_length,adj_close)).abs())
    return APBMA

def get_BBIC(adj_close):
    BBIC=get_BBI(adj_close)/adj_close
    return BBIC

def get_MA10Close(adj_close):
    MA10Close=get_MA(10,adj_close)/adj_close
    return MA10Close

def get_BIAS(date_length,adj_close):
    BIAS=(adj_close/get_MA(date_length,adj_close)-1)*100
    return BIAS

def get_CCI(date_length,adj_close,adj_high,adj_low):
    TYP=(adj_close+adj_high+adj_low)/3
    MATYP=get_MA(date_length,TYP)
    DEV=((TYP-MATYP).abs()).rolling(window=date_length,center=False).sum()
    CCI=(TYP-MATYP)/0.015/DEV
    return CCI

def get_ROC(date_length,adj_close):
    ROC=100*(adj_close.diff(date_length)-1)
    return ROC

def get_SRMI(adj_close,date_length=10):
    temp1=adj_close.shift(date_length)
    temp2=adj_close.mask(temp1>adj_close,temp1)
    SRMI=(adj_close-temp1)/temp2
    return SRMI

def get_ChandeSD(adj_close,date_length=20):
    temp1=adj_close.shift(1)-adj_close
    temp2=temp1.mask(temp1<0,0)
    SD=temp2.rolling(window=date_length,center=False).sum()
    return SD

def get_ChandeSU(adj_close,date_length=20):
    temp1 = adj_close - adj_close.shift(1)
    temp2 = temp1.mask(temp1 < 0, 0)
    SU = temp2.rolling(window=date_length, center=False).sum()
    return SU

def get_CMO(adj_close,date_length=20):
    SU=get_ChandeSU(adj_close,date_length)
    SD=get_ChandeSD(adj_close,date_length)
    CMO=(SU-SD)/(SU+SD)*100
    return CMO

def get_REVS(date_length,adj_close):
    """文档中的停牌处理暂时未考虑"""
    REVS=adj_close/adj_close.shift(date_length)
    return REVS

def get_REVS_M(date_length1,date_length2,adj_close):
    REVS_M=get_REVS(date_length1,adj_close)-get_REVS(date_length2,adj_close)
    return REVS_M

def get_Fiftytwoweekhigh(adj_close):
    r=adj_close.pct_change()
    r.iloc[0,:]=0
    r=r+1
    Price=r.cumprod()
    min=Price.rolling(window=240,center=False).min()
    max=Price.rolling(window=240,center=False).max()
    ftwh=(Price-min)/(max-min)
    return ftwh

def get_Price1M(adj_close):
    Price1M=20*adj_close/adj_close.rolling(window=20,center=False).sum()-1
    return Price1M

def get_Price3M(adj_close):
    Price3M=60*adj_close/adj_close.rolling(window=60,center=False).sum()-1
    return Price3M


def get_Price1Y(adj_close):
    """按照文档定义，N取250"""
    Price1Y=250*adj_close/adj_close.rolling(window=250,center=False).sum()-1
    return Price1Y

def get_Rank1M(adj_close):
    REVS20=get_REVS(20,adj_close)
    rank_value=REVS20.rank(axis=1,ascending=False)
    N=len(REVS20.columns)
    Rank1M=1-rank_value/N
    return Rank1M

def get_RC(date_length,adj_close):
    RC=adj_close/adj_close.shift(date_length)
    return RC

def get_RSTR(date_length,adj_close):
    "材料中是月收益率序列，但与日收益率进行操作等价"
    temp=adj_close/adj_close.shift(1)
    temp1=np.log(temp)
    RSTRN=temp1.rolling(window=date_length,center=False).sum()
    return RSTRN

def get_RSTR504():
    """未完成"""
    pass

def get_WMA_forCoppock(date_length,adj_close):
    if date_length==1:
        return adj_close
    else:
        return date_length*adj_close+get_WMA_forCoppock(date_length-1,adj_close.shift(1))

def get_CoppockCurve(adj_close,date_length_N1,date_length_N2,date_length_M):
    RC=100*(adj_close/adj_close.shift(date_length_N1)+adj_close/adj_close.shift(date_length_N2))
    temp=(date_length_M+1)*date_length_M/2
    Coppock=get_WMA_forCoppock(date_length_M,RC)/temp
    return Coppock

for_aroon_up=lambda x:pd.Series(x).idxmax(axis=1)
for_aroon_down=lambda x:pd.Series(x).idxmin(axis=1)

def get_AroonDown(adj_low,date_length=26):
    adj_low.index=range(len(adj_low.index))
    y=date_length-adj_low.rolling(window=date_length,center=False).apply(func=for_aroon_down)-1
    AroonDown=(date_length-y)/date_length
    return AroonDown

def get_AroonUp(adj_high,date_length=26):
    adj_high.index=range(len(adj_high.index))
    x=date_length-adj_high.rolling(window=date_length,center=False).apply(func=for_aroon_up)-1
    AroonUp=(date_length-x)/date_length
    return AroonUp

def get_Aroon(adj_low,adj_high,date_length=26):
    aroonup=get_AroonUp(adj_high,date_length)
    aroondown=get_AroonDown(adj_low,date_length)
    aroon=aroonup-aroondown
    return aroon

def get_DMZ(adj_high,adj_low):
    temp1=adj_high+adj_low
    temp2=adj_high.shift(1)+adj_low.shift(1)
    temp3=(adj_high-adj_high.shift(1)).abs()
    temp4=(adj_low-adj_low.shift(1)).abs()
    DMZ=temp3.mask(temp4>temp3,temp4)
    DMZ=DMZ.mask(temp1<=temp2,0)
    return DMZ

def get_DMF(adj_high,adj_low):
    temp1=adj_high+adj_low
    temp2=adj_high.shift(1)+adj_low.shift(1)
    temp3=(adj_high-adj_high.shift(1)).abs()
    temp4=(adj_low-adj_low.shift(1)).abs()
    DMF=temp3.mask(temp4>temp3,temp4)
    DMF=DMF.mask(temp1>=temp2,0)
    return DMF

def get_DIZ(date_length,adj_high,adj_low):
    DMZ=get_DMZ(adj_high,adj_low)
    DMF=get_DMF(adj_high,adj_low)
    DIZ=DMZ.rolling(window=date_length,center=False).sum()/(DMZ.rolling(window=date_length,center=False).sum()+
                                                            DMF.rolling(window=date_length, center=False).sum())
    return DIZ

def get_DIF(date_length,adj_high,adj_low):
    DMZ=get_DMZ(adj_high,adj_low)
    DMF=get_DMF(adj_high,adj_low)
    DIF=DMF.rolling(window=date_length,center=False).sum()/(DMZ.rolling(window=date_length,center=False).sum()+
                                                            DMF.rolling(window=date_length, center=False).sum())
    return DIF

def get_DDI(adj_high,adj_low,date_length=13):
    DIZ=get_DIZ(date_length,adj_high,adj_low)
    DIF=get_DIF(date_length,adj_high,adj_low)
    DDI=DIZ-DIF
    return DDI

def get_PVT(adj_close,adj_volume,date_length=1):
    PVT=(adj_close-adj_close.shift(1))/adj_close.shift(1)*adj_volume
    PVT_ac=PVT.rolling(window=date_length,center=False).sum()/1e6
    return PVT_ac

def get_TRIX(date_length,adj_close):
    EMA3=get_EMA(date_length,get_EMA(date_length,get_EMA(date_length,adj_close)))
    TRIX=EMA3/EMA3.shift(1)-1
    return TRIX

def get_MA10RegressCoeff(date_length,adj_close,ascending=True):
    if ascending:
        a=list(range(1,date_length+1))
    else:
        a=list(range(date_length,0,-1))
    MA=get_MA(date_length,adj_close)
    fit = lambda t: sm.OLS(t, sm.add_constant(a)).fit().params[1]
    MARegressCoef = MA.rolling(window=date_length, center=False).apply(func=fit)
    return MARegressCoef

def get_PLRC(date_length,adj_close,ascending=True):
    if ascending:
        a=list(range(1,date_length+1))
    else:
        a=list(range(date_length,0,-1))
    fit = lambda t: sm.OLS(t, sm.add_constant(a)).fit().params[1]
    PLRC=adj_close.rolling(window=date_length,center=False).apply(func=fit)
    return PLRC

def get_AR(adj_open,adj_high,adj_low,date_length=26):
    AR=(adj_high-adj_open).rolling(window=date_length,center=False).sum()/\
       (adj_open-adj_low).rolling(window=date_length,center=False).sum()
    return AR

def get_BR(adj_close,adj_high,adj_low,date_length=26):
    temp1=adj_high-adj_close.shift(1)
    temp1=temp1.mask(temp1<0,0)
    temp2=adj_close.shift(1)-adj_low
    temp2=temp2.mask(temp2<0,0)
    BR=temp1.rolling(window=date_length,center=False).sum()/temp2.rolling(window=date_length,center=False).sum()
    return BR

def get_ARBR(adj_open,adj_close,adj_high,adj_low,date_length=26):
    AR=get_AR(adj_open,adj_high,adj_low,date_length)
    BR=get_BR(adj_close,adj_high,adj_low,date_length)
    ARBR=AR-BR
    return ARBR

def get_PSY(date_length,adj_close):
    temp=adj_close.diff(1)
    temp=temp.mask(temp>0,1)
    temp=temp.mask(temp!=1,0)
    PSY=temp.rolling(window=date_length,center=False).sum()/date_length
    return PSY
"""这里两个迭代法不使用"""
def get_NVI2(df_length,adj_close,adj_volume):
    """
    这里给出了按行迭代的方法，但并没有进行验证
    df_length为输入的数据的时间长度，缺乏该参数不方便迭代，实际使用时可传递len(adj_close.index)作为实参
    """
    temp1=adj_close.diff(1)/adj_close
    temp2=adj_volume.diff(1)
    delta=temp1.mask(temp2>=0,0)
    delta=delta.where(delta.notnull(),0)
    if df_length==1:
        return delta+100
    else:
        return delta+get_NVI2(df_length-1,adj_close.shift(1),adj_volume.shift(1))

def get_PVI2(df_length,adj_close,adj_volume):
    """
    这里给出了按行迭代的方法，但并没有进行验证
    df_length为输入的数据的时间长度，缺乏该参数不方便迭代，实际使用时可传递len(adj_close.index)作为实参
    """
    temp1=adj_close.diff(1)/adj_close
    temp2=adj_volume.diff(1)
    delta=temp1.mask(temp2<=0,0)
    delta=delta.where(delta.notnull(),0)
    if df_length==1:
        return delta+100
    else:
        return delta+get_PVI2(df_length-1,adj_close.shift(1),adj_volume.shift(1))


def get_NVI(adj_close,adj_volume):
    temp1 = adj_close.diff(1) / adj_close
    temp2 = adj_volume.diff(1)
    delta = temp1.mask(temp2 >= 0, 0)
    delta=delta.where(delta.notnull(), 0)
    NVI=delta.cumsum()+100
    return NVI

def get_PVI(adj_close,adj_volume):
    temp1 = adj_close.diff(1) / adj_close
    temp2 = adj_volume.diff(1)
    delta = temp1.mask(temp2 <= 0, 0)
    delta=delta.where(delta.notnull(), 0)
    PVI=delta.cumsum()+100
    return PVI

def get_JDQS20():
    """缺乏大盘开盘数据，未完成"""
    pass

def get_DTM(adj_open,adj_high):
    temp1 = adj_high - adj_open
    temp2 = adj_open - adj_open.shift(1)
    DTM = temp1.mask(temp2 > temp1, temp2)
    DTM = DTM.mask(temp2 <= 0, 0)
    return DTM

def get_DBM(adj_open,adj_low):
    temp2 = adj_open - adj_open.shift(1)
    temp3 = adj_open - adj_low
    DBM = temp3.mask(temp2 > temp3, temp2)
    DBM = DBM.mask(temp2 >= 0, 0)
    return DBM

def get_STM(adj_open,adj_high,date_length=20):
    DTM=get_DTM(adj_open,adj_high)
    STM = DTM.rolling(window=date_length, center=False).sum()
    return STM

def get_SBM(adj_open,adj_low,date_length=20):
    DBM=get_DBM(adj_open,adj_low)
    SBM = DBM.rolling(window=date_length, center=False).sum()
    return SBM

def get_ADTM(date_length,adj_open,adj_high,adj_low):
    STM=get_STM(adj_open,adj_high,date_length)
    SBM=get_SBM(adj_open,adj_low,date_length)
    ADTM=(STM-SBM)/STM.mask(SBM>STM,SBM)
    return ADTM
"""
def get_iterforATR(MA,TR,iter_length,date_length):
    if iter_length==1:
        return MA
    else:
        temp1=get_iterforATR(MA.shift(1),TR.shift(1),iter_length-1,date_length)
        temp1.iloc[date_length:2*date_length-iter_length,:]=0
        temp2=TR
        temp2.iloc[date_length:2*date_length-iter_length,:]=0
        temp=((date_length-1)/date_length)*temp1+1/date_length*temp2
        return temp

def get_ATR(date_length,adj_close,adj_high,adj_low):
    temp1=adj_high-adj_low
    temp2=(adj_high-adj_close.shift(1)).abs()
    temp3=(adj_low-adj_close.shift(1)).abs()
    TR=temp1.mask(temp2>temp1,temp2)
    TR=TR.mask(temp3>TR,temp3)
    MA=get_MA(date_length,TR)
    ATR=get_iterforATR(MA,TR,date_length,date_length)
    return ATR
"""
def get_ATR(date_length,adj_close,adj_high,adj_low):
    temp1 = adj_high - adj_low
    temp2 = (adj_high - adj_close.shift(1)).abs()
    temp3 = (adj_low - adj_close.shift(1)).abs()
    TR=temp1.mask(temp2>temp1,temp2)
    TR=TR.mask(temp3>TR,temp3)
    MA = get_MA(date_length, TR)
    TR.iloc[0:date_length,:]=MA.iloc[0:date_length,:]
    ATR=TR.ewm(alpha=1/date_length,adjust=False,min_periods=1).mean()
    return ATR

def get_RSI(date_length,adj_close):
    U_i=adj_close.diff(1)
    U_i=U_i.mask(U_i<=0,0)
    D_i=adj_close.shift(1)-adj_close
    D_i=D_i.mask(D_i<=0,0)
    RS=get_MA(date_length,U_i)/get_MA(date_length,D_i)
    RSI=100-100/(1+RS)
    return RSI

def get_Volatility(turnover_rate):
    sd=turnover_rate.rolling(window=20,center=False).std()
    mean=turnover_rate.rolling(window=20,center=False).mean()
    volatility=sd/mean
    return volatility

def get_WVAD(date_length,adj_close,adj_open,adj_high,adj_low,adj_volume):
    temp=(adj_close-adj_open)/(adj_high-adj_low)*adj_volume
    WVAD=temp.rolling(window=date_length,center=False).sum()
    return WVAD

def get_MAWVAD(date_length1,date_length2,adj_close,adj_open,adj_high,adj_low,adj_volume):
    """date_length1用于计算WVAD，date_length2用于计算MA"""
    WVAD=get_WVAD(date_length1,adj_close,adj_open,adj_high,adj_low,adj_volume)
    MAWVAD=get_MA(date_length2,WVAD)
    return MAWVAD

def get_Volumn1M(adj_close,adj_volume):
    REVS20=get_REVS(20,adj_close)
    volumn1m=(20*adj_volume/adj_volume.rolling(window=20,center=False).sum()-1)*REVS20
    return volumn1m

def get_Volumn3M(adj_close,adj_volume):
    REVS60=get_REVS(60,adj_close)
    volumn3m=(12*adj_volume.rolling(window=5,center=False).sum()/adj_volume.rolling(window=60,center=False).sum()-1)*REVS60
    return volumn3m

def get_ACD(date_length,adj_close,adj_high,adj_low):
    temp1=adj_low.mask(adj_close.shift(1)<adj_low,adj_close.shift(1))
    buy=adj_close-temp1
    buy=buy.where(adj_close>adj_close.shift(1),0)
    temp2=adj_high.mask(adj_close.shift(1)>adj_high,adj_close.shift(1))
    sell=adj_close-temp2
    sell=sell.where(adj_close<adj_close.shift(1),0)
    ACD=buy.rolling(window=date_length,center=False).sum()+sell.rolling(window=date_length,center=False).sum()
    return ACD

def get_OBV(date_length,adj_close,adj_volume):
    temp=adj_close.diff(1)
    signal=temp.mask(temp>0,1)
    signal=signal.mask(signal!=1,-1)
    signal=signal.where(signal.notnull(),1)
    adj_volume=adj_volume/100
    temp1=signal*adj_volume
    OBV=temp1.rolling(window=date_length,center=False).sum()
    return OBV

def get_VDIFF(adj_volume):
    VDIFF=get_EMA(12,adj_volume)-get_EMA(26,adj_volume)
    return VDIFF

def get_VDEA(adj_volume,date_length=9):
    VDIFF=get_VDIFF(adj_volume)
    VDEA=get_EMA(date_length,VDIFF)
    return VDEA

def get_VMACD(adj_volume,date_length=9):
    VDIFF=get_VDIFF(adj_volume)
    VDEA=get_VDEA(adj_volume,date_length)
    VMACD=VDIFF-VDEA
    return VMACD

def get_VEMA(date_length,adj_volume):
    VEMA=get_EMA(date_length,adj_volume)
    return VEMA

def get_TVMA(date_length,deal_amount):
    """单位可能有问题"""
    TVMA=get_MA(date_length,deal_amount)
    return TVMA

def get_TVSTD(date_length,deal_amount):
    """单位可能有问题"""
    TVSTD=deal_amount.rolling(window=date_length,center=False).std()
    return TVSTD

def get_VOL(date_length,turnover_rate):
    mean=turnover_rate.rolling(window=date_length,center=False).mean()
    return mean

def get_VOSC(adj_volume):
    VOSC=100*(get_MA(12,adj_volume)-get_MA(26,adj_volume))/get_MA(12,adj_volume)
    return VOSC

def get_VR(adj_close,adj_volume,date_length=24):
    AV=BV=CV=adj_volume
    diff=adj_close.diff(1)
    AV=AV.where(diff>0,0)
    BV=BV.where(diff<0,0)
    CV=CV.where(diff==0,0)
    AVS=AV.rolling(window=date_length,center=False).sum()
    BVS=BV.rolling(window=date_length,center=False).sum()
    CVS=CV.rolling(window=date_length,center=False).sum()
    VR=(AVS+CVS/2)/(BVS+CVS/2)
    return VR

def get_VROC(date_length,adj_volume):
    VROC=100*(adj_volume/adj_volume.shift(date_length)-1)
    return VROC

def get_VSTD(date_length,adj_volume):
    VSTD=adj_volume.rolling(window=date_length,center=False).std()
    return VSTD

def get_KlingerOscillator(adj_close,adj_high,adj_low,adj_volume):
    TYP=(adj_close+adj_high+adj_low)/3
    temp=TYP.diff(1)
    adj_volume=adj_volume.where(temp>0,-adj_volume)
    KO=get_EMA(6,(get_EMA(34,adj_volume)-get_EMA(55,adj_volume)))/1e6
    return KO

def get_MoneyFlow(adj_close,adj_high,adj_low,adj_volume,date_length=20):
    MF=(adj_close+adj_low+adj_high)*adj_volume/3
    MF=MF.rolling(window=date_length,center=False).sum()
    return MF

def get_DAVOL(date_length,turnover_rate):
    DAVOL=get_VOL(date_length,turnover_rate)-get_VOL(120,turnover_rate)
    return DAVOL

def get_STOM(turnover_rate):
    temp=turnover_rate.rolling(window=20,center=False).sum()
    STOM=np.log(temp)
    return STOM

def get_STOQ(turnover_rate):
    """按文档定义，每月21个交易日"""
    STOM=get_STOM(turnover_rate)
    temp1=np.exp(STOM)
    temp2=temp1.rolling(window=63,center=False).sum()/63
    STOQ=np.log(temp2)
    return STOQ

def get_STOA(turnover_rate):
    """按文档定义，每月21个交易日"""
    STOM=get_STOM(turnover_rate)
    temp1=np.exp(STOM)
    temp2=temp1.rolling(window=252,center=False).sum()/252
    STOA=np.log(temp2)
    return STOA

def get_Variance(date_length,adj_close):
    temp=adj_close.pct_change()
    var=temp.rolling(window=date_length,center=False).var()*250
    return var

def get_Kurtosis(date_length,adj_close):
    temp = adj_close.pct_change()
    kurt=temp.rolling(window=date_length,center=False).kurt()
    return kurt

def get_Skewness(date_length,adj_close):
    skew=adj_close.rolling(window=date_length,center=False).skew()
    return skew

def get_Beta(date_length,adj_close,close_index):
    r = adj_close.pct_change()
    r_m=close_index.pct_change()
    var_index=r_m.rolling(window=date_length,center=False).var()
    corr=r.rolling(window=date_length).cov(other=r_m,pairwise=True)
    beta=corr.div(var_index.squeeze(),axis=0)
    return beta

def get_Alpha(date_length,adj_close,close_index):
    r_m = close_index.pct_change()
    r=adj_close.pct_change()
    temp1=r.rolling(window=date_length,center=False).mean()
    beta=get_Beta(date_length,adj_close,close_index)
    temp2=beta.mul((r_m.rolling(window=date_length,center=False).mean()).squeeze(),axis=0)
    alpha=temp1.sub(temp2.squeeze(),axis=0)*250
    return alpha

def get_SharpeRatio(date_length,adj_close):
    """与原文档定义有差异"""
    r=adj_close.pct_change()
    temp1=r.rolling(window=date_length,center=False).mean()*250
    temp2=r.rolling(window=date_length,center=False).std()*np.sqrt(250)
    sharpe_ratio=temp1/temp2
    return sharpe_ratio

def get_TreynorRatio(date_length,adj_close,close_index):
    r = adj_close.pct_change()
    temp1 = r.rolling(window=date_length,center=False).mean()
    beta=get_Beta(date_length,adj_close,close_index)
    TR=temp1/beta*250
    return TR

def get_InformationRatio(date_length,adj_close,close_index):
    r_m = close_index.pct_change()
    r=adj_close.pct_change()
    temp=r.sub(r_m.squeeze(),axis=0)
    temp1=temp.rolling(window=date_length,center=False).mean()
    temp2=temp.rolling(window=date_length,center=False).std()
    IR=temp1/temp2
    return IR

def get_GainVariance(date_length,adj_close):
    r=adj_close.pct_change()
    r=r.mask(r<0,np.nan)
    temp=r.rolling(window=date_length,min_periods=1).mean()
    temp.iloc[:date_length-1,:]=np.nan
    temp2=np.square(temp)
    temp3=np.square(r)
    temp4=temp3.rolling(window=date_length,min_periods=1).mean()
    temp4.iloc[:date_length - 1, :] = np.nan
    GV=(temp4-temp2)*250
    """调整数值计算误差"""
    GV=GV.mask(GV<-1e-8,0)
    return GV

def get_LossVariance(date_length,adj_close):
    r = adj_close.pct_change()
    r = r.mask(r > 0, np.nan)
    temp = r.rolling(window=date_length, min_periods=1).mean()
    temp.iloc[:date_length - 1, :] = np.nan
    temp2 = np.square(temp)
    temp3 = np.square(r)
    temp4 = temp3.rolling(window=date_length, min_periods=1).mean()
    temp4.iloc[:date_length - 1, :] = np.nan
    LV = (temp4 - temp2) * 250
    """调整数值计算误差"""
    LV = LV.mask(LV < -1e-8, 0)
    return LV

def get_GainLossVarianceRatio(date_length,adj_close):
    GLRatio=get_GainVariance(date_length,adj_close)/get_LossVariance(date_length,adj_close)
    return GLRatio

def get_DASTD(adj_close,daily_riskfree):
    r=adj_close.pct_change()
    temp=r.sub(daily_riskfree.squeeze(), axis=0)
    DASTD=temp.rolling(window=252,center=False).std()
    return DASTD

def get_CmraCNE5(adj_close,daily_riskfree):
    """未完成"""
    pass
def get_Cmra(adj_close,adj_volume,daily_riskfree):
    """未完成"""
    pass

def get_Hbeta(adj_close,close_index):
    """未完成"""
    r = adj_close.pct_change()
    rm = close_index.pct_change()

    pass

def get_Hsigma(adj_close,close_index):
    """未完成"""
    pass

def get_DDNSR(adj_close,close_index):
    r = adj_close.pct_change()
    rm=close_index.pct_change()
    nrm=rm.mask(rm>=0,np.nan)
    r["r_index"]=rm
    r[r["r_index"]>=0]=np.nan
    del r["r_index"]
    sd_r=r.rolling(window=240,min_periods=1).std()
    sd_nrm=nrm.rolling(window=240,min_periods=1).std()
    sd_r.iloc[:239, :] = np.nan
    sd_nrm[:239] = np.nan
    DDNSR=sd_r.div(sd_nrm.squeeze(),axis=0)
    return DDNSR

def get_DDNCR(adj_close,close_index):
    r=adj_close.pct_change()
    rm=close_index.pct_change()
    nrm=rm.mask(rm>=0,np.nan)
    r["r_index"]=rm
    r[r["r_index"]>=0]=np.nan
    del r["r_index"]
    DDNCR=r.rolling(window=240,min_periods=1).corr(other=nrm,pairwise=True)
    DDNCR.iloc[:239,:]=np.nan
    return DDNCR

def get_Dvrat(adj_close):
    """文档注明了无风险利率按0处理,认为文档中存在问题，代码部分编写与文档有出入"""
    r=adj_close.pct_change()
    sigma=(np.square(r)).rolling(window=480,center=False).sum()/479
    m=10*(480-10+1)*(1-10/480)
    temp1=r.rolling(window=10,center=False).sum()
    temp2=np.square(temp1)
    temp3=temp2.rolling(window=240-10+1,center=False).sum()
    sigmaq=temp3/m
    DVRAT=sigmaq/sigma-1
    return DVRAT
"""
Hurst未改好
def get_Hurst(adj_close):
    array=[4,7,15,30,60]
    Y=adj_close.applymap(lambda x:[])
    for n in array:
        X=(np.log(adj_close)).diff()
        mu=X.rolling(window=n,center=False).mean()
        sigma=X.rolling(window=n,center=False).std()
        Z=(X-mu).rolling(window=n,center=False).sum()
        R=Z.rolling(window=n,center=False).max()-Z.rolling(window=n,center=False).min()
        Y_n=(np.log(R/sigma)).applymap(lambda x:[x])
        Y=Y+Y_n
    array=np.log(array)
    Hurst=Y.applymap(lambda Y:sm.OLS(Y, sm.add_constant(array)).fit().params[1])
    return Hurst
"""
def get_Ddnbt(adj_close,close_index):
    r = adj_close.pct_change()
    rm = close_index.pct_change()
    nrm = rm.mask(rm >= 0, np.nan)
    corrcoef=r.rolling(window=240,center=False,min_periods=1).corr(other=nrm,pairwise=True)
    std_x= nrm.rolling(window=240,center=False,min_periods=1).std()
    beta=(corrcoef*r.rolling(window=240,center=False).std()).div(std_x.squeeze(),axis=0)
    return beta
