import numpy as np
import matplotlib.pyplot as plt
import pandas
import cvxpy as cvx
from xlrd import open_workbook
import functools
import math
import cvxopt as opt
from cvxopt import blas, solvers
import numpy
from scipy.stats import norm
import scipy as scipy
import cvxopt as cvxopt
import matplotlib.pyplot as plt
import copy
from numpy.lib.stride_tricks import as_strided

#function to calculate monthly geometric mean of a series
def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


#load in the data
df_AGG = pandas.read_excel('Project_Data_Part1.xlsx', sheetname='AGG') #AGG adj close data
df_TBill = pandas.read_excel('Project_Data_Part1.xlsx', sheetname='13W Treasury') #13WTreasury r (%)
df_RNR = pandas.read_excel('RNR-2.xlsx') #RNR data
df_RNR = df_RNR['Adj Close'].values[102:270] #since the last row is a day of the month other than 1st
df_AGG = df_AGG['Adjusted Close'].values
df_TBill = df_TBill['Rate (annualized %)'].values


#now for the adj. close and r data, get the monthly returns (R) - ((r_t+1 - r_t) / 100) + 1
AGG_R = list() #return (R) of Feb 2004 to December 2017 - AGG
TBill_R = list() #return (R) of Feb 2004 to December 2017 - TBill
RNR_R = list() #return (R) RNR
for i in range(len(df_AGG) - 1):
    R_AGG = float((df_AGG[i + 1] - df_AGG[i]) / 100) + 1
    AGG_R.append(R_AGG)
    R_TBill = float((df_TBill[i + 1] + 1) ** (1 / float(12))) #converting from annualized to monthly
    TBill_R.append(R_TBill)
    R_RNR = float((df_RNR[i + 1] - df_RNR[i]) / 100) + 1
    RNR_R.append(R_RNR)

#R's for TBills which would essentially be the returns at the beginning of each period
R_TBill_beg = list()
for i in range(len(df_TBill)):
    TBill_beg = float((df_TBill[i] + 1) ** (1 / float(12))) #converting from annualized to monthly
    R_TBill_beg.append(TBill_beg)

capital = 1.3e9 #initial capital (in Billions of $) - this is what we have in January 2004 * returns due to
#investment in TBills
yellow_mark = capital * 0.2
num_yellow = 0
num_red = 0
inverted_yield = 0
y = 1 #static overlay strategy of investment policy, we start with y = 2
#At every month, y*capital - long on AGG, y*capital - short on 13W TBills, the remaining capital
# (initial capital) - invest in T-Bills
#at each month beginning February, we will add the results of long and short and see how the previous
#T-Bill investment paid off - returns of assets - losses due to liabilities
capital_per_month = list() #the amount of capital available at the beginning of each month
asset_investments = list() #beginning of each month how much money is invested - long
asset_investments_rnr = list() #beginning of each month how much money is invested - long rnr
liabilities_shorted = list() #beginning of each month how much money is shorted
net_profit_list = list()
capital_returns_per_month = list() #Total Returns (R) per month -
# beginning from February 2004 - 167 entries
diff = float(0.0) #amount to be recapitalized by in the event of a red card
for i in range(len(R_TBill_beg)):
    if i == 0: #for the very first period it will be just return from T-Bills
        asset = y * capital
        asset_investments.append(asset)
        asset_investments_rnr.append(asset)
        liability = 2 * y * capital
        liabilities_shorted.append(liability)
        capital = capital * R_TBill_beg[0]
        capital_per_month.append(capital)
    else: #take in the assets - liabilities from the pervious month which would add to capital flows this month
        assets_returns_prev_year = asset_investments[i - 1] * AGG_R[i - 1] + \
                                   asset_investments_rnr[i - 1] * RNR_R[i - 1]
        #assets_returns_prev_year = asset_investments_tlt[i - 1] * TLT_R[i - 1]
        liabilities_prev_year = liabilities_shorted[i - 1] * TBill_R[i - 1]
        net_profit = assets_returns_prev_year - liabilities_prev_year
        net_profit_list.append(net_profit)
        prev_month_capital = capital
        capital = (capital + net_profit) * R_TBill_beg[i] #this is the capital for beginning of a new period
        R_this_month = float(capital / prev_month_capital) #formula for R
        spread = spread = (AGG_R[i - 1] + RNR_R[i - 1]) - TBill_R[i - 1] #the spread
        if spread < 0:
            inverted_yield = int(inverted_yield + 1)
        #asset and liability allocation for the next month
        asset = y * capital
        asset_investments.append(asset)
        asset_investments_rnr.append(asset)
        liability = 2 * y * capital
        liabilities_shorted.append(liability)
        capital_per_month.append(capital) #capital for this month
        capital_returns_per_month.append(R_this_month)
    if capital < 0:  # we will have to recapitalize the bank
        # print('Red Flag.')
        num_red = int(num_red + 1)
        diff = float((-1 * capital + 1.3e9))

    elif capital < yellow_mark:
        # print('Yellow Flag.')
        num_yellow = int(num_yellow + 1)

#print(capital_per_month)
print('Number of red cards = %d ' % num_red)
print('Number of yellow cards = %d ' % num_yellow)
print('Number of inverted yields = %d ' % inverted_yield)
#Calculate reward and risk measures
#find the geometric returns, monthly then convert to annualized
annual_geometric_return_R = math.pow(geo_mean(capital_returns_per_month), 12)
print('Annual geometric return (R) = %f ' % annual_geometric_return_R)
#annualized volatility of returns
annual_volatility = 12 * math.pow(np.std(np.array(capital_returns_per_month)), 2)
print('Annualized volatility = %f' % annual_volatility)
#Sharpe Ratio
'''all_TBills_R = list()
val = float((df_TBill[0] ** (1 / float(12))) + 1) #converting from annualized to monthly - for 1st month
all_TBills_R.append(val)
for i in range(len(TBill_R)):
    all_TBills_R.append(TBill_R[i])'''
all_TBills_R = list()
for i in range(len(df_TBill)):
    val = float((df_TBill[i] + 1) ** (1 / float(12)))  # converting from annualized to monthly - for each month
    all_TBills_R.append(val)
#print(all_TBills_R)
risk_free_R = math.pow(geo_mean(all_TBills_R), 12) #the risk free rate is the annualized avg of T-Bills
print('Risk Free rate of return = %f' % risk_free_R)
Sharpe_Ratio = float((annual_geometric_return_R - risk_free_R) / math.sqrt(annual_volatility))
print('Sharpe Ratio = %f' % Sharpe_Ratio)
#Sortino Ratio
#first we have to calculate the downside variance wrt risk_free_R
N = len(capital_returns_per_month)

risk_free_R_monthly = geo_mean(all_TBills_R) #monthly avg risk free return
#print(risk_free_R_monthly)
downsides = list()
for i in range(N):
    comp = risk_free_R_monthly - capital_returns_per_month[i] #using monthly avg risk free return
    if comp > 0:
        downsides.append(comp)
    else:
        downsides.append(0)
downside_variance = 12 * math.pow(np.std(np.array(downsides)), 2) #annualized from mothly
print('Annualized downside variance = %f' % downside_variance)
Sortino_Ratio = float((annual_geometric_return_R - risk_free_R) / math.sqrt(downside_variance))
print('Sortino Ratio = %f' % Sortino_Ratio)

#Maximum drawdown calculation
X = pandas.Series(capital_per_month) #pandas series for max drawdown calculation
#print(X)
def create_drawdowns(equity_curve):
    """
    Calculate the largest peak-to-trough drawdown of the equity curve
    as well as the duration of the drawdown. Requires that the
    equity returns is a pandas Series.

    Parameters:
    equity_curve - A pandas Series representing period net returns.

    Returns:
    drawdown, duration - Highest peak-to-trough drawdown and duration.
    """

    # Calculate the cumulative returns curve
    # and set up the High Water Mark
    # Then create the drawdown and duration series
    hwm = [0]
    eq_idx = equity_curve.index
    #print(eq_idx)
    drawdown = pandas.Series(index = eq_idx)
    duration = pandas.Series(index = eq_idx)

    # Loop over the index range
    for t in range(1, len(eq_idx)):
        cur_hwm = max(hwm[t-1], equity_curve[t])
        hwm.append(cur_hwm)
        drawdown[t]= hwm[t] - equity_curve[t]
        duration[t]= 0 if drawdown[t] == 0 else duration[t-1] + 1
    '''print(drawdown)
    print(duration)'''
    return drawdown.max(), duration.max()
ans = create_drawdowns(X)
#print(ans)
MDD = ans[0]
print('Maximum DrawDown = %f' % MDD)

#VaR at h = 0.01 (1%)
h = 0.01
VaR_point = int(h * len(capital_per_month))
'''print(capital_per_month)
print(VaR_point)'''
capital_per_month_sorted = np.sort(capital_per_month)
VaR = capital_per_month_sorted[VaR_point]
print('VaR = %f' % VaR)
#CVaR at h = 0.01 (1%)
CVaR = 0
for i in range(VaR_point):
    CVaR = float(CVaR + capital_per_month_sorted[i])
CVaR = float(CVaR / VaR_point)
print('CVaR = %f' % CVaR)









