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
from scipy.optimize import *

#function to calculate monthly geometric mean of a series
def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


#load in the data
df_AGG = pandas.read_excel('Project_Data_Part1.xlsx', sheetname='AGG') #AGG adj close data
df_TLT = pandas.read_excel('TLT.xlsx') #TLT data
df_TBill = pandas.read_excel('Project_Data_Part1.xlsx', sheetname='13W Treasury') #13WTreasury r (%)

df_AGG = df_AGG['Adjusted Close'].values
df_TLT = df_TLT['Adj Close'].values[18:186] #since we want only from Jan '04 - Dec '17
df_TBill = df_TBill['Rate (annualized %)'].values


#now for the adj. close and r data, get the monthly returns (R) - ((r_t+1 - r_t) / 100) + 1
AGG_R = list() #return (R) of Feb 2004 to December 2017 - AGG
TLT_R = list() #return (R) of Feb 2004 to December 2017 - TLT
TBill_R = list() #return (R) of Feb 2004 to December 2017 - TBill
for i in range(len(df_AGG) - 1):
    R_AGG = float((df_AGG[i + 1] - df_AGG[i]) / 100) + 1
    AGG_R.append(R_AGG)
    R_TLT = float((df_TLT[i + 1] - df_TLT[i]) / 100) + 1
    TLT_R.append(R_TLT)
    R_TBill = float((df_TBill[i + 1] + 1) ** (1 / float(12))) #converting from annualized to monthly
    TBill_R.append(R_TBill)

#R's for TBills which would essentially be the returns at the beginning of each period
R_TBill_beg = list()
for i in range(len(df_TBill)):
    TBill_beg = float((df_TBill[i] + 1) ** (1 / float(12))) #converting from annualized to monthly
    R_TBill_beg.append(TBill_beg)

ys = np.linspace(0.0, 5.0, num=1000, endpoint=True) #grid search spacings - for the sake of practicality,
#we are considering 0 to 5 with both endpoints included as the range of leverage
#print(ys)
#now, since we are performing gridsearch, for each of the ys we will see what happens and pick the best
#y based on if they fulfill the constraints

#CVaR constraint
def CVaR_constraint(capital_per_month1):
    h = 0.01  # tolerance level is 1%
    #print(len(capital_per_month1))
    VaR_point = int(h * len(capital_per_month1))
    capital_per_month_sorted = np.sort(capital_per_month1)
    # CVaR at h = 0.01 (1%)
    CVaR = 0
    for i in range(VaR_point):
        CVaR = float(CVaR + capital_per_month_sorted[i])
    # print(VaR_point)
    CVaR = float(CVaR / VaR_point)
    return CVaR

y = 0.0 #initialization of the optimization variable
max_geometric_R = 0.0 #this will be the optimizer
min_CVaR = 0.0 #best CVaR so far
for j in range(len(ys)): #basically for each potential y within ys
    y_i = ys[j]
    capital_per_month = list()  # the amount of capital available at the beginning of each month
    asset_investments_tlt = list()  # beginning of each month how much money is invested - long
    liabilities_shorted = list()  # beginning of each month how much money is shorted
    net_profit_list = list()
    capital_returns_per_month = list()  # Total Returns (R) per month -
    capital = 1.3e9  # initial capital (in Billions of $) - this is what we have in January 2004 * returns due to
    for i in range(len(R_TBill_beg)):
        if i == 0:  # for the very first period it will be just return from T-Bills
            asset = y_i * capital
            asset_investments_tlt.append(asset)
            liability = y_i * capital
            liabilities_shorted.append(liability)
            capital = capital * R_TBill_beg[0]
            capital_per_month.append(capital)
            if capital < 0: #in the event of a red flag, immediately go to the next y
                continue
        else:  # take in the assets - liabilities from the pervious month which would add to capital flows this month
            assets_returns_prev_year = asset_investments_tlt[i - 1] * TLT_R[i - 1]
            liabilities_prev_year = liabilities_shorted[i - 1] * TBill_R[i - 1]
            net_profit = assets_returns_prev_year - liabilities_prev_year
            net_profit_list.append(net_profit)
            prev_month_capital = capital
            capital = (capital + net_profit) * R_TBill_beg[i]  # this is the capital for beginning of a new period
            R_this_month = float(capital / prev_month_capital)  # formula for R
            if capital < 0: #in the event of a red flag, immediately go to the next y
                continue

            # asset and liability allocation for the next month
            asset = y_i * capital
            asset_investments_tlt.append(asset)
            liability = y_i * capital
            liabilities_shorted.append(liability)
            capital_per_month.append(capital)  # capital for this month
            capital_returns_per_month.append(R_this_month)

    # now calculate the geometric mean of the capital returns
    annual_geometric_return_R = math.pow(geo_mean(capital_returns_per_month), 12)
    CVaR = CVaR_constraint(capital_per_month) #CVaR - check if it is <= $200 million
    #update for decision variable whenever we have a higher return and CVaR constraint is satisfied
    if (annual_geometric_return_R >= max_geometric_R and CVaR <= 2e8):
        y = y_i
        max_geometric_R = annual_geometric_return_R
        min_CVaR = CVaR



#now, our y will store the optimized decision variable. Now we will just conduct the analysis of
#risk-reward as before
num_yellow = 0
num_red = 0
inverted_yield = 0
capital = 1.3e9
yellow_mark = capital * 0.2

#At every month, y*capital - long on AGG, y*capital - short on 13W TBills, the remaining capital
# (initial capital) - invest in T-Bills
#at each month beginning February, we will add the results of long and short and see how the previous
#T-Bill investment paid off - returns of assets - losses due to liabilities
capital_per_month = list() #the amount of capital available at the beginning of each month
asset_investments_tlt = list() #beginning of each month how much money is invested - long
liabilities_shorted = list() #beginning of each month how much money is shorted
net_profit_list = list()
capital_returns_per_month = list() #Total Returns (R) per month -

for i in range(len(R_TBill_beg)):
    if i == 0: #for the very first period it will be just return from T-Bills
        asset = y * capital
        asset_investments_tlt.append(asset)
        liability = y * capital
        liabilities_shorted.append(liability)
        capital = capital * R_TBill_beg[0]
        capital_per_month.append(capital)
    else: #take in the assets - liabilities from the pervious month which would add to capital flows this month
        assets_returns_prev_year = asset_investments_tlt[i - 1] * TLT_R[i - 1]
        liabilities_prev_year = liabilities_shorted[i - 1] * TBill_R[i - 1]
        net_profit = assets_returns_prev_year - liabilities_prev_year
        net_profit_list.append(net_profit)
        prev_month_capital = capital
        capital = (capital + net_profit) * R_TBill_beg[i] #this is the capital for beginning of a new period
        R_this_month = float(capital / prev_month_capital) #formula for R

        spread = TLT_R[i - 1] - TBill_R[i - 1]  # the spread
        if spread < 0:
            inverted_yield = int(inverted_yield + 1)

        #asset and liability allocation for the next month
        asset = y * capital
        asset_investments_tlt.append(asset)
        liability = y * capital
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

print('Value of optimized decision variable (y) = %f' % y)
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


