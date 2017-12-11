import numpy as np
from scipy.optimize import minimize
import math


def objective(u):
    alpha = 0.1
    g_u = (0.05 * (1 - math.exp(5 * (alpha - u))))
    tax = 0.20
    non_tax = 1 - tax
    interest = 0.09
    Y_0 = 10

    equation_a = non_tax + ((tax * alpha * u) / (g_u + alpha)) - u
    equation_b = (1 / (interest - g_u)) * (1 + interest)

    equation = -1 * equation_a * equation_b * Y_0
    return equation

u = np.zeros(1)
bound = [(0.0, 1.0)] * 1

sol = minimize(objective, u.flatten(), method='SLSQP', bounds = bound)
print(sol)