import math
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt 


r = 0.05 

# This project is inspired by Joshi The Concepts and Practice of Mathematical Finance
def main():
    S = 100
    K = 105
    sigma = 0.27
    T = 0.25
    bs_call_price = black_scholes_engine.call_price(S, K, T, sigma, r)
    def call_payoff(S, K):
        return max(S-K, 0)
    call_price_numerical = option_pricing(S, K, T, r, sigma, steps=50, payoff=call_payoff, paths=10000)
    print(f"Monte Carlo call price is {call_price_numerical}")
    print(f"Black Scholes call price is {bs_call_price}")
    # print(f"The price for the call option with sopt price {S}, strike price {K}, expires in {T*12} months is {call_price}")
    # x = brownian_motion(T, 50, 1)
    # y = geometric_brownian_motion(T, 50, S0=10, mu=0, sigma=0.8)
    # print(y)
    # plt.plot(y)
    # plt.show()


class black_scholes_engine:
    def black_scholes_forward_price(S, K, sigma, T, r):
        pass 


    def call_price(S, K, T, sigma, r):
        d_1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt (T))
        d_2 = d_1 - sigma * math.sqrt(T)
        K_pv = K * math.exp(-r * T)

        c = ss.norm.cdf(d_1) * S - ss.norm.cdf(d_2) * K_pv

        return c


def brownian_motion(T, n, sigma=1):
    x = np.linspace(0, T, n)
    x_priv = 0
    with np.nditer(x, op_flags=['readwrite']) as it:
        for w in it:
            w[...] = x_priv + np.random.normal(0, sigma)
            x_priv = w[...]
    return x 

def geometric_brownian_motion(T, steps, S0, mu, sigma):
    dt = T/steps

    x = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=steps))
    return S0 * x.cumprod()

# class monte_carlo:
#     def __init__(self, paths, steps):
#         self.paths = paths
#         self.steps = steps

#     def option_pricing(S0, K, T, r, steps, payoff, drift=0, sigma=1):
#         prices = []
#         for p in range(self.paths):
#             simulated_path = geometric_brownian_motion(T, steps, S0, K, drift=0, sigma=1)
#             simulated_price = simulated_path[-1]
#             price = payoff(simulated_price, K)
#             prices = price * math.exp(-r*T)
#         return sum(prices) / len(prices)


def option_pricing(S0, K, T, r, sigma, steps, payoff, paths=500):
    prices = []
    drift  = r - 0.5 * sigma ** 2
    for p in range(paths):
        simulated_path = geometric_brownian_motion(T, steps, S0, r, sigma)
        simulated_price = simulated_path[-1]
        price = payoff(simulated_price, K)
        price = price * math.exp(-r*T)
         
        prices.append(price)
    return sum(prices) / len(prices)

main()
