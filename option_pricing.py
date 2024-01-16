import math
import itertools
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt 


# This project is inspired by Joshi The Concepts and Practice of Mathematical Finance
def main():
    r = 0.05 
    S = 1
    K = 1.05
    sigma = 0.27
    T = 0.25
    bs_call_prices = []
    mc_call_prices = []
    grid = []
    strike_gird = np.linspace(0.1, 2*S, 10)
    sigma_gird = np.linspace(0.01, 1, 10)
    expiration_time_gird = np.linspace(0.1, 1, 52)

    for i in itertools.product(strike_gird, sigma_gird, expiration_time_gird):
        K = i[0]
        sigma = i[1]
        T = i[2]
        bsc = BlackScholesEngine.call_price(S, K, T, sigma, r)
        mcc = mc_option_pricing(S, K, T, sigma, r, steps=50, payoff=call_payoff, paths=1000)

        bs_call_prices.append(bsc)
        mc_call_prices.append(mcc)
    bs_call_prices = np.array(bs_call_prices)
    mc_call_prices = np.array(mc_call_prices)
    mse = ((bs_call_prices - mc_call_prices) ** 2).mean(axis=0)
    print(f"The mean squared error is {mse}")

    bs_call_price = BlackScholesEngine.call_price(S, K, T, sigma, r)
    mc_call_price = mc_option_pricing(S, K, T, sigma, r, steps=50, payoff=call_payoff, paths=1000)

    # print(f"The price for the call option with sopt price {S}, strike price {K}, expires in {T*12} months is {call_price}")
    # x = brownian_motion(T, 50, 1)
    # y = geometric_brownian_motion(T, 50, S0=10, mu=0, sigma=0.8)
    # print(y)
    # plt.plot(y)
    # plt.show()


def prices_of_range_of_inputs(S, pricing_method):
    strike = np.linespace(0, 2*S, S*0.1)
    sigma = np.linspace(0, 1, 0.1)
    expiration_time = np.linespace(0, 1, 1/52)



def call_payoff(S, K):
    return max(S-K, 0)

def put_payoff(S, K):
    return max(K-S, 0)

def forward_payoff(S, K):
    return (S-K)

def digital_call_payoff(S, K):
    return 1 if S-K>0 else 0

def digital_put_payoff(S, K):
    return 1 if S-K<0 else 0

class BlackScholesEngine:
    def forward_price(S, K, sigma, T, r):
        return (S - K) * math.exp(-r*T) 

    def call_price(S, K, T, sigma, r):
        d_1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt (T))
        d_2 = d_1 - sigma * math.sqrt(T)
        K_pv = K * math.exp(-r * T)

        c = ss.norm.cdf(d_1) * S - ss.norm.cdf(d_2) * K_pv
        return c
    
    def put_price(S, K, T, sigma, r):
        d_1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt (T))
        d_2 = d_1 - sigma * math.sqrt(T)
        K_pv = K * math.exp(-r * T)

        p = ss.norm.cdf(-d_2) * K_pv -  ss.norm.cdf(-d_1) * S
        return p
    
    def zero_coupon_bond_price(K, T, r):
        return 
    

class MonteCarloEngine:
    def __init__(self, mc_paths):
        self.mc_paths = mc_paths

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

def mc_option_pricing(S0, K, T, sigma, r, steps, payoff, paths=500):
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
