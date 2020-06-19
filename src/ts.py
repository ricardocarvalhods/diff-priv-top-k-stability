import numpy as np


def get_delta_q(delta, kbar, c):
    """
    Calculates the delta_q used on TS subject to delta_max <= \delta/kbar
    """
    delta_q = delta
    delta_max = kbar * ( ( 2*((delta_q)**c) + delta_q - c*(delta_q**c + 2*delta_q) )/(4 - 4*c) )
    while delta_max > delta:
        delta_q = delta_q*0.99
        delta_max = kbar * ( ( 2*((delta_q)**c) + delta_q - c*(delta_q**c + 2*delta_q) )/(4 - 4*c) )
    return delta_q



def topk_via_ts(sorted_data, eps2, eps1, k, delta_q, kbar, eps_em):
    """
    Performs DP top-k selection with TS on sorted data (descending order, i.e. higher counts first).
    Returns indices of elements selected (bot is not returned).
    """
    noisy_thresh = np.log(1/delta_q)/(eps2/2) + np.random.laplace(0, 1/eps1, 1)
    for i in list(range(k, kbar+1, 1)) + list(range(k-1,0,-1)): # Search order can be any/arbitrary
        qi = max(sorted_data[i] - sorted_data[i+1] - 1, 0)
        qi_n = qi + np.random.laplace(0, 1/(eps2/2), 1)
        if qi_n > noisy_thresh:
            if i > k:
                data_selec = sorted_data[:i]
                prob = np.exp(eps_em*data_selec)/np.sum(np.exp(eps_em*data_selec))
                output = np.random.choice(range(i), p=prob, size=k, replace=False)
            else:
                output = range(i)
            return output
    return []

