import numpy as np


def iter_priv_budget(eps_total, delta_prime, k, prec=0.0001):
    """
    Gives epsilon budget per iteration using composition from [Durfee and Rogers, NeurIPS 2019]
    """
    eps = eps_total/k
    t_min = 0
    while t_min <= eps_total:
        eps = eps+prec
        t_1 = k*eps
        t_2 = k*eps*((np.exp(eps)-1)/(np.exp(eps)+1))+eps*np.sqrt(2*k*np.log(1/delta_prime))
        t_3 = (k*eps**2)/2 + eps*np.sqrt(0.5*k*np.log(1/delta_prime))
        t_min = min(t_1,t_2,t_3)
    return eps-prec


def set_kbar_prob(sorted_data, k, d, delta_ld, eps_it):
    """
    Sets probability to select kbar for LD using optimization from [Durfee and Rogers, NeurIPS 2019] with budget eps_it
    We use EM formulation instead of Gumbel noise
    """
    thresh = sorted_data[(k-1):(d-1)] + np.log(np.array(range(k, d, 1))/delta_ld)/eps_it
    prob = np.exp(eps_it*(thresh - max(thresh)*0.95))/np.sum(np.exp(eps_it*(thresh - max(thresh)*0.95)))
    prob = prob/np.sum(prob)
    return prob

    
def topk_via_ld(sorted_data, eps_it, k, delta, kbar):
    """
    Performs DP top-k selection with LD on sorted data (descending order, i.e. higher counts first).
    Returns indices of elements selected (bot is not returned).
    """
    noisy_data = sorted_data[:kbar] + np.random.gumbel(0, 1/eps_it, kbar)
    bot = sorted_data[kbar] + 1 + np.log(kbar/delta)/eps_it + np.random.gumbel(0, 1/eps_it, 1)

    noisy_sort_order = noisy_data.argsort()[::-1]
    indices = np.array(range(kbar))[noisy_sort_order]
    noisy_data = noisy_data[noisy_sort_order]

    indices_sel = indices[noisy_data > bot]
    output = indices_sel[:k]

    return output

