import numpy as np
from utils import plot_results, get_metrics
from ts import get_delta_q, topk_via_ts
from ld import iter_priv_budget, set_kbar_prob, topk_via_ld


def run_experiment(usr_counts, eps_list, k_list, delta, nr_trials, verbose=True, show_plot=False):
    """
    Compare TS and LD and plot tabulated (for verbose=True) results 
    together with comparison plot (for show_plot=True) for
    sorted data (descending, higher counts first).
    """
    for eps in eps_list:
        if verbose:
            print("\n- epsilon = ", eps)

        TS_P = []
        LD_P = []
        TS_S = []
        LD_S = []

        for k in k_list:

            if verbose:
                print("\nk =", k)

            ################################################################################
            # SETUP: TS
            kbar = k
            eps_em = 0
            eps_left = eps - eps_em
            eps1 = 0.37*eps_left
            eps2 = 0.63*eps_left
            c = 2/(eps2/eps1)
            delta_q = get_delta_q(delta, kbar, c)

            P = []
            S = []

            # RUN TRIALS
            for tr in range(nr_trials):
                output = topk_via_ts(usr_counts, eps2, eps1, k, delta_q, kbar, eps_em)
                m1, m2 = get_metrics(output, k, usr_counts)
                P.append(m1)
                S.append(m2)

            if verbose:
                print("TS =", f"P: {np.mean(P):.3f} | S: {np.mean(S):.3f}")

            TS_P.append(np.mean(P))
            TS_S.append(np.mean(S))

            ################################################################################
            # SETUP: LD
            delta_ld = delta/2 # half goes to delta_prime, then total is delta 
            eps_it = iter_priv_budget(eps, delta_ld, k+1, 0.0001)

            kbar_max = 5*k
            kbar_prob = set_kbar_prob(usr_counts, k, kbar_max, delta_ld, eps_it)
            kbar_set = np.array(range(k, kbar_max, 1))

            P = []
            S = []

            # RUN TRIALS
            for tr in range(nr_trials):
                kbar = int(np.random.choice(kbar_set, p=kbar_prob, size=1, replace=False))
                output = topk_via_ld(usr_counts, eps_it, k, delta, kbar)

                m1, m2 = get_metrics(output, k, usr_counts)
                P.append(m1)
                S.append(m2)

            if verbose:
                print("LD =", f"P: {np.mean(P):.3f} | S: {np.mean(S):.3f}")

            LD_P.append(np.mean(P))
            LD_S.append(np.mean(S))

        # Always shows plots
        if show_plot:
            #plot_results(LD_P, TS_P, "P", eps, nr_trials, k_list)
            plot_results(LD_S, TS_S, "S", eps, nr_trials, k_list)
