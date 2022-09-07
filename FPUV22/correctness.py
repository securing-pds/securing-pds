import itertools
import argparse
from tqdm import tqdm
from sage.all import RealField, exp, log, ceil, floor, round, oo, save, line
from tikz import TikzPlot


RRR = RealField(100) # have custom real field with high enough precision
PRF_EPS = RRR(2)**-128
ACCEPTABLE_ERR = 2**-128


class BF:

    @staticmethod
    def filter_bitlength(pp):
        m, k = pp
        return m

    @staticmethod
    def honest_pfp_ub(n, pp, exact=False, return_log=False):
        """ Bloom filter P[FP] honest upper bound. If `exact=True`, it computes the exact value of P[FP] instead.
        
            :param n:   number of inserted entries
            :param pp:  tuple of public parameters (m, k)
        """

        m, k = pp
        if exact:
            log_p = k *  log(RRR(1.) - exp(-RRR(n*k)/m), 2)
        else:
            log_p = k *  log(RRR(1.) - exp(-RRR((n+0.5)*k)/(m-1)), 2)
        if return_log:
            return log_p
        return RRR(2)**log_p

    @staticmethod
    def pp_from_hon_fpp_n_queries(target_hon_fpp, query_budget):
        """ Generates recommended public parameters for Bloom filters given a query budget and a target maximum P[FP]
            following the established "honest setting" analysis.

            Given the number of initially inserted elements `n` and the total number of update queries `q_u`,
            it performs the analysis assuming `n + q_u` elements being inserted.
        """

        n, q_u, q_t, q_v = query_budget
        tot_n = n + q_u
        m = ceil((RRR(tot_n) * log(RRR(target_hon_fpp))) / log(1 / RRR(2)**log(2)))
        k = floor((m / RRR(tot_n)) * log(2))
        pp = (m, k)
        return pp

    @staticmethod
    def admissible_params(m_range, k_range, query_budget=None):
        """ Returns generator of BF parameters ordered from the cheapest to the more expensive to run under a metric where it's always better to have a smaller state at the cost of a larger number of PRF evaluations.

            EXAMPLE:
                >>> m_range = (2**log_m for log_m in range(10, 24))
                >>> k_range = (2**log_k for log_k in range(0, 10))
                >>> admissible_params = BF.admissible_params(m_range, k_range)
        """
        params = itertools.product(m_range, k_range)
        return params

    @staticmethod
    def cost(pp, strict=True):
        """ Given public parameters, we assign them a runtime cost per query.
        """
        m, k = pp

        if strict:
            raise NotImplementedError("Should consider what makes sense, and come up with an algorithm to sort admissible_params by their cost")

        return m

        # assuming k < m, the following gives absolute priority to m, with k coming later
        # return m**2 + k


class CF:

    @staticmethod
    def load_factor(s, deletions=True):
        if deletions:
            alpha = {
                1: .5,
                2: .84,
                4: .95,
                8: .98
            }
        else:
            raise ValueError("Load factors for insertion-only not yet covered.")

        return alpha[s]

    @staticmethod
    def lambda_T_lb(target_hon_fpp, s):
        return ceil(log(2*s, 2) - log(target_hon_fpp, 2))

    @staticmethod
    def space_cost(target_hon_fpp, s, deletions=True):
        return CF.lambda_T_lb(target_hon_fpp, s)/CF.load_factor(s, deletions=deletions)

    @staticmethod
    def honest_pfp_ub(n, pp, **kwargs):
        """ Cuckoo filter P[FP] average upper bound.
        
            :param n:   number of inserted entries
            :param pp:  tuple of public parameters (s, lambda_I, lambda_T, num)
        """

        s, lambda_I, lambda_T, num = pp
        p = RRR(1) - (RRR(1) - RRR(2)**(-lambda_T))**(2*s+1)
        return p

    @staticmethod
    def filter_bitlength(pp):
        s, lambda_I, lambda_T, num = pp
        return 2**lambda_I * s * lambda_T

    @staticmethod
    def pp_from_hon_fpp_n_queries(target_hon_fpp, query_budget, deletions=True):
        """ Generates recommended public parameters for Cuckoo filters given a query budget and a target maximum P[FP]
            following the established "honest setting" analysis.

            Given the number of initially inserted elements `n` and the total number of update queries `q_u`,
            it performs the analysis assuming `n + q_u` elements being inserted.
        """
        b = 2 # "buckets searched per negative lookup"
        num = 500 # NOTE: only experimental analysis at the base of this number
        n, q_u, q_t, q_v = query_budget
        # NOTE: Morton paper claims better results by marking some slots?
        cheapest_pp = None
        cheapest_pp_cost = oo
        for s in [1,2,4,8]:
            alpha = CF.load_factor(s, deletions=deletions) # NOTE: expected load factor, experimentally derived assuming num = 500
            # lambda_T = ceil(log((b * s + 1) / RRR(target_hon_fpp), 2)) # approximation
            lambda_T = ceil(-log(1-(1-RRR(target_hon_fpp))**(RRR(1)/(s*b+1)),2)) # exact inversion
            lambda_I = ceil(log(RRR(n + q_u)/(s * alpha),2))
            pp = (s, lambda_I, lambda_T, num)
            pp_cost = CF.filter_bitlength(pp)
            if pp_cost < cheapest_pp_cost:
                cheapest_pp_cost = pp_cost
                cheapest_pp = pp
        return cheapest_pp

    @staticmethod
    def admissible_params(s_range, lambda_I_range, lambda_T_range, query_budget=None, adversarial_query_budget=False, deletions=True, max_size=oo):
        """ Returns generator of BF parameters ordered from the cheapest to the more expensive to run under a metric where it's always better to have a smaller state at the cost of a larger number of PRF evaluations.
        """
        num = 500

        n, q_u, q_t, q_v = query_budget
        # assumes: bigger runtime cost is lambda_I, lambda_T comes later (ie reduce storage at any cost)
        params = itertools.product(s_range, lambda_I_range, lambda_T_range)
        for s, lambda_I, lambda_T in params:
            # compute minimal lambda_I to support n+q_u insertions
            if adversarial_query_budget:
                q = n + q_u + q_t
            else:
                q = n + q_u
            alpha = CF.load_factor(s, deletions=deletions) # expected load factor
            min_lambda_I = ceil(log(RRR(n + q_u)/(s * alpha),2))
            size = s * 2**lambda_I * lambda_T
            if lambda_I >= min_lambda_I and size < max_size:
                yield (s, lambda_I, lambda_T, num)

    @staticmethod
    def cost(pp, strict=True):
        """ Given public parameters, we assign them a runtime cost per query.
        """
        s, lambda_I, lambda_T, num = pp

        if strict:
            raise NotImplementedError("Should consider what makes sense, and come up with an algorithm to sort admissible_params by their cost")

        # storage
        return 2**lambda_I * s * lambda_T


def AdvRoI(amq, query_budget, pp, prf_eps=PRF_EPS, alpha=1, beta=1):
    """ Computation for the distinguishing advantage between Real and Ideal in Theorem 1.
        It accounts for the caveat regarding immutable filters.

        :param amq:             AMQ-PDS namespace
        :param query_budget:    tuple with number of allowed queries (n, q_u, q_t, q_v)
        :param pp:              tuple of public parameters
        :param eps:             assumed PRF advantage
        :param alpha:           number of calls made to F in up
        :param beta:            number of calls made to F in qry
    """

    # obtain query budget
    n, q_u, q_t, q_v = query_budget

    # account for PRF advantage
    amq_adv = prf_eps

    # immutable setting
    if q_u <= 0:
        return amq_adv

    # upper bound P[E]
    amq_adv += 2 * q_t * amq.honest_pfp_ub(n+q_u, pp)

    return amq_adv


def pp_from_adv_fpp_n_queries(amq, target_pfp, query_budget, prf_eps=PRF_EPS, **kwargs):
    """ Generates recommended public parameters for a provided AMQ given a PRF distinguishing advantage,
        a query budget and a target maximum P[FP], following our analysis of the "adversarial correctness" setting.
        That is, parameters using
            P[adv FP | n + q_u insertions] <= AdvRoI + P[hon FP | n + q_u insertions].
                                            = eps + (2 q_t + 1) * P[hon FP | n + q_u insertions].

        To compute the parameters it first recovers the required P[hon FP | n + q_u insertions], and then
        computes the public parameters from the established "honest" setting analysis for the respective AMQ.

        Given the number of initially inserted elements `n` and the total number of update queries `q_u`,
        it performs the analysis assuming `n + q_u` elements being inserted.
    """
    n, q_u, q_t, q_v = query_budget
    tot_n = n + q_u
    target_hon_fpp = (target_pfp - prf_eps) / (2 * RRR(q_t) + 1)
    return amq.pp_from_hon_fpp_n_queries(target_hon_fpp, query_budget, **kwargs)


def guarantee_from_params(query_budget, amq, pp, explore_best=True, tradeoff_plot=None, verbose=False):
    """ This function takes a query budget, an AMQ class and a set of parameters for the AMQ,
        and computes the honest and adversarial correctness guarantees we can provide.

        If `explore_best=True`, it will define q = q_u + q_t, and will look for the choice of
        (q_u, q_t) that maximises the P[FP].

        :params query_budget:   query budget
        :params amq:            AMQ's class
        :params pp:             tuple of AMQ's public parameters

        :returns tuple:         (honest_pfp, adversarial_pfp)

        EXAMPLE:
            >>> query_budget = (n, q_u, q_t, q_v) = (10, 100, 100, 1)
            >>> amq = BF
            >>> PP = (m, k) = (1024, 3)
            >>> guarantee_from_params(query_budget, amq, pp)
            ...
            >>> amq = CF
            >>> pp = (s, lambda_I, lambda_T, num) = (4, 10, 20, 500)
            >>> guarantee_from_params(query_budget, amq, pp)
            ...
    """

    n, q_u, q_t, q_v = query_budget
    if explore_best:
        q = q_u + q_t
        log_q = ceil(log(q, 2))

        fun_pfp = []
        max_pt = (-1, {"hon": -oo, "adv": -oo})
        for new_log_qt in range(log_q+1):
            new_q_t = 2**new_log_qt
            new_q_u = q - new_q_t
            new_query_budget = (n, new_q_u, new_q_t, q_v)

            pfp = guarantee_from_params(new_query_budget, amq, pp, explore_best=False)
            fun_pfp.append((new_log_qt, log(pfp["adv"],2)))
            if pfp["adv"] > max_pt[1]["adv"]:
                max_pt = (new_log_qt, pfp)

        if isinstance(tradeoff_plot, str):
            g = line(fun_pfp, title="$\\log \\Pr[Adv FP]$ vs $\\log(q_t)$", axes_labels=["$\\log(q_t)$", "$\\log \\Pr[Adv FP]$"])
            save(g, tradeoff_plot)
        return max_pt[1]
    else:
        honest_pfp = amq.honest_pfp_ub(n, pp)
        adv_roi = AdvRoI(amq, query_budget, pp)
        adversarial_pfp = honest_pfp + adv_roi
        if verbose:
            print (pp, query_budget, adversarial_pfp)
        return {"hon": float(honest_pfp), "adv": float(adversarial_pfp)}


def params_from_guarantee(query_budget, amq, target_pfp, prf_eps=PRF_EPS, explore_best=True, admissible_params=None, verbose=True, **kwargs):
    """ Generates recommended public parameters for a provided AMQ given a PRF distinguishing advantage,
        a query budget and a target maximum P[FP], following our analysis of the "adversarial correctness" setting.

        This function wraps pp_from_adv_fpp_n_queries, and adds extra sanity checks.

        EXAMPLE:
            >>> query_budget = (n, q_u, q_t, q_v) = (2**32-1, 1, 2**32, 1)
            >>> amq = BF
            >>> target_pfp = 2**-17
            >>> params_from_guarantee(query_budget, amq, target_pfp)
            ...
            >>> amq = CF
            >>> params_from_guarantee(query_budget, amq, target_pfp)
            ...
    """

    # TODO: this function should also do a query tradeoff between q_u and q_t to get best parameters given total number of queries.

    n, q_u, q_t, q_v = query_budget

    if not explore_best:
        tot_n = n + q_u

        # honest params sanity check
        honest_pp = amq.pp_from_hon_fpp_n_queries(target_pfp, query_budget)
        honest_pfp = amq.honest_pfp_ub(tot_n, honest_pp, exact=True)
        if verbose:
            print("target", log(target_pfp, 2).n())
            print("obatined honest", log(honest_pfp, 2))
        assert(honest_pfp < target_pfp)

        adversarial_pp = pp_from_adv_fpp_n_queries(amq, target_pfp, query_budget, prf_eps=prf_eps, **kwargs)
        adversarial_pfp = prf_eps + (2 * q_t + 1) * amq.honest_pfp_ub(tot_n, adversarial_pp, exact=True)
        if verbose:
            print("obtained adversarial", log(adversarial_pfp, 2))
        assert(adversarial_pfp < target_pfp)

        return adversarial_pp
    else:
        if not admissible_params:
            raise ValueError("If searching for guarantee over all possible query strategies, you need to provide a generator of admissible parameter sets from the cheapest to the most expensive to run.")
        # find parameters satisfying any equivalent query strategies by the adversary
        q = q_u + q_t
        log_q = ceil(log(q, 2))
        if ceil(log(q, 2)) == floor(log(q, 2)):
            # passed q = power of two
            log_q += 1

        cheapest_pp = None
        cheapest_pp_cost = oo
        for pp in admissible_params:
            satisfied = True
            new_log_qt = 0
            while satisfied and new_log_qt < log_q:
                new_q_t = 2**new_log_qt
                new_q_u = q - new_q_t
                new_query_budget = (n, new_q_u, new_q_t, 1)
                assert(new_q_u >= 0)
                assert(new_q_t >= 0)

                pfp = guarantee_from_params(new_query_budget, amq, pp, explore_best=False)
                if pfp["adv"] > target_pfp:
                    satisfied = False
                    break

                # next iteration
                new_log_qt += 1

            if satisfied:
                # found solution
                pp_cost = amq.filter_bitlength(pp)
                if pp_cost < cheapest_pp_cost:
                    cheapest_pp_cost = pp_cost
                    cheapest_pp = pp

        if not cheapest_pp:
            # print("ERROR: no good parameters, try a higher target P[FP]")
            pass

        return cheapest_pp


def example():
    """ Example use of this script.
    """
    print("Guarantee from params")
    query_budget = (n, q_u, q_t, q_v) = (10, 100, 100, 1)

    amq = BF
    pp = (m, k) = (1024, 3)
    pfp = guarantee_from_params(query_budget, amq, pp, tradeoff_plot="bf.png")
    print("BF", pp, pfp, log(pfp["adv"],2))

    amq = CF
    pp = (s, lambda_I, lambda_T, num) = (4, 10, 20, 500)
    pfp = guarantee_from_params(query_budget, amq, pp, tradeoff_plot="cf.png")
    print("CF", pp, pfp, log(pfp["adv"],2))

    print()

    print("Params from guarantee")

    amq = BF
    query_budget = (n, q_u, q_t, q_v) = (10, 2**8, 1, 1)
    target_pfp = 2**-20
    m_range = (2**log_m for log_m in range(10, 24))
    k_range = (2**log_k for log_k in range(1, 10))
    admissible_params = amq.admissible_params(m_range, k_range)
    pp = params_from_guarantee(query_budget, amq, target_pfp, admissible_params=admissible_params)
    print("BF", float(log(target_pfp, 2)), pp)

    amq = CF
    query_budget = (n, q_u, q_t, q_v) = (100, 10, 1, 1)
    target_pfp = 2**-20
    s_range = [1, 2, 4, 8]
    lamda_I_range = range(2, 10)
    lambda_T_range = range(1, 32+1)
    admissible_params = amq.admissible_params(s_range, lamda_I_range, lambda_T_range, query_budget=query_budget)
    pp = params_from_guarantee(query_budget, amq, target_pfp, admissible_params=admissible_params)
    print("CF", float(log(target_pfp, 2)), "~", float(log(guarantee_from_params(query_budget, amq, pp)['adv'],2)), pp)
    _s, _lambda_I = pp[:2]
    print(f"n: {n}, entries: {_s * 2**_lambda_I}")


def secure_instances(bloom=True, cuckoo=True):

    if cuckoo:
        # CF filters: pick n, q, and check what PFP can be achieved as the size grows (for various lambda_T vs b tradeoffs)
        amq = CF

        sm_sm_xticks = [0, 2048, 4096, 6144, 8192, 10240, 12288, 16384]
        sm_sm_xticklabels = ["$0$ B", "$256$ B", "$512$ B", "$768$ B", "$1024$ B", "$1280$ B", "$1536$ B", "$2048$ B"]

        sm_bi_xticks = [8589934592,25769803776,42949672960,60129542144,77309411328,94489280512,111669149696,128849018880,146028888064,163208757248]
        sm_bi_xtickslabels = ["$1$ GiB", "$3$ GiB", "$5$ GiB", "$7$ GiB", "$9$ GiB", "$11$ GiB", "$13$ GiB", "$15$ GiB", "$17$ GiB", "$19$ GiB"]

        bi_bi_xticks = [8589934592,25769803776,42949672960,60129542144,77309411328,94489280512,111669149696,128849018880,146028888064,163208757248]
        bi_bi_xtickslabels = ["$1$ GiB", "$3$ GiB", "$5$ GiB", "$7$ GiB", "$9$ GiB", "$11$ GiB", "$13$ GiB", "$15$ GiB", "$17$ GiB", "$19$ GiB"]

        settings = [
            # (fn, log_q, log_n, min_size, max_size, skip_size, xticks, xticklabels)
            ("small_n_small_q", 8, 7,
                8,
                int(8 * 1024 * 1.5),
                128,
                sm_sm_xticks, sm_sm_xticklabels),
            ("small_n_big_q", int(log(10**9, 2)) + 1, 7,
                int(8 * 1024 * 1024 * 1024 * 5),
                int(8 * 1024 * 1024 * 1024 * 15),
                8 * 1024 * 1024 * 128,
                sm_bi_xticks, sm_bi_xtickslabels),
            ("big_n_big_q", int(log(10**9, 2)) + 1, int(log(10**9, 2)),
                int(8 * 1024 * 1024 * 1024 * 5),
                int(8 * 1024 * 1024 * 1024 * 15),
                8 * 1024 * 1024 * 128,
                bi_bi_xticks, bi_bi_xtickslabels)
        ]

        for setting_params in settings:

            print(setting_params[:-2])

            fn, log_q, log_n, min_size, max_size, skip_size, xticks, xticklabels = setting_params
            size_range = range(min_size, max_size, skip_size)

            range_s = [1, 2, 4, 8]
            num = 500
            q = 2**log_q
            n = 2**log_n
            lambda_T_min, lambda_T_max, lambda_T_skip = 6, 64, 1
            lambda_I_min, lambda_I_max, lambda_I_skip = 1, 32, 1

            min_log_target_pfp = -20
            g = line([])
            gt_2 = TikzPlot(legend_pos="north east")

            colors = (x for x in ['red', 'blue', 'orange', 'green', 'purple', 'black', 'brown', 'violet'])
            for s in range_s:
                color = next(colors)

                honest_log_pfp_vs_size = []
                log_pfp_vs_size = []

                # pick params using honest setting
                for size in tqdm(range(min_size//4, max_size, skip_size)):
                    # pfp for honest setting
                    # size = s * 2**lambda_I * lambda_T
                    # need to satisfy s * 2**lambda_I * alpha > n + q
                    #         size =  s * 2**lambda_I * lambda_T > lambda_T * (n+q)/alpha
                    min_log_honest_pfp = oo
                    min_log_honest_pfp_pp = None
                    range_lambda_T = range(lambda_T_min, lambda_T_max+1, lambda_T_skip)
                    for lambda_T in range_lambda_T:
                        lambda_I = floor(log(size / (s * lambda_T), 2))
                        if lambda_I < lambda_I_min or lambda_I > lambda_I_max:
                            continue
                        if s * 2**lambda_I * CF.load_factor(s) < n + q:
                            continue
                        _pp = (s, lambda_I, lambda_T, num)
                        _log_honest_pfp = float(log(amq.honest_pfp_ub(n+q, _pp),2))
                        if _log_honest_pfp < min_log_honest_pfp:
                            min_log_honest_pfp = _log_honest_pfp
                            min_log_honest_pfp_pp = _pp
                    log_honest_pfp = min_log_honest_pfp
                    if min_log_honest_pfp_pp:
                        honest_log_pfp_vs_size.append((size, log_honest_pfp))

                # pfp for adversarial setting
                for size in tqdm(size_range):
                    # size = s * 2**lambda_I * lambda_T
                    # need to satisfy s * 2**lambda_I * alpha > n + q
                    #         size =  s * 2**lambda_I * lambda_T > lambda_T * (n+q)/alpha
                    query_budget = (n, q_u, q_t, q_v) = (n, q-1, 1, 1)

                    found_pp = False
                    for log_target_pfp in (x/1 for x in range(1 * min_log_target_pfp, 1, 4)):
                        target_pfp = 2**log_target_pfp
                        # need to create admissible_params from scratch to avoid generator depletion
                        # generate parameters that could store n+q_u elements
                        range_lambda_T = range(lambda_T_min, lambda_T_max+1, lambda_T_skip)
                        range_lambda_I = range(lambda_I_min, lambda_I_max+1, lambda_I_skip)
                        # NOTE: admissible_params enforces "fitting n+q_u" elements but also storing within size
                        admissible_params = amq.admissible_params([s], range_lambda_I, range_lambda_T, query_budget=query_budget, max_size=size)
                        # return cheapest parameters that achieve target_pfp for all possible query budget balancing
                        pp = params_from_guarantee(query_budget, amq, target_pfp, admissible_params=admissible_params)
                        if pp:
                            print("CF", float(log(target_pfp, 2)), "~", float(log(guarantee_from_params(query_budget, amq, pp)['adv'],2)), pp)
                            found_pp = True
                            break
                    if found_pp:
                        _s, _lambda_I, _lambda_T = pp[:3]
                        _size = _s * 2**_lambda_I * _lambda_T 
                        log_pfp_vs_size.append((size, log_target_pfp))
                        if log_target_pfp == min_log_target_pfp:
                            break

                legend_label = f"$s = {s}$"
                title = f"$\\log \\Pr [FP]$ vs size given $\\log(n)={float(log(n,2))}$, $\\log(q_u + q_t) = {float(log(q, 2))}$"
                axes_labels = ["size", "$\\log \\Pr {[FP]}$"]
                g += line(log_pfp_vs_size, color=color, legend_label=legend_label, title=title, axes_labels=axes_labels)
                g += line(honest_log_pfp_vs_size, color=color, linestyle="--")
                gt_2.line(log_pfp_vs_size, color=color, legend_label=legend_label, axes_labels=axes_labels)
                gt_2.line(honest_log_pfp_vs_size, color=color, linestyle="--")
            # save(g, f"./plots/cf_log_pfp_vs_size_{fn}.png")
            gt_2.save(f"./plots/cf_log_pfp_vs_size_{fn}.tikz", xticks=xticks, xticklabels=xticklabels, ymin=min_log_target_pfp-2) #, xmax=int(xticks[-1]*1.2))

    if bloom:
        # Bloom filters: pick n, q, and check what PFP can be achieved as m grows (for various k)
        amq = BF

        sm_sm_xticks = [0, 2048, 4096, 6144, 8192, 10240, 12288, 16384]
        sm_sm_xticklabels = ["$0$ B", "$256$ B", "$512$ B", "$768$ B", "$1024$ B", "$1280$ B", "$1536$ B", "$2048$ B"]

        sm_bi_xticks = [8589934592,25769803776,42949672960,60129542144,77309411328,94489280512,111669149696,128849018880,146028888064,163208757248]
        sm_bi_xtickslabels = ["$1$ GiB", "$3$ GiB", "$5$ GiB", "$7$ GiB", "$9$ GiB", "$11$ GiB", "$13$ GiB", "$15$ GiB", "$17$ GiB", "$19$ GiB"]

        bi_bi_xticks = [8589934592,25769803776,42949672960,60129542144,77309411328,94489280512,111669149696,128849018880,146028888064,163208757248]
        bi_bi_xtickslabels = ["$1$ GiB", "$3$ GiB", "$5$ GiB", "$7$ GiB", "$9$ GiB", "$11$ GiB", "$13$ GiB", "$15$ GiB", "$17$ GiB", "$19$ GiB"]

        settings = [
            # (fn, log_q, log_n, min_m, max_m, skip_m, xticks, xticklabels)
            ("small_n_small_q", 8, 7, 8, int(8 * 1024 * 1.5), 8, sm_sm_xticks, sm_sm_xticklabels),
            ("small_n_big_q", int(log(10**9, 2)) + 1, 7, int(8 * 1024 * 1024 * 1024 * 5), int(8 * 1024 * 1024 * 1024 * 15), 8 * 1024 * 1024 * 10, sm_bi_xticks, sm_bi_xtickslabels),
            ("big_n_big_q", int(log(10**9, 2)) + 1, int(log(10**9, 2)), int(8 * 1024 * 1024 * 1024 * 5), int(8 * 1024 * 1024 * 1024 * 15), 8 * 1024 * 1024 * 10, bi_bi_xticks, bi_bi_xtickslabels)
        ]

        for setting_params in settings:

            print(setting_params[:-2])

            fn, log_q, log_n, min_m, max_m, skip_m, xticks, xticklabels = setting_params
            m_range = range(min_m, max_m, skip_m)

            range_set_k = range(8, 32+1, 4)
            q = 2**log_q
            n = 2**log_n

            min_log_target_pfp = -20
            g = line([])
            gt_2 = TikzPlot(legend_pos="north east")

            colors = (x for x in ['red', 'blue', 'orange', 'green', 'purple', 'black', 'brown', 'violet'])
            for set_k in range_set_k:
                color = next(colors)

                honest_log_pfp_vs_m = []
                log_pfp_vs_m = []

                for m in tqdm(range(min_m//4, max_m, skip_m)):
                    # pfp for honest setting
                    log_honest_pfp = amq.honest_pfp_ub(n+q, (m, set_k), return_log=True)
                    honest_log_pfp_vs_m.append((m, log_honest_pfp))

                for m in tqdm(m_range):
                    # pfp for adversarial setting
                    query_budget = (n, q_u, q_t, q_v) = (n, q-1, 1, 1)

                    found_pp = False
                    for log_target_pfp in (x/1 for x in range(1 * min_log_target_pfp, 1)):
                        target_pfp = 2**log_target_pfp
                        # need to create admissible_params from scratch to avoid generator depletion
                        if set_k:
                            k_range = [set_k]
                        else:
                            k_range = range(1, 65)

                        admissible_params = amq.admissible_params([m], k_range)
                        pp = params_from_guarantee(query_budget, amq, target_pfp, admissible_params=admissible_params)
                        if pp:
                            found_pp = True
                            break
                    if found_pp:
                        log_pfp_vs_m.append((pp[0], log_target_pfp))
                        if log_target_pfp == min_log_target_pfp:
                            break

                legend_label = f"$k = {set_k}$" if set_k else "best $k$"
                title = f"$\\log \\Pr [FP]$ vs $m$ given $\\log(n)={float(log(n,2))}$, $\\log(q_u + q_t) = {float(log(q, 2))}$"
                axes_labels = ["$m$", "$\\log \\Pr {[FP]}$"]
                g += line(log_pfp_vs_m, color=color, legend_label=legend_label, title=title, axes_labels=axes_labels)
                g += line(honest_log_pfp_vs_m, color=color, linestyle="--")
                gt_2.line(log_pfp_vs_m, color=color, legend_label=legend_label, axes_labels=axes_labels)
                gt_2.line(honest_log_pfp_vs_m, color=color, linestyle="--")
            # save(g, f"./plots/bf_log_pfp_vs_m_{fn}.png")
            gt_2.save(f"./plots/bf_log_pfp_vs_m_{fn}.tikz", xticks=xticks, xticklabels=xticklabels, ymin=min_log_target_pfp-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bloom', action='store_true')
    parser.add_argument('--cuckoo', action='store_true')
    args = parser.parse_args()
    # example()
    secure_instances(bloom=args.bloom, cuckoo=args.cuckoo)

