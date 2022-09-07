import argparse
import json
import tqdm
from cuckoo import CuckooFilter, StreamlinedCF, PRFWrappedCF, PRNG, Hash
from sage.all import parallel, mean, line, save


H = Hash(32)


instances = [
    {'m_len': 15, 'b_len': 2, 'kickmax': 500, 'tag_len_min': 3, 'tag_len_max': 20},
    {'m_len': 20, 'b_len': 2, 'kickmax': 500, 'tag_len_min': 3, 'tag_len_max': 20},
    {'m_len': 15, 'b_len': 3, 'kickmax': 500, 'tag_len_min': 4, 'tag_len_max': 20},
    {'m_len': 20, 'b_len': 3, 'kickmax': 500, 'tag_len_min': 4, 'tag_len_max': 20},
]


def singleThreadExperiment(CF, params, tag_len, t):
    m_len, b_len, kickmax = params['m_len'], params['b_len'], params['kickmax']
    _seed = json.dumps(params) + str(tag_len) + str(t)
    seed = int.from_bytes(H.eval(bytes(_seed, encoding="utf8")), 'big')
    prng = PRNG(seed)
    cf = CF(m_len, b_len, kickmax, tag_len, prng)
    x = 0
    cnt = 0
    enabled = True
    while enabled:
        # first look for an true-negative element
        dont_insert = True
        while dont_insert:
            x += 1
            dont_insert = cf.check(x)

        # then insert it
        enabled = cf.insert(x)
        cnt += 1
    # here x = number of insertions before up-disabling
    return cnt


def run_experiments(cftype, results_fn, trials=16, ncpus=4):
    if cftype == "original":
        CF = CuckooFilter
    elif cftype == "streamlined":
        CF = StreamlinedCF
    elif cftype == "prfwrapped":
        CF = PRFWrappedCF
    else:
        raise ValueError("Cuckoo filter `type` can only be `original`, `streamlined` or `prfwrapped`.")
    results = []
    for params in instances:
        m_len, b_len, kickmax, tag_len_min, tag_len_max = params['m_len'], params['b_len'], params['kickmax'], params['tag_len_min'], params['tag_len_max']
        inst_results = {
            "params": params,
            "res": []
        }

        for tag_len in tqdm.tqdm(range(tag_len_max, tag_len_min-1, -1)):
            res = []

            if ncpus > 1:
                @parallel(ncpus=ncpus)
                def parallelExperiment(CF, params, tag_len, t):
                    return singleThreadExperiment(CF, params, tag_len, t)

                out = list(parallelExperiment(
                    ((CF, params, tag_len, t) for t in range(trials))
                ))

                for entry in out:
                    res.append(entry[1])
            else:
                for t in range(trials):
                    x = singleThreadExperiment(params, tag_len, t)
                    res.append(x)

            inst_results['res'].append((tag_len, res))
        results.append(inst_results)
    json.dump(results, open(results_fn, 'w'))


def plot_results(results_fn, cftype, plots_dir="plots/", boxplot=True):
    if cftype == "original":
        CF = CuckooFilter
    elif cftype == "streamlined":
        CF = StreamlinedCF
    elif cftype == "prfwrapped":
        CF = PRFWrappedCF
    else:
        raise ValueError("Cuckoo filter `type` can only be `original`, `streamlined` or `prfwrapped`.")
    data = json.load(open(results_fn))
    for experiment in data:
        results = experiment['res']
        params = experiment['params']
        m_len = params['m_len']
        b_len = params['b_len']
        kickmax = params['kickmax']
        
        b = 2**b_len
        m = 2**m_len
        total_slots = b * m

        if boxplot:
            frame = {}
        else:
            data_points = []

        for entry in results:
            tag_len = entry[0]
            insertions = entry[1]
            avg_insertions = mean(insertions)
            load_factor = avg_insertions/total_slots
            print(tag_len, total_slots, float(avg_insertions), insertions)
            if boxplot:
                frame[tag_len] = list(map(lambda x: x/total_slots, insertions))
            else:
                data_points.append((tag_len, load_factor))
        
        fn = f"{plots_dir}/{cftype}-m_len-{m_len}-b_len-{b_len}-kickmax-{kickmax}.png"
        if boxplot:
            import pandas as pd
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            df = pd.DataFrame(frame)
            df.sort_index(axis=1, inplace=True)
            df.plot.box(grid=True, ax=ax, ylim=(0.9, 1), showfliers=False)
            ax.set_xlabel("$\lambda_T$")
            ax.set_ylabel("load factor")
            boxplot = df.boxplot()
            plt.savefig(fn)
            plt.close()
        else:
            g = line(data_points, title=f"|m| = {m_len}, b = {b}", axes_labels=["$\lambda_T$", "load factor"])
            save(g, fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="res_load_factor.json", help='raw data output file')
    parser.add_argument('--trials', type=int, default=16, help='trials per data point')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--no-run', action='store_true')
    parser.add_argument('--ncpus', type=int, default=4)
    parser.add_argument('--type', type=str, default="original")
    args = parser.parse_args()

    res_fn = args.out
    trials = args.trials
    ncpus = args.ncpus
    cftype = args.type
    if not args.no_run:
        run_experiments(cftype, res_fn, trials=args.trials, ncpus=ncpus)
    if not args.no_plots:
        plot_results(res_fn, cftype)

