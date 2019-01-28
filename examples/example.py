import loopy as lp
import numpy as np
from kernel_profiler import KernelProfiler
from kernel_profiler import KernelStatOptions as stat_opts


knl = lp.make_kernel(
        "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
        [
            "c[i, j] = sum(k, a[i, k]*b[k, j])"
        ],
        name="matmul",
        assumptions="n,m,ell >= 1",
        lang_version=(2018, 2),
        )

knl = lp.add_and_infer_dtypes(knl, dict(a=np.float32, b=np.float32))

lsize = 16
knl = lp.split_iname(knl, "i", lsize, outer_tag="g.0", inner_tag="l.1")
knl = lp.split_iname(knl, "j", lsize, outer_tag="g.1", inner_tag="l.0")
knl = lp.split_iname(knl, "k", lsize)
knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"], default_tag="l.auto")
knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner"], default_tag="l.auto")

n = 512
m = 256
ell = 128
param_dict = {'n': n, 'm': m, 'ell': ell}

kp = KernelProfiler("NVIDIA", "GEFORCE")
stats = kp.profile(
        knl,
        [
            stat_opts.WALL_TIME,
            stat_opts.MEMORY_ACCESS,
            stat_opts.ARITHMETIC_OPS,
            stat_opts.SYNCHRONIZATION,
            stat_opts.GRID_SIZES,
        ],
        param_dict=param_dict)
print("\nWall time:", stats[stat_opts.WALL_TIME], "\n")
print(lp.stringify_stats_mapping(stats[stat_opts.MEMORY_ACCESS]))
print(lp.stringify_stats_mapping(stats[stat_opts.ARITHMETIC_OPS]))
print(lp.stringify_stats_mapping(stats[stat_opts.SYNCHRONIZATION]))
print(stats[stat_opts.GRID_SIZES], "\n")

interactive_kp = KernelProfiler(interactive=True)
interactive_stats = interactive_kp.profile(
        knl, [stat_opts.WALL_TIME], param_dict=param_dict)
print(interactive_stats[stat_opts.WALL_TIME], "\n")
