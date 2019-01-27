import loopy as lp
import numpy as np
from kernel_profiler import KernelProfiler
from kernel_profiler import KernelStatOptions as stat_opts


knl = lp.make_kernel(
        "{[i,k,j]: 0<=i<n and 0<=k<m and 0<=j<ell}",
        [
            "c[i, j] = sum(k, a[i, k]*b[k, j])"
        ],
        name="matmul", assumptions="n,m,ell >= 1")

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
stats = kp.get_stats(knl, [stat_opts.WALL_TIME], param_dict=param_dict)
print(stats[stat_opts.WALL_TIME])

interactive_kp = KernelProfiler(interactive=True)
interactive_stats = interactive_kp.get_stats(
        knl, [stat_opts.WALL_TIME], param_dict=param_dict)
print(interactive_stats[stat_opts.WALL_TIME])
