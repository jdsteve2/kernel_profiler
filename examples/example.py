import loopy as lp
import numpy as np
from kernel_profiler import KernelProfiler
from kernel_profiler import KernelStatOptions as kso  # noqa


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

n = 2**10
m = 2**11
ell = 2**9
param_dict = {'n': n, 'm': m, 'ell': ell}

kp = KernelProfiler(
        #"NVIDIA", "GEFORCE",
        #"NVIDIA", "K40C",
        evaluate_polys=True,
        include_kernel_params_in_ptx_filename=True,
        )
stats = kp.profile(
        knl,
        [
            kso.WALL_TIME,
            kso.MEM_ACCESS_MAP,
            kso.OP_MAP,
            kso.SYNC_MAP,
            kso.GRID_SIZES,
            kso.FLOP_RATE,
            kso.MEM_BANDWIDTH,
            kso.SAVE_PTX,
            kso.GENERATED_CODE,
        ],
        param_dict=param_dict,
        )
print("\nWall time:", stats[kso.WALL_TIME], "\n")
print(lp.stringify_stats_mapping(stats[kso.MEM_ACCESS_MAP]))
print(lp.stringify_stats_mapping(stats[kso.OP_MAP]))
print(lp.stringify_stats_mapping(stats[kso.SYNC_MAP]))
print(stats[kso.GRID_SIZES], "\n")
print(stats[kso.FLOP_RATE], "\n")
print(stats[kso.MEM_BANDWIDTH][0], stats[kso.MEM_BANDWIDTH][1], "\n")

kp.update_options(evaluate_polys=False)

stats = kp.profile(
        knl, [kso.WALL_TIME], param_dict=param_dict)
print(stats[kso.WALL_TIME], "\n")
