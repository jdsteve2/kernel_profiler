import pyopencl as cl
import loopy as lp


def create_rand_args(ctx, knl, param_dict):

    queue = cl.CommandQueue(ctx)
    info = lp.generate_code_v2(knl).implemented_data_info
    args, arg_data = lp.auto_test.make_ref_args(
                        knl,
                        info,
                        queue, param_dict)
    args.clear()
    del args

    rand_args = lp.auto_test.make_args(knl, info,
                                       queue, arg_data, param_dict)

    del arg_data[:]
    del arg_data
    return rand_args


def find_cl_device_candidates(platform_name, device_name):
    candidates = [
            dev
            for plat in cl.get_platforms()
            if platform_name.lower() in plat.name.lower()
            for dev in plat.get_devices()
            if device_name.lower() in dev.name.lower()]

    if not candidates:
        raise ValueError("no device matched the platform/device ID tuple "
                "'%s' and '%s'" % (platform_name, device_name))

    return candidates


class KernelStatOptions:
    WALL_TIME = "wall_time"
    MEMORY_ACCESS = "memory_access"
    ARITHMETIC_OPS = "arithmetic_ops"
    SYNCHRONIZATION = "synchronization"
    GRID_SIZES = "grid_sizes"
    # TODO add other stat options here


class KernelProfiler(object):

    def __init__(
                self,
                platform_name=None,
                device_name=None,
                interactive=False,
                n_warmup_time_trials=4,
                n_time_trials=64,
                evaluate_polys=True,
                subgroup_size=32,
                count_redundant_work=True,
                count_madds=True,
                count_within_subscripts=False,
                ):

        self.ctx_cache = {}
        self.platform_name = platform_name
        self.device_name = device_name
        self.interactive = interactive

        self.n_warmup_time_trials = n_warmup_time_trials
        self.n_time_trials = n_time_trials

        self.evaluate_polys = evaluate_polys
        self.subgroup_size = subgroup_size
        self.count_redundant_work = count_redundant_work
        self.count_madds = count_madds
        self.count_within_subscripts = count_within_subscripts

    def get_cl_context(self):

        if self.interactive:
            return cl.create_some_context()
        else:
            if self.platform_name is None or self.device_name is None:
                raise ValueError(
                        "Wall time requires platform name, and device name.")

            cache_key = (self.platform_name, self.device_name, "ctx")
            try:
                return self.ctx_cache[cache_key]
            except KeyError:
                ctx = cl.Context([find_cl_device_candidates(
                        self.platform_name, self.device_name)[-1]]
                        )
                self.ctx_cache[cache_key] = ctx
                return ctx

    def time_kernel(
                self,
                knl,
                param_dict,
                n_warmup_trials=None,
                n_trials=None,
                ):

        n_warmup_trials = self.n_warmup_time_trials if not n_warmup_trials \
                else n_warmup_trials
        n_trials = self.n_time_trials if not n_trials else n_trials

        ctx = self.get_cl_context()
        queue = cl.CommandQueue(ctx)

        arg_arrays = create_rand_args(ctx, knl, param_dict)
        knl = lp.set_options(knl, no_numpy=True)
        compiled = lp.CompiledKernel(ctx, knl)

        wtimes = []

        import time
        for t in range(n_trials + n_warmup_trials):
            queue.finish()
            tstart = time.time()
            evt, out = compiled(queue, **arg_arrays)
            queue.finish()
            tend = time.time()
            wtimes.append(tend-tstart)

        import numpy as np
        return np.average(wtimes[n_warmup_trials:])

    def get_mem_access_stats(
                self,
                knl,
                evaluate_polys=None,
                param_dict=None,
                count_redundant_work=None,
                subgroup_size=None,
                ):

        from loopy.statistics import get_mem_access_map

        # if no value passed, set to defaults
        evaluate_polys = self.evaluate_polys \
                if not evaluate_polys else evaluate_polys
        count_redundant_work = self.count_redundant_work \
                if not count_redundant_work else count_redundant_work
        subgroup_size = self.subgroup_size if not subgroup_size else subgroup_size

        mem_access_map = get_mem_access_map(
                knl,
                count_redundant_work=count_redundant_work,
                subgroup_size=subgroup_size,
                )

        if evaluate_polys:
            if param_dict is None:
                raise ValueError("Cannont evaluate polynomials without param_dict.")
            return mem_access_map.eval(param_dict)
        else:
            return mem_access_map

    def get_op_stats(
                self,
                knl,
                evaluate_polys=None,
                param_dict=None,
                count_redundant_work=None,
                subgroup_size=None,
                count_madds=None,
                count_within_subscripts=None,
                ):

        from loopy.statistics import get_op_map

        # if no value passed, set to defaults
        evaluate_polys = self.evaluate_polys \
                if not evaluate_polys else evaluate_polys
        count_redundant_work = self.count_redundant_work \
                if not count_redundant_work else count_redundant_work
        subgroup_size = self.subgroup_size if not subgroup_size else subgroup_size
        count_madds = self.count_madds if not count_madds else count_madds
        count_within_subscripts = self.count_within_subscripts \
                if not count_within_subscripts else count_within_subscripts

        op_map = get_op_map(
            knl,
            count_redundant_work=count_redundant_work,
            count_within_subscripts=count_within_subscripts,
            subgroup_size=subgroup_size,
            count_madds=count_madds,
            )

        if evaluate_polys:
            if param_dict is None:
                raise ValueError("Cannont evaluate polynomials without param_dict.")
            return op_map.eval(param_dict)
        else:
            return op_map

    def get_synchronization_stats(
                self,
                knl,
                evaluate_polys=None,
                param_dict=None,
                subgroup_size=None,
                ):

        from loopy.statistics import get_synchronization_map

        # if no value passed, set to defaults
        evaluate_polys = self.evaluate_polys \
                if not evaluate_polys else evaluate_polys
        subgroup_size = self.subgroup_size if not subgroup_size else subgroup_size

        sync_map = get_synchronization_map(
            knl,
            subgroup_size=subgroup_size,
            )

        if evaluate_polys:
            if param_dict is None:
                raise ValueError("Cannont evaluate polynomials without param_dict.")
            return sync_map.eval(param_dict)
        else:
            return sync_map

    def get_grid_sizes(
                self,
                knl,
                evaluate_polys=None,
                param_dict=None,
                ):

        # if no value passed, set to defaults
        evaluate_polys = self.evaluate_polys \
                if not evaluate_polys else evaluate_polys

        global_size, local_size = knl.get_grid_size_upper_bounds()

        from islpy import PwQPolynomial
        gsize_pwqs = []
        lsize_pwqs = []
        for gsize in global_size:
            gsize_pwqs.append(PwQPolynomial.from_pw_aff(gsize))
        for lsize in local_size:
            lsize_pwqs.append(PwQPolynomial.from_pw_aff(lsize))

        if evaluate_polys:
            if param_dict is None:
                raise ValueError("Cannont evaluate polynomials without param_dict.")
            return [g.eval_with_dict(param_dict) for g in gsize_pwqs], \
                   [l.eval_with_dict(param_dict) for l in lsize_pwqs]
        else:
            return gsize_pwqs, lsize_pwqs

    def profile(
                self,
                knl,
                stat_options=[],
                param_dict=None,
                n_warmup_wtime_trials=None,
                n_wtime_trials=None,
                evaluate_polys=True,
                count_redundant_work=None,
                subgroup_size=None,
                count_madds=True,
                count_within_subscripts=False,
                ):

        stats_found = {}

        if KernelStatOptions.WALL_TIME in stat_options:

            # if no value passed, set to defaults
            #TODO these checks are redundant
            n_warmup_wtime_trials = self.n_warmup_time_trials \
                    if n_warmup_wtime_trials is None else n_warmup_wtime_trials
            n_wtime_trials = self.n_time_trials \
                    if n_wtime_trials is None else n_wtime_trials

            if param_dict is None:
                raise ValueError(
                        "Wall time requires dictionary of kernel parameters.")

            stats_found[KernelStatOptions.WALL_TIME] = self.time_kernel(
                    knl, param_dict, n_warmup_wtime_trials, n_wtime_trials)

        if KernelStatOptions.MEMORY_ACCESS in stat_options:
            # if no value passed, set to defaults
            evaluate_polys = self.evaluate_polys \
                    if not evaluate_polys else evaluate_polys
            count_redundant_work = self.count_redundant_work \
                    if not count_redundant_work else count_redundant_work
            subgroup_size = self.subgroup_size \
                    if not subgroup_size else subgroup_size

            stats_found[KernelStatOptions.MEMORY_ACCESS] = self.get_mem_access_stats(
                    knl,
                    evaluate_polys=evaluate_polys,
                    param_dict=param_dict,
                    count_redundant_work=count_redundant_work,
                    subgroup_size=subgroup_size,
                    )

        if KernelStatOptions.ARITHMETIC_OPS in stat_options:
            # if no value passed, set to defaults
            evaluate_polys = self.evaluate_polys \
                    if not evaluate_polys else evaluate_polys
            count_redundant_work = self.count_redundant_work \
                    if not count_redundant_work else count_redundant_work
            subgroup_size = self.subgroup_size \
                    if not subgroup_size else subgroup_size
            count_madds = self.count_madds if not count_madds else count_madds
            count_within_subscripts = self.count_within_subscripts \
                    if not count_within_subscripts else count_within_subscripts

            stats_found[KernelStatOptions.ARITHMETIC_OPS] = self.get_op_stats(
                    knl,
                    evaluate_polys=evaluate_polys,
                    param_dict=param_dict,
                    count_redundant_work=count_redundant_work,
                    subgroup_size=subgroup_size,
                    count_madds=count_madds,
                    count_within_subscripts=count_within_subscripts,
                    )

        if KernelStatOptions.SYNCHRONIZATION in stat_options:
            # if no value passed, set to defaults
            evaluate_polys = self.evaluate_polys \
                    if not evaluate_polys else evaluate_polys
            subgroup_size = self.subgroup_size \
                    if not subgroup_size else subgroup_size

            stats_found[KernelStatOptions.SYNCHRONIZATION] = \
                    self.get_synchronization_stats(
                    knl,
                    evaluate_polys=evaluate_polys,
                    param_dict=param_dict,
                    subgroup_size=subgroup_size,
                    )

        if KernelStatOptions.GRID_SIZES in stat_options:
            # if no value passed, set to defaults
            evaluate_polys = self.evaluate_polys \
                    if not evaluate_polys else evaluate_polys

            stats_found[KernelStatOptions.GRID_SIZES] = self.get_grid_sizes(
                    knl,
                    evaluate_polys=evaluate_polys,
                    param_dict=param_dict,
                    )

        return stats_found
