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
    # TODO add other stat options here


class KernelProfiler(object):

    N_WARMUP_TIME_TRIALS = 4
    N_TIME_TRIALS = 64

    def __init__(
                self,
                platform_name=None,
                device_name=None,
                interactive=False,
                ):
        self.ctx_cache = {}
        self.platform_name = platform_name
        self.device_name = device_name
        self.interactive = interactive

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
                ctx = cl.Context(
                        [find_cl_device_candidates(self.platform_name, self.device_name)[-1]]
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

        n_warmup_trials = self.N_WARMUP_TIME_TRIALS if n_warmup_trials is None \
                else n_warmup_trials
        n_trials = self.N_TIME_TRIALS if n_trials is None else n_trials

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

    def get_stats(
            self,
            knl,
            stat_options=[],
            param_dict=None,
            n_warmup_wtime_trials=None,
            n_wtime_trials=None,
            ):

        stats_found = {}

        if KernelStatOptions.WALL_TIME in stat_options:

            n_warmup_wtime_trials = self.N_WARMUP_TIME_TRIALS \
                    if n_warmup_wtime_trials is None else n_warmup_wtime_trials
            n_wtime_trials = self.N_TIME_TRIALS \
                    if n_wtime_trials is None else n_wtime_trials

            if param_dict is None:
                raise ValueError(
                        "Wall time requires dictionary of kernel parameters.")

            stats_found[KernelStatOptions.WALL_TIME] = self.time_kernel(
                    knl, param_dict, n_warmup_wtime_trials, n_wtime_trials)

        # TODO add other stat options here

        return stats_found
