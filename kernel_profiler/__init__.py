import pyopencl as cl
import loopy as lp
from loopy.preprocess import prepare_for_caching


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


def write_ptx(ctx, knl, filename=None):
    cl_program = cl.Program(
                            ctx, lp.generate_code_v2(knl).device_code()
                           ).build(options=knl.options.cl_build_options)
    ptx_src = cl_program.binaries[0]
    if not filename:
        filename = "ptx_"+knl.name+".ptx"
    ptx_src_file = open(filename, 'w')
    ptx_src_file.write(ptx_src.decode('utf-8', 'ignore'))


class KernelStatOptions:
    WALL_TIME = "wall_time"
    MEM_ACCESS_MAP = "mem_access_map"
    OP_MAP = "op_map"
    SYNC_MAP = "sync_map"
    GRID_SIZES = "grid_sizes"
    FLOP_RATE = "flop_rate"
    MEM_BANDWIDTH = "mem_bandwidth"
    SAVE_PTX = "save_ptx"
    GENERATED_CODE = "generated_code"
    # TODO mem access to footprint ratio


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
                include_kernel_params_in_ptx_filename=False,
                ptx_filename_suffix="",
                ):

        # TODO create cache to store kernels (for executing w/different params)
        self.ctx_cache = {}
        self.stat_cache = {}
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

        self.include_kernel_params_in_ptx_filename = \
                include_kernel_params_in_ptx_filename
        self.ptx_filename_suffix = ptx_filename_suffix

    def update_options(
                self,
                platform_name=None,
                device_name=None,
                interactive=None,
                n_warmup_wtime_trials=None,
                n_wtime_trials=None,
                evaluate_polys=None,
                count_redundant_work=None,
                subgroup_size=None,
                count_madds=None,
                count_within_subscripts=None,
                include_kernel_params_in_ptx_filename=None,
                ptx_filename_suffix=None,
                ):
        if platform_name is not None:
            self.platform_name = platform_name
        if device_name is not None:
            self.device_name = device_name
        if interactive is not None:
            self.interactive = interactive
        if n_warmup_wtime_trials is not None:
            self.n_warmup_wtime_trials = n_warmup_wtime_trials
        if n_wtime_trials is not None:
            self.n_wtime_trials = n_wtime_trials
        if evaluate_polys is not None:
            self.evaluate_polys = evaluate_polys
        if count_redundant_work is not None:
            self.count_redundant_work = count_redundant_work
        if subgroup_size is not None:
            self.subgroup_size = subgroup_size
        if count_madds is not None:
            self.count_madds = count_madds
        if count_within_subscripts is not None:
            self.count_within_subscripts = count_within_subscripts
        if include_kernel_params_in_ptx_filename is not None:
            self.include_kernel_params_in_ptx_filename = \
                    include_kernel_params_in_ptx_filename
        if ptx_filename_suffix is not None:
            self.ptx_filename_suffix = ptx_filename_suffix

    def get_cl_context(self, knl):

        if knl.target is not None and knl.target.device is not None:
            # kernel has a device already, see if we can use it
            knl_platform_name = knl.target.device.platform.name
            knl_device_name = knl.target.device.name

            # check for mismatch between platforms/devices
            if (self.platform_name is not None
                    and self.platform_name not in knl_platform_name) or (
                    self.device_name is not None
                    and self.device_name not in knl_device_name):
                raise ValueError("kernel target platform %s and/or device %s do "
                        "not match profiler platform %s and/or device %s."
                        % (knl_platform_name, knl_device_name,
                        self.platform_name, self.device_name))

            cache_key = (knl_platform_name, knl_device_name, "ctx")
            try:
                return self.ctx_cache[cache_key]
            except KeyError:
                ctx = cl.Context([find_cl_device_candidates(
                        knl_platform_name, knl_device_name)[-1]]
                        )
                self.ctx_cache[cache_key] = ctx
                return ctx

        elif self.platform_name is None or self.device_name is None:
            # kernel does not have a pre-specified device,
            # and profiler does not know platform+device
            ctx = cl.create_some_context()  # interactive mode
            self.platform_name = ctx.devices[0].platform.name
            self.device_name = ctx.devices[0].name
            self.ctx_cache[(self.platform_name, self.device_name, "ctx")] = ctx
            return ctx

        else:
            # profiler knows both platform and device already
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
                ):

        if param_dict is None:
            raise ValueError(
                    "Wall time requires dictionary of kernel parameters.")

        ctx = self.get_cl_context(knl)
        queue = cl.CommandQueue(ctx)

        arg_arrays = create_rand_args(ctx, knl, param_dict)
        knl = lp.set_options(knl, no_numpy=True)
        compiled = lp.CompiledKernel(ctx, knl)

        wtimes = []

        import time
        for t in range(self.n_time_trials + self.n_warmup_time_trials):
            queue.finish()
            tstart = time.time()
            evt, out = compiled(queue, **arg_arrays)
            queue.finish()
            tend = time.time()
            wtimes.append(tend-tstart)

        import numpy as np
        return np.average(wtimes[self.n_warmup_time_trials:])

    def save_ptx(
            self,
            knl,
            param_dict=None,
            ):

        if self.include_kernel_params_in_ptx_filename:
            if param_dict is None:
                raise ValueError("Cannot include kernel params "
                        "in ptx filename, no param dict passed.")
            write_ptx(
                    self.get_cl_context(knl),
                    knl,
                    filename="ptx_"+knl.name+"_"+"_".join(
                        ["%s%d" % (p, v) for p, v in param_dict.items()]
                        )+self.ptx_filename_suffix+".ptx"
                    )
        else:
            write_ptx(self.get_cl_context(knl), knl)

    def get_cached_stats_map(
                self,
                knl,
                stat_option,  # KernelStatOptions
                ):

        cache_key = (prepare_for_caching(knl), stat_option)
        # TODO avoid multiple calls to prepare_for_caching()?

        try:
            return self.stat_cache[cache_key]
        except KeyError:
            if stat_option == KernelStatOptions.MEM_ACCESS_MAP:
                from loopy.statistics import get_mem_access_map
                stats_map = get_mem_access_map(
                        knl,
                        count_redundant_work=self.count_redundant_work,
                        subgroup_size=self.subgroup_size,
                        )
            elif stat_option == KernelStatOptions.OP_MAP:
                from loopy.statistics import get_op_map
                stats_map = get_op_map(
                        knl,
                        count_redundant_work=self.count_redundant_work,
                        count_within_subscripts=self.count_within_subscripts,
                        subgroup_size=self.subgroup_size,
                        count_madds=self.count_madds,
                        )
            elif stat_option == KernelStatOptions.SYNC_MAP:
                from loopy.statistics import get_synchronization_map
                stats_map = get_synchronization_map(
                        knl,
                        subgroup_size=self.subgroup_size,
                        )
            self.stat_cache[cache_key] = stats_map
            return stats_map

    def get_stats_map_and_evaluate_if_required(
                self,
                knl,
                stat_option,  # KernelStatOptions
                param_dict=None,
                ):
        stats_map = self.get_cached_stats_map(knl, stat_option)
        if self.evaluate_polys:
            if param_dict is None:
                raise ValueError(
                        "Cannot evaluate polynomials without param_dict.")
            return stats_map.eval(param_dict)
        else:
            return stats_map

    def get_grid_sizes(
                self,
                knl,
                param_dict=None,
                ):

        cache_key = (prepare_for_caching(knl), KernelStatOptions.GRID_SIZES)
        # TODO avoid multiple calls to prepare_for_caching()?

        try:
            grid_sizes = self.stat_cache[cache_key]
        except KeyError:

            global_size, local_size = knl.get_grid_size_upper_bounds()

            from islpy import PwQPolynomial
            gsize_pwqs = []
            lsize_pwqs = []
            for gsize in global_size:
                gsize_pwqs.append(PwQPolynomial.from_pw_aff(gsize))
            for lsize in local_size:
                lsize_pwqs.append(PwQPolynomial.from_pw_aff(lsize))
            grid_sizes = [gsize_pwqs, lsize_pwqs]
            self.stat_cache[cache_key] = grid_sizes

        if self.evaluate_polys:
            if param_dict is None:
                raise ValueError("Cannont evaluate polynomials without param_dict.")
            return [g.eval_with_dict(param_dict) for g in grid_sizes[0]], \
                   [l.eval_with_dict(param_dict) for l in grid_sizes[1]]
        else:
            return grid_sizes

    def profile(
                self,
                knl,
                stat_options=[],
                param_dict=None,
                ):

        stats_found = {}
        kso = KernelStatOptions

        if kso.WALL_TIME in stat_options or \
                kso.FLOP_RATE in stat_options or \
                kso.MEM_BANDWIDTH in stat_options:
            stats_found[kso.WALL_TIME] = self.time_kernel(
                    knl,
                    param_dict,
                    )

        if kso.MEM_ACCESS_MAP in stat_options or \
                kso.MEM_BANDWIDTH in stat_options:
            stats_found[kso.MEM_ACCESS_MAP] = \
                    self.get_stats_map_and_evaluate_if_required(
                            knl,
                            kso.MEM_ACCESS_MAP,
                            param_dict=param_dict,
                            )

        if kso.OP_MAP in stat_options or \
                kso.FLOP_RATE in stat_options:
            stats_found[kso.OP_MAP] = \
                    self.get_stats_map_and_evaluate_if_required(
                            knl,
                            kso.OP_MAP,
                            param_dict=param_dict,
                            )

        if kso.SYNC_MAP in stat_options:
            stats_found[kso.SYNC_MAP] = \
                    self.get_stats_map_and_evaluate_if_required(
                            knl,
                            kso.SYNC_MAP,
                            param_dict=param_dict,
                            )

        if kso.GRID_SIZES in stat_options:
            stats_found[kso.GRID_SIZES] = self.get_grid_sizes(
                    knl,
                    param_dict=param_dict,
                    )

        if kso.FLOP_RATE in stat_options:
            import numpy as np
            # count madds as 2 ops
            # (count all flops once and then count the madds again)

            # flops counted w/subgroup granularity
            float_ops = self.subgroup_size*(
                    stats_found[kso.OP_MAP].filter_by(
                        dtype=[np.float32, np.float64],
                        count_granularity=[lp.CountGranularity.SUBGROUP],
                        ).sum() +
                    stats_found[kso.OP_MAP].filter_by(
                        dtype=[np.float32, np.float64],
                        count_granularity=[lp.CountGranularity.SUBGROUP],
                        name=["madd"]
                        ).sum())

            # flops counted w/workitem granularity (should be zero)
            float_ops += stats_found[kso.OP_MAP].filter_by(
                    dtype=[np.float32, np.float64],
                    count_granularity=[lp.CountGranularity.WORKITEM],
                    ).sum() + stats_found[kso.OP_MAP].filter_by(
                    dtype=[np.float32, np.float64],
                    count_granularity=[lp.CountGranularity.WORKITEM],
                    name=["madd"]
                    ).sum()
            # TODO after ToCountMap gets version of sum that allows
            # counting w/specified count granularity, update this

            if not self.evaluate_polys:
                float_ops = float_ops.eval_with_dict(param_dict)
            stats_found[kso.FLOP_RATE] = float_ops/stats_found[kso.WALL_TIME]

        if kso.MEM_BANDWIDTH in stat_options:
            # first get footprint of data moved
            from loopy import gather_access_footprint_bytes
            footsize_bytes = 0
            for access, count in stats_found[kso.MEM_ACCESS_MAP].filter_by(
                        mtype=["global"]).items():
                direction = "write" if access.direction == "store" else "read"
                footsize_bytes += gather_access_footprint_bytes(knl)[
                        (access.variable, direction)].eval_with_dict(param_dict)

            # mem access counted w/subgroup granularity
            data_moved_bytes = stats_found[kso.MEM_ACCESS_MAP].filter_by(
                    mtype=["global"],
                    count_granularity=[lp.CountGranularity.SUBGROUP],
                    ).to_bytes().sum()*self.subgroup_size
            # mem access counted w/workitem granularity
            data_moved_bytes += stats_found[kso.MEM_ACCESS_MAP].filter_by(
                    mtype=["global"],
                    count_granularity=[lp.CountGranularity.WORKITEM],
                    ).to_bytes().sum()
            # if these polys have not alread been evaluated, evaluate them
            if not self.evaluate_polys:
                data_moved_bytes = data_moved_bytes.eval_with_dict(param_dict)

            stats_found[kso.MEM_BANDWIDTH] = (
                    data_moved_bytes/stats_found[kso.WALL_TIME],
                    footsize_bytes/stats_found[kso.WALL_TIME]
                    )

        if kso.SAVE_PTX in stat_options:
            self.save_ptx(knl, param_dict)

        if kso.GENERATED_CODE in stat_options:
            stats_found[kso.GENERATED_CODE] = lp.generate_code_v2(knl).device_code()

        return stats_found
