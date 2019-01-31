# kernel_profiler

**Install**:

`python setup.py install`

**Developer install** (source changes take immediate effect):

`python setup.py develop`

# Stat options

* **KernelStatOptions.WALL_TIME**  
   Kernel execution time using random input data.

* **KernelStatOptions.MEM_ACCESS_MAP**  
   A [**loopy.ToCountMap**](https://documen.tician.de/loopy/ref_other.html#loopy.ToCountMap) mapping memory accesses to counts. Also see [**loopy.get_mem_access_map**](https://documen.tician.de/loopy/ref_other.html#loopy.get_mem_access_map).

* **KernelStatOptions.OP_MAP**  
   A [**loopy.ToCountMap**](https://documen.tician.de/loopy/ref_other.html#loopy.ToCountMap) mapping operations to counts. Also see [**loopy.get_op_map**](https://documen.tician.de/loopy/ref_other.html#loopy.get_op_map).

* **KernelStatOptions.SYNC_MAP**  
   A [**loopy.ToCountMap**](https://documen.tician.de/loopy/ref_other.html#loopy.ToCountMap) mapping synchronization operations to counts. Also see [**loopy.get_synchronization_map**](https://documen.tician.de/loopy/ref_other.html#loopy.get_synchronization_map).

* **KernelStatOptions.GRID_SIZES**  
   A tuple containing (local sizes, global sizes).

* **KernelStatOptions.FLOP_RATE**  
   Number of 32-bit and 64-bit floating point operations per second.

* **KernelStatOptions.MEM_BANDWIDTH**  
   Global memory bytes accessed per second.

* **KernelStatOptions.GENERATED_CODE**  
   Generated opencl code.

* **KernelStatOptions.SAVE_PTX**  
   Save PTX (Portable Thread eXecution) file.
