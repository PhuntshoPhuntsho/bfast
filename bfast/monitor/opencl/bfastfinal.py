import sys
import numpy as np
import ctypes as ct
# Stub code for OpenCL setup.

import pyopencl as cl
import numpy as np
import sys

if cl.version.VERSION < (2015,2):
    raise Exception('Futhark requires at least PyOpenCL version 2015.2.  Installed version is %s.' %
                    cl.version.VERSION_TEXT)

def parse_preferred_device(s):
    pref_num = 0
    if len(s) > 1 and s[0] == '#':
        i = 1
        while i < len(s):
            if not s[i].isdigit():
                break
            else:
                pref_num = pref_num * 10 + int(s[i])
            i += 1
        while i < len(s) and s[i].isspace():
            i += 1
        return (s[i:], pref_num)
    else:
        return (s, 0)

def get_prefered_context(interactive=False, platform_pref=None, device_pref=None):
    if device_pref != None:
        (device_pref, device_num) = parse_preferred_device(device_pref)
    else:
        device_num = 0

    if interactive:
        return cl.create_some_context(interactive=True)

    def blacklisted(p, d):
        return platform_pref == None and device_pref == None and \
            p.name == "Apple" and d.name.find("Intel(R) Core(TM)") >= 0
    def platform_ok(p):
        return not platform_pref or p.name.find(platform_pref) >= 0
    def device_ok(d):
        return not device_pref or d.name.find(device_pref) >= 0

    device_matches = 0

    for p in cl.get_platforms():
        if not platform_ok(p):
            continue
        for d in p.get_devices():
            if blacklisted(p,d) or not device_ok(d):
                continue
            if device_matches == device_num:
                return cl.Context(devices=[d])
            else:
                device_matches += 1
    raise Exception('No OpenCL platform and device matching constraints found.')

def size_assignment(s):
    name, value = s.split('=')
    return (name, int(value))

def check_types(self, required_types):
    if 'f64' in required_types:
        if self.device.get_info(cl.device_info.PREFERRED_VECTOR_WIDTH_DOUBLE) == 0:
            raise Exception('Program uses double-precision floats, but this is not supported on chosen device: %s' % self.device.name)

def apply_size_heuristics(self, size_heuristics, sizes):
    for (platform_name, device_type, size, valuef) in size_heuristics:
        if sizes[size] == None \
           and self.platform.name.find(platform_name) >= 0 \
           and (self.device.type & device_type) == device_type:
               sizes[size] = valuef(self.device)
    return sizes

def initialise_opencl_object(self,
                             program_src='',
                             command_queue=None,
                             interactive=False,
                             platform_pref=None,
                             device_pref=None,
                             default_group_size=None,
                             default_num_groups=None,
                             default_tile_size=None,
                             default_threshold=None,
                             size_heuristics=[],
                             required_types=[],
                             all_sizes={},
                             user_sizes={}):
    if command_queue is None:
        self.ctx = get_prefered_context(interactive, platform_pref, device_pref)
        self.queue = cl.CommandQueue(self.ctx)
    else:
        self.ctx = command_queue.context
        self.queue = command_queue
    self.device = self.queue.device
    self.platform = self.device.platform
    self.pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))
    device_type = self.device.type

    check_types(self, required_types)

    max_group_size = int(self.device.max_work_group_size)
    max_tile_size = int(np.sqrt(self.device.max_work_group_size))

    self.max_group_size = max_group_size
    self.max_tile_size = max_tile_size
    self.max_threshold = 0
    self.max_num_groups = 0

    self.max_local_memory = int(self.device.local_mem_size)

    # Futhark reserves 4 bytes of local memory for its own purposes.
    self.max_local_memory -= 4

    # See comment in rts/c/opencl.h.
    if self.platform.name.find('NVIDIA CUDA') >= 0:
        self.max_local_memory -= 12

    self.free_list = {}

    self.global_failure = self.pool.allocate(np.int32().itemsize)
    cl.enqueue_fill_buffer(self.queue, self.global_failure, np.int32(-1), 0, np.int32().itemsize)
    self.global_failure_args = self.pool.allocate(np.int32().itemsize *
                                                  (self.global_failure_args_max+1))
    self.failure_is_an_option = np.int32(0)

    if 'default_group_size' in sizes:
        default_group_size = sizes['default_group_size']
        del sizes['default_group_size']

    if 'default_num_groups' in sizes:
        default_num_groups = sizes['default_num_groups']
        del sizes['default_num_groups']

    if 'default_tile_size' in sizes:
        default_tile_size = sizes['default_tile_size']
        del sizes['default_tile_size']

    if 'default_threshold' in sizes:
        default_threshold = sizes['default_threshold']
        del sizes['default_threshold']

    default_group_size_set = default_group_size != None
    default_tile_size_set = default_tile_size != None
    default_sizes = apply_size_heuristics(self, size_heuristics,
                                          {'group_size': default_group_size,
                                           'tile_size': default_tile_size,
                                           'num_groups': default_num_groups,
                                           'lockstep_width': None,
                                           'threshold': default_threshold})
    default_group_size = default_sizes['group_size']
    default_num_groups = default_sizes['num_groups']
    default_threshold = default_sizes['threshold']
    default_tile_size = default_sizes['tile_size']
    lockstep_width = default_sizes['lockstep_width']

    if default_group_size > max_group_size:
        if default_group_size_set:
            sys.stderr.write('Note: Device limits group size to {} (down from {})\n'.
                             format(max_tile_size, default_group_size))
        default_group_size = max_group_size

    if default_tile_size > max_tile_size:
        if default_tile_size_set:
            sys.stderr.write('Note: Device limits tile size to {} (down from {})\n'.
                             format(max_tile_size, default_tile_size))
        default_tile_size = max_tile_size

    for (k,v) in user_sizes.items():
        if k in all_sizes:
            all_sizes[k]['value'] = v
        else:
            raise Exception('Unknown size: {}\nKnown sizes: {}'.format(k, ' '.join(all_sizes.keys())))

    self.sizes = {}
    for (k,v) in all_sizes.items():
        if v['class'] == 'group_size':
            max_value = max_group_size
            default_value = default_group_size
        elif v['class'] == 'num_groups':
            max_value = max_group_size # Intentional!
            default_value = default_num_groups
        elif v['class'] == 'tile_size':
            max_value = max_tile_size
            default_value = default_tile_size
        elif v['class'].startswith('threshold'):
            max_value = None
            default_value = default_threshold
        else:
            # Bespoke sizes have no limit or default.
            max_value = None
        if v['value'] == None:
            self.sizes[k] = default_value
        elif max_value != None and v['value'] > max_value:
            sys.stderr.write('Note: Device limits {} to {} (down from {}\n'.
                             format(k, max_value, v['value']))
            self.sizes[k] = max_value
        else:
            self.sizes[k] = v['value']

    # XXX: we perform only a subset of z-encoding here.  Really, the
    # compiler should provide us with the variables to which
    # parameters are mapped.
    if (len(program_src) >= 0):
        return cl.Program(self.ctx, program_src).build(
            ["-DLOCKSTEP_WIDTH={}".format(lockstep_width)]
            + ["-D{}={}".format(s.replace('z', 'zz').replace('.', 'zi').replace('#', 'zh'),v) for (s,v) in self.sizes.items()])

def opencl_alloc(self, min_size, tag):
    min_size = 1 if min_size == 0 else min_size
    assert min_size > 0
    return self.pool.allocate(min_size)

def opencl_free_all(self):
    self.pool.free_held()

def sync(self):
    failure = np.empty(1, dtype=np.int32)
    cl.enqueue_copy(self.queue, failure, self.global_failure, is_blocking=True)
    self.failure_is_an_option = np.int32(0)
    if failure[0] >= 0:
        # Reset failure information.
        cl.enqueue_fill_buffer(self.queue, self.global_failure, np.int32(-1), 0, np.int32().itemsize)

        # Read failure args.
        failure_args = np.empty(self.global_failure_args_max+1, dtype=np.int32)
        cl.enqueue_copy(self.queue, failure_args, self.global_failure_args, is_blocking=True)

        raise Exception(self.failure_msgs[failure[0]].format(*failure_args))
import pyopencl.array
import time
import argparse
sizes = {}
synchronous = False
preferred_platform = None
preferred_device = None
default_threshold = None
default_group_size = None
default_num_groups = None
default_tile_size = None
fut_opencl_src = """#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
__kernel void dummy_kernel(__global unsigned char *dummy, int n)
{
    const int thread_gid = get_global_id(0);
    
    if (thread_gid >= n)
        return;
}
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;
#ifdef cl_nv_pragma_unroll
static inline void mem_fence_global()
{
    asm("membar.gl;");
}
#else
static inline void mem_fence_global()
{
    mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
#endif
static inline void mem_fence_local()
{
    mem_fence(CLK_LOCAL_MEM_FENCE);
}
static inline uint8_t add8(uint8_t x, uint8_t y)
{
    return x + y;
}
static inline uint16_t add16(uint16_t x, uint16_t y)
{
    return x + y;
}
static inline uint32_t add32(uint32_t x, uint32_t y)
{
    return x + y;
}
static inline uint64_t add64(uint64_t x, uint64_t y)
{
    return x + y;
}
static inline uint8_t sub8(uint8_t x, uint8_t y)
{
    return x - y;
}
static inline uint16_t sub16(uint16_t x, uint16_t y)
{
    return x - y;
}
static inline uint32_t sub32(uint32_t x, uint32_t y)
{
    return x - y;
}
static inline uint64_t sub64(uint64_t x, uint64_t y)
{
    return x - y;
}
static inline uint8_t mul8(uint8_t x, uint8_t y)
{
    return x * y;
}
static inline uint16_t mul16(uint16_t x, uint16_t y)
{
    return x * y;
}
static inline uint32_t mul32(uint32_t x, uint32_t y)
{
    return x * y;
}
static inline uint64_t mul64(uint64_t x, uint64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t udiv_up8(uint8_t x, uint8_t y)
{
    return (x + y - 1) / y;
}
static inline uint16_t udiv_up16(uint16_t x, uint16_t y)
{
    return (x + y - 1) / y;
}
static inline uint32_t udiv_up32(uint32_t x, uint32_t y)
{
    return (x + y - 1) / y;
}
static inline uint64_t udiv_up64(uint64_t x, uint64_t y)
{
    return (x + y - 1) / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline uint8_t udiv_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint16_t udiv_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint32_t udiv_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint64_t udiv_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint8_t udiv_up_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint16_t udiv_up_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint32_t udiv_up_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint64_t udiv_up_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint8_t umod_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint16_t umod_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint32_t umod_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint64_t umod_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t sdiv_up8(int8_t x, int8_t y)
{
    return sdiv8(x + y - 1, y);
}
static inline int16_t sdiv_up16(int16_t x, int16_t y)
{
    return sdiv16(x + y - 1, y);
}
static inline int32_t sdiv_up32(int32_t x, int32_t y)
{
    return sdiv32(x + y - 1, y);
}
static inline int64_t sdiv_up64(int64_t x, int64_t y)
{
    return sdiv64(x + y - 1, y);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t sdiv_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : sdiv8(x, y);
}
static inline int16_t sdiv_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : sdiv16(x, y);
}
static inline int32_t sdiv_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : sdiv32(x, y);
}
static inline int64_t sdiv_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : sdiv64(x, y);
}
static inline int8_t sdiv_up_safe8(int8_t x, int8_t y)
{
    return sdiv_safe8(x + y - 1, y);
}
static inline int16_t sdiv_up_safe16(int16_t x, int16_t y)
{
    return sdiv_safe16(x + y - 1, y);
}
static inline int32_t sdiv_up_safe32(int32_t x, int32_t y)
{
    return sdiv_safe32(x + y - 1, y);
}
static inline int64_t sdiv_up_safe64(int64_t x, int64_t y)
{
    return sdiv_safe64(x + y - 1, y);
}
static inline int8_t smod_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : smod8(x, y);
}
static inline int16_t smod_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : smod16(x, y);
}
static inline int32_t smod_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : smod32(x, y);
}
static inline int64_t smod_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : smod64(x, y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t squot_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int16_t squot_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int32_t squot_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int64_t squot_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int8_t srem_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int16_t srem_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int32_t srem_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int64_t srem_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline bool ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline bool ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline bool ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline bool ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline bool ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline bool ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline bool ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline bool ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline bool slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline bool slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline bool slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline bool slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline bool sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline bool sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline bool sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline bool sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((int8_t) (uint8_t) x)
#define zext_i8_i16(x) ((int16_t) (uint8_t) x)
#define zext_i8_i32(x) ((int32_t) (uint8_t) x)
#define zext_i8_i64(x) ((int64_t) (uint8_t) x)
#define zext_i16_i8(x) ((int8_t) (uint16_t) x)
#define zext_i16_i16(x) ((int16_t) (uint16_t) x)
#define zext_i16_i32(x) ((int32_t) (uint16_t) x)
#define zext_i16_i64(x) ((int64_t) (uint16_t) x)
#define zext_i32_i8(x) ((int8_t) (uint32_t) x)
#define zext_i32_i16(x) ((int16_t) (uint32_t) x)
#define zext_i32_i32(x) ((int32_t) (uint32_t) x)
#define zext_i32_i64(x) ((int64_t) (uint32_t) x)
#define zext_i64_i8(x) ((int8_t) (uint64_t) x)
#define zext_i64_i16(x) ((int16_t) (uint64_t) x)
#define zext_i64_i32(x) ((int32_t) (uint64_t) x)
#define zext_i64_i64(x) ((int64_t) (uint64_t) x)
#if defined(__OPENCL_VERSION__)
static int32_t futrts_popc8(int8_t x)
{
    return popcount(x);
}
static int32_t futrts_popc16(int16_t x)
{
    return popcount(x);
}
static int32_t futrts_popc32(int32_t x)
{
    return popcount(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return popcount(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_popc8(int8_t x)
{
    return __popc(zext_i8_i32(x));
}
static int32_t futrts_popc16(int16_t x)
{
    return __popc(zext_i16_i32(x));
}
static int32_t futrts_popc32(int32_t x)
{
    return __popc(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return __popcll(x);
}
#else
static int32_t futrts_popc8(int8_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc16(int16_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc32(int32_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc64(int64_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    return mul_hi(a, b);
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    return mul_hi(a, b);
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mul_hi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul_hi(a, b);
}
#elif defined(__CUDA_ARCH__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mulhi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul64hi(a, b);
}
#else
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    uint64_t aa = a;
    uint64_t bb = b;
    
    return aa * bb >> 32;
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    __uint128_t aa = a;
    __uint128_t bb = b;
    
    return aa * bb >> 64;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return mad_hi(a, b, c);
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return mad_hi(a, b, c);
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return mad_hi(a, b, c);
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return mad_hi(a, b, c);
}
#else
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return futrts_mul_hi8(a, b) + c;
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return futrts_mul_hi16(a, b) + c;
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return futrts_mul_hi32(a, b) + c;
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return futrts_mul_hi64(a, b) + c;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_clzz8(int8_t x)
{
    return clz(x);
}
static int32_t futrts_clzz16(int16_t x)
{
    return clz(x);
}
static int32_t futrts_clzz32(int32_t x)
{
    return clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return clz(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_clzz8(int8_t x)
{
    return __clz(zext_i8_i32(x)) - 24;
}
static int32_t futrts_clzz16(int16_t x)
{
    return __clz(zext_i16_i32(x)) - 16;
}
static int32_t futrts_clzz32(int32_t x)
{
    return __clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return __clzll(x);
}
#else
static int32_t futrts_clzz8(int8_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz16(int16_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz32(int32_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz64(int64_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_ctzz8(int8_t x)
{
    int i = 0;
    
    for (; i < 8 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int i = 0;
    
    for (; i < 16 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int i = 0;
    
    for (; i < 32 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int i = 0;
    
    for (; i < 64 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_ctzz8(int8_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 8 : y - 1;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 16 : y - 1;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 32 : y - 1;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int y = __ffsll(x);
    
    return y == 0 ? 64 : y - 1;
}
#else
static int32_t futrts_ctzz8(int8_t x)
{
    return x == 0 ? 8 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz16(int16_t x)
{
    return x == 0 ? 16 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz32(int32_t x)
{
    return x == 0 ? 32 : __builtin_ctz(x);
}
static int32_t futrts_ctzz64(int64_t x)
{
    return x == 0 ? 64 : __builtin_ctzl(x);
}
#endif
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return fmin(x, y);
}
static inline float fmax32(float x, float y)
{
    return fmax(x, y);
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline bool cmplt32(float x, float y)
{
    return x < y;
}
static inline bool cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return (float) x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return (float) x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return (float) x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return (float) x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return (float) x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return (float) x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return (float) x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return (float) x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return (uint64_t) x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_cosh32(float x)
{
    return cosh(x);
}
static inline float futrts_sinh32(float x)
{
    return sinh(x);
}
static inline float futrts_tanh32(float x)
{
    return tanh(x);
}
static inline float futrts_acosh32(float x)
{
    return acosh(x);
}
static inline float futrts_asinh32(float x)
{
    return asinh(x);
}
static inline float futrts_atanh32(float x)
{
    return atanh(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline bool futrts_isnan32(float x)
{
    return isnan(x);
}
static inline bool futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
#ifdef __OPENCL_VERSION__
static inline float fmod32(float x, float y)
{
    return fmod(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline float futrts_floor32(float x)
{
    return floor(x);
}
static inline float futrts_ceil32(float x)
{
    return ceil(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return mix(v0, v1, t);
}
static inline float futrts_mad32(float a, float b, float c)
{
    return mad(a, b, c);
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fma(a, b, c);
}
#else
static inline float fmod32(float x, float y)
{
    return fmodf(x, y);
}
static inline float futrts_round32(float x)
{
    return rintf(x);
}
static inline float futrts_floor32(float x)
{
    return floorf(x);
}
static inline float futrts_ceil32(float x)
{
    return ceilf(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return v0 + (v1 - v0) * t;
}
static inline float futrts_mad32(float a, float b, float c)
{
    return a * b + c;
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fmaf(a, b, c);
}
#endif
// Start of atomics.h

inline int32_t atomic_add_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((int32_t*)p, x);
#else
  return atomic_add(p, x);
#endif
}

inline int32_t atomic_add_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((int32_t*)p, x);
#else
  return atomic_add(p, x);
#endif
}

inline float atomic_fadd_f32_global(volatile __global float *p, float x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((float*)p, x);
#else
  union { int32_t i; float f; } old;
  union { int32_t i; float f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg((volatile __global int32_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
#endif
}

inline float atomic_fadd_f32_local(volatile __local float *p, float x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((float*)p, x);
#else
  union { int32_t i; float f; } old;
  union { int32_t i; float f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg((volatile __local int32_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
#endif
}

inline int32_t atomic_smax_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((int32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline int32_t atomic_smax_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((int32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline int32_t atomic_smin_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((int32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline int32_t atomic_smin_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((int32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline uint32_t atomic_umax_i32_global(volatile __global uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((uint32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline uint32_t atomic_umax_i32_local(volatile __local uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((uint32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline uint32_t atomic_umin_i32_global(volatile __global uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((uint32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline uint32_t atomic_umin_i32_local(volatile __local uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((uint32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline int32_t atomic_and_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAnd((int32_t*)p, x);
#else
  return atomic_and(p, x);
#endif
}

inline int32_t atomic_and_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAnd((int32_t*)p, x);
#else
  return atomic_and(p, x);
#endif
}

inline int32_t atomic_or_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicOr((int32_t*)p, x);
#else
  return atomic_or(p, x);
#endif
}

inline int32_t atomic_or_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicOr((int32_t*)p, x);
#else
  return atomic_or(p, x);
#endif
}

inline int32_t atomic_xor_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicXor((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xor_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicXor((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xchg_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicExch((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xchg_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicExch((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_cmpxchg_i32_global(volatile __global int32_t *p,
                                         int32_t cmp, int32_t val) {
#ifdef FUTHARK_CUDA
  return atomicCAS((int32_t*)p, cmp, val);
#else
  return atomic_cmpxchg(p, cmp, val);
#endif
}

inline int32_t atomic_cmpxchg_i32_local(volatile __local int32_t *p,
                                         int32_t cmp, int32_t val) {
#ifdef FUTHARK_CUDA
  return atomicCAS((int32_t*)p, cmp, val);
#else
  return atomic_cmpxchg(p, cmp, val);
#endif
}

// End of atomics.h




__kernel void builtinzhreplicate_f32zireplicate_24498(__global
                                                      unsigned char *mem_24494,
                                                      int32_t num_elems_24495,
                                                      float val_24496)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_24498;
    int32_t replicate_ltid_24499;
    int32_t replicate_gid_24500;
    
    replicate_gtid_24498 = get_global_id(0);
    replicate_ltid_24499 = get_local_id(0);
    replicate_gid_24500 = get_group_id(0);
    if (slt64(replicate_gtid_24498, sext_i32_i64(num_elems_24495))) {
        ((__global float *) mem_24494)[sext_i32_i64(replicate_gtid_24498)] =
            val_24496;
    }
    
  error_0:
    return;
}
__kernel void builtinzhreplicate_i32zireplicate_24507(__global
                                                      unsigned char *mem_24503,
                                                      int32_t num_elems_24504,
                                                      int32_t val_24505)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_24507;
    int32_t replicate_ltid_24508;
    int32_t replicate_gid_24509;
    
    replicate_gtid_24507 = get_global_id(0);
    replicate_ltid_24508 = get_local_id(0);
    replicate_gid_24509 = get_group_id(0);
    if (slt64(replicate_gtid_24507, sext_i32_i64(num_elems_24504))) {
        ((__global int32_t *) mem_24503)[sext_i32_i64(replicate_gtid_24507)] =
            val_24505;
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32(__local volatile
                                    int64_t *block_9_backing_aligned_0,
                                    int32_t destoffset_1, int32_t srcoffset_3,
                                    int32_t num_arrays_4, int32_t x_elems_5,
                                    int32_t y_elems_6, int32_t mulx_7,
                                    int32_t muly_8, __global
                                    unsigned char *destmem_0, __global
                                    unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_global_id_0_37;
    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;
    
    if (slt32(x_index_31, x_elems_5)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, y_elems_6)) {
                ((__local float *) block_9)[sext_i32_i64((get_local_id_1_39 +
                                                          j_43 * 8) * 33 +
                                            get_local_id_0_38)] = ((__global
                                                                    float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                                       index_in_35)];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;
    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;
    if (slt32(x_index_31, y_elems_6)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, x_elems_5)) {
                ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                               index_out_36)] = ((__local
                                                                  float *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                                    33 +
                                                                                    get_local_id_1_39 +
                                                                                    j_43 *
                                                                                    8)];
            }
        }
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32_low_height(__local volatile
                                               int64_t *block_9_backing_aligned_0,
                                               int32_t destoffset_1,
                                               int32_t srcoffset_3,
                                               int32_t num_arrays_4,
                                               int32_t x_elems_5,
                                               int32_t y_elems_6,
                                               int32_t mulx_7, int32_t muly_8,
                                               __global
                                               unsigned char *destmem_0,
                                               __global unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_0_38 +
            srem32(get_local_id_1_39, mulx_7) * 16;
    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,
                                                          mulx_7);
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local float *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +
                                    get_local_id_0_38)] = ((__global
                                                            float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                               index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_7);
    y_index_32 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_1_39 +
        srem32(get_local_id_0_38, mulx_7) * 16;
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                       index_out_36)] = ((__local
                                                          float *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                            17 +
                                                                            get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32_low_width(__local volatile
                                              int64_t *block_9_backing_aligned_0,
                                              int32_t destoffset_1,
                                              int32_t srcoffset_3,
                                              int32_t num_arrays_4,
                                              int32_t x_elems_5,
                                              int32_t y_elems_6, int32_t mulx_7,
                                              int32_t muly_8, __global
                                              unsigned char *destmem_0, __global
                                              unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,
                                                          muly_8);
    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_8 + get_local_id_1_39 +
            srem32(get_local_id_0_38, muly_8) * 16;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local float *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +
                                    get_local_id_0_38)] = ((__global
                                                            float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                               index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 * muly_8 + get_local_id_0_38 +
        srem32(get_local_id_1_39, muly_8) * 16;
    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_8);
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                       index_out_36)] = ((__local
                                                          float *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                            17 +
                                                                            get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32_small(__local volatile
                                          int64_t *block_9_backing_aligned_0,
                                          int32_t destoffset_1,
                                          int32_t srcoffset_3,
                                          int32_t num_arrays_4,
                                          int32_t x_elems_5, int32_t y_elems_6,
                                          int32_t mulx_7, int32_t muly_8,
                                          __global unsigned char *destmem_0,
                                          __global unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *
                                          x_elems_5) * (y_elems_6 * x_elems_5);
    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *
                                        x_elems_5), y_elems_6);
    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;
    
    if (slt32(get_global_id_0_37, x_elems_5 * y_elems_6 * num_arrays_4)) {
        ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                       index_out_36)] = ((__global
                                                          float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                             index_in_35)];
    }
    
  error_0:
    return;
}
__kernel void mainzicopy_24205(int32_t m_18055, int32_t nm_18194,
                               int32_t ctx_param_ext_23378,
                               int32_t ctx_param_ext_23379,
                               int32_t ctx_param_ext_23381, __global
                               unsigned char *mem_param_23383, __global
                               unsigned char *mem_23390)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_24205;
    int32_t copy_ltid_24206;
    int32_t copy_gid_24207;
    
    copy_gtid_24205 = get_global_id(0);
    copy_ltid_24206 = get_local_id(0);
    copy_gid_24207 = get_group_id(0);
    if (slt32(copy_gtid_24205, sext_i64_i32(sext_i32_i64(m_18055) *
              sext_i32_i64(nm_18194)))) {
        ((__global float *) mem_23390)[sext_i32_i64(copy_gtid_24205 -
                                       squot32(copy_gtid_24205, nm_18194) *
                                       nm_18194) * sext_i32_i64(m_18055) +
                                       sext_i32_i64(squot32(copy_gtid_24205,
                                                            nm_18194))] =
            ((__global
              float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                        (sext_i32_i64(squot32(copy_gtid_24205,
                                                              nm_18194)) *
                                         sext_i32_i64(ctx_param_ext_23379) +
                                         sext_i32_i64(copy_gtid_24205 -
                                         squot32(copy_gtid_24205, nm_18194) *
                                         nm_18194) *
                                         sext_i32_i64(ctx_param_ext_23381))];
    }
    
  error_0:
    return;
}
__kernel void mainzicopy_24582(int32_t N_18054, int32_t m_18055,
                               int32_t i_18298, __global
                               unsigned char *mem_23777, __global
                               unsigned char *mem_23785)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_24582;
    int32_t copy_ltid_24583;
    int32_t copy_gid_24584;
    
    copy_gtid_24582 = get_global_id(0);
    copy_ltid_24583 = get_local_id(0);
    copy_gid_24584 = get_group_id(0);
    if (slt32(copy_gtid_24582, sext_i64_i32(sext_i32_i64(m_18055)))) {
        ((__global int32_t *) mem_23785)[sext_i32_i64(copy_gtid_24582)] =
            ((__global int32_t *) mem_23777)[sext_i32_i64(i_18298) +
                                             sext_i32_i64(copy_gtid_24582) *
                                             sext_i32_i64(N_18054)];
    }
    
  error_0:
    return;
}
__kernel void mainziscan_stage1_21202(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_24536_backing_aligned_0,
                                      int32_t N_18054, int32_t m_18055,
                                      int32_t N_18056, __global
                                      unsigned char *images_mem_23189, __global
                                      unsigned char *res_mem_23733, __global
                                      unsigned char *mem_23777, __global
                                      unsigned char *mem_23782,
                                      int32_t num_threads_24530)
{
    #define segscan_group_sizze_21221 (mainzisegscan_group_sizze_21196)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_24536_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_24536_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24531;
    int32_t local_tid_24532;
    int32_t group_sizze_24535;
    int32_t wave_sizze_24534;
    int32_t group_tid_24533;
    
    global_tid_24531 = get_global_id(0);
    local_tid_24532 = get_local_id(0);
    group_sizze_24535 = get_local_size(0);
    wave_sizze_24534 = LOCKSTEP_WIDTH;
    group_tid_24533 = get_group_id(0);
    
    int32_t phys_tid_21202;
    
    phys_tid_21202 = global_tid_24531;
    
    __local char *scan_arr_mem_24536;
    
    scan_arr_mem_24536 = (__local char *) scan_arr_mem_24536_backing_0;
    
    int32_t x_21226;
    int32_t x_21227;
    
    x_21226 = 0;
    for (int32_t j_24538 = 0; j_24538 < sdiv_up32(m_18055 * N_18054,
                                                  num_threads_24530);
         j_24538++) {
        int32_t chunk_offset_24539 = segscan_group_sizze_21221 * j_24538 +
                group_tid_24533 * (segscan_group_sizze_21221 *
                                   sdiv_up32(m_18055 * N_18054,
                                             num_threads_24530));
        int32_t flat_idx_24540 = chunk_offset_24539 + local_tid_24532;
        int32_t gtid_21191 = squot32(flat_idx_24540, N_18054);
        int32_t gtid_21201 = flat_idx_24540 - squot32(flat_idx_24540, N_18054) *
                N_18054;
        
        // threads in bounds read input
        {
            if (slt32(gtid_21191, m_18055) && slt32(gtid_21201, N_18054)) {
                float x_21231 = ((__global
                                  float *) images_mem_23189)[sext_i32_i64(gtid_21191) *
                                                             sext_i32_i64(N_18056) +
                                                             sext_i32_i64(gtid_21201)];
                bool res_21233;
                
                res_21233 = futrts_isnan32(x_21231);
                
                bool cond_21234 = !res_21233;
                float res_21235;
                
                if (cond_21234) {
                    float x_21232 = ((__global
                                      float *) res_mem_23733)[sext_i32_i64(gtid_21191) *
                                                              sext_i32_i64(N_18054) +
                                                              sext_i32_i64(gtid_21201)];
                    float res_21236 = x_21231 - x_21232;
                    
                    res_21235 = res_21236;
                } else {
                    res_21235 = NAN;
                }
                
                bool res_21237;
                
                res_21237 = futrts_isnan32(res_21235);
                
                bool res_21238 = !res_21237;
                int32_t res_21239 = btoi_bool_i32(res_21238);
                
                // write to-scan values to parameters
                {
                    x_21227 = res_21239;
                }
                // write mapped values results to global memory
                {
                    ((__global float *) mem_23782)[sext_i32_i64(gtid_21191) *
                                                   sext_i32_i64(N_18054) +
                                                   sext_i32_i64(gtid_21201)] =
                        res_21235;
                }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_21191, m_18055) && slt32(gtid_21201,
                                                          N_18054))) {
                    x_21227 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                int32_t res_21228 = add32(x_21226, x_21227);
                
                ((__local
                  int32_t *) scan_arr_mem_24536)[sext_i32_i64(local_tid_24532)] =
                    res_21228;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_24541;
            int32_t x_24542;
            int32_t x_24544;
            int32_t x_24545;
            int32_t skip_threads_24547;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_24532, segscan_group_sizze_21221)) {
                    x_24542 = ((volatile __local
                                int32_t *) scan_arr_mem_24536)[sext_i32_i64(local_tid_24532)];
                    if ((local_tid_24532 - squot32(local_tid_24532, 32) * 32) ==
                        0) {
                        x_24541 = x_24542;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_24547 = 1;
                while (slt32(skip_threads_24547, 32)) {
                    if (sle32(skip_threads_24547, local_tid_24532 -
                              squot32(local_tid_24532, 32) * 32) &&
                        slt32(local_tid_24532, segscan_group_sizze_21221)) {
                        // read operands
                        {
                            x_24541 = ((volatile __local
                                        int32_t *) scan_arr_mem_24536)[sext_i32_i64(local_tid_24532 -
                                                                       skip_threads_24547)];
                        }
                        // perform operation
                        {
                            bool inactive_24548 = slt32(srem32(local_tid_24532 +
                                                               chunk_offset_24539,
                                                               N_18054),
                                                        local_tid_24532 +
                                                        chunk_offset_24539 -
                                                        (local_tid_24532 -
                                                         skip_threads_24547 +
                                                         chunk_offset_24539));
                            
                            if (inactive_24548) {
                                x_24541 = x_24542;
                            }
                            if (!inactive_24548) {
                                int32_t res_24543 = add32(x_24541, x_24542);
                                
                                x_24541 = res_24543;
                            }
                        }
                    }
                    if (sle32(wave_sizze_24534, skip_threads_24547)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_24547, local_tid_24532 -
                              squot32(local_tid_24532, 32) * 32) &&
                        slt32(local_tid_24532, segscan_group_sizze_21221)) {
                        // write result
                        {
                            ((volatile __local
                              int32_t *) scan_arr_mem_24536)[sext_i32_i64(local_tid_24532)] =
                                x_24541;
                            x_24542 = x_24541;
                        }
                    }
                    if (sle32(wave_sizze_24534, skip_threads_24547)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_24547 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_24532 - squot32(local_tid_24532, 32) * 32) ==
                    31 && slt32(local_tid_24532, segscan_group_sizze_21221)) {
                    ((volatile __local
                      int32_t *) scan_arr_mem_24536)[sext_i32_i64(squot32(local_tid_24532,
                                                                          32))] =
                        x_24541;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_24549;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_24532, 32) == 0 &&
                        slt32(local_tid_24532, segscan_group_sizze_21221)) {
                        x_24545 = ((volatile __local
                                    int32_t *) scan_arr_mem_24536)[sext_i32_i64(local_tid_24532)];
                        if ((local_tid_24532 - squot32(local_tid_24532, 32) *
                             32) == 0) {
                            x_24544 = x_24545;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24549 = 1;
                    while (slt32(skip_threads_24549, 32)) {
                        if (sle32(skip_threads_24549, local_tid_24532 -
                                  squot32(local_tid_24532, 32) * 32) &&
                            (squot32(local_tid_24532, 32) == 0 &&
                             slt32(local_tid_24532,
                                   segscan_group_sizze_21221))) {
                            // read operands
                            {
                                x_24544 = ((volatile __local
                                            int32_t *) scan_arr_mem_24536)[sext_i32_i64(local_tid_24532 -
                                                                           skip_threads_24549)];
                            }
                            // perform operation
                            {
                                bool inactive_24550 =
                                     slt32(srem32(local_tid_24532 * 32 + 32 -
                                                  1 + chunk_offset_24539,
                                                  N_18054), local_tid_24532 *
                                           32 + 32 - 1 + chunk_offset_24539 -
                                           ((local_tid_24532 -
                                             skip_threads_24549) * 32 + 32 - 1 +
                                            chunk_offset_24539));
                                
                                if (inactive_24550) {
                                    x_24544 = x_24545;
                                }
                                if (!inactive_24550) {
                                    int32_t res_24546 = add32(x_24544, x_24545);
                                    
                                    x_24544 = res_24546;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24534, skip_threads_24549)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24549, local_tid_24532 -
                                  squot32(local_tid_24532, 32) * 32) &&
                            (squot32(local_tid_24532, 32) == 0 &&
                             slt32(local_tid_24532,
                                   segscan_group_sizze_21221))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) scan_arr_mem_24536)[sext_i32_i64(local_tid_24532)] =
                                    x_24544;
                                x_24545 = x_24544;
                            }
                        }
                        if (sle32(wave_sizze_24534, skip_threads_24549)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24549 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_24532, 32) == 0 ||
                      !slt32(local_tid_24532, segscan_group_sizze_21221))) {
                    // read operands
                    {
                        x_24542 = x_24541;
                        x_24541 = ((__local
                                    int32_t *) scan_arr_mem_24536)[sext_i32_i64(squot32(local_tid_24532,
                                                                                        32) -
                                                                   1)];
                    }
                    // perform operation
                    {
                        bool inactive_24551 = slt32(srem32(local_tid_24532 +
                                                           chunk_offset_24539,
                                                           N_18054),
                                                    local_tid_24532 +
                                                    chunk_offset_24539 -
                                                    (squot32(local_tid_24532,
                                                             32) * 32 - 1 +
                                                     chunk_offset_24539));
                        
                        if (inactive_24551) {
                            x_24541 = x_24542;
                        }
                        if (!inactive_24551) {
                            int32_t res_24543 = add32(x_24541, x_24542);
                            
                            x_24541 = res_24543;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          int32_t *) scan_arr_mem_24536)[sext_i32_i64(local_tid_24532)] =
                            x_24541;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_24532, 32) == 0) {
                    ((__local
                      int32_t *) scan_arr_mem_24536)[sext_i32_i64(local_tid_24532)] =
                        x_24542;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_21191, m_18055) && slt32(gtid_21201, N_18054)) {
                    ((__global int32_t *) mem_23777)[sext_i32_i64(gtid_21191) *
                                                     sext_i32_i64(N_18054) +
                                                     sext_i32_i64(gtid_21201)] =
                        ((__local
                          int32_t *) scan_arr_mem_24536)[sext_i32_i64(local_tid_24532)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_24552 = slt32(srem32(chunk_offset_24539 +
                                                          segscan_group_sizze_21221,
                                                          N_18054),
                                                   chunk_offset_24539 +
                                                   segscan_group_sizze_21221 -
                                                   (chunk_offset_24539 +
                                                    segscan_group_sizze_21221 -
                                                    1));
                bool should_load_carry_24553 = local_tid_24532 == 0 &&
                     !crosses_segment_24552;
                
                if (should_load_carry_24553) {
                    x_21226 = ((__local
                                int32_t *) scan_arr_mem_24536)[sext_i32_i64(segscan_group_sizze_21221 -
                                                               1)];
                }
                if (!should_load_carry_24553) {
                    x_21226 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_21221
}
__kernel void mainziscan_stage1_22167(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_24887_backing_aligned_0,
                                      int32_t N_18054, int32_t m_18055,
                                      int32_t iota_arg_18403, __global
                                      unsigned char *res_mem_23788, __global
                                      unsigned char *res_mem_23840, __global
                                      unsigned char *res_mem_23841, __global
                                      unsigned char *res_mem_23854, __global
                                      unsigned char *mem_23877, __global
                                      unsigned char *mem_23883,
                                      int32_t num_threads_24881)
{
    #define segscan_group_sizze_22249 (mainzisegscan_group_sizze_22161)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_24887_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_24887_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24882;
    int32_t local_tid_24883;
    int32_t group_sizze_24886;
    int32_t wave_sizze_24885;
    int32_t group_tid_24884;
    
    global_tid_24882 = get_global_id(0);
    local_tid_24883 = get_local_id(0);
    group_sizze_24886 = get_local_size(0);
    wave_sizze_24885 = LOCKSTEP_WIDTH;
    group_tid_24884 = get_group_id(0);
    
    int32_t phys_tid_22167;
    
    phys_tid_22167 = global_tid_24882;
    
    __local char *scan_arr_mem_24887;
    
    scan_arr_mem_24887 = (__local char *) scan_arr_mem_24887_backing_0;
    
    float x_22253;
    float x_22254;
    
    x_22253 = 0.0F;
    for (int32_t j_24889 = 0; j_24889 < sdiv_up32(m_18055 * iota_arg_18403,
                                                  num_threads_24881);
         j_24889++) {
        int32_t chunk_offset_24890 = segscan_group_sizze_22249 * j_24889 +
                group_tid_24884 * (segscan_group_sizze_22249 *
                                   sdiv_up32(m_18055 * iota_arg_18403,
                                             num_threads_24881));
        int32_t flat_idx_24891 = chunk_offset_24890 + local_tid_24883;
        int32_t gtid_22156 = squot32(flat_idx_24891, iota_arg_18403);
        int32_t gtid_22166 = flat_idx_24891 - squot32(flat_idx_24891,
                                                      iota_arg_18403) *
                iota_arg_18403;
        
        // threads in bounds read input
        {
            if (slt32(gtid_22156, m_18055) && slt32(gtid_22166,
                                                    iota_arg_18403)) {
                int32_t y_22260 = ((__global
                                    int32_t *) mem_23877)[sext_i32_i64(gtid_22156)];
                bool cond_22263 = sle32(y_22260, gtid_22166);
                float res_22264;
                
                if (cond_22263) {
                    res_22264 = 0.0F;
                } else {
                    int32_t x_22256 = ((__global
                                        int32_t *) res_mem_23841)[sext_i32_i64(gtid_22156)];
                    int32_t x_22257 = ((__global
                                        int32_t *) res_mem_23840)[sext_i32_i64(gtid_22156)];
                    float x_22258 = ((__global
                                      float *) res_mem_23854)[sext_i32_i64(gtid_22156)];
                    bool cond_22265 = gtid_22166 == 0;
                    float res_22266;
                    
                    if (cond_22265) {
                        res_22266 = x_22258;
                    } else {
                        int32_t x_22267 = sub32(x_22256, x_22257);
                        int32_t i_22268 = add32(gtid_22166, x_22267);
                        float negate_arg_22269 = ((__global
                                                   float *) res_mem_23788)[sext_i32_i64(gtid_22156) *
                                                                           sext_i32_i64(N_18054) +
                                                                           sext_i32_i64(i_22268)];
                        float x_22270 = 0.0F - negate_arg_22269;
                        int32_t i_22271 = add32(gtid_22166, x_22256);
                        float y_22272 = ((__global
                                          float *) res_mem_23788)[sext_i32_i64(gtid_22156) *
                                                                  sext_i32_i64(N_18054) +
                                                                  sext_i32_i64(i_22271)];
                        float res_22273 = x_22270 + y_22272;
                        
                        res_22266 = res_22273;
                    }
                    res_22264 = res_22266;
                }
                // write to-scan values to parameters
                {
                    x_22254 = res_22264;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_22156, m_18055) && slt32(gtid_22166,
                                                          iota_arg_18403))) {
                    x_22254 = 0.0F;
                }
            }
            // combine with carry and write to local memory
            {
                float res_22255 = x_22253 + x_22254;
                
                ((__local
                  float *) scan_arr_mem_24887)[sext_i32_i64(local_tid_24883)] =
                    res_22255;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            float x_24892;
            float x_24893;
            float x_24895;
            float x_24896;
            int32_t skip_threads_24898;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_24883, segscan_group_sizze_22249)) {
                    x_24893 = ((volatile __local
                                float *) scan_arr_mem_24887)[sext_i32_i64(local_tid_24883)];
                    if ((local_tid_24883 - squot32(local_tid_24883, 32) * 32) ==
                        0) {
                        x_24892 = x_24893;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_24898 = 1;
                while (slt32(skip_threads_24898, 32)) {
                    if (sle32(skip_threads_24898, local_tid_24883 -
                              squot32(local_tid_24883, 32) * 32) &&
                        slt32(local_tid_24883, segscan_group_sizze_22249)) {
                        // read operands
                        {
                            x_24892 = ((volatile __local
                                        float *) scan_arr_mem_24887)[sext_i32_i64(local_tid_24883 -
                                                                     skip_threads_24898)];
                        }
                        // perform operation
                        {
                            bool inactive_24899 = slt32(srem32(local_tid_24883 +
                                                               chunk_offset_24890,
                                                               iota_arg_18403),
                                                        local_tid_24883 +
                                                        chunk_offset_24890 -
                                                        (local_tid_24883 -
                                                         skip_threads_24898 +
                                                         chunk_offset_24890));
                            
                            if (inactive_24899) {
                                x_24892 = x_24893;
                            }
                            if (!inactive_24899) {
                                float res_24894 = x_24892 + x_24893;
                                
                                x_24892 = res_24894;
                            }
                        }
                    }
                    if (sle32(wave_sizze_24885, skip_threads_24898)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_24898, local_tid_24883 -
                              squot32(local_tid_24883, 32) * 32) &&
                        slt32(local_tid_24883, segscan_group_sizze_22249)) {
                        // write result
                        {
                            ((volatile __local
                              float *) scan_arr_mem_24887)[sext_i32_i64(local_tid_24883)] =
                                x_24892;
                            x_24893 = x_24892;
                        }
                    }
                    if (sle32(wave_sizze_24885, skip_threads_24898)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_24898 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_24883 - squot32(local_tid_24883, 32) * 32) ==
                    31 && slt32(local_tid_24883, segscan_group_sizze_22249)) {
                    ((volatile __local
                      float *) scan_arr_mem_24887)[sext_i32_i64(squot32(local_tid_24883,
                                                                        32))] =
                        x_24892;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_24900;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_24883, 32) == 0 &&
                        slt32(local_tid_24883, segscan_group_sizze_22249)) {
                        x_24896 = ((volatile __local
                                    float *) scan_arr_mem_24887)[sext_i32_i64(local_tid_24883)];
                        if ((local_tid_24883 - squot32(local_tid_24883, 32) *
                             32) == 0) {
                            x_24895 = x_24896;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24900 = 1;
                    while (slt32(skip_threads_24900, 32)) {
                        if (sle32(skip_threads_24900, local_tid_24883 -
                                  squot32(local_tid_24883, 32) * 32) &&
                            (squot32(local_tid_24883, 32) == 0 &&
                             slt32(local_tid_24883,
                                   segscan_group_sizze_22249))) {
                            // read operands
                            {
                                x_24895 = ((volatile __local
                                            float *) scan_arr_mem_24887)[sext_i32_i64(local_tid_24883 -
                                                                         skip_threads_24900)];
                            }
                            // perform operation
                            {
                                bool inactive_24901 =
                                     slt32(srem32(local_tid_24883 * 32 + 32 -
                                                  1 + chunk_offset_24890,
                                                  iota_arg_18403),
                                           local_tid_24883 * 32 + 32 - 1 +
                                           chunk_offset_24890 -
                                           ((local_tid_24883 -
                                             skip_threads_24900) * 32 + 32 - 1 +
                                            chunk_offset_24890));
                                
                                if (inactive_24901) {
                                    x_24895 = x_24896;
                                }
                                if (!inactive_24901) {
                                    float res_24897 = x_24895 + x_24896;
                                    
                                    x_24895 = res_24897;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24885, skip_threads_24900)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24900, local_tid_24883 -
                                  squot32(local_tid_24883, 32) * 32) &&
                            (squot32(local_tid_24883, 32) == 0 &&
                             slt32(local_tid_24883,
                                   segscan_group_sizze_22249))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) scan_arr_mem_24887)[sext_i32_i64(local_tid_24883)] =
                                    x_24895;
                                x_24896 = x_24895;
                            }
                        }
                        if (sle32(wave_sizze_24885, skip_threads_24900)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24900 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_24883, 32) == 0 ||
                      !slt32(local_tid_24883, segscan_group_sizze_22249))) {
                    // read operands
                    {
                        x_24893 = x_24892;
                        x_24892 = ((__local
                                    float *) scan_arr_mem_24887)[sext_i32_i64(squot32(local_tid_24883,
                                                                                      32) -
                                                                 1)];
                    }
                    // perform operation
                    {
                        bool inactive_24902 = slt32(srem32(local_tid_24883 +
                                                           chunk_offset_24890,
                                                           iota_arg_18403),
                                                    local_tid_24883 +
                                                    chunk_offset_24890 -
                                                    (squot32(local_tid_24883,
                                                             32) * 32 - 1 +
                                                     chunk_offset_24890));
                        
                        if (inactive_24902) {
                            x_24892 = x_24893;
                        }
                        if (!inactive_24902) {
                            float res_24894 = x_24892 + x_24893;
                            
                            x_24892 = res_24894;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          float *) scan_arr_mem_24887)[sext_i32_i64(local_tid_24883)] =
                            x_24892;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_24883, 32) == 0) {
                    ((__local
                      float *) scan_arr_mem_24887)[sext_i32_i64(local_tid_24883)] =
                        x_24893;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_22156, m_18055) && slt32(gtid_22166,
                                                        iota_arg_18403)) {
                    ((__global float *) mem_23883)[sext_i32_i64(gtid_22156) *
                                                   sext_i32_i64(iota_arg_18403) +
                                                   sext_i32_i64(gtid_22166)] =
                        ((__local
                          float *) scan_arr_mem_24887)[sext_i32_i64(local_tid_24883)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_24903 = slt32(srem32(chunk_offset_24890 +
                                                          segscan_group_sizze_22249,
                                                          iota_arg_18403),
                                                   chunk_offset_24890 +
                                                   segscan_group_sizze_22249 -
                                                   (chunk_offset_24890 +
                                                    segscan_group_sizze_22249 -
                                                    1));
                bool should_load_carry_24904 = local_tid_24883 == 0 &&
                     !crosses_segment_24903;
                
                if (should_load_carry_24904) {
                    x_22253 = ((__local
                                float *) scan_arr_mem_24887)[sext_i32_i64(segscan_group_sizze_22249 -
                                                             1)];
                }
                if (!should_load_carry_24904) {
                    x_22253 = 0.0F;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_22249
}
__kernel void mainziscan_stage2_21202(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_24559_backing_aligned_0,
                                      int32_t N_18054, int32_t m_18055, __global
                                      unsigned char *mem_23777,
                                      int32_t stage1_num_groups_24529,
                                      int32_t num_threads_24530)
{
    #define segscan_group_sizze_21221 (mainzisegscan_group_sizze_21196)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_24559_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_24559_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24554;
    int32_t local_tid_24555;
    int32_t group_sizze_24558;
    int32_t wave_sizze_24557;
    int32_t group_tid_24556;
    
    global_tid_24554 = get_global_id(0);
    local_tid_24555 = get_local_id(0);
    group_sizze_24558 = get_local_size(0);
    wave_sizze_24557 = LOCKSTEP_WIDTH;
    group_tid_24556 = get_group_id(0);
    
    int32_t phys_tid_21202;
    
    phys_tid_21202 = global_tid_24554;
    
    __local char *scan_arr_mem_24559;
    
    scan_arr_mem_24559 = (__local char *) scan_arr_mem_24559_backing_0;
    
    int32_t flat_idx_24561;
    
    flat_idx_24561 = (local_tid_24555 + 1) * (segscan_group_sizze_21221 *
                                              sdiv_up32(m_18055 * N_18054,
                                                        num_threads_24530)) - 1;
    
    int32_t gtid_21191;
    
    gtid_21191 = squot32(flat_idx_24561, N_18054);
    
    int32_t gtid_21201;
    
    gtid_21201 = flat_idx_24561 - squot32(flat_idx_24561, N_18054) * N_18054;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_21191, m_18055) && slt32(gtid_21201, N_18054)) {
            ((__local
              int32_t *) scan_arr_mem_24559)[sext_i32_i64(local_tid_24555)] =
                ((__global int32_t *) mem_23777)[sext_i32_i64(gtid_21191) *
                                                 sext_i32_i64(N_18054) +
                                                 sext_i32_i64(gtid_21201)];
        } else {
            ((__local
              int32_t *) scan_arr_mem_24559)[sext_i32_i64(local_tid_24555)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t x_21226;
    int32_t x_21227;
    int32_t x_24562;
    int32_t x_24563;
    int32_t skip_threads_24565;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_24555, stage1_num_groups_24529)) {
            x_21227 = ((volatile __local
                        int32_t *) scan_arr_mem_24559)[sext_i32_i64(local_tid_24555)];
            if ((local_tid_24555 - squot32(local_tid_24555, 32) * 32) == 0) {
                x_21226 = x_21227;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_24565 = 1;
        while (slt32(skip_threads_24565, 32)) {
            if (sle32(skip_threads_24565, local_tid_24555 -
                      squot32(local_tid_24555, 32) * 32) &&
                slt32(local_tid_24555, stage1_num_groups_24529)) {
                // read operands
                {
                    x_21226 = ((volatile __local
                                int32_t *) scan_arr_mem_24559)[sext_i32_i64(local_tid_24555 -
                                                               skip_threads_24565)];
                }
                // perform operation
                {
                    bool inactive_24566 = slt32(srem32((local_tid_24555 + 1) *
                                                       (segscan_group_sizze_21221 *
                                                        sdiv_up32(m_18055 *
                                                                  N_18054,
                                                                  num_threads_24530)) -
                                                       1, N_18054),
                                                (local_tid_24555 + 1) *
                                                (segscan_group_sizze_21221 *
                                                 sdiv_up32(m_18055 * N_18054,
                                                           num_threads_24530)) -
                                                1 - ((local_tid_24555 -
                                                      skip_threads_24565 + 1) *
                                                     (segscan_group_sizze_21221 *
                                                      sdiv_up32(m_18055 *
                                                                N_18054,
                                                                num_threads_24530)) -
                                                     1));
                    
                    if (inactive_24566) {
                        x_21226 = x_21227;
                    }
                    if (!inactive_24566) {
                        int32_t res_21228 = add32(x_21226, x_21227);
                        
                        x_21226 = res_21228;
                    }
                }
            }
            if (sle32(wave_sizze_24557, skip_threads_24565)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_24565, local_tid_24555 -
                      squot32(local_tid_24555, 32) * 32) &&
                slt32(local_tid_24555, stage1_num_groups_24529)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) scan_arr_mem_24559)[sext_i32_i64(local_tid_24555)] =
                        x_21226;
                    x_21227 = x_21226;
                }
            }
            if (sle32(wave_sizze_24557, skip_threads_24565)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_24565 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_24555 - squot32(local_tid_24555, 32) * 32) == 31 &&
            slt32(local_tid_24555, stage1_num_groups_24529)) {
            ((volatile __local
              int32_t *) scan_arr_mem_24559)[sext_i32_i64(squot32(local_tid_24555,
                                                                  32))] =
                x_21226;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_24567;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_24555, 32) == 0 && slt32(local_tid_24555,
                                                           stage1_num_groups_24529)) {
                x_24563 = ((volatile __local
                            int32_t *) scan_arr_mem_24559)[sext_i32_i64(local_tid_24555)];
                if ((local_tid_24555 - squot32(local_tid_24555, 32) * 32) ==
                    0) {
                    x_24562 = x_24563;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_24567 = 1;
            while (slt32(skip_threads_24567, 32)) {
                if (sle32(skip_threads_24567, local_tid_24555 -
                          squot32(local_tid_24555, 32) * 32) &&
                    (squot32(local_tid_24555, 32) == 0 && slt32(local_tid_24555,
                                                                stage1_num_groups_24529))) {
                    // read operands
                    {
                        x_24562 = ((volatile __local
                                    int32_t *) scan_arr_mem_24559)[sext_i32_i64(local_tid_24555 -
                                                                   skip_threads_24567)];
                    }
                    // perform operation
                    {
                        bool inactive_24568 = slt32(srem32((local_tid_24555 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_21221 *
                                                            sdiv_up32(m_18055 *
                                                                      N_18054,
                                                                      num_threads_24530)) -
                                                           1, N_18054),
                                                    (local_tid_24555 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_21221 *
                                                     sdiv_up32(m_18055 *
                                                               N_18054,
                                                               num_threads_24530)) -
                                                    1 - (((local_tid_24555 -
                                                           skip_threads_24567) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_21221 *
                                                          sdiv_up32(m_18055 *
                                                                    N_18054,
                                                                    num_threads_24530)) -
                                                         1));
                        
                        if (inactive_24568) {
                            x_24562 = x_24563;
                        }
                        if (!inactive_24568) {
                            int32_t res_24564 = add32(x_24562, x_24563);
                            
                            x_24562 = res_24564;
                        }
                    }
                }
                if (sle32(wave_sizze_24557, skip_threads_24567)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_24567, local_tid_24555 -
                          squot32(local_tid_24555, 32) * 32) &&
                    (squot32(local_tid_24555, 32) == 0 && slt32(local_tid_24555,
                                                                stage1_num_groups_24529))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) scan_arr_mem_24559)[sext_i32_i64(local_tid_24555)] =
                            x_24562;
                        x_24563 = x_24562;
                    }
                }
                if (sle32(wave_sizze_24557, skip_threads_24567)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_24567 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_24555, 32) == 0 || !slt32(local_tid_24555,
                                                          stage1_num_groups_24529))) {
            // read operands
            {
                x_21227 = x_21226;
                x_21226 = ((__local
                            int32_t *) scan_arr_mem_24559)[sext_i32_i64(squot32(local_tid_24555,
                                                                                32) -
                                                           1)];
            }
            // perform operation
            {
                bool inactive_24569 = slt32(srem32((local_tid_24555 + 1) *
                                                   (segscan_group_sizze_21221 *
                                                    sdiv_up32(m_18055 * N_18054,
                                                              num_threads_24530)) -
                                                   1, N_18054),
                                            (local_tid_24555 + 1) *
                                            (segscan_group_sizze_21221 *
                                             sdiv_up32(m_18055 * N_18054,
                                                       num_threads_24530)) - 1 -
                                            ((squot32(local_tid_24555, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_21221 *
                                              sdiv_up32(m_18055 * N_18054,
                                                        num_threads_24530)) -
                                             1));
                
                if (inactive_24569) {
                    x_21226 = x_21227;
                }
                if (!inactive_24569) {
                    int32_t res_21228 = add32(x_21226, x_21227);
                    
                    x_21226 = res_21228;
                }
            }
            // write final result
            {
                ((__local
                  int32_t *) scan_arr_mem_24559)[sext_i32_i64(local_tid_24555)] =
                    x_21226;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_24555, 32) == 0) {
            ((__local
              int32_t *) scan_arr_mem_24559)[sext_i32_i64(local_tid_24555)] =
                x_21227;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_21191, m_18055) && slt32(gtid_21201, N_18054)) {
            ((__global int32_t *) mem_23777)[sext_i32_i64(gtid_21191) *
                                             sext_i32_i64(N_18054) +
                                             sext_i32_i64(gtid_21201)] =
                ((__local
                  int32_t *) scan_arr_mem_24559)[sext_i32_i64(local_tid_24555)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_21221
}
__kernel void mainziscan_stage2_22167(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_24910_backing_aligned_0,
                                      int32_t m_18055, int32_t iota_arg_18403,
                                      __global unsigned char *mem_23883,
                                      int32_t stage1_num_groups_24880,
                                      int32_t num_threads_24881)
{
    #define segscan_group_sizze_22249 (mainzisegscan_group_sizze_22161)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_24910_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_24910_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24905;
    int32_t local_tid_24906;
    int32_t group_sizze_24909;
    int32_t wave_sizze_24908;
    int32_t group_tid_24907;
    
    global_tid_24905 = get_global_id(0);
    local_tid_24906 = get_local_id(0);
    group_sizze_24909 = get_local_size(0);
    wave_sizze_24908 = LOCKSTEP_WIDTH;
    group_tid_24907 = get_group_id(0);
    
    int32_t phys_tid_22167;
    
    phys_tid_22167 = global_tid_24905;
    
    __local char *scan_arr_mem_24910;
    
    scan_arr_mem_24910 = (__local char *) scan_arr_mem_24910_backing_0;
    
    int32_t flat_idx_24912;
    
    flat_idx_24912 = (local_tid_24906 + 1) * (segscan_group_sizze_22249 *
                                              sdiv_up32(m_18055 *
                                                        iota_arg_18403,
                                                        num_threads_24881)) - 1;
    
    int32_t gtid_22156;
    
    gtid_22156 = squot32(flat_idx_24912, iota_arg_18403);
    
    int32_t gtid_22166;
    
    gtid_22166 = flat_idx_24912 - squot32(flat_idx_24912, iota_arg_18403) *
        iota_arg_18403;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_22156, m_18055) && slt32(gtid_22166, iota_arg_18403)) {
            ((__local
              float *) scan_arr_mem_24910)[sext_i32_i64(local_tid_24906)] =
                ((__global float *) mem_23883)[sext_i32_i64(gtid_22156) *
                                               sext_i32_i64(iota_arg_18403) +
                                               sext_i32_i64(gtid_22166)];
        } else {
            ((__local
              float *) scan_arr_mem_24910)[sext_i32_i64(local_tid_24906)] =
                0.0F;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float x_22253;
    float x_22254;
    float x_24913;
    float x_24914;
    int32_t skip_threads_24916;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_24906, stage1_num_groups_24880)) {
            x_22254 = ((volatile __local
                        float *) scan_arr_mem_24910)[sext_i32_i64(local_tid_24906)];
            if ((local_tid_24906 - squot32(local_tid_24906, 32) * 32) == 0) {
                x_22253 = x_22254;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_24916 = 1;
        while (slt32(skip_threads_24916, 32)) {
            if (sle32(skip_threads_24916, local_tid_24906 -
                      squot32(local_tid_24906, 32) * 32) &&
                slt32(local_tid_24906, stage1_num_groups_24880)) {
                // read operands
                {
                    x_22253 = ((volatile __local
                                float *) scan_arr_mem_24910)[sext_i32_i64(local_tid_24906 -
                                                             skip_threads_24916)];
                }
                // perform operation
                {
                    bool inactive_24917 = slt32(srem32((local_tid_24906 + 1) *
                                                       (segscan_group_sizze_22249 *
                                                        sdiv_up32(m_18055 *
                                                                  iota_arg_18403,
                                                                  num_threads_24881)) -
                                                       1, iota_arg_18403),
                                                (local_tid_24906 + 1) *
                                                (segscan_group_sizze_22249 *
                                                 sdiv_up32(m_18055 *
                                                           iota_arg_18403,
                                                           num_threads_24881)) -
                                                1 - ((local_tid_24906 -
                                                      skip_threads_24916 + 1) *
                                                     (segscan_group_sizze_22249 *
                                                      sdiv_up32(m_18055 *
                                                                iota_arg_18403,
                                                                num_threads_24881)) -
                                                     1));
                    
                    if (inactive_24917) {
                        x_22253 = x_22254;
                    }
                    if (!inactive_24917) {
                        float res_22255 = x_22253 + x_22254;
                        
                        x_22253 = res_22255;
                    }
                }
            }
            if (sle32(wave_sizze_24908, skip_threads_24916)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_24916, local_tid_24906 -
                      squot32(local_tid_24906, 32) * 32) &&
                slt32(local_tid_24906, stage1_num_groups_24880)) {
                // write result
                {
                    ((volatile __local
                      float *) scan_arr_mem_24910)[sext_i32_i64(local_tid_24906)] =
                        x_22253;
                    x_22254 = x_22253;
                }
            }
            if (sle32(wave_sizze_24908, skip_threads_24916)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_24916 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_24906 - squot32(local_tid_24906, 32) * 32) == 31 &&
            slt32(local_tid_24906, stage1_num_groups_24880)) {
            ((volatile __local
              float *) scan_arr_mem_24910)[sext_i32_i64(squot32(local_tid_24906,
                                                                32))] = x_22253;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_24918;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_24906, 32) == 0 && slt32(local_tid_24906,
                                                           stage1_num_groups_24880)) {
                x_24914 = ((volatile __local
                            float *) scan_arr_mem_24910)[sext_i32_i64(local_tid_24906)];
                if ((local_tid_24906 - squot32(local_tid_24906, 32) * 32) ==
                    0) {
                    x_24913 = x_24914;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_24918 = 1;
            while (slt32(skip_threads_24918, 32)) {
                if (sle32(skip_threads_24918, local_tid_24906 -
                          squot32(local_tid_24906, 32) * 32) &&
                    (squot32(local_tid_24906, 32) == 0 && slt32(local_tid_24906,
                                                                stage1_num_groups_24880))) {
                    // read operands
                    {
                        x_24913 = ((volatile __local
                                    float *) scan_arr_mem_24910)[sext_i32_i64(local_tid_24906 -
                                                                 skip_threads_24918)];
                    }
                    // perform operation
                    {
                        bool inactive_24919 = slt32(srem32((local_tid_24906 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_22249 *
                                                            sdiv_up32(m_18055 *
                                                                      iota_arg_18403,
                                                                      num_threads_24881)) -
                                                           1, iota_arg_18403),
                                                    (local_tid_24906 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_22249 *
                                                     sdiv_up32(m_18055 *
                                                               iota_arg_18403,
                                                               num_threads_24881)) -
                                                    1 - (((local_tid_24906 -
                                                           skip_threads_24918) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_22249 *
                                                          sdiv_up32(m_18055 *
                                                                    iota_arg_18403,
                                                                    num_threads_24881)) -
                                                         1));
                        
                        if (inactive_24919) {
                            x_24913 = x_24914;
                        }
                        if (!inactive_24919) {
                            float res_24915 = x_24913 + x_24914;
                            
                            x_24913 = res_24915;
                        }
                    }
                }
                if (sle32(wave_sizze_24908, skip_threads_24918)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_24918, local_tid_24906 -
                          squot32(local_tid_24906, 32) * 32) &&
                    (squot32(local_tid_24906, 32) == 0 && slt32(local_tid_24906,
                                                                stage1_num_groups_24880))) {
                    // write result
                    {
                        ((volatile __local
                          float *) scan_arr_mem_24910)[sext_i32_i64(local_tid_24906)] =
                            x_24913;
                        x_24914 = x_24913;
                    }
                }
                if (sle32(wave_sizze_24908, skip_threads_24918)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_24918 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_24906, 32) == 0 || !slt32(local_tid_24906,
                                                          stage1_num_groups_24880))) {
            // read operands
            {
                x_22254 = x_22253;
                x_22253 = ((__local
                            float *) scan_arr_mem_24910)[sext_i32_i64(squot32(local_tid_24906,
                                                                              32) -
                                                         1)];
            }
            // perform operation
            {
                bool inactive_24920 = slt32(srem32((local_tid_24906 + 1) *
                                                   (segscan_group_sizze_22249 *
                                                    sdiv_up32(m_18055 *
                                                              iota_arg_18403,
                                                              num_threads_24881)) -
                                                   1, iota_arg_18403),
                                            (local_tid_24906 + 1) *
                                            (segscan_group_sizze_22249 *
                                             sdiv_up32(m_18055 * iota_arg_18403,
                                                       num_threads_24881)) - 1 -
                                            ((squot32(local_tid_24906, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_22249 *
                                              sdiv_up32(m_18055 *
                                                        iota_arg_18403,
                                                        num_threads_24881)) -
                                             1));
                
                if (inactive_24920) {
                    x_22253 = x_22254;
                }
                if (!inactive_24920) {
                    float res_22255 = x_22253 + x_22254;
                    
                    x_22253 = res_22255;
                }
            }
            // write final result
            {
                ((__local
                  float *) scan_arr_mem_24910)[sext_i32_i64(local_tid_24906)] =
                    x_22253;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_24906, 32) == 0) {
            ((__local
              float *) scan_arr_mem_24910)[sext_i32_i64(local_tid_24906)] =
                x_22254;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_22156, m_18055) && slt32(gtid_22166, iota_arg_18403)) {
            ((__global float *) mem_23883)[sext_i32_i64(gtid_22156) *
                                           sext_i32_i64(iota_arg_18403) +
                                           sext_i32_i64(gtid_22166)] = ((__local
                                                                         float *) scan_arr_mem_24910)[sext_i32_i64(local_tid_24906)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_22249
}
__kernel void mainziscan_stage3_21202(__global int *global_failure,
                                      int32_t N_18054, int32_t m_18055,
                                      int32_t num_groups_21222, __global
                                      unsigned char *mem_23777,
                                      int32_t num_threads_24530,
                                      int32_t required_groups_24570)
{
    #define segscan_group_sizze_21221 (mainzisegscan_group_sizze_21196)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24571;
    int32_t local_tid_24572;
    int32_t group_sizze_24575;
    int32_t wave_sizze_24574;
    int32_t group_tid_24573;
    
    global_tid_24571 = get_global_id(0);
    local_tid_24572 = get_local_id(0);
    group_sizze_24575 = get_local_size(0);
    wave_sizze_24574 = LOCKSTEP_WIDTH;
    group_tid_24573 = get_group_id(0);
    
    int32_t phys_tid_21202;
    
    phys_tid_21202 = global_tid_24571;
    
    int32_t phys_group_id_24576;
    
    phys_group_id_24576 = get_group_id(0);
    for (int32_t i_24577 = 0; i_24577 < sdiv_up32(required_groups_24570 -
                                                  phys_group_id_24576,
                                                  num_groups_21222);
         i_24577++) {
        int32_t virt_group_id_24578 = phys_group_id_24576 + i_24577 *
                num_groups_21222;
        int32_t flat_idx_24579 = virt_group_id_24578 *
                segscan_group_sizze_21221 + local_tid_24572;
        int32_t gtid_21191 = squot32(flat_idx_24579, N_18054);
        int32_t gtid_21201 = flat_idx_24579 - squot32(flat_idx_24579, N_18054) *
                N_18054;
        int32_t orig_group_24580 = squot32(flat_idx_24579,
                                           segscan_group_sizze_21221 *
                                           sdiv_up32(m_18055 * N_18054,
                                                     num_threads_24530));
        int32_t carry_in_flat_idx_24581 = orig_group_24580 *
                (segscan_group_sizze_21221 * sdiv_up32(m_18055 * N_18054,
                                                       num_threads_24530)) - 1;
        
        if (slt32(gtid_21191, m_18055) && slt32(gtid_21201, N_18054)) {
            if (!(orig_group_24580 == 0 || (flat_idx_24579 ==
                                            (orig_group_24580 + 1) *
                                            (segscan_group_sizze_21221 *
                                             sdiv_up32(m_18055 * N_18054,
                                                       num_threads_24530)) -
                                            1 || slt32(srem32(flat_idx_24579,
                                                              N_18054),
                                                       flat_idx_24579 -
                                                       carry_in_flat_idx_24581)))) {
                int32_t x_21226;
                int32_t x_21227;
                
                x_21226 = ((__global
                            int32_t *) mem_23777)[sext_i32_i64(squot32(carry_in_flat_idx_24581,
                                                                       N_18054)) *
                                                  sext_i32_i64(N_18054) +
                                                  sext_i32_i64(carry_in_flat_idx_24581 -
                                                  squot32(carry_in_flat_idx_24581,
                                                          N_18054) * N_18054)];
                x_21227 = ((__global
                            int32_t *) mem_23777)[sext_i32_i64(gtid_21191) *
                                                  sext_i32_i64(N_18054) +
                                                  sext_i32_i64(gtid_21201)];
                
                int32_t res_21228;
                
                res_21228 = add32(x_21226, x_21227);
                x_21226 = res_21228;
                ((__global int32_t *) mem_23777)[sext_i32_i64(gtid_21191) *
                                                 sext_i32_i64(N_18054) +
                                                 sext_i32_i64(gtid_21201)] =
                    x_21226;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_21221
}
__kernel void mainziscan_stage3_22167(__global int *global_failure,
                                      int32_t m_18055, int32_t iota_arg_18403,
                                      int32_t num_groups_22250, __global
                                      unsigned char *mem_23883,
                                      int32_t num_threads_24881,
                                      int32_t required_groups_24921)
{
    #define segscan_group_sizze_22249 (mainzisegscan_group_sizze_22161)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24922;
    int32_t local_tid_24923;
    int32_t group_sizze_24926;
    int32_t wave_sizze_24925;
    int32_t group_tid_24924;
    
    global_tid_24922 = get_global_id(0);
    local_tid_24923 = get_local_id(0);
    group_sizze_24926 = get_local_size(0);
    wave_sizze_24925 = LOCKSTEP_WIDTH;
    group_tid_24924 = get_group_id(0);
    
    int32_t phys_tid_22167;
    
    phys_tid_22167 = global_tid_24922;
    
    int32_t phys_group_id_24927;
    
    phys_group_id_24927 = get_group_id(0);
    for (int32_t i_24928 = 0; i_24928 < sdiv_up32(required_groups_24921 -
                                                  phys_group_id_24927,
                                                  num_groups_22250);
         i_24928++) {
        int32_t virt_group_id_24929 = phys_group_id_24927 + i_24928 *
                num_groups_22250;
        int32_t flat_idx_24930 = virt_group_id_24929 *
                segscan_group_sizze_22249 + local_tid_24923;
        int32_t gtid_22156 = squot32(flat_idx_24930, iota_arg_18403);
        int32_t gtid_22166 = flat_idx_24930 - squot32(flat_idx_24930,
                                                      iota_arg_18403) *
                iota_arg_18403;
        int32_t orig_group_24931 = squot32(flat_idx_24930,
                                           segscan_group_sizze_22249 *
                                           sdiv_up32(m_18055 * iota_arg_18403,
                                                     num_threads_24881));
        int32_t carry_in_flat_idx_24932 = orig_group_24931 *
                (segscan_group_sizze_22249 * sdiv_up32(m_18055 * iota_arg_18403,
                                                       num_threads_24881)) - 1;
        
        if (slt32(gtid_22156, m_18055) && slt32(gtid_22166, iota_arg_18403)) {
            if (!(orig_group_24931 == 0 || (flat_idx_24930 ==
                                            (orig_group_24931 + 1) *
                                            (segscan_group_sizze_22249 *
                                             sdiv_up32(m_18055 * iota_arg_18403,
                                                       num_threads_24881)) -
                                            1 || slt32(srem32(flat_idx_24930,
                                                              iota_arg_18403),
                                                       flat_idx_24930 -
                                                       carry_in_flat_idx_24932)))) {
                float x_22253;
                float x_22254;
                
                x_22253 = ((__global
                            float *) mem_23883)[sext_i32_i64(squot32(carry_in_flat_idx_24932,
                                                                     iota_arg_18403)) *
                                                sext_i32_i64(iota_arg_18403) +
                                                sext_i32_i64(carry_in_flat_idx_24932 -
                                                squot32(carry_in_flat_idx_24932,
                                                        iota_arg_18403) *
                                                iota_arg_18403)];
                x_22254 = ((__global
                            float *) mem_23883)[sext_i32_i64(gtid_22156) *
                                                sext_i32_i64(iota_arg_18403) +
                                                sext_i32_i64(gtid_22166)];
                
                float res_22255;
                
                res_22255 = x_22253 + x_22254;
                x_22253 = res_22255;
                ((__global float *) mem_23883)[sext_i32_i64(gtid_22156) *
                                               sext_i32_i64(iota_arg_18403) +
                                               sext_i32_i64(gtid_22166)] =
                    x_22253;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_22249
}
__kernel void mainzisegmap_18799(__global int *global_failure, int32_t N_18054,
                                 float freq_18060, int32_t k2p2zq_18071,
                                 __global
                                 unsigned char *mappingindices_mem_23188,
                                 __global unsigned char *mem_23195)
{
    #define segmap_group_sizze_18889 (mainzisegmap_group_sizze_18804)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24064;
    int32_t local_tid_24065;
    int32_t group_sizze_24068;
    int32_t wave_sizze_24067;
    int32_t group_tid_24066;
    
    global_tid_24064 = get_global_id(0);
    local_tid_24065 = get_local_id(0);
    group_sizze_24068 = get_local_size(0);
    wave_sizze_24067 = LOCKSTEP_WIDTH;
    group_tid_24066 = get_group_id(0);
    
    int32_t phys_tid_18799;
    
    phys_tid_18799 = global_tid_24064;
    
    int32_t gtid_18797;
    
    gtid_18797 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24066) *
                                      sext_i32_i64(segmap_group_sizze_18889) +
                                      sext_i32_i64(local_tid_24065),
                                      sext_i32_i64(N_18054)));
    
    int32_t gtid_18798;
    
    gtid_18798 = sext_i64_i32(sext_i32_i64(group_tid_24066) *
        sext_i32_i64(segmap_group_sizze_18889) + sext_i32_i64(local_tid_24065) -
        squot64(sext_i32_i64(group_tid_24066) *
                sext_i32_i64(segmap_group_sizze_18889) +
                sext_i32_i64(local_tid_24065), sext_i32_i64(N_18054)) *
        sext_i32_i64(N_18054));
    if (slt32(gtid_18797, k2p2zq_18071) && slt32(gtid_18798, N_18054)) {
        bool index_primexp_22375 = gtid_18797 == 0;
        float res_18897;
        
        if (index_primexp_22375) {
            res_18897 = 1.0F;
        } else {
            int32_t x_18896 = ((__global
                                int32_t *) mappingindices_mem_23188)[sext_i32_i64(gtid_18798)];
            bool cond_18898 = gtid_18797 == 1;
            float res_18899;
            
            if (cond_18898) {
                float res_18900 = sitofp_i32_f32(x_18896);
                
                res_18899 = res_18900;
            } else {
                int32_t r32_arg_18901 = sdiv32(gtid_18797, 2);
                float res_18902 = sitofp_i32_f32(r32_arg_18901);
                float res_18903 = sitofp_i32_f32(x_18896);
                float x_18904 = 6.2831855F * res_18902;
                float x_18905 = res_18903 * x_18904;
                float angle_18906 = x_18905 / freq_18060;
                int32_t x_18907 = smod32(gtid_18797, 2);
                bool cond_18908 = x_18907 == 0;
                float res_18909;
                
                if (cond_18908) {
                    float res_18910;
                    
                    res_18910 = futrts_sin32(angle_18906);
                    res_18909 = res_18910;
                } else {
                    float res_18911;
                    
                    res_18911 = futrts_cos32(angle_18906);
                    res_18909 = res_18911;
                }
                res_18899 = res_18909;
            }
            res_18897 = res_18899;
        }
        ((__global float *) mem_23195)[sext_i32_i64(gtid_18797) *
                                       sext_i32_i64(N_18054) +
                                       sext_i32_i64(gtid_18798)] = res_18897;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_18889
}
__kernel void mainzisegmap_18999(__global int *global_failure, int32_t N_18054,
                                 float freq_18060, int32_t k2p2zq_18071,
                                 __global
                                 unsigned char *mappingindices_mem_23188,
                                 __global unsigned char *mem_23201)
{
    #define segmap_group_sizze_19085 (mainzisegmap_group_sizze_19004)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24069;
    int32_t local_tid_24070;
    int32_t group_sizze_24073;
    int32_t wave_sizze_24072;
    int32_t group_tid_24071;
    
    global_tid_24069 = get_global_id(0);
    local_tid_24070 = get_local_id(0);
    group_sizze_24073 = get_local_size(0);
    wave_sizze_24072 = LOCKSTEP_WIDTH;
    group_tid_24071 = get_group_id(0);
    
    int32_t phys_tid_18999;
    
    phys_tid_18999 = global_tid_24069;
    
    int32_t gtid_18997;
    
    gtid_18997 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24071) *
                                      sext_i32_i64(segmap_group_sizze_19085) +
                                      sext_i32_i64(local_tid_24070),
                                      sext_i32_i64(N_18054)));
    
    int32_t gtid_18998;
    
    gtid_18998 = sext_i64_i32(sext_i32_i64(group_tid_24071) *
        sext_i32_i64(segmap_group_sizze_19085) + sext_i32_i64(local_tid_24070) -
        squot64(sext_i32_i64(group_tid_24071) *
                sext_i32_i64(segmap_group_sizze_19085) +
                sext_i32_i64(local_tid_24070), sext_i32_i64(N_18054)) *
        sext_i32_i64(N_18054));
    if (slt32(gtid_18997, k2p2zq_18071) && slt32(gtid_18998, N_18054)) {
        bool index_primexp_22382 = gtid_18997 == 0;
        float res_19093;
        
        if (index_primexp_22382) {
            res_19093 = 1.0F;
        } else {
            int32_t x_19092 = ((__global
                                int32_t *) mappingindices_mem_23188)[sext_i32_i64(gtid_18998)];
            int32_t i_19094 = add32(1, gtid_18997);
            int32_t r32_arg_19095 = sdiv32(i_19094, 2);
            float res_19096 = sitofp_i32_f32(r32_arg_19095);
            float res_19097 = sitofp_i32_f32(x_19092);
            float x_19098 = 6.2831855F * res_19096;
            float x_19099 = res_19097 * x_19098;
            float angle_19100 = x_19099 / freq_18060;
            int32_t x_19101 = smod32(i_19094, 2);
            bool cond_19102 = x_19101 == 0;
            float res_19103;
            
            if (cond_19102) {
                float res_19104;
                
                res_19104 = futrts_sin32(angle_19100);
                res_19103 = res_19104;
            } else {
                float res_19105;
                
                res_19105 = futrts_cos32(angle_19100);
                res_19103 = res_19105;
            }
            res_19093 = res_19103;
        }
        ((__global float *) mem_23201)[sext_i32_i64(gtid_18997) *
                                       sext_i32_i64(N_18054) +
                                       sext_i32_i64(gtid_18998)] = res_19093;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19085
}
__kernel void mainzisegmap_19152(__global int *global_failure, int32_t N_18054,
                                 int32_t k2p2zq_18071, float res_18134, __global
                                 unsigned char *mem_23207, __global
                                 unsigned char *mem_23213)
{
    #define segmap_group_sizze_19190 (mainzisegmap_group_sizze_19157)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24074;
    int32_t local_tid_24075;
    int32_t group_sizze_24078;
    int32_t wave_sizze_24077;
    int32_t group_tid_24076;
    
    global_tid_24074 = get_global_id(0);
    local_tid_24075 = get_local_id(0);
    group_sizze_24078 = get_local_size(0);
    wave_sizze_24077 = LOCKSTEP_WIDTH;
    group_tid_24076 = get_group_id(0);
    
    int32_t phys_tid_19152;
    
    phys_tid_19152 = global_tid_24074;
    
    int32_t gtid_19150;
    
    gtid_19150 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24076) *
                                      sext_i32_i64(segmap_group_sizze_19190) +
                                      sext_i32_i64(local_tid_24075),
                                      sext_i32_i64(k2p2zq_18071)));
    
    int32_t gtid_19151;
    
    gtid_19151 = sext_i64_i32(sext_i32_i64(group_tid_24076) *
        sext_i32_i64(segmap_group_sizze_19190) + sext_i32_i64(local_tid_24075) -
        squot64(sext_i32_i64(group_tid_24076) *
                sext_i32_i64(segmap_group_sizze_19190) +
                sext_i32_i64(local_tid_24075), sext_i32_i64(k2p2zq_18071)) *
        sext_i32_i64(k2p2zq_18071));
    if (slt32(gtid_19150, N_18054) && slt32(gtid_19151, k2p2zq_18071)) {
        float x_19195 = ((__global
                          float *) mem_23207)[sext_i32_i64(gtid_19150) *
                                              sext_i32_i64(k2p2zq_18071) +
                                              sext_i32_i64(gtid_19151)];
        float res_19196 = res_18134 + x_19195;
        
        ((__global float *) mem_23213)[sext_i32_i64(gtid_19150) *
                                       sext_i32_i64(k2p2zq_18071) +
                                       sext_i32_i64(gtid_19151)] = res_19196;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19190
}
__kernel void mainzisegmap_19201(__global int *global_failure, int32_t N_18054,
                                 int32_t m_18055, int32_t n_18059,
                                 int32_t k2p2zq_18071, int32_t num_groups_19228,
                                 __global unsigned char *binop_p_mem_23202,
                                 __global unsigned char *mem_23213, __global
                                 unsigned char *mem_23218, __global
                                 unsigned char *mem_23224, __global
                                 unsigned char *mem_23271)
{
    #define segmap_group_sizze_19227 (mainzisegmap_group_sizze_19204)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24082;
    int32_t local_tid_24083;
    int32_t group_sizze_24086;
    int32_t wave_sizze_24085;
    int32_t group_tid_24084;
    
    global_tid_24082 = get_global_id(0);
    local_tid_24083 = get_local_id(0);
    group_sizze_24086 = get_local_size(0);
    wave_sizze_24085 = LOCKSTEP_WIDTH;
    group_tid_24084 = get_group_id(0);
    
    int32_t phys_tid_19201;
    
    phys_tid_19201 = global_tid_24082;
    
    int32_t phys_group_id_24087;
    
    phys_group_id_24087 = get_group_id(0);
    for (int32_t i_24088 = 0; i_24088 < sdiv_up32(sdiv_up32(m_18055,
                                                            segmap_group_sizze_19227) -
                                                  phys_group_id_24087,
                                                  num_groups_19228);
         i_24088++) {
        int32_t virt_group_id_24089 = phys_group_id_24087 + i_24088 *
                num_groups_19228;
        int32_t gtid_19200 = sext_i64_i32(sext_i32_i64(virt_group_id_24089) *
                sext_i32_i64(segmap_group_sizze_19227) +
                sext_i32_i64(local_tid_24083));
        
        if (slt32(gtid_19200, m_18055)) {
            for (int32_t i_23133 = 0; i_23133 < k2p2zq_18071; i_23133++) {
                for (int32_t i_23137 = 0; i_23137 < k2p2zq_18071; i_23137++) {
                    float res_19236;
                    float redout_23139 = 0.0F;
                    
                    for (int32_t i_23140 = 0; i_23140 < n_18059; i_23140++) {
                        float x_19240 = ((__global
                                          float *) mem_23218)[sext_i32_i64(i_23140) *
                                                              sext_i32_i64(m_18055) +
                                                              sext_i32_i64(gtid_19200)];
                        float x_19241 = ((__global
                                          float *) binop_p_mem_23202)[sext_i32_i64(i_23133) *
                                                                      sext_i32_i64(N_18054) +
                                                                      sext_i32_i64(i_23140)];
                        float x_19242 = ((__global
                                          float *) mem_23213)[sext_i32_i64(i_23140) *
                                                              sext_i32_i64(k2p2zq_18071) +
                                                              sext_i32_i64(i_23137)];
                        float x_19243 = x_19241 * x_19242;
                        bool res_19244;
                        
                        res_19244 = futrts_isnan32(x_19240);
                        
                        float y_19245;
                        
                        if (res_19244) {
                            y_19245 = 0.0F;
                        } else {
                            y_19245 = 1.0F;
                        }
                        
                        float res_19246 = x_19243 * y_19245;
                        float res_19239 = res_19246 + redout_23139;
                        float redout_tmp_24092 = res_19239;
                        
                        redout_23139 = redout_tmp_24092;
                    }
                    res_19236 = redout_23139;
                    ((__global
                      float *) mem_23224)[sext_i32_i64(phys_tid_19201) +
                                          (sext_i32_i64(i_23133) *
                                           sext_i32_i64(num_groups_19228 *
                                           segmap_group_sizze_19227 *
                                           k2p2zq_18071) +
                                           sext_i32_i64(i_23137) *
                                           sext_i32_i64(num_groups_19228 *
                                           segmap_group_sizze_19227))] =
                        res_19236;
                }
            }
            for (int32_t i_24093 = 0; i_24093 < k2p2zq_18071; i_24093++) {
                for (int32_t i_24094 = 0; i_24094 < k2p2zq_18071; i_24094++) {
                    ((__global float *) mem_23271)[sext_i32_i64(i_24093) *
                                                   sext_i32_i64(m_18055 *
                                                   k2p2zq_18071) +
                                                   sext_i32_i64(i_24094) *
                                                   sext_i32_i64(m_18055) +
                                                   sext_i32_i64(gtid_19200)] =
                        ((__global
                          float *) mem_23224)[sext_i32_i64(phys_tid_19201) +
                                              (sext_i32_i64(i_24093) *
                                               sext_i32_i64(num_groups_19228 *
                                               segmap_group_sizze_19227 *
                                               k2p2zq_18071) +
                                               sext_i32_i64(i_24094) *
                                               sext_i32_i64(num_groups_19228 *
                                               segmap_group_sizze_19227))];
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19227
}
__kernel void mainzisegmap_19249(__global int *global_failure, int32_t m_18055,
                                 int32_t N_18056, int32_t n_18059,
                                 int32_t k2p2zq_18071, int32_t num_groups_19431,
                                 __global unsigned char *images_mem_23189,
                                 __global unsigned char *mem_23207, __global
                                 unsigned char *mem_23213, __global
                                 unsigned char *mem_23275, __global
                                 unsigned char *mem_23294)
{
    #define segmap_group_sizze_19430 (mainzisegmap_group_sizze_19254)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24095;
    int32_t local_tid_24096;
    int32_t group_sizze_24099;
    int32_t wave_sizze_24098;
    int32_t group_tid_24097;
    
    global_tid_24095 = get_global_id(0);
    local_tid_24096 = get_local_id(0);
    group_sizze_24099 = get_local_size(0);
    wave_sizze_24098 = LOCKSTEP_WIDTH;
    group_tid_24097 = get_group_id(0);
    
    int32_t phys_tid_19249;
    
    phys_tid_19249 = global_tid_24095;
    
    int32_t phys_group_id_24100;
    
    phys_group_id_24100 = get_group_id(0);
    for (int32_t i_24101 = 0; i_24101 < sdiv_up32(sdiv_up32(m_18055 *
                                                            k2p2zq_18071,
                                                            segmap_group_sizze_19430) -
                                                  phys_group_id_24100,
                                                  num_groups_19431);
         i_24101++) {
        int32_t virt_group_id_24102 = phys_group_id_24100 + i_24101 *
                num_groups_19431;
        int32_t gtid_19247 =
                sext_i64_i32(squot64(sext_i32_i64(virt_group_id_24102) *
                                     sext_i32_i64(segmap_group_sizze_19430) +
                                     sext_i32_i64(local_tid_24096),
                                     sext_i32_i64(k2p2zq_18071)));
        int32_t gtid_19248 = sext_i64_i32(sext_i32_i64(virt_group_id_24102) *
                sext_i32_i64(segmap_group_sizze_19430) +
                sext_i32_i64(local_tid_24096) -
                squot64(sext_i32_i64(virt_group_id_24102) *
                        sext_i32_i64(segmap_group_sizze_19430) +
                        sext_i32_i64(local_tid_24096),
                        sext_i32_i64(k2p2zq_18071)) *
                sext_i32_i64(k2p2zq_18071));
        
        if (slt32(gtid_19247, m_18055) && slt32(gtid_19248, k2p2zq_18071)) {
            for (int32_t i_23143 = 0; i_23143 < k2p2zq_18071; i_23143++) {
                float res_19442;
                float redout_23145 = 0.0F;
                
                for (int32_t i_23146 = 0; i_23146 < n_18059; i_23146++) {
                    float x_19446 = ((__global
                                      float *) images_mem_23189)[sext_i32_i64(gtid_19247) *
                                                                 sext_i32_i64(N_18056) +
                                                                 sext_i32_i64(i_23146)];
                    float x_19447 = ((__global
                                      float *) mem_23207)[sext_i32_i64(i_23146) *
                                                          sext_i32_i64(k2p2zq_18071) +
                                                          sext_i32_i64(gtid_19248)];
                    float x_19448 = ((__global
                                      float *) mem_23213)[sext_i32_i64(i_23146) *
                                                          sext_i32_i64(k2p2zq_18071) +
                                                          sext_i32_i64(i_23143)];
                    float x_19449 = x_19447 * x_19448;
                    bool res_19450;
                    
                    res_19450 = futrts_isnan32(x_19446);
                    
                    float y_19451;
                    
                    if (res_19450) {
                        y_19451 = 0.0F;
                    } else {
                        y_19451 = 1.0F;
                    }
                    
                    float res_19452 = x_19449 * y_19451;
                    float res_19445 = res_19452 + redout_23145;
                    float redout_tmp_24104 = res_19445;
                    
                    redout_23145 = redout_tmp_24104;
                }
                res_19442 = redout_23145;
                ((__global float *) mem_23275)[sext_i32_i64(phys_tid_19249) +
                                               sext_i32_i64(i_23143) *
                                               sext_i32_i64(num_groups_19431 *
                                               segmap_group_sizze_19430)] =
                    res_19442;
            }
            for (int32_t i_24105 = 0; i_24105 < k2p2zq_18071; i_24105++) {
                ((__global float *) mem_23294)[sext_i32_i64(i_24105) *
                                               sext_i32_i64(k2p2zq_18071 *
                                               m_18055) +
                                               sext_i32_i64(gtid_19247) *
                                               sext_i32_i64(k2p2zq_18071) +
                                               sext_i32_i64(gtid_19248)] =
                    ((__global
                      float *) mem_23275)[sext_i32_i64(phys_tid_19249) +
                                          sext_i32_i64(i_24105) *
                                          sext_i32_i64(num_groups_19431 *
                                          segmap_group_sizze_19430)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19430
}
__kernel void mainzisegmap_19281(__global int *global_failure, int32_t m_18055,
                                 int32_t N_18056, int32_t n_18059,
                                 int32_t k2p2zq_18071, __global
                                 unsigned char *images_mem_23189, __global
                                 unsigned char *mem_23207, __global
                                 unsigned char *mem_23213, __global
                                 unsigned char *mem_23302)
{
    #define segmap_group_sizze_19459 (mainzisegmap_group_sizze_19288)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24106;
    int32_t local_tid_24107;
    int32_t group_sizze_24110;
    int32_t wave_sizze_24109;
    int32_t group_tid_24108;
    
    global_tid_24106 = get_global_id(0);
    local_tid_24107 = get_local_id(0);
    group_sizze_24110 = get_local_size(0);
    wave_sizze_24109 = LOCKSTEP_WIDTH;
    group_tid_24108 = get_group_id(0);
    
    int32_t phys_tid_19281;
    
    phys_tid_19281 = global_tid_24106;
    
    int32_t gtid_19278;
    
    gtid_19278 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24108) *
                                      sext_i32_i64(segmap_group_sizze_19459) +
                                      sext_i32_i64(local_tid_24107),
                                      sext_i32_i64(k2p2zq_18071) *
                                      sext_i32_i64(k2p2zq_18071)));
    
    int32_t gtid_19279;
    
    gtid_19279 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24108) *
                                      sext_i32_i64(segmap_group_sizze_19459) +
                                      sext_i32_i64(local_tid_24107) -
                                      squot64(sext_i32_i64(group_tid_24108) *
                                              sext_i32_i64(segmap_group_sizze_19459) +
                                              sext_i32_i64(local_tid_24107),
                                              sext_i32_i64(k2p2zq_18071) *
                                              sext_i32_i64(k2p2zq_18071)) *
                                      (sext_i32_i64(k2p2zq_18071) *
                                       sext_i32_i64(k2p2zq_18071)),
                                      sext_i32_i64(k2p2zq_18071)));
    
    int32_t gtid_19280;
    
    gtid_19280 = sext_i64_i32(sext_i32_i64(group_tid_24108) *
        sext_i32_i64(segmap_group_sizze_19459) + sext_i32_i64(local_tid_24107) -
        squot64(sext_i32_i64(group_tid_24108) *
                sext_i32_i64(segmap_group_sizze_19459) +
                sext_i32_i64(local_tid_24107), sext_i32_i64(k2p2zq_18071) *
                sext_i32_i64(k2p2zq_18071)) * (sext_i32_i64(k2p2zq_18071) *
                                               sext_i32_i64(k2p2zq_18071)) -
        squot64(sext_i32_i64(group_tid_24108) *
                sext_i32_i64(segmap_group_sizze_19459) +
                sext_i32_i64(local_tid_24107) -
                squot64(sext_i32_i64(group_tid_24108) *
                        sext_i32_i64(segmap_group_sizze_19459) +
                        sext_i32_i64(local_tid_24107),
                        sext_i32_i64(k2p2zq_18071) *
                        sext_i32_i64(k2p2zq_18071)) *
                (sext_i32_i64(k2p2zq_18071) * sext_i32_i64(k2p2zq_18071)),
                sext_i32_i64(k2p2zq_18071)) * sext_i32_i64(k2p2zq_18071));
    if ((slt32(gtid_19278, m_18055) && slt32(gtid_19279, k2p2zq_18071)) &&
        slt32(gtid_19280, k2p2zq_18071)) {
        float res_19472;
        float redout_23147 = 0.0F;
        
        for (int32_t i_23148 = 0; i_23148 < n_18059; i_23148++) {
            float x_19476 = ((__global
                              float *) images_mem_23189)[sext_i32_i64(gtid_19278) *
                                                         sext_i32_i64(N_18056) +
                                                         sext_i32_i64(i_23148)];
            float x_19477 = ((__global
                              float *) mem_23207)[sext_i32_i64(i_23148) *
                                                  sext_i32_i64(k2p2zq_18071) +
                                                  sext_i32_i64(gtid_19279)];
            float x_19478 = ((__global
                              float *) mem_23213)[sext_i32_i64(i_23148) *
                                                  sext_i32_i64(k2p2zq_18071) +
                                                  sext_i32_i64(gtid_19280)];
            float x_19479 = x_19477 * x_19478;
            bool res_19480;
            
            res_19480 = futrts_isnan32(x_19476);
            
            float y_19481;
            
            if (res_19480) {
                y_19481 = 0.0F;
            } else {
                y_19481 = 1.0F;
            }
            
            float res_19482 = x_19479 * y_19481;
            float res_19475 = res_19482 + redout_23147;
            float redout_tmp_24111 = res_19475;
            
            redout_23147 = redout_tmp_24111;
        }
        res_19472 = redout_23147;
        ((__global float *) mem_23302)[sext_i32_i64(gtid_19278) *
                                       sext_i32_i64(k2p2zq_18071 *
                                       k2p2zq_18071) +
                                       sext_i32_i64(gtid_19279) *
                                       sext_i32_i64(k2p2zq_18071) +
                                       sext_i32_i64(gtid_19280)] = res_19472;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19459
}
__kernel void mainzisegmap_19798(__global int *global_failure, int32_t m_18055,
                                 int32_t k2p2zq_18071, int32_t m_18193,
                                 int32_t res_r_ixfn_23422,
                                 int32_t res_r_ixfn_23423,
                                 int32_t res_r_ixfn_23425, __global
                                 unsigned char *res_r_mem_23427, __global
                                 unsigned char *mem_23435)
{
    #define segmap_group_sizze_20481 (mainzisegmap_group_sizze_19805)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24233;
    int32_t local_tid_24234;
    int32_t group_sizze_24237;
    int32_t wave_sizze_24236;
    int32_t group_tid_24235;
    
    global_tid_24233 = get_global_id(0);
    local_tid_24234 = get_local_id(0);
    group_sizze_24237 = get_local_size(0);
    wave_sizze_24236 = LOCKSTEP_WIDTH;
    group_tid_24235 = get_group_id(0);
    
    int32_t phys_tid_19798;
    
    phys_tid_19798 = global_tid_24233;
    
    int32_t gtid_19795;
    
    gtid_19795 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24235) *
                                      sext_i32_i64(segmap_group_sizze_20481) +
                                      sext_i32_i64(local_tid_24234),
                                      sext_i32_i64(k2p2zq_18071) *
                                      sext_i32_i64(k2p2zq_18071)));
    
    int32_t gtid_19796;
    
    gtid_19796 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24235) *
                                      sext_i32_i64(segmap_group_sizze_20481) +
                                      sext_i32_i64(local_tid_24234) -
                                      squot64(sext_i32_i64(group_tid_24235) *
                                              sext_i32_i64(segmap_group_sizze_20481) +
                                              sext_i32_i64(local_tid_24234),
                                              sext_i32_i64(k2p2zq_18071) *
                                              sext_i32_i64(k2p2zq_18071)) *
                                      (sext_i32_i64(k2p2zq_18071) *
                                       sext_i32_i64(k2p2zq_18071)),
                                      sext_i32_i64(k2p2zq_18071)));
    
    int32_t gtid_19797;
    
    gtid_19797 = sext_i64_i32(sext_i32_i64(group_tid_24235) *
        sext_i32_i64(segmap_group_sizze_20481) + sext_i32_i64(local_tid_24234) -
        squot64(sext_i32_i64(group_tid_24235) *
                sext_i32_i64(segmap_group_sizze_20481) +
                sext_i32_i64(local_tid_24234), sext_i32_i64(k2p2zq_18071) *
                sext_i32_i64(k2p2zq_18071)) * (sext_i32_i64(k2p2zq_18071) *
                                               sext_i32_i64(k2p2zq_18071)) -
        squot64(sext_i32_i64(group_tid_24235) *
                sext_i32_i64(segmap_group_sizze_20481) +
                sext_i32_i64(local_tid_24234) -
                squot64(sext_i32_i64(group_tid_24235) *
                        sext_i32_i64(segmap_group_sizze_20481) +
                        sext_i32_i64(local_tid_24234),
                        sext_i32_i64(k2p2zq_18071) *
                        sext_i32_i64(k2p2zq_18071)) *
                (sext_i32_i64(k2p2zq_18071) * sext_i32_i64(k2p2zq_18071)),
                sext_i32_i64(k2p2zq_18071)) * sext_i32_i64(k2p2zq_18071));
    if ((slt32(gtid_19795, m_18055) && slt32(gtid_19796, k2p2zq_18071)) &&
        slt32(gtid_19797, k2p2zq_18071)) {
        int32_t index_primexp_22414 = m_18193 * gtid_19796;
        int32_t i_20489 = add32(k2p2zq_18071, gtid_19797);
        int32_t new_index_20490 = i_20489 + index_primexp_22414;
        float res_20491 = ((__global
                            float *) res_r_mem_23427)[sext_i32_i64(res_r_ixfn_23422) +
                                                      (sext_i32_i64(gtid_19795) *
                                                       sext_i32_i64(res_r_ixfn_23423) +
                                                       sext_i32_i64(new_index_20490) *
                                                       sext_i32_i64(res_r_ixfn_23425))];
        
        ((__global float *) mem_23435)[sext_i32_i64(gtid_19795) *
                                       sext_i32_i64(k2p2zq_18071 *
                                       k2p2zq_18071) +
                                       sext_i32_i64(gtid_19796) *
                                       sext_i32_i64(k2p2zq_18071) +
                                       sext_i32_i64(gtid_19797)] = res_20491;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20481
}
__kernel void mainzisegmap_20027(__global int *global_failure, int32_t m_18055,
                                 int32_t nm_18194, int32_t ctx_param_ext_23378,
                                 int32_t ctx_param_ext_23379,
                                 int32_t ctx_param_ext_23381, __global
                                 unsigned char *mem_param_23383, __global
                                 unsigned char *mem_23410)
{
    #define segmap_group_sizze_20430 (mainzisegmap_group_sizze_20032)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24227;
    int32_t local_tid_24228;
    int32_t group_sizze_24231;
    int32_t wave_sizze_24230;
    int32_t group_tid_24229;
    
    global_tid_24227 = get_global_id(0);
    local_tid_24228 = get_local_id(0);
    group_sizze_24231 = get_local_size(0);
    wave_sizze_24230 = LOCKSTEP_WIDTH;
    group_tid_24229 = get_group_id(0);
    
    int32_t phys_tid_20027;
    
    phys_tid_20027 = global_tid_24227;
    
    int32_t gtid_20025;
    
    gtid_20025 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24229) *
                                      sext_i32_i64(segmap_group_sizze_20430) +
                                      sext_i32_i64(local_tid_24228),
                                      sext_i32_i64(nm_18194)));
    
    int32_t gtid_20026;
    
    gtid_20026 = sext_i64_i32(sext_i32_i64(group_tid_24229) *
        sext_i32_i64(segmap_group_sizze_20430) + sext_i32_i64(local_tid_24228) -
        squot64(sext_i32_i64(group_tid_24229) *
                sext_i32_i64(segmap_group_sizze_20430) +
                sext_i32_i64(local_tid_24228), sext_i32_i64(nm_18194)) *
        sext_i32_i64(nm_18194));
    if (slt32(gtid_20025, m_18055) && slt32(gtid_20026, nm_18194)) {
        float write_value_20438 = ((__global
                                    float *) mem_23410)[sext_i32_i64(gtid_20025) *
                                                        sext_i32_i64(nm_18194) +
                                                        sext_i32_i64(gtid_20026)];
        
        if ((sle32(0, gtid_20025) && slt32(gtid_20025, m_18055)) && (sle32(0,
                                                                           gtid_20026) &&
                                                                     slt32(gtid_20026,
                                                                           nm_18194))) {
            ((__global
              float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                        (sext_i32_i64(gtid_20025) *
                                         sext_i32_i64(ctx_param_ext_23379) +
                                         sext_i32_i64(gtid_20026) *
                                         sext_i32_i64(ctx_param_ext_23381))] =
                write_value_20438;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20430
}
__kernel void mainzisegmap_20084(__global int *global_failure, int32_t m_18055,
                                 int32_t k2p2zq_18071, int32_t m_18193,
                                 int32_t nm_18194, int32_t i_20329,
                                 int32_t ctx_param_ext_23378,
                                 int32_t ctx_param_ext_23379,
                                 int32_t ctx_param_ext_23381, __global
                                 unsigned char *mem_param_23383, __global
                                 unsigned char *mem_23404, __global
                                 unsigned char *mem_23410)
{
    #define segmap_group_sizze_20397 (mainzisegmap_group_sizze_20089)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24222;
    int32_t local_tid_24223;
    int32_t group_sizze_24226;
    int32_t wave_sizze_24225;
    int32_t group_tid_24224;
    
    global_tid_24222 = get_global_id(0);
    local_tid_24223 = get_local_id(0);
    group_sizze_24226 = get_local_size(0);
    wave_sizze_24225 = LOCKSTEP_WIDTH;
    group_tid_24224 = get_group_id(0);
    
    int32_t phys_tid_20084;
    
    phys_tid_20084 = global_tid_24222;
    
    int32_t gtid_20082;
    
    gtid_20082 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24224) *
                                      sext_i32_i64(segmap_group_sizze_20397) +
                                      sext_i32_i64(local_tid_24223),
                                      sext_i32_i64(nm_18194)));
    
    int32_t gtid_20083;
    
    gtid_20083 = sext_i64_i32(sext_i32_i64(group_tid_24224) *
        sext_i32_i64(segmap_group_sizze_20397) + sext_i32_i64(local_tid_24223) -
        squot64(sext_i32_i64(group_tid_24224) *
                sext_i32_i64(segmap_group_sizze_20397) +
                sext_i32_i64(local_tid_24223), sext_i32_i64(nm_18194)) *
        sext_i32_i64(nm_18194));
    if (slt32(gtid_20082, m_18055) && slt32(gtid_20083, nm_18194)) {
        bool cond_20404 = ((__global
                            bool *) mem_23404)[sext_i32_i64(gtid_20082)];
        int32_t res_20406 = sdiv32(gtid_20083, m_18193);
        int32_t res_20407 = smod32(gtid_20083, m_18193);
        float res_20408;
        
        if (cond_20404) {
            int32_t x_20409 = mul32(m_18193, res_20406);
            int32_t i_20410 = add32(res_20407, x_20409);
            float res_20411 = ((__global
                                float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                                          (sext_i32_i64(gtid_20082) *
                                                           sext_i32_i64(ctx_param_ext_23379) +
                                                           sext_i32_i64(i_20410) *
                                                           sext_i32_i64(ctx_param_ext_23381))];
            
            res_20408 = res_20411;
        } else {
            float v1_20403 = ((__global
                               float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                                         (sext_i32_i64(gtid_20082) *
                                                          sext_i32_i64(ctx_param_ext_23379) +
                                                          sext_i32_i64(i_20329) *
                                                          sext_i32_i64(ctx_param_ext_23381))];
            float x_20412 = ((__global
                              float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                                        (sext_i32_i64(gtid_20082) *
                                                         sext_i32_i64(ctx_param_ext_23379) +
                                                         sext_i32_i64(res_20407) *
                                                         sext_i32_i64(ctx_param_ext_23381))];
            float x_20413 = x_20412 / v1_20403;
            int32_t y_20414 = sub32(k2p2zq_18071, 1);
            bool cond_20415 = slt32(res_20406, y_20414);
            float res_20416;
            
            if (cond_20415) {
                int32_t x_20417 = add32(1, res_20406);
                int32_t x_20418 = mul32(m_18193, x_20417);
                int32_t i_20419 = add32(res_20407, x_20418);
                float x_20420 = ((__global
                                  float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                                            (sext_i32_i64(gtid_20082) *
                                                             sext_i32_i64(ctx_param_ext_23379) +
                                                             sext_i32_i64(i_20419) *
                                                             sext_i32_i64(ctx_param_ext_23381))];
                int32_t i_20421 = add32(i_20329, x_20418);
                float x_20422 = ((__global
                                  float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                                            (sext_i32_i64(gtid_20082) *
                                                             sext_i32_i64(ctx_param_ext_23379) +
                                                             sext_i32_i64(i_20421) *
                                                             sext_i32_i64(ctx_param_ext_23381))];
                float y_20423 = x_20413 * x_20422;
                float res_20424 = x_20420 - y_20423;
                
                res_20416 = res_20424;
            } else {
                res_20416 = x_20413;
            }
            res_20408 = res_20416;
        }
        ((__global float *) mem_23410)[sext_i32_i64(gtid_20082) *
                                       sext_i32_i64(nm_18194) +
                                       sext_i32_i64(gtid_20083)] = res_20408;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20397
}
__kernel void mainzisegmap_20150(__global int *global_failure, int32_t m_18055,
                                 int32_t i_20329, int32_t ctx_param_ext_23378,
                                 int32_t ctx_param_ext_23379,
                                 int32_t ctx_param_ext_23381, __global
                                 unsigned char *mem_param_23383, __global
                                 unsigned char *mem_23404)
{
    #define segmap_group_sizze_20373 (mainzisegmap_group_sizze_20153)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24217;
    int32_t local_tid_24218;
    int32_t group_sizze_24221;
    int32_t wave_sizze_24220;
    int32_t group_tid_24219;
    
    global_tid_24217 = get_global_id(0);
    local_tid_24218 = get_local_id(0);
    group_sizze_24221 = get_local_size(0);
    wave_sizze_24220 = LOCKSTEP_WIDTH;
    group_tid_24219 = get_group_id(0);
    
    int32_t phys_tid_20150;
    
    phys_tid_20150 = global_tid_24217;
    
    int32_t gtid_20149;
    
    gtid_20149 = sext_i64_i32(sext_i32_i64(group_tid_24219) *
        sext_i32_i64(segmap_group_sizze_20373) + sext_i32_i64(local_tid_24218));
    if (slt32(gtid_20149, m_18055)) {
        float v1_20380 = ((__global
                           float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                                     (sext_i32_i64(gtid_20149) *
                                                      sext_i32_i64(ctx_param_ext_23379) +
                                                      sext_i32_i64(i_20329) *
                                                      sext_i32_i64(ctx_param_ext_23381))];
        bool cond_20381 = v1_20380 == 0.0F;
        
        ((__global bool *) mem_23404)[sext_i32_i64(gtid_20149)] = cond_20381;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20373
}
__kernel void mainzisegmap_20258(__global int *global_failure, int32_t m_18055,
                                 int32_t k2p2zq_18071, int32_t m_18193,
                                 int32_t nm_18194, __global
                                 unsigned char *res_mem_23334, __global
                                 unsigned char *mem_23375)
{
    #define segmap_group_sizze_20312 (mainzisegmap_group_sizze_20263)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24184;
    int32_t local_tid_24185;
    int32_t group_sizze_24188;
    int32_t wave_sizze_24187;
    int32_t group_tid_24186;
    
    global_tid_24184 = get_global_id(0);
    local_tid_24185 = get_local_id(0);
    group_sizze_24188 = get_local_size(0);
    wave_sizze_24187 = LOCKSTEP_WIDTH;
    group_tid_24186 = get_group_id(0);
    
    int32_t phys_tid_20258;
    
    phys_tid_20258 = global_tid_24184;
    
    int32_t gtid_20256;
    
    gtid_20256 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24186) *
                                      sext_i32_i64(segmap_group_sizze_20312) +
                                      sext_i32_i64(local_tid_24185),
                                      sext_i32_i64(nm_18194)));
    
    int32_t gtid_20257;
    
    gtid_20257 = sext_i64_i32(sext_i32_i64(group_tid_24186) *
        sext_i32_i64(segmap_group_sizze_20312) + sext_i32_i64(local_tid_24185) -
        squot64(sext_i32_i64(group_tid_24186) *
                sext_i32_i64(segmap_group_sizze_20312) +
                sext_i32_i64(local_tid_24185), sext_i32_i64(nm_18194)) *
        sext_i32_i64(nm_18194));
    if (slt32(gtid_20256, m_18055) && slt32(gtid_20257, nm_18194)) {
        int32_t res_20319 = sdiv32(gtid_20257, m_18193);
        int32_t res_20320 = smod32(gtid_20257, m_18193);
        bool cond_20321 = slt32(res_20320, k2p2zq_18071);
        float res_20322;
        
        if (cond_20321) {
            float res_20323 = ((__global
                                float *) res_mem_23334)[sext_i32_i64(gtid_20256) *
                                                        sext_i32_i64(k2p2zq_18071 *
                                                        k2p2zq_18071) +
                                                        sext_i32_i64(res_20319) *
                                                        sext_i32_i64(k2p2zq_18071) +
                                                        sext_i32_i64(res_20320)];
            
            res_20322 = res_20323;
        } else {
            int32_t y_20324 = add32(k2p2zq_18071, res_20319);
            bool cond_20325 = res_20320 == y_20324;
            float res_20326;
            
            if (cond_20325) {
                res_20326 = 1.0F;
            } else {
                res_20326 = 0.0F;
            }
            res_20322 = res_20326;
        }
        ((__global float *) mem_23375)[sext_i32_i64(gtid_20256) *
                                       sext_i32_i64(nm_18194) +
                                       sext_i32_i64(gtid_20257)] = res_20322;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20312
}
__kernel void mainzisegmap_20498(__global int *global_failure, int32_t N_18054,
                                 int32_t m_18055, int32_t n_18059,
                                 int32_t k2p2zq_18071, int32_t num_groups_20521,
                                 __global unsigned char *binop_p_mem_23202,
                                 __global unsigned char *mem_23441, __global
                                 unsigned char *mem_23445, __global
                                 unsigned char *mem_23462)
{
    #define segmap_group_sizze_20520 (mainzisegmap_group_sizze_20501)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24241;
    int32_t local_tid_24242;
    int32_t group_sizze_24245;
    int32_t wave_sizze_24244;
    int32_t group_tid_24243;
    
    global_tid_24241 = get_global_id(0);
    local_tid_24242 = get_local_id(0);
    group_sizze_24245 = get_local_size(0);
    wave_sizze_24244 = LOCKSTEP_WIDTH;
    group_tid_24243 = get_group_id(0);
    
    int32_t phys_tid_20498;
    
    phys_tid_20498 = global_tid_24241;
    
    int32_t phys_group_id_24246;
    
    phys_group_id_24246 = get_group_id(0);
    for (int32_t i_24247 = 0; i_24247 < sdiv_up32(sdiv_up32(m_18055,
                                                            segmap_group_sizze_20520) -
                                                  phys_group_id_24246,
                                                  num_groups_20521);
         i_24247++) {
        int32_t virt_group_id_24248 = phys_group_id_24246 + i_24247 *
                num_groups_20521;
        int32_t gtid_20497 = sext_i64_i32(sext_i32_i64(virt_group_id_24248) *
                sext_i32_i64(segmap_group_sizze_20520) +
                sext_i32_i64(local_tid_24242));
        
        if (slt32(gtid_20497, m_18055)) {
            for (int32_t i_23151 = 0; i_23151 < k2p2zq_18071; i_23151++) {
                float res_20527;
                float redout_23153 = 0.0F;
                
                for (int32_t i_23154 = 0; i_23154 < n_18059; i_23154++) {
                    float x_20532 = ((__global
                                      float *) mem_23441)[sext_i32_i64(i_23154) *
                                                          sext_i32_i64(m_18055) +
                                                          sext_i32_i64(gtid_20497)];
                    bool res_20533;
                    
                    res_20533 = futrts_isnan32(x_20532);
                    
                    float res_20534;
                    
                    if (res_20533) {
                        res_20534 = 0.0F;
                    } else {
                        float x_20531 = ((__global
                                          float *) binop_p_mem_23202)[sext_i32_i64(i_23151) *
                                                                      sext_i32_i64(N_18054) +
                                                                      sext_i32_i64(i_23154)];
                        float res_20535 = x_20531 * x_20532;
                        
                        res_20534 = res_20535;
                    }
                    
                    float res_20530 = res_20534 + redout_23153;
                    float redout_tmp_24250 = res_20530;
                    
                    redout_23153 = redout_tmp_24250;
                }
                res_20527 = redout_23153;
                ((__global float *) mem_23445)[sext_i32_i64(phys_tid_20498) +
                                               sext_i32_i64(i_23151) *
                                               sext_i32_i64(num_groups_20521 *
                                               segmap_group_sizze_20520)] =
                    res_20527;
            }
            for (int32_t i_24251 = 0; i_24251 < k2p2zq_18071; i_24251++) {
                ((__global float *) mem_23462)[sext_i32_i64(i_24251) *
                                               sext_i32_i64(m_18055) +
                                               sext_i32_i64(gtid_20497)] =
                    ((__global
                      float *) mem_23445)[sext_i32_i64(phys_tid_20498) +
                                          sext_i32_i64(i_24251) *
                                          sext_i32_i64(num_groups_20521 *
                                          segmap_group_sizze_20520)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20520
}
__kernel void mainzisegmap_20658(__global int *global_failure, int32_t m_18055,
                                 int32_t k2p2zq_18071, int32_t num_groups_20680,
                                 __global unsigned char *mem_23559, __global
                                 unsigned char *mem_23564, __global
                                 unsigned char *mem_23568, __global
                                 unsigned char *mem_23585)
{
    #define segmap_group_sizze_20679 (mainzisegmap_group_sizze_20661)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24328;
    int32_t local_tid_24329;
    int32_t group_sizze_24332;
    int32_t wave_sizze_24331;
    int32_t group_tid_24330;
    
    global_tid_24328 = get_global_id(0);
    local_tid_24329 = get_local_id(0);
    group_sizze_24332 = get_local_size(0);
    wave_sizze_24331 = LOCKSTEP_WIDTH;
    group_tid_24330 = get_group_id(0);
    
    int32_t phys_tid_20658;
    
    phys_tid_20658 = global_tid_24328;
    
    int32_t phys_group_id_24333;
    
    phys_group_id_24333 = get_group_id(0);
    for (int32_t i_24334 = 0; i_24334 < sdiv_up32(sdiv_up32(m_18055,
                                                            segmap_group_sizze_20679) -
                                                  phys_group_id_24333,
                                                  num_groups_20680);
         i_24334++) {
        int32_t virt_group_id_24335 = phys_group_id_24333 + i_24334 *
                num_groups_20680;
        int32_t gtid_20657 = sext_i64_i32(sext_i32_i64(virt_group_id_24335) *
                sext_i32_i64(segmap_group_sizze_20679) +
                sext_i32_i64(local_tid_24329));
        
        if (slt32(gtid_20657, m_18055)) {
            for (int32_t i_23161 = 0; i_23161 < k2p2zq_18071; i_23161++) {
                float res_20687;
                float redout_23163 = 0.0F;
                
                for (int32_t i_23164 = 0; i_23164 < k2p2zq_18071; i_23164++) {
                    float x_20691 = ((__global
                                      float *) mem_23564)[sext_i32_i64(i_23164) *
                                                          sext_i32_i64(m_18055) +
                                                          sext_i32_i64(gtid_20657)];
                    float x_20692 = ((__global
                                      float *) mem_23559)[sext_i32_i64(i_23161) *
                                                          sext_i32_i64(m_18055 *
                                                          k2p2zq_18071) +
                                                          sext_i32_i64(i_23164) *
                                                          sext_i32_i64(m_18055) +
                                                          sext_i32_i64(gtid_20657)];
                    float res_20693 = x_20691 * x_20692;
                    float res_20690 = res_20693 + redout_23163;
                    float redout_tmp_24337 = res_20690;
                    
                    redout_23163 = redout_tmp_24337;
                }
                res_20687 = redout_23163;
                ((__global float *) mem_23568)[sext_i32_i64(phys_tid_20658) +
                                               sext_i32_i64(i_23161) *
                                               sext_i32_i64(num_groups_20680 *
                                               segmap_group_sizze_20679)] =
                    res_20687;
            }
            for (int32_t i_24338 = 0; i_24338 < k2p2zq_18071; i_24338++) {
                ((__global float *) mem_23585)[sext_i32_i64(i_24338) *
                                               sext_i32_i64(m_18055) +
                                               sext_i32_i64(gtid_20657)] =
                    ((__global
                      float *) mem_23568)[sext_i32_i64(phys_tid_20658) +
                                          sext_i32_i64(i_24338) *
                                          sext_i32_i64(num_groups_20680 *
                                          segmap_group_sizze_20679)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20679
}
__kernel void mainzisegmap_20696(__global int *global_failure, int32_t m_18055,
                                 int32_t k2p2zq_18071, __global
                                 unsigned char *res_mem_23552, __global
                                 unsigned char *mem_23592, __global
                                 unsigned char *mem_23598)
{
    #define segmap_group_sizze_20767 (mainzisegmap_group_sizze_20701)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24339;
    int32_t local_tid_24340;
    int32_t group_sizze_24343;
    int32_t wave_sizze_24342;
    int32_t group_tid_24341;
    
    global_tid_24339 = get_global_id(0);
    local_tid_24340 = get_local_id(0);
    group_sizze_24343 = get_local_size(0);
    wave_sizze_24342 = LOCKSTEP_WIDTH;
    group_tid_24341 = get_group_id(0);
    
    int32_t phys_tid_20696;
    
    phys_tid_20696 = global_tid_24339;
    
    int32_t gtid_20694;
    
    gtid_20694 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24341) *
                                      sext_i32_i64(segmap_group_sizze_20767) +
                                      sext_i32_i64(local_tid_24340),
                                      sext_i32_i64(k2p2zq_18071)));
    
    int32_t gtid_20695;
    
    gtid_20695 = sext_i64_i32(sext_i32_i64(group_tid_24341) *
        sext_i32_i64(segmap_group_sizze_20767) + sext_i32_i64(local_tid_24340) -
        squot64(sext_i32_i64(group_tid_24341) *
                sext_i32_i64(segmap_group_sizze_20767) +
                sext_i32_i64(local_tid_24340), sext_i32_i64(k2p2zq_18071)) *
        sext_i32_i64(k2p2zq_18071));
    if (slt32(gtid_20694, m_18055) && slt32(gtid_20695, k2p2zq_18071)) {
        float res_20778;
        float redout_23165 = 0.0F;
        
        for (int32_t i_23166 = 0; i_23166 < k2p2zq_18071; i_23166++) {
            float x_20782 = ((__global
                              float *) res_mem_23552)[sext_i32_i64(gtid_20694) *
                                                      sext_i32_i64(k2p2zq_18071) +
                                                      sext_i32_i64(i_23166)];
            float x_20783 = ((__global
                              float *) mem_23592)[sext_i32_i64(i_23166) *
                                                  sext_i32_i64(k2p2zq_18071 *
                                                  m_18055) +
                                                  sext_i32_i64(gtid_20694) *
                                                  sext_i32_i64(k2p2zq_18071) +
                                                  sext_i32_i64(gtid_20695)];
            float res_20784 = x_20782 * x_20783;
            float res_20781 = res_20784 + redout_23165;
            float redout_tmp_24344 = res_20781;
            
            redout_23165 = redout_tmp_24344;
        }
        res_20778 = redout_23165;
        ((__global float *) mem_23598)[sext_i32_i64(gtid_20694) *
                                       sext_i32_i64(k2p2zq_18071) +
                                       sext_i32_i64(gtid_20695)] = res_20778;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20767
}
__kernel void mainzisegmap_20809(__global int *global_failure, int32_t N_18054,
                                 int32_t m_18055, int32_t k2p2zq_18071,
                                 int32_t num_groups_20830, __global
                                 unsigned char *mem_23213, __global
                                 unsigned char *mem_23617, __global
                                 unsigned char *mem_23621, __global
                                 unsigned char *mem_23638)
{
    #define segmap_group_sizze_20829 (mainzisegmap_group_sizze_20812)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24408;
    int32_t local_tid_24409;
    int32_t group_sizze_24412;
    int32_t wave_sizze_24411;
    int32_t group_tid_24410;
    
    global_tid_24408 = get_global_id(0);
    local_tid_24409 = get_local_id(0);
    group_sizze_24412 = get_local_size(0);
    wave_sizze_24411 = LOCKSTEP_WIDTH;
    group_tid_24410 = get_group_id(0);
    
    int32_t phys_tid_20809;
    
    phys_tid_20809 = global_tid_24408;
    
    int32_t phys_group_id_24413;
    
    phys_group_id_24413 = get_group_id(0);
    for (int32_t i_24414 = 0; i_24414 < sdiv_up32(sdiv_up32(m_18055,
                                                            segmap_group_sizze_20829) -
                                                  phys_group_id_24413,
                                                  num_groups_20830);
         i_24414++) {
        int32_t virt_group_id_24415 = phys_group_id_24413 + i_24414 *
                num_groups_20830;
        int32_t gtid_20808 = sext_i64_i32(sext_i32_i64(virt_group_id_24415) *
                sext_i32_i64(segmap_group_sizze_20829) +
                sext_i32_i64(local_tid_24409));
        
        if (slt32(gtid_20808, m_18055)) {
            for (int32_t i_23169 = 0; i_23169 < N_18054; i_23169++) {
                float res_20836;
                float redout_23171 = 0.0F;
                
                for (int32_t i_23172 = 0; i_23172 < k2p2zq_18071; i_23172++) {
                    float x_20840 = ((__global
                                      float *) mem_23617)[sext_i32_i64(i_23172) *
                                                          sext_i32_i64(m_18055) +
                                                          sext_i32_i64(gtid_20808)];
                    float x_20841 = ((__global
                                      float *) mem_23213)[sext_i32_i64(i_23169) *
                                                          sext_i32_i64(k2p2zq_18071) +
                                                          sext_i32_i64(i_23172)];
                    float res_20842 = x_20840 * x_20841;
                    float res_20839 = res_20842 + redout_23171;
                    float redout_tmp_24417 = res_20839;
                    
                    redout_23171 = redout_tmp_24417;
                }
                res_20836 = redout_23171;
                ((__global float *) mem_23621)[sext_i32_i64(phys_tid_20809) +
                                               sext_i32_i64(i_23169) *
                                               sext_i32_i64(num_groups_20830 *
                                               segmap_group_sizze_20829)] =
                    res_20836;
            }
            for (int32_t i_24418 = 0; i_24418 < N_18054; i_24418++) {
                ((__global float *) mem_23638)[sext_i32_i64(i_24418) *
                                               sext_i32_i64(m_18055) +
                                               sext_i32_i64(gtid_20808)] =
                    ((__global
                      float *) mem_23621)[sext_i32_i64(phys_tid_20809) +
                                          sext_i32_i64(i_24418) *
                                          sext_i32_i64(num_groups_20830 *
                                          segmap_group_sizze_20829)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20829
}
__kernel void mainzisegmap_21077(__global int *global_failure, int32_t N_18054,
                                 int32_t m_18055, __global
                                 unsigned char *mem_23738, __global
                                 unsigned char *mem_23743, __global
                                 unsigned char *mem_23777, __global
                                 unsigned char *mem_23782)
{
    #define segmap_group_sizze_21297 (mainzisegmap_group_sizze_21082)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24587;
    int32_t local_tid_24588;
    int32_t group_sizze_24591;
    int32_t wave_sizze_24590;
    int32_t group_tid_24589;
    
    global_tid_24587 = get_global_id(0);
    local_tid_24588 = get_local_id(0);
    group_sizze_24591 = get_local_size(0);
    wave_sizze_24590 = LOCKSTEP_WIDTH;
    group_tid_24589 = get_group_id(0);
    
    int32_t phys_tid_21077;
    
    phys_tid_21077 = global_tid_24587;
    
    int32_t gtid_21075;
    
    gtid_21075 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24589) *
                                      sext_i32_i64(segmap_group_sizze_21297) +
                                      sext_i32_i64(local_tid_24588),
                                      sext_i32_i64(N_18054)));
    
    int32_t gtid_21076;
    
    gtid_21076 = sext_i64_i32(sext_i32_i64(group_tid_24589) *
        sext_i32_i64(segmap_group_sizze_21297) + sext_i32_i64(local_tid_24588) -
        squot64(sext_i32_i64(group_tid_24589) *
                sext_i32_i64(segmap_group_sizze_21297) +
                sext_i32_i64(local_tid_24588), sext_i32_i64(N_18054)) *
        sext_i32_i64(N_18054));
    if (slt32(gtid_21075, m_18055) && slt32(gtid_21076, N_18054)) {
        float x_21307 = ((__global
                          float *) mem_23782)[sext_i32_i64(gtid_21075) *
                                              sext_i32_i64(N_18054) +
                                              sext_i32_i64(gtid_21076)];
        bool res_21310;
        
        res_21310 = futrts_isnan32(x_21307);
        
        bool res_21311 = !res_21310;
        int32_t res_21312;
        
        if (res_21311) {
            int32_t x_21308 = ((__global
                                int32_t *) mem_23777)[sext_i32_i64(gtid_21075) *
                                                      sext_i32_i64(N_18054) +
                                                      sext_i32_i64(gtid_21076)];
            int32_t res_21313 = sub32(x_21308, 1);
            
            res_21312 = res_21313;
        } else {
            res_21312 = -1;
        }
        if ((sle32(0, gtid_21075) && slt32(gtid_21075, m_18055)) && (sle32(0,
                                                                           res_21312) &&
                                                                     slt32(res_21312,
                                                                           N_18054))) {
            ((__global int32_t *) mem_23743)[sext_i32_i64(gtid_21075) *
                                             sext_i32_i64(N_18054) +
                                             sext_i32_i64(res_21312)] =
                gtid_21076;
        }
        if ((sle32(0, gtid_21075) && slt32(gtid_21075, m_18055)) && (sle32(0,
                                                                           res_21312) &&
                                                                     slt32(res_21312,
                                                                           N_18054))) {
            ((__global float *) mem_23738)[sext_i32_i64(gtid_21075) *
                                           sext_i32_i64(N_18054) +
                                           sext_i32_i64(res_21312)] = x_21307;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_21297
}
__kernel void mainzisegmap_21328(__global int *global_failure, int32_t m_18055,
                                 int32_t n_18059, float hfrac_18061,
                                 int32_t k2p2_18069, __global
                                 unsigned char *mem_23794, __global
                                 unsigned char *mem_23799, __global
                                 unsigned char *mem_23803, __global
                                 unsigned char *mem_23806, __global
                                 unsigned char *mem_23809)
{
    #define segmap_group_sizze_21364 (mainzisegmap_group_sizze_21331)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24595;
    int32_t local_tid_24596;
    int32_t group_sizze_24599;
    int32_t wave_sizze_24598;
    int32_t group_tid_24597;
    
    global_tid_24595 = get_global_id(0);
    local_tid_24596 = get_local_id(0);
    group_sizze_24599 = get_local_size(0);
    wave_sizze_24598 = LOCKSTEP_WIDTH;
    group_tid_24597 = get_group_id(0);
    
    int32_t phys_tid_21328;
    
    phys_tid_21328 = global_tid_24595;
    
    int32_t gtid_21327;
    
    gtid_21327 = sext_i64_i32(sext_i32_i64(group_tid_24597) *
        sext_i32_i64(segmap_group_sizze_21364) + sext_i32_i64(local_tid_24596));
    if (slt32(gtid_21327, m_18055)) {
        int32_t res_21373;
        int32_t redout_23177 = 0;
        
        for (int32_t i_23178 = 0; i_23178 < n_18059; i_23178++) {
            float x_21377 = ((__global
                              float *) mem_23794)[sext_i32_i64(i_23178) *
                                                  sext_i32_i64(m_18055) +
                                                  sext_i32_i64(gtid_21327)];
            bool res_21378;
            
            res_21378 = futrts_isnan32(x_21377);
            
            bool cond_21379 = !res_21378;
            int32_t res_21380 = btoi_bool_i32(cond_21379);
            int32_t res_21376 = add32(res_21380, redout_23177);
            int32_t redout_tmp_24600 = res_21376;
            
            redout_23177 = redout_tmp_24600;
        }
        res_21373 = redout_23177;
        
        float res_21381;
        float redout_23179 = 0.0F;
        
        for (int32_t i_23180 = 0; i_23180 < n_18059; i_23180++) {
            bool cond_21387 = slt32(i_23180, res_21373);
            float res_21388;
            
            if (cond_21387) {
                float x_elem_21386 = ((__global
                                       float *) mem_23799)[sext_i32_i64(i_23180) *
                                                           sext_i32_i64(m_18055) +
                                                           sext_i32_i64(gtid_21327)];
                
                res_21388 = x_elem_21386;
            } else {
                res_21388 = 0.0F;
            }
            
            float res_21389 = res_21388 * res_21388;
            float res_21384 = res_21389 + redout_23179;
            float redout_tmp_24601 = res_21384;
            
            redout_23179 = redout_tmp_24601;
        }
        res_21381 = redout_23179;
        
        int32_t r32_arg_21390 = sub32(res_21373, k2p2_18069);
        float res_21391 = sitofp_i32_f32(r32_arg_21390);
        float sqrt_arg_21392 = res_21381 / res_21391;
        float res_21393;
        
        res_21393 = futrts_sqrt32(sqrt_arg_21392);
        
        float res_21394 = sitofp_i32_f32(res_21373);
        float t32_arg_21395 = hfrac_18061 * res_21394;
        int32_t res_21396 = fptosi_f32_i32(t32_arg_21395);
        
        ((__global int32_t *) mem_23803)[sext_i32_i64(gtid_21327)] = res_21396;
        ((__global int32_t *) mem_23806)[sext_i32_i64(gtid_21327)] = res_21373;
        ((__global float *) mem_23809)[sext_i32_i64(gtid_21327)] = res_21393;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_21364
}
__kernel void mainzisegmap_21433(__global int *global_failure, int32_t m_18055,
                                 float hfrac_18061, int32_t k2p2_18069, __global
                                 unsigned char *mem_23825, __global
                                 unsigned char *mem_23829, __global
                                 unsigned char *mem_23833, __global
                                 unsigned char *mem_23836)
{
    #define segmap_group_sizze_21528 (mainzisegmap_group_sizze_21436)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24734;
    int32_t local_tid_24735;
    int32_t group_sizze_24738;
    int32_t wave_sizze_24737;
    int32_t group_tid_24736;
    
    global_tid_24734 = get_global_id(0);
    local_tid_24735 = get_local_id(0);
    group_sizze_24738 = get_local_size(0);
    wave_sizze_24737 = LOCKSTEP_WIDTH;
    group_tid_24736 = get_group_id(0);
    
    int32_t phys_tid_21433;
    
    phys_tid_21433 = global_tid_24734;
    
    int32_t gtid_21432;
    
    gtid_21432 = sext_i64_i32(sext_i32_i64(group_tid_24736) *
        sext_i32_i64(segmap_group_sizze_21528) + sext_i32_i64(local_tid_24735));
    if (slt32(gtid_21432, m_18055)) {
        int32_t res_21534 = ((__global
                              int32_t *) mem_23825)[sext_i32_i64(gtid_21432)];
        float res_21535 = ((__global
                            float *) mem_23829)[sext_i32_i64(gtid_21432)];
        int32_t r32_arg_21536 = sub32(res_21534, k2p2_18069);
        float res_21537 = sitofp_i32_f32(r32_arg_21536);
        float sqrt_arg_21538 = res_21535 / res_21537;
        float res_21539;
        
        res_21539 = futrts_sqrt32(sqrt_arg_21538);
        
        float res_21540 = sitofp_i32_f32(res_21534);
        float t32_arg_21541 = hfrac_18061 * res_21540;
        int32_t res_21542 = fptosi_f32_i32(t32_arg_21541);
        
        ((__global int32_t *) mem_23833)[sext_i32_i64(gtid_21432)] = res_21542;
        ((__global float *) mem_23836)[sext_i32_i64(gtid_21432)] = res_21539;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_21528
}
__kernel void mainzisegmap_21563(__global int *global_failure, int32_t N_18054,
                                 int32_t m_18055, int32_t res_18381, __global
                                 unsigned char *res_mem_23788, __global
                                 unsigned char *res_mem_23840, __global
                                 unsigned char *res_mem_23841, __global
                                 unsigned char *mem_23849)
{
    #define segmap_group_sizze_21587 (mainzisegmap_group_sizze_21566)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24773;
    int32_t local_tid_24774;
    int32_t group_sizze_24777;
    int32_t wave_sizze_24776;
    int32_t group_tid_24775;
    
    global_tid_24773 = get_global_id(0);
    local_tid_24774 = get_local_id(0);
    group_sizze_24777 = get_local_size(0);
    wave_sizze_24776 = LOCKSTEP_WIDTH;
    group_tid_24775 = get_group_id(0);
    
    int32_t phys_tid_21563;
    
    phys_tid_21563 = global_tid_24773;
    
    int32_t gtid_21562;
    
    gtid_21562 = sext_i64_i32(sext_i32_i64(group_tid_24775) *
        sext_i32_i64(segmap_group_sizze_21587) + sext_i32_i64(local_tid_24774));
    if (slt32(gtid_21562, m_18055)) {
        int32_t x_21593 = ((__global
                            int32_t *) res_mem_23841)[sext_i32_i64(gtid_21562)];
        int32_t x_21594 = ((__global
                            int32_t *) res_mem_23840)[sext_i32_i64(gtid_21562)];
        float res_21595;
        float redout_22424 = 0.0F;
        
        for (int32_t i_22425 = 0; i_22425 < res_18381; i_22425++) {
            bool cond_21600 = slt32(i_22425, x_21594);
            float res_21601;
            
            if (cond_21600) {
                int32_t x_21602 = add32(x_21593, i_22425);
                int32_t x_21603 = sub32(x_21602, x_21594);
                int32_t i_21604 = add32(1, x_21603);
                float res_21605 = ((__global
                                    float *) res_mem_23788)[sext_i32_i64(gtid_21562) *
                                                            sext_i32_i64(N_18054) +
                                                            sext_i32_i64(i_21604)];
                
                res_21601 = res_21605;
            } else {
                res_21601 = 0.0F;
            }
            
            float res_21598 = res_21601 + redout_22424;
            float redout_tmp_24778 = res_21598;
            
            redout_22424 = redout_tmp_24778;
        }
        res_21595 = redout_22424;
        ((__global float *) mem_23849)[sext_i32_i64(gtid_21562)] = res_21595;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_21587
}
__kernel void mainzisegmap_21694(__global int *global_failure, float lam_18062,
                                 int32_t iota_arg_18403, int32_t x_18408,
                                 float res_18411, __global
                                 unsigned char *mappingindices_mem_23188,
                                 __global unsigned char *mem_23858)
{
    #define segmap_group_sizze_21715 (mainzisegmap_group_sizze_21697)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24839;
    int32_t local_tid_24840;
    int32_t group_sizze_24843;
    int32_t wave_sizze_24842;
    int32_t group_tid_24841;
    
    global_tid_24839 = get_global_id(0);
    local_tid_24840 = get_local_id(0);
    group_sizze_24843 = get_local_size(0);
    wave_sizze_24842 = LOCKSTEP_WIDTH;
    group_tid_24841 = get_group_id(0);
    
    int32_t phys_tid_21694;
    
    phys_tid_21694 = global_tid_24839;
    
    int32_t gtid_21693;
    
    gtid_21693 = sext_i64_i32(sext_i32_i64(group_tid_24841) *
        sext_i32_i64(segmap_group_sizze_21715) + sext_i32_i64(local_tid_24840));
    if (slt32(gtid_21693, iota_arg_18403)) {
        int32_t t_21721 = add32(x_18408, gtid_21693);
        int32_t i_21722 = sub32(t_21721, 1);
        int32_t time_21723 = ((__global
                               int32_t *) mappingindices_mem_23188)[sext_i32_i64(i_21722)];
        float res_21724 = sitofp_i32_f32(time_21723);
        float logplus_arg_21725 = res_21724 / res_18411;
        bool cond_21726 = 2.7182817F < logplus_arg_21725;
        float res_21727;
        
        if (cond_21726) {
            float res_21728;
            
            res_21728 = futrts_log32(logplus_arg_21725);
            res_21727 = res_21728;
        } else {
            res_21727 = 1.0F;
        }
        
        float res_21729;
        
        res_21729 = futrts_sqrt32(res_21727);
        
        float res_21730 = lam_18062 * res_21729;
        
        ((__global float *) mem_23858)[sext_i32_i64(gtid_21693)] = res_21730;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_21715
}
__kernel void mainzisegmap_22053(__global int *global_failure, int32_t N_18054,
                                 int32_t m_18055, int32_t n_18059, __global
                                 unsigned char *res_mem_23789, __global
                                 unsigned char *res_mem_23841, __global
                                 unsigned char *mem_23877, __global
                                 unsigned char *mem_23886, __global
                                 unsigned char *mem_23889, __global
                                 unsigned char *mem_23899)
{
    #define segmap_group_sizze_22331 (mainzisegmap_group_sizze_22056)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25030;
    int32_t local_tid_25031;
    int32_t group_sizze_25034;
    int32_t wave_sizze_25033;
    int32_t group_tid_25032;
    
    global_tid_25030 = get_global_id(0);
    local_tid_25031 = get_local_id(0);
    group_sizze_25034 = get_local_size(0);
    wave_sizze_25033 = LOCKSTEP_WIDTH;
    group_tid_25032 = get_group_id(0);
    
    int32_t phys_tid_22053;
    
    phys_tid_22053 = global_tid_25030;
    
    int32_t gtid_22052;
    
    gtid_22052 = sext_i64_i32(sext_i32_i64(group_tid_25032) *
        sext_i32_i64(segmap_group_sizze_22331) + sext_i32_i64(local_tid_25031));
    if (slt32(gtid_22052, m_18055)) {
        int32_t x_22337 = ((__global
                            int32_t *) res_mem_23841)[sext_i32_i64(gtid_22052)];
        int32_t y_22339 = ((__global
                            int32_t *) mem_23877)[sext_i32_i64(gtid_22052)];
        bool acc0_22341 = ((__global
                            bool *) mem_23886)[sext_i32_i64(gtid_22052)];
        bool x_22346 = acc0_22341 && acc0_22341;
        int32_t res_22350;
        
        if (acc0_22341) {
            int32_t acc0_22342 = ((__global
                                   int32_t *) mem_23889)[sext_i32_i64(gtid_22052)];
            
            res_22350 = acc0_22342;
        } else {
            res_22350 = -1;
        }
        
        bool cond_22356 = !x_22346;
        int32_t fst_breakzq_22357;
        
        if (cond_22356) {
            fst_breakzq_22357 = -1;
        } else {
            bool cond_22358 = slt32(res_22350, y_22339);
            int32_t res_22359;
            
            if (cond_22358) {
                int32_t i_22360 = add32(x_22337, res_22350);
                int32_t x_22361 = ((__global
                                    int32_t *) res_mem_23789)[sext_i32_i64(gtid_22052) *
                                                              sext_i32_i64(N_18054) +
                                                              sext_i32_i64(i_22360)];
                int32_t res_22362 = sub32(x_22361, n_18059);
                
                res_22359 = res_22362;
            } else {
                res_22359 = -1;
            }
            fst_breakzq_22357 = res_22359;
        }
        
        bool cond_22363 = sle32(x_22337, 5);
        bool res_22364 = sle32(y_22339, 5);
        bool x_22365 = !cond_22363;
        bool y_22366 = res_22364 && x_22365;
        bool cond_22367 = cond_22363 || y_22366;
        int32_t fst_breakzq_22368;
        
        if (cond_22367) {
            fst_breakzq_22368 = -2;
        } else {
            fst_breakzq_22368 = fst_breakzq_22357;
        }
        ((__global int32_t *) mem_23899)[sext_i32_i64(gtid_22052)] =
            fst_breakzq_22368;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_22331
}
__kernel void mainzisegmap_22200(__global int *global_failure, int32_t m_18055,
                                 int32_t num_groups_22225, __global
                                 unsigned char *res_mem_23787, __global
                                 unsigned char *res_mem_23841, __global
                                 unsigned char *res_mem_23842, __global
                                 unsigned char *mem_23874, __global
                                 unsigned char *mem_23877)
{
    #define segmap_group_sizze_22224 (mainzisegmap_group_sizze_22203)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24871;
    int32_t local_tid_24872;
    int32_t group_sizze_24875;
    int32_t wave_sizze_24874;
    int32_t group_tid_24873;
    
    global_tid_24871 = get_global_id(0);
    local_tid_24872 = get_local_id(0);
    group_sizze_24875 = get_local_size(0);
    wave_sizze_24874 = LOCKSTEP_WIDTH;
    group_tid_24873 = get_group_id(0);
    
    int32_t phys_tid_22200;
    
    phys_tid_22200 = global_tid_24871;
    
    int32_t phys_group_id_24876;
    
    phys_group_id_24876 = get_group_id(0);
    for (int32_t i_24877 = 0; i_24877 < sdiv_up32(sdiv_up32(m_18055,
                                                            segmap_group_sizze_22224) -
                                                  phys_group_id_24876,
                                                  num_groups_22225);
         i_24877++) {
        int32_t virt_group_id_24878 = phys_group_id_24876 + i_24877 *
                num_groups_22225;
        int32_t gtid_22199 = sext_i64_i32(sext_i32_i64(virt_group_id_24878) *
                sext_i32_i64(segmap_group_sizze_22224) +
                sext_i32_i64(local_tid_24872));
        
        if (slt32(gtid_22199, m_18055)) {
            int32_t x_22231 = ((__global
                                int32_t *) res_mem_23787)[sext_i32_i64(gtid_22199)];
            int32_t x_22232 = ((__global
                                int32_t *) res_mem_23841)[sext_i32_i64(gtid_22199)];
            float x_22233 = ((__global
                              float *) res_mem_23842)[sext_i32_i64(gtid_22199)];
            int32_t y_22234 = sub32(x_22231, x_22232);
            float res_22235 = sitofp_i32_f32(x_22232);
            float res_22236;
            
            res_22236 = futrts_sqrt32(res_22235);
            
            float y_22237 = x_22233 * res_22236;
            
            ((__global float *) mem_23874)[sext_i32_i64(gtid_22199)] = y_22237;
            ((__global int32_t *) mem_23877)[sext_i32_i64(gtid_22199)] =
                y_22234;
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_22224
}
__kernel void mainzisegmap_intragroup_19569(__global int *global_failure,
                                            int failure_is_an_option, __global
                                            int *global_failure_args,
                                            __local volatile
                                            int64_t *mem_23362_backing_aligned_0,
                                            __local volatile
                                            int64_t *mem_23350_backing_aligned_1,
                                            __local volatile
                                            int64_t *mem_23339_backing_aligned_2,
                                            int32_t k2p2zq_18071,
                                            int32_t m_18193, int32_t nm_18194,
                                            int32_t computed_group_sizze_19515,
                                            __global
                                            unsigned char *res_mem_23334,
                                            __global unsigned char *mem_23369)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_23362_backing_2 = (__local volatile
                                                           char *) mem_23362_backing_aligned_0;
    __local volatile char *restrict mem_23350_backing_1 = (__local volatile
                                                           char *) mem_23350_backing_aligned_1;
    __local volatile char *restrict mem_23339_backing_0 = (__local volatile
                                                           char *) mem_23339_backing_aligned_2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_24174;
    int32_t local_tid_24175;
    int32_t group_sizze_24178;
    int32_t wave_sizze_24177;
    int32_t group_tid_24176;
    
    global_tid_24174 = get_global_id(0);
    local_tid_24175 = get_local_id(0);
    group_sizze_24178 = get_local_size(0);
    wave_sizze_24177 = LOCKSTEP_WIDTH;
    group_tid_24176 = get_group_id(0);
    
    int32_t phys_tid_19569;
    
    phys_tid_19569 = group_tid_24176;
    
    int32_t ltid_pre_24179;
    
    ltid_pre_24179 = squot32(local_tid_24175, k2p2zq_18071);
    
    int32_t ltid_pre_24180;
    
    ltid_pre_24180 = local_tid_24175 - squot32(local_tid_24175, k2p2zq_18071) *
        k2p2zq_18071;
    
    int32_t ltid_pre_24181;
    
    ltid_pre_24181 = local_tid_24175;
    
    int32_t gtid_19513;
    
    gtid_19513 = group_tid_24176;
    
    __local char *mem_23339;
    
    mem_23339 = (__local char *) mem_23339_backing_0;
    
    int32_t gtid_19516 = ltid_pre_24181;
    int32_t phys_tid_19517 = local_tid_24175;
    
    if (slt32(gtid_19516, nm_18194)) {
        int32_t res_19694 = sdiv32(gtid_19516, m_18193);
        int32_t res_19695 = smod32(gtid_19516, m_18193);
        bool cond_19696 = slt32(res_19695, k2p2zq_18071);
        float res_19697;
        
        if (cond_19696) {
            float res_19698 = ((__global
                                float *) res_mem_23334)[sext_i32_i64(gtid_19513) *
                                                        sext_i32_i64(k2p2zq_18071 *
                                                        k2p2zq_18071) +
                                                        sext_i32_i64(res_19694) *
                                                        sext_i32_i64(k2p2zq_18071) +
                                                        sext_i32_i64(res_19695)];
            
            res_19697 = res_19698;
        } else {
            int32_t y_19699 = add32(k2p2zq_18071, res_19694);
            bool cond_19700 = res_19695 == y_19699;
            float res_19701;
            
            if (cond_19700) {
                res_19701 = 1.0F;
            } else {
                res_19701 = 0.0F;
            }
            res_19697 = res_19701;
        }
        ((__local float *) mem_23339)[sext_i32_i64(gtid_19516)] = res_19697;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_23350;
    
    mem_23350 = (__local char *) mem_23350_backing_1;
    for (int32_t i_19703 = 0; i_19703 < k2p2zq_18071; i_19703++) {
        bool y_19705 = slt32(i_19703, nm_18194);
        bool index_certs_19706;
        
        if (!y_19705) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 0) == -1) {
                    global_failure_args[0] = i_19703;
                    global_failure_args[1] = nm_18194;
                    ;
                }
                local_failure = true;
                goto error_1;
            }
        }
        
        float v1_19707 = ((__local float *) mem_23339)[sext_i32_i64(i_19703)];
        bool cond_19708 = v1_19707 == 0.0F;
        int32_t gtid_19527 = ltid_pre_24181;
        int32_t phys_tid_19528 = local_tid_24175;
        
        if (slt32(gtid_19527, nm_18194)) {
            int32_t res_19711 = sdiv32(gtid_19527, m_18193);
            int32_t res_19712 = smod32(gtid_19527, m_18193);
            float res_19713;
            
            if (cond_19708) {
                int32_t x_19714 = mul32(m_18193, res_19711);
                int32_t i_19715 = add32(res_19712, x_19714);
                float res_19716 = ((__local
                                    float *) mem_23339)[sext_i32_i64(i_19715)];
                
                res_19713 = res_19716;
            } else {
                float x_19717 = ((__local
                                  float *) mem_23339)[sext_i32_i64(res_19712)];
                float x_19718 = x_19717 / v1_19707;
                int32_t y_19719 = sub32(k2p2zq_18071, 1);
                bool cond_19720 = slt32(res_19711, y_19719);
                float res_19721;
                
                if (cond_19720) {
                    int32_t x_19722 = add32(1, res_19711);
                    int32_t x_19723 = mul32(m_18193, x_19722);
                    int32_t i_19724 = add32(res_19712, x_19723);
                    float x_19725 = ((__local
                                      float *) mem_23339)[sext_i32_i64(i_19724)];
                    int32_t i_19726 = add32(i_19703, x_19723);
                    float x_19727 = ((__local
                                      float *) mem_23339)[sext_i32_i64(i_19726)];
                    float y_19728 = x_19718 * x_19727;
                    float res_19729 = x_19725 - y_19728;
                    
                    res_19721 = res_19729;
                } else {
                    res_19721 = x_19718;
                }
                res_19713 = res_19721;
            }
            ((__local float *) mem_23350)[sext_i32_i64(gtid_19527)] = res_19713;
        }
        
      error_1:
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_failure)
            return;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t write_i_19549 = ltid_pre_24181;
        int32_t phys_tid_19550 = local_tid_24175;
        
        if (slt32(write_i_19549, nm_18194)) {
            float write_value_19732 = ((__local
                                        float *) mem_23350)[sext_i32_i64(write_i_19549)];
            
            if (sle32(0, write_i_19549) && slt32(write_i_19549, nm_18194)) {
                ((__local float *) mem_23339)[sext_i32_i64(write_i_19549)] =
                    write_value_19732;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    __local char *mem_23362;
    
    mem_23362 = (__local char *) mem_23362_backing_2;
    
    int32_t gtid_19552 = ltid_pre_24179;
    int32_t gtid_19553 = ltid_pre_24180;
    int32_t phys_tid_19554 = local_tid_24175;
    
    if (slt32(gtid_19552, k2p2zq_18071) && slt32(gtid_19553, k2p2zq_18071)) {
        int32_t index_primexp_22397 = m_18193 * gtid_19552;
        int32_t i_19739 = add32(k2p2zq_18071, gtid_19553);
        int32_t new_index_19740 = i_19739 + index_primexp_22397;
        float res_19741 = ((__local
                            float *) mem_23339)[sext_i32_i64(new_index_19740)];
        
        ((__local float *) mem_23362)[sext_i32_i64(gtid_19552) *
                                      sext_i32_i64(k2p2zq_18071) +
                                      sext_i32_i64(gtid_19553)] = res_19741;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t i_24183 = 0; i_24183 < sdiv_up32(k2p2zq_18071 * k2p2zq_18071 -
                                                  local_tid_24175,
                                                  computed_group_sizze_19515);
         i_24183++) {
        ((__global float *) mem_23369)[sext_i32_i64(gtid_19513) *
                                       sext_i32_i64(k2p2zq_18071 *
                                       k2p2zq_18071) +
                                       sext_i32_i64(squot32(i_24183 *
                                                            computed_group_sizze_19515 +
                                                            local_tid_24175,
                                                            k2p2zq_18071)) *
                                       sext_i32_i64(k2p2zq_18071) +
                                       sext_i32_i64(i_24183 *
                                       computed_group_sizze_19515 +
                                       local_tid_24175 - squot32(i_24183 *
                                                                 computed_group_sizze_19515 +
                                                                 local_tid_24175,
                                                                 k2p2zq_18071) *
                                       k2p2zq_18071)] = ((__local
                                                          float *) mem_23362)[sext_i32_i64(squot32(i_24183 *
                                                                                                   computed_group_sizze_19515 +
                                                                                                   local_tid_24175,
                                                                                                   k2p2zq_18071)) *
                                                                              sext_i32_i64(k2p2zq_18071) +
                                                                              sext_i32_i64(i_24183 *
                                                                              computed_group_sizze_19515 +
                                                                              local_tid_24175 -
                                                                              squot32(i_24183 *
                                                                                      computed_group_sizze_19515 +
                                                                                      local_tid_24175,
                                                                                      k2p2zq_18071) *
                                                                              k2p2zq_18071)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
  error_4:
    return;
}
__kernel void mainzisegmap_intragroup_19921(__global int *global_failure,
                                            __local volatile
                                            int64_t *mem_23395_backing_aligned_0,
                                            int32_t m_18055,
                                            int32_t k2p2zq_18071,
                                            int32_t m_18193, int32_t nm_18194,
                                            int32_t i_20329,
                                            int32_t ctx_param_ext_23378,
                                            int32_t ctx_param_ext_23379,
                                            int32_t ctx_param_ext_23381,
                                            __global
                                            unsigned char *mem_param_23383,
                                            __global unsigned char *mem_23390,
                                            __global unsigned char *mem_23401)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_23395_backing_0 = (__local volatile
                                                           char *) mem_23395_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24210;
    int32_t local_tid_24211;
    int32_t group_sizze_24214;
    int32_t wave_sizze_24213;
    int32_t group_tid_24212;
    
    global_tid_24210 = get_global_id(0);
    local_tid_24211 = get_local_id(0);
    group_sizze_24214 = get_local_size(0);
    wave_sizze_24213 = LOCKSTEP_WIDTH;
    group_tid_24212 = get_group_id(0);
    
    int32_t phys_tid_19921;
    
    phys_tid_19921 = group_tid_24212;
    
    int32_t ltid_pre_24215;
    
    ltid_pre_24215 = local_tid_24211;
    
    int32_t gtid_19894;
    
    gtid_19894 = group_tid_24212;
    
    float v1_20345 = ((__global
                       float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                                 (sext_i32_i64(gtid_19894) *
                                                  sext_i32_i64(ctx_param_ext_23379) +
                                                  sext_i32_i64(i_20329) *
                                                  sext_i32_i64(ctx_param_ext_23381))];
    bool cond_20346 = v1_20345 == 0.0F;
    __local char *mem_23395;
    
    mem_23395 = (__local char *) mem_23395_backing_0;
    
    int32_t gtid_19897 = ltid_pre_24215;
    int32_t phys_tid_19898 = local_tid_24211;
    
    if (slt32(gtid_19897, nm_18194)) {
        int32_t res_20349 = sdiv32(gtid_19897, m_18193);
        int32_t res_20350 = smod32(gtid_19897, m_18193);
        float res_20351;
        
        if (cond_20346) {
            int32_t x_20352 = mul32(m_18193, res_20349);
            int32_t i_20353 = add32(res_20350, x_20352);
            float res_20354 = ((__global
                                float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                                          (sext_i32_i64(gtid_19894) *
                                                           sext_i32_i64(ctx_param_ext_23379) +
                                                           sext_i32_i64(i_20353) *
                                                           sext_i32_i64(ctx_param_ext_23381))];
            
            res_20351 = res_20354;
        } else {
            float x_20355 = ((__global
                              float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                                        (sext_i32_i64(gtid_19894) *
                                                         sext_i32_i64(ctx_param_ext_23379) +
                                                         sext_i32_i64(res_20350) *
                                                         sext_i32_i64(ctx_param_ext_23381))];
            float x_20356 = x_20355 / v1_20345;
            int32_t y_20357 = sub32(k2p2zq_18071, 1);
            bool cond_20358 = slt32(res_20349, y_20357);
            float res_20359;
            
            if (cond_20358) {
                int32_t x_20360 = add32(1, res_20349);
                int32_t x_20361 = mul32(m_18193, x_20360);
                int32_t i_20362 = add32(res_20350, x_20361);
                float x_20363 = ((__global
                                  float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                                            (sext_i32_i64(gtid_19894) *
                                                             sext_i32_i64(ctx_param_ext_23379) +
                                                             sext_i32_i64(i_20362) *
                                                             sext_i32_i64(ctx_param_ext_23381))];
                int32_t i_20364 = add32(i_20329, x_20361);
                float x_20365 = ((__global
                                  float *) mem_param_23383)[sext_i32_i64(ctx_param_ext_23378) +
                                                            (sext_i32_i64(gtid_19894) *
                                                             sext_i32_i64(ctx_param_ext_23379) +
                                                             sext_i32_i64(i_20364) *
                                                             sext_i32_i64(ctx_param_ext_23381))];
                float y_20366 = x_20356 * x_20365;
                float res_20367 = x_20363 - y_20366;
                
                res_20359 = res_20367;
            } else {
                res_20359 = x_20356;
            }
            res_20351 = res_20359;
        }
        ((__local float *) mem_23395)[sext_i32_i64(gtid_19897)] = res_20351;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t write_i_19919 = ltid_pre_24215;
    int32_t phys_tid_19920 = local_tid_24211;
    
    if (slt32(write_i_19919, nm_18194)) {
        float write_value_20370 = ((__local
                                    float *) mem_23395)[sext_i32_i64(write_i_19919)];
        
        if (sle32(0, write_i_19919) && slt32(write_i_19919, nm_18194)) {
            ((__global float *) mem_23390)[sext_i32_i64(gtid_19894) +
                                           sext_i32_i64(write_i_19919) *
                                           sext_i32_i64(m_18055)] =
                write_value_20370;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_tid_24211 == 0) {
        for (int32_t i_24216 = 0; i_24216 < nm_18194; i_24216++) {
            ((__global float *) mem_23401)[sext_i32_i64(gtid_19894) *
                                           sext_i32_i64(nm_18194) +
                                           sext_i32_i64(i_24216)] = ((__global
                                                                      float *) mem_23390)[sext_i32_i64(gtid_19894) +
                                                                                          sext_i32_i64(i_24216) *
                                                                                          sext_i32_i64(m_18055)];
        }
    }
    
  error_2:
    return;
}
__kernel void mainzisegmap_intragroup_20961(__global int *global_failure,
                                            __local volatile
                                            int64_t *mem_23757_backing_aligned_0,
                                            __local volatile
                                            int64_t *mem_23754_backing_aligned_1,
                                            __local volatile
                                            int64_t *mem_23751_backing_aligned_2,
                                            __local volatile
                                            int64_t *mem_23748_backing_aligned_3,
                                            int32_t N_18054, int32_t N_18056,
                                            int32_t i_18298, __global
                                            unsigned char *images_mem_23189,
                                            __global
                                            unsigned char *res_mem_23733,
                                            __global unsigned char *mem_23761,
                                            __global unsigned char *mem_23766,
                                            __global unsigned char *mem_23771)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_23757_backing_3 = (__local volatile
                                                           char *) mem_23757_backing_aligned_0;
    __local volatile char *restrict mem_23754_backing_2 = (__local volatile
                                                           char *) mem_23754_backing_aligned_1;
    __local volatile char *restrict mem_23751_backing_1 = (__local volatile
                                                           char *) mem_23751_backing_aligned_2;
    __local volatile char *restrict mem_23748_backing_0 = (__local volatile
                                                           char *) mem_23748_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24512;
    int32_t local_tid_24513;
    int32_t group_sizze_24516;
    int32_t wave_sizze_24515;
    int32_t group_tid_24514;
    
    global_tid_24512 = get_global_id(0);
    local_tid_24513 = get_local_id(0);
    group_sizze_24516 = get_local_size(0);
    wave_sizze_24515 = LOCKSTEP_WIDTH;
    group_tid_24514 = get_group_id(0);
    
    int32_t phys_tid_20961;
    
    phys_tid_20961 = group_tid_24514;
    
    int32_t ltid_pre_24517;
    
    ltid_pre_24517 = local_tid_24513;
    
    int32_t gtid_20954;
    
    gtid_20954 = group_tid_24514;
    
    __local char *mem_23748;
    
    mem_23748 = (__local char *) mem_23748_backing_0;
    
    __local char *mem_23751;
    
    mem_23751 = (__local char *) mem_23751_backing_1;
    
    int32_t gtid_20957 = ltid_pre_24517;
    int32_t phys_tid_20958 = local_tid_24513;
    
    if (slt32(gtid_20957, N_18054)) {
        float x_21050 = ((__global
                          float *) images_mem_23189)[sext_i32_i64(gtid_20954) *
                                                     sext_i32_i64(N_18056) +
                                                     sext_i32_i64(gtid_20957)];
        bool res_21052;
        
        res_21052 = futrts_isnan32(x_21050);
        
        bool cond_21053 = !res_21052;
        float res_21054;
        
        if (cond_21053) {
            float x_21051 = ((__global
                              float *) res_mem_23733)[sext_i32_i64(gtid_20954) *
                                                      sext_i32_i64(N_18054) +
                                                      sext_i32_i64(gtid_20957)];
            float res_21055 = x_21050 - x_21051;
            
            res_21054 = res_21055;
        } else {
            res_21054 = NAN;
        }
        
        bool res_21056;
        
        res_21056 = futrts_isnan32(res_21054);
        
        bool res_21057 = !res_21056;
        int32_t res_21058 = btoi_bool_i32(res_21057);
        
        ((__local int32_t *) mem_23748)[sext_i32_i64(gtid_20957)] = res_21058;
        ((__local float *) mem_23751)[sext_i32_i64(gtid_20957)] = res_21054;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_24518;
    
    dims_flat_24518 = N_18054;
    
    int32_t x_21047;
    int32_t x_21048;
    int32_t x_24520;
    int32_t x_24521;
    int32_t skip_threads_24523;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_24513, N_18054)) {
            x_21048 = ((volatile __local
                        int32_t *) mem_23748)[sext_i32_i64(local_tid_24513)];
            if ((local_tid_24513 - squot32(local_tid_24513, 32) * 32) == 0) {
                x_21047 = x_21048;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_24523 = 1;
        while (slt32(skip_threads_24523, 32)) {
            if (sle32(skip_threads_24523, local_tid_24513 -
                      squot32(local_tid_24513, 32) * 32) &&
                slt32(local_tid_24513, N_18054)) {
                // read operands
                {
                    x_21047 = ((volatile __local
                                int32_t *) mem_23748)[sext_i32_i64(local_tid_24513 -
                                                      skip_threads_24523)];
                }
                // perform operation
                {
                    bool inactive_24524 = slt32(srem32(local_tid_24513,
                                                       N_18054),
                                                local_tid_24513 -
                                                (local_tid_24513 -
                                                 skip_threads_24523));
                    
                    if (inactive_24524) {
                        x_21047 = x_21048;
                    }
                    if (!inactive_24524) {
                        int32_t res_21049 = add32(x_21047, x_21048);
                        
                        x_21047 = res_21049;
                    }
                }
            }
            if (sle32(wave_sizze_24515, skip_threads_24523)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_24523, local_tid_24513 -
                      squot32(local_tid_24513, 32) * 32) &&
                slt32(local_tid_24513, N_18054)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) mem_23748)[sext_i32_i64(local_tid_24513)] =
                        x_21047;
                    x_21048 = x_21047;
                }
            }
            if (sle32(wave_sizze_24515, skip_threads_24523)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_24523 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_24513 - squot32(local_tid_24513, 32) * 32) == 31 &&
            slt32(local_tid_24513, N_18054)) {
            ((volatile __local
              int32_t *) mem_23748)[sext_i32_i64(squot32(local_tid_24513,
                                                         32))] = x_21047;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_24525;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_24513, 32) == 0 && slt32(local_tid_24513,
                                                           N_18054)) {
                x_24521 = ((volatile __local
                            int32_t *) mem_23748)[sext_i32_i64(local_tid_24513)];
                if ((local_tid_24513 - squot32(local_tid_24513, 32) * 32) ==
                    0) {
                    x_24520 = x_24521;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_24525 = 1;
            while (slt32(skip_threads_24525, 32)) {
                if (sle32(skip_threads_24525, local_tid_24513 -
                          squot32(local_tid_24513, 32) * 32) &&
                    (squot32(local_tid_24513, 32) == 0 && slt32(local_tid_24513,
                                                                N_18054))) {
                    // read operands
                    {
                        x_24520 = ((volatile __local
                                    int32_t *) mem_23748)[sext_i32_i64(local_tid_24513 -
                                                          skip_threads_24525)];
                    }
                    // perform operation
                    {
                        bool inactive_24526 = slt32(srem32(local_tid_24513 *
                                                           32 + 32 - 1,
                                                           N_18054),
                                                    local_tid_24513 * 32 + 32 -
                                                    1 - ((local_tid_24513 -
                                                          skip_threads_24525) *
                                                         32 + 32 - 1));
                        
                        if (inactive_24526) {
                            x_24520 = x_24521;
                        }
                        if (!inactive_24526) {
                            int32_t res_24522 = add32(x_24520, x_24521);
                            
                            x_24520 = res_24522;
                        }
                    }
                }
                if (sle32(wave_sizze_24515, skip_threads_24525)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_24525, local_tid_24513 -
                          squot32(local_tid_24513, 32) * 32) &&
                    (squot32(local_tid_24513, 32) == 0 && slt32(local_tid_24513,
                                                                N_18054))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) mem_23748)[sext_i32_i64(local_tid_24513)] =
                            x_24520;
                        x_24521 = x_24520;
                    }
                }
                if (sle32(wave_sizze_24515, skip_threads_24525)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_24525 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_24513, 32) == 0 || !slt32(local_tid_24513,
                                                          N_18054))) {
            // read operands
            {
                x_21048 = x_21047;
                x_21047 = ((__local
                            int32_t *) mem_23748)[sext_i32_i64(squot32(local_tid_24513,
                                                                       32) -
                                                  1)];
            }
            // perform operation
            {
                bool inactive_24527 = slt32(srem32(local_tid_24513, N_18054),
                                            local_tid_24513 -
                                            (squot32(local_tid_24513, 32) * 32 -
                                             1));
                
                if (inactive_24527) {
                    x_21047 = x_21048;
                }
                if (!inactive_24527) {
                    int32_t res_21049 = add32(x_21047, x_21048);
                    
                    x_21047 = res_21049;
                }
            }
            // write final result
            {
                ((__local int32_t *) mem_23748)[sext_i32_i64(local_tid_24513)] =
                    x_21047;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_24513, 32) == 0) {
            ((__local int32_t *) mem_23748)[sext_i32_i64(local_tid_24513)] =
                x_21048;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t res_21059 = ((__local int32_t *) mem_23748)[sext_i32_i64(i_18298)];
    __local char *mem_23754;
    
    mem_23754 = (__local char *) mem_23754_backing_2;
    ((__local float *) mem_23754)[sext_i32_i64(local_tid_24513)] = NAN;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_23757;
    
    mem_23757 = (__local char *) mem_23757_backing_3;
    ((__local int32_t *) mem_23757)[sext_i32_i64(local_tid_24513)] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t write_i_20959 = ltid_pre_24517;
    int32_t phys_tid_20960 = local_tid_24513;
    
    if (slt32(write_i_20959, N_18054)) {
        float x_21064 = ((__local
                          float *) mem_23751)[sext_i32_i64(write_i_20959)];
        bool res_21067;
        
        res_21067 = futrts_isnan32(x_21064);
        
        bool res_21068 = !res_21067;
        int32_t res_21069;
        
        if (res_21068) {
            int32_t x_21065 = ((__local
                                int32_t *) mem_23748)[sext_i32_i64(write_i_20959)];
            int32_t res_21070 = sub32(x_21065, 1);
            
            res_21069 = res_21070;
        } else {
            res_21069 = -1;
        }
        if (sle32(0, res_21069) && slt32(res_21069, N_18054)) {
            ((__local int32_t *) mem_23757)[sext_i32_i64(res_21069)] =
                write_i_20959;
        }
        if (sle32(0, res_21069) && slt32(res_21069, N_18054)) {
            ((__local float *) mem_23754)[sext_i32_i64(res_21069)] = x_21064;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_tid_24513 == 0) {
        ((__global int32_t *) mem_23761)[sext_i32_i64(gtid_20954)] = res_21059;
    }
    ((__global float *) mem_23766)[sext_i32_i64(gtid_20954) *
                                   sext_i32_i64(N_18054) +
                                   sext_i32_i64(local_tid_24513)] = ((__local
                                                                      float *) mem_23754)[sext_i32_i64(local_tid_24513)];
    barrier(CLK_LOCAL_MEM_FENCE);
    ((__global int32_t *) mem_23771)[sext_i32_i64(gtid_20954) *
                                     sext_i32_i64(N_18054) +
                                     sext_i32_i64(local_tid_24513)] = ((__local
                                                                        int32_t *) mem_23757)[sext_i32_i64(local_tid_24513)];
    barrier(CLK_LOCAL_MEM_FENCE);
    
  error_2:
    return;
}
__kernel void mainzisegmap_intragroup_21326(__global int *global_failure,
                                            __local volatile
                                            int64_t *red_arr_mem_24612_backing_aligned_0,
                                            __local volatile
                                            int64_t *red_arr_mem_24608_backing_aligned_1,
                                            int32_t N_18054, int32_t N_18056,
                                            int32_t n_18059, float hfrac_18061,
                                            int32_t k2p2_18069, __global
                                            unsigned char *images_mem_23189,
                                            __global
                                            unsigned char *res_mem_23788,
                                            __global unsigned char *mem_23815,
                                            __global unsigned char *mem_23818,
                                            __global unsigned char *mem_23821)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24612_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_24612_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24608_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24608_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24602;
    int32_t local_tid_24603;
    int32_t group_sizze_24606;
    int32_t wave_sizze_24605;
    int32_t group_tid_24604;
    
    global_tid_24602 = get_global_id(0);
    local_tid_24603 = get_local_id(0);
    group_sizze_24606 = get_local_size(0);
    wave_sizze_24605 = LOCKSTEP_WIDTH;
    group_tid_24604 = get_group_id(0);
    
    int32_t phys_tid_21326;
    
    phys_tid_21326 = group_tid_24604;
    
    int32_t ltid_pre_24607;
    
    ltid_pre_24607 = local_tid_24603;
    
    int32_t gtid_21319;
    
    gtid_21319 = group_tid_24604;
    
    int32_t res_21407;
    int32_t gtid_21322 = ltid_pre_24607;
    int32_t phys_tid_21323 = local_tid_24603;
    __local char *red_arr_mem_24608;
    
    red_arr_mem_24608 = (__local char *) red_arr_mem_24608_backing_0;
    if (slt32(gtid_21322, n_18059)) {
        float x_21411 = ((__global
                          float *) images_mem_23189)[sext_i32_i64(gtid_21319) *
                                                     sext_i32_i64(N_18056) +
                                                     sext_i32_i64(gtid_21322)];
        bool res_21412;
        
        res_21412 = futrts_isnan32(x_21411);
        
        bool cond_21413 = !res_21412;
        int32_t res_21414 = btoi_bool_i32(cond_21413);
        
        ((__local int32_t *) red_arr_mem_24608)[sext_i32_i64(gtid_21322)] =
            res_21414;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_24610;
    int32_t skip_waves_24611;
    int32_t x_21408;
    int32_t x_21409;
    
    offset_24610 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_24603, n_18059)) {
            x_21408 = ((__local
                        int32_t *) red_arr_mem_24608)[sext_i32_i64(local_tid_24603 +
                                                      offset_24610)];
        }
    }
    offset_24610 = 1;
    while (slt32(offset_24610, wave_sizze_24605)) {
        if (slt32(local_tid_24603 + offset_24610, n_18059) &&
            ((local_tid_24603 - squot32(local_tid_24603, wave_sizze_24605) *
              wave_sizze_24605) & (2 * offset_24610 - 1)) == 0) {
            // read array element
            {
                x_21409 = ((volatile __local
                            int32_t *) red_arr_mem_24608)[sext_i32_i64(local_tid_24603 +
                                                          offset_24610)];
            }
            // apply reduction operation
            {
                int32_t res_21410 = add32(x_21408, x_21409);
                
                x_21408 = res_21410;
            }
            // write result of operation
            {
                ((volatile __local
                  int32_t *) red_arr_mem_24608)[sext_i32_i64(local_tid_24603)] =
                    x_21408;
            }
        }
        offset_24610 *= 2;
    }
    skip_waves_24611 = 1;
    while (slt32(skip_waves_24611, squot32(n_18059 + wave_sizze_24605 - 1,
                                           wave_sizze_24605))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_24610 = skip_waves_24611 * wave_sizze_24605;
        if (slt32(local_tid_24603 + offset_24610, n_18059) &&
            ((local_tid_24603 - squot32(local_tid_24603, wave_sizze_24605) *
              wave_sizze_24605) == 0 && (squot32(local_tid_24603,
                                                 wave_sizze_24605) & (2 *
                                                                      skip_waves_24611 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_21409 = ((__local
                            int32_t *) red_arr_mem_24608)[sext_i32_i64(local_tid_24603 +
                                                          offset_24610)];
            }
            // apply reduction operation
            {
                int32_t res_21410 = add32(x_21408, x_21409);
                
                x_21408 = res_21410;
            }
            // write result of operation
            {
                ((__local
                  int32_t *) red_arr_mem_24608)[sext_i32_i64(local_tid_24603)] =
                    x_21408;
            }
        }
        skip_waves_24611 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_21407 = ((__local int32_t *) red_arr_mem_24608)[0];
    
    float res_21415;
    int32_t gtid_21324 = ltid_pre_24607;
    int32_t phys_tid_21325 = local_tid_24603;
    __local char *red_arr_mem_24612;
    
    red_arr_mem_24612 = (__local char *) red_arr_mem_24612_backing_1;
    if (slt32(gtid_21324, n_18059)) {
        bool cond_21421 = slt32(gtid_21324, res_21407);
        float res_21422;
        
        if (cond_21421) {
            float x_elem_21420 = ((__global
                                   float *) res_mem_23788)[sext_i32_i64(gtid_21319) *
                                                           sext_i32_i64(N_18054) +
                                                           sext_i32_i64(gtid_21324)];
            
            res_21422 = x_elem_21420;
        } else {
            res_21422 = 0.0F;
        }
        
        float res_21423 = res_21422 * res_21422;
        
        ((__local float *) red_arr_mem_24612)[sext_i32_i64(gtid_21324)] =
            res_21423;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_24614;
    int32_t skip_waves_24615;
    float x_21416;
    float x_21417;
    
    offset_24614 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_24603, n_18059)) {
            x_21416 = ((__local
                        float *) red_arr_mem_24612)[sext_i32_i64(local_tid_24603 +
                                                    offset_24614)];
        }
    }
    offset_24614 = 1;
    while (slt32(offset_24614, wave_sizze_24605)) {
        if (slt32(local_tid_24603 + offset_24614, n_18059) &&
            ((local_tid_24603 - squot32(local_tid_24603, wave_sizze_24605) *
              wave_sizze_24605) & (2 * offset_24614 - 1)) == 0) {
            // read array element
            {
                x_21417 = ((volatile __local
                            float *) red_arr_mem_24612)[sext_i32_i64(local_tid_24603 +
                                                        offset_24614)];
            }
            // apply reduction operation
            {
                float res_21418 = x_21416 + x_21417;
                
                x_21416 = res_21418;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_24612)[sext_i32_i64(local_tid_24603)] =
                    x_21416;
            }
        }
        offset_24614 *= 2;
    }
    skip_waves_24615 = 1;
    while (slt32(skip_waves_24615, squot32(n_18059 + wave_sizze_24605 - 1,
                                           wave_sizze_24605))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_24614 = skip_waves_24615 * wave_sizze_24605;
        if (slt32(local_tid_24603 + offset_24614, n_18059) &&
            ((local_tid_24603 - squot32(local_tid_24603, wave_sizze_24605) *
              wave_sizze_24605) == 0 && (squot32(local_tid_24603,
                                                 wave_sizze_24605) & (2 *
                                                                      skip_waves_24615 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_21417 = ((__local
                            float *) red_arr_mem_24612)[sext_i32_i64(local_tid_24603 +
                                                        offset_24614)];
            }
            // apply reduction operation
            {
                float res_21418 = x_21416 + x_21417;
                
                x_21416 = res_21418;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_24612)[sext_i32_i64(local_tid_24603)] =
                    x_21416;
            }
        }
        skip_waves_24615 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_21415 = ((__local float *) red_arr_mem_24612)[0];
    
    int32_t r32_arg_21424 = sub32(res_21407, k2p2_18069);
    float res_21425 = sitofp_i32_f32(r32_arg_21424);
    float sqrt_arg_21426 = res_21415 / res_21425;
    float res_21427;
    
    res_21427 = futrts_sqrt32(sqrt_arg_21426);
    
    float res_21428 = sitofp_i32_f32(res_21407);
    float t32_arg_21429 = hfrac_18061 * res_21428;
    int32_t res_21430 = fptosi_f32_i32(t32_arg_21429);
    
    if (local_tid_24603 == 0) {
        ((__global int32_t *) mem_23815)[sext_i32_i64(gtid_21319)] = res_21430;
    }
    if (local_tid_24603 == 0) {
        ((__global int32_t *) mem_23818)[sext_i32_i64(gtid_21319)] = res_21407;
    }
    if (local_tid_24603 == 0) {
        ((__global float *) mem_23821)[sext_i32_i64(gtid_21319)] = res_21427;
    }
    
  error_4:
    return;
}
__kernel void mainzisegmap_intragroup_21740(__global int *global_failure,
                                            __local volatile
                                            int64_t *red_arr_mem_24867_backing_aligned_0,
                                            __local volatile
                                            int64_t *red_arr_mem_24865_backing_aligned_1,
                                            __local volatile
                                            int64_t *red_arr_mem_24863_backing_aligned_2,
                                            __local volatile
                                            int64_t *mem_23863_backing_aligned_3,
                                            int32_t N_18054, int32_t n_18059,
                                            int32_t iota_arg_18403, __global
                                            unsigned char *res_mem_23787,
                                            __global
                                            unsigned char *res_mem_23788,
                                            __global
                                            unsigned char *res_mem_23789,
                                            __global
                                            unsigned char *res_mem_23840,
                                            __global
                                            unsigned char *res_mem_23841,
                                            __global
                                            unsigned char *res_mem_23842,
                                            __global
                                            unsigned char *res_mem_23854,
                                            __global unsigned char *mem_23858,
                                            __global unsigned char *mem_23867,
                                            __global unsigned char *mem_23870)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24867_backing_3 =
                          (__local volatile
                           char *) red_arr_mem_24867_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24865_backing_2 =
                          (__local volatile
                           char *) red_arr_mem_24865_backing_aligned_1;
    __local volatile char *restrict red_arr_mem_24863_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_24863_backing_aligned_2;
    __local volatile char *restrict mem_23863_backing_0 = (__local volatile
                                                           char *) mem_23863_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24847;
    int32_t local_tid_24848;
    int32_t group_sizze_24851;
    int32_t wave_sizze_24850;
    int32_t group_tid_24849;
    
    global_tid_24847 = get_global_id(0);
    local_tid_24848 = get_local_id(0);
    group_sizze_24851 = get_local_size(0);
    wave_sizze_24850 = LOCKSTEP_WIDTH;
    group_tid_24849 = get_group_id(0);
    
    int32_t phys_tid_21740;
    
    phys_tid_21740 = group_tid_24849;
    
    int32_t ltid_pre_24852;
    
    ltid_pre_24852 = local_tid_24848;
    
    int32_t gtid_21733;
    
    gtid_21733 = group_tid_24849;
    
    int32_t x_21953;
    
    x_21953 = ((__global int32_t *) res_mem_23787)[sext_i32_i64(gtid_21733)];
    
    int32_t x_21954 = ((__global
                        int32_t *) res_mem_23841)[sext_i32_i64(gtid_21733)];
    float x_21955 = ((__global
                      float *) res_mem_23842)[sext_i32_i64(gtid_21733)];
    int32_t x_21956 = ((__global
                        int32_t *) res_mem_23840)[sext_i32_i64(gtid_21733)];
    float x_21957 = ((__global
                      float *) res_mem_23854)[sext_i32_i64(gtid_21733)];
    int32_t y_21960 = sub32(x_21953, x_21954);
    float res_21961 = sitofp_i32_f32(x_21954);
    float res_21962;
    
    res_21962 = futrts_sqrt32(res_21961);
    
    float y_21963 = x_21955 * res_21962;
    __local char *mem_23863;
    
    mem_23863 = (__local char *) mem_23863_backing_0;
    
    int32_t gtid_21736 = ltid_pre_24852;
    int32_t phys_tid_21737 = local_tid_24848;
    
    if (slt32(gtid_21736, iota_arg_18403)) {
        bool cond_21976 = sle32(y_21960, gtid_21736);
        float res_21977;
        
        if (cond_21976) {
            res_21977 = 0.0F;
        } else {
            bool cond_21978 = gtid_21736 == 0;
            float res_21979;
            
            if (cond_21978) {
                res_21979 = x_21957;
            } else {
                int32_t x_21980 = sub32(x_21954, x_21956);
                int32_t i_21981 = add32(gtid_21736, x_21980);
                float negate_arg_21982 = ((__global
                                           float *) res_mem_23788)[sext_i32_i64(gtid_21733) *
                                                                   sext_i32_i64(N_18054) +
                                                                   sext_i32_i64(i_21981)];
                float x_21983 = 0.0F - negate_arg_21982;
                int32_t i_21984 = add32(gtid_21736, x_21954);
                float y_21985 = ((__global
                                  float *) res_mem_23788)[sext_i32_i64(gtid_21733) *
                                                          sext_i32_i64(N_18054) +
                                                          sext_i32_i64(i_21984)];
                float res_21986 = x_21983 + y_21985;
                
                res_21979 = res_21986;
            }
            res_21977 = res_21979;
        }
        ((__local float *) mem_23863)[sext_i32_i64(gtid_21736)] = res_21977;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_24853;
    
    dims_flat_24853 = iota_arg_18403;
    
    float x_21972;
    float x_21973;
    float x_24855;
    float x_24856;
    int32_t skip_threads_24858;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_24848, iota_arg_18403)) {
            x_21973 = ((volatile __local
                        float *) mem_23863)[sext_i32_i64(local_tid_24848)];
            if ((local_tid_24848 - squot32(local_tid_24848, 32) * 32) == 0) {
                x_21972 = x_21973;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_24858 = 1;
        while (slt32(skip_threads_24858, 32)) {
            if (sle32(skip_threads_24858, local_tid_24848 -
                      squot32(local_tid_24848, 32) * 32) &&
                slt32(local_tid_24848, iota_arg_18403)) {
                // read operands
                {
                    x_21972 = ((volatile __local
                                float *) mem_23863)[sext_i32_i64(local_tid_24848 -
                                                    skip_threads_24858)];
                }
                // perform operation
                {
                    bool inactive_24859 = slt32(srem32(local_tid_24848,
                                                       iota_arg_18403),
                                                local_tid_24848 -
                                                (local_tid_24848 -
                                                 skip_threads_24858));
                    
                    if (inactive_24859) {
                        x_21972 = x_21973;
                    }
                    if (!inactive_24859) {
                        float res_21974 = x_21972 + x_21973;
                        
                        x_21972 = res_21974;
                    }
                }
            }
            if (sle32(wave_sizze_24850, skip_threads_24858)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_24858, local_tid_24848 -
                      squot32(local_tid_24848, 32) * 32) &&
                slt32(local_tid_24848, iota_arg_18403)) {
                // write result
                {
                    ((volatile __local
                      float *) mem_23863)[sext_i32_i64(local_tid_24848)] =
                        x_21972;
                    x_21973 = x_21972;
                }
            }
            if (sle32(wave_sizze_24850, skip_threads_24858)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_24858 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_24848 - squot32(local_tid_24848, 32) * 32) == 31 &&
            slt32(local_tid_24848, iota_arg_18403)) {
            ((volatile __local
              float *) mem_23863)[sext_i32_i64(squot32(local_tid_24848, 32))] =
                x_21972;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_24860;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_24848, 32) == 0 && slt32(local_tid_24848,
                                                           iota_arg_18403)) {
                x_24856 = ((volatile __local
                            float *) mem_23863)[sext_i32_i64(local_tid_24848)];
                if ((local_tid_24848 - squot32(local_tid_24848, 32) * 32) ==
                    0) {
                    x_24855 = x_24856;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_24860 = 1;
            while (slt32(skip_threads_24860, 32)) {
                if (sle32(skip_threads_24860, local_tid_24848 -
                          squot32(local_tid_24848, 32) * 32) &&
                    (squot32(local_tid_24848, 32) == 0 && slt32(local_tid_24848,
                                                                iota_arg_18403))) {
                    // read operands
                    {
                        x_24855 = ((volatile __local
                                    float *) mem_23863)[sext_i32_i64(local_tid_24848 -
                                                        skip_threads_24860)];
                    }
                    // perform operation
                    {
                        bool inactive_24861 = slt32(srem32(local_tid_24848 *
                                                           32 + 32 - 1,
                                                           iota_arg_18403),
                                                    local_tid_24848 * 32 + 32 -
                                                    1 - ((local_tid_24848 -
                                                          skip_threads_24860) *
                                                         32 + 32 - 1));
                        
                        if (inactive_24861) {
                            x_24855 = x_24856;
                        }
                        if (!inactive_24861) {
                            float res_24857 = x_24855 + x_24856;
                            
                            x_24855 = res_24857;
                        }
                    }
                }
                if (sle32(wave_sizze_24850, skip_threads_24860)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_24860, local_tid_24848 -
                          squot32(local_tid_24848, 32) * 32) &&
                    (squot32(local_tid_24848, 32) == 0 && slt32(local_tid_24848,
                                                                iota_arg_18403))) {
                    // write result
                    {
                        ((volatile __local
                          float *) mem_23863)[sext_i32_i64(local_tid_24848)] =
                            x_24855;
                        x_24856 = x_24855;
                    }
                }
                if (sle32(wave_sizze_24850, skip_threads_24860)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_24860 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_24848, 32) == 0 || !slt32(local_tid_24848,
                                                          iota_arg_18403))) {
            // read operands
            {
                x_21973 = x_21972;
                x_21972 = ((__local
                            float *) mem_23863)[sext_i32_i64(squot32(local_tid_24848,
                                                                     32) - 1)];
            }
            // perform operation
            {
                bool inactive_24862 = slt32(srem32(local_tid_24848,
                                                   iota_arg_18403),
                                            local_tid_24848 -
                                            (squot32(local_tid_24848, 32) * 32 -
                                             1));
                
                if (inactive_24862) {
                    x_21972 = x_21973;
                }
                if (!inactive_24862) {
                    float res_21974 = x_21972 + x_21973;
                    
                    x_21972 = res_21974;
                }
            }
            // write final result
            {
                ((__local float *) mem_23863)[sext_i32_i64(local_tid_24848)] =
                    x_21972;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_24848, 32) == 0) {
            ((__local float *) mem_23863)[sext_i32_i64(local_tid_24848)] =
                x_21973;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bool acc0_21992;
    int32_t acc0_21993;
    float acc0_21994;
    int32_t gtid_21738 = ltid_pre_24852;
    int32_t phys_tid_21739 = local_tid_24848;
    __local char *red_arr_mem_24863;
    
    red_arr_mem_24863 = (__local char *) red_arr_mem_24863_backing_1;
    
    __local char *red_arr_mem_24865;
    
    red_arr_mem_24865 = (__local char *) red_arr_mem_24865_backing_2;
    
    __local char *red_arr_mem_24867;
    
    red_arr_mem_24867 = (__local char *) red_arr_mem_24867_backing_3;
    if (slt32(gtid_21738, iota_arg_18403)) {
        float x_22009 = ((__local float *) mem_23863)[sext_i32_i64(gtid_21738)];
        float x_22010 = ((__global
                          float *) mem_23858)[sext_i32_i64(gtid_21738)];
        float res_22013 = x_22009 / y_21963;
        bool cond_22014 = slt32(gtid_21738, y_21960);
        bool res_22015;
        
        res_22015 = futrts_isnan32(res_22013);
        
        bool res_22016 = !res_22015;
        bool x_22017 = cond_22014 && res_22016;
        float res_22018 = (float) fabs(res_22013);
        bool res_22019 = x_22010 < res_22018;
        bool x_22020 = x_22017 && res_22019;
        float res_22021;
        
        if (cond_22014) {
            res_22021 = res_22013;
        } else {
            res_22021 = 0.0F;
        }
        ((__local bool *) red_arr_mem_24863)[sext_i32_i64(gtid_21738)] =
            x_22020;
        ((__local int32_t *) red_arr_mem_24865)[sext_i32_i64(gtid_21738)] =
            gtid_21738;
        ((__local float *) red_arr_mem_24867)[sext_i32_i64(gtid_21738)] =
            res_22021;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_24869;
    int32_t skip_waves_24870;
    bool x_21995;
    int32_t x_21996;
    float x_21997;
    bool x_21998;
    int32_t x_21999;
    float x_22000;
    
    offset_24869 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_24848, iota_arg_18403)) {
            x_21995 = ((__local
                        bool *) red_arr_mem_24863)[sext_i32_i64(local_tid_24848 +
                                                   offset_24869)];
            x_21996 = ((__local
                        int32_t *) red_arr_mem_24865)[sext_i32_i64(local_tid_24848 +
                                                      offset_24869)];
            x_21997 = ((__local
                        float *) red_arr_mem_24867)[sext_i32_i64(local_tid_24848 +
                                                    offset_24869)];
        }
    }
    offset_24869 = 1;
    while (slt32(offset_24869, wave_sizze_24850)) {
        if (slt32(local_tid_24848 + offset_24869, iota_arg_18403) &&
            ((local_tid_24848 - squot32(local_tid_24848, wave_sizze_24850) *
              wave_sizze_24850) & (2 * offset_24869 - 1)) == 0) {
            // read array element
            {
                x_21998 = ((volatile __local
                            bool *) red_arr_mem_24863)[sext_i32_i64(local_tid_24848 +
                                                       offset_24869)];
                x_21999 = ((volatile __local
                            int32_t *) red_arr_mem_24865)[sext_i32_i64(local_tid_24848 +
                                                          offset_24869)];
                x_22000 = ((volatile __local
                            float *) red_arr_mem_24867)[sext_i32_i64(local_tid_24848 +
                                                        offset_24869)];
            }
            // apply reduction operation
            {
                bool res_22001;
                int32_t res_22002;
                
                if (x_21995) {
                    res_22001 = x_21995;
                    res_22002 = x_21996;
                } else {
                    bool x_22003 = x_21998 && x_21998;
                    bool x_22004 = !x_21998;
                    bool y_22005 = x_21995 && x_22004;
                    bool res_22006 = x_22003 || y_22005;
                    int32_t res_22007;
                    
                    if (x_21998) {
                        res_22007 = x_21999;
                    } else {
                        res_22007 = x_21996;
                    }
                    res_22001 = res_22006;
                    res_22002 = res_22007;
                }
                
                float res_22008 = x_21997 + x_22000;
                
                x_21995 = res_22001;
                x_21996 = res_22002;
                x_21997 = res_22008;
            }
            // write result of operation
            {
                ((volatile __local
                  bool *) red_arr_mem_24863)[sext_i32_i64(local_tid_24848)] =
                    x_21995;
                ((volatile __local
                  int32_t *) red_arr_mem_24865)[sext_i32_i64(local_tid_24848)] =
                    x_21996;
                ((volatile __local
                  float *) red_arr_mem_24867)[sext_i32_i64(local_tid_24848)] =
                    x_21997;
            }
        }
        offset_24869 *= 2;
    }
    skip_waves_24870 = 1;
    while (slt32(skip_waves_24870, squot32(iota_arg_18403 + wave_sizze_24850 -
                                           1, wave_sizze_24850))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_24869 = skip_waves_24870 * wave_sizze_24850;
        if (slt32(local_tid_24848 + offset_24869, iota_arg_18403) &&
            ((local_tid_24848 - squot32(local_tid_24848, wave_sizze_24850) *
              wave_sizze_24850) == 0 && (squot32(local_tid_24848,
                                                 wave_sizze_24850) & (2 *
                                                                      skip_waves_24870 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_21998 = ((__local
                            bool *) red_arr_mem_24863)[sext_i32_i64(local_tid_24848 +
                                                       offset_24869)];
                x_21999 = ((__local
                            int32_t *) red_arr_mem_24865)[sext_i32_i64(local_tid_24848 +
                                                          offset_24869)];
                x_22000 = ((__local
                            float *) red_arr_mem_24867)[sext_i32_i64(local_tid_24848 +
                                                        offset_24869)];
            }
            // apply reduction operation
            {
                bool res_22001;
                int32_t res_22002;
                
                if (x_21995) {
                    res_22001 = x_21995;
                    res_22002 = x_21996;
                } else {
                    bool x_22003 = x_21998 && x_21998;
                    bool x_22004 = !x_21998;
                    bool y_22005 = x_21995 && x_22004;
                    bool res_22006 = x_22003 || y_22005;
                    int32_t res_22007;
                    
                    if (x_21998) {
                        res_22007 = x_21999;
                    } else {
                        res_22007 = x_21996;
                    }
                    res_22001 = res_22006;
                    res_22002 = res_22007;
                }
                
                float res_22008 = x_21997 + x_22000;
                
                x_21995 = res_22001;
                x_21996 = res_22002;
                x_21997 = res_22008;
            }
            // write result of operation
            {
                ((__local
                  bool *) red_arr_mem_24863)[sext_i32_i64(local_tid_24848)] =
                    x_21995;
                ((__local
                  int32_t *) red_arr_mem_24865)[sext_i32_i64(local_tid_24848)] =
                    x_21996;
                ((__local
                  float *) red_arr_mem_24867)[sext_i32_i64(local_tid_24848)] =
                    x_21997;
            }
        }
        skip_waves_24870 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    acc0_21992 = ((__local bool *) red_arr_mem_24863)[0];
    acc0_21993 = ((__local int32_t *) red_arr_mem_24865)[0];
    acc0_21994 = ((__local float *) red_arr_mem_24867)[0];
    
    bool x_22024 = acc0_21992 && acc0_21992;
    int32_t res_22028;
    
    if (acc0_21992) {
        res_22028 = acc0_21993;
    } else {
        res_22028 = -1;
    }
    
    bool cond_22034 = !x_22024;
    int32_t fst_breakzq_22035;
    
    if (cond_22034) {
        fst_breakzq_22035 = -1;
    } else {
        bool cond_22036 = slt32(res_22028, y_21960);
        int32_t res_22037;
        
        if (cond_22036) {
            int32_t i_22038 = add32(x_21954, res_22028);
            int32_t x_22039 = ((__global
                                int32_t *) res_mem_23789)[sext_i32_i64(gtid_21733) *
                                                          sext_i32_i64(N_18054) +
                                                          sext_i32_i64(i_22038)];
            int32_t res_22040 = sub32(x_22039, n_18059);
            
            res_22037 = res_22040;
        } else {
            res_22037 = -1;
        }
        fst_breakzq_22035 = res_22037;
    }
    
    bool cond_22041 = sle32(x_21954, 5);
    bool res_22042 = sle32(y_21960, 5);
    bool x_22043 = !cond_22041;
    bool y_22044 = res_22042 && x_22043;
    bool cond_22045 = cond_22041 || y_22044;
    int32_t fst_breakzq_22046;
    
    if (cond_22045) {
        fst_breakzq_22046 = -2;
    } else {
        fst_breakzq_22046 = fst_breakzq_22035;
    }
    if (local_tid_24848 == 0) {
        ((__global int32_t *) mem_23867)[sext_i32_i64(gtid_21733)] =
            fst_breakzq_22046;
    }
    if (local_tid_24848 == 0) {
        ((__global float *) mem_23870)[sext_i32_i64(gtid_21733)] = acc0_21994;
    }
    
  error_3:
    return;
}
__kernel void mainzisegmap_intragroup_22505(__global int *global_failure,
                                            __local volatile
                                            int64_t *mem_23519_backing_aligned_0,
                                            __local volatile
                                            int64_t *mem_23514_backing_aligned_1,
                                            __local volatile
                                            int64_t *mem_23491_backing_aligned_2,
                                            __local volatile
                                            int64_t *mem_23486_backing_aligned_3,
                                            int32_t m_18055, int32_t N_18056,
                                            int32_t n_18059,
                                            int32_t k2p2zq_18071,
                                            int32_t num_groups_y_22503,
                                            int32_t num_whole_tiles_22521,
                                            int32_t residual_input_22671,
                                            unsigned char cond_22672, __global
                                            unsigned char *images_mem_23189,
                                            __global unsigned char *mem_23207,
                                            __global unsigned char *mem_23538)
{
    #define tile_sizze_22500 (mainzitile_sizze_22499)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_23519_backing_7 = (__local volatile
                                                           char *) mem_23519_backing_aligned_0;
    __local volatile char *restrict mem_23514_backing_6 = (__local volatile
                                                           char *) mem_23514_backing_aligned_1;
    __local volatile char *restrict mem_23491_backing_1 = (__local volatile
                                                           char *) mem_23491_backing_aligned_2;
    __local volatile char *restrict mem_23486_backing_0 = (__local volatile
                                                           char *) mem_23486_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24252;
    int32_t local_tid_24253;
    int32_t group_sizze_24256;
    int32_t wave_sizze_24255;
    int32_t group_tid_24254;
    
    global_tid_24252 = get_global_id(0);
    local_tid_24253 = get_local_id(0);
    group_sizze_24256 = get_local_size(0);
    wave_sizze_24255 = LOCKSTEP_WIDTH;
    group_tid_24254 = get_group_id(0);
    
    int32_t gid_flat_22505;
    
    gid_flat_22505 = group_tid_24254;
    
    int32_t ltid_pre_24257;
    
    ltid_pre_24257 = squot32(local_tid_24253, tile_sizze_22500);
    
    int32_t ltid_pre_24258;
    
    ltid_pre_24258 = local_tid_24253 - squot32(local_tid_24253,
                                               tile_sizze_22500) *
        tile_sizze_22500;
    
    int32_t gid_x_22497;
    
    gid_x_22497 = squot32(group_tid_24254, num_groups_y_22503);
    
    int32_t gid_y_22498;
    
    gid_y_22498 = group_tid_24254 - squot32(group_tid_24254,
                                            num_groups_y_22503) *
        num_groups_y_22503;
    
    float mem_23469[1];
    int32_t ltid_x_22522 = ltid_pre_24257;
    int32_t ltid_y_22523 = ltid_pre_24258;
    int32_t ltid_flat_22524 = local_tid_24253;
    
    if (slt32(ltid_x_22522, tile_sizze_22500) && slt32(ltid_y_22523,
                                                       tile_sizze_22500)) {
        mem_23469[0] = 0.0F;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_22621 = gid_x_22497 * tile_sizze_22500;
    int32_t binop_x_22623 = gid_y_22498 * tile_sizze_22500;
    __local char *mem_23486;
    
    mem_23486 = (__local char *) mem_23486_backing_0;
    
    __local char *mem_23491;
    
    mem_23491 = (__local char *) mem_23491_backing_1;
    
    float accs_mem_23508[1];
    float mem_param_23477[1];
    
    for (int32_t i_2 = 0; i_2 < 1; i_2++)
        mem_param_23477[i_2] = mem_23469[i_2];
    for (int32_t tile_id_22533 = 0; tile_id_22533 < num_whole_tiles_22521;
         tile_id_22533++) {
        int32_t binop_x_22617 = tile_sizze_22500 * tile_id_22533;
        int32_t ltid_x_22534 = ltid_pre_24257;
        int32_t ltid_y_22535 = ltid_pre_24258;
        int32_t ltid_flat_22536 = local_tid_24253;
        int32_t i_22618 = ltid_x_22534 + binop_x_22617;
        int32_t j_22620 = ltid_y_22535 + binop_x_22617;
        int32_t gtid_22622 = ltid_x_22534 + binop_x_22621;
        int32_t gtid_22624 = ltid_y_22535 + binop_x_22623;
        bool binop_x_22627 = slt32(i_22618, n_18059);
        bool binop_y_22628 = slt32(gtid_22624, k2p2zq_18071);
        bool cond_22629 = binop_x_22627 && binop_y_22628;
        float pre_22630;
        
        if (cond_22629) {
            float x_22631 = ((__global
                              float *) mem_23207)[sext_i32_i64(i_22618) *
                                                  sext_i32_i64(k2p2zq_18071) +
                                                  sext_i32_i64(gtid_22624)];
            
            pre_22630 = x_22631;
        } else {
            pre_22630 = 0.0F;
        }
        
        bool binop_x_22633 = slt32(j_22620, n_18059);
        bool binop_y_22634 = slt32(gtid_22622, m_18055);
        bool cond_22635 = binop_x_22633 && binop_y_22634;
        float pre_22636;
        
        if (cond_22635) {
            float x_22637 = ((__global
                              float *) images_mem_23189)[sext_i32_i64(gtid_22622) *
                                                         sext_i32_i64(N_18056) +
                                                         sext_i32_i64(j_22620)];
            
            pre_22636 = x_22637;
        } else {
            pre_22636 = 0.0F;
        }
        ((__local float *) mem_23486)[sext_i32_i64(ltid_x_22534) *
                                      sext_i32_i64(tile_sizze_22500) +
                                      sext_i32_i64(ltid_y_22535)] = pre_22630;
        ((__local float *) mem_23491)[sext_i32_i64(ltid_x_22534) *
                                      sext_i32_i64(tile_sizze_22500) +
                                      sext_i32_i64(ltid_y_22535)] = pre_22636;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float mem_23497[1];
        int32_t ltid_x_22579 = ltid_pre_24257;
        int32_t ltid_y_22580 = ltid_pre_24258;
        int32_t ltid_flat_22581 = local_tid_24253;
        int32_t gtid_22641 = ltid_x_22579 + binop_x_22621;
        int32_t gtid_22643 = ltid_y_22580 + binop_x_22623;
        float acc_22646 = mem_param_23477[0];
        bool binop_x_22649 = slt32(gtid_22641, m_18055);
        bool binop_y_22650 = slt32(gtid_22643, k2p2zq_18071);
        bool cond_22651 = binop_x_22649 && binop_y_22650;
        float acc_22652;
        
        if (cond_22651) {
            float x_22653;
            float redout_23155 = acc_22646;
            
            for (int32_t i_23156 = 0; i_23156 < tile_sizze_22500; i_23156++) {
                float x_22658 = ((__local
                                  float *) mem_23491)[sext_i32_i64(ltid_x_22579) *
                                                      sext_i32_i64(tile_sizze_22500) +
                                                      sext_i32_i64(i_23156)];
                bool res_22659;
                
                res_22659 = futrts_isnan32(x_22658);
                
                float res_22660;
                
                if (res_22659) {
                    res_22660 = 0.0F;
                } else {
                    float x_22657 = ((__local
                                      float *) mem_23486)[sext_i32_i64(i_23156) *
                                                          sext_i32_i64(tile_sizze_22500) +
                                                          sext_i32_i64(ltid_y_22580)];
                    float res_22661 = x_22657 * x_22658;
                    
                    res_22660 = res_22661;
                }
                
                float res_22656 = res_22660 + redout_23155;
                float redout_tmp_24261 = res_22656;
                
                redout_23155 = redout_tmp_24261;
            }
            x_22653 = redout_23155;
            acc_22652 = x_22653;
        } else {
            acc_22652 = acc_22646;
        }
        mem_23497[0] = acc_22652;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float mem_param_tmp_24259[1];
        
        for (int32_t i_3 = 0; i_3 < 1; i_3++)
            mem_param_tmp_24259[i_3] = mem_23497[i_3];
        for (int32_t i_4 = 0; i_4 < 1; i_4++)
            mem_param_23477[i_4] = mem_param_tmp_24259[i_4];
    }
    for (int32_t i_5 = 0; i_5 < 1; i_5++)
        accs_mem_23508[i_5] = mem_param_23477[i_5];
    
    __local char *mem_23514;
    
    mem_23514 = (__local char *) mem_23514_backing_6;
    
    __local char *mem_23519;
    
    mem_23519 = (__local char *) mem_23519_backing_7;
    
    float mem_23525[1];
    float mem_23915[1];
    
    if (cond_22672) {
        mem_23915[0] = accs_mem_23508[0];
    } else {
        int32_t binop_x_22758 = tile_sizze_22500 * num_whole_tiles_22521;
        int32_t ltid_x_22673 = ltid_pre_24257;
        int32_t ltid_y_22674 = ltid_pre_24258;
        int32_t ltid_flat_22675 = local_tid_24253;
        int32_t i_22759 = ltid_x_22673 + binop_x_22758;
        int32_t j_22761 = ltid_y_22674 + binop_x_22758;
        int32_t gtid_22763 = binop_x_22621 + ltid_x_22673;
        int32_t gtid_22765 = binop_x_22623 + ltid_y_22674;
        bool binop_x_22768 = slt32(i_22759, n_18059);
        bool binop_y_22769 = slt32(gtid_22765, k2p2zq_18071);
        bool cond_22770 = binop_x_22768 && binop_y_22769;
        float pre_22771;
        
        if (cond_22770) {
            float x_22772 = ((__global
                              float *) mem_23207)[sext_i32_i64(i_22759) *
                                                  sext_i32_i64(k2p2zq_18071) +
                                                  sext_i32_i64(gtid_22765)];
            
            pre_22771 = x_22772;
        } else {
            pre_22771 = 0.0F;
        }
        
        bool binop_x_22774 = slt32(j_22761, n_18059);
        bool binop_y_22775 = slt32(gtid_22763, m_18055);
        bool cond_22776 = binop_x_22774 && binop_y_22775;
        float pre_22777;
        
        if (cond_22776) {
            float x_22778 = ((__global
                              float *) images_mem_23189)[sext_i32_i64(gtid_22763) *
                                                         sext_i32_i64(N_18056) +
                                                         sext_i32_i64(j_22761)];
            
            pre_22777 = x_22778;
        } else {
            pre_22777 = 0.0F;
        }
        ((__local float *) mem_23514)[sext_i32_i64(ltid_x_22673) *
                                      sext_i32_i64(tile_sizze_22500) +
                                      sext_i32_i64(ltid_y_22674)] = pre_22771;
        ((__local float *) mem_23519)[sext_i32_i64(ltid_x_22673) *
                                      sext_i32_i64(tile_sizze_22500) +
                                      sext_i32_i64(ltid_y_22674)] = pre_22777;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_22720 = ltid_pre_24257;
        int32_t ltid_y_22721 = ltid_pre_24258;
        int32_t ltid_flat_22722 = local_tid_24253;
        int32_t gtid_22784 = binop_x_22621 + ltid_x_22720;
        int32_t gtid_22786 = binop_x_22623 + ltid_y_22721;
        float acc_22789 = accs_mem_23508[0];
        bool binop_x_22792 = slt32(gtid_22784, m_18055);
        bool binop_y_22793 = slt32(gtid_22786, k2p2zq_18071);
        bool cond_22794 = binop_x_22792 && binop_y_22793;
        float acc_22795;
        
        if (cond_22794) {
            float x_22796;
            float redout_23157 = acc_22789;
            
            for (int32_t i_23158 = 0; i_23158 < residual_input_22671;
                 i_23158++) {
                float x_22801 = ((__local
                                  float *) mem_23519)[sext_i32_i64(ltid_x_22720) *
                                                      sext_i32_i64(tile_sizze_22500) +
                                                      sext_i32_i64(i_23158)];
                bool res_22802;
                
                res_22802 = futrts_isnan32(x_22801);
                
                float res_22803;
                
                if (res_22802) {
                    res_22803 = 0.0F;
                } else {
                    float x_22800 = ((__local
                                      float *) mem_23514)[sext_i32_i64(i_23158) *
                                                          sext_i32_i64(tile_sizze_22500) +
                                                          sext_i32_i64(ltid_y_22721)];
                    float res_22804 = x_22800 * x_22801;
                    
                    res_22803 = res_22804;
                }
                
                float res_22799 = res_22803 + redout_23157;
                float redout_tmp_24262 = res_22799;
                
                redout_23157 = redout_tmp_24262;
            }
            x_22796 = redout_23157;
            acc_22795 = x_22796;
        } else {
            acc_22795 = acc_22789;
        }
        mem_23525[0] = acc_22795;
        barrier(CLK_LOCAL_MEM_FENCE);
        mem_23915[0] = mem_23525[0];
    }
    
    int32_t thread_out_index_24263 = gid_x_22497 * tile_sizze_22500 +
            ltid_pre_24257;
    int32_t thread_out_index_24264 = gid_y_22498 * tile_sizze_22500 +
            ltid_pre_24258;
    
    if (slt32(thread_out_index_24263, m_18055) && slt32(thread_out_index_24264,
                                                        k2p2zq_18071)) {
        ((__global float *) mem_23538)[sext_i32_i64(thread_out_index_24263) *
                                       sext_i32_i64(k2p2zq_18071) +
                                       sext_i32_i64(thread_out_index_24264)] =
            mem_23915[0];
    }
    
  error_5:
    return;
    #undef tile_sizze_22500
}
__kernel void mainzisegmap_intragroup_22827(__global int *global_failure,
                                            __local volatile
                                            int64_t *mem_23700_backing_aligned_0,
                                            __local volatile
                                            int64_t *mem_23695_backing_aligned_1,
                                            __local volatile
                                            int64_t *mem_23672_backing_aligned_2,
                                            __local volatile
                                            int64_t *mem_23667_backing_aligned_3,
                                            int32_t N_18054, int32_t m_18055,
                                            int32_t k2p2zq_18071,
                                            int32_t num_groups_y_22825,
                                            int32_t num_whole_tiles_22843,
                                            int32_t residual_input_22987,
                                            unsigned char cond_22988, __global
                                            unsigned char *res_mem_23612,
                                            __global unsigned char *mem_23643,
                                            __global unsigned char *mem_23719)
{
    #define tile_sizze_22822 (mainzitile_sizze_22821)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_23700_backing_7 = (__local volatile
                                                           char *) mem_23700_backing_aligned_0;
    __local volatile char *restrict mem_23695_backing_6 = (__local volatile
                                                           char *) mem_23695_backing_aligned_1;
    __local volatile char *restrict mem_23672_backing_1 = (__local volatile
                                                           char *) mem_23672_backing_aligned_2;
    __local volatile char *restrict mem_23667_backing_0 = (__local volatile
                                                           char *) mem_23667_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24419;
    int32_t local_tid_24420;
    int32_t group_sizze_24423;
    int32_t wave_sizze_24422;
    int32_t group_tid_24421;
    
    global_tid_24419 = get_global_id(0);
    local_tid_24420 = get_local_id(0);
    group_sizze_24423 = get_local_size(0);
    wave_sizze_24422 = LOCKSTEP_WIDTH;
    group_tid_24421 = get_group_id(0);
    
    int32_t gid_flat_22827;
    
    gid_flat_22827 = group_tid_24421;
    
    int32_t ltid_pre_24424;
    
    ltid_pre_24424 = squot32(local_tid_24420, tile_sizze_22822);
    
    int32_t ltid_pre_24425;
    
    ltid_pre_24425 = local_tid_24420 - squot32(local_tid_24420,
                                               tile_sizze_22822) *
        tile_sizze_22822;
    
    int32_t gid_x_22819;
    
    gid_x_22819 = squot32(group_tid_24421, num_groups_y_22825);
    
    int32_t gid_y_22820;
    
    gid_y_22820 = group_tid_24421 - squot32(group_tid_24421,
                                            num_groups_y_22825) *
        num_groups_y_22825;
    
    float mem_23650[1];
    int32_t ltid_x_22844 = ltid_pre_24424;
    int32_t ltid_y_22845 = ltid_pre_24425;
    int32_t ltid_flat_22846 = local_tid_24420;
    
    if (slt32(ltid_x_22844, tile_sizze_22822) && slt32(ltid_y_22845,
                                                       tile_sizze_22822)) {
        mem_23650[0] = 0.0F;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_22941 = gid_x_22819 * tile_sizze_22822;
    int32_t binop_x_22943 = gid_y_22820 * tile_sizze_22822;
    __local char *mem_23667;
    
    mem_23667 = (__local char *) mem_23667_backing_0;
    
    __local char *mem_23672;
    
    mem_23672 = (__local char *) mem_23672_backing_1;
    
    float accs_mem_23689[1];
    float mem_param_23658[1];
    
    for (int32_t i_2 = 0; i_2 < 1; i_2++)
        mem_param_23658[i_2] = mem_23650[i_2];
    for (int32_t tile_id_22855 = 0; tile_id_22855 < num_whole_tiles_22843;
         tile_id_22855++) {
        int32_t binop_x_22937 = tile_sizze_22822 * tile_id_22855;
        int32_t ltid_x_22856 = ltid_pre_24424;
        int32_t ltid_y_22857 = ltid_pre_24425;
        int32_t ltid_flat_22858 = local_tid_24420;
        int32_t i_22938 = ltid_x_22856 + binop_x_22937;
        int32_t j_22940 = ltid_y_22857 + binop_x_22937;
        int32_t gtid_22942 = ltid_x_22856 + binop_x_22941;
        int32_t gtid_22944 = ltid_y_22857 + binop_x_22943;
        bool binop_x_22947 = slt32(j_22940, k2p2zq_18071);
        bool binop_y_22948 = slt32(gtid_22942, m_18055);
        bool cond_22949 = binop_x_22947 && binop_y_22948;
        float pre_22950;
        
        if (cond_22949) {
            float x_22951 = ((__global
                              float *) res_mem_23612)[sext_i32_i64(gtid_22942) *
                                                      sext_i32_i64(k2p2zq_18071) +
                                                      sext_i32_i64(j_22940)];
            
            pre_22950 = x_22951;
        } else {
            pre_22950 = 0.0F;
        }
        
        bool binop_x_22953 = slt32(i_22938, k2p2zq_18071);
        bool binop_y_22954 = slt32(gtid_22944, N_18054);
        bool cond_22955 = binop_x_22953 && binop_y_22954;
        float pre_22956;
        
        if (cond_22955) {
            float x_22957 = ((__global
                              float *) mem_23643)[sext_i32_i64(i_22938) *
                                                  sext_i32_i64(N_18054) +
                                                  sext_i32_i64(gtid_22944)];
            
            pre_22956 = x_22957;
        } else {
            pre_22956 = 0.0F;
        }
        ((__local float *) mem_23667)[sext_i32_i64(ltid_x_22856) *
                                      sext_i32_i64(tile_sizze_22822) +
                                      sext_i32_i64(ltid_y_22857)] = pre_22950;
        ((__local float *) mem_23672)[sext_i32_i64(ltid_x_22856) *
                                      sext_i32_i64(tile_sizze_22822) +
                                      sext_i32_i64(ltid_y_22857)] = pre_22956;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float mem_23678[1];
        int32_t ltid_x_22901 = ltid_pre_24424;
        int32_t ltid_y_22902 = ltid_pre_24425;
        int32_t ltid_flat_22903 = local_tid_24420;
        int32_t gtid_22961 = ltid_x_22901 + binop_x_22941;
        int32_t gtid_22963 = ltid_y_22902 + binop_x_22943;
        float acc_22966 = mem_param_23658[0];
        bool binop_x_22969 = slt32(gtid_22961, m_18055);
        bool binop_y_22970 = slt32(gtid_22963, N_18054);
        bool cond_22971 = binop_x_22969 && binop_y_22970;
        float acc_22972;
        
        if (cond_22971) {
            float x_22973;
            float redout_23173 = acc_22966;
            
            for (int32_t i_23174 = 0; i_23174 < tile_sizze_22822; i_23174++) {
                float x_22977 = ((__local
                                  float *) mem_23667)[sext_i32_i64(ltid_x_22901) *
                                                      sext_i32_i64(tile_sizze_22822) +
                                                      sext_i32_i64(i_23174)];
                float x_22978 = ((__local
                                  float *) mem_23672)[sext_i32_i64(i_23174) *
                                                      sext_i32_i64(tile_sizze_22822) +
                                                      sext_i32_i64(ltid_y_22902)];
                float res_22979 = x_22977 * x_22978;
                float res_22976 = res_22979 + redout_23173;
                float redout_tmp_24428 = res_22976;
                
                redout_23173 = redout_tmp_24428;
            }
            x_22973 = redout_23173;
            acc_22972 = x_22973;
        } else {
            acc_22972 = acc_22966;
        }
        mem_23678[0] = acc_22972;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float mem_param_tmp_24426[1];
        
        for (int32_t i_3 = 0; i_3 < 1; i_3++)
            mem_param_tmp_24426[i_3] = mem_23678[i_3];
        for (int32_t i_4 = 0; i_4 < 1; i_4++)
            mem_param_23658[i_4] = mem_param_tmp_24426[i_4];
    }
    for (int32_t i_5 = 0; i_5 < 1; i_5++)
        accs_mem_23689[i_5] = mem_param_23658[i_5];
    
    __local char *mem_23695;
    
    mem_23695 = (__local char *) mem_23695_backing_6;
    
    __local char *mem_23700;
    
    mem_23700 = (__local char *) mem_23700_backing_7;
    
    float mem_23706[1];
    float mem_23929[1];
    
    if (cond_22988) {
        mem_23929[0] = accs_mem_23689[0];
    } else {
        int32_t binop_x_23072 = tile_sizze_22822 * num_whole_tiles_22843;
        int32_t ltid_x_22989 = ltid_pre_24424;
        int32_t ltid_y_22990 = ltid_pre_24425;
        int32_t ltid_flat_22991 = local_tid_24420;
        int32_t i_23073 = ltid_x_22989 + binop_x_23072;
        int32_t j_23075 = ltid_y_22990 + binop_x_23072;
        int32_t gtid_23077 = binop_x_22941 + ltid_x_22989;
        int32_t gtid_23079 = binop_x_22943 + ltid_y_22990;
        bool binop_x_23082 = slt32(j_23075, k2p2zq_18071);
        bool binop_y_23083 = slt32(gtid_23077, m_18055);
        bool cond_23084 = binop_x_23082 && binop_y_23083;
        float pre_23085;
        
        if (cond_23084) {
            float x_23086 = ((__global
                              float *) res_mem_23612)[sext_i32_i64(gtid_23077) *
                                                      sext_i32_i64(k2p2zq_18071) +
                                                      sext_i32_i64(j_23075)];
            
            pre_23085 = x_23086;
        } else {
            pre_23085 = 0.0F;
        }
        
        bool binop_x_23088 = slt32(i_23073, k2p2zq_18071);
        bool binop_y_23089 = slt32(gtid_23079, N_18054);
        bool cond_23090 = binop_x_23088 && binop_y_23089;
        float pre_23091;
        
        if (cond_23090) {
            float x_23092 = ((__global
                              float *) mem_23643)[sext_i32_i64(i_23073) *
                                                  sext_i32_i64(N_18054) +
                                                  sext_i32_i64(gtid_23079)];
            
            pre_23091 = x_23092;
        } else {
            pre_23091 = 0.0F;
        }
        ((__local float *) mem_23695)[sext_i32_i64(ltid_x_22989) *
                                      sext_i32_i64(tile_sizze_22822) +
                                      sext_i32_i64(ltid_y_22990)] = pre_23085;
        ((__local float *) mem_23700)[sext_i32_i64(ltid_x_22989) *
                                      sext_i32_i64(tile_sizze_22822) +
                                      sext_i32_i64(ltid_y_22990)] = pre_23091;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_23036 = ltid_pre_24424;
        int32_t ltid_y_23037 = ltid_pre_24425;
        int32_t ltid_flat_23038 = local_tid_24420;
        int32_t gtid_23098 = binop_x_22941 + ltid_x_23036;
        int32_t gtid_23100 = binop_x_22943 + ltid_y_23037;
        float acc_23103 = accs_mem_23689[0];
        bool binop_x_23106 = slt32(gtid_23098, m_18055);
        bool binop_y_23107 = slt32(gtid_23100, N_18054);
        bool cond_23108 = binop_x_23106 && binop_y_23107;
        float acc_23109;
        
        if (cond_23108) {
            float x_23110;
            float redout_23175 = acc_23103;
            
            for (int32_t i_23176 = 0; i_23176 < residual_input_22987;
                 i_23176++) {
                float x_23114 = ((__local
                                  float *) mem_23695)[sext_i32_i64(ltid_x_23036) *
                                                      sext_i32_i64(tile_sizze_22822) +
                                                      sext_i32_i64(i_23176)];
                float x_23115 = ((__local
                                  float *) mem_23700)[sext_i32_i64(i_23176) *
                                                      sext_i32_i64(tile_sizze_22822) +
                                                      sext_i32_i64(ltid_y_23037)];
                float res_23116 = x_23114 * x_23115;
                float res_23113 = res_23116 + redout_23175;
                float redout_tmp_24429 = res_23113;
                
                redout_23175 = redout_tmp_24429;
            }
            x_23110 = redout_23175;
            acc_23109 = x_23110;
        } else {
            acc_23109 = acc_23103;
        }
        mem_23706[0] = acc_23109;
        barrier(CLK_LOCAL_MEM_FENCE);
        mem_23929[0] = mem_23706[0];
    }
    
    int32_t thread_out_index_24430 = gid_x_22819 * tile_sizze_22822 +
            ltid_pre_24424;
    int32_t thread_out_index_24431 = gid_y_22820 * tile_sizze_22822 +
            ltid_pre_24425;
    
    if (slt32(thread_out_index_24430, m_18055) && slt32(thread_out_index_24431,
                                                        N_18054)) {
        ((__global float *) mem_23719)[sext_i32_i64(thread_out_index_24430) *
                                       sext_i32_i64(N_18054) +
                                       sext_i32_i64(thread_out_index_24431)] =
            mem_23929[0];
    }
    
  error_5:
    return;
    #undef tile_sizze_22822
}
__kernel void mainzisegred_large_19329(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_24148_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24146_backing_aligned_1,
                                       int32_t N_18054, int32_t N_18056,
                                       int32_t n_18059, int32_t k2p2zq_18071,
                                       int32_t num_groups_19492, __global
                                       unsigned char *images_mem_23189, __global
                                       unsigned char *binop_p_mem_23202,
                                       __global unsigned char *mem_23307,
                                       __global unsigned char *mem_23315,
                                       int32_t groups_per_segment_24132,
                                       int32_t elements_per_thread_24133,
                                       int32_t virt_num_groups_24134,
                                       int32_t threads_per_segment_24136,
                                       __global
                                       unsigned char *group_res_arr_mem_24137,
                                       __global
                                       unsigned char *mainzicounter_mem_24139)
{
    #define segred_group_sizze_19491 (mainzisegred_group_sizze_19323)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_24148_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_24148_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24146_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24146_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24141;
    int32_t local_tid_24142;
    int32_t group_sizze_24145;
    int32_t wave_sizze_24144;
    int32_t group_tid_24143;
    
    global_tid_24141 = get_global_id(0);
    local_tid_24142 = get_local_id(0);
    group_sizze_24145 = get_local_size(0);
    wave_sizze_24144 = LOCKSTEP_WIDTH;
    group_tid_24143 = get_group_id(0);
    
    int32_t phys_tid_19329;
    
    phys_tid_19329 = global_tid_24141;
    
    __local char *red_arr_mem_24146;
    
    red_arr_mem_24146 = (__local char *) red_arr_mem_24146_backing_0;
    
    __local char *sync_arr_mem_24148;
    
    sync_arr_mem_24148 = (__local char *) sync_arr_mem_24148_backing_1;
    
    int32_t phys_group_id_24150;
    
    phys_group_id_24150 = get_group_id(0);
    for (int32_t i_24151 = 0; i_24151 < sdiv_up32(virt_num_groups_24134 -
                                                  phys_group_id_24150,
                                                  num_groups_19492);
         i_24151++) {
        int32_t virt_group_id_24152 = phys_group_id_24150 + i_24151 *
                num_groups_19492;
        int32_t flat_segment_id_24153 = squot32(virt_group_id_24152,
                                                groups_per_segment_24132);
        int32_t global_tid_24154 = srem32(virt_group_id_24152 *
                                          segred_group_sizze_19491 +
                                          local_tid_24142,
                                          segred_group_sizze_19491 *
                                          groups_per_segment_24132);
        int32_t gtid_19312 = squot32(flat_segment_id_24153, k2p2zq_18071 *
                                     k2p2zq_18071);
        int32_t gtid_19313 = squot32(flat_segment_id_24153 -
                                     squot32(flat_segment_id_24153,
                                             k2p2zq_18071 * k2p2zq_18071) *
                                     (k2p2zq_18071 * k2p2zq_18071),
                                     k2p2zq_18071);
        int32_t gtid_19314 = flat_segment_id_24153 -
                squot32(flat_segment_id_24153, k2p2zq_18071 * k2p2zq_18071) *
                (k2p2zq_18071 * k2p2zq_18071) - squot32(flat_segment_id_24153 -
                                                        squot32(flat_segment_id_24153,
                                                                k2p2zq_18071 *
                                                                k2p2zq_18071) *
                                                        (k2p2zq_18071 *
                                                         k2p2zq_18071),
                                                        k2p2zq_18071) *
                k2p2zq_18071;
        int32_t gtid_19328;
        float x_acc_24155;
        int32_t chunk_sizze_24156;
        
        chunk_sizze_24156 = smin32(elements_per_thread_24133,
                                   sdiv_up32(n_18059 - global_tid_24154,
                                             threads_per_segment_24136));
        
        float x_19495;
        float x_19496;
        
        // neutral-initialise the accumulators
        {
            x_acc_24155 = 0.0F;
        }
        for (int32_t i_24160 = 0; i_24160 < chunk_sizze_24156; i_24160++) {
            gtid_19328 = global_tid_24154 + threads_per_segment_24136 * i_24160;
            // apply map function
            {
                float x_19501 = ((__global
                                  float *) images_mem_23189)[sext_i32_i64(gtid_19312) *
                                                             sext_i32_i64(N_18056) +
                                                             sext_i32_i64(gtid_19328)];
                float x_19502 = ((__global
                                  float *) binop_p_mem_23202)[sext_i32_i64(gtid_19313) *
                                                              sext_i32_i64(N_18054) +
                                                              sext_i32_i64(gtid_19328)];
                float x_19503 = ((__global
                                  float *) mem_23307)[sext_i32_i64(gtid_19314) *
                                                      sext_i32_i64(N_18054) +
                                                      sext_i32_i64(gtid_19328)];
                float x_19504 = x_19502 * x_19503;
                bool res_19505;
                
                res_19505 = futrts_isnan32(x_19501);
                
                float y_19506;
                
                if (res_19505) {
                    y_19506 = 0.0F;
                } else {
                    y_19506 = 1.0F;
                }
                
                float res_19507 = x_19504 * y_19506;
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_19495 = x_acc_24155;
                }
                // load new values
                {
                    x_19496 = res_19507;
                }
                // apply reduction operator
                {
                    float res_19497 = x_19495 + x_19496;
                    
                    // store in accumulator
                    {
                        x_acc_24155 = res_19497;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_19495 = x_acc_24155;
            ((__local
              float *) red_arr_mem_24146)[sext_i32_i64(local_tid_24142)] =
                x_19495;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_24161;
        int32_t skip_waves_24162;
        float x_24157;
        float x_24158;
        
        offset_24161 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_24142, segred_group_sizze_19491)) {
                x_24157 = ((__local
                            float *) red_arr_mem_24146)[sext_i32_i64(local_tid_24142 +
                                                        offset_24161)];
            }
        }
        offset_24161 = 1;
        while (slt32(offset_24161, wave_sizze_24144)) {
            if (slt32(local_tid_24142 + offset_24161,
                      segred_group_sizze_19491) && ((local_tid_24142 -
                                                     squot32(local_tid_24142,
                                                             wave_sizze_24144) *
                                                     wave_sizze_24144) & (2 *
                                                                          offset_24161 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_24158 = ((volatile __local
                                float *) red_arr_mem_24146)[sext_i32_i64(local_tid_24142 +
                                                            offset_24161)];
                }
                // apply reduction operation
                {
                    float res_24159 = x_24157 + x_24158;
                    
                    x_24157 = res_24159;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_24146)[sext_i32_i64(local_tid_24142)] =
                        x_24157;
                }
            }
            offset_24161 *= 2;
        }
        skip_waves_24162 = 1;
        while (slt32(skip_waves_24162, squot32(segred_group_sizze_19491 +
                                               wave_sizze_24144 - 1,
                                               wave_sizze_24144))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_24161 = skip_waves_24162 * wave_sizze_24144;
            if (slt32(local_tid_24142 + offset_24161,
                      segred_group_sizze_19491) && ((local_tid_24142 -
                                                     squot32(local_tid_24142,
                                                             wave_sizze_24144) *
                                                     wave_sizze_24144) == 0 &&
                                                    (squot32(local_tid_24142,
                                                             wave_sizze_24144) &
                                                     (2 * skip_waves_24162 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_24158 = ((__local
                                float *) red_arr_mem_24146)[sext_i32_i64(local_tid_24142 +
                                                            offset_24161)];
                }
                // apply reduction operation
                {
                    float res_24159 = x_24157 + x_24158;
                    
                    x_24157 = res_24159;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_24146)[sext_i32_i64(local_tid_24142)] =
                        x_24157;
                }
            }
            skip_waves_24162 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_24142 == 0) {
                x_acc_24155 = x_24157;
            }
        }
        if (groups_per_segment_24132 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_24142 == 0) {
                    ((__global float *) mem_23315)[sext_i32_i64(gtid_19312) *
                                                   sext_i32_i64(k2p2zq_18071 *
                                                   k2p2zq_18071) +
                                                   sext_i32_i64(gtid_19313) *
                                                   sext_i32_i64(k2p2zq_18071) +
                                                   sext_i32_i64(gtid_19314)] =
                        x_acc_24155;
                }
            }
        } else {
            int32_t old_counter_24163;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_24142 == 0) {
                    ((__global
                      float *) group_res_arr_mem_24137)[sext_i32_i64(virt_group_id_24152) *
                                                        sext_i32_i64(segred_group_sizze_19491)] =
                        x_acc_24155;
                    mem_fence_global();
                    old_counter_24163 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24139)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24153,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_24148)[0] =
                        old_counter_24163 == groups_per_segment_24132 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_24164;
            
            is_last_group_24164 = ((__local bool *) sync_arr_mem_24148)[0];
            if (is_last_group_24164) {
                if (local_tid_24142 == 0) {
                    old_counter_24163 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24139)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24153,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_24132));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_24165 =
                            sdiv_up32(groups_per_segment_24132,
                                      segred_group_sizze_19491);
                    
                    x_19495 = 0.0F;
                    for (int32_t i_24166 = 0; i_24166 < read_per_thread_24165;
                         i_24166++) {
                        int32_t group_res_id_24167 = local_tid_24142 *
                                read_per_thread_24165 + i_24166;
                        int32_t index_of_group_res_24168 =
                                flat_segment_id_24153 *
                                groups_per_segment_24132 + group_res_id_24167;
                        
                        if (slt32(group_res_id_24167,
                                  groups_per_segment_24132)) {
                            x_19496 = ((__global
                                        float *) group_res_arr_mem_24137)[sext_i32_i64(index_of_group_res_24168) *
                                                                          sext_i32_i64(segred_group_sizze_19491)];
                            
                            float res_19497;
                            
                            res_19497 = x_19495 + x_19496;
                            x_19495 = res_19497;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_24146)[sext_i32_i64(local_tid_24142)] =
                    x_19495;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_24169;
                    int32_t skip_waves_24170;
                    float x_24157;
                    float x_24158;
                    
                    offset_24169 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_24142, segred_group_sizze_19491)) {
                            x_24157 = ((__local
                                        float *) red_arr_mem_24146)[sext_i32_i64(local_tid_24142 +
                                                                    offset_24169)];
                        }
                    }
                    offset_24169 = 1;
                    while (slt32(offset_24169, wave_sizze_24144)) {
                        if (slt32(local_tid_24142 + offset_24169,
                                  segred_group_sizze_19491) &&
                            ((local_tid_24142 - squot32(local_tid_24142,
                                                        wave_sizze_24144) *
                              wave_sizze_24144) & (2 * offset_24169 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_24158 = ((volatile __local
                                            float *) red_arr_mem_24146)[sext_i32_i64(local_tid_24142 +
                                                                        offset_24169)];
                            }
                            // apply reduction operation
                            {
                                float res_24159 = x_24157 + x_24158;
                                
                                x_24157 = res_24159;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24146)[sext_i32_i64(local_tid_24142)] =
                                    x_24157;
                            }
                        }
                        offset_24169 *= 2;
                    }
                    skip_waves_24170 = 1;
                    while (slt32(skip_waves_24170,
                                 squot32(segred_group_sizze_19491 +
                                         wave_sizze_24144 - 1,
                                         wave_sizze_24144))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_24169 = skip_waves_24170 * wave_sizze_24144;
                        if (slt32(local_tid_24142 + offset_24169,
                                  segred_group_sizze_19491) &&
                            ((local_tid_24142 - squot32(local_tid_24142,
                                                        wave_sizze_24144) *
                              wave_sizze_24144) == 0 &&
                             (squot32(local_tid_24142, wave_sizze_24144) & (2 *
                                                                            skip_waves_24170 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_24158 = ((__local
                                            float *) red_arr_mem_24146)[sext_i32_i64(local_tid_24142 +
                                                                        offset_24169)];
                            }
                            // apply reduction operation
                            {
                                float res_24159 = x_24157 + x_24158;
                                
                                x_24157 = res_24159;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_24146)[sext_i32_i64(local_tid_24142)] =
                                    x_24157;
                            }
                        }
                        skip_waves_24170 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_24142 == 0) {
                            ((__global
                              float *) mem_23315)[sext_i32_i64(gtid_19312) *
                                                  sext_i32_i64(k2p2zq_18071 *
                                                  k2p2zq_18071) +
                                                  sext_i32_i64(gtid_19313) *
                                                  sext_i32_i64(k2p2zq_18071) +
                                                  sext_i32_i64(gtid_19314)] =
                                x_24157;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_19491
}
__kernel void mainzisegred_large_20577(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_24301_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24299_backing_aligned_1,
                                       int32_t N_18054, int32_t N_18056,
                                       int32_t n_18059, int32_t k2p2zq_18071,
                                       int32_t num_groups_20640, __global
                                       unsigned char *images_mem_23189, __global
                                       unsigned char *binop_p_mem_23202,
                                       __global unsigned char *mem_23544,
                                       int32_t groups_per_segment_24285,
                                       int32_t elements_per_thread_24286,
                                       int32_t virt_num_groups_24287,
                                       int32_t threads_per_segment_24289,
                                       __global
                                       unsigned char *group_res_arr_mem_24290,
                                       __global
                                       unsigned char *mainzicounter_mem_24292)
{
    #define segred_group_sizze_20639 (mainzisegred_group_sizze_20571)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_24301_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_24301_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24299_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24299_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24294;
    int32_t local_tid_24295;
    int32_t group_sizze_24298;
    int32_t wave_sizze_24297;
    int32_t group_tid_24296;
    
    global_tid_24294 = get_global_id(0);
    local_tid_24295 = get_local_id(0);
    group_sizze_24298 = get_local_size(0);
    wave_sizze_24297 = LOCKSTEP_WIDTH;
    group_tid_24296 = get_group_id(0);
    
    int32_t phys_tid_20577;
    
    phys_tid_20577 = global_tid_24294;
    
    __local char *red_arr_mem_24299;
    
    red_arr_mem_24299 = (__local char *) red_arr_mem_24299_backing_0;
    
    __local char *sync_arr_mem_24301;
    
    sync_arr_mem_24301 = (__local char *) sync_arr_mem_24301_backing_1;
    
    int32_t phys_group_id_24303;
    
    phys_group_id_24303 = get_group_id(0);
    for (int32_t i_24304 = 0; i_24304 < sdiv_up32(virt_num_groups_24287 -
                                                  phys_group_id_24303,
                                                  num_groups_20640);
         i_24304++) {
        int32_t virt_group_id_24305 = phys_group_id_24303 + i_24304 *
                num_groups_20640;
        int32_t flat_segment_id_24306 = squot32(virt_group_id_24305,
                                                groups_per_segment_24285);
        int32_t global_tid_24307 = srem32(virt_group_id_24305 *
                                          segred_group_sizze_20639 +
                                          local_tid_24295,
                                          segred_group_sizze_20639 *
                                          groups_per_segment_24285);
        int32_t gtid_20563 = squot32(flat_segment_id_24306, k2p2zq_18071);
        int32_t gtid_20564 = flat_segment_id_24306 -
                squot32(flat_segment_id_24306, k2p2zq_18071) * k2p2zq_18071;
        int32_t gtid_20576;
        float x_acc_24308;
        int32_t chunk_sizze_24309;
        
        chunk_sizze_24309 = smin32(elements_per_thread_24286,
                                   sdiv_up32(n_18059 - global_tid_24307,
                                             threads_per_segment_24289));
        
        float x_20643;
        float x_20644;
        
        // neutral-initialise the accumulators
        {
            x_acc_24308 = 0.0F;
        }
        for (int32_t i_24313 = 0; i_24313 < chunk_sizze_24309; i_24313++) {
            gtid_20576 = global_tid_24307 + threads_per_segment_24289 * i_24313;
            // apply map function
            {
                float x_20649 = ((__global
                                  float *) images_mem_23189)[sext_i32_i64(gtid_20563) *
                                                             sext_i32_i64(N_18056) +
                                                             sext_i32_i64(gtid_20576)];
                bool res_20650;
                
                res_20650 = futrts_isnan32(x_20649);
                
                float res_20651;
                
                if (res_20650) {
                    res_20651 = 0.0F;
                } else {
                    float x_20648 = ((__global
                                      float *) binop_p_mem_23202)[sext_i32_i64(gtid_20564) *
                                                                  sext_i32_i64(N_18054) +
                                                                  sext_i32_i64(gtid_20576)];
                    float res_20652 = x_20648 * x_20649;
                    
                    res_20651 = res_20652;
                }
                // save map-out results
                { }
                // load accumulator
                {
                    x_20643 = x_acc_24308;
                }
                // load new values
                {
                    x_20644 = res_20651;
                }
                // apply reduction operator
                {
                    float res_20645 = x_20643 + x_20644;
                    
                    // store in accumulator
                    {
                        x_acc_24308 = res_20645;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_20643 = x_acc_24308;
            ((__local
              float *) red_arr_mem_24299)[sext_i32_i64(local_tid_24295)] =
                x_20643;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_24314;
        int32_t skip_waves_24315;
        float x_24310;
        float x_24311;
        
        offset_24314 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_24295, segred_group_sizze_20639)) {
                x_24310 = ((__local
                            float *) red_arr_mem_24299)[sext_i32_i64(local_tid_24295 +
                                                        offset_24314)];
            }
        }
        offset_24314 = 1;
        while (slt32(offset_24314, wave_sizze_24297)) {
            if (slt32(local_tid_24295 + offset_24314,
                      segred_group_sizze_20639) && ((local_tid_24295 -
                                                     squot32(local_tid_24295,
                                                             wave_sizze_24297) *
                                                     wave_sizze_24297) & (2 *
                                                                          offset_24314 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_24311 = ((volatile __local
                                float *) red_arr_mem_24299)[sext_i32_i64(local_tid_24295 +
                                                            offset_24314)];
                }
                // apply reduction operation
                {
                    float res_24312 = x_24310 + x_24311;
                    
                    x_24310 = res_24312;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_24299)[sext_i32_i64(local_tid_24295)] =
                        x_24310;
                }
            }
            offset_24314 *= 2;
        }
        skip_waves_24315 = 1;
        while (slt32(skip_waves_24315, squot32(segred_group_sizze_20639 +
                                               wave_sizze_24297 - 1,
                                               wave_sizze_24297))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_24314 = skip_waves_24315 * wave_sizze_24297;
            if (slt32(local_tid_24295 + offset_24314,
                      segred_group_sizze_20639) && ((local_tid_24295 -
                                                     squot32(local_tid_24295,
                                                             wave_sizze_24297) *
                                                     wave_sizze_24297) == 0 &&
                                                    (squot32(local_tid_24295,
                                                             wave_sizze_24297) &
                                                     (2 * skip_waves_24315 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_24311 = ((__local
                                float *) red_arr_mem_24299)[sext_i32_i64(local_tid_24295 +
                                                            offset_24314)];
                }
                // apply reduction operation
                {
                    float res_24312 = x_24310 + x_24311;
                    
                    x_24310 = res_24312;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_24299)[sext_i32_i64(local_tid_24295)] =
                        x_24310;
                }
            }
            skip_waves_24315 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_24295 == 0) {
                x_acc_24308 = x_24310;
            }
        }
        if (groups_per_segment_24285 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_24295 == 0) {
                    ((__global float *) mem_23544)[sext_i32_i64(gtid_20563) *
                                                   sext_i32_i64(k2p2zq_18071) +
                                                   sext_i32_i64(gtid_20564)] =
                        x_acc_24308;
                }
            }
        } else {
            int32_t old_counter_24316;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_24295 == 0) {
                    ((__global
                      float *) group_res_arr_mem_24290)[sext_i32_i64(virt_group_id_24305) *
                                                        sext_i32_i64(segred_group_sizze_20639)] =
                        x_acc_24308;
                    mem_fence_global();
                    old_counter_24316 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24292)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24306,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_24301)[0] =
                        old_counter_24316 == groups_per_segment_24285 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_24317;
            
            is_last_group_24317 = ((__local bool *) sync_arr_mem_24301)[0];
            if (is_last_group_24317) {
                if (local_tid_24295 == 0) {
                    old_counter_24316 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24292)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24306,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_24285));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_24318 =
                            sdiv_up32(groups_per_segment_24285,
                                      segred_group_sizze_20639);
                    
                    x_20643 = 0.0F;
                    for (int32_t i_24319 = 0; i_24319 < read_per_thread_24318;
                         i_24319++) {
                        int32_t group_res_id_24320 = local_tid_24295 *
                                read_per_thread_24318 + i_24319;
                        int32_t index_of_group_res_24321 =
                                flat_segment_id_24306 *
                                groups_per_segment_24285 + group_res_id_24320;
                        
                        if (slt32(group_res_id_24320,
                                  groups_per_segment_24285)) {
                            x_20644 = ((__global
                                        float *) group_res_arr_mem_24290)[sext_i32_i64(index_of_group_res_24321) *
                                                                          sext_i32_i64(segred_group_sizze_20639)];
                            
                            float res_20645;
                            
                            res_20645 = x_20643 + x_20644;
                            x_20643 = res_20645;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_24299)[sext_i32_i64(local_tid_24295)] =
                    x_20643;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_24322;
                    int32_t skip_waves_24323;
                    float x_24310;
                    float x_24311;
                    
                    offset_24322 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_24295, segred_group_sizze_20639)) {
                            x_24310 = ((__local
                                        float *) red_arr_mem_24299)[sext_i32_i64(local_tid_24295 +
                                                                    offset_24322)];
                        }
                    }
                    offset_24322 = 1;
                    while (slt32(offset_24322, wave_sizze_24297)) {
                        if (slt32(local_tid_24295 + offset_24322,
                                  segred_group_sizze_20639) &&
                            ((local_tid_24295 - squot32(local_tid_24295,
                                                        wave_sizze_24297) *
                              wave_sizze_24297) & (2 * offset_24322 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_24311 = ((volatile __local
                                            float *) red_arr_mem_24299)[sext_i32_i64(local_tid_24295 +
                                                                        offset_24322)];
                            }
                            // apply reduction operation
                            {
                                float res_24312 = x_24310 + x_24311;
                                
                                x_24310 = res_24312;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24299)[sext_i32_i64(local_tid_24295)] =
                                    x_24310;
                            }
                        }
                        offset_24322 *= 2;
                    }
                    skip_waves_24323 = 1;
                    while (slt32(skip_waves_24323,
                                 squot32(segred_group_sizze_20639 +
                                         wave_sizze_24297 - 1,
                                         wave_sizze_24297))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_24322 = skip_waves_24323 * wave_sizze_24297;
                        if (slt32(local_tid_24295 + offset_24322,
                                  segred_group_sizze_20639) &&
                            ((local_tid_24295 - squot32(local_tid_24295,
                                                        wave_sizze_24297) *
                              wave_sizze_24297) == 0 &&
                             (squot32(local_tid_24295, wave_sizze_24297) & (2 *
                                                                            skip_waves_24323 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_24311 = ((__local
                                            float *) red_arr_mem_24299)[sext_i32_i64(local_tid_24295 +
                                                                        offset_24322)];
                            }
                            // apply reduction operation
                            {
                                float res_24312 = x_24310 + x_24311;
                                
                                x_24310 = res_24312;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_24299)[sext_i32_i64(local_tid_24295)] =
                                    x_24310;
                            }
                        }
                        skip_waves_24323 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_24295 == 0) {
                            ((__global
                              float *) mem_23544)[sext_i32_i64(gtid_20563) *
                                                  sext_i32_i64(k2p2zq_18071) +
                                                  sext_i32_i64(gtid_20564)] =
                                x_24310;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_20639
}
__kernel void mainzisegred_large_20733(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_24381_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24379_backing_aligned_1,
                                       int32_t k2p2zq_18071,
                                       int32_t num_groups_20792, __global
                                       unsigned char *res_mem_23436, __global
                                       unsigned char *res_mem_23552, __global
                                       unsigned char *mem_23604,
                                       int32_t groups_per_segment_24365,
                                       int32_t elements_per_thread_24366,
                                       int32_t virt_num_groups_24367,
                                       int32_t threads_per_segment_24369,
                                       __global
                                       unsigned char *group_res_arr_mem_24370,
                                       __global
                                       unsigned char *mainzicounter_mem_24372)
{
    #define segred_group_sizze_20791 (mainzisegred_group_sizze_20727)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_24381_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_24381_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24379_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24379_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24374;
    int32_t local_tid_24375;
    int32_t group_sizze_24378;
    int32_t wave_sizze_24377;
    int32_t group_tid_24376;
    
    global_tid_24374 = get_global_id(0);
    local_tid_24375 = get_local_id(0);
    group_sizze_24378 = get_local_size(0);
    wave_sizze_24377 = LOCKSTEP_WIDTH;
    group_tid_24376 = get_group_id(0);
    
    int32_t phys_tid_20733;
    
    phys_tid_20733 = global_tid_24374;
    
    __local char *red_arr_mem_24379;
    
    red_arr_mem_24379 = (__local char *) red_arr_mem_24379_backing_0;
    
    __local char *sync_arr_mem_24381;
    
    sync_arr_mem_24381 = (__local char *) sync_arr_mem_24381_backing_1;
    
    int32_t phys_group_id_24383;
    
    phys_group_id_24383 = get_group_id(0);
    for (int32_t i_24384 = 0; i_24384 < sdiv_up32(virt_num_groups_24367 -
                                                  phys_group_id_24383,
                                                  num_groups_20792);
         i_24384++) {
        int32_t virt_group_id_24385 = phys_group_id_24383 + i_24384 *
                num_groups_20792;
        int32_t flat_segment_id_24386 = squot32(virt_group_id_24385,
                                                groups_per_segment_24365);
        int32_t global_tid_24387 = srem32(virt_group_id_24385 *
                                          segred_group_sizze_20791 +
                                          local_tid_24375,
                                          segred_group_sizze_20791 *
                                          groups_per_segment_24365);
        int32_t gtid_20719 = squot32(flat_segment_id_24386, k2p2zq_18071);
        int32_t gtid_20720 = flat_segment_id_24386 -
                squot32(flat_segment_id_24386, k2p2zq_18071) * k2p2zq_18071;
        int32_t gtid_20732;
        float x_acc_24388;
        int32_t chunk_sizze_24389;
        
        chunk_sizze_24389 = smin32(elements_per_thread_24366,
                                   sdiv_up32(k2p2zq_18071 - global_tid_24387,
                                             threads_per_segment_24369));
        
        float x_20795;
        float x_20796;
        
        // neutral-initialise the accumulators
        {
            x_acc_24388 = 0.0F;
        }
        for (int32_t i_24393 = 0; i_24393 < chunk_sizze_24389; i_24393++) {
            gtid_20732 = global_tid_24387 + threads_per_segment_24369 * i_24393;
            // apply map function
            {
                float x_20801 = ((__global
                                  float *) res_mem_23552)[sext_i32_i64(gtid_20719) *
                                                          sext_i32_i64(k2p2zq_18071) +
                                                          sext_i32_i64(gtid_20732)];
                float x_20802 = ((__global
                                  float *) res_mem_23436)[sext_i32_i64(gtid_20719) *
                                                          sext_i32_i64(k2p2zq_18071 *
                                                          k2p2zq_18071) +
                                                          sext_i32_i64(gtid_20720) *
                                                          sext_i32_i64(k2p2zq_18071) +
                                                          sext_i32_i64(gtid_20732)];
                float res_20803 = x_20801 * x_20802;
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_20795 = x_acc_24388;
                }
                // load new values
                {
                    x_20796 = res_20803;
                }
                // apply reduction operator
                {
                    float res_20797 = x_20795 + x_20796;
                    
                    // store in accumulator
                    {
                        x_acc_24388 = res_20797;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_20795 = x_acc_24388;
            ((__local
              float *) red_arr_mem_24379)[sext_i32_i64(local_tid_24375)] =
                x_20795;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_24394;
        int32_t skip_waves_24395;
        float x_24390;
        float x_24391;
        
        offset_24394 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_24375, segred_group_sizze_20791)) {
                x_24390 = ((__local
                            float *) red_arr_mem_24379)[sext_i32_i64(local_tid_24375 +
                                                        offset_24394)];
            }
        }
        offset_24394 = 1;
        while (slt32(offset_24394, wave_sizze_24377)) {
            if (slt32(local_tid_24375 + offset_24394,
                      segred_group_sizze_20791) && ((local_tid_24375 -
                                                     squot32(local_tid_24375,
                                                             wave_sizze_24377) *
                                                     wave_sizze_24377) & (2 *
                                                                          offset_24394 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_24391 = ((volatile __local
                                float *) red_arr_mem_24379)[sext_i32_i64(local_tid_24375 +
                                                            offset_24394)];
                }
                // apply reduction operation
                {
                    float res_24392 = x_24390 + x_24391;
                    
                    x_24390 = res_24392;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_24379)[sext_i32_i64(local_tid_24375)] =
                        x_24390;
                }
            }
            offset_24394 *= 2;
        }
        skip_waves_24395 = 1;
        while (slt32(skip_waves_24395, squot32(segred_group_sizze_20791 +
                                               wave_sizze_24377 - 1,
                                               wave_sizze_24377))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_24394 = skip_waves_24395 * wave_sizze_24377;
            if (slt32(local_tid_24375 + offset_24394,
                      segred_group_sizze_20791) && ((local_tid_24375 -
                                                     squot32(local_tid_24375,
                                                             wave_sizze_24377) *
                                                     wave_sizze_24377) == 0 &&
                                                    (squot32(local_tid_24375,
                                                             wave_sizze_24377) &
                                                     (2 * skip_waves_24395 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_24391 = ((__local
                                float *) red_arr_mem_24379)[sext_i32_i64(local_tid_24375 +
                                                            offset_24394)];
                }
                // apply reduction operation
                {
                    float res_24392 = x_24390 + x_24391;
                    
                    x_24390 = res_24392;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_24379)[sext_i32_i64(local_tid_24375)] =
                        x_24390;
                }
            }
            skip_waves_24395 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_24375 == 0) {
                x_acc_24388 = x_24390;
            }
        }
        if (groups_per_segment_24365 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_24375 == 0) {
                    ((__global float *) mem_23604)[sext_i32_i64(gtid_20719) *
                                                   sext_i32_i64(k2p2zq_18071) +
                                                   sext_i32_i64(gtid_20720)] =
                        x_acc_24388;
                }
            }
        } else {
            int32_t old_counter_24396;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_24375 == 0) {
                    ((__global
                      float *) group_res_arr_mem_24370)[sext_i32_i64(virt_group_id_24385) *
                                                        sext_i32_i64(segred_group_sizze_20791)] =
                        x_acc_24388;
                    mem_fence_global();
                    old_counter_24396 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24372)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24386,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_24381)[0] =
                        old_counter_24396 == groups_per_segment_24365 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_24397;
            
            is_last_group_24397 = ((__local bool *) sync_arr_mem_24381)[0];
            if (is_last_group_24397) {
                if (local_tid_24375 == 0) {
                    old_counter_24396 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24372)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24386,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_24365));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_24398 =
                            sdiv_up32(groups_per_segment_24365,
                                      segred_group_sizze_20791);
                    
                    x_20795 = 0.0F;
                    for (int32_t i_24399 = 0; i_24399 < read_per_thread_24398;
                         i_24399++) {
                        int32_t group_res_id_24400 = local_tid_24375 *
                                read_per_thread_24398 + i_24399;
                        int32_t index_of_group_res_24401 =
                                flat_segment_id_24386 *
                                groups_per_segment_24365 + group_res_id_24400;
                        
                        if (slt32(group_res_id_24400,
                                  groups_per_segment_24365)) {
                            x_20796 = ((__global
                                        float *) group_res_arr_mem_24370)[sext_i32_i64(index_of_group_res_24401) *
                                                                          sext_i32_i64(segred_group_sizze_20791)];
                            
                            float res_20797;
                            
                            res_20797 = x_20795 + x_20796;
                            x_20795 = res_20797;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_24379)[sext_i32_i64(local_tid_24375)] =
                    x_20795;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_24402;
                    int32_t skip_waves_24403;
                    float x_24390;
                    float x_24391;
                    
                    offset_24402 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_24375, segred_group_sizze_20791)) {
                            x_24390 = ((__local
                                        float *) red_arr_mem_24379)[sext_i32_i64(local_tid_24375 +
                                                                    offset_24402)];
                        }
                    }
                    offset_24402 = 1;
                    while (slt32(offset_24402, wave_sizze_24377)) {
                        if (slt32(local_tid_24375 + offset_24402,
                                  segred_group_sizze_20791) &&
                            ((local_tid_24375 - squot32(local_tid_24375,
                                                        wave_sizze_24377) *
                              wave_sizze_24377) & (2 * offset_24402 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_24391 = ((volatile __local
                                            float *) red_arr_mem_24379)[sext_i32_i64(local_tid_24375 +
                                                                        offset_24402)];
                            }
                            // apply reduction operation
                            {
                                float res_24392 = x_24390 + x_24391;
                                
                                x_24390 = res_24392;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24379)[sext_i32_i64(local_tid_24375)] =
                                    x_24390;
                            }
                        }
                        offset_24402 *= 2;
                    }
                    skip_waves_24403 = 1;
                    while (slt32(skip_waves_24403,
                                 squot32(segred_group_sizze_20791 +
                                         wave_sizze_24377 - 1,
                                         wave_sizze_24377))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_24402 = skip_waves_24403 * wave_sizze_24377;
                        if (slt32(local_tid_24375 + offset_24402,
                                  segred_group_sizze_20791) &&
                            ((local_tid_24375 - squot32(local_tid_24375,
                                                        wave_sizze_24377) *
                              wave_sizze_24377) == 0 &&
                             (squot32(local_tid_24375, wave_sizze_24377) & (2 *
                                                                            skip_waves_24403 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_24391 = ((__local
                                            float *) red_arr_mem_24379)[sext_i32_i64(local_tid_24375 +
                                                                        offset_24402)];
                            }
                            // apply reduction operation
                            {
                                float res_24392 = x_24390 + x_24391;
                                
                                x_24390 = res_24392;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_24379)[sext_i32_i64(local_tid_24375)] =
                                    x_24390;
                            }
                        }
                        skip_waves_24403 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_24375 == 0) {
                            ((__global
                              float *) mem_23604)[sext_i32_i64(gtid_20719) *
                                                  sext_i32_i64(k2p2zq_18071) +
                                                  sext_i32_i64(gtid_20720)] =
                                x_24390;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_20791
}
__kernel void mainzisegred_large_20882(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_24468_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24466_backing_aligned_1,
                                       int32_t N_18054, int32_t k2p2zq_18071,
                                       int32_t num_groups_20939, __global
                                       unsigned char *mem_23213, __global
                                       unsigned char *res_mem_23612, __global
                                       unsigned char *mem_23725,
                                       int32_t groups_per_segment_24452,
                                       int32_t elements_per_thread_24453,
                                       int32_t virt_num_groups_24454,
                                       int32_t threads_per_segment_24456,
                                       __global
                                       unsigned char *group_res_arr_mem_24457,
                                       __global
                                       unsigned char *mainzicounter_mem_24459)
{
    #define segred_group_sizze_20938 (mainzisegred_group_sizze_20876)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_24468_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_24468_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24466_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24466_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24461;
    int32_t local_tid_24462;
    int32_t group_sizze_24465;
    int32_t wave_sizze_24464;
    int32_t group_tid_24463;
    
    global_tid_24461 = get_global_id(0);
    local_tid_24462 = get_local_id(0);
    group_sizze_24465 = get_local_size(0);
    wave_sizze_24464 = LOCKSTEP_WIDTH;
    group_tid_24463 = get_group_id(0);
    
    int32_t phys_tid_20882;
    
    phys_tid_20882 = global_tid_24461;
    
    __local char *red_arr_mem_24466;
    
    red_arr_mem_24466 = (__local char *) red_arr_mem_24466_backing_0;
    
    __local char *sync_arr_mem_24468;
    
    sync_arr_mem_24468 = (__local char *) sync_arr_mem_24468_backing_1;
    
    int32_t phys_group_id_24470;
    
    phys_group_id_24470 = get_group_id(0);
    for (int32_t i_24471 = 0; i_24471 < sdiv_up32(virt_num_groups_24454 -
                                                  phys_group_id_24470,
                                                  num_groups_20939);
         i_24471++) {
        int32_t virt_group_id_24472 = phys_group_id_24470 + i_24471 *
                num_groups_20939;
        int32_t flat_segment_id_24473 = squot32(virt_group_id_24472,
                                                groups_per_segment_24452);
        int32_t global_tid_24474 = srem32(virt_group_id_24472 *
                                          segred_group_sizze_20938 +
                                          local_tid_24462,
                                          segred_group_sizze_20938 *
                                          groups_per_segment_24452);
        int32_t gtid_20868 = squot32(flat_segment_id_24473, N_18054);
        int32_t gtid_20869 = flat_segment_id_24473 -
                squot32(flat_segment_id_24473, N_18054) * N_18054;
        int32_t gtid_20881;
        float x_acc_24475;
        int32_t chunk_sizze_24476;
        
        chunk_sizze_24476 = smin32(elements_per_thread_24453,
                                   sdiv_up32(k2p2zq_18071 - global_tid_24474,
                                             threads_per_segment_24456));
        
        float x_20942;
        float x_20943;
        
        // neutral-initialise the accumulators
        {
            x_acc_24475 = 0.0F;
        }
        for (int32_t i_24480 = 0; i_24480 < chunk_sizze_24476; i_24480++) {
            gtid_20881 = global_tid_24474 + threads_per_segment_24456 * i_24480;
            // apply map function
            {
                float x_20947 = ((__global
                                  float *) res_mem_23612)[sext_i32_i64(gtid_20868) *
                                                          sext_i32_i64(k2p2zq_18071) +
                                                          sext_i32_i64(gtid_20881)];
                float x_20948 = ((__global
                                  float *) mem_23213)[sext_i32_i64(gtid_20869) *
                                                      sext_i32_i64(k2p2zq_18071) +
                                                      sext_i32_i64(gtid_20881)];
                float res_20949 = x_20947 * x_20948;
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_20942 = x_acc_24475;
                }
                // load new values
                {
                    x_20943 = res_20949;
                }
                // apply reduction operator
                {
                    float res_20944 = x_20942 + x_20943;
                    
                    // store in accumulator
                    {
                        x_acc_24475 = res_20944;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_20942 = x_acc_24475;
            ((__local
              float *) red_arr_mem_24466)[sext_i32_i64(local_tid_24462)] =
                x_20942;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_24481;
        int32_t skip_waves_24482;
        float x_24477;
        float x_24478;
        
        offset_24481 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_24462, segred_group_sizze_20938)) {
                x_24477 = ((__local
                            float *) red_arr_mem_24466)[sext_i32_i64(local_tid_24462 +
                                                        offset_24481)];
            }
        }
        offset_24481 = 1;
        while (slt32(offset_24481, wave_sizze_24464)) {
            if (slt32(local_tid_24462 + offset_24481,
                      segred_group_sizze_20938) && ((local_tid_24462 -
                                                     squot32(local_tid_24462,
                                                             wave_sizze_24464) *
                                                     wave_sizze_24464) & (2 *
                                                                          offset_24481 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_24478 = ((volatile __local
                                float *) red_arr_mem_24466)[sext_i32_i64(local_tid_24462 +
                                                            offset_24481)];
                }
                // apply reduction operation
                {
                    float res_24479 = x_24477 + x_24478;
                    
                    x_24477 = res_24479;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_24466)[sext_i32_i64(local_tid_24462)] =
                        x_24477;
                }
            }
            offset_24481 *= 2;
        }
        skip_waves_24482 = 1;
        while (slt32(skip_waves_24482, squot32(segred_group_sizze_20938 +
                                               wave_sizze_24464 - 1,
                                               wave_sizze_24464))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_24481 = skip_waves_24482 * wave_sizze_24464;
            if (slt32(local_tid_24462 + offset_24481,
                      segred_group_sizze_20938) && ((local_tid_24462 -
                                                     squot32(local_tid_24462,
                                                             wave_sizze_24464) *
                                                     wave_sizze_24464) == 0 &&
                                                    (squot32(local_tid_24462,
                                                             wave_sizze_24464) &
                                                     (2 * skip_waves_24482 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_24478 = ((__local
                                float *) red_arr_mem_24466)[sext_i32_i64(local_tid_24462 +
                                                            offset_24481)];
                }
                // apply reduction operation
                {
                    float res_24479 = x_24477 + x_24478;
                    
                    x_24477 = res_24479;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_24466)[sext_i32_i64(local_tid_24462)] =
                        x_24477;
                }
            }
            skip_waves_24482 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_24462 == 0) {
                x_acc_24475 = x_24477;
            }
        }
        if (groups_per_segment_24452 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_24462 == 0) {
                    ((__global float *) mem_23725)[sext_i32_i64(gtid_20868) *
                                                   sext_i32_i64(N_18054) +
                                                   sext_i32_i64(gtid_20869)] =
                        x_acc_24475;
                }
            }
        } else {
            int32_t old_counter_24483;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_24462 == 0) {
                    ((__global
                      float *) group_res_arr_mem_24457)[sext_i32_i64(virt_group_id_24472) *
                                                        sext_i32_i64(segred_group_sizze_20938)] =
                        x_acc_24475;
                    mem_fence_global();
                    old_counter_24483 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24459)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24473,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_24468)[0] =
                        old_counter_24483 == groups_per_segment_24452 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_24484;
            
            is_last_group_24484 = ((__local bool *) sync_arr_mem_24468)[0];
            if (is_last_group_24484) {
                if (local_tid_24462 == 0) {
                    old_counter_24483 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24459)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24473,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_24452));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_24485 =
                            sdiv_up32(groups_per_segment_24452,
                                      segred_group_sizze_20938);
                    
                    x_20942 = 0.0F;
                    for (int32_t i_24486 = 0; i_24486 < read_per_thread_24485;
                         i_24486++) {
                        int32_t group_res_id_24487 = local_tid_24462 *
                                read_per_thread_24485 + i_24486;
                        int32_t index_of_group_res_24488 =
                                flat_segment_id_24473 *
                                groups_per_segment_24452 + group_res_id_24487;
                        
                        if (slt32(group_res_id_24487,
                                  groups_per_segment_24452)) {
                            x_20943 = ((__global
                                        float *) group_res_arr_mem_24457)[sext_i32_i64(index_of_group_res_24488) *
                                                                          sext_i32_i64(segred_group_sizze_20938)];
                            
                            float res_20944;
                            
                            res_20944 = x_20942 + x_20943;
                            x_20942 = res_20944;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_24466)[sext_i32_i64(local_tid_24462)] =
                    x_20942;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_24489;
                    int32_t skip_waves_24490;
                    float x_24477;
                    float x_24478;
                    
                    offset_24489 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_24462, segred_group_sizze_20938)) {
                            x_24477 = ((__local
                                        float *) red_arr_mem_24466)[sext_i32_i64(local_tid_24462 +
                                                                    offset_24489)];
                        }
                    }
                    offset_24489 = 1;
                    while (slt32(offset_24489, wave_sizze_24464)) {
                        if (slt32(local_tid_24462 + offset_24489,
                                  segred_group_sizze_20938) &&
                            ((local_tid_24462 - squot32(local_tid_24462,
                                                        wave_sizze_24464) *
                              wave_sizze_24464) & (2 * offset_24489 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_24478 = ((volatile __local
                                            float *) red_arr_mem_24466)[sext_i32_i64(local_tid_24462 +
                                                                        offset_24489)];
                            }
                            // apply reduction operation
                            {
                                float res_24479 = x_24477 + x_24478;
                                
                                x_24477 = res_24479;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24466)[sext_i32_i64(local_tid_24462)] =
                                    x_24477;
                            }
                        }
                        offset_24489 *= 2;
                    }
                    skip_waves_24490 = 1;
                    while (slt32(skip_waves_24490,
                                 squot32(segred_group_sizze_20938 +
                                         wave_sizze_24464 - 1,
                                         wave_sizze_24464))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_24489 = skip_waves_24490 * wave_sizze_24464;
                        if (slt32(local_tid_24462 + offset_24489,
                                  segred_group_sizze_20938) &&
                            ((local_tid_24462 - squot32(local_tid_24462,
                                                        wave_sizze_24464) *
                              wave_sizze_24464) == 0 &&
                             (squot32(local_tid_24462, wave_sizze_24464) & (2 *
                                                                            skip_waves_24490 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_24478 = ((__local
                                            float *) red_arr_mem_24466)[sext_i32_i64(local_tid_24462 +
                                                                        offset_24489)];
                            }
                            // apply reduction operation
                            {
                                float res_24479 = x_24477 + x_24478;
                                
                                x_24477 = res_24479;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_24466)[sext_i32_i64(local_tid_24462)] =
                                    x_24477;
                            }
                        }
                        skip_waves_24490 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_24462 == 0) {
                            ((__global
                              float *) mem_23725)[sext_i32_i64(gtid_20868) *
                                                  sext_i32_i64(N_18054) +
                                                  sext_i32_i64(gtid_20869)] =
                                x_24477;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_20938
}
__kernel void mainzisegred_large_21461(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_24711_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24709_backing_aligned_1,
                                       int32_t N_18054, int32_t n_18059,
                                       int32_t num_groups_21513, __global
                                       unsigned char *res_mem_23788, __global
                                       unsigned char *mem_23825, __global
                                       unsigned char *mem_23829,
                                       int32_t groups_per_segment_24695,
                                       int32_t elements_per_thread_24696,
                                       int32_t virt_num_groups_24697,
                                       int32_t threads_per_segment_24699,
                                       __global
                                       unsigned char *group_res_arr_mem_24700,
                                       __global
                                       unsigned char *mainzicounter_mem_24702)
{
    #define segred_group_sizze_21512 (mainzisegred_group_sizze_21455)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_24711_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_24711_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24709_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24709_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24704;
    int32_t local_tid_24705;
    int32_t group_sizze_24708;
    int32_t wave_sizze_24707;
    int32_t group_tid_24706;
    
    global_tid_24704 = get_global_id(0);
    local_tid_24705 = get_local_id(0);
    group_sizze_24708 = get_local_size(0);
    wave_sizze_24707 = LOCKSTEP_WIDTH;
    group_tid_24706 = get_group_id(0);
    
    int32_t phys_tid_21461;
    
    phys_tid_21461 = global_tid_24704;
    
    __local char *red_arr_mem_24709;
    
    red_arr_mem_24709 = (__local char *) red_arr_mem_24709_backing_0;
    
    __local char *sync_arr_mem_24711;
    
    sync_arr_mem_24711 = (__local char *) sync_arr_mem_24711_backing_1;
    
    int32_t phys_group_id_24713;
    
    phys_group_id_24713 = get_group_id(0);
    for (int32_t i_24714 = 0; i_24714 < sdiv_up32(virt_num_groups_24697 -
                                                  phys_group_id_24713,
                                                  num_groups_21513);
         i_24714++) {
        int32_t virt_group_id_24715 = phys_group_id_24713 + i_24714 *
                num_groups_21513;
        int32_t flat_segment_id_24716 = squot32(virt_group_id_24715,
                                                groups_per_segment_24695);
        int32_t global_tid_24717 = srem32(virt_group_id_24715 *
                                          segred_group_sizze_21512 +
                                          local_tid_24705,
                                          segred_group_sizze_21512 *
                                          groups_per_segment_24695);
        int32_t gtid_21450 = flat_segment_id_24716;
        int32_t gtid_21460;
        float x_acc_24718;
        int32_t chunk_sizze_24719;
        
        chunk_sizze_24719 = smin32(elements_per_thread_24696,
                                   sdiv_up32(n_18059 - global_tid_24717,
                                             threads_per_segment_24699));
        
        float x_21516;
        float x_21517;
        
        // neutral-initialise the accumulators
        {
            x_acc_24718 = 0.0F;
        }
        for (int32_t i_24723 = 0; i_24723 < chunk_sizze_24719; i_24723++) {
            gtid_21460 = global_tid_24717 + threads_per_segment_24699 * i_24723;
            // apply map function
            {
                int32_t res_21520 = ((__global
                                      int32_t *) mem_23825)[sext_i32_i64(gtid_21450)];
                bool cond_21523 = slt32(gtid_21460, res_21520);
                float res_21524;
                
                if (cond_21523) {
                    float x_elem_21522 = ((__global
                                           float *) res_mem_23788)[sext_i32_i64(gtid_21450) *
                                                                   sext_i32_i64(N_18054) +
                                                                   sext_i32_i64(gtid_21460)];
                    
                    res_21524 = x_elem_21522;
                } else {
                    res_21524 = 0.0F;
                }
                
                float res_21525 = res_21524 * res_21524;
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_21516 = x_acc_24718;
                }
                // load new values
                {
                    x_21517 = res_21525;
                }
                // apply reduction operator
                {
                    float res_21518 = x_21516 + x_21517;
                    
                    // store in accumulator
                    {
                        x_acc_24718 = res_21518;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_21516 = x_acc_24718;
            ((__local
              float *) red_arr_mem_24709)[sext_i32_i64(local_tid_24705)] =
                x_21516;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_24724;
        int32_t skip_waves_24725;
        float x_24720;
        float x_24721;
        
        offset_24724 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_24705, segred_group_sizze_21512)) {
                x_24720 = ((__local
                            float *) red_arr_mem_24709)[sext_i32_i64(local_tid_24705 +
                                                        offset_24724)];
            }
        }
        offset_24724 = 1;
        while (slt32(offset_24724, wave_sizze_24707)) {
            if (slt32(local_tid_24705 + offset_24724,
                      segred_group_sizze_21512) && ((local_tid_24705 -
                                                     squot32(local_tid_24705,
                                                             wave_sizze_24707) *
                                                     wave_sizze_24707) & (2 *
                                                                          offset_24724 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_24721 = ((volatile __local
                                float *) red_arr_mem_24709)[sext_i32_i64(local_tid_24705 +
                                                            offset_24724)];
                }
                // apply reduction operation
                {
                    float res_24722 = x_24720 + x_24721;
                    
                    x_24720 = res_24722;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_24709)[sext_i32_i64(local_tid_24705)] =
                        x_24720;
                }
            }
            offset_24724 *= 2;
        }
        skip_waves_24725 = 1;
        while (slt32(skip_waves_24725, squot32(segred_group_sizze_21512 +
                                               wave_sizze_24707 - 1,
                                               wave_sizze_24707))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_24724 = skip_waves_24725 * wave_sizze_24707;
            if (slt32(local_tid_24705 + offset_24724,
                      segred_group_sizze_21512) && ((local_tid_24705 -
                                                     squot32(local_tid_24705,
                                                             wave_sizze_24707) *
                                                     wave_sizze_24707) == 0 &&
                                                    (squot32(local_tid_24705,
                                                             wave_sizze_24707) &
                                                     (2 * skip_waves_24725 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_24721 = ((__local
                                float *) red_arr_mem_24709)[sext_i32_i64(local_tid_24705 +
                                                            offset_24724)];
                }
                // apply reduction operation
                {
                    float res_24722 = x_24720 + x_24721;
                    
                    x_24720 = res_24722;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_24709)[sext_i32_i64(local_tid_24705)] =
                        x_24720;
                }
            }
            skip_waves_24725 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_24705 == 0) {
                x_acc_24718 = x_24720;
            }
        }
        if (groups_per_segment_24695 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_24705 == 0) {
                    ((__global float *) mem_23829)[sext_i32_i64(gtid_21450)] =
                        x_acc_24718;
                }
            }
        } else {
            int32_t old_counter_24726;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_24705 == 0) {
                    ((__global
                      float *) group_res_arr_mem_24700)[sext_i32_i64(virt_group_id_24715) *
                                                        sext_i32_i64(segred_group_sizze_21512)] =
                        x_acc_24718;
                    mem_fence_global();
                    old_counter_24726 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24702)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24716,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_24711)[0] =
                        old_counter_24726 == groups_per_segment_24695 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_24727;
            
            is_last_group_24727 = ((__local bool *) sync_arr_mem_24711)[0];
            if (is_last_group_24727) {
                if (local_tid_24705 == 0) {
                    old_counter_24726 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24702)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24716,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_24695));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_24728 =
                            sdiv_up32(groups_per_segment_24695,
                                      segred_group_sizze_21512);
                    
                    x_21516 = 0.0F;
                    for (int32_t i_24729 = 0; i_24729 < read_per_thread_24728;
                         i_24729++) {
                        int32_t group_res_id_24730 = local_tid_24705 *
                                read_per_thread_24728 + i_24729;
                        int32_t index_of_group_res_24731 =
                                flat_segment_id_24716 *
                                groups_per_segment_24695 + group_res_id_24730;
                        
                        if (slt32(group_res_id_24730,
                                  groups_per_segment_24695)) {
                            x_21517 = ((__global
                                        float *) group_res_arr_mem_24700)[sext_i32_i64(index_of_group_res_24731) *
                                                                          sext_i32_i64(segred_group_sizze_21512)];
                            
                            float res_21518;
                            
                            res_21518 = x_21516 + x_21517;
                            x_21516 = res_21518;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_24709)[sext_i32_i64(local_tid_24705)] =
                    x_21516;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_24732;
                    int32_t skip_waves_24733;
                    float x_24720;
                    float x_24721;
                    
                    offset_24732 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_24705, segred_group_sizze_21512)) {
                            x_24720 = ((__local
                                        float *) red_arr_mem_24709)[sext_i32_i64(local_tid_24705 +
                                                                    offset_24732)];
                        }
                    }
                    offset_24732 = 1;
                    while (slt32(offset_24732, wave_sizze_24707)) {
                        if (slt32(local_tid_24705 + offset_24732,
                                  segred_group_sizze_21512) &&
                            ((local_tid_24705 - squot32(local_tid_24705,
                                                        wave_sizze_24707) *
                              wave_sizze_24707) & (2 * offset_24732 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_24721 = ((volatile __local
                                            float *) red_arr_mem_24709)[sext_i32_i64(local_tid_24705 +
                                                                        offset_24732)];
                            }
                            // apply reduction operation
                            {
                                float res_24722 = x_24720 + x_24721;
                                
                                x_24720 = res_24722;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24709)[sext_i32_i64(local_tid_24705)] =
                                    x_24720;
                            }
                        }
                        offset_24732 *= 2;
                    }
                    skip_waves_24733 = 1;
                    while (slt32(skip_waves_24733,
                                 squot32(segred_group_sizze_21512 +
                                         wave_sizze_24707 - 1,
                                         wave_sizze_24707))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_24732 = skip_waves_24733 * wave_sizze_24707;
                        if (slt32(local_tid_24705 + offset_24732,
                                  segred_group_sizze_21512) &&
                            ((local_tid_24705 - squot32(local_tid_24705,
                                                        wave_sizze_24707) *
                              wave_sizze_24707) == 0 &&
                             (squot32(local_tid_24705, wave_sizze_24707) & (2 *
                                                                            skip_waves_24733 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_24721 = ((__local
                                            float *) red_arr_mem_24709)[sext_i32_i64(local_tid_24705 +
                                                                        offset_24732)];
                            }
                            // apply reduction operation
                            {
                                float res_24722 = x_24720 + x_24721;
                                
                                x_24720 = res_24722;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_24709)[sext_i32_i64(local_tid_24705)] =
                                    x_24720;
                            }
                        }
                        skip_waves_24733 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_24705 == 0) {
                            ((__global
                              float *) mem_23829)[sext_i32_i64(gtid_21450)] =
                                x_24720;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21512
}
__kernel void mainzisegred_large_21483(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_24652_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24650_backing_aligned_1,
                                       int32_t N_18056, int32_t n_18059,
                                       int32_t num_groups_21497, __global
                                       unsigned char *images_mem_23189, __global
                                       unsigned char *mem_23825,
                                       int32_t groups_per_segment_24636,
                                       int32_t elements_per_thread_24637,
                                       int32_t virt_num_groups_24638,
                                       int32_t threads_per_segment_24640,
                                       __global
                                       unsigned char *group_res_arr_mem_24641,
                                       __global
                                       unsigned char *mainzicounter_mem_24643)
{
    #define segred_group_sizze_21496 (mainzisegred_group_sizze_21477)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_24652_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_24652_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24650_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24650_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24645;
    int32_t local_tid_24646;
    int32_t group_sizze_24649;
    int32_t wave_sizze_24648;
    int32_t group_tid_24647;
    
    global_tid_24645 = get_global_id(0);
    local_tid_24646 = get_local_id(0);
    group_sizze_24649 = get_local_size(0);
    wave_sizze_24648 = LOCKSTEP_WIDTH;
    group_tid_24647 = get_group_id(0);
    
    int32_t phys_tid_21483;
    
    phys_tid_21483 = global_tid_24645;
    
    __local char *red_arr_mem_24650;
    
    red_arr_mem_24650 = (__local char *) red_arr_mem_24650_backing_0;
    
    __local char *sync_arr_mem_24652;
    
    sync_arr_mem_24652 = (__local char *) sync_arr_mem_24652_backing_1;
    
    int32_t phys_group_id_24654;
    
    phys_group_id_24654 = get_group_id(0);
    for (int32_t i_24655 = 0; i_24655 < sdiv_up32(virt_num_groups_24638 -
                                                  phys_group_id_24654,
                                                  num_groups_21497);
         i_24655++) {
        int32_t virt_group_id_24656 = phys_group_id_24654 + i_24655 *
                num_groups_21497;
        int32_t flat_segment_id_24657 = squot32(virt_group_id_24656,
                                                groups_per_segment_24636);
        int32_t global_tid_24658 = srem32(virt_group_id_24656 *
                                          segred_group_sizze_21496 +
                                          local_tid_24646,
                                          segred_group_sizze_21496 *
                                          groups_per_segment_24636);
        int32_t gtid_21472 = flat_segment_id_24657;
        int32_t gtid_21482;
        int32_t x_acc_24659;
        int32_t chunk_sizze_24660;
        
        chunk_sizze_24660 = smin32(elements_per_thread_24637,
                                   sdiv_up32(n_18059 - global_tid_24658,
                                             threads_per_segment_24640));
        
        int32_t x_21500;
        int32_t x_21501;
        
        // neutral-initialise the accumulators
        {
            x_acc_24659 = 0;
        }
        for (int32_t i_24664 = 0; i_24664 < chunk_sizze_24660; i_24664++) {
            gtid_21482 = global_tid_24658 + threads_per_segment_24640 * i_24664;
            // apply map function
            {
                float x_21504 = ((__global
                                  float *) images_mem_23189)[sext_i32_i64(gtid_21472) *
                                                             sext_i32_i64(N_18056) +
                                                             sext_i32_i64(gtid_21482)];
                bool res_21505;
                
                res_21505 = futrts_isnan32(x_21504);
                
                bool cond_21506 = !res_21505;
                int32_t res_21507 = btoi_bool_i32(cond_21506);
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_21500 = x_acc_24659;
                }
                // load new values
                {
                    x_21501 = res_21507;
                }
                // apply reduction operator
                {
                    int32_t res_21502 = add32(x_21500, x_21501);
                    
                    // store in accumulator
                    {
                        x_acc_24659 = res_21502;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_21500 = x_acc_24659;
            ((__local
              int32_t *) red_arr_mem_24650)[sext_i32_i64(local_tid_24646)] =
                x_21500;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_24665;
        int32_t skip_waves_24666;
        int32_t x_24661;
        int32_t x_24662;
        
        offset_24665 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_24646, segred_group_sizze_21496)) {
                x_24661 = ((__local
                            int32_t *) red_arr_mem_24650)[sext_i32_i64(local_tid_24646 +
                                                          offset_24665)];
            }
        }
        offset_24665 = 1;
        while (slt32(offset_24665, wave_sizze_24648)) {
            if (slt32(local_tid_24646 + offset_24665,
                      segred_group_sizze_21496) && ((local_tid_24646 -
                                                     squot32(local_tid_24646,
                                                             wave_sizze_24648) *
                                                     wave_sizze_24648) & (2 *
                                                                          offset_24665 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_24662 = ((volatile __local
                                int32_t *) red_arr_mem_24650)[sext_i32_i64(local_tid_24646 +
                                                              offset_24665)];
                }
                // apply reduction operation
                {
                    int32_t res_24663 = add32(x_24661, x_24662);
                    
                    x_24661 = res_24663;
                }
                // write result of operation
                {
                    ((volatile __local
                      int32_t *) red_arr_mem_24650)[sext_i32_i64(local_tid_24646)] =
                        x_24661;
                }
            }
            offset_24665 *= 2;
        }
        skip_waves_24666 = 1;
        while (slt32(skip_waves_24666, squot32(segred_group_sizze_21496 +
                                               wave_sizze_24648 - 1,
                                               wave_sizze_24648))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_24665 = skip_waves_24666 * wave_sizze_24648;
            if (slt32(local_tid_24646 + offset_24665,
                      segred_group_sizze_21496) && ((local_tid_24646 -
                                                     squot32(local_tid_24646,
                                                             wave_sizze_24648) *
                                                     wave_sizze_24648) == 0 &&
                                                    (squot32(local_tid_24646,
                                                             wave_sizze_24648) &
                                                     (2 * skip_waves_24666 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_24662 = ((__local
                                int32_t *) red_arr_mem_24650)[sext_i32_i64(local_tid_24646 +
                                                              offset_24665)];
                }
                // apply reduction operation
                {
                    int32_t res_24663 = add32(x_24661, x_24662);
                    
                    x_24661 = res_24663;
                }
                // write result of operation
                {
                    ((__local
                      int32_t *) red_arr_mem_24650)[sext_i32_i64(local_tid_24646)] =
                        x_24661;
                }
            }
            skip_waves_24666 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_24646 == 0) {
                x_acc_24659 = x_24661;
            }
        }
        if (groups_per_segment_24636 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_24646 == 0) {
                    ((__global int32_t *) mem_23825)[sext_i32_i64(gtid_21472)] =
                        x_acc_24659;
                }
            }
        } else {
            int32_t old_counter_24667;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_24646 == 0) {
                    ((__global
                      int32_t *) group_res_arr_mem_24641)[sext_i32_i64(virt_group_id_24656) *
                                                          sext_i32_i64(segred_group_sizze_21496)] =
                        x_acc_24659;
                    mem_fence_global();
                    old_counter_24667 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24643)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24657,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_24652)[0] =
                        old_counter_24667 == groups_per_segment_24636 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_24668;
            
            is_last_group_24668 = ((__local bool *) sync_arr_mem_24652)[0];
            if (is_last_group_24668) {
                if (local_tid_24646 == 0) {
                    old_counter_24667 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24643)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24657,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_24636));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_24669 =
                            sdiv_up32(groups_per_segment_24636,
                                      segred_group_sizze_21496);
                    
                    x_21500 = 0;
                    for (int32_t i_24670 = 0; i_24670 < read_per_thread_24669;
                         i_24670++) {
                        int32_t group_res_id_24671 = local_tid_24646 *
                                read_per_thread_24669 + i_24670;
                        int32_t index_of_group_res_24672 =
                                flat_segment_id_24657 *
                                groups_per_segment_24636 + group_res_id_24671;
                        
                        if (slt32(group_res_id_24671,
                                  groups_per_segment_24636)) {
                            x_21501 = ((__global
                                        int32_t *) group_res_arr_mem_24641)[sext_i32_i64(index_of_group_res_24672) *
                                                                            sext_i32_i64(segred_group_sizze_21496)];
                            
                            int32_t res_21502;
                            
                            res_21502 = add32(x_21500, x_21501);
                            x_21500 = res_21502;
                        }
                    }
                }
                ((__local
                  int32_t *) red_arr_mem_24650)[sext_i32_i64(local_tid_24646)] =
                    x_21500;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_24673;
                    int32_t skip_waves_24674;
                    int32_t x_24661;
                    int32_t x_24662;
                    
                    offset_24673 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_24646, segred_group_sizze_21496)) {
                            x_24661 = ((__local
                                        int32_t *) red_arr_mem_24650)[sext_i32_i64(local_tid_24646 +
                                                                      offset_24673)];
                        }
                    }
                    offset_24673 = 1;
                    while (slt32(offset_24673, wave_sizze_24648)) {
                        if (slt32(local_tid_24646 + offset_24673,
                                  segred_group_sizze_21496) &&
                            ((local_tid_24646 - squot32(local_tid_24646,
                                                        wave_sizze_24648) *
                              wave_sizze_24648) & (2 * offset_24673 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_24662 = ((volatile __local
                                            int32_t *) red_arr_mem_24650)[sext_i32_i64(local_tid_24646 +
                                                                          offset_24673)];
                            }
                            // apply reduction operation
                            {
                                int32_t res_24663 = add32(x_24661, x_24662);
                                
                                x_24661 = res_24663;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  int32_t *) red_arr_mem_24650)[sext_i32_i64(local_tid_24646)] =
                                    x_24661;
                            }
                        }
                        offset_24673 *= 2;
                    }
                    skip_waves_24674 = 1;
                    while (slt32(skip_waves_24674,
                                 squot32(segred_group_sizze_21496 +
                                         wave_sizze_24648 - 1,
                                         wave_sizze_24648))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_24673 = skip_waves_24674 * wave_sizze_24648;
                        if (slt32(local_tid_24646 + offset_24673,
                                  segred_group_sizze_21496) &&
                            ((local_tid_24646 - squot32(local_tid_24646,
                                                        wave_sizze_24648) *
                              wave_sizze_24648) == 0 &&
                             (squot32(local_tid_24646, wave_sizze_24648) & (2 *
                                                                            skip_waves_24674 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_24662 = ((__local
                                            int32_t *) red_arr_mem_24650)[sext_i32_i64(local_tid_24646 +
                                                                          offset_24673)];
                            }
                            // apply reduction operation
                            {
                                int32_t res_24663 = add32(x_24661, x_24662);
                                
                                x_24661 = res_24663;
                            }
                            // write result of operation
                            {
                                ((__local
                                  int32_t *) red_arr_mem_24650)[sext_i32_i64(local_tid_24646)] =
                                    x_24661;
                            }
                        }
                        skip_waves_24674 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_24646 == 0) {
                            ((__global
                              int32_t *) mem_23825)[sext_i32_i64(gtid_21472)] =
                                x_24661;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21496
}
__kernel void mainzisegred_large_21617(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_24815_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24813_backing_aligned_1,
                                       int32_t N_18054, int32_t res_18381,
                                       int32_t num_groups_21636, __global
                                       unsigned char *res_mem_23788, __global
                                       unsigned char *res_mem_23840, __global
                                       unsigned char *res_mem_23841, __global
                                       unsigned char *mem_23853,
                                       int32_t groups_per_segment_24799,
                                       int32_t elements_per_thread_24800,
                                       int32_t virt_num_groups_24801,
                                       int32_t threads_per_segment_24803,
                                       __global
                                       unsigned char *group_res_arr_mem_24804,
                                       __global
                                       unsigned char *mainzicounter_mem_24806)
{
    #define segred_group_sizze_21635 (mainzisegred_group_sizze_21611)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_24815_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_24815_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24813_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24813_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24808;
    int32_t local_tid_24809;
    int32_t group_sizze_24812;
    int32_t wave_sizze_24811;
    int32_t group_tid_24810;
    
    global_tid_24808 = get_global_id(0);
    local_tid_24809 = get_local_id(0);
    group_sizze_24812 = get_local_size(0);
    wave_sizze_24811 = LOCKSTEP_WIDTH;
    group_tid_24810 = get_group_id(0);
    
    int32_t phys_tid_21617;
    
    phys_tid_21617 = global_tid_24808;
    
    __local char *red_arr_mem_24813;
    
    red_arr_mem_24813 = (__local char *) red_arr_mem_24813_backing_0;
    
    __local char *sync_arr_mem_24815;
    
    sync_arr_mem_24815 = (__local char *) sync_arr_mem_24815_backing_1;
    
    int32_t phys_group_id_24817;
    
    phys_group_id_24817 = get_group_id(0);
    for (int32_t i_24818 = 0; i_24818 < sdiv_up32(virt_num_groups_24801 -
                                                  phys_group_id_24817,
                                                  num_groups_21636);
         i_24818++) {
        int32_t virt_group_id_24819 = phys_group_id_24817 + i_24818 *
                num_groups_21636;
        int32_t flat_segment_id_24820 = squot32(virt_group_id_24819,
                                                groups_per_segment_24799);
        int32_t global_tid_24821 = srem32(virt_group_id_24819 *
                                          segred_group_sizze_21635 +
                                          local_tid_24809,
                                          segred_group_sizze_21635 *
                                          groups_per_segment_24799);
        int32_t gtid_21606 = flat_segment_id_24820;
        int32_t gtid_21616;
        float x_acc_24822;
        int32_t chunk_sizze_24823;
        
        chunk_sizze_24823 = smin32(elements_per_thread_24800,
                                   sdiv_up32(res_18381 - global_tid_24821,
                                             threads_per_segment_24803));
        
        float x_21639;
        float x_21640;
        
        // neutral-initialise the accumulators
        {
            x_acc_24822 = 0.0F;
        }
        for (int32_t i_24827 = 0; i_24827 < chunk_sizze_24823; i_24827++) {
            gtid_21616 = global_tid_24821 + threads_per_segment_24803 * i_24827;
            // apply map function
            {
                int32_t x_21644 = ((__global
                                    int32_t *) res_mem_23840)[sext_i32_i64(gtid_21606)];
                bool cond_21646 = slt32(gtid_21616, x_21644);
                float res_21647;
                
                if (cond_21646) {
                    int32_t x_21643 = ((__global
                                        int32_t *) res_mem_23841)[sext_i32_i64(gtid_21606)];
                    int32_t x_21648 = add32(gtid_21616, x_21643);
                    int32_t x_21649 = sub32(x_21648, x_21644);
                    int32_t i_21650 = add32(1, x_21649);
                    float res_21651 = ((__global
                                        float *) res_mem_23788)[sext_i32_i64(gtid_21606) *
                                                                sext_i32_i64(N_18054) +
                                                                sext_i32_i64(i_21650)];
                    
                    res_21647 = res_21651;
                } else {
                    res_21647 = 0.0F;
                }
                // save map-out results
                { }
                // load accumulator
                {
                    x_21639 = x_acc_24822;
                }
                // load new values
                {
                    x_21640 = res_21647;
                }
                // apply reduction operator
                {
                    float res_21641 = x_21639 + x_21640;
                    
                    // store in accumulator
                    {
                        x_acc_24822 = res_21641;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_21639 = x_acc_24822;
            ((__local
              float *) red_arr_mem_24813)[sext_i32_i64(local_tid_24809)] =
                x_21639;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_24828;
        int32_t skip_waves_24829;
        float x_24824;
        float x_24825;
        
        offset_24828 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_24809, segred_group_sizze_21635)) {
                x_24824 = ((__local
                            float *) red_arr_mem_24813)[sext_i32_i64(local_tid_24809 +
                                                        offset_24828)];
            }
        }
        offset_24828 = 1;
        while (slt32(offset_24828, wave_sizze_24811)) {
            if (slt32(local_tid_24809 + offset_24828,
                      segred_group_sizze_21635) && ((local_tid_24809 -
                                                     squot32(local_tid_24809,
                                                             wave_sizze_24811) *
                                                     wave_sizze_24811) & (2 *
                                                                          offset_24828 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_24825 = ((volatile __local
                                float *) red_arr_mem_24813)[sext_i32_i64(local_tid_24809 +
                                                            offset_24828)];
                }
                // apply reduction operation
                {
                    float res_24826 = x_24824 + x_24825;
                    
                    x_24824 = res_24826;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_24813)[sext_i32_i64(local_tid_24809)] =
                        x_24824;
                }
            }
            offset_24828 *= 2;
        }
        skip_waves_24829 = 1;
        while (slt32(skip_waves_24829, squot32(segred_group_sizze_21635 +
                                               wave_sizze_24811 - 1,
                                               wave_sizze_24811))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_24828 = skip_waves_24829 * wave_sizze_24811;
            if (slt32(local_tid_24809 + offset_24828,
                      segred_group_sizze_21635) && ((local_tid_24809 -
                                                     squot32(local_tid_24809,
                                                             wave_sizze_24811) *
                                                     wave_sizze_24811) == 0 &&
                                                    (squot32(local_tid_24809,
                                                             wave_sizze_24811) &
                                                     (2 * skip_waves_24829 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_24825 = ((__local
                                float *) red_arr_mem_24813)[sext_i32_i64(local_tid_24809 +
                                                            offset_24828)];
                }
                // apply reduction operation
                {
                    float res_24826 = x_24824 + x_24825;
                    
                    x_24824 = res_24826;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_24813)[sext_i32_i64(local_tid_24809)] =
                        x_24824;
                }
            }
            skip_waves_24829 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_24809 == 0) {
                x_acc_24822 = x_24824;
            }
        }
        if (groups_per_segment_24799 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_24809 == 0) {
                    ((__global float *) mem_23853)[sext_i32_i64(gtid_21606)] =
                        x_acc_24822;
                }
            }
        } else {
            int32_t old_counter_24830;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_24809 == 0) {
                    ((__global
                      float *) group_res_arr_mem_24804)[sext_i32_i64(virt_group_id_24819) *
                                                        sext_i32_i64(segred_group_sizze_21635)] =
                        x_acc_24822;
                    mem_fence_global();
                    old_counter_24830 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24806)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24820,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_24815)[0] =
                        old_counter_24830 == groups_per_segment_24799 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_24831;
            
            is_last_group_24831 = ((__local bool *) sync_arr_mem_24815)[0];
            if (is_last_group_24831) {
                if (local_tid_24809 == 0) {
                    old_counter_24830 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24806)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24820,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_24799));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_24832 =
                            sdiv_up32(groups_per_segment_24799,
                                      segred_group_sizze_21635);
                    
                    x_21639 = 0.0F;
                    for (int32_t i_24833 = 0; i_24833 < read_per_thread_24832;
                         i_24833++) {
                        int32_t group_res_id_24834 = local_tid_24809 *
                                read_per_thread_24832 + i_24833;
                        int32_t index_of_group_res_24835 =
                                flat_segment_id_24820 *
                                groups_per_segment_24799 + group_res_id_24834;
                        
                        if (slt32(group_res_id_24834,
                                  groups_per_segment_24799)) {
                            x_21640 = ((__global
                                        float *) group_res_arr_mem_24804)[sext_i32_i64(index_of_group_res_24835) *
                                                                          sext_i32_i64(segred_group_sizze_21635)];
                            
                            float res_21641;
                            
                            res_21641 = x_21639 + x_21640;
                            x_21639 = res_21641;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_24813)[sext_i32_i64(local_tid_24809)] =
                    x_21639;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_24836;
                    int32_t skip_waves_24837;
                    float x_24824;
                    float x_24825;
                    
                    offset_24836 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_24809, segred_group_sizze_21635)) {
                            x_24824 = ((__local
                                        float *) red_arr_mem_24813)[sext_i32_i64(local_tid_24809 +
                                                                    offset_24836)];
                        }
                    }
                    offset_24836 = 1;
                    while (slt32(offset_24836, wave_sizze_24811)) {
                        if (slt32(local_tid_24809 + offset_24836,
                                  segred_group_sizze_21635) &&
                            ((local_tid_24809 - squot32(local_tid_24809,
                                                        wave_sizze_24811) *
                              wave_sizze_24811) & (2 * offset_24836 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_24825 = ((volatile __local
                                            float *) red_arr_mem_24813)[sext_i32_i64(local_tid_24809 +
                                                                        offset_24836)];
                            }
                            // apply reduction operation
                            {
                                float res_24826 = x_24824 + x_24825;
                                
                                x_24824 = res_24826;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24813)[sext_i32_i64(local_tid_24809)] =
                                    x_24824;
                            }
                        }
                        offset_24836 *= 2;
                    }
                    skip_waves_24837 = 1;
                    while (slt32(skip_waves_24837,
                                 squot32(segred_group_sizze_21635 +
                                         wave_sizze_24811 - 1,
                                         wave_sizze_24811))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_24836 = skip_waves_24837 * wave_sizze_24811;
                        if (slt32(local_tid_24809 + offset_24836,
                                  segred_group_sizze_21635) &&
                            ((local_tid_24809 - squot32(local_tid_24809,
                                                        wave_sizze_24811) *
                              wave_sizze_24811) == 0 &&
                             (squot32(local_tid_24809, wave_sizze_24811) & (2 *
                                                                            skip_waves_24837 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_24825 = ((__local
                                            float *) red_arr_mem_24813)[sext_i32_i64(local_tid_24809 +
                                                                        offset_24836)];
                            }
                            // apply reduction operation
                            {
                                float res_24826 = x_24824 + x_24825;
                                
                                x_24824 = res_24826;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_24813)[sext_i32_i64(local_tid_24809)] =
                                    x_24824;
                            }
                        }
                        skip_waves_24837 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_24809 == 0) {
                            ((__global
                              float *) mem_23853)[sext_i32_i64(gtid_21606)] =
                                x_24824;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21635
}
__kernel void mainzisegred_large_22108(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_24992_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24990_backing_aligned_1,
                                       __local volatile
                                       int64_t *red_arr_mem_24988_backing_aligned_2,
                                       __local volatile
                                       int64_t *red_arr_mem_24986_backing_aligned_3,
                                       int32_t iota_arg_18403,
                                       int32_t num_groups_22292, __global
                                       unsigned char *mem_23858, __global
                                       unsigned char *mem_23874, __global
                                       unsigned char *mem_23877, __global
                                       unsigned char *mem_23883, __global
                                       unsigned char *mem_23886, __global
                                       unsigned char *mem_23889, __global
                                       unsigned char *mem_23892,
                                       int32_t groups_per_segment_24968,
                                       int32_t elements_per_thread_24969,
                                       int32_t virt_num_groups_24970, __global
                                       unsigned char *group_res_arr_mem_24973,
                                       __global
                                       unsigned char *group_res_arr_mem_24975,
                                       __global
                                       unsigned char *group_res_arr_mem_24977,
                                       __global
                                       unsigned char *mainzicounter_mem_24979)
{
    #define segred_group_sizze_22291 (mainzisegred_group_sizze_22102)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_24992_backing_3 =
                          (__local volatile
                           char *) sync_arr_mem_24992_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24990_backing_2 =
                          (__local volatile
                           char *) red_arr_mem_24990_backing_aligned_1;
    __local volatile char *restrict red_arr_mem_24988_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_24988_backing_aligned_2;
    __local volatile char *restrict red_arr_mem_24986_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24986_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24981;
    int32_t local_tid_24982;
    int32_t group_sizze_24985;
    int32_t wave_sizze_24984;
    int32_t group_tid_24983;
    
    global_tid_24981 = get_global_id(0);
    local_tid_24982 = get_local_id(0);
    group_sizze_24985 = get_local_size(0);
    wave_sizze_24984 = LOCKSTEP_WIDTH;
    group_tid_24983 = get_group_id(0);
    
    int32_t phys_tid_22108;
    
    phys_tid_22108 = global_tid_24981;
    
    __local char *red_arr_mem_24986;
    
    red_arr_mem_24986 = (__local char *) red_arr_mem_24986_backing_0;
    
    __local char *red_arr_mem_24988;
    
    red_arr_mem_24988 = (__local char *) red_arr_mem_24988_backing_1;
    
    __local char *red_arr_mem_24990;
    
    red_arr_mem_24990 = (__local char *) red_arr_mem_24990_backing_2;
    
    __local char *sync_arr_mem_24992;
    
    sync_arr_mem_24992 = (__local char *) sync_arr_mem_24992_backing_3;
    
    int32_t phys_group_id_24994;
    
    phys_group_id_24994 = get_group_id(0);
    for (int32_t i_24995 = 0; i_24995 < sdiv_up32(virt_num_groups_24970 -
                                                  phys_group_id_24994,
                                                  num_groups_22292);
         i_24995++) {
        int32_t virt_group_id_24996 = phys_group_id_24994 + i_24995 *
                num_groups_22292;
        int32_t flat_segment_id_24997 = squot32(virt_group_id_24996,
                                                groups_per_segment_24968);
        int32_t global_tid_24998 = srem32(virt_group_id_24996 *
                                          segred_group_sizze_22291 +
                                          local_tid_24982,
                                          segred_group_sizze_22291 *
                                          groups_per_segment_24968);
        int32_t gtid_22097 = flat_segment_id_24997;
        int32_t gtid_22107;
        bool x_acc_24999;
        int32_t x_acc_25000;
        float x_acc_25001;
        int32_t chunk_sizze_25002;
        int32_t starting_point_25003;
        
        starting_point_25003 = global_tid_24998 * elements_per_thread_24969;
        
        int32_t remaining_elements_25004;
        
        remaining_elements_25004 = iota_arg_18403 - starting_point_25003;
        if (sle32(remaining_elements_25004, 0) || sle32(iota_arg_18403,
                                                        starting_point_25003)) {
            chunk_sizze_25002 = 0;
        } else {
            if (slt32(iota_arg_18403, (global_tid_24998 + 1) *
                      elements_per_thread_24969)) {
                chunk_sizze_25002 = iota_arg_18403 - global_tid_24998 *
                    elements_per_thread_24969;
            } else {
                chunk_sizze_25002 = elements_per_thread_24969;
            }
        }
        
        bool x_22297;
        int32_t x_22298;
        float x_22299;
        bool x_22300;
        int32_t x_22301;
        float x_22302;
        
        // neutral-initialise the accumulators
        {
            x_acc_24999 = 0;
            x_acc_25000 = -1;
            x_acc_25001 = 0.0F;
        }
        for (int32_t i_25019 = 0; i_25019 < elements_per_thread_24969;
             i_25019++) {
            gtid_22107 = local_tid_24982 + (squot32(global_tid_24998,
                                                    segred_group_sizze_22291) *
                                            elements_per_thread_24969 +
                                            i_25019) * segred_group_sizze_22291;
            if (slt32(gtid_22107, iota_arg_18403)) {
                // apply map function
                {
                    int32_t y_22311 = ((__global
                                        int32_t *) mem_23877)[sext_i32_i64(gtid_22097)];
                    float y_22312 = ((__global
                                      float *) mem_23874)[sext_i32_i64(gtid_22097)];
                    float x_22316 = ((__global
                                      float *) mem_23883)[sext_i32_i64(gtid_22097) *
                                                          sext_i32_i64(iota_arg_18403) +
                                                          sext_i32_i64(gtid_22107)];
                    float x_22317 = ((__global
                                      float *) mem_23858)[sext_i32_i64(gtid_22107)];
                    float res_22320 = x_22316 / y_22312;
                    bool cond_22321 = slt32(gtid_22107, y_22311);
                    bool res_22322;
                    
                    res_22322 = futrts_isnan32(res_22320);
                    
                    bool res_22323 = !res_22322;
                    bool x_22324 = cond_22321 && res_22323;
                    float res_22325 = (float) fabs(res_22320);
                    bool res_22326 = x_22317 < res_22325;
                    bool x_22327 = x_22324 && res_22326;
                    float res_22328;
                    
                    if (cond_22321) {
                        res_22328 = res_22320;
                    } else {
                        res_22328 = 0.0F;
                    }
                    // save map-out results
                    { }
                    // load accumulator
                    {
                        x_22297 = x_acc_24999;
                        x_22298 = x_acc_25000;
                        x_22299 = x_acc_25001;
                    }
                    // load new values
                    {
                        x_22300 = x_22327;
                        x_22301 = gtid_22107;
                        x_22302 = res_22328;
                    }
                    // apply reduction operator
                    {
                        bool res_22303;
                        int32_t res_22304;
                        
                        if (x_22297) {
                            res_22303 = x_22297;
                            res_22304 = x_22298;
                        } else {
                            bool x_22305 = x_22300 && x_22300;
                            bool x_22306 = !x_22300;
                            bool y_22307 = x_22297 && x_22306;
                            bool res_22308 = x_22305 || y_22307;
                            int32_t res_22309;
                            
                            if (x_22300) {
                                res_22309 = x_22301;
                            } else {
                                res_22309 = x_22298;
                            }
                            res_22303 = res_22308;
                            res_22304 = res_22309;
                        }
                        
                        float res_22310 = x_22299 + x_22302;
                        
                        // store in accumulator
                        {
                            x_acc_24999 = res_22303;
                            x_acc_25000 = res_22304;
                            x_acc_25001 = res_22310;
                        }
                    }
                }
            }
            // to reduce current chunk, first store our result in memory
            {
                x_22297 = x_acc_24999;
                x_22298 = x_acc_25000;
                x_22299 = x_acc_25001;
                ((__local
                  bool *) red_arr_mem_24986)[sext_i32_i64(local_tid_24982)] =
                    x_22297;
                ((__local
                  int32_t *) red_arr_mem_24988)[sext_i32_i64(local_tid_24982)] =
                    x_22298;
                ((__local
                  float *) red_arr_mem_24990)[sext_i32_i64(local_tid_24982)] =
                    x_22299;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t offset_25020;
            int32_t skip_waves_25021;
            bool x_25005;
            int32_t x_25006;
            float x_25007;
            bool x_25008;
            int32_t x_25009;
            float x_25010;
            
            offset_25020 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_24982, segred_group_sizze_22291)) {
                    x_25005 = ((__local
                                bool *) red_arr_mem_24986)[sext_i32_i64(local_tid_24982 +
                                                           offset_25020)];
                    x_25006 = ((__local
                                int32_t *) red_arr_mem_24988)[sext_i32_i64(local_tid_24982 +
                                                              offset_25020)];
                    x_25007 = ((__local
                                float *) red_arr_mem_24990)[sext_i32_i64(local_tid_24982 +
                                                            offset_25020)];
                }
            }
            offset_25020 = 1;
            while (slt32(offset_25020, wave_sizze_24984)) {
                if (slt32(local_tid_24982 + offset_25020,
                          segred_group_sizze_22291) && ((local_tid_24982 -
                                                         squot32(local_tid_24982,
                                                                 wave_sizze_24984) *
                                                         wave_sizze_24984) &
                                                        (2 * offset_25020 -
                                                         1)) == 0) {
                    // read array element
                    {
                        x_25008 = ((volatile __local
                                    bool *) red_arr_mem_24986)[sext_i32_i64(local_tid_24982 +
                                                               offset_25020)];
                        x_25009 = ((volatile __local
                                    int32_t *) red_arr_mem_24988)[sext_i32_i64(local_tid_24982 +
                                                                  offset_25020)];
                        x_25010 = ((volatile __local
                                    float *) red_arr_mem_24990)[sext_i32_i64(local_tid_24982 +
                                                                offset_25020)];
                    }
                    // apply reduction operation
                    {
                        bool res_25011;
                        int32_t res_25012;
                        
                        if (x_25005) {
                            res_25011 = x_25005;
                            res_25012 = x_25006;
                        } else {
                            bool x_25013 = x_25008 && x_25008;
                            bool x_25014 = !x_25008;
                            bool y_25015 = x_25005 && x_25014;
                            bool res_25016 = x_25013 || y_25015;
                            int32_t res_25017;
                            
                            if (x_25008) {
                                res_25017 = x_25009;
                            } else {
                                res_25017 = x_25006;
                            }
                            res_25011 = res_25016;
                            res_25012 = res_25017;
                        }
                        
                        float res_25018 = x_25007 + x_25010;
                        
                        x_25005 = res_25011;
                        x_25006 = res_25012;
                        x_25007 = res_25018;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          bool *) red_arr_mem_24986)[sext_i32_i64(local_tid_24982)] =
                            x_25005;
                        ((volatile __local
                          int32_t *) red_arr_mem_24988)[sext_i32_i64(local_tid_24982)] =
                            x_25006;
                        ((volatile __local
                          float *) red_arr_mem_24990)[sext_i32_i64(local_tid_24982)] =
                            x_25007;
                    }
                }
                offset_25020 *= 2;
            }
            skip_waves_25021 = 1;
            while (slt32(skip_waves_25021, squot32(segred_group_sizze_22291 +
                                                   wave_sizze_24984 - 1,
                                                   wave_sizze_24984))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_25020 = skip_waves_25021 * wave_sizze_24984;
                if (slt32(local_tid_24982 + offset_25020,
                          segred_group_sizze_22291) && ((local_tid_24982 -
                                                         squot32(local_tid_24982,
                                                                 wave_sizze_24984) *
                                                         wave_sizze_24984) ==
                                                        0 &&
                                                        (squot32(local_tid_24982,
                                                                 wave_sizze_24984) &
                                                         (2 * skip_waves_25021 -
                                                          1)) == 0)) {
                    // read array element
                    {
                        x_25008 = ((__local
                                    bool *) red_arr_mem_24986)[sext_i32_i64(local_tid_24982 +
                                                               offset_25020)];
                        x_25009 = ((__local
                                    int32_t *) red_arr_mem_24988)[sext_i32_i64(local_tid_24982 +
                                                                  offset_25020)];
                        x_25010 = ((__local
                                    float *) red_arr_mem_24990)[sext_i32_i64(local_tid_24982 +
                                                                offset_25020)];
                    }
                    // apply reduction operation
                    {
                        bool res_25011;
                        int32_t res_25012;
                        
                        if (x_25005) {
                            res_25011 = x_25005;
                            res_25012 = x_25006;
                        } else {
                            bool x_25013 = x_25008 && x_25008;
                            bool x_25014 = !x_25008;
                            bool y_25015 = x_25005 && x_25014;
                            bool res_25016 = x_25013 || y_25015;
                            int32_t res_25017;
                            
                            if (x_25008) {
                                res_25017 = x_25009;
                            } else {
                                res_25017 = x_25006;
                            }
                            res_25011 = res_25016;
                            res_25012 = res_25017;
                        }
                        
                        float res_25018 = x_25007 + x_25010;
                        
                        x_25005 = res_25011;
                        x_25006 = res_25012;
                        x_25007 = res_25018;
                    }
                    // write result of operation
                    {
                        ((__local
                          bool *) red_arr_mem_24986)[sext_i32_i64(local_tid_24982)] =
                            x_25005;
                        ((__local
                          int32_t *) red_arr_mem_24988)[sext_i32_i64(local_tid_24982)] =
                            x_25006;
                        ((__local
                          float *) red_arr_mem_24990)[sext_i32_i64(local_tid_24982)] =
                            x_25007;
                    }
                }
                skip_waves_25021 *= 2;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread saves the result in accumulator
            {
                if (local_tid_24982 == 0) {
                    x_acc_24999 = x_25005;
                    x_acc_25000 = x_25006;
                    x_acc_25001 = x_25007;
                }
            }
            // first thread keeps accumulator; others reset to neutral element
            {
                if (!(local_tid_24982 == 0)) {
                    x_acc_24999 = 0;
                    x_acc_25000 = -1;
                    x_acc_25001 = 0.0F;
                }
            }
        }
        x_22297 = x_acc_24999;
        x_22298 = x_acc_25000;
        x_22299 = x_acc_25001;
        if (groups_per_segment_24968 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_24982 == 0) {
                    ((__global bool *) mem_23886)[sext_i32_i64(gtid_22097)] =
                        x_acc_24999;
                    ((__global int32_t *) mem_23889)[sext_i32_i64(gtid_22097)] =
                        x_acc_25000;
                    ((__global float *) mem_23892)[sext_i32_i64(gtid_22097)] =
                        x_acc_25001;
                }
            }
        } else {
            int32_t old_counter_25022;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_24982 == 0) {
                    ((__global
                      bool *) group_res_arr_mem_24973)[sext_i32_i64(virt_group_id_24996) *
                                                       sext_i32_i64(segred_group_sizze_22291)] =
                        x_acc_24999;
                    ((__global
                      int32_t *) group_res_arr_mem_24975)[sext_i32_i64(virt_group_id_24996) *
                                                          sext_i32_i64(segred_group_sizze_22291)] =
                        x_acc_25000;
                    ((__global
                      float *) group_res_arr_mem_24977)[sext_i32_i64(virt_group_id_24996) *
                                                        sext_i32_i64(segred_group_sizze_22291)] =
                        x_acc_25001;
                    mem_fence_global();
                    old_counter_25022 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24979)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24997,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_24992)[0] =
                        old_counter_25022 == groups_per_segment_24968 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_25023;
            
            is_last_group_25023 = ((__local bool *) sync_arr_mem_24992)[0];
            if (is_last_group_25023) {
                if (local_tid_24982 == 0) {
                    old_counter_25022 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24979)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24997,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_24968));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_25024 =
                            sdiv_up32(groups_per_segment_24968,
                                      segred_group_sizze_22291);
                    
                    x_22297 = 0;
                    x_22298 = -1;
                    x_22299 = 0.0F;
                    for (int32_t i_25025 = 0; i_25025 < read_per_thread_25024;
                         i_25025++) {
                        int32_t group_res_id_25026 = local_tid_24982 *
                                read_per_thread_25024 + i_25025;
                        int32_t index_of_group_res_25027 =
                                flat_segment_id_24997 *
                                groups_per_segment_24968 + group_res_id_25026;
                        
                        if (slt32(group_res_id_25026,
                                  groups_per_segment_24968)) {
                            x_22300 = ((__global
                                        bool *) group_res_arr_mem_24973)[sext_i32_i64(index_of_group_res_25027) *
                                                                         sext_i32_i64(segred_group_sizze_22291)];
                            x_22301 = ((__global
                                        int32_t *) group_res_arr_mem_24975)[sext_i32_i64(index_of_group_res_25027) *
                                                                            sext_i32_i64(segred_group_sizze_22291)];
                            x_22302 = ((__global
                                        float *) group_res_arr_mem_24977)[sext_i32_i64(index_of_group_res_25027) *
                                                                          sext_i32_i64(segred_group_sizze_22291)];
                            
                            bool res_22303;
                            int32_t res_22304;
                            
                            if (x_22297) {
                                res_22303 = x_22297;
                                res_22304 = x_22298;
                            } else {
                                bool x_22305 = x_22300 && x_22300;
                                bool x_22306 = !x_22300;
                                bool y_22307 = x_22297 && x_22306;
                                bool res_22308 = x_22305 || y_22307;
                                int32_t res_22309;
                                
                                if (x_22300) {
                                    res_22309 = x_22301;
                                } else {
                                    res_22309 = x_22298;
                                }
                                res_22303 = res_22308;
                                res_22304 = res_22309;
                            }
                            
                            float res_22310 = x_22299 + x_22302;
                            
                            x_22297 = res_22303;
                            x_22298 = res_22304;
                            x_22299 = res_22310;
                        }
                    }
                }
                ((__local
                  bool *) red_arr_mem_24986)[sext_i32_i64(local_tid_24982)] =
                    x_22297;
                ((__local
                  int32_t *) red_arr_mem_24988)[sext_i32_i64(local_tid_24982)] =
                    x_22298;
                ((__local
                  float *) red_arr_mem_24990)[sext_i32_i64(local_tid_24982)] =
                    x_22299;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_25028;
                    int32_t skip_waves_25029;
                    bool x_25005;
                    int32_t x_25006;
                    float x_25007;
                    bool x_25008;
                    int32_t x_25009;
                    float x_25010;
                    
                    offset_25028 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_24982, segred_group_sizze_22291)) {
                            x_25005 = ((__local
                                        bool *) red_arr_mem_24986)[sext_i32_i64(local_tid_24982 +
                                                                   offset_25028)];
                            x_25006 = ((__local
                                        int32_t *) red_arr_mem_24988)[sext_i32_i64(local_tid_24982 +
                                                                      offset_25028)];
                            x_25007 = ((__local
                                        float *) red_arr_mem_24990)[sext_i32_i64(local_tid_24982 +
                                                                    offset_25028)];
                        }
                    }
                    offset_25028 = 1;
                    while (slt32(offset_25028, wave_sizze_24984)) {
                        if (slt32(local_tid_24982 + offset_25028,
                                  segred_group_sizze_22291) &&
                            ((local_tid_24982 - squot32(local_tid_24982,
                                                        wave_sizze_24984) *
                              wave_sizze_24984) & (2 * offset_25028 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_25008 = ((volatile __local
                                            bool *) red_arr_mem_24986)[sext_i32_i64(local_tid_24982 +
                                                                       offset_25028)];
                                x_25009 = ((volatile __local
                                            int32_t *) red_arr_mem_24988)[sext_i32_i64(local_tid_24982 +
                                                                          offset_25028)];
                                x_25010 = ((volatile __local
                                            float *) red_arr_mem_24990)[sext_i32_i64(local_tid_24982 +
                                                                        offset_25028)];
                            }
                            // apply reduction operation
                            {
                                bool res_25011;
                                int32_t res_25012;
                                
                                if (x_25005) {
                                    res_25011 = x_25005;
                                    res_25012 = x_25006;
                                } else {
                                    bool x_25013 = x_25008 && x_25008;
                                    bool x_25014 = !x_25008;
                                    bool y_25015 = x_25005 && x_25014;
                                    bool res_25016 = x_25013 || y_25015;
                                    int32_t res_25017;
                                    
                                    if (x_25008) {
                                        res_25017 = x_25009;
                                    } else {
                                        res_25017 = x_25006;
                                    }
                                    res_25011 = res_25016;
                                    res_25012 = res_25017;
                                }
                                
                                float res_25018 = x_25007 + x_25010;
                                
                                x_25005 = res_25011;
                                x_25006 = res_25012;
                                x_25007 = res_25018;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  bool *) red_arr_mem_24986)[sext_i32_i64(local_tid_24982)] =
                                    x_25005;
                                ((volatile __local
                                  int32_t *) red_arr_mem_24988)[sext_i32_i64(local_tid_24982)] =
                                    x_25006;
                                ((volatile __local
                                  float *) red_arr_mem_24990)[sext_i32_i64(local_tid_24982)] =
                                    x_25007;
                            }
                        }
                        offset_25028 *= 2;
                    }
                    skip_waves_25029 = 1;
                    while (slt32(skip_waves_25029,
                                 squot32(segred_group_sizze_22291 +
                                         wave_sizze_24984 - 1,
                                         wave_sizze_24984))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_25028 = skip_waves_25029 * wave_sizze_24984;
                        if (slt32(local_tid_24982 + offset_25028,
                                  segred_group_sizze_22291) &&
                            ((local_tid_24982 - squot32(local_tid_24982,
                                                        wave_sizze_24984) *
                              wave_sizze_24984) == 0 &&
                             (squot32(local_tid_24982, wave_sizze_24984) & (2 *
                                                                            skip_waves_25029 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_25008 = ((__local
                                            bool *) red_arr_mem_24986)[sext_i32_i64(local_tid_24982 +
                                                                       offset_25028)];
                                x_25009 = ((__local
                                            int32_t *) red_arr_mem_24988)[sext_i32_i64(local_tid_24982 +
                                                                          offset_25028)];
                                x_25010 = ((__local
                                            float *) red_arr_mem_24990)[sext_i32_i64(local_tid_24982 +
                                                                        offset_25028)];
                            }
                            // apply reduction operation
                            {
                                bool res_25011;
                                int32_t res_25012;
                                
                                if (x_25005) {
                                    res_25011 = x_25005;
                                    res_25012 = x_25006;
                                } else {
                                    bool x_25013 = x_25008 && x_25008;
                                    bool x_25014 = !x_25008;
                                    bool y_25015 = x_25005 && x_25014;
                                    bool res_25016 = x_25013 || y_25015;
                                    int32_t res_25017;
                                    
                                    if (x_25008) {
                                        res_25017 = x_25009;
                                    } else {
                                        res_25017 = x_25006;
                                    }
                                    res_25011 = res_25016;
                                    res_25012 = res_25017;
                                }
                                
                                float res_25018 = x_25007 + x_25010;
                                
                                x_25005 = res_25011;
                                x_25006 = res_25012;
                                x_25007 = res_25018;
                            }
                            // write result of operation
                            {
                                ((__local
                                  bool *) red_arr_mem_24986)[sext_i32_i64(local_tid_24982)] =
                                    x_25005;
                                ((__local
                                  int32_t *) red_arr_mem_24988)[sext_i32_i64(local_tid_24982)] =
                                    x_25006;
                                ((__local
                                  float *) red_arr_mem_24990)[sext_i32_i64(local_tid_24982)] =
                                    x_25007;
                            }
                        }
                        skip_waves_25029 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_24982 == 0) {
                            ((__global
                              bool *) mem_23886)[sext_i32_i64(gtid_22097)] =
                                x_25005;
                            ((__global
                              int32_t *) mem_23889)[sext_i32_i64(gtid_22097)] =
                                x_25006;
                            ((__global
                              float *) mem_23892)[sext_i32_i64(gtid_22097)] =
                                x_25007;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_22291
}
__kernel void mainzisegred_nonseg_21559(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_24754_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_24752_backing_aligned_1,
                                        int32_t m_18055,
                                        int32_t num_groups_21554, __global
                                        unsigned char *res_mem_23840, __global
                                        unsigned char *mem_23845, __global
                                        unsigned char *mainzicounter_mem_24742,
                                        __global
                                        unsigned char *group_res_arr_mem_24744,
                                        int32_t num_threads_24746)
{
    #define segred_group_sizze_21552 (mainzisegred_group_sizze_21551)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24754_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_24754_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_24752_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_24752_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24747;
    int32_t local_tid_24748;
    int32_t group_sizze_24751;
    int32_t wave_sizze_24750;
    int32_t group_tid_24749;
    
    global_tid_24747 = get_global_id(0);
    local_tid_24748 = get_local_id(0);
    group_sizze_24751 = get_local_size(0);
    wave_sizze_24750 = LOCKSTEP_WIDTH;
    group_tid_24749 = get_group_id(0);
    
    int32_t phys_tid_21559;
    
    phys_tid_21559 = global_tid_24747;
    
    __local char *sync_arr_mem_24752;
    
    sync_arr_mem_24752 = (__local char *) sync_arr_mem_24752_backing_0;
    
    __local char *red_arr_mem_24754;
    
    red_arr_mem_24754 = (__local char *) red_arr_mem_24754_backing_1;
    
    int32_t dummy_21557;
    
    dummy_21557 = 0;
    
    int32_t gtid_21558;
    
    gtid_21558 = 0;
    
    int32_t x_acc_24756;
    int32_t chunk_sizze_24757;
    
    chunk_sizze_24757 = smin32(sdiv_up32(m_18055, segred_group_sizze_21552 *
                                         num_groups_21554), sdiv_up32(m_18055 -
                                                                      phys_tid_21559,
                                                                      num_threads_24746));
    
    int32_t x_18382;
    int32_t x_18383;
    
    // neutral-initialise the accumulators
    {
        x_acc_24756 = 0;
    }
    for (int32_t i_24761 = 0; i_24761 < chunk_sizze_24757; i_24761++) {
        gtid_21558 = phys_tid_21559 + num_threads_24746 * i_24761;
        // apply map function
        {
            int32_t x_18385 = ((__global
                                int32_t *) res_mem_23840)[sext_i32_i64(gtid_21558)];
            
            // save map-out results
            { }
            // load accumulator
            {
                x_18382 = x_acc_24756;
            }
            // load new values
            {
                x_18383 = x_18385;
            }
            // apply reduction operator
            {
                int32_t res_18384 = smax32(x_18382, x_18383);
                
                // store in accumulator
                {
                    x_acc_24756 = res_18384;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_18382 = x_acc_24756;
        ((__local int32_t *) red_arr_mem_24754)[sext_i32_i64(local_tid_24748)] =
            x_18382;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_24762;
    int32_t skip_waves_24763;
    int32_t x_24758;
    int32_t x_24759;
    
    offset_24762 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_24748, segred_group_sizze_21552)) {
            x_24758 = ((__local
                        int32_t *) red_arr_mem_24754)[sext_i32_i64(local_tid_24748 +
                                                      offset_24762)];
        }
    }
    offset_24762 = 1;
    while (slt32(offset_24762, wave_sizze_24750)) {
        if (slt32(local_tid_24748 + offset_24762, segred_group_sizze_21552) &&
            ((local_tid_24748 - squot32(local_tid_24748, wave_sizze_24750) *
              wave_sizze_24750) & (2 * offset_24762 - 1)) == 0) {
            // read array element
            {
                x_24759 = ((volatile __local
                            int32_t *) red_arr_mem_24754)[sext_i32_i64(local_tid_24748 +
                                                          offset_24762)];
            }
            // apply reduction operation
            {
                int32_t res_24760 = smax32(x_24758, x_24759);
                
                x_24758 = res_24760;
            }
            // write result of operation
            {
                ((volatile __local
                  int32_t *) red_arr_mem_24754)[sext_i32_i64(local_tid_24748)] =
                    x_24758;
            }
        }
        offset_24762 *= 2;
    }
    skip_waves_24763 = 1;
    while (slt32(skip_waves_24763, squot32(segred_group_sizze_21552 +
                                           wave_sizze_24750 - 1,
                                           wave_sizze_24750))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_24762 = skip_waves_24763 * wave_sizze_24750;
        if (slt32(local_tid_24748 + offset_24762, segred_group_sizze_21552) &&
            ((local_tid_24748 - squot32(local_tid_24748, wave_sizze_24750) *
              wave_sizze_24750) == 0 && (squot32(local_tid_24748,
                                                 wave_sizze_24750) & (2 *
                                                                      skip_waves_24763 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_24759 = ((__local
                            int32_t *) red_arr_mem_24754)[sext_i32_i64(local_tid_24748 +
                                                          offset_24762)];
            }
            // apply reduction operation
            {
                int32_t res_24760 = smax32(x_24758, x_24759);
                
                x_24758 = res_24760;
            }
            // write result of operation
            {
                ((__local
                  int32_t *) red_arr_mem_24754)[sext_i32_i64(local_tid_24748)] =
                    x_24758;
            }
        }
        skip_waves_24763 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (local_tid_24748 == 0) {
            x_acc_24756 = x_24758;
        }
    }
    
    int32_t old_counter_24764;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_24748 == 0) {
            ((__global
              int32_t *) group_res_arr_mem_24744)[sext_i32_i64(group_tid_24749) *
                                                  sext_i32_i64(segred_group_sizze_21552)] =
                x_acc_24756;
            mem_fence_global();
            old_counter_24764 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_24742)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_24752)[0] = old_counter_24764 ==
                num_groups_21554 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_24765;
    
    is_last_group_24765 = ((__local bool *) sync_arr_mem_24752)[0];
    if (is_last_group_24765) {
        if (local_tid_24748 == 0) {
            old_counter_24764 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_24742)[0],
                                                      (int) (0 -
                                                             num_groups_21554));
        }
        // read in the per-group-results
        {
            int32_t read_per_thread_24766 = sdiv_up32(num_groups_21554,
                                                      segred_group_sizze_21552);
            
            x_18382 = 0;
            for (int32_t i_24767 = 0; i_24767 < read_per_thread_24766;
                 i_24767++) {
                int32_t group_res_id_24768 = local_tid_24748 *
                        read_per_thread_24766 + i_24767;
                int32_t index_of_group_res_24769 = group_res_id_24768;
                
                if (slt32(group_res_id_24768, num_groups_21554)) {
                    x_18383 = ((__global
                                int32_t *) group_res_arr_mem_24744)[sext_i32_i64(index_of_group_res_24769) *
                                                                    sext_i32_i64(segred_group_sizze_21552)];
                    
                    int32_t res_18384;
                    
                    res_18384 = smax32(x_18382, x_18383);
                    x_18382 = res_18384;
                }
            }
        }
        ((__local int32_t *) red_arr_mem_24754)[sext_i32_i64(local_tid_24748)] =
            x_18382;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_24770;
            int32_t skip_waves_24771;
            int32_t x_24758;
            int32_t x_24759;
            
            offset_24770 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_24748, segred_group_sizze_21552)) {
                    x_24758 = ((__local
                                int32_t *) red_arr_mem_24754)[sext_i32_i64(local_tid_24748 +
                                                              offset_24770)];
                }
            }
            offset_24770 = 1;
            while (slt32(offset_24770, wave_sizze_24750)) {
                if (slt32(local_tid_24748 + offset_24770,
                          segred_group_sizze_21552) && ((local_tid_24748 -
                                                         squot32(local_tid_24748,
                                                                 wave_sizze_24750) *
                                                         wave_sizze_24750) &
                                                        (2 * offset_24770 -
                                                         1)) == 0) {
                    // read array element
                    {
                        x_24759 = ((volatile __local
                                    int32_t *) red_arr_mem_24754)[sext_i32_i64(local_tid_24748 +
                                                                  offset_24770)];
                    }
                    // apply reduction operation
                    {
                        int32_t res_24760 = smax32(x_24758, x_24759);
                        
                        x_24758 = res_24760;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          int32_t *) red_arr_mem_24754)[sext_i32_i64(local_tid_24748)] =
                            x_24758;
                    }
                }
                offset_24770 *= 2;
            }
            skip_waves_24771 = 1;
            while (slt32(skip_waves_24771, squot32(segred_group_sizze_21552 +
                                                   wave_sizze_24750 - 1,
                                                   wave_sizze_24750))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_24770 = skip_waves_24771 * wave_sizze_24750;
                if (slt32(local_tid_24748 + offset_24770,
                          segred_group_sizze_21552) && ((local_tid_24748 -
                                                         squot32(local_tid_24748,
                                                                 wave_sizze_24750) *
                                                         wave_sizze_24750) ==
                                                        0 &&
                                                        (squot32(local_tid_24748,
                                                                 wave_sizze_24750) &
                                                         (2 * skip_waves_24771 -
                                                          1)) == 0)) {
                    // read array element
                    {
                        x_24759 = ((__local
                                    int32_t *) red_arr_mem_24754)[sext_i32_i64(local_tid_24748 +
                                                                  offset_24770)];
                    }
                    // apply reduction operation
                    {
                        int32_t res_24760 = smax32(x_24758, x_24759);
                        
                        x_24758 = res_24760;
                    }
                    // write result of operation
                    {
                        ((__local
                          int32_t *) red_arr_mem_24754)[sext_i32_i64(local_tid_24748)] =
                            x_24758;
                    }
                }
                skip_waves_24771 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_24748 == 0) {
                    ((__global int32_t *) mem_23845)[0] = x_24758;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21552
}
__kernel void mainzisegred_small_19329(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_24119_backing_aligned_0,
                                       int32_t N_18054, int32_t m_18055,
                                       int32_t N_18056, int32_t n_18059,
                                       int32_t k2p2zq_18071,
                                       int32_t num_groups_19492, __global
                                       unsigned char *images_mem_23189, __global
                                       unsigned char *binop_p_mem_23202,
                                       __global unsigned char *mem_23307,
                                       __global unsigned char *mem_23315,
                                       int32_t segment_sizze_nonzzero_24112)
{
    #define segred_group_sizze_19491 (mainzisegred_group_sizze_19323)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24119_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24119_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24114;
    int32_t local_tid_24115;
    int32_t group_sizze_24118;
    int32_t wave_sizze_24117;
    int32_t group_tid_24116;
    
    global_tid_24114 = get_global_id(0);
    local_tid_24115 = get_local_id(0);
    group_sizze_24118 = get_local_size(0);
    wave_sizze_24117 = LOCKSTEP_WIDTH;
    group_tid_24116 = get_group_id(0);
    
    int32_t phys_tid_19329;
    
    phys_tid_19329 = global_tid_24114;
    
    __local char *red_arr_mem_24119;
    
    red_arr_mem_24119 = (__local char *) red_arr_mem_24119_backing_0;
    
    int32_t phys_group_id_24121;
    
    phys_group_id_24121 = get_group_id(0);
    for (int32_t i_24122 = 0; i_24122 < sdiv_up32(sdiv_up32(m_18055 *
                                                            k2p2zq_18071 *
                                                            k2p2zq_18071,
                                                            squot32(segred_group_sizze_19491,
                                                                    segment_sizze_nonzzero_24112)) -
                                                  phys_group_id_24121,
                                                  num_groups_19492);
         i_24122++) {
        int32_t virt_group_id_24123 = phys_group_id_24121 + i_24122 *
                num_groups_19492;
        int32_t gtid_19312 = squot32(squot32(local_tid_24115,
                                             segment_sizze_nonzzero_24112) +
                                     virt_group_id_24123 *
                                     squot32(segred_group_sizze_19491,
                                             segment_sizze_nonzzero_24112),
                                     k2p2zq_18071 * k2p2zq_18071);
        int32_t gtid_19313 = squot32(squot32(local_tid_24115,
                                             segment_sizze_nonzzero_24112) +
                                     virt_group_id_24123 *
                                     squot32(segred_group_sizze_19491,
                                             segment_sizze_nonzzero_24112) -
                                     squot32(squot32(local_tid_24115,
                                                     segment_sizze_nonzzero_24112) +
                                             virt_group_id_24123 *
                                             squot32(segred_group_sizze_19491,
                                                     segment_sizze_nonzzero_24112),
                                             k2p2zq_18071 * k2p2zq_18071) *
                                     (k2p2zq_18071 * k2p2zq_18071),
                                     k2p2zq_18071);
        int32_t gtid_19314 = squot32(local_tid_24115,
                                     segment_sizze_nonzzero_24112) +
                virt_group_id_24123 * squot32(segred_group_sizze_19491,
                                              segment_sizze_nonzzero_24112) -
                squot32(squot32(local_tid_24115, segment_sizze_nonzzero_24112) +
                        virt_group_id_24123 * squot32(segred_group_sizze_19491,
                                                      segment_sizze_nonzzero_24112),
                        k2p2zq_18071 * k2p2zq_18071) * (k2p2zq_18071 *
                                                        k2p2zq_18071) -
                squot32(squot32(local_tid_24115, segment_sizze_nonzzero_24112) +
                        virt_group_id_24123 * squot32(segred_group_sizze_19491,
                                                      segment_sizze_nonzzero_24112) -
                        squot32(squot32(local_tid_24115,
                                        segment_sizze_nonzzero_24112) +
                                virt_group_id_24123 *
                                squot32(segred_group_sizze_19491,
                                        segment_sizze_nonzzero_24112),
                                k2p2zq_18071 * k2p2zq_18071) * (k2p2zq_18071 *
                                                                k2p2zq_18071),
                        k2p2zq_18071) * k2p2zq_18071;
        int32_t gtid_19328 = srem32(local_tid_24115, n_18059);
        
        // apply map function if in bounds
        {
            if (slt32(0, n_18059) && (((slt32(gtid_19312, m_18055) &&
                                        slt32(gtid_19313, k2p2zq_18071)) &&
                                       slt32(gtid_19314, k2p2zq_18071)) &&
                                      slt32(local_tid_24115, n_18059 *
                                            squot32(segred_group_sizze_19491,
                                                    segment_sizze_nonzzero_24112)))) {
                float x_19501 = ((__global
                                  float *) images_mem_23189)[sext_i32_i64(gtid_19312) *
                                                             sext_i32_i64(N_18056) +
                                                             sext_i32_i64(gtid_19328)];
                float x_19502 = ((__global
                                  float *) binop_p_mem_23202)[sext_i32_i64(gtid_19313) *
                                                              sext_i32_i64(N_18054) +
                                                              sext_i32_i64(gtid_19328)];
                float x_19503 = ((__global
                                  float *) mem_23307)[sext_i32_i64(gtid_19314) *
                                                      sext_i32_i64(N_18054) +
                                                      sext_i32_i64(gtid_19328)];
                float x_19504 = x_19502 * x_19503;
                bool res_19505;
                
                res_19505 = futrts_isnan32(x_19501);
                
                float y_19506;
                
                if (res_19505) {
                    y_19506 = 0.0F;
                } else {
                    y_19506 = 1.0F;
                }
                
                float res_19507 = x_19504 * y_19506;
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_24119)[sext_i32_i64(local_tid_24115)] =
                        res_19507;
                }
            } else {
                ((__local
                  float *) red_arr_mem_24119)[sext_i32_i64(local_tid_24115)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, n_18059)) {
            // perform segmented scan to imitate reduction
            {
                float x_19495;
                float x_19496;
                float x_24124;
                float x_24125;
                int32_t skip_threads_24127;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_24115, n_18059 *
                              squot32(segred_group_sizze_19491,
                                      segment_sizze_nonzzero_24112))) {
                        x_19496 = ((volatile __local
                                    float *) red_arr_mem_24119)[sext_i32_i64(local_tid_24115)];
                        if ((local_tid_24115 - squot32(local_tid_24115, 32) *
                             32) == 0) {
                            x_19495 = x_19496;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24127 = 1;
                    while (slt32(skip_threads_24127, 32)) {
                        if (sle32(skip_threads_24127, local_tid_24115 -
                                  squot32(local_tid_24115, 32) * 32) &&
                            slt32(local_tid_24115, n_18059 *
                                  squot32(segred_group_sizze_19491,
                                          segment_sizze_nonzzero_24112))) {
                            // read operands
                            {
                                x_19495 = ((volatile __local
                                            float *) red_arr_mem_24119)[sext_i32_i64(local_tid_24115 -
                                                                        skip_threads_24127)];
                            }
                            // perform operation
                            {
                                bool inactive_24128 =
                                     slt32(srem32(local_tid_24115, n_18059),
                                           local_tid_24115 - (local_tid_24115 -
                                                              skip_threads_24127));
                                
                                if (inactive_24128) {
                                    x_19495 = x_19496;
                                }
                                if (!inactive_24128) {
                                    float res_19497 = x_19495 + x_19496;
                                    
                                    x_19495 = res_19497;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24117, skip_threads_24127)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24127, local_tid_24115 -
                                  squot32(local_tid_24115, 32) * 32) &&
                            slt32(local_tid_24115, n_18059 *
                                  squot32(segred_group_sizze_19491,
                                          segment_sizze_nonzzero_24112))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24119)[sext_i32_i64(local_tid_24115)] =
                                    x_19495;
                                x_19496 = x_19495;
                            }
                        }
                        if (sle32(wave_sizze_24117, skip_threads_24127)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24127 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_24115 - squot32(local_tid_24115, 32) * 32) ==
                        31 && slt32(local_tid_24115, n_18059 *
                                    squot32(segred_group_sizze_19491,
                                            segment_sizze_nonzzero_24112))) {
                        ((volatile __local
                          float *) red_arr_mem_24119)[sext_i32_i64(squot32(local_tid_24115,
                                                                           32))] =
                            x_19495;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_24129;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_24115, 32) == 0 &&
                            slt32(local_tid_24115, n_18059 *
                                  squot32(segred_group_sizze_19491,
                                          segment_sizze_nonzzero_24112))) {
                            x_24125 = ((volatile __local
                                        float *) red_arr_mem_24119)[sext_i32_i64(local_tid_24115)];
                            if ((local_tid_24115 - squot32(local_tid_24115,
                                                           32) * 32) == 0) {
                                x_24124 = x_24125;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_24129 = 1;
                        while (slt32(skip_threads_24129, 32)) {
                            if (sle32(skip_threads_24129, local_tid_24115 -
                                      squot32(local_tid_24115, 32) * 32) &&
                                (squot32(local_tid_24115, 32) == 0 &&
                                 slt32(local_tid_24115, n_18059 *
                                       squot32(segred_group_sizze_19491,
                                               segment_sizze_nonzzero_24112)))) {
                                // read operands
                                {
                                    x_24124 = ((volatile __local
                                                float *) red_arr_mem_24119)[sext_i32_i64(local_tid_24115 -
                                                                            skip_threads_24129)];
                                }
                                // perform operation
                                {
                                    bool inactive_24130 =
                                         slt32(srem32(local_tid_24115 * 32 +
                                                      32 - 1, n_18059),
                                               local_tid_24115 * 32 + 32 - 1 -
                                               ((local_tid_24115 -
                                                 skip_threads_24129) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_24130) {
                                        x_24124 = x_24125;
                                    }
                                    if (!inactive_24130) {
                                        float res_24126 = x_24124 + x_24125;
                                        
                                        x_24124 = res_24126;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_24117, skip_threads_24129)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_24129, local_tid_24115 -
                                      squot32(local_tid_24115, 32) * 32) &&
                                (squot32(local_tid_24115, 32) == 0 &&
                                 slt32(local_tid_24115, n_18059 *
                                       squot32(segred_group_sizze_19491,
                                               segment_sizze_nonzzero_24112)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_24119)[sext_i32_i64(local_tid_24115)] =
                                        x_24124;
                                    x_24125 = x_24124;
                                }
                            }
                            if (sle32(wave_sizze_24117, skip_threads_24129)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_24129 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_24115, 32) == 0 ||
                          !slt32(local_tid_24115, n_18059 *
                                 squot32(segred_group_sizze_19491,
                                         segment_sizze_nonzzero_24112)))) {
                        // read operands
                        {
                            x_19496 = x_19495;
                            x_19495 = ((__local
                                        float *) red_arr_mem_24119)[sext_i32_i64(squot32(local_tid_24115,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_24131 = slt32(srem32(local_tid_24115,
                                                               n_18059),
                                                        local_tid_24115 -
                                                        (squot32(local_tid_24115,
                                                                 32) * 32 - 1));
                            
                            if (inactive_24131) {
                                x_19495 = x_19496;
                            }
                            if (!inactive_24131) {
                                float res_19497 = x_19495 + x_19496;
                                
                                x_19495 = res_19497;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_24119)[sext_i32_i64(local_tid_24115)] =
                                x_19495;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_24115, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_24119)[sext_i32_i64(local_tid_24115)] =
                            x_19496;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_24123 * squot32(segred_group_sizze_19491,
                                                    segment_sizze_nonzzero_24112) +
                      local_tid_24115, m_18055 * k2p2zq_18071 * k2p2zq_18071) &&
                slt32(local_tid_24115, squot32(segred_group_sizze_19491,
                                               segment_sizze_nonzzero_24112))) {
                ((__global
                  float *) mem_23315)[sext_i32_i64(squot32(virt_group_id_24123 *
                                                           squot32(segred_group_sizze_19491,
                                                                   segment_sizze_nonzzero_24112) +
                                                           local_tid_24115,
                                                           k2p2zq_18071 *
                                                           k2p2zq_18071)) *
                                      sext_i32_i64(k2p2zq_18071 *
                                      k2p2zq_18071) +
                                      sext_i32_i64(squot32(virt_group_id_24123 *
                                                           squot32(segred_group_sizze_19491,
                                                                   segment_sizze_nonzzero_24112) +
                                                           local_tid_24115 -
                                                           squot32(virt_group_id_24123 *
                                                                   squot32(segred_group_sizze_19491,
                                                                           segment_sizze_nonzzero_24112) +
                                                                   local_tid_24115,
                                                                   k2p2zq_18071 *
                                                                   k2p2zq_18071) *
                                                           (k2p2zq_18071 *
                                                            k2p2zq_18071),
                                                           k2p2zq_18071)) *
                                      sext_i32_i64(k2p2zq_18071) +
                                      sext_i32_i64(virt_group_id_24123 *
                                      squot32(segred_group_sizze_19491,
                                              segment_sizze_nonzzero_24112) +
                                      local_tid_24115 -
                                      squot32(virt_group_id_24123 *
                                              squot32(segred_group_sizze_19491,
                                                      segment_sizze_nonzzero_24112) +
                                              local_tid_24115, k2p2zq_18071 *
                                              k2p2zq_18071) * (k2p2zq_18071 *
                                                               k2p2zq_18071) -
                                      squot32(virt_group_id_24123 *
                                              squot32(segred_group_sizze_19491,
                                                      segment_sizze_nonzzero_24112) +
                                              local_tid_24115 -
                                              squot32(virt_group_id_24123 *
                                                      squot32(segred_group_sizze_19491,
                                                              segment_sizze_nonzzero_24112) +
                                                      local_tid_24115,
                                                      k2p2zq_18071 *
                                                      k2p2zq_18071) *
                                              (k2p2zq_18071 * k2p2zq_18071),
                                              k2p2zq_18071) * k2p2zq_18071)] =
                    ((__local
                      float *) red_arr_mem_24119)[sext_i32_i64((local_tid_24115 +
                                                                1) *
                                                  segment_sizze_nonzzero_24112 -
                                                  1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_19491
}
__kernel void mainzisegred_small_20577(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_24272_backing_aligned_0,
                                       int32_t N_18054, int32_t m_18055,
                                       int32_t N_18056, int32_t n_18059,
                                       int32_t k2p2zq_18071,
                                       int32_t num_groups_20640, __global
                                       unsigned char *images_mem_23189, __global
                                       unsigned char *binop_p_mem_23202,
                                       __global unsigned char *mem_23544,
                                       int32_t segment_sizze_nonzzero_24265)
{
    #define segred_group_sizze_20639 (mainzisegred_group_sizze_20571)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24272_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24272_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24267;
    int32_t local_tid_24268;
    int32_t group_sizze_24271;
    int32_t wave_sizze_24270;
    int32_t group_tid_24269;
    
    global_tid_24267 = get_global_id(0);
    local_tid_24268 = get_local_id(0);
    group_sizze_24271 = get_local_size(0);
    wave_sizze_24270 = LOCKSTEP_WIDTH;
    group_tid_24269 = get_group_id(0);
    
    int32_t phys_tid_20577;
    
    phys_tid_20577 = global_tid_24267;
    
    __local char *red_arr_mem_24272;
    
    red_arr_mem_24272 = (__local char *) red_arr_mem_24272_backing_0;
    
    int32_t phys_group_id_24274;
    
    phys_group_id_24274 = get_group_id(0);
    for (int32_t i_24275 = 0; i_24275 < sdiv_up32(sdiv_up32(m_18055 *
                                                            k2p2zq_18071,
                                                            squot32(segred_group_sizze_20639,
                                                                    segment_sizze_nonzzero_24265)) -
                                                  phys_group_id_24274,
                                                  num_groups_20640);
         i_24275++) {
        int32_t virt_group_id_24276 = phys_group_id_24274 + i_24275 *
                num_groups_20640;
        int32_t gtid_20563 = squot32(squot32(local_tid_24268,
                                             segment_sizze_nonzzero_24265) +
                                     virt_group_id_24276 *
                                     squot32(segred_group_sizze_20639,
                                             segment_sizze_nonzzero_24265),
                                     k2p2zq_18071);
        int32_t gtid_20564 = squot32(local_tid_24268,
                                     segment_sizze_nonzzero_24265) +
                virt_group_id_24276 * squot32(segred_group_sizze_20639,
                                              segment_sizze_nonzzero_24265) -
                squot32(squot32(local_tid_24268, segment_sizze_nonzzero_24265) +
                        virt_group_id_24276 * squot32(segred_group_sizze_20639,
                                                      segment_sizze_nonzzero_24265),
                        k2p2zq_18071) * k2p2zq_18071;
        int32_t gtid_20576 = srem32(local_tid_24268, n_18059);
        
        // apply map function if in bounds
        {
            if (slt32(0, n_18059) && ((slt32(gtid_20563, m_18055) &&
                                       slt32(gtid_20564, k2p2zq_18071)) &&
                                      slt32(local_tid_24268, n_18059 *
                                            squot32(segred_group_sizze_20639,
                                                    segment_sizze_nonzzero_24265)))) {
                float x_20649 = ((__global
                                  float *) images_mem_23189)[sext_i32_i64(gtid_20563) *
                                                             sext_i32_i64(N_18056) +
                                                             sext_i32_i64(gtid_20576)];
                bool res_20650;
                
                res_20650 = futrts_isnan32(x_20649);
                
                float res_20651;
                
                if (res_20650) {
                    res_20651 = 0.0F;
                } else {
                    float x_20648 = ((__global
                                      float *) binop_p_mem_23202)[sext_i32_i64(gtid_20564) *
                                                                  sext_i32_i64(N_18054) +
                                                                  sext_i32_i64(gtid_20576)];
                    float res_20652 = x_20648 * x_20649;
                    
                    res_20651 = res_20652;
                }
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_24272)[sext_i32_i64(local_tid_24268)] =
                        res_20651;
                }
            } else {
                ((__local
                  float *) red_arr_mem_24272)[sext_i32_i64(local_tid_24268)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, n_18059)) {
            // perform segmented scan to imitate reduction
            {
                float x_20643;
                float x_20644;
                float x_24277;
                float x_24278;
                int32_t skip_threads_24280;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_24268, n_18059 *
                              squot32(segred_group_sizze_20639,
                                      segment_sizze_nonzzero_24265))) {
                        x_20644 = ((volatile __local
                                    float *) red_arr_mem_24272)[sext_i32_i64(local_tid_24268)];
                        if ((local_tid_24268 - squot32(local_tid_24268, 32) *
                             32) == 0) {
                            x_20643 = x_20644;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24280 = 1;
                    while (slt32(skip_threads_24280, 32)) {
                        if (sle32(skip_threads_24280, local_tid_24268 -
                                  squot32(local_tid_24268, 32) * 32) &&
                            slt32(local_tid_24268, n_18059 *
                                  squot32(segred_group_sizze_20639,
                                          segment_sizze_nonzzero_24265))) {
                            // read operands
                            {
                                x_20643 = ((volatile __local
                                            float *) red_arr_mem_24272)[sext_i32_i64(local_tid_24268 -
                                                                        skip_threads_24280)];
                            }
                            // perform operation
                            {
                                bool inactive_24281 =
                                     slt32(srem32(local_tid_24268, n_18059),
                                           local_tid_24268 - (local_tid_24268 -
                                                              skip_threads_24280));
                                
                                if (inactive_24281) {
                                    x_20643 = x_20644;
                                }
                                if (!inactive_24281) {
                                    float res_20645 = x_20643 + x_20644;
                                    
                                    x_20643 = res_20645;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24270, skip_threads_24280)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24280, local_tid_24268 -
                                  squot32(local_tid_24268, 32) * 32) &&
                            slt32(local_tid_24268, n_18059 *
                                  squot32(segred_group_sizze_20639,
                                          segment_sizze_nonzzero_24265))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24272)[sext_i32_i64(local_tid_24268)] =
                                    x_20643;
                                x_20644 = x_20643;
                            }
                        }
                        if (sle32(wave_sizze_24270, skip_threads_24280)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24280 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_24268 - squot32(local_tid_24268, 32) * 32) ==
                        31 && slt32(local_tid_24268, n_18059 *
                                    squot32(segred_group_sizze_20639,
                                            segment_sizze_nonzzero_24265))) {
                        ((volatile __local
                          float *) red_arr_mem_24272)[sext_i32_i64(squot32(local_tid_24268,
                                                                           32))] =
                            x_20643;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_24282;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_24268, 32) == 0 &&
                            slt32(local_tid_24268, n_18059 *
                                  squot32(segred_group_sizze_20639,
                                          segment_sizze_nonzzero_24265))) {
                            x_24278 = ((volatile __local
                                        float *) red_arr_mem_24272)[sext_i32_i64(local_tid_24268)];
                            if ((local_tid_24268 - squot32(local_tid_24268,
                                                           32) * 32) == 0) {
                                x_24277 = x_24278;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_24282 = 1;
                        while (slt32(skip_threads_24282, 32)) {
                            if (sle32(skip_threads_24282, local_tid_24268 -
                                      squot32(local_tid_24268, 32) * 32) &&
                                (squot32(local_tid_24268, 32) == 0 &&
                                 slt32(local_tid_24268, n_18059 *
                                       squot32(segred_group_sizze_20639,
                                               segment_sizze_nonzzero_24265)))) {
                                // read operands
                                {
                                    x_24277 = ((volatile __local
                                                float *) red_arr_mem_24272)[sext_i32_i64(local_tid_24268 -
                                                                            skip_threads_24282)];
                                }
                                // perform operation
                                {
                                    bool inactive_24283 =
                                         slt32(srem32(local_tid_24268 * 32 +
                                                      32 - 1, n_18059),
                                               local_tid_24268 * 32 + 32 - 1 -
                                               ((local_tid_24268 -
                                                 skip_threads_24282) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_24283) {
                                        x_24277 = x_24278;
                                    }
                                    if (!inactive_24283) {
                                        float res_24279 = x_24277 + x_24278;
                                        
                                        x_24277 = res_24279;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_24270, skip_threads_24282)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_24282, local_tid_24268 -
                                      squot32(local_tid_24268, 32) * 32) &&
                                (squot32(local_tid_24268, 32) == 0 &&
                                 slt32(local_tid_24268, n_18059 *
                                       squot32(segred_group_sizze_20639,
                                               segment_sizze_nonzzero_24265)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_24272)[sext_i32_i64(local_tid_24268)] =
                                        x_24277;
                                    x_24278 = x_24277;
                                }
                            }
                            if (sle32(wave_sizze_24270, skip_threads_24282)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_24282 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_24268, 32) == 0 ||
                          !slt32(local_tid_24268, n_18059 *
                                 squot32(segred_group_sizze_20639,
                                         segment_sizze_nonzzero_24265)))) {
                        // read operands
                        {
                            x_20644 = x_20643;
                            x_20643 = ((__local
                                        float *) red_arr_mem_24272)[sext_i32_i64(squot32(local_tid_24268,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_24284 = slt32(srem32(local_tid_24268,
                                                               n_18059),
                                                        local_tid_24268 -
                                                        (squot32(local_tid_24268,
                                                                 32) * 32 - 1));
                            
                            if (inactive_24284) {
                                x_20643 = x_20644;
                            }
                            if (!inactive_24284) {
                                float res_20645 = x_20643 + x_20644;
                                
                                x_20643 = res_20645;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_24272)[sext_i32_i64(local_tid_24268)] =
                                x_20643;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_24268, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_24272)[sext_i32_i64(local_tid_24268)] =
                            x_20644;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_24276 * squot32(segred_group_sizze_20639,
                                                    segment_sizze_nonzzero_24265) +
                      local_tid_24268, m_18055 * k2p2zq_18071) &&
                slt32(local_tid_24268, squot32(segred_group_sizze_20639,
                                               segment_sizze_nonzzero_24265))) {
                ((__global
                  float *) mem_23544)[sext_i32_i64(squot32(virt_group_id_24276 *
                                                           squot32(segred_group_sizze_20639,
                                                                   segment_sizze_nonzzero_24265) +
                                                           local_tid_24268,
                                                           k2p2zq_18071)) *
                                      sext_i32_i64(k2p2zq_18071) +
                                      sext_i32_i64(virt_group_id_24276 *
                                      squot32(segred_group_sizze_20639,
                                              segment_sizze_nonzzero_24265) +
                                      local_tid_24268 -
                                      squot32(virt_group_id_24276 *
                                              squot32(segred_group_sizze_20639,
                                                      segment_sizze_nonzzero_24265) +
                                              local_tid_24268, k2p2zq_18071) *
                                      k2p2zq_18071)] = ((__local
                                                         float *) red_arr_mem_24272)[sext_i32_i64((local_tid_24268 +
                                                                                                   1) *
                                                                                     segment_sizze_nonzzero_24265 -
                                                                                     1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_20639
}
__kernel void mainzisegred_small_20733(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_24352_backing_aligned_0,
                                       int32_t m_18055, int32_t k2p2zq_18071,
                                       int32_t num_groups_20792, __global
                                       unsigned char *res_mem_23436, __global
                                       unsigned char *res_mem_23552, __global
                                       unsigned char *mem_23604,
                                       int32_t segment_sizze_nonzzero_24345)
{
    #define segred_group_sizze_20791 (mainzisegred_group_sizze_20727)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24352_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24352_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24347;
    int32_t local_tid_24348;
    int32_t group_sizze_24351;
    int32_t wave_sizze_24350;
    int32_t group_tid_24349;
    
    global_tid_24347 = get_global_id(0);
    local_tid_24348 = get_local_id(0);
    group_sizze_24351 = get_local_size(0);
    wave_sizze_24350 = LOCKSTEP_WIDTH;
    group_tid_24349 = get_group_id(0);
    
    int32_t phys_tid_20733;
    
    phys_tid_20733 = global_tid_24347;
    
    __local char *red_arr_mem_24352;
    
    red_arr_mem_24352 = (__local char *) red_arr_mem_24352_backing_0;
    
    int32_t phys_group_id_24354;
    
    phys_group_id_24354 = get_group_id(0);
    for (int32_t i_24355 = 0; i_24355 < sdiv_up32(sdiv_up32(m_18055 *
                                                            k2p2zq_18071,
                                                            squot32(segred_group_sizze_20791,
                                                                    segment_sizze_nonzzero_24345)) -
                                                  phys_group_id_24354,
                                                  num_groups_20792);
         i_24355++) {
        int32_t virt_group_id_24356 = phys_group_id_24354 + i_24355 *
                num_groups_20792;
        int32_t gtid_20719 = squot32(squot32(local_tid_24348,
                                             segment_sizze_nonzzero_24345) +
                                     virt_group_id_24356 *
                                     squot32(segred_group_sizze_20791,
                                             segment_sizze_nonzzero_24345),
                                     k2p2zq_18071);
        int32_t gtid_20720 = squot32(local_tid_24348,
                                     segment_sizze_nonzzero_24345) +
                virt_group_id_24356 * squot32(segred_group_sizze_20791,
                                              segment_sizze_nonzzero_24345) -
                squot32(squot32(local_tid_24348, segment_sizze_nonzzero_24345) +
                        virt_group_id_24356 * squot32(segred_group_sizze_20791,
                                                      segment_sizze_nonzzero_24345),
                        k2p2zq_18071) * k2p2zq_18071;
        int32_t gtid_20732 = srem32(local_tid_24348, k2p2zq_18071);
        
        // apply map function if in bounds
        {
            if (slt32(0, k2p2zq_18071) && ((slt32(gtid_20719, m_18055) &&
                                            slt32(gtid_20720, k2p2zq_18071)) &&
                                           slt32(local_tid_24348, k2p2zq_18071 *
                                                 squot32(segred_group_sizze_20791,
                                                         segment_sizze_nonzzero_24345)))) {
                float x_20801 = ((__global
                                  float *) res_mem_23552)[sext_i32_i64(gtid_20719) *
                                                          sext_i32_i64(k2p2zq_18071) +
                                                          sext_i32_i64(gtid_20732)];
                float x_20802 = ((__global
                                  float *) res_mem_23436)[sext_i32_i64(gtid_20719) *
                                                          sext_i32_i64(k2p2zq_18071 *
                                                          k2p2zq_18071) +
                                                          sext_i32_i64(gtid_20720) *
                                                          sext_i32_i64(k2p2zq_18071) +
                                                          sext_i32_i64(gtid_20732)];
                float res_20803 = x_20801 * x_20802;
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_24352)[sext_i32_i64(local_tid_24348)] =
                        res_20803;
                }
            } else {
                ((__local
                  float *) red_arr_mem_24352)[sext_i32_i64(local_tid_24348)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, k2p2zq_18071)) {
            // perform segmented scan to imitate reduction
            {
                float x_20795;
                float x_20796;
                float x_24357;
                float x_24358;
                int32_t skip_threads_24360;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_24348, k2p2zq_18071 *
                              squot32(segred_group_sizze_20791,
                                      segment_sizze_nonzzero_24345))) {
                        x_20796 = ((volatile __local
                                    float *) red_arr_mem_24352)[sext_i32_i64(local_tid_24348)];
                        if ((local_tid_24348 - squot32(local_tid_24348, 32) *
                             32) == 0) {
                            x_20795 = x_20796;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24360 = 1;
                    while (slt32(skip_threads_24360, 32)) {
                        if (sle32(skip_threads_24360, local_tid_24348 -
                                  squot32(local_tid_24348, 32) * 32) &&
                            slt32(local_tid_24348, k2p2zq_18071 *
                                  squot32(segred_group_sizze_20791,
                                          segment_sizze_nonzzero_24345))) {
                            // read operands
                            {
                                x_20795 = ((volatile __local
                                            float *) red_arr_mem_24352)[sext_i32_i64(local_tid_24348 -
                                                                        skip_threads_24360)];
                            }
                            // perform operation
                            {
                                bool inactive_24361 =
                                     slt32(srem32(local_tid_24348,
                                                  k2p2zq_18071),
                                           local_tid_24348 - (local_tid_24348 -
                                                              skip_threads_24360));
                                
                                if (inactive_24361) {
                                    x_20795 = x_20796;
                                }
                                if (!inactive_24361) {
                                    float res_20797 = x_20795 + x_20796;
                                    
                                    x_20795 = res_20797;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24350, skip_threads_24360)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24360, local_tid_24348 -
                                  squot32(local_tid_24348, 32) * 32) &&
                            slt32(local_tid_24348, k2p2zq_18071 *
                                  squot32(segred_group_sizze_20791,
                                          segment_sizze_nonzzero_24345))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24352)[sext_i32_i64(local_tid_24348)] =
                                    x_20795;
                                x_20796 = x_20795;
                            }
                        }
                        if (sle32(wave_sizze_24350, skip_threads_24360)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24360 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_24348 - squot32(local_tid_24348, 32) * 32) ==
                        31 && slt32(local_tid_24348, k2p2zq_18071 *
                                    squot32(segred_group_sizze_20791,
                                            segment_sizze_nonzzero_24345))) {
                        ((volatile __local
                          float *) red_arr_mem_24352)[sext_i32_i64(squot32(local_tid_24348,
                                                                           32))] =
                            x_20795;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_24362;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_24348, 32) == 0 &&
                            slt32(local_tid_24348, k2p2zq_18071 *
                                  squot32(segred_group_sizze_20791,
                                          segment_sizze_nonzzero_24345))) {
                            x_24358 = ((volatile __local
                                        float *) red_arr_mem_24352)[sext_i32_i64(local_tid_24348)];
                            if ((local_tid_24348 - squot32(local_tid_24348,
                                                           32) * 32) == 0) {
                                x_24357 = x_24358;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_24362 = 1;
                        while (slt32(skip_threads_24362, 32)) {
                            if (sle32(skip_threads_24362, local_tid_24348 -
                                      squot32(local_tid_24348, 32) * 32) &&
                                (squot32(local_tid_24348, 32) == 0 &&
                                 slt32(local_tid_24348, k2p2zq_18071 *
                                       squot32(segred_group_sizze_20791,
                                               segment_sizze_nonzzero_24345)))) {
                                // read operands
                                {
                                    x_24357 = ((volatile __local
                                                float *) red_arr_mem_24352)[sext_i32_i64(local_tid_24348 -
                                                                            skip_threads_24362)];
                                }
                                // perform operation
                                {
                                    bool inactive_24363 =
                                         slt32(srem32(local_tid_24348 * 32 +
                                                      32 - 1, k2p2zq_18071),
                                               local_tid_24348 * 32 + 32 - 1 -
                                               ((local_tid_24348 -
                                                 skip_threads_24362) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_24363) {
                                        x_24357 = x_24358;
                                    }
                                    if (!inactive_24363) {
                                        float res_24359 = x_24357 + x_24358;
                                        
                                        x_24357 = res_24359;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_24350, skip_threads_24362)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_24362, local_tid_24348 -
                                      squot32(local_tid_24348, 32) * 32) &&
                                (squot32(local_tid_24348, 32) == 0 &&
                                 slt32(local_tid_24348, k2p2zq_18071 *
                                       squot32(segred_group_sizze_20791,
                                               segment_sizze_nonzzero_24345)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_24352)[sext_i32_i64(local_tid_24348)] =
                                        x_24357;
                                    x_24358 = x_24357;
                                }
                            }
                            if (sle32(wave_sizze_24350, skip_threads_24362)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_24362 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_24348, 32) == 0 ||
                          !slt32(local_tid_24348, k2p2zq_18071 *
                                 squot32(segred_group_sizze_20791,
                                         segment_sizze_nonzzero_24345)))) {
                        // read operands
                        {
                            x_20796 = x_20795;
                            x_20795 = ((__local
                                        float *) red_arr_mem_24352)[sext_i32_i64(squot32(local_tid_24348,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_24364 = slt32(srem32(local_tid_24348,
                                                               k2p2zq_18071),
                                                        local_tid_24348 -
                                                        (squot32(local_tid_24348,
                                                                 32) * 32 - 1));
                            
                            if (inactive_24364) {
                                x_20795 = x_20796;
                            }
                            if (!inactive_24364) {
                                float res_20797 = x_20795 + x_20796;
                                
                                x_20795 = res_20797;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_24352)[sext_i32_i64(local_tid_24348)] =
                                x_20795;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_24348, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_24352)[sext_i32_i64(local_tid_24348)] =
                            x_20796;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_24356 * squot32(segred_group_sizze_20791,
                                                    segment_sizze_nonzzero_24345) +
                      local_tid_24348, m_18055 * k2p2zq_18071) &&
                slt32(local_tid_24348, squot32(segred_group_sizze_20791,
                                               segment_sizze_nonzzero_24345))) {
                ((__global
                  float *) mem_23604)[sext_i32_i64(squot32(virt_group_id_24356 *
                                                           squot32(segred_group_sizze_20791,
                                                                   segment_sizze_nonzzero_24345) +
                                                           local_tid_24348,
                                                           k2p2zq_18071)) *
                                      sext_i32_i64(k2p2zq_18071) +
                                      sext_i32_i64(virt_group_id_24356 *
                                      squot32(segred_group_sizze_20791,
                                              segment_sizze_nonzzero_24345) +
                                      local_tid_24348 -
                                      squot32(virt_group_id_24356 *
                                              squot32(segred_group_sizze_20791,
                                                      segment_sizze_nonzzero_24345) +
                                              local_tid_24348, k2p2zq_18071) *
                                      k2p2zq_18071)] = ((__local
                                                         float *) red_arr_mem_24352)[sext_i32_i64((local_tid_24348 +
                                                                                                   1) *
                                                                                     segment_sizze_nonzzero_24345 -
                                                                                     1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_20791
}
__kernel void mainzisegred_small_20882(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_24439_backing_aligned_0,
                                       int32_t N_18054, int32_t m_18055,
                                       int32_t k2p2zq_18071,
                                       int32_t num_groups_20939, __global
                                       unsigned char *mem_23213, __global
                                       unsigned char *res_mem_23612, __global
                                       unsigned char *mem_23725,
                                       int32_t segment_sizze_nonzzero_24432)
{
    #define segred_group_sizze_20938 (mainzisegred_group_sizze_20876)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24439_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24439_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24434;
    int32_t local_tid_24435;
    int32_t group_sizze_24438;
    int32_t wave_sizze_24437;
    int32_t group_tid_24436;
    
    global_tid_24434 = get_global_id(0);
    local_tid_24435 = get_local_id(0);
    group_sizze_24438 = get_local_size(0);
    wave_sizze_24437 = LOCKSTEP_WIDTH;
    group_tid_24436 = get_group_id(0);
    
    int32_t phys_tid_20882;
    
    phys_tid_20882 = global_tid_24434;
    
    __local char *red_arr_mem_24439;
    
    red_arr_mem_24439 = (__local char *) red_arr_mem_24439_backing_0;
    
    int32_t phys_group_id_24441;
    
    phys_group_id_24441 = get_group_id(0);
    for (int32_t i_24442 = 0; i_24442 < sdiv_up32(sdiv_up32(m_18055 * N_18054,
                                                            squot32(segred_group_sizze_20938,
                                                                    segment_sizze_nonzzero_24432)) -
                                                  phys_group_id_24441,
                                                  num_groups_20939);
         i_24442++) {
        int32_t virt_group_id_24443 = phys_group_id_24441 + i_24442 *
                num_groups_20939;
        int32_t gtid_20868 = squot32(squot32(local_tid_24435,
                                             segment_sizze_nonzzero_24432) +
                                     virt_group_id_24443 *
                                     squot32(segred_group_sizze_20938,
                                             segment_sizze_nonzzero_24432),
                                     N_18054);
        int32_t gtid_20869 = squot32(local_tid_24435,
                                     segment_sizze_nonzzero_24432) +
                virt_group_id_24443 * squot32(segred_group_sizze_20938,
                                              segment_sizze_nonzzero_24432) -
                squot32(squot32(local_tid_24435, segment_sizze_nonzzero_24432) +
                        virt_group_id_24443 * squot32(segred_group_sizze_20938,
                                                      segment_sizze_nonzzero_24432),
                        N_18054) * N_18054;
        int32_t gtid_20881 = srem32(local_tid_24435, k2p2zq_18071);
        
        // apply map function if in bounds
        {
            if (slt32(0, k2p2zq_18071) && ((slt32(gtid_20868, m_18055) &&
                                            slt32(gtid_20869, N_18054)) &&
                                           slt32(local_tid_24435, k2p2zq_18071 *
                                                 squot32(segred_group_sizze_20938,
                                                         segment_sizze_nonzzero_24432)))) {
                float x_20947 = ((__global
                                  float *) res_mem_23612)[sext_i32_i64(gtid_20868) *
                                                          sext_i32_i64(k2p2zq_18071) +
                                                          sext_i32_i64(gtid_20881)];
                float x_20948 = ((__global
                                  float *) mem_23213)[sext_i32_i64(gtid_20869) *
                                                      sext_i32_i64(k2p2zq_18071) +
                                                      sext_i32_i64(gtid_20881)];
                float res_20949 = x_20947 * x_20948;
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_24439)[sext_i32_i64(local_tid_24435)] =
                        res_20949;
                }
            } else {
                ((__local
                  float *) red_arr_mem_24439)[sext_i32_i64(local_tid_24435)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, k2p2zq_18071)) {
            // perform segmented scan to imitate reduction
            {
                float x_20942;
                float x_20943;
                float x_24444;
                float x_24445;
                int32_t skip_threads_24447;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_24435, k2p2zq_18071 *
                              squot32(segred_group_sizze_20938,
                                      segment_sizze_nonzzero_24432))) {
                        x_20943 = ((volatile __local
                                    float *) red_arr_mem_24439)[sext_i32_i64(local_tid_24435)];
                        if ((local_tid_24435 - squot32(local_tid_24435, 32) *
                             32) == 0) {
                            x_20942 = x_20943;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24447 = 1;
                    while (slt32(skip_threads_24447, 32)) {
                        if (sle32(skip_threads_24447, local_tid_24435 -
                                  squot32(local_tid_24435, 32) * 32) &&
                            slt32(local_tid_24435, k2p2zq_18071 *
                                  squot32(segred_group_sizze_20938,
                                          segment_sizze_nonzzero_24432))) {
                            // read operands
                            {
                                x_20942 = ((volatile __local
                                            float *) red_arr_mem_24439)[sext_i32_i64(local_tid_24435 -
                                                                        skip_threads_24447)];
                            }
                            // perform operation
                            {
                                bool inactive_24448 =
                                     slt32(srem32(local_tid_24435,
                                                  k2p2zq_18071),
                                           local_tid_24435 - (local_tid_24435 -
                                                              skip_threads_24447));
                                
                                if (inactive_24448) {
                                    x_20942 = x_20943;
                                }
                                if (!inactive_24448) {
                                    float res_20944 = x_20942 + x_20943;
                                    
                                    x_20942 = res_20944;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24437, skip_threads_24447)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24447, local_tid_24435 -
                                  squot32(local_tid_24435, 32) * 32) &&
                            slt32(local_tid_24435, k2p2zq_18071 *
                                  squot32(segred_group_sizze_20938,
                                          segment_sizze_nonzzero_24432))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24439)[sext_i32_i64(local_tid_24435)] =
                                    x_20942;
                                x_20943 = x_20942;
                            }
                        }
                        if (sle32(wave_sizze_24437, skip_threads_24447)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24447 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_24435 - squot32(local_tid_24435, 32) * 32) ==
                        31 && slt32(local_tid_24435, k2p2zq_18071 *
                                    squot32(segred_group_sizze_20938,
                                            segment_sizze_nonzzero_24432))) {
                        ((volatile __local
                          float *) red_arr_mem_24439)[sext_i32_i64(squot32(local_tid_24435,
                                                                           32))] =
                            x_20942;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_24449;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_24435, 32) == 0 &&
                            slt32(local_tid_24435, k2p2zq_18071 *
                                  squot32(segred_group_sizze_20938,
                                          segment_sizze_nonzzero_24432))) {
                            x_24445 = ((volatile __local
                                        float *) red_arr_mem_24439)[sext_i32_i64(local_tid_24435)];
                            if ((local_tid_24435 - squot32(local_tid_24435,
                                                           32) * 32) == 0) {
                                x_24444 = x_24445;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_24449 = 1;
                        while (slt32(skip_threads_24449, 32)) {
                            if (sle32(skip_threads_24449, local_tid_24435 -
                                      squot32(local_tid_24435, 32) * 32) &&
                                (squot32(local_tid_24435, 32) == 0 &&
                                 slt32(local_tid_24435, k2p2zq_18071 *
                                       squot32(segred_group_sizze_20938,
                                               segment_sizze_nonzzero_24432)))) {
                                // read operands
                                {
                                    x_24444 = ((volatile __local
                                                float *) red_arr_mem_24439)[sext_i32_i64(local_tid_24435 -
                                                                            skip_threads_24449)];
                                }
                                // perform operation
                                {
                                    bool inactive_24450 =
                                         slt32(srem32(local_tid_24435 * 32 +
                                                      32 - 1, k2p2zq_18071),
                                               local_tid_24435 * 32 + 32 - 1 -
                                               ((local_tid_24435 -
                                                 skip_threads_24449) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_24450) {
                                        x_24444 = x_24445;
                                    }
                                    if (!inactive_24450) {
                                        float res_24446 = x_24444 + x_24445;
                                        
                                        x_24444 = res_24446;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_24437, skip_threads_24449)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_24449, local_tid_24435 -
                                      squot32(local_tid_24435, 32) * 32) &&
                                (squot32(local_tid_24435, 32) == 0 &&
                                 slt32(local_tid_24435, k2p2zq_18071 *
                                       squot32(segred_group_sizze_20938,
                                               segment_sizze_nonzzero_24432)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_24439)[sext_i32_i64(local_tid_24435)] =
                                        x_24444;
                                    x_24445 = x_24444;
                                }
                            }
                            if (sle32(wave_sizze_24437, skip_threads_24449)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_24449 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_24435, 32) == 0 ||
                          !slt32(local_tid_24435, k2p2zq_18071 *
                                 squot32(segred_group_sizze_20938,
                                         segment_sizze_nonzzero_24432)))) {
                        // read operands
                        {
                            x_20943 = x_20942;
                            x_20942 = ((__local
                                        float *) red_arr_mem_24439)[sext_i32_i64(squot32(local_tid_24435,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_24451 = slt32(srem32(local_tid_24435,
                                                               k2p2zq_18071),
                                                        local_tid_24435 -
                                                        (squot32(local_tid_24435,
                                                                 32) * 32 - 1));
                            
                            if (inactive_24451) {
                                x_20942 = x_20943;
                            }
                            if (!inactive_24451) {
                                float res_20944 = x_20942 + x_20943;
                                
                                x_20942 = res_20944;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_24439)[sext_i32_i64(local_tid_24435)] =
                                x_20942;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_24435, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_24439)[sext_i32_i64(local_tid_24435)] =
                            x_20943;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_24443 * squot32(segred_group_sizze_20938,
                                                    segment_sizze_nonzzero_24432) +
                      local_tid_24435, m_18055 * N_18054) &&
                slt32(local_tid_24435, squot32(segred_group_sizze_20938,
                                               segment_sizze_nonzzero_24432))) {
                ((__global
                  float *) mem_23725)[sext_i32_i64(squot32(virt_group_id_24443 *
                                                           squot32(segred_group_sizze_20938,
                                                                   segment_sizze_nonzzero_24432) +
                                                           local_tid_24435,
                                                           N_18054)) *
                                      sext_i32_i64(N_18054) +
                                      sext_i32_i64(virt_group_id_24443 *
                                      squot32(segred_group_sizze_20938,
                                              segment_sizze_nonzzero_24432) +
                                      local_tid_24435 -
                                      squot32(virt_group_id_24443 *
                                              squot32(segred_group_sizze_20938,
                                                      segment_sizze_nonzzero_24432) +
                                              local_tid_24435, N_18054) *
                                      N_18054)] = ((__local
                                                    float *) red_arr_mem_24439)[sext_i32_i64((local_tid_24435 +
                                                                                              1) *
                                                                                segment_sizze_nonzzero_24432 -
                                                                                1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_20938
}
__kernel void mainzisegred_small_21461(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_24682_backing_aligned_0,
                                       int32_t N_18054, int32_t m_18055,
                                       int32_t n_18059,
                                       int32_t num_groups_21513, __global
                                       unsigned char *res_mem_23788, __global
                                       unsigned char *mem_23825, __global
                                       unsigned char *mem_23829,
                                       int32_t segment_sizze_nonzzero_24675)
{
    #define segred_group_sizze_21512 (mainzisegred_group_sizze_21455)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24682_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24682_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24677;
    int32_t local_tid_24678;
    int32_t group_sizze_24681;
    int32_t wave_sizze_24680;
    int32_t group_tid_24679;
    
    global_tid_24677 = get_global_id(0);
    local_tid_24678 = get_local_id(0);
    group_sizze_24681 = get_local_size(0);
    wave_sizze_24680 = LOCKSTEP_WIDTH;
    group_tid_24679 = get_group_id(0);
    
    int32_t phys_tid_21461;
    
    phys_tid_21461 = global_tid_24677;
    
    __local char *red_arr_mem_24682;
    
    red_arr_mem_24682 = (__local char *) red_arr_mem_24682_backing_0;
    
    int32_t phys_group_id_24684;
    
    phys_group_id_24684 = get_group_id(0);
    for (int32_t i_24685 = 0; i_24685 < sdiv_up32(sdiv_up32(m_18055,
                                                            squot32(segred_group_sizze_21512,
                                                                    segment_sizze_nonzzero_24675)) -
                                                  phys_group_id_24684,
                                                  num_groups_21513);
         i_24685++) {
        int32_t virt_group_id_24686 = phys_group_id_24684 + i_24685 *
                num_groups_21513;
        int32_t gtid_21450 = squot32(local_tid_24678,
                                     segment_sizze_nonzzero_24675) +
                virt_group_id_24686 * squot32(segred_group_sizze_21512,
                                              segment_sizze_nonzzero_24675);
        int32_t gtid_21460 = srem32(local_tid_24678, n_18059);
        
        // apply map function if in bounds
        {
            if (slt32(0, n_18059) && (slt32(gtid_21450, m_18055) &&
                                      slt32(local_tid_24678, n_18059 *
                                            squot32(segred_group_sizze_21512,
                                                    segment_sizze_nonzzero_24675)))) {
                int32_t res_21520 = ((__global
                                      int32_t *) mem_23825)[sext_i32_i64(gtid_21450)];
                bool cond_21523 = slt32(gtid_21460, res_21520);
                float res_21524;
                
                if (cond_21523) {
                    float x_elem_21522 = ((__global
                                           float *) res_mem_23788)[sext_i32_i64(gtid_21450) *
                                                                   sext_i32_i64(N_18054) +
                                                                   sext_i32_i64(gtid_21460)];
                    
                    res_21524 = x_elem_21522;
                } else {
                    res_21524 = 0.0F;
                }
                
                float res_21525 = res_21524 * res_21524;
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_24682)[sext_i32_i64(local_tid_24678)] =
                        res_21525;
                }
            } else {
                ((__local
                  float *) red_arr_mem_24682)[sext_i32_i64(local_tid_24678)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, n_18059)) {
            // perform segmented scan to imitate reduction
            {
                float x_21516;
                float x_21517;
                float x_24687;
                float x_24688;
                int32_t skip_threads_24690;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_24678, n_18059 *
                              squot32(segred_group_sizze_21512,
                                      segment_sizze_nonzzero_24675))) {
                        x_21517 = ((volatile __local
                                    float *) red_arr_mem_24682)[sext_i32_i64(local_tid_24678)];
                        if ((local_tid_24678 - squot32(local_tid_24678, 32) *
                             32) == 0) {
                            x_21516 = x_21517;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24690 = 1;
                    while (slt32(skip_threads_24690, 32)) {
                        if (sle32(skip_threads_24690, local_tid_24678 -
                                  squot32(local_tid_24678, 32) * 32) &&
                            slt32(local_tid_24678, n_18059 *
                                  squot32(segred_group_sizze_21512,
                                          segment_sizze_nonzzero_24675))) {
                            // read operands
                            {
                                x_21516 = ((volatile __local
                                            float *) red_arr_mem_24682)[sext_i32_i64(local_tid_24678 -
                                                                        skip_threads_24690)];
                            }
                            // perform operation
                            {
                                bool inactive_24691 =
                                     slt32(srem32(local_tid_24678, n_18059),
                                           local_tid_24678 - (local_tid_24678 -
                                                              skip_threads_24690));
                                
                                if (inactive_24691) {
                                    x_21516 = x_21517;
                                }
                                if (!inactive_24691) {
                                    float res_21518 = x_21516 + x_21517;
                                    
                                    x_21516 = res_21518;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24680, skip_threads_24690)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24690, local_tid_24678 -
                                  squot32(local_tid_24678, 32) * 32) &&
                            slt32(local_tid_24678, n_18059 *
                                  squot32(segred_group_sizze_21512,
                                          segment_sizze_nonzzero_24675))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24682)[sext_i32_i64(local_tid_24678)] =
                                    x_21516;
                                x_21517 = x_21516;
                            }
                        }
                        if (sle32(wave_sizze_24680, skip_threads_24690)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24690 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_24678 - squot32(local_tid_24678, 32) * 32) ==
                        31 && slt32(local_tid_24678, n_18059 *
                                    squot32(segred_group_sizze_21512,
                                            segment_sizze_nonzzero_24675))) {
                        ((volatile __local
                          float *) red_arr_mem_24682)[sext_i32_i64(squot32(local_tid_24678,
                                                                           32))] =
                            x_21516;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_24692;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_24678, 32) == 0 &&
                            slt32(local_tid_24678, n_18059 *
                                  squot32(segred_group_sizze_21512,
                                          segment_sizze_nonzzero_24675))) {
                            x_24688 = ((volatile __local
                                        float *) red_arr_mem_24682)[sext_i32_i64(local_tid_24678)];
                            if ((local_tid_24678 - squot32(local_tid_24678,
                                                           32) * 32) == 0) {
                                x_24687 = x_24688;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_24692 = 1;
                        while (slt32(skip_threads_24692, 32)) {
                            if (sle32(skip_threads_24692, local_tid_24678 -
                                      squot32(local_tid_24678, 32) * 32) &&
                                (squot32(local_tid_24678, 32) == 0 &&
                                 slt32(local_tid_24678, n_18059 *
                                       squot32(segred_group_sizze_21512,
                                               segment_sizze_nonzzero_24675)))) {
                                // read operands
                                {
                                    x_24687 = ((volatile __local
                                                float *) red_arr_mem_24682)[sext_i32_i64(local_tid_24678 -
                                                                            skip_threads_24692)];
                                }
                                // perform operation
                                {
                                    bool inactive_24693 =
                                         slt32(srem32(local_tid_24678 * 32 +
                                                      32 - 1, n_18059),
                                               local_tid_24678 * 32 + 32 - 1 -
                                               ((local_tid_24678 -
                                                 skip_threads_24692) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_24693) {
                                        x_24687 = x_24688;
                                    }
                                    if (!inactive_24693) {
                                        float res_24689 = x_24687 + x_24688;
                                        
                                        x_24687 = res_24689;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_24680, skip_threads_24692)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_24692, local_tid_24678 -
                                      squot32(local_tid_24678, 32) * 32) &&
                                (squot32(local_tid_24678, 32) == 0 &&
                                 slt32(local_tid_24678, n_18059 *
                                       squot32(segred_group_sizze_21512,
                                               segment_sizze_nonzzero_24675)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_24682)[sext_i32_i64(local_tid_24678)] =
                                        x_24687;
                                    x_24688 = x_24687;
                                }
                            }
                            if (sle32(wave_sizze_24680, skip_threads_24692)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_24692 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_24678, 32) == 0 ||
                          !slt32(local_tid_24678, n_18059 *
                                 squot32(segred_group_sizze_21512,
                                         segment_sizze_nonzzero_24675)))) {
                        // read operands
                        {
                            x_21517 = x_21516;
                            x_21516 = ((__local
                                        float *) red_arr_mem_24682)[sext_i32_i64(squot32(local_tid_24678,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_24694 = slt32(srem32(local_tid_24678,
                                                               n_18059),
                                                        local_tid_24678 -
                                                        (squot32(local_tid_24678,
                                                                 32) * 32 - 1));
                            
                            if (inactive_24694) {
                                x_21516 = x_21517;
                            }
                            if (!inactive_24694) {
                                float res_21518 = x_21516 + x_21517;
                                
                                x_21516 = res_21518;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_24682)[sext_i32_i64(local_tid_24678)] =
                                x_21516;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_24678, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_24682)[sext_i32_i64(local_tid_24678)] =
                            x_21517;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_24686 * squot32(segred_group_sizze_21512,
                                                    segment_sizze_nonzzero_24675) +
                      local_tid_24678, m_18055) && slt32(local_tid_24678,
                                                         squot32(segred_group_sizze_21512,
                                                                 segment_sizze_nonzzero_24675))) {
                ((__global
                  float *) mem_23829)[sext_i32_i64(virt_group_id_24686 *
                                      squot32(segred_group_sizze_21512,
                                              segment_sizze_nonzzero_24675) +
                                      local_tid_24678)] = ((__local
                                                            float *) red_arr_mem_24682)[sext_i32_i64((local_tid_24678 +
                                                                                                      1) *
                                                                                        segment_sizze_nonzzero_24675 -
                                                                                        1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21512
}
__kernel void mainzisegred_small_21483(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_24623_backing_aligned_0,
                                       int32_t m_18055, int32_t N_18056,
                                       int32_t n_18059,
                                       int32_t num_groups_21497, __global
                                       unsigned char *images_mem_23189, __global
                                       unsigned char *mem_23825,
                                       int32_t segment_sizze_nonzzero_24616)
{
    #define segred_group_sizze_21496 (mainzisegred_group_sizze_21477)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24623_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24623_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24618;
    int32_t local_tid_24619;
    int32_t group_sizze_24622;
    int32_t wave_sizze_24621;
    int32_t group_tid_24620;
    
    global_tid_24618 = get_global_id(0);
    local_tid_24619 = get_local_id(0);
    group_sizze_24622 = get_local_size(0);
    wave_sizze_24621 = LOCKSTEP_WIDTH;
    group_tid_24620 = get_group_id(0);
    
    int32_t phys_tid_21483;
    
    phys_tid_21483 = global_tid_24618;
    
    __local char *red_arr_mem_24623;
    
    red_arr_mem_24623 = (__local char *) red_arr_mem_24623_backing_0;
    
    int32_t phys_group_id_24625;
    
    phys_group_id_24625 = get_group_id(0);
    for (int32_t i_24626 = 0; i_24626 < sdiv_up32(sdiv_up32(m_18055,
                                                            squot32(segred_group_sizze_21496,
                                                                    segment_sizze_nonzzero_24616)) -
                                                  phys_group_id_24625,
                                                  num_groups_21497);
         i_24626++) {
        int32_t virt_group_id_24627 = phys_group_id_24625 + i_24626 *
                num_groups_21497;
        int32_t gtid_21472 = squot32(local_tid_24619,
                                     segment_sizze_nonzzero_24616) +
                virt_group_id_24627 * squot32(segred_group_sizze_21496,
                                              segment_sizze_nonzzero_24616);
        int32_t gtid_21482 = srem32(local_tid_24619, n_18059);
        
        // apply map function if in bounds
        {
            if (slt32(0, n_18059) && (slt32(gtid_21472, m_18055) &&
                                      slt32(local_tid_24619, n_18059 *
                                            squot32(segred_group_sizze_21496,
                                                    segment_sizze_nonzzero_24616)))) {
                float x_21504 = ((__global
                                  float *) images_mem_23189)[sext_i32_i64(gtid_21472) *
                                                             sext_i32_i64(N_18056) +
                                                             sext_i32_i64(gtid_21482)];
                bool res_21505;
                
                res_21505 = futrts_isnan32(x_21504);
                
                bool cond_21506 = !res_21505;
                int32_t res_21507 = btoi_bool_i32(cond_21506);
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      int32_t *) red_arr_mem_24623)[sext_i32_i64(local_tid_24619)] =
                        res_21507;
                }
            } else {
                ((__local
                  int32_t *) red_arr_mem_24623)[sext_i32_i64(local_tid_24619)] =
                    0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, n_18059)) {
            // perform segmented scan to imitate reduction
            {
                int32_t x_21500;
                int32_t x_21501;
                int32_t x_24628;
                int32_t x_24629;
                int32_t skip_threads_24631;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_24619, n_18059 *
                              squot32(segred_group_sizze_21496,
                                      segment_sizze_nonzzero_24616))) {
                        x_21501 = ((volatile __local
                                    int32_t *) red_arr_mem_24623)[sext_i32_i64(local_tid_24619)];
                        if ((local_tid_24619 - squot32(local_tid_24619, 32) *
                             32) == 0) {
                            x_21500 = x_21501;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24631 = 1;
                    while (slt32(skip_threads_24631, 32)) {
                        if (sle32(skip_threads_24631, local_tid_24619 -
                                  squot32(local_tid_24619, 32) * 32) &&
                            slt32(local_tid_24619, n_18059 *
                                  squot32(segred_group_sizze_21496,
                                          segment_sizze_nonzzero_24616))) {
                            // read operands
                            {
                                x_21500 = ((volatile __local
                                            int32_t *) red_arr_mem_24623)[sext_i32_i64(local_tid_24619 -
                                                                          skip_threads_24631)];
                            }
                            // perform operation
                            {
                                bool inactive_24632 =
                                     slt32(srem32(local_tid_24619, n_18059),
                                           local_tid_24619 - (local_tid_24619 -
                                                              skip_threads_24631));
                                
                                if (inactive_24632) {
                                    x_21500 = x_21501;
                                }
                                if (!inactive_24632) {
                                    int32_t res_21502 = add32(x_21500, x_21501);
                                    
                                    x_21500 = res_21502;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24621, skip_threads_24631)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24631, local_tid_24619 -
                                  squot32(local_tid_24619, 32) * 32) &&
                            slt32(local_tid_24619, n_18059 *
                                  squot32(segred_group_sizze_21496,
                                          segment_sizze_nonzzero_24616))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) red_arr_mem_24623)[sext_i32_i64(local_tid_24619)] =
                                    x_21500;
                                x_21501 = x_21500;
                            }
                        }
                        if (sle32(wave_sizze_24621, skip_threads_24631)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24631 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_24619 - squot32(local_tid_24619, 32) * 32) ==
                        31 && slt32(local_tid_24619, n_18059 *
                                    squot32(segred_group_sizze_21496,
                                            segment_sizze_nonzzero_24616))) {
                        ((volatile __local
                          int32_t *) red_arr_mem_24623)[sext_i32_i64(squot32(local_tid_24619,
                                                                             32))] =
                            x_21500;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_24633;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_24619, 32) == 0 &&
                            slt32(local_tid_24619, n_18059 *
                                  squot32(segred_group_sizze_21496,
                                          segment_sizze_nonzzero_24616))) {
                            x_24629 = ((volatile __local
                                        int32_t *) red_arr_mem_24623)[sext_i32_i64(local_tid_24619)];
                            if ((local_tid_24619 - squot32(local_tid_24619,
                                                           32) * 32) == 0) {
                                x_24628 = x_24629;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_24633 = 1;
                        while (slt32(skip_threads_24633, 32)) {
                            if (sle32(skip_threads_24633, local_tid_24619 -
                                      squot32(local_tid_24619, 32) * 32) &&
                                (squot32(local_tid_24619, 32) == 0 &&
                                 slt32(local_tid_24619, n_18059 *
                                       squot32(segred_group_sizze_21496,
                                               segment_sizze_nonzzero_24616)))) {
                                // read operands
                                {
                                    x_24628 = ((volatile __local
                                                int32_t *) red_arr_mem_24623)[sext_i32_i64(local_tid_24619 -
                                                                              skip_threads_24633)];
                                }
                                // perform operation
                                {
                                    bool inactive_24634 =
                                         slt32(srem32(local_tid_24619 * 32 +
                                                      32 - 1, n_18059),
                                               local_tid_24619 * 32 + 32 - 1 -
                                               ((local_tid_24619 -
                                                 skip_threads_24633) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_24634) {
                                        x_24628 = x_24629;
                                    }
                                    if (!inactive_24634) {
                                        int32_t res_24630 = add32(x_24628,
                                                                  x_24629);
                                        
                                        x_24628 = res_24630;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_24621, skip_threads_24633)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_24633, local_tid_24619 -
                                      squot32(local_tid_24619, 32) * 32) &&
                                (squot32(local_tid_24619, 32) == 0 &&
                                 slt32(local_tid_24619, n_18059 *
                                       squot32(segred_group_sizze_21496,
                                               segment_sizze_nonzzero_24616)))) {
                                // write result
                                {
                                    ((volatile __local
                                      int32_t *) red_arr_mem_24623)[sext_i32_i64(local_tid_24619)] =
                                        x_24628;
                                    x_24629 = x_24628;
                                }
                            }
                            if (sle32(wave_sizze_24621, skip_threads_24633)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_24633 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_24619, 32) == 0 ||
                          !slt32(local_tid_24619, n_18059 *
                                 squot32(segred_group_sizze_21496,
                                         segment_sizze_nonzzero_24616)))) {
                        // read operands
                        {
                            x_21501 = x_21500;
                            x_21500 = ((__local
                                        int32_t *) red_arr_mem_24623)[sext_i32_i64(squot32(local_tid_24619,
                                                                                           32) -
                                                                      1)];
                        }
                        // perform operation
                        {
                            bool inactive_24635 = slt32(srem32(local_tid_24619,
                                                               n_18059),
                                                        local_tid_24619 -
                                                        (squot32(local_tid_24619,
                                                                 32) * 32 - 1));
                            
                            if (inactive_24635) {
                                x_21500 = x_21501;
                            }
                            if (!inactive_24635) {
                                int32_t res_21502 = add32(x_21500, x_21501);
                                
                                x_21500 = res_21502;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              int32_t *) red_arr_mem_24623)[sext_i32_i64(local_tid_24619)] =
                                x_21500;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_24619, 32) == 0) {
                        ((__local
                          int32_t *) red_arr_mem_24623)[sext_i32_i64(local_tid_24619)] =
                            x_21501;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_24627 * squot32(segred_group_sizze_21496,
                                                    segment_sizze_nonzzero_24616) +
                      local_tid_24619, m_18055) && slt32(local_tid_24619,
                                                         squot32(segred_group_sizze_21496,
                                                                 segment_sizze_nonzzero_24616))) {
                ((__global
                  int32_t *) mem_23825)[sext_i32_i64(virt_group_id_24627 *
                                        squot32(segred_group_sizze_21496,
                                                segment_sizze_nonzzero_24616) +
                                        local_tid_24619)] = ((__local
                                                              int32_t *) red_arr_mem_24623)[sext_i32_i64((local_tid_24619 +
                                                                                                          1) *
                                                                                            segment_sizze_nonzzero_24616 -
                                                                                            1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21496
}
__kernel void mainzisegred_small_21617(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_24786_backing_aligned_0,
                                       int32_t N_18054, int32_t m_18055,
                                       int32_t res_18381,
                                       int32_t num_groups_21636, __global
                                       unsigned char *res_mem_23788, __global
                                       unsigned char *res_mem_23840, __global
                                       unsigned char *res_mem_23841, __global
                                       unsigned char *mem_23853,
                                       int32_t segment_sizze_nonzzero_24779)
{
    #define segred_group_sizze_21635 (mainzisegred_group_sizze_21611)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24786_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24786_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24781;
    int32_t local_tid_24782;
    int32_t group_sizze_24785;
    int32_t wave_sizze_24784;
    int32_t group_tid_24783;
    
    global_tid_24781 = get_global_id(0);
    local_tid_24782 = get_local_id(0);
    group_sizze_24785 = get_local_size(0);
    wave_sizze_24784 = LOCKSTEP_WIDTH;
    group_tid_24783 = get_group_id(0);
    
    int32_t phys_tid_21617;
    
    phys_tid_21617 = global_tid_24781;
    
    __local char *red_arr_mem_24786;
    
    red_arr_mem_24786 = (__local char *) red_arr_mem_24786_backing_0;
    
    int32_t phys_group_id_24788;
    
    phys_group_id_24788 = get_group_id(0);
    for (int32_t i_24789 = 0; i_24789 < sdiv_up32(sdiv_up32(m_18055,
                                                            squot32(segred_group_sizze_21635,
                                                                    segment_sizze_nonzzero_24779)) -
                                                  phys_group_id_24788,
                                                  num_groups_21636);
         i_24789++) {
        int32_t virt_group_id_24790 = phys_group_id_24788 + i_24789 *
                num_groups_21636;
        int32_t gtid_21606 = squot32(local_tid_24782,
                                     segment_sizze_nonzzero_24779) +
                virt_group_id_24790 * squot32(segred_group_sizze_21635,
                                              segment_sizze_nonzzero_24779);
        int32_t gtid_21616 = srem32(local_tid_24782, res_18381);
        
        // apply map function if in bounds
        {
            if (slt32(0, res_18381) && (slt32(gtid_21606, m_18055) &&
                                        slt32(local_tid_24782, res_18381 *
                                              squot32(segred_group_sizze_21635,
                                                      segment_sizze_nonzzero_24779)))) {
                int32_t x_21644 = ((__global
                                    int32_t *) res_mem_23840)[sext_i32_i64(gtid_21606)];
                bool cond_21646 = slt32(gtid_21616, x_21644);
                float res_21647;
                
                if (cond_21646) {
                    int32_t x_21643 = ((__global
                                        int32_t *) res_mem_23841)[sext_i32_i64(gtid_21606)];
                    int32_t x_21648 = add32(gtid_21616, x_21643);
                    int32_t x_21649 = sub32(x_21648, x_21644);
                    int32_t i_21650 = add32(1, x_21649);
                    float res_21651 = ((__global
                                        float *) res_mem_23788)[sext_i32_i64(gtid_21606) *
                                                                sext_i32_i64(N_18054) +
                                                                sext_i32_i64(i_21650)];
                    
                    res_21647 = res_21651;
                } else {
                    res_21647 = 0.0F;
                }
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_24786)[sext_i32_i64(local_tid_24782)] =
                        res_21647;
                }
            } else {
                ((__local
                  float *) red_arr_mem_24786)[sext_i32_i64(local_tid_24782)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, res_18381)) {
            // perform segmented scan to imitate reduction
            {
                float x_21639;
                float x_21640;
                float x_24791;
                float x_24792;
                int32_t skip_threads_24794;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_24782, res_18381 *
                              squot32(segred_group_sizze_21635,
                                      segment_sizze_nonzzero_24779))) {
                        x_21640 = ((volatile __local
                                    float *) red_arr_mem_24786)[sext_i32_i64(local_tid_24782)];
                        if ((local_tid_24782 - squot32(local_tid_24782, 32) *
                             32) == 0) {
                            x_21639 = x_21640;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24794 = 1;
                    while (slt32(skip_threads_24794, 32)) {
                        if (sle32(skip_threads_24794, local_tid_24782 -
                                  squot32(local_tid_24782, 32) * 32) &&
                            slt32(local_tid_24782, res_18381 *
                                  squot32(segred_group_sizze_21635,
                                          segment_sizze_nonzzero_24779))) {
                            // read operands
                            {
                                x_21639 = ((volatile __local
                                            float *) red_arr_mem_24786)[sext_i32_i64(local_tid_24782 -
                                                                        skip_threads_24794)];
                            }
                            // perform operation
                            {
                                bool inactive_24795 =
                                     slt32(srem32(local_tid_24782, res_18381),
                                           local_tid_24782 - (local_tid_24782 -
                                                              skip_threads_24794));
                                
                                if (inactive_24795) {
                                    x_21639 = x_21640;
                                }
                                if (!inactive_24795) {
                                    float res_21641 = x_21639 + x_21640;
                                    
                                    x_21639 = res_21641;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24784, skip_threads_24794)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24794, local_tid_24782 -
                                  squot32(local_tid_24782, 32) * 32) &&
                            slt32(local_tid_24782, res_18381 *
                                  squot32(segred_group_sizze_21635,
                                          segment_sizze_nonzzero_24779))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24786)[sext_i32_i64(local_tid_24782)] =
                                    x_21639;
                                x_21640 = x_21639;
                            }
                        }
                        if (sle32(wave_sizze_24784, skip_threads_24794)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24794 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_24782 - squot32(local_tid_24782, 32) * 32) ==
                        31 && slt32(local_tid_24782, res_18381 *
                                    squot32(segred_group_sizze_21635,
                                            segment_sizze_nonzzero_24779))) {
                        ((volatile __local
                          float *) red_arr_mem_24786)[sext_i32_i64(squot32(local_tid_24782,
                                                                           32))] =
                            x_21639;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_24796;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_24782, 32) == 0 &&
                            slt32(local_tid_24782, res_18381 *
                                  squot32(segred_group_sizze_21635,
                                          segment_sizze_nonzzero_24779))) {
                            x_24792 = ((volatile __local
                                        float *) red_arr_mem_24786)[sext_i32_i64(local_tid_24782)];
                            if ((local_tid_24782 - squot32(local_tid_24782,
                                                           32) * 32) == 0) {
                                x_24791 = x_24792;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_24796 = 1;
                        while (slt32(skip_threads_24796, 32)) {
                            if (sle32(skip_threads_24796, local_tid_24782 -
                                      squot32(local_tid_24782, 32) * 32) &&
                                (squot32(local_tid_24782, 32) == 0 &&
                                 slt32(local_tid_24782, res_18381 *
                                       squot32(segred_group_sizze_21635,
                                               segment_sizze_nonzzero_24779)))) {
                                // read operands
                                {
                                    x_24791 = ((volatile __local
                                                float *) red_arr_mem_24786)[sext_i32_i64(local_tid_24782 -
                                                                            skip_threads_24796)];
                                }
                                // perform operation
                                {
                                    bool inactive_24797 =
                                         slt32(srem32(local_tid_24782 * 32 +
                                                      32 - 1, res_18381),
                                               local_tid_24782 * 32 + 32 - 1 -
                                               ((local_tid_24782 -
                                                 skip_threads_24796) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_24797) {
                                        x_24791 = x_24792;
                                    }
                                    if (!inactive_24797) {
                                        float res_24793 = x_24791 + x_24792;
                                        
                                        x_24791 = res_24793;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_24784, skip_threads_24796)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_24796, local_tid_24782 -
                                      squot32(local_tid_24782, 32) * 32) &&
                                (squot32(local_tid_24782, 32) == 0 &&
                                 slt32(local_tid_24782, res_18381 *
                                       squot32(segred_group_sizze_21635,
                                               segment_sizze_nonzzero_24779)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_24786)[sext_i32_i64(local_tid_24782)] =
                                        x_24791;
                                    x_24792 = x_24791;
                                }
                            }
                            if (sle32(wave_sizze_24784, skip_threads_24796)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_24796 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_24782, 32) == 0 ||
                          !slt32(local_tid_24782, res_18381 *
                                 squot32(segred_group_sizze_21635,
                                         segment_sizze_nonzzero_24779)))) {
                        // read operands
                        {
                            x_21640 = x_21639;
                            x_21639 = ((__local
                                        float *) red_arr_mem_24786)[sext_i32_i64(squot32(local_tid_24782,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_24798 = slt32(srem32(local_tid_24782,
                                                               res_18381),
                                                        local_tid_24782 -
                                                        (squot32(local_tid_24782,
                                                                 32) * 32 - 1));
                            
                            if (inactive_24798) {
                                x_21639 = x_21640;
                            }
                            if (!inactive_24798) {
                                float res_21641 = x_21639 + x_21640;
                                
                                x_21639 = res_21641;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_24786)[sext_i32_i64(local_tid_24782)] =
                                x_21639;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_24782, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_24786)[sext_i32_i64(local_tid_24782)] =
                            x_21640;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_24790 * squot32(segred_group_sizze_21635,
                                                    segment_sizze_nonzzero_24779) +
                      local_tid_24782, m_18055) && slt32(local_tid_24782,
                                                         squot32(segred_group_sizze_21635,
                                                                 segment_sizze_nonzzero_24779))) {
                ((__global
                  float *) mem_23853)[sext_i32_i64(virt_group_id_24790 *
                                      squot32(segred_group_sizze_21635,
                                              segment_sizze_nonzzero_24779) +
                                      local_tid_24782)] = ((__local
                                                            float *) red_arr_mem_24786)[sext_i32_i64((local_tid_24782 +
                                                                                                      1) *
                                                                                        segment_sizze_nonzzero_24779 -
                                                                                        1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21635
}
__kernel void mainzisegred_small_22108(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_24944_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24942_backing_aligned_1,
                                       __local volatile
                                       int64_t *red_arr_mem_24940_backing_aligned_2,
                                       int32_t m_18055, int32_t iota_arg_18403,
                                       int32_t num_groups_22292, __global
                                       unsigned char *mem_23858, __global
                                       unsigned char *mem_23874, __global
                                       unsigned char *mem_23877, __global
                                       unsigned char *mem_23883, __global
                                       unsigned char *mem_23886, __global
                                       unsigned char *mem_23889, __global
                                       unsigned char *mem_23892,
                                       int32_t segment_sizze_nonzzero_24933)
{
    #define segred_group_sizze_22291 (mainzisegred_group_sizze_22102)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24944_backing_2 =
                          (__local volatile
                           char *) red_arr_mem_24944_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24942_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_24942_backing_aligned_1;
    __local volatile char *restrict red_arr_mem_24940_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24940_backing_aligned_2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24935;
    int32_t local_tid_24936;
    int32_t group_sizze_24939;
    int32_t wave_sizze_24938;
    int32_t group_tid_24937;
    
    global_tid_24935 = get_global_id(0);
    local_tid_24936 = get_local_id(0);
    group_sizze_24939 = get_local_size(0);
    wave_sizze_24938 = LOCKSTEP_WIDTH;
    group_tid_24937 = get_group_id(0);
    
    int32_t phys_tid_22108;
    
    phys_tid_22108 = global_tid_24935;
    
    __local char *red_arr_mem_24940;
    
    red_arr_mem_24940 = (__local char *) red_arr_mem_24940_backing_0;
    
    __local char *red_arr_mem_24942;
    
    red_arr_mem_24942 = (__local char *) red_arr_mem_24942_backing_1;
    
    __local char *red_arr_mem_24944;
    
    red_arr_mem_24944 = (__local char *) red_arr_mem_24944_backing_2;
    
    int32_t phys_group_id_24946;
    
    phys_group_id_24946 = get_group_id(0);
    for (int32_t i_24947 = 0; i_24947 < sdiv_up32(sdiv_up32(m_18055,
                                                            squot32(segred_group_sizze_22291,
                                                                    segment_sizze_nonzzero_24933)) -
                                                  phys_group_id_24946,
                                                  num_groups_22292);
         i_24947++) {
        int32_t virt_group_id_24948 = phys_group_id_24946 + i_24947 *
                num_groups_22292;
        int32_t gtid_22097 = squot32(local_tid_24936,
                                     segment_sizze_nonzzero_24933) +
                virt_group_id_24948 * squot32(segred_group_sizze_22291,
                                              segment_sizze_nonzzero_24933);
        int32_t gtid_22107 = srem32(local_tid_24936, iota_arg_18403);
        
        // apply map function if in bounds
        {
            if (slt32(0, iota_arg_18403) && (slt32(gtid_22097, m_18055) &&
                                             slt32(local_tid_24936,
                                                   iota_arg_18403 *
                                                   squot32(segred_group_sizze_22291,
                                                           segment_sizze_nonzzero_24933)))) {
                int32_t y_22311 = ((__global
                                    int32_t *) mem_23877)[sext_i32_i64(gtid_22097)];
                float y_22312 = ((__global
                                  float *) mem_23874)[sext_i32_i64(gtid_22097)];
                float x_22316 = ((__global
                                  float *) mem_23883)[sext_i32_i64(gtid_22097) *
                                                      sext_i32_i64(iota_arg_18403) +
                                                      sext_i32_i64(gtid_22107)];
                float x_22317 = ((__global
                                  float *) mem_23858)[sext_i32_i64(gtid_22107)];
                float res_22320 = x_22316 / y_22312;
                bool cond_22321 = slt32(gtid_22107, y_22311);
                bool res_22322;
                
                res_22322 = futrts_isnan32(res_22320);
                
                bool res_22323 = !res_22322;
                bool x_22324 = cond_22321 && res_22323;
                float res_22325 = (float) fabs(res_22320);
                bool res_22326 = x_22317 < res_22325;
                bool x_22327 = x_22324 && res_22326;
                float res_22328;
                
                if (cond_22321) {
                    res_22328 = res_22320;
                } else {
                    res_22328 = 0.0F;
                }
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      bool *) red_arr_mem_24940)[sext_i32_i64(local_tid_24936)] =
                        x_22327;
                    ((__local
                      int32_t *) red_arr_mem_24942)[sext_i32_i64(local_tid_24936)] =
                        gtid_22107;
                    ((__local
                      float *) red_arr_mem_24944)[sext_i32_i64(local_tid_24936)] =
                        res_22328;
                }
            } else {
                ((__local
                  bool *) red_arr_mem_24940)[sext_i32_i64(local_tid_24936)] = 0;
                ((__local
                  int32_t *) red_arr_mem_24942)[sext_i32_i64(local_tid_24936)] =
                    -1;
                ((__local
                  float *) red_arr_mem_24944)[sext_i32_i64(local_tid_24936)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, iota_arg_18403)) {
            // perform segmented scan to imitate reduction
            {
                bool x_22297;
                int32_t x_22298;
                float x_22299;
                bool x_22300;
                int32_t x_22301;
                float x_22302;
                bool x_24949;
                int32_t x_24950;
                float x_24951;
                bool x_24952;
                int32_t x_24953;
                float x_24954;
                int32_t skip_threads_24963;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_24936, iota_arg_18403 *
                              squot32(segred_group_sizze_22291,
                                      segment_sizze_nonzzero_24933))) {
                        x_22300 = ((volatile __local
                                    bool *) red_arr_mem_24940)[sext_i32_i64(local_tid_24936)];
                        x_22301 = ((volatile __local
                                    int32_t *) red_arr_mem_24942)[sext_i32_i64(local_tid_24936)];
                        x_22302 = ((volatile __local
                                    float *) red_arr_mem_24944)[sext_i32_i64(local_tid_24936)];
                        if ((local_tid_24936 - squot32(local_tid_24936, 32) *
                             32) == 0) {
                            x_22297 = x_22300;
                            x_22298 = x_22301;
                            x_22299 = x_22302;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24963 = 1;
                    while (slt32(skip_threads_24963, 32)) {
                        if (sle32(skip_threads_24963, local_tid_24936 -
                                  squot32(local_tid_24936, 32) * 32) &&
                            slt32(local_tid_24936, iota_arg_18403 *
                                  squot32(segred_group_sizze_22291,
                                          segment_sizze_nonzzero_24933))) {
                            // read operands
                            {
                                x_22297 = ((volatile __local
                                            bool *) red_arr_mem_24940)[sext_i32_i64(local_tid_24936 -
                                                                       skip_threads_24963)];
                                x_22298 = ((volatile __local
                                            int32_t *) red_arr_mem_24942)[sext_i32_i64(local_tid_24936 -
                                                                          skip_threads_24963)];
                                x_22299 = ((volatile __local
                                            float *) red_arr_mem_24944)[sext_i32_i64(local_tid_24936 -
                                                                        skip_threads_24963)];
                            }
                            // perform operation
                            {
                                bool inactive_24964 =
                                     slt32(srem32(local_tid_24936,
                                                  iota_arg_18403),
                                           local_tid_24936 - (local_tid_24936 -
                                                              skip_threads_24963));
                                
                                if (inactive_24964) {
                                    x_22297 = x_22300;
                                    x_22298 = x_22301;
                                    x_22299 = x_22302;
                                }
                                if (!inactive_24964) {
                                    bool res_22303;
                                    int32_t res_22304;
                                    
                                    if (x_22297) {
                                        res_22303 = x_22297;
                                        res_22304 = x_22298;
                                    } else {
                                        bool x_22305 = x_22300 && x_22300;
                                        bool x_22306 = !x_22300;
                                        bool y_22307 = x_22297 && x_22306;
                                        bool res_22308 = x_22305 || y_22307;
                                        int32_t res_22309;
                                        
                                        if (x_22300) {
                                            res_22309 = x_22301;
                                        } else {
                                            res_22309 = x_22298;
                                        }
                                        res_22303 = res_22308;
                                        res_22304 = res_22309;
                                    }
                                    
                                    float res_22310 = x_22299 + x_22302;
                                    
                                    x_22297 = res_22303;
                                    x_22298 = res_22304;
                                    x_22299 = res_22310;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24938, skip_threads_24963)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24963, local_tid_24936 -
                                  squot32(local_tid_24936, 32) * 32) &&
                            slt32(local_tid_24936, iota_arg_18403 *
                                  squot32(segred_group_sizze_22291,
                                          segment_sizze_nonzzero_24933))) {
                            // write result
                            {
                                ((volatile __local
                                  bool *) red_arr_mem_24940)[sext_i32_i64(local_tid_24936)] =
                                    x_22297;
                                x_22300 = x_22297;
                                ((volatile __local
                                  int32_t *) red_arr_mem_24942)[sext_i32_i64(local_tid_24936)] =
                                    x_22298;
                                x_22301 = x_22298;
                                ((volatile __local
                                  float *) red_arr_mem_24944)[sext_i32_i64(local_tid_24936)] =
                                    x_22299;
                                x_22302 = x_22299;
                            }
                        }
                        if (sle32(wave_sizze_24938, skip_threads_24963)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24963 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_24936 - squot32(local_tid_24936, 32) * 32) ==
                        31 && slt32(local_tid_24936, iota_arg_18403 *
                                    squot32(segred_group_sizze_22291,
                                            segment_sizze_nonzzero_24933))) {
                        ((volatile __local
                          bool *) red_arr_mem_24940)[sext_i32_i64(squot32(local_tid_24936,
                                                                          32))] =
                            x_22297;
                        ((volatile __local
                          int32_t *) red_arr_mem_24942)[sext_i32_i64(squot32(local_tid_24936,
                                                                             32))] =
                            x_22298;
                        ((volatile __local
                          float *) red_arr_mem_24944)[sext_i32_i64(squot32(local_tid_24936,
                                                                           32))] =
                            x_22299;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_24965;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_24936, 32) == 0 &&
                            slt32(local_tid_24936, iota_arg_18403 *
                                  squot32(segred_group_sizze_22291,
                                          segment_sizze_nonzzero_24933))) {
                            x_24952 = ((volatile __local
                                        bool *) red_arr_mem_24940)[sext_i32_i64(local_tid_24936)];
                            x_24953 = ((volatile __local
                                        int32_t *) red_arr_mem_24942)[sext_i32_i64(local_tid_24936)];
                            x_24954 = ((volatile __local
                                        float *) red_arr_mem_24944)[sext_i32_i64(local_tid_24936)];
                            if ((local_tid_24936 - squot32(local_tid_24936,
                                                           32) * 32) == 0) {
                                x_24949 = x_24952;
                                x_24950 = x_24953;
                                x_24951 = x_24954;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_24965 = 1;
                        while (slt32(skip_threads_24965, 32)) {
                            if (sle32(skip_threads_24965, local_tid_24936 -
                                      squot32(local_tid_24936, 32) * 32) &&
                                (squot32(local_tid_24936, 32) == 0 &&
                                 slt32(local_tid_24936, iota_arg_18403 *
                                       squot32(segred_group_sizze_22291,
                                               segment_sizze_nonzzero_24933)))) {
                                // read operands
                                {
                                    x_24949 = ((volatile __local
                                                bool *) red_arr_mem_24940)[sext_i32_i64(local_tid_24936 -
                                                                           skip_threads_24965)];
                                    x_24950 = ((volatile __local
                                                int32_t *) red_arr_mem_24942)[sext_i32_i64(local_tid_24936 -
                                                                              skip_threads_24965)];
                                    x_24951 = ((volatile __local
                                                float *) red_arr_mem_24944)[sext_i32_i64(local_tid_24936 -
                                                                            skip_threads_24965)];
                                }
                                // perform operation
                                {
                                    bool inactive_24966 =
                                         slt32(srem32(local_tid_24936 * 32 +
                                                      32 - 1, iota_arg_18403),
                                               local_tid_24936 * 32 + 32 - 1 -
                                               ((local_tid_24936 -
                                                 skip_threads_24965) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_24966) {
                                        x_24949 = x_24952;
                                        x_24950 = x_24953;
                                        x_24951 = x_24954;
                                    }
                                    if (!inactive_24966) {
                                        bool res_24955;
                                        int32_t res_24956;
                                        
                                        if (x_24949) {
                                            res_24955 = x_24949;
                                            res_24956 = x_24950;
                                        } else {
                                            bool x_24957 = x_24952 && x_24952;
                                            bool x_24958 = !x_24952;
                                            bool y_24959 = x_24949 && x_24958;
                                            bool res_24960 = x_24957 || y_24959;
                                            int32_t res_24961;
                                            
                                            if (x_24952) {
                                                res_24961 = x_24953;
                                            } else {
                                                res_24961 = x_24950;
                                            }
                                            res_24955 = res_24960;
                                            res_24956 = res_24961;
                                        }
                                        
                                        float res_24962 = x_24951 + x_24954;
                                        
                                        x_24949 = res_24955;
                                        x_24950 = res_24956;
                                        x_24951 = res_24962;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_24938, skip_threads_24965)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_24965, local_tid_24936 -
                                      squot32(local_tid_24936, 32) * 32) &&
                                (squot32(local_tid_24936, 32) == 0 &&
                                 slt32(local_tid_24936, iota_arg_18403 *
                                       squot32(segred_group_sizze_22291,
                                               segment_sizze_nonzzero_24933)))) {
                                // write result
                                {
                                    ((volatile __local
                                      bool *) red_arr_mem_24940)[sext_i32_i64(local_tid_24936)] =
                                        x_24949;
                                    x_24952 = x_24949;
                                    ((volatile __local
                                      int32_t *) red_arr_mem_24942)[sext_i32_i64(local_tid_24936)] =
                                        x_24950;
                                    x_24953 = x_24950;
                                    ((volatile __local
                                      float *) red_arr_mem_24944)[sext_i32_i64(local_tid_24936)] =
                                        x_24951;
                                    x_24954 = x_24951;
                                }
                            }
                            if (sle32(wave_sizze_24938, skip_threads_24965)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_24965 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_24936, 32) == 0 ||
                          !slt32(local_tid_24936, iota_arg_18403 *
                                 squot32(segred_group_sizze_22291,
                                         segment_sizze_nonzzero_24933)))) {
                        // read operands
                        {
                            x_22300 = x_22297;
                            x_22301 = x_22298;
                            x_22302 = x_22299;
                            x_22297 = ((__local
                                        bool *) red_arr_mem_24940)[sext_i32_i64(squot32(local_tid_24936,
                                                                                        32) -
                                                                   1)];
                            x_22298 = ((__local
                                        int32_t *) red_arr_mem_24942)[sext_i32_i64(squot32(local_tid_24936,
                                                                                           32) -
                                                                      1)];
                            x_22299 = ((__local
                                        float *) red_arr_mem_24944)[sext_i32_i64(squot32(local_tid_24936,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_24967 = slt32(srem32(local_tid_24936,
                                                               iota_arg_18403),
                                                        local_tid_24936 -
                                                        (squot32(local_tid_24936,
                                                                 32) * 32 - 1));
                            
                            if (inactive_24967) {
                                x_22297 = x_22300;
                                x_22298 = x_22301;
                                x_22299 = x_22302;
                            }
                            if (!inactive_24967) {
                                bool res_22303;
                                int32_t res_22304;
                                
                                if (x_22297) {
                                    res_22303 = x_22297;
                                    res_22304 = x_22298;
                                } else {
                                    bool x_22305 = x_22300 && x_22300;
                                    bool x_22306 = !x_22300;
                                    bool y_22307 = x_22297 && x_22306;
                                    bool res_22308 = x_22305 || y_22307;
                                    int32_t res_22309;
                                    
                                    if (x_22300) {
                                        res_22309 = x_22301;
                                    } else {
                                        res_22309 = x_22298;
                                    }
                                    res_22303 = res_22308;
                                    res_22304 = res_22309;
                                }
                                
                                float res_22310 = x_22299 + x_22302;
                                
                                x_22297 = res_22303;
                                x_22298 = res_22304;
                                x_22299 = res_22310;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              bool *) red_arr_mem_24940)[sext_i32_i64(local_tid_24936)] =
                                x_22297;
                            ((__local
                              int32_t *) red_arr_mem_24942)[sext_i32_i64(local_tid_24936)] =
                                x_22298;
                            ((__local
                              float *) red_arr_mem_24944)[sext_i32_i64(local_tid_24936)] =
                                x_22299;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_24936, 32) == 0) {
                        ((__local
                          bool *) red_arr_mem_24940)[sext_i32_i64(local_tid_24936)] =
                            x_22300;
                        ((__local
                          int32_t *) red_arr_mem_24942)[sext_i32_i64(local_tid_24936)] =
                            x_22301;
                        ((__local
                          float *) red_arr_mem_24944)[sext_i32_i64(local_tid_24936)] =
                            x_22302;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_24948 * squot32(segred_group_sizze_22291,
                                                    segment_sizze_nonzzero_24933) +
                      local_tid_24936, m_18055) && slt32(local_tid_24936,
                                                         squot32(segred_group_sizze_22291,
                                                                 segment_sizze_nonzzero_24933))) {
                ((__global bool *) mem_23886)[sext_i32_i64(virt_group_id_24948 *
                                              squot32(segred_group_sizze_22291,
                                                      segment_sizze_nonzzero_24933) +
                                              local_tid_24936)] = ((__local
                                                                    bool *) red_arr_mem_24940)[sext_i32_i64((local_tid_24936 +
                                                                                                             1) *
                                                                                               segment_sizze_nonzzero_24933 -
                                                                                               1)];
                ((__global
                  int32_t *) mem_23889)[sext_i32_i64(virt_group_id_24948 *
                                        squot32(segred_group_sizze_22291,
                                                segment_sizze_nonzzero_24933) +
                                        local_tid_24936)] = ((__local
                                                              int32_t *) red_arr_mem_24942)[sext_i32_i64((local_tid_24936 +
                                                                                                          1) *
                                                                                            segment_sizze_nonzzero_24933 -
                                                                                            1)];
                ((__global
                  float *) mem_23892)[sext_i32_i64(virt_group_id_24948 *
                                      squot32(segred_group_sizze_22291,
                                              segment_sizze_nonzzero_24933) +
                                      local_tid_24936)] = ((__local
                                                            float *) red_arr_mem_24944)[sext_i32_i64((local_tid_24936 +
                                                                                                      1) *
                                                                                        segment_sizze_nonzzero_24933 -
                                                                                        1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_22291
}
__kernel void remove_nanszisegmap_18605(__global int *global_failure,
                                        int32_t m_18040, int32_t n_18041,
                                        int32_t p_18042,
                                        int16_t nan_value_18043, __global
                                        unsigned char *images_mem_23188,
                                        __global unsigned char *mem_23196)
{
    #define segmap_group_sizze_18696 (remove_nanszisegmap_group_sizze_18612)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24064;
    int32_t local_tid_24065;
    int32_t group_sizze_24068;
    int32_t wave_sizze_24067;
    int32_t group_tid_24066;
    
    global_tid_24064 = get_global_id(0);
    local_tid_24065 = get_local_id(0);
    group_sizze_24068 = get_local_size(0);
    wave_sizze_24067 = LOCKSTEP_WIDTH;
    group_tid_24066 = get_group_id(0);
    
    int32_t phys_tid_18605;
    
    phys_tid_18605 = global_tid_24064;
    
    int32_t gtid_18602;
    
    gtid_18602 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24066) *
                                      sext_i32_i64(segmap_group_sizze_18696) +
                                      sext_i32_i64(local_tid_24065),
                                      sext_i32_i64(n_18041) *
                                      sext_i32_i64(p_18042)));
    
    int32_t gtid_18603;
    
    gtid_18603 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24066) *
                                      sext_i32_i64(segmap_group_sizze_18696) +
                                      sext_i32_i64(local_tid_24065) -
                                      squot64(sext_i32_i64(group_tid_24066) *
                                              sext_i32_i64(segmap_group_sizze_18696) +
                                              sext_i32_i64(local_tid_24065),
                                              sext_i32_i64(n_18041) *
                                              sext_i32_i64(p_18042)) *
                                      (sext_i32_i64(n_18041) *
                                       sext_i32_i64(p_18042)),
                                      sext_i32_i64(p_18042)));
    
    int32_t gtid_18604;
    
    gtid_18604 = sext_i64_i32(sext_i32_i64(group_tid_24066) *
        sext_i32_i64(segmap_group_sizze_18696) + sext_i32_i64(local_tid_24065) -
        squot64(sext_i32_i64(group_tid_24066) *
                sext_i32_i64(segmap_group_sizze_18696) +
                sext_i32_i64(local_tid_24065), sext_i32_i64(n_18041) *
                sext_i32_i64(p_18042)) * (sext_i32_i64(n_18041) *
                                          sext_i32_i64(p_18042)) -
        squot64(sext_i32_i64(group_tid_24066) *
                sext_i32_i64(segmap_group_sizze_18696) +
                sext_i32_i64(local_tid_24065) -
                squot64(sext_i32_i64(group_tid_24066) *
                        sext_i32_i64(segmap_group_sizze_18696) +
                        sext_i32_i64(local_tid_24065), sext_i32_i64(n_18041) *
                        sext_i32_i64(p_18042)) * (sext_i32_i64(n_18041) *
                                                  sext_i32_i64(p_18042)),
                sext_i32_i64(p_18042)) * sext_i32_i64(p_18042));
    if ((slt32(gtid_18602, m_18040) && slt32(gtid_18603, n_18041)) &&
        slt32(gtid_18604, p_18042)) {
        int16_t x_18701 = ((__global
                            int16_t *) images_mem_23188)[sext_i32_i64(gtid_18602) *
                                                         sext_i32_i64(p_18042 *
                                                         n_18041) +
                                                         sext_i32_i64(gtid_18603) *
                                                         sext_i32_i64(p_18042) +
                                                         sext_i32_i64(gtid_18604)];
        bool cond_18702 = x_18701 == nan_value_18043;
        float res_18703;
        
        if (cond_18702) {
            res_18703 = NAN;
        } else {
            float res_18704 = sitofp_i16_f32(x_18701);
            
            res_18703 = res_18704;
        }
        ((__global float *) mem_23196)[sext_i32_i64(gtid_18602) *
                                       sext_i32_i64(p_18042 * n_18041) +
                                       sext_i32_i64(gtid_18603) *
                                       sext_i32_i64(p_18042) +
                                       sext_i32_i64(gtid_18604)] = res_18703;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_18696
}
"""
# Start of values.py.

# Hacky parser/reader/writer for values written in Futhark syntax.
# Used for reading stdin when compiling standalone programs with the
# Python code generator.

import numpy as np
import string
import struct
import sys

class ReaderInput:
    def __init__(self, f):
        self.f = f
        self.lookahead_buffer = []

    def get_char(self):
        if len(self.lookahead_buffer) == 0:
            return self.f.read(1)
        else:
            c = self.lookahead_buffer[0]
            self.lookahead_buffer = self.lookahead_buffer[1:]
            return c

    def unget_char(self, c):
        self.lookahead_buffer = [c] + self.lookahead_buffer

    def get_chars(self, n):
        n1 = min(n, len(self.lookahead_buffer))
        s = b''.join(self.lookahead_buffer[:n1])
        self.lookahead_buffer = self.lookahead_buffer[n1:]
        n2 = n - n1
        if n2 > 0:
            s += self.f.read(n2)
        return s

    def peek_char(self):
        c = self.get_char()
        if c:
            self.unget_char(c)
        return c

def skip_spaces(f):
    c = f.get_char()
    while c != None:
        if c.isspace():
            c = f.get_char()
        elif c == b'-':
          # May be line comment.
          if f.peek_char() == b'-':
            # Yes, line comment. Skip to end of line.
            while (c != b'\n' and c != None):
              c = f.get_char()
          else:
            break
        else:
          break
    if c:
        f.unget_char(c)

def parse_specific_char(f, expected):
    got = f.get_char()
    if got != expected:
        f.unget_char(got)
        raise ValueError
    return True

def parse_specific_string(f, s):
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    read = []
    try:
        for c in bs:
            parse_specific_char(f, c)
            read.append(c)
        return True
    except ValueError:
        for c in read[::-1]:
            f.unget_char(c)
        raise

def optional(p, *args):
    try:
        return p(*args)
    except ValueError:
        return None

def optional_specific_string(f, s):
    c = f.peek_char()
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    if c == bs[0]:
        return parse_specific_string(f, s)
    else:
        return False

def sepBy(p, sep, *args):
    elems = []
    x = optional(p, *args)
    if x != None:
        elems += [x]
        while optional(sep, *args) != None:
            x = p(*args)
            elems += [x]
    return elems

# Assumes '0x' has already been read
def parse_hex_int(f):
    s = b''
    c = f.get_char()
    while c != None:
        if c in b'01234556789ABCDEFabcdef':
            s += c
            c = f.get_char()
        elif c == b'_':
            c = f.get_char() # skip _
        else:
            f.unget_char(c)
            break
    return str(int(s, 16)).encode('utf8') # ugh

def parse_int(f):
    s = b''
    c = f.get_char()
    if c == b'0' and f.peek_char() in b'xX':
        c = f.get_char() # skip X
        return parse_hex_int(f)
    else:
        while c != None:
            if c.isdigit():
                s += c
                c = f.get_char()
            elif c == b'_':
                c = f.get_char() # skip _
            else:
                f.unget_char(c)
                break
        if len(s) == 0:
            raise ValueError
        return s

def parse_int_signed(f):
    s = b''
    c = f.get_char()

    if c == b'-' and f.peek_char().isdigit():
      return c + parse_int(f)
    else:
      if c != b'+':
          f.unget_char(c)
      return parse_int(f)

def read_str_comma(f):
    skip_spaces(f)
    parse_specific_char(f, b',')
    return b','

def read_str_int(f, s):
    skip_spaces(f)
    x = int(parse_int_signed(f))
    optional_specific_string(f, s)
    return x

def read_str_uint(f, s):
    skip_spaces(f)
    x = int(parse_int(f))
    optional_specific_string(f, s)
    return x

def read_str_i8(f):
    return np.int8(read_str_int(f, 'i8'))
def read_str_i16(f):
    return np.int16(read_str_int(f, 'i16'))
def read_str_i32(f):
    return np.int32(read_str_int(f, 'i32'))
def read_str_i64(f):
    return np.int64(read_str_int(f, 'i64'))

def read_str_u8(f):
    return np.uint8(read_str_int(f, 'u8'))
def read_str_u16(f):
    return np.uint16(read_str_int(f, 'u16'))
def read_str_u32(f):
    return np.uint32(read_str_int(f, 'u32'))
def read_str_u64(f):
    return np.uint64(read_str_int(f, 'u64'))

def read_char(f):
    skip_spaces(f)
    parse_specific_char(f, b'\'')
    c = f.get_char()
    parse_specific_char(f, b'\'')
    return c

def read_str_hex_float(f, sign):
    int_part = parse_hex_int(f)
    parse_specific_char(f, b'.')
    frac_part = parse_hex_int(f)
    parse_specific_char(f, b'p')
    exponent = parse_int(f)

    int_val = int(int_part, 16)
    frac_val = float(int(frac_part, 16)) / (16 ** len(frac_part))
    exp_val = int(exponent)

    total_val = (int_val + frac_val) * (2.0 ** exp_val)
    if sign == b'-':
        total_val = -1 * total_val

    return float(total_val)


def read_str_decimal(f):
    skip_spaces(f)
    c = f.get_char()
    if (c == b'-'):
      sign = b'-'
    else:
      f.unget_char(c)
      sign = b''

    # Check for hexadecimal float
    c = f.get_char()
    if (c == '0' and (f.peek_char() in ['x', 'X'])):
        f.get_char()
        return read_str_hex_float(f, sign)
    else:
        f.unget_char(c)

    bef = optional(parse_int, f)
    if bef == None:
        bef = b'0'
        parse_specific_char(f, b'.')
        aft = parse_int(f)
    elif optional(parse_specific_char, f, b'.'):
        aft = parse_int(f)
    else:
        aft = b'0'
    if (optional(parse_specific_char, f, b'E') or
        optional(parse_specific_char, f, b'e')):
        expt = parse_int_signed(f)
    else:
        expt = b'0'
    return float(sign + bef + b'.' + aft + b'E' + expt)

def read_str_f32(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f32.nan')
        return np.float32(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f32.inf')
            return np.float32(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f32.inf')
               return np.float32(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f32')
               return x

def read_str_f64(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f64.nan')
        return np.float64(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f64.inf')
            return np.float64(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f64.inf')
               return np.float64(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f64')
               return x

def read_str_bool(f):
    skip_spaces(f)
    if f.peek_char() == b't':
        parse_specific_string(f, 'true')
        return True
    elif f.peek_char() == b'f':
        parse_specific_string(f, 'false')
        return False
    else:
        raise ValueError

def read_str_empty_array(f, type_name, rank):
    parse_specific_string(f, 'empty')
    parse_specific_char(f, b'(')
    dims = []
    for i in range(rank):
        parse_specific_string(f, '[')
        dims += [int(parse_int(f))]
        parse_specific_string(f, ']')
    if np.product(dims) != 0:
        raise ValueError
    parse_specific_string(f, type_name)
    parse_specific_char(f, b')')

    return tuple(dims)

def read_str_array_elems(f, elem_reader, type_name, rank):
    skip_spaces(f)
    try:
        parse_specific_char(f, b'[')
    except ValueError:
        return read_str_empty_array(f, type_name, rank)
    else:
        xs = sepBy(elem_reader, read_str_comma, f)
        skip_spaces(f)
        parse_specific_char(f, b']')
        return xs

def read_str_array_helper(f, elem_reader, type_name, rank):
    def nested_row_reader(_):
        return read_str_array_helper(f, elem_reader, type_name, rank-1)
    if rank == 1:
        row_reader = elem_reader
    else:
        row_reader = nested_row_reader
    return read_str_array_elems(f, row_reader, type_name, rank)

def expected_array_dims(l, rank):
  if rank > 1:
      n = len(l)
      if n == 0:
          elem = []
      else:
          elem = l[0]
      return [n] + expected_array_dims(elem, rank-1)
  else:
      return [len(l)]

def verify_array_dims(l, dims):
    if dims[0] != len(l):
        raise ValueError
    if len(dims) > 1:
        for x in l:
            verify_array_dims(x, dims[1:])

def read_str_array(f, elem_reader, type_name, rank, bt):
    elems = read_str_array_helper(f, elem_reader, type_name, rank)
    if type(elems) == tuple:
        # Empty array
        return np.empty(elems, dtype=bt)
    else:
        dims = expected_array_dims(elems, rank)
        verify_array_dims(elems, dims)
        return np.array(elems, dtype=bt)

################################################################################

READ_BINARY_VERSION = 2

# struct format specified at
# https://docs.python.org/2/library/struct.html#format-characters

def mk_bin_scalar_reader(t):
    def bin_reader(f):
        fmt = FUTHARK_PRIMTYPES[t]['bin_format']
        size = FUTHARK_PRIMTYPES[t]['size']
        return struct.unpack('<' + fmt, f.get_chars(size))[0]
    return bin_reader

read_bin_i8 = mk_bin_scalar_reader('i8')
read_bin_i16 = mk_bin_scalar_reader('i16')
read_bin_i32 = mk_bin_scalar_reader('i32')
read_bin_i64 = mk_bin_scalar_reader('i64')

read_bin_u8 = mk_bin_scalar_reader('u8')
read_bin_u16 = mk_bin_scalar_reader('u16')
read_bin_u32 = mk_bin_scalar_reader('u32')
read_bin_u64 = mk_bin_scalar_reader('u64')

read_bin_f32 = mk_bin_scalar_reader('f32')
read_bin_f64 = mk_bin_scalar_reader('f64')

read_bin_bool = mk_bin_scalar_reader('bool')

def read_is_binary(f):
    skip_spaces(f)
    c = f.get_char()
    if c == b'b':
        bin_version = read_bin_u8(f)
        if bin_version != READ_BINARY_VERSION:
            panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
                  bin_version, READ_BINARY_VERSION)
        return True
    else:
        f.unget_char(c)
        return False

FUTHARK_PRIMTYPES = {
    'i8':  {'binname' : b"  i8",
            'size' : 1,
            'bin_reader': read_bin_i8,
            'str_reader': read_str_i8,
            'bin_format': 'b',
            'numpy_type': np.int8 },

    'i16': {'binname' : b" i16",
            'size' : 2,
            'bin_reader': read_bin_i16,
            'str_reader': read_str_i16,
            'bin_format': 'h',
            'numpy_type': np.int16 },

    'i32': {'binname' : b" i32",
            'size' : 4,
            'bin_reader': read_bin_i32,
            'str_reader': read_str_i32,
            'bin_format': 'i',
            'numpy_type': np.int32 },

    'i64': {'binname' : b" i64",
            'size' : 8,
            'bin_reader': read_bin_i64,
            'str_reader': read_str_i64,
            'bin_format': 'q',
            'numpy_type': np.int64},

    'u8':  {'binname' : b"  u8",
            'size' : 1,
            'bin_reader': read_bin_u8,
            'str_reader': read_str_u8,
            'bin_format': 'B',
            'numpy_type': np.uint8 },

    'u16': {'binname' : b" u16",
            'size' : 2,
            'bin_reader': read_bin_u16,
            'str_reader': read_str_u16,
            'bin_format': 'H',
            'numpy_type': np.uint16 },

    'u32': {'binname' : b" u32",
            'size' : 4,
            'bin_reader': read_bin_u32,
            'str_reader': read_str_u32,
            'bin_format': 'I',
            'numpy_type': np.uint32 },

    'u64': {'binname' : b" u64",
            'size' : 8,
            'bin_reader': read_bin_u64,
            'str_reader': read_str_u64,
            'bin_format': 'Q',
            'numpy_type': np.uint64 },

    'f32': {'binname' : b" f32",
            'size' : 4,
            'bin_reader': read_bin_f32,
            'str_reader': read_str_f32,
            'bin_format': 'f',
            'numpy_type': np.float32 },

    'f64': {'binname' : b" f64",
            'size' : 8,
            'bin_reader': read_bin_f64,
            'str_reader': read_str_f64,
            'bin_format': 'd',
            'numpy_type': np.float64 },

    'bool': {'binname' : b"bool",
             'size' : 1,
             'bin_reader': read_bin_bool,
             'str_reader': read_str_bool,
             'bin_format': 'b',
             'numpy_type': np.bool }
}

def read_bin_read_type(f):
    read_binname = f.get_chars(4)

    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['binname'] == read_binname:
            return k
    panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname)

def numpy_type_to_type_name(t):
    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['numpy_type'] == t:
            return k
    raise Exception('Unknown Numpy type: {}'.format(t))

def read_bin_ensure_scalar(f, expected_type):
  dims = read_bin_i8(f)

  if dims != 0:
      panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n", dims)

  bin_type = read_bin_read_type(f)
  if bin_type != expected_type:
      panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
            expected_type, bin_type)

# ------------------------------------------------------------------------------
# General interface for reading Primitive Futhark Values
# ------------------------------------------------------------------------------

def read_scalar(f, ty):
    if read_is_binary(f):
        read_bin_ensure_scalar(f, ty)
        return FUTHARK_PRIMTYPES[ty]['bin_reader'](f)
    return FUTHARK_PRIMTYPES[ty]['str_reader'](f)

def read_array(f, expected_type, rank):
    if not read_is_binary(f):
        str_reader = FUTHARK_PRIMTYPES[expected_type]['str_reader']
        return read_str_array(f, str_reader, expected_type, rank,
                              FUTHARK_PRIMTYPES[expected_type]['numpy_type'])

    bin_rank = read_bin_u8(f)

    if bin_rank != rank:
        panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
              rank, bin_rank)

    bin_type_enum = read_bin_read_type(f)
    if expected_type != bin_type_enum:
        panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
              rank, expected_type, bin_rank, bin_type_enum)

    shape = []
    elem_count = 1
    for i in range(rank):
        bin_size = read_bin_u64(f)
        elem_count *= bin_size
        shape.append(bin_size)

    bin_fmt = FUTHARK_PRIMTYPES[bin_type_enum]['bin_format']

    # We first read the expected number of types into a bytestring,
    # then use np.fromstring.  This is because np.fromfile does not
    # work on things that are insufficiently file-like, like a network
    # stream.
    bytes = f.get_chars(elem_count * FUTHARK_PRIMTYPES[expected_type]['size'])
    arr = np.fromstring(bytes, dtype=FUTHARK_PRIMTYPES[bin_type_enum]['numpy_type'])
    arr.shape = shape

    return arr

if sys.version_info >= (3,0):
    input_reader = ReaderInput(sys.stdin.buffer)
else:
    input_reader = ReaderInput(sys.stdin)

import re

def read_value(type_desc, reader=input_reader):
    """Read a value of the given type.  The type is a string
representation of the Futhark type."""
    m = re.match(r'((?:\[\])*)([a-z0-9]+)$', type_desc)
    if m:
        dims = int(len(m.group(1))/2)
        basetype = m.group(2)
        assert basetype in FUTHARK_PRIMTYPES, "Unknown type: {}".format(type_desc)
        if dims > 0:
            return read_array(reader, basetype, dims)
        else:
            return read_scalar(reader, basetype)
        return (dims, basetype)

def end_of_input(entry, f=input_reader):
    skip_spaces(f)
    if f.get_char() != b'':
        panic(1, "Expected EOF on stdin after reading input for \"%s\".", entry)

def write_value_text(v, out=sys.stdout):
    if type(v) == np.uint8:
        out.write("%uu8" % v)
    elif type(v) == np.uint16:
        out.write("%uu16" % v)
    elif type(v) == np.uint32:
        out.write("%uu32" % v)
    elif type(v) == np.uint64:
        out.write("%uu64" % v)
    elif type(v) == np.int8:
        out.write("%di8" % v)
    elif type(v) == np.int16:
        out.write("%di16" % v)
    elif type(v) == np.int32:
        out.write("%di32" % v)
    elif type(v) == np.int64:
        out.write("%di64" % v)
    elif type(v) in [np.bool, np.bool_]:
        if v:
            out.write("true")
        else:
            out.write("false")
    elif type(v) == np.float32:
        if np.isnan(v):
            out.write('f32.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f32.inf')
            else:
                out.write('-f32.inf')
        else:
            out.write("%.6ff32" % v)
    elif type(v) == np.float64:
        if np.isnan(v):
            out.write('f64.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f64.inf')
            else:
                out.write('-f64.inf')
        else:
            out.write("%.6ff64" % v)
    elif type(v) == np.ndarray:
        if np.product(v.shape) == 0:
            tname = numpy_type_to_type_name(v.dtype)
            out.write('empty({}{})'.format(''.join(['[{}]'.format(d)
                                                    for d in v.shape]), tname))
        else:
            first = True
            out.write('[')
            for x in v:
                if not first: out.write(', ')
                first = False
                write_value(x, out=out)
            out.write(']')
    else:
        raise Exception("Cannot print value of type {}: {}".format(type(v), v))

type_strs = { np.dtype('int8'): b'  i8',
              np.dtype('int16'): b' i16',
              np.dtype('int32'): b' i32',
              np.dtype('int64'): b' i64',
              np.dtype('uint8'): b'  u8',
              np.dtype('uint16'): b' u16',
              np.dtype('uint32'): b' u32',
              np.dtype('uint64'): b' u64',
              np.dtype('float32'): b' f32',
              np.dtype('float64'): b' f64',
              np.dtype('bool'): b'bool'}

def construct_binary_value(v):
    t = v.dtype
    shape = v.shape

    elems = 1
    for d in shape:
        elems *= d

    num_bytes = 1 + 1 + 1 + 4 + len(shape) * 8 + elems * t.itemsize
    bytes = bytearray(num_bytes)
    bytes[0] = np.int8(ord('b'))
    bytes[1] = 2
    bytes[2] = np.int8(len(shape))
    bytes[3:7] = type_strs[t]

    for i in range(len(shape)):
        bytes[7+i*8:7+(i+1)*8] = np.int64(shape[i]).tostring()

    bytes[7+len(shape)*8:] = np.ascontiguousarray(v).tostring()

    return bytes

def write_value_binary(v, out=sys.stdout):
    if sys.version_info >= (3,0):
        out = out.buffer
    out.write(construct_binary_value(v))

def write_value(v, out=sys.stdout, binary=False):
    if binary:
        return write_value_binary(v, out=out)
    else:
        return write_value_text(v, out=out)

# End of values.py.
# Start of memory.py.

import ctypes as ct

def addressOffset(x, offset, bt):
  return ct.cast(ct.addressof(x.contents)+int(offset), ct.POINTER(bt))

def allocateMem(size):
  return ct.cast((ct.c_byte * max(0,size))(), ct.POINTER(ct.c_byte))

# Copy an array if its is not-None.  This is important for treating
# Numpy arrays as flat memory, but has some overhead.
def normaliseArray(x):
  if (x.base is x) or (x.base is None):
    return x
  else:
    return x.copy()

def unwrapArray(x):
  return normaliseArray(x).ctypes.data_as(ct.POINTER(ct.c_byte))

def createArray(x, shape):
  # HACK: np.ctypeslib.as_array may fail if the shape contains zeroes,
  # for some reason.
  if any(map(lambda x: x == 0, shape)):
      return np.ndarray(shape, dtype=x._type_)
  else:
      return np.ctypeslib.as_array(x, shape=shape)

def indexArray(x, offset, bt, nptype):
  return nptype(addressOffset(x, offset*ct.sizeof(bt), bt)[0])

def writeScalarArray(x, offset, v):
  ct.memmove(ct.addressof(x.contents)+int(offset)*ct.sizeof(v), ct.addressof(v), ct.sizeof(v))

# An opaque Futhark value.
class opaque(object):
  def __init__(self, desc, *payload):
    self.data = payload
    self.desc = desc

  def __repr__(self):
    return "<opaque Futhark value of type {}>".format(self.desc)

# End of memory.py.
# Start of panic.py.

def panic(exitcode, fmt, *args):
    sys.stderr.write('%s: ' % sys.argv[0])
    sys.stderr.write(fmt % args)
    sys.stderr.write('\n')
    sys.exit(exitcode)

# End of panic.py.
# Start of tuning.py

def read_tuning_file(kvs, f):
    for line in f.read().splitlines():
        size, value = line.split('=')
        kvs[size] = int(value)
    return kvs

# End of tuning.py.
# Start of scalar.py.

import numpy as np
import math
import struct

def intlit(t, x):
  if t == np.int8:
    return np.int8(x)
  elif t == np.int16:
    return np.int16(x)
  elif t == np.int32:
    return np.int32(x)
  else:
    return np.int64(x)

def signed(x):
  if type(x) == np.uint8:
    return np.int8(x)
  elif type(x) == np.uint16:
    return np.int16(x)
  elif type(x) == np.uint32:
    return np.int32(x)
  else:
    return np.int64(x)

def unsigned(x):
  if type(x) == np.int8:
    return np.uint8(x)
  elif type(x) == np.int16:
    return np.uint16(x)
  elif type(x) == np.int32:
    return np.uint32(x)
  else:
    return np.uint64(x)

def shlN(x,y):
  return x << y

def ashrN(x,y):
  return x >> y

# Python is so slow that we just make all the unsafe operations safe,
# always.

def sdivN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return x // y

def sdiv_upN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return (x+y-intlit(type(x), 1)) // y

def smodN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return x % y

def udivN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return signed(unsigned(x) // unsigned(y))

def udiv_upN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return signed((unsigned(x)+unsigned(y)-unsigned(intlit(type(x),1))) // unsigned(y))

def umodN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return np.floor_divide(np.abs(x), np.abs(y)) * np.sign(x) * np.sign(y)

def sremN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return np.remainder(np.abs(x), np.abs(y)) * np.sign(x)

def sminN(x,y):
  return min(x,y)

def smaxN(x,y):
  return max(x,y)

def uminN(x,y):
  return signed(min(unsigned(x),unsigned(y)))

def umaxN(x,y):
  return signed(max(unsigned(x),unsigned(y)))

def fminN(x,y):
  return min(x,y)

def fmaxN(x,y):
  return max(x,y)

def powN(x,y):
  return x ** y

def fpowN(x,y):
  return x ** y

def sleN(x,y):
  return x <= y

def sltN(x,y):
  return x < y

def uleN(x,y):
  return unsigned(x) <= unsigned(y)

def ultN(x,y):
  return unsigned(x) < unsigned(y)

def lshr8(x,y):
  return np.int8(np.uint8(x) >> np.uint8(y))

def lshr16(x,y):
  return np.int16(np.uint16(x) >> np.uint16(y))

def lshr32(x,y):
  return np.int32(np.uint32(x) >> np.uint32(y))

def lshr64(x,y):
  return np.int64(np.uint64(x) >> np.uint64(y))

def sext_T_i8(x):
  return np.int8(x)

def sext_T_i16(x):
  return np.int16(x)

def sext_T_i32(x):
  return np.int32(x)

def sext_T_i64(x):
  return np.int64(x)

def itob_T_bool(x):
  return np.bool(x)

def btoi_bool_i8(x):
  return np.int8(x)

def btoi_bool_i16(x):
  return np.int8(x)

def btoi_bool_i32(x):
  return np.int8(x)

def btoi_bool_i64(x):
  return np.int8(x)

def zext_i8_i8(x):
  return np.int8(np.uint8(x))

def zext_i8_i16(x):
  return np.int16(np.uint8(x))

def zext_i8_i32(x):
  return np.int32(np.uint8(x))

def zext_i8_i64(x):
  return np.int64(np.uint8(x))

def zext_i16_i8(x):
  return np.int8(np.uint16(x))

def zext_i16_i16(x):
  return np.int16(np.uint16(x))

def zext_i16_i32(x):
  return np.int32(np.uint16(x))

def zext_i16_i64(x):
  return np.int64(np.uint16(x))

def zext_i32_i8(x):
  return np.int8(np.uint32(x))

def zext_i32_i16(x):
  return np.int16(np.uint32(x))

def zext_i32_i32(x):
  return np.int32(np.uint32(x))

def zext_i32_i64(x):
  return np.int64(np.uint32(x))

def zext_i64_i8(x):
  return np.int8(np.uint64(x))

def zext_i64_i16(x):
  return np.int16(np.uint64(x))

def zext_i64_i32(x):
  return np.int32(np.uint64(x))

def zext_i64_i64(x):
  return np.int64(np.uint64(x))

sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
sdiv_up8 = sdiv1_up6 = sdiv_up32 = sdiv_up64 = sdiv_upN
sdiv_safe8 = sdiv1_safe6 = sdiv_safe32 = sdiv_safe64 = sdivN
sdiv_up_safe8 = sdiv_up1_safe6 = sdiv_up_safe32 = sdiv_up_safe64 = sdiv_upN
smod8 = smod16 = smod32 = smod64 = smodN
smod_safe8 = smod_safe16 = smod_safe32 = smod_safe64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
udiv_up8 = udiv_up16 = udiv_up32 = udiv_up64 = udivN
udiv_safe8 = udiv_safe16 = udiv_safe32 = udiv_safe64 = udiv_upN
udiv_up_safe8 = udiv_up_safe16 = udiv_up_safe32 = udiv_up_safe64 = udiv_upN
umod8 = umod16 = umod32 = umod64 = umodN
umod_safe8 = umod_safe16 = umod_safe32 = umod_safe64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
squot_safe8 = squot_safe16 = squot_safe32 = squot_safe64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
srem_safe8 = srem_safe16 = srem_safe32 = srem_safe64 = sremN

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
smax8 = smax16 = smax32 = smax64 = smaxN
smin8 = smin16 = smin32 = smin64 = sminN
umax8 = umax16 = umax32 = umax64 = umaxN
umin8 = umin16 = umin32 = umin64 = uminN
pow8 = pow16 = pow32 = pow64 = powN
fpow32 = fpow64 = fpowN
fmax32 = fmax64 = fmaxN
fmin32 = fmin64 = fminN
sle8 = sle16 = sle32 = sle64 = sleN
slt8 = slt16 = slt32 = slt64 = sltN
ule8 = ule16 = ule32 = ule64 = uleN
ult8 = ult16 = ult32 = ult64 = ultN
sext_i8_i8 = sext_i16_i8 = sext_i32_i8 = sext_i64_i8 = sext_T_i8
sext_i8_i16 = sext_i16_i16 = sext_i32_i16 = sext_i64_i16 = sext_T_i16
sext_i8_i32 = sext_i16_i32 = sext_i32_i32 = sext_i64_i32 = sext_T_i32
sext_i8_i64 = sext_i16_i64 = sext_i32_i64 = sext_i64_i64 = sext_T_i64
itob_i8_bool = itob_i16_bool = itob_i32_bool = itob_i64_bool = itob_T_bool

def clz_T(x):
  n = np.int32(0)
  bits = x.itemsize * 8
  for i in range(bits):
    if x < 0:
      break
    n += 1
    x <<= np.int8(1)
  return n

def ctz_T(x):
  n = np.int32(0)
  bits = x.itemsize * 8
  for i in range(bits):
    if (x & 1) == 1:
      break
    n += 1
    x >>= np.int8(1)
  return n

def popc_T(x):
  c = np.int32(0)
  while x != 0:
    x &= x - np.int8(1)
    c += np.int8(1)
  return c

futhark_popc8 = futhark_popc16 = futhark_popc32 = futhark_popc64 = popc_T
futhark_clzz8 = futhark_clzz16 = futhark_clzz32 = futhark_clzz64 = clz_T
futhark_ctzz8 = futhark_ctzz16 = futhark_ctzz32 = futhark_ctzz64 = ctz_T

def ssignum(x):
  return np.sign(x)

def usignum(x):
  if x < 0:
    return ssignum(-x)
  else:
    return ssignum(x)

def sitofp_T_f32(x):
  return np.float32(x)
sitofp_i8_f32 = sitofp_i16_f32 = sitofp_i32_f32 = sitofp_i64_f32 = sitofp_T_f32

def sitofp_T_f64(x):
  return np.float64(x)
sitofp_i8_f64 = sitofp_i16_f64 = sitofp_i32_f64 = sitofp_i64_f64 = sitofp_T_f64

def uitofp_T_f32(x):
  return np.float32(unsigned(x))
uitofp_i8_f32 = uitofp_i16_f32 = uitofp_i32_f32 = uitofp_i64_f32 = uitofp_T_f32

def uitofp_T_f64(x):
  return np.float64(unsigned(x))
uitofp_i8_f64 = uitofp_i16_f64 = uitofp_i32_f64 = uitofp_i64_f64 = uitofp_T_f64

def fptosi_T_i8(x):
  return np.int8(np.trunc(x))
fptosi_f32_i8 = fptosi_f64_i8 = fptosi_T_i8

def fptosi_T_i16(x):
  return np.int16(np.trunc(x))
fptosi_f32_i16 = fptosi_f64_i16 = fptosi_T_i16

def fptosi_T_i32(x):
  return np.int32(np.trunc(x))
fptosi_f32_i32 = fptosi_f64_i32 = fptosi_T_i32

def fptosi_T_i64(x):
  return np.int64(np.trunc(x))
fptosi_f32_i64 = fptosi_f64_i64 = fptosi_T_i64

def fptoui_T_i8(x):
  return np.uint8(np.trunc(x))
fptoui_f32_i8 = fptoui_f64_i8 = fptoui_T_i8

def fptoui_T_i16(x):
  return np.uint16(np.trunc(x))
fptoui_f32_i16 = fptoui_f64_i16 = fptoui_T_i16

def fptoui_T_i32(x):
  return np.uint32(np.trunc(x))
fptoui_f32_i32 = fptoui_f64_i32 = fptoui_T_i32

def fptoui_T_i64(x):
  return np.uint64(np.trunc(x))
fptoui_f32_i64 = fptoui_f64_i64 = fptoui_T_i64

def fpconv_f32_f64(x):
  return np.float64(x)

def fpconv_f64_f32(x):
  return np.float32(x)

def futhark_mul_hi8(a, b):
  a = np.uint64(np.uint8(a))
  b = np.uint64(np.uint8(b))
  return np.int8((a*b) >> np.uint64(8))

def futhark_mul_hi16(a, b):
  a = np.uint64(np.uint16(a))
  b = np.uint64(np.uint16(b))
  return np.int16((a*b) >> np.uint64(16))

def futhark_mul_hi32(a, b):
  a = np.uint64(np.uint32(a))
  b = np.uint64(np.uint32(b))
  return np.int32((a*b) >> np.uint64(32))

# This one is done with arbitrary-precision integers.
def futhark_mul_hi64(a, b):
  a = int(np.uint64(a))
  b = int(np.uint64(b))
  return np.int64(np.uint64(a*b >> 64))

def futhark_mad_hi8(a, b, c):
  return futhark_mul_hi8(a,b) + c

def futhark_mad_hi16(a, b, c):
  return futhark_mul_hi16(a,b) + c

def futhark_mad_hi32(a, b, c):
  return futhark_mul_hi32(a,b) + c

def futhark_mad_hi64(a, b, c):
  return futhark_mul_hi64(a,b) + c

def futhark_log64(x):
  return np.float64(np.log(x))

def futhark_log2_64(x):
  return np.float64(np.log2(x))

def futhark_log10_64(x):
  return np.float64(np.log10(x))

def futhark_sqrt64(x):
  return np.sqrt(x)

def futhark_exp64(x):
  return np.exp(x)

def futhark_cos64(x):
  return np.cos(x)

def futhark_sin64(x):
  return np.sin(x)

def futhark_tan64(x):
  return np.tan(x)

def futhark_acos64(x):
  return np.arccos(x)

def futhark_asin64(x):
  return np.arcsin(x)

def futhark_atan64(x):
  return np.arctan(x)

def futhark_cosh64(x):
  return np.cosh(x)

def futhark_sinh64(x):
  return np.sinh(x)

def futhark_tanh64(x):
  return np.tanh(x)

def futhark_acosh64(x):
  return np.arccosh(x)

def futhark_asinh64(x):
  return np.arcsinh(x)

def futhark_atanh64(x):
  return np.arctanh(x)

def futhark_atan2_64(x, y):
  return np.arctan2(x, y)

def futhark_gamma64(x):
  return np.float64(math.gamma(x))

def futhark_lgamma64(x):
  return np.float64(math.lgamma(x))

def futhark_round64(x):
  return np.round(x)

def futhark_ceil64(x):
  return np.ceil(x)

def futhark_floor64(x):
  return np.floor(x)

def futhark_isnan64(x):
  return np.isnan(x)

def futhark_isinf64(x):
  return np.isinf(x)

def futhark_to_bits64(x):
  s = struct.pack('>d', x)
  return np.int64(struct.unpack('>q', s)[0])

def futhark_from_bits64(x):
  s = struct.pack('>q', x)
  return np.float64(struct.unpack('>d', s)[0])

def futhark_log32(x):
  return np.float32(np.log(x))

def futhark_log2_32(x):
  return np.float32(np.log2(x))

def futhark_log10_32(x):
  return np.float32(np.log10(x))

def futhark_sqrt32(x):
  return np.float32(np.sqrt(x))

def futhark_exp32(x):
  return np.exp(x)

def futhark_cos32(x):
  return np.cos(x)

def futhark_sin32(x):
  return np.sin(x)

def futhark_tan32(x):
  return np.tan(x)

def futhark_acos32(x):
  return np.arccos(x)

def futhark_asin32(x):
  return np.arcsin(x)

def futhark_atan32(x):
  return np.arctan(x)

def futhark_cosh32(x):
  return np.cosh(x)

def futhark_sinh32(x):
  return np.sinh(x)

def futhark_tanh32(x):
  return np.tanh(x)

def futhark_acosh32(x):
  return np.arccosh(x)

def futhark_asinh32(x):
  return np.arcsinh(x)

def futhark_atanh32(x):
  return np.arctanh(x)

def futhark_atan2_32(x, y):
  return np.arctan2(x, y)

def futhark_gamma32(x):
  return np.float32(math.gamma(x))

def futhark_lgamma32(x):
  return np.float32(math.lgamma(x))

def futhark_round32(x):
  return np.round(x)

def futhark_ceil32(x):
  return np.ceil(x)

def futhark_floor32(x):
  return np.floor(x)

def futhark_isnan32(x):
  return np.isnan(x)

def futhark_isinf32(x):
  return np.isinf(x)

def futhark_to_bits32(x):
  s = struct.pack('>f', x)
  return np.int32(struct.unpack('>l', s)[0])

def futhark_from_bits32(x):
  s = struct.pack('>l', x)
  return np.float32(struct.unpack('>f', s)[0])

def futhark_lerp32(v0, v1, t):
  return v0 + (v1-v0)*t

def futhark_lerp64(v0, v1, t):
  return v0 + (v1-v0)*t

def futhark_mad32(a, b, c):
  return a * b + c

def futhark_mad64(a, b, c):
  return a * b + c

def futhark_fma32(a, b, c):
  return a * b + c

def futhark_fma64(a, b, c):
  return a * b + c

# End of scalar.py.
class bfastfinal:
  entry_points = {"main": (["i32", "i32", "i32", "f32", "f32", "f32", "[]i32",
                            "[][]f32"], ["[]i32", "[]f32"]),
                  "remove_nans": (["i16", "[][][]i16"], ["[][][]f32"]),
                  "reshapeTransp": (["[][][]f32"], ["[][]f32"])}
  def __init__(self, command_queue=None, interactive=False,
               platform_pref=preferred_platform, device_pref=preferred_device,
               default_group_size=default_group_size,
               default_num_groups=default_num_groups,
               default_tile_size=default_tile_size,
               default_threshold=default_threshold, sizes=sizes):
    size_heuristics=[("NVIDIA CUDA", cl.device_type.GPU, "lockstep_width",
      lambda device: np.int32(32)), ("AMD Accelerated Parallel Processing",
                                     cl.device_type.GPU, "lockstep_width",
                                     lambda device: np.int32(32)), ("",
                                                                    cl.device_type.GPU,
                                                                    "lockstep_width",
                                                                    lambda device: np.int32(1)),
     ("", cl.device_type.GPU, "num_groups",
      lambda device: (np.int32(4) * device.get_info(getattr(cl.device_info,
                                                            "MAX_COMPUTE_UNITS")))),
     ("", cl.device_type.GPU, "group_size", lambda device: np.int32(256)), ("",
                                                                            cl.device_type.GPU,
                                                                            "tile_size",
                                                                            lambda device: np.int32(32)),
     ("", cl.device_type.GPU, "threshold", lambda device: np.int32(32768)), ("",
                                                                             cl.device_type.CPU,
                                                                             "lockstep_width",
                                                                             lambda device: np.int32(1)),
     ("", cl.device_type.CPU, "num_groups",
      lambda device: device.get_info(getattr(cl.device_info, "MAX_COMPUTE_UNITS"))),
     ("", cl.device_type.CPU, "group_size", lambda device: np.int32(32)), ("",
                                                                           cl.device_type.CPU,
                                                                           "tile_size",
                                                                           lambda device: np.int32(4)),
     ("", cl.device_type.CPU, "threshold",
      lambda device: device.get_info(getattr(cl.device_info, "MAX_COMPUTE_UNITS")))]
    self.global_failure_args_max = 2
    self.failure_msgs=["Index [{}] out of bounds for array of shape [{}].\n-> #0  helpers.fut:52:16-19\n   #1  helpers.fut:72:15-33\n   #2  bfastfinal.fut:52:35-50\n   #3  bfastfinal.fut:19:1-147:20\n"]
    program = initialise_opencl_object(self,
                                       program_src=fut_opencl_src,
                                       command_queue=command_queue,
                                       interactive=interactive,
                                       platform_pref=platform_pref,
                                       device_pref=device_pref,
                                       default_group_size=default_group_size,
                                       default_num_groups=default_num_groups,
                                       default_tile_size=default_tile_size,
                                       default_threshold=default_threshold,
                                       size_heuristics=size_heuristics,
                                       required_types=["i16", "i32", "i64", "f32", "bool", "cert"],
                                       user_sizes=sizes,
                                       all_sizes={"builtin#replicate_f32.group_size_24501": {"class": "group_size",
                                                                                   "value": None},
                                        "builtin#replicate_i32.group_size_24510": {"class": "group_size",
                                                                                   "value": None},
                                        "main.group_size_24208": {"class": "group_size", "value": None},
                                        "main.group_size_24585": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_18804": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19004": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19157": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19204": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19254": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19288": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19805": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20032": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20089": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20153": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20263": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20501": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20661": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20701": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20812": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_21082": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_21331": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_21436": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_21566": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_21697": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_22056": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_22203": {"class": "group_size", "value": None},
                                        "main.segmap_num_groups_19206": {"class": "num_groups", "value": None},
                                        "main.segmap_num_groups_19256": {"class": "num_groups", "value": None},
                                        "main.segmap_num_groups_20503": {"class": "num_groups", "value": None},
                                        "main.segmap_num_groups_20663": {"class": "num_groups", "value": None},
                                        "main.segmap_num_groups_20814": {"class": "num_groups", "value": None},
                                        "main.segmap_num_groups_22205": {"class": "num_groups", "value": None},
                                        "main.segred_group_size_19323": {"class": "group_size", "value": None},
                                        "main.segred_group_size_20571": {"class": "group_size", "value": None},
                                        "main.segred_group_size_20727": {"class": "group_size", "value": None},
                                        "main.segred_group_size_20876": {"class": "group_size", "value": None},
                                        "main.segred_group_size_21455": {"class": "group_size", "value": None},
                                        "main.segred_group_size_21477": {"class": "group_size", "value": None},
                                        "main.segred_group_size_21551": {"class": "group_size", "value": None},
                                        "main.segred_group_size_21611": {"class": "group_size", "value": None},
                                        "main.segred_group_size_22102": {"class": "group_size", "value": None},
                                        "main.segred_num_groups_19325": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_20573": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_20729": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_20878": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_21457": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_21479": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_21553": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_21613": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_22104": {"class": "num_groups", "value": None},
                                        "main.segscan_group_size_21196": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_22161": {"class": "group_size", "value": None},
                                        "main.segscan_num_groups_21198": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_22163": {"class": "num_groups", "value": None},
                                        "main.suff_intra_par_10": {"class": "threshold ()", "value": 32},
                                        "main.suff_intra_par_14": {"class": "threshold (!main.suff_intra_par_10)",
                                                                   "value": 32},
                                        "main.suff_intra_par_24": {"class": "threshold ()", "value": 32},
                                        "main.suff_intra_par_28": {"class": "threshold (!main.suff_outer_par_27)",
                                                                   "value": 32},
                                        "main.suff_intra_par_32": {"class": "threshold ()", "value": 32},
                                        "main.suff_outer_par_17": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_18": {"class": "threshold (!main.suff_outer_par_17)",
                                                                   "value": None},
                                        "main.suff_outer_par_19": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_20": {"class": "threshold (!main.suff_outer_par_19)",
                                                                   "value": None},
                                        "main.suff_outer_par_21": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_22": {"class": "threshold (!main.suff_outer_par_21)",
                                                                   "value": None},
                                        "main.suff_outer_par_27": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_29": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_6": {"class": "threshold ()", "value": None},
                                        "main.suff_outer_par_7": {"class": "threshold (!main.suff_outer_par_6)",
                                                                  "value": None},
                                        "main.suff_outer_par_8": {"class": "threshold (!main.suff_outer_par_7 !main.suff_outer_par_6)",
                                                                  "value": None},
                                        "main.tile_size_22499": {"class": "tile_size", "value": None},
                                        "main.tile_size_22821": {"class": "tile_size", "value": None},
                                        "remove_nans.segmap_group_size_18612": {"class": "group_size", "value": None}})
    self.builtinzhreplicate_f32zireplicate_24498_var = program.builtinzhreplicate_f32zireplicate_24498
    self.builtinzhreplicate_i32zireplicate_24507_var = program.builtinzhreplicate_i32zireplicate_24507
    self.gpu_map_transpose_f32_var = program.gpu_map_transpose_f32
    self.gpu_map_transpose_f32_low_height_var = program.gpu_map_transpose_f32_low_height
    self.gpu_map_transpose_f32_low_width_var = program.gpu_map_transpose_f32_low_width
    self.gpu_map_transpose_f32_small_var = program.gpu_map_transpose_f32_small
    self.mainzicopy_24205_var = program.mainzicopy_24205
    self.mainzicopy_24582_var = program.mainzicopy_24582
    self.mainziscan_stage1_21202_var = program.mainziscan_stage1_21202
    self.mainziscan_stage1_22167_var = program.mainziscan_stage1_22167
    self.mainziscan_stage2_21202_var = program.mainziscan_stage2_21202
    self.mainziscan_stage2_22167_var = program.mainziscan_stage2_22167
    self.mainziscan_stage3_21202_var = program.mainziscan_stage3_21202
    self.mainziscan_stage3_22167_var = program.mainziscan_stage3_22167
    self.mainzisegmap_18799_var = program.mainzisegmap_18799
    self.mainzisegmap_18999_var = program.mainzisegmap_18999
    self.mainzisegmap_19152_var = program.mainzisegmap_19152
    self.mainzisegmap_19201_var = program.mainzisegmap_19201
    self.mainzisegmap_19249_var = program.mainzisegmap_19249
    self.mainzisegmap_19281_var = program.mainzisegmap_19281
    self.mainzisegmap_19798_var = program.mainzisegmap_19798
    self.mainzisegmap_20027_var = program.mainzisegmap_20027
    self.mainzisegmap_20084_var = program.mainzisegmap_20084
    self.mainzisegmap_20150_var = program.mainzisegmap_20150
    self.mainzisegmap_20258_var = program.mainzisegmap_20258
    self.mainzisegmap_20498_var = program.mainzisegmap_20498
    self.mainzisegmap_20658_var = program.mainzisegmap_20658
    self.mainzisegmap_20696_var = program.mainzisegmap_20696
    self.mainzisegmap_20809_var = program.mainzisegmap_20809
    self.mainzisegmap_21077_var = program.mainzisegmap_21077
    self.mainzisegmap_21328_var = program.mainzisegmap_21328
    self.mainzisegmap_21433_var = program.mainzisegmap_21433
    self.mainzisegmap_21563_var = program.mainzisegmap_21563
    self.mainzisegmap_21694_var = program.mainzisegmap_21694
    self.mainzisegmap_22053_var = program.mainzisegmap_22053
    self.mainzisegmap_22200_var = program.mainzisegmap_22200
    self.mainzisegmap_intragroup_19569_var = program.mainzisegmap_intragroup_19569
    self.mainzisegmap_intragroup_19921_var = program.mainzisegmap_intragroup_19921
    self.mainzisegmap_intragroup_20961_var = program.mainzisegmap_intragroup_20961
    self.mainzisegmap_intragroup_21326_var = program.mainzisegmap_intragroup_21326
    self.mainzisegmap_intragroup_21740_var = program.mainzisegmap_intragroup_21740
    self.mainzisegmap_intragroup_22505_var = program.mainzisegmap_intragroup_22505
    self.mainzisegmap_intragroup_22827_var = program.mainzisegmap_intragroup_22827
    self.mainzisegred_large_19329_var = program.mainzisegred_large_19329
    self.mainzisegred_large_20577_var = program.mainzisegred_large_20577
    self.mainzisegred_large_20733_var = program.mainzisegred_large_20733
    self.mainzisegred_large_20882_var = program.mainzisegred_large_20882
    self.mainzisegred_large_21461_var = program.mainzisegred_large_21461
    self.mainzisegred_large_21483_var = program.mainzisegred_large_21483
    self.mainzisegred_large_21617_var = program.mainzisegred_large_21617
    self.mainzisegred_large_22108_var = program.mainzisegred_large_22108
    self.mainzisegred_nonseg_21559_var = program.mainzisegred_nonseg_21559
    self.mainzisegred_small_19329_var = program.mainzisegred_small_19329
    self.mainzisegred_small_20577_var = program.mainzisegred_small_20577
    self.mainzisegred_small_20733_var = program.mainzisegred_small_20733
    self.mainzisegred_small_20882_var = program.mainzisegred_small_20882
    self.mainzisegred_small_21461_var = program.mainzisegred_small_21461
    self.mainzisegred_small_21483_var = program.mainzisegred_small_21483
    self.mainzisegred_small_21617_var = program.mainzisegred_small_21617
    self.mainzisegred_small_22108_var = program.mainzisegred_small_22108
    self.remove_nanszisegmap_18605_var = program.remove_nanszisegmap_18605
    self.constants = {}
    mainzicounter_mem_24139 = np.zeros(10240, dtype=np.int32)
    static_mem_25036 = opencl_alloc(self, 40960, "static_mem_25036")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25036,
                      normaliseArray(mainzicounter_mem_24139),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24139 = static_mem_25036
    mainzicounter_mem_24292 = np.zeros(10240, dtype=np.int32)
    static_mem_25039 = opencl_alloc(self, 40960, "static_mem_25039")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25039,
                      normaliseArray(mainzicounter_mem_24292),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24292 = static_mem_25039
    mainzicounter_mem_24372 = np.zeros(10240, dtype=np.int32)
    static_mem_25040 = opencl_alloc(self, 40960, "static_mem_25040")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25040,
                      normaliseArray(mainzicounter_mem_24372),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24372 = static_mem_25040
    mainzicounter_mem_24459 = np.zeros(10240, dtype=np.int32)
    static_mem_25041 = opencl_alloc(self, 40960, "static_mem_25041")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25041,
                      normaliseArray(mainzicounter_mem_24459),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24459 = static_mem_25041
    mainzicounter_mem_24643 = np.zeros(10240, dtype=np.int32)
    static_mem_25042 = opencl_alloc(self, 40960, "static_mem_25042")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25042,
                      normaliseArray(mainzicounter_mem_24643),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24643 = static_mem_25042
    mainzicounter_mem_24702 = np.zeros(10240, dtype=np.int32)
    static_mem_25043 = opencl_alloc(self, 40960, "static_mem_25043")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25043,
                      normaliseArray(mainzicounter_mem_24702),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24702 = static_mem_25043
    mainzicounter_mem_24742 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_25044 = opencl_alloc(self, 40, "static_mem_25044")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_25044,
                      normaliseArray(mainzicounter_mem_24742),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24742 = static_mem_25044
    mainzicounter_mem_24806 = np.zeros(10240, dtype=np.int32)
    static_mem_25046 = opencl_alloc(self, 40960, "static_mem_25046")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25046,
                      normaliseArray(mainzicounter_mem_24806),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24806 = static_mem_25046
    mainzicounter_mem_24979 = np.zeros(10240, dtype=np.int32)
    static_mem_25048 = opencl_alloc(self, 40960, "static_mem_25048")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25048,
                      normaliseArray(mainzicounter_mem_24979),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24979 = static_mem_25048
  def futhark_builtinzhgpu_map_transpose_f32(self, destmem_0, destoffset_1,
                                             srcmem_2, srcoffset_3,
                                             num_arrays_4, x_elems_5,
                                             y_elems_6):
    if ((num_arrays_4 == np.int32(0)) or ((x_elems_5 == np.int32(0)) or (y_elems_6 == np.int32(0)))):
      pass
    else:
      muly_8 = squot32(np.int32(16), x_elems_5)
      mulx_7 = squot32(np.int32(16), y_elems_6)
      if ((num_arrays_4 == np.int32(1)) and ((x_elems_5 == np.int32(1)) or (y_elems_6 == np.int32(1)))):
        if (sext_i32_i64(((x_elems_5 * y_elems_6) * np.int32(4))) != 0):
          cl.enqueue_copy(self.queue, destmem_0, srcmem_2,
                          dest_offset=np.long(sext_i32_i64(destoffset_1)),
                          src_offset=np.long(sext_i32_i64(srcoffset_3)),
                          byte_count=np.long(sext_i32_i64(((x_elems_5 * y_elems_6) * np.int32(4)))))
        if synchronous:
          sync(self)
      else:
        if (sle32(x_elems_5, np.int32(8)) and slt32(np.int32(16), y_elems_6)):
          if ((((1 * (np.long(sdiv_up32(x_elems_5,
                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(sdiv_up32(sdiv_up32(y_elems_6,
                                                                                                                muly_8),
                                                                                                      np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
            self.gpu_map_transpose_f32_low_width_var.set_args(cl.LocalMemory(np.long(np.int64(1088))),
                                                              np.int32(destoffset_1),
                                                              np.int32(srcoffset_3),
                                                              np.int32(num_arrays_4),
                                                              np.int32(x_elems_5),
                                                              np.int32(y_elems_6),
                                                              np.int32(mulx_7),
                                                              np.int32(muly_8),
                                                              destmem_0,
                                                              srcmem_2)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.gpu_map_transpose_f32_low_width_var,
                                       ((np.long(sdiv_up32(x_elems_5,
                                                           np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(sdiv_up32(sdiv_up32(y_elems_6,
                                                                     muly_8),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                       (np.long(np.int32(16)),
                                        np.long(np.int32(16)),
                                        np.long(np.int32(1))))
            if synchronous:
              sync(self)
        else:
          if (sle32(y_elems_6, np.int32(8)) and slt32(np.int32(16), x_elems_5)):
            if ((((1 * (np.long(sdiv_up32(sdiv_up32(x_elems_5, mulx_7),
                                          np.int32(16))) * np.long(np.int32(16)))) * (np.long(sdiv_up32(y_elems_6,
                                                                                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
              self.gpu_map_transpose_f32_low_height_var.set_args(cl.LocalMemory(np.long(np.int64(1088))),
                                                                 np.int32(destoffset_1),
                                                                 np.int32(srcoffset_3),
                                                                 np.int32(num_arrays_4),
                                                                 np.int32(x_elems_5),
                                                                 np.int32(y_elems_6),
                                                                 np.int32(mulx_7),
                                                                 np.int32(muly_8),
                                                                 destmem_0,
                                                                 srcmem_2)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.gpu_map_transpose_f32_low_height_var,
                                         ((np.long(sdiv_up32(sdiv_up32(x_elems_5,
                                                                       mulx_7),
                                                             np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(sdiv_up32(y_elems_6,
                                                             np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                         (np.long(np.int32(16)),
                                          np.long(np.int32(16)),
                                          np.long(np.int32(1))))
              if synchronous:
                sync(self)
          else:
            if (sle32(x_elems_5, np.int32(8)) and sle32(y_elems_6,
                                                        np.int32(8))):
              if ((1 * (np.long(sdiv_up32(((num_arrays_4 * x_elems_5) * y_elems_6),
                                          np.int32(256))) * np.long(np.int32(256)))) != 0):
                self.gpu_map_transpose_f32_small_var.set_args(cl.LocalMemory(np.long(np.int64(1))),
                                                              np.int32(destoffset_1),
                                                              np.int32(srcoffset_3),
                                                              np.int32(num_arrays_4),
                                                              np.int32(x_elems_5),
                                                              np.int32(y_elems_6),
                                                              np.int32(mulx_7),
                                                              np.int32(muly_8),
                                                              destmem_0,
                                                              srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.gpu_map_transpose_f32_small_var,
                                           ((np.long(sdiv_up32(((num_arrays_4 * x_elems_5) * y_elems_6),
                                                               np.int32(256))) * np.long(np.int32(256))),),
                                           (np.long(np.int32(256)),))
                if synchronous:
                  sync(self)
            else:
              if ((((1 * (np.long(sdiv_up32(x_elems_5,
                                            np.int32(32))) * np.long(np.int32(32)))) * (np.long(sdiv_up32(y_elems_6,
                                                                                                          np.int32(32))) * np.long(np.int32(8)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
                self.gpu_map_transpose_f32_var.set_args(cl.LocalMemory(np.long(np.int64(4224))),
                                                        np.int32(destoffset_1),
                                                        np.int32(srcoffset_3),
                                                        np.int32(num_arrays_4),
                                                        np.int32(x_elems_5),
                                                        np.int32(y_elems_6),
                                                        np.int32(mulx_7),
                                                        np.int32(muly_8),
                                                        destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.gpu_map_transpose_f32_var,
                                           ((np.long(sdiv_up32(x_elems_5,
                                                               np.int32(32))) * np.long(np.int32(32))),
                                            (np.long(sdiv_up32(y_elems_6,
                                                               np.int32(32))) * np.long(np.int32(8))),
                                            (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                           (np.long(np.int32(32)),
                                            np.long(np.int32(8)),
                                            np.long(np.int32(1))))
                if synchronous:
                  sync(self)
    return ()
  def futhark_builtinzhreplicate_f32(self, mem_24494, num_elems_24495,
                                     val_24496):
    group_sizze_24501 = self.sizes["builtin#replicate_f32.group_size_24501"]
    num_groups_24502 = sdiv_up64(sext_i32_i64(num_elems_24495),
                                 sext_i32_i64(group_sizze_24501))
    if ((1 * (np.long(sext_i64_i32(num_groups_24502)) * np.long(group_sizze_24501))) != 0):
      self.builtinzhreplicate_f32zireplicate_24498_var.set_args(mem_24494,
                                                                np.int32(num_elems_24495),
                                                                np.float32(val_24496))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_f32zireplicate_24498_var,
                                 ((np.long(sext_i64_i32(num_groups_24502)) * np.long(group_sizze_24501)),),
                                 (np.long(group_sizze_24501),))
      if synchronous:
        sync(self)
    return ()
  def futhark_builtinzhreplicate_i32(self, mem_24503, num_elems_24504,
                                     val_24505):
    group_sizze_24510 = self.sizes["builtin#replicate_i32.group_size_24510"]
    num_groups_24511 = sdiv_up64(sext_i32_i64(num_elems_24504),
                                 sext_i32_i64(group_sizze_24510))
    if ((1 * (np.long(sext_i64_i32(num_groups_24511)) * np.long(group_sizze_24510))) != 0):
      self.builtinzhreplicate_i32zireplicate_24507_var.set_args(mem_24503,
                                                                np.int32(num_elems_24504),
                                                                np.int32(val_24505))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_i32zireplicate_24507_var,
                                 ((np.long(sext_i64_i32(num_groups_24511)) * np.long(group_sizze_24510)),),
                                 (np.long(group_sizze_24510),))
      if synchronous:
        sync(self)
    return ()
  def futhark_main(self, mappingindices_mem_23188, images_mem_23189, N_18054,
                   m_18055, N_18056, trend_18057, k_18058, n_18059, freq_18060,
                   hfrac_18061, lam_18062):
    dim_match_18065 = (N_18054 == N_18056)
    empty_or_match_cert_18066 = True
    assert dim_match_18065, ("Error: %s\n\nBacktrace:\n-> #0  bfastfinal.fut:19:1-147:20\n" % ("function arguments of wrong shape",))
    x_18068 = (np.int32(2) * k_18058)
    k2p2_18069 = (np.int32(2) + x_18068)
    cond_18070 = slt32(np.int32(0), trend_18057)
    if cond_18070:
      k2p2zq_18071 = k2p2_18069
    else:
      res_18072 = (k2p2_18069 - np.int32(1))
      k2p2zq_18071 = res_18072
    binop_x_23192 = sext_i32_i64(k2p2zq_18071)
    binop_y_23193 = sext_i32_i64(N_18054)
    binop_x_23194 = (binop_x_23192 * binop_y_23193)
    bytes_23191 = (np.int64(4) * binop_x_23194)
    if cond_18070:
      bounds_invalid_upwards_18074 = slt32(k2p2zq_18071, np.int32(0))
      valid_18075 = not(bounds_invalid_upwards_18074)
      range_valid_c_18076 = True
      assert valid_18075, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  helpers.fut:31:10-18\n   #2  bfastfinal.fut:30:16-55\n   #3  bfastfinal.fut:19:1-147:20\n" % ("Range ",
                                                                                                                                                                                                         np.int32(0),
                                                                                                                                                                                                         "..",
                                                                                                                                                                                                         np.int32(1),
                                                                                                                                                                                                         "..<",
                                                                                                                                                                                                         k2p2zq_18071,
                                                                                                                                                                                                         " is invalid."))
      segmap_group_sizze_18889 = self.sizes["main.segmap_group_size_18804"]
      segmap_group_sizze_18890 = sext_i32_i64(segmap_group_sizze_18889)
      segmap_usable_groups_64_18891 = sdiv_up64(binop_x_23194,
                                                segmap_group_sizze_18890)
      segmap_usable_groups_18892 = sext_i64_i32(segmap_usable_groups_64_18891)
      mem_23195 = opencl_alloc(self, bytes_23191, "mem_23195")
      if ((1 * (np.long(segmap_usable_groups_18892) * np.long(segmap_group_sizze_18889))) != 0):
        self.mainzisegmap_18799_var.set_args(self.global_failure,
                                             np.int32(N_18054),
                                             np.float32(freq_18060),
                                             np.int32(k2p2zq_18071),
                                             mappingindices_mem_23188,
                                             mem_23195)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_18799_var,
                                   ((np.long(segmap_usable_groups_18892) * np.long(segmap_group_sizze_18889)),),
                                   (np.long(segmap_group_sizze_18889),))
        if synchronous:
          sync(self)
      binop_p_mem_23202 = mem_23195
    else:
      bounds_invalid_upwards_18099 = slt32(k2p2zq_18071, np.int32(0))
      valid_18100 = not(bounds_invalid_upwards_18099)
      range_valid_c_18101 = True
      assert valid_18100, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  helpers.fut:43:10-20\n   #2  bfastfinal.fut:31:10-49\n   #3  bfastfinal.fut:19:1-147:20\n" % ("Range ",
                                                                                                                                                                                                         np.int32(0),
                                                                                                                                                                                                         "..",
                                                                                                                                                                                                         np.int32(1),
                                                                                                                                                                                                         "..<",
                                                                                                                                                                                                         k2p2zq_18071,
                                                                                                                                                                                                         " is invalid."))
      segmap_group_sizze_19085 = self.sizes["main.segmap_group_size_19004"]
      segmap_group_sizze_19086 = sext_i32_i64(segmap_group_sizze_19085)
      segmap_usable_groups_64_19087 = sdiv_up64(binop_x_23194,
                                                segmap_group_sizze_19086)
      segmap_usable_groups_19088 = sext_i64_i32(segmap_usable_groups_64_19087)
      mem_23201 = opencl_alloc(self, bytes_23191, "mem_23201")
      if ((1 * (np.long(segmap_usable_groups_19088) * np.long(segmap_group_sizze_19085))) != 0):
        self.mainzisegmap_18999_var.set_args(self.global_failure,
                                             np.int32(N_18054),
                                             np.float32(freq_18060),
                                             np.int32(k2p2zq_18071),
                                             mappingindices_mem_23188,
                                             mem_23201)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_18999_var,
                                   ((np.long(segmap_usable_groups_19088) * np.long(segmap_group_sizze_19085)),),
                                   (np.long(segmap_group_sizze_19085),))
        if synchronous:
          sync(self)
      binop_p_mem_23202 = mem_23201
    x_18123 = (N_18054 * N_18054)
    y_18124 = (np.int32(2) * N_18054)
    x_18125 = (x_18123 + y_18124)
    x_18126 = (np.int32(1) + x_18125)
    y_18127 = (np.int32(1) + N_18054)
    zzero_18128 = (y_18127 == np.int32(0))
    nonzzero_18129 = not(zzero_18128)
    nonzzero_cert_18130 = True
    assert nonzzero_18129, ("Error: %s\n\nBacktrace:\n-> #0  bfastfinal.fut:37:21-45\n   #1  bfastfinal.fut:19:1-147:20\n" % ("division by zero",))
    x_18131 = sdiv32(x_18126, y_18127)
    x_18132 = (x_18131 - N_18054)
    binop_p_18133 = (x_18132 - np.int32(1))
    res_18134 = sitofp_i32_f32(binop_p_18133)
    nest_sizze_19189 = (binop_x_23192 * binop_y_23193)
    segmap_group_sizze_19190 = self.sizes["main.segmap_group_size_19157"]
    segmap_group_sizze_19191 = sext_i32_i64(segmap_group_sizze_19190)
    segmap_usable_groups_64_19192 = sdiv_up64(nest_sizze_19189,
                                              segmap_group_sizze_19191)
    segmap_usable_groups_19193 = sext_i64_i32(segmap_usable_groups_64_19192)
    bytes_23203 = (np.int64(4) * nest_sizze_19189)
    mem_23207 = opencl_alloc(self, bytes_23203, "mem_23207")
    self.futhark_builtinzhgpu_map_transpose_f32(mem_23207, np.int32(0),
                                                binop_p_mem_23202, np.int32(0),
                                                np.int32(1), N_18054,
                                                k2p2zq_18071)
    mem_23213 = opencl_alloc(self, bytes_23203, "mem_23213")
    if ((1 * (np.long(segmap_usable_groups_19193) * np.long(segmap_group_sizze_19190))) != 0):
      self.mainzisegmap_19152_var.set_args(self.global_failure,
                                           np.int32(N_18054),
                                           np.int32(k2p2zq_18071),
                                           np.float32(res_18134), mem_23207,
                                           mem_23213)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19152_var,
                                 ((np.long(segmap_usable_groups_19193) * np.long(segmap_group_sizze_19190)),),
                                 (np.long(segmap_group_sizze_19190),))
      if synchronous:
        sync(self)
    empty_slice_18142 = (k2p2zq_18071 == np.int32(0))
    m_18143 = (k2p2zq_18071 - np.int32(1))
    zzero_leq_i_p_m_t_s_18144 = sle32(np.int32(0), m_18143)
    i_p_m_t_s_leq_w_18145 = slt32(m_18143, k2p2zq_18071)
    i_lte_j_18146 = sle32(np.int32(0), k2p2zq_18071)
    y_18147 = (zzero_leq_i_p_m_t_s_18144 and i_p_m_t_s_leq_w_18145)
    y_18148 = (i_lte_j_18146 and y_18147)
    ok_or_empty_18149 = (empty_slice_18142 or y_18148)
    empty_slice_18150 = (n_18059 == np.int32(0))
    m_18151 = (n_18059 - np.int32(1))
    zzero_leq_i_p_m_t_s_18152 = sle32(np.int32(0), m_18151)
    i_p_m_t_s_leq_w_18153 = slt32(m_18151, N_18054)
    i_lte_j_18154 = sle32(np.int32(0), n_18059)
    y_18155 = (zzero_leq_i_p_m_t_s_18152 and i_p_m_t_s_leq_w_18153)
    y_18156 = (i_lte_j_18154 and y_18155)
    ok_or_empty_18157 = (empty_slice_18150 or y_18156)
    index_ok_18158 = (ok_or_empty_18149 and ok_or_empty_18157)
    index_certs_18159 = True
    assert index_ok_18158, ("Error: %s%d%s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  bfastfinal.fut:40:15-21\n   #1  bfastfinal.fut:19:1-147:20\n" % ("Index [",
                                                                                                                                              np.int32(0),
                                                                                                                                              ":, :",
                                                                                                                                              n_18059,
                                                                                                                                              "] out of bounds for array of shape [",
                                                                                                                                              k2p2zq_18071,
                                                                                                                                              "][",
                                                                                                                                              N_18054,
                                                                                                                                              "]."))
    index_certs_18161 = True
    assert index_ok_18158, ("Error: %s%d%s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  bfastfinal.fut:41:15-22\n   #1  bfastfinal.fut:19:1-147:20\n" % ("Index [:",
                                                                                                                                              n_18059,
                                                                                                                                              ", ",
                                                                                                                                              np.int32(0),
                                                                                                                                              ":] out of bounds for array of shape [",
                                                                                                                                              N_18054,
                                                                                                                                              "][",
                                                                                                                                              k2p2zq_18071,
                                                                                                                                              "]."))
    empty_slice_18163 = (m_18055 == np.int32(0))
    m_18164 = (m_18055 - np.int32(1))
    zzero_leq_i_p_m_t_s_18165 = sle32(np.int32(0), m_18164)
    i_p_m_t_s_leq_w_18166 = slt32(m_18164, m_18055)
    i_lte_j_18167 = sle32(np.int32(0), m_18055)
    y_18168 = (zzero_leq_i_p_m_t_s_18165 and i_p_m_t_s_leq_w_18166)
    y_18169 = (i_lte_j_18167 and y_18168)
    ok_or_empty_18170 = (empty_slice_18163 or y_18169)
    index_ok_18171 = (ok_or_empty_18157 and ok_or_empty_18170)
    index_certs_18172 = True
    assert index_ok_18171, ("Error: %s%d%s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  bfastfinal.fut:42:15-26\n   #1  bfastfinal.fut:19:1-147:20\n" % ("Index [",
                                                                                                                                              np.int32(0),
                                                                                                                                              ":, :",
                                                                                                                                              n_18059,
                                                                                                                                              "] out of bounds for array of shape [",
                                                                                                                                              m_18055,
                                                                                                                                              "][",
                                                                                                                                              N_18054,
                                                                                                                                              "]."))
    suff_outer_par_19199 = (self.sizes["main.suff_outer_par_6"] <= m_18055)
    m_19225 = sext_i32_i64(m_18055)
    segmap_group_sizze_19227 = self.sizes["main.segmap_group_size_19204"]
    max_num_groups_24079 = self.sizes["main.segmap_num_groups_19206"]
    num_groups_19228 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(m_19225,
                                                            sext_i32_i64(segmap_group_sizze_19227)),
                                                  sext_i32_i64(max_num_groups_24079))))
    nest_sizze_19429 = (m_19225 * binop_x_23192)
    segmap_group_sizze_19430 = self.sizes["main.segmap_group_size_19254"]
    max_num_groups_24080 = self.sizes["main.segmap_num_groups_19256"]
    num_groups_19431 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_19429,
                                                            sext_i32_i64(segmap_group_sizze_19430)),
                                                  sext_i32_i64(max_num_groups_24080))))
    comparatee_19434 = (m_18055 * k2p2zq_18071)
    suff_outer_par_19435 = (self.sizes["main.suff_outer_par_7"] <= comparatee_19434)
    y_19457 = (binop_x_23192 * binop_x_23192)
    nest_sizze_19458 = (m_19225 * y_19457)
    segmap_group_sizze_19459 = self.sizes["main.segmap_group_size_19288"]
    segmap_group_sizze_19460 = sext_i32_i64(segmap_group_sizze_19459)
    y_19464 = (k2p2zq_18071 * k2p2zq_18071)
    comparatee_19465 = (m_18055 * y_19464)
    suff_outer_par_19466 = (self.sizes["main.suff_outer_par_8"] <= comparatee_19465)
    n_19483 = sext_i32_i64(n_18059)
    nest_sizze_19490 = (nest_sizze_19458 * n_19483)
    segred_group_sizze_19491 = self.sizes["main.segred_group_size_19323"]
    max_num_groups_24081 = self.sizes["main.segred_num_groups_19325"]
    num_groups_19492 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_19490,
                                                            sext_i32_i64(segred_group_sizze_19491)),
                                                  sext_i32_i64(max_num_groups_24081))))
    binop_x_23215 = sext_i32_i64(N_18056)
    binop_x_23217 = (m_19225 * binop_x_23215)
    bytes_23214 = (np.int64(4) * binop_x_23217)
    binop_x_23270 = (m_19225 * y_19457)
    bytes_23265 = (np.int64(4) * binop_x_23270)
    bytes_23220 = (np.int64(4) * y_19457)
    binop_x_23329 = (m_19225 * binop_x_23192)
    binop_x_23331 = (binop_x_23192 * binop_x_23329)
    bytes_23326 = (np.int64(4) * binop_x_23331)
    binop_x_23293 = (nest_sizze_19429 * binop_x_23192)
    bytes_23288 = (np.int64(4) * binop_x_23293)
    bytes_23273 = (np.int64(4) * binop_x_23192)
    num_threads_23944 = (segmap_group_sizze_19227 * num_groups_19228)
    num_threads64_23946 = sext_i32_i64(num_threads_23944)
    total_sizze_23947 = (bytes_23220 * num_threads64_23946)
    num_threads_23948 = (segmap_group_sizze_19430 * num_groups_19431)
    num_threads64_23950 = sext_i32_i64(num_threads_23948)
    total_sizze_23951 = (bytes_23273 * num_threads64_23950)
    local_memory_capacity_24173 = self.max_local_memory
    if (sle64(np.int64(0),
              sext_i32_i64(local_memory_capacity_24173)) and suff_outer_par_19199):
      mem_23218 = opencl_alloc(self, bytes_23214, "mem_23218")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23218, np.int32(0),
                                                  images_mem_23189, np.int32(0),
                                                  np.int32(1), N_18056, m_18055)
      mem_23271 = opencl_alloc(self, bytes_23265, "mem_23271")
      mem_23224 = opencl_alloc(self, total_sizze_23947, "mem_23224")
      if ((1 * (np.long(num_groups_19228) * np.long(segmap_group_sizze_19227))) != 0):
        self.mainzisegmap_19201_var.set_args(self.global_failure,
                                             np.int32(N_18054),
                                             np.int32(m_18055),
                                             np.int32(n_18059),
                                             np.int32(k2p2zq_18071),
                                             np.int32(num_groups_19228),
                                             binop_p_mem_23202, mem_23213,
                                             mem_23218, mem_23224, mem_23271)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19201_var,
                                   ((np.long(num_groups_19228) * np.long(segmap_group_sizze_19227)),),
                                   (np.long(segmap_group_sizze_19227),))
        if synchronous:
          sync(self)
      mem_23218 = None
      mem_23224 = None
      mem_23332 = opencl_alloc(self, bytes_23326, "mem_23332")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23332, np.int32(0),
                                                  mem_23271, np.int32(0),
                                                  np.int32(1), m_18055,
                                                  (k2p2zq_18071 * k2p2zq_18071))
      mem_23271 = None
      res_mem_23334 = mem_23332
    else:
      local_memory_capacity_24172 = self.max_local_memory
      if (sle64(np.int64(0),
                sext_i32_i64(local_memory_capacity_24172)) and suff_outer_par_19435):
        mem_23294 = opencl_alloc(self, bytes_23288, "mem_23294")
        mem_23275 = opencl_alloc(self, total_sizze_23951, "mem_23275")
        if ((1 * (np.long(num_groups_19431) * np.long(segmap_group_sizze_19430))) != 0):
          self.mainzisegmap_19249_var.set_args(self.global_failure,
                                               np.int32(m_18055),
                                               np.int32(N_18056),
                                               np.int32(n_18059),
                                               np.int32(k2p2zq_18071),
                                               np.int32(num_groups_19431),
                                               images_mem_23189, mem_23207,
                                               mem_23213, mem_23275, mem_23294)
          cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19249_var,
                                     ((np.long(num_groups_19431) * np.long(segmap_group_sizze_19430)),),
                                     (np.long(segmap_group_sizze_19430),))
          if synchronous:
            sync(self)
        mem_23275 = None
        mem_23323 = opencl_alloc(self, bytes_23326, "mem_23323")
        self.futhark_builtinzhgpu_map_transpose_f32(mem_23323, np.int32(0),
                                                    mem_23294, np.int32(0),
                                                    np.int32(1),
                                                    (m_18055 * k2p2zq_18071),
                                                    k2p2zq_18071)
        mem_23294 = None
        res_mem_23325 = mem_23323
      else:
        segmap_usable_groups_64_19461 = sdiv_up64(nest_sizze_19458,
                                                  segmap_group_sizze_19460)
        segmap_usable_groups_19462 = sext_i64_i32(segmap_usable_groups_64_19461)
        local_memory_capacity_24171 = self.max_local_memory
        if (sle64(np.int64(0),
                  sext_i32_i64(local_memory_capacity_24171)) and suff_outer_par_19466):
          mem_23302 = opencl_alloc(self, bytes_23326, "mem_23302")
          if ((1 * (np.long(segmap_usable_groups_19462) * np.long(segmap_group_sizze_19459))) != 0):
            self.mainzisegmap_19281_var.set_args(self.global_failure,
                                                 np.int32(m_18055),
                                                 np.int32(N_18056),
                                                 np.int32(n_18059),
                                                 np.int32(k2p2zq_18071),
                                                 images_mem_23189, mem_23207,
                                                 mem_23213, mem_23302)
            cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19281_var,
                                       ((np.long(segmap_usable_groups_19462) * np.long(segmap_group_sizze_19459)),),
                                       (np.long(segmap_group_sizze_19459),))
            if synchronous:
              sync(self)
          res_mem_23316 = mem_23302
        else:
          mem_23307 = opencl_alloc(self, bytes_23191, "mem_23307")
          self.futhark_builtinzhgpu_map_transpose_f32(mem_23307, np.int32(0),
                                                      mem_23213, np.int32(0),
                                                      np.int32(1), k2p2zq_18071,
                                                      N_18054)
          mem_23315 = opencl_alloc(self, bytes_23326, "mem_23315")
          if slt32((n_18059 * np.int32(2)), segred_group_sizze_19491):
            segment_sizze_nonzzero_24112 = smax32(np.int32(1), n_18059)
            num_threads_24113 = (num_groups_19492 * segred_group_sizze_19491)
            if ((1 * (np.long(num_groups_19492) * np.long(segred_group_sizze_19491))) != 0):
              self.mainzisegred_small_19329_var.set_args(self.global_failure,
                                                         cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_19491)))),
                                                         np.int32(N_18054),
                                                         np.int32(m_18055),
                                                         np.int32(N_18056),
                                                         np.int32(n_18059),
                                                         np.int32(k2p2zq_18071),
                                                         np.int32(num_groups_19492),
                                                         images_mem_23189,
                                                         binop_p_mem_23202,
                                                         mem_23307, mem_23315,
                                                         np.int32(segment_sizze_nonzzero_24112))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainzisegred_small_19329_var,
                                         ((np.long(num_groups_19492) * np.long(segred_group_sizze_19491)),),
                                         (np.long(segred_group_sizze_19491),))
              if synchronous:
                sync(self)
          else:
            groups_per_segment_24132 = sdiv_up32(num_groups_19492,
                                                 smax32(np.int32(1),
                                                        ((m_18055 * k2p2zq_18071) * k2p2zq_18071)))
            elements_per_thread_24133 = sdiv_up32(n_18059,
                                                  (segred_group_sizze_19491 * groups_per_segment_24132))
            virt_num_groups_24134 = (groups_per_segment_24132 * ((m_18055 * k2p2zq_18071) * k2p2zq_18071))
            num_threads_24135 = (num_groups_19492 * segred_group_sizze_19491)
            threads_per_segment_24136 = (groups_per_segment_24132 * segred_group_sizze_19491)
            group_res_arr_mem_24137 = opencl_alloc(self,
                                                   (np.int32(4) * (sext_i32_i64(segred_group_sizze_19491) * sext_i32_i64(virt_num_groups_24134))),
                                                   "group_res_arr_mem_24137")
            mainzicounter_mem_24139 = self.mainzicounter_mem_24139
            if ((1 * (np.long(num_groups_19492) * np.long(segred_group_sizze_19491))) != 0):
              self.mainzisegred_large_19329_var.set_args(self.global_failure,
                                                         cl.LocalMemory(np.long(np.int32(1))),
                                                         cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_19491)))),
                                                         np.int32(N_18054),
                                                         np.int32(N_18056),
                                                         np.int32(n_18059),
                                                         np.int32(k2p2zq_18071),
                                                         np.int32(num_groups_19492),
                                                         images_mem_23189,
                                                         binop_p_mem_23202,
                                                         mem_23307, mem_23315,
                                                         np.int32(groups_per_segment_24132),
                                                         np.int32(elements_per_thread_24133),
                                                         np.int32(virt_num_groups_24134),
                                                         np.int32(threads_per_segment_24136),
                                                         group_res_arr_mem_24137,
                                                         mainzicounter_mem_24139)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainzisegred_large_19329_var,
                                         ((np.long(num_groups_19492) * np.long(segred_group_sizze_19491)),),
                                         (np.long(segred_group_sizze_19491),))
              if synchronous:
                sync(self)
          mem_23307 = None
          res_mem_23316 = mem_23315
        res_mem_23325 = res_mem_23316
      res_mem_23334 = res_mem_23325
    m_18193 = (np.int32(2) * k2p2zq_18071)
    nm_18194 = (k2p2zq_18071 * m_18193)
    bounds_invalid_upwards_18195 = slt32(nm_18194, np.int32(0))
    valid_18196 = not(bounds_invalid_upwards_18195)
    range_valid_c_18197 = True
    assert valid_18196, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  helpers.fut:71:21-27\n   #2  bfastfinal.fut:52:35-50\n   #3  bfastfinal.fut:19:1-147:20\n" % ("Range ",
                                                                                                                                                                                                       np.int32(0),
                                                                                                                                                                                                       "..",
                                                                                                                                                                                                       np.int32(1),
                                                                                                                                                                                                       "..<",
                                                                                                                                                                                                       nm_18194,
                                                                                                                                                                                                       " is invalid."))
    zzero_18199 = (m_18193 == np.int32(0))
    nonzzero_18200 = not(zzero_18199)
    nonzzero_cert_18201 = True
    assert nonzzero_18200, ("Error: %s\n\nBacktrace:\n-> #0  helpers.fut:66:41-47\n   #1  helpers.fut:66:14-71:28\n   #2  bfastfinal.fut:52:35-50\n   #3  bfastfinal.fut:19:1-147:20\n" % ("division by zero",))
    loop_nonempty_18202 = slt32(np.int32(0), k2p2zq_18071)
    loop_not_taken_18203 = not(loop_nonempty_18202)
    protect_assert_disj_18204 = (nonzzero_18200 or loop_not_taken_18203)
    nonzzero_cert_18205 = True
    assert protect_assert_disj_18204, ("Error: %s\n\nBacktrace:\n-> #0  helpers.fut:53:43-49\n   #1  helpers.fut:53:16-59:30\n   #2  helpers.fut:72:15-33\n   #3  bfastfinal.fut:52:35-50\n   #4  bfastfinal.fut:19:1-147:20\n" % ("division by zero",))
    y_19566 = smin32(k2p2zq_18071, nm_18194)
    intra_avail_par_19567 = smin32(y_19464, y_19566)
    y_19568 = smax32(k2p2zq_18071, nm_18194)
    computed_group_sizze_19515 = smax32(y_19464, y_19568)
    max_group_sizze_19687 = self.max_group_size
    fits_19688 = sle32(computed_group_sizze_19515, max_group_sizze_19687)
    suff_intra_par_19686 = (self.sizes["main.suff_intra_par_10"] <= intra_avail_par_19567)
    intra_suff_and_fits_19689 = (suff_intra_par_19686 and fits_19688)
    nm_20309 = sext_i32_i64(nm_18194)
    nest_sizze_20311 = (m_19225 * nm_20309)
    segmap_group_sizze_20312 = self.sizes["main.segmap_group_size_20263"]
    segmap_group_sizze_20313 = sext_i32_i64(segmap_group_sizze_20312)
    fits_20338 = sle32(nm_18194, max_group_sizze_19687)
    suff_intra_par_20340 = (self.sizes["main.suff_intra_par_14"] <= nm_18194)
    intra_suff_and_fits_20341 = (fits_20338 and suff_intra_par_20340)
    segmap_group_sizze_20373 = self.sizes["main.segmap_group_size_20153"]
    segmap_group_sizze_20374 = sext_i32_i64(segmap_group_sizze_20373)
    segmap_group_sizze_20397 = self.sizes["main.segmap_group_size_20089"]
    segmap_group_sizze_20398 = sext_i32_i64(segmap_group_sizze_20397)
    segmap_group_sizze_20430 = self.sizes["main.segmap_group_size_20032"]
    segmap_group_sizze_20431 = sext_i32_i64(segmap_group_sizze_20430)
    segmap_group_sizze_20481 = self.sizes["main.segmap_group_size_19805"]
    segmap_group_sizze_20482 = sext_i32_i64(segmap_group_sizze_20481)
    segmap_usable_groups_64_20375 = sdiv_up_safe64(m_19225,
                                                   segmap_group_sizze_20374)
    segmap_usable_groups_20376 = sext_i64_i32(segmap_usable_groups_64_20375)
    segmap_usable_groups_64_20399 = sdiv_up_safe64(nest_sizze_20311,
                                                   segmap_group_sizze_20398)
    segmap_usable_groups_20400 = sext_i64_i32(segmap_usable_groups_64_20399)
    segmap_usable_groups_64_20432 = sdiv_up_safe64(nest_sizze_20311,
                                                   segmap_group_sizze_20431)
    segmap_usable_groups_20433 = sext_i64_i32(segmap_usable_groups_64_20432)
    bytes_23337 = (np.int64(4) * nm_20309)
    bytes_23371 = (np.int64(4) * nest_sizze_20311)
    binop_x_23389 = (m_19225 * nm_20309)
    bytes_23386 = (np.int64(4) * binop_x_23389)
    local_memory_capacity_24238 = self.max_local_memory
    if (sle64(((bytes_23337 + bytes_23337) + bytes_23220),
              sext_i32_i64(local_memory_capacity_24238)) and intra_suff_and_fits_19689):
      mem_23369 = opencl_alloc(self, bytes_23326, "mem_23369")
      if ((1 * (np.long(m_18055) * np.long(computed_group_sizze_19515))) != 0):
        self.mainzisegmap_intragroup_19569_var.set_args(self.global_failure,
                                                        self.failure_is_an_option,
                                                        self.global_failure_args,
                                                        cl.LocalMemory(np.long(bytes_23220)),
                                                        cl.LocalMemory(np.long(bytes_23337)),
                                                        cl.LocalMemory(np.long(bytes_23337)),
                                                        np.int32(k2p2zq_18071),
                                                        np.int32(m_18193),
                                                        np.int32(nm_18194),
                                                        np.int32(computed_group_sizze_19515),
                                                        res_mem_23334,
                                                        mem_23369)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegmap_intragroup_19569_var,
                                   ((np.long(m_18055) * np.long(computed_group_sizze_19515)),),
                                   (np.long(computed_group_sizze_19515),))
        if synchronous:
          sync(self)
      self.failure_is_an_option = np.int32(1)
      res_mem_23436 = mem_23369
    else:
      segmap_usable_groups_64_20314 = sdiv_up64(nest_sizze_20311,
                                                segmap_group_sizze_20313)
      segmap_usable_groups_20315 = sext_i64_i32(segmap_usable_groups_64_20314)
      mem_23375 = opencl_alloc(self, bytes_23371, "mem_23375")
      if ((1 * (np.long(segmap_usable_groups_20315) * np.long(segmap_group_sizze_20312))) != 0):
        self.mainzisegmap_20258_var.set_args(self.global_failure,
                                             np.int32(m_18055),
                                             np.int32(k2p2zq_18071),
                                             np.int32(m_18193),
                                             np.int32(nm_18194), res_mem_23334,
                                             mem_23375)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20258_var,
                                   ((np.long(segmap_usable_groups_20315) * np.long(segmap_group_sizze_20312)),),
                                   (np.long(segmap_group_sizze_20312),))
        if synchronous:
          sync(self)
      ctx_param_ext_23376 = m_18055
      ctx_param_ext_23377 = nm_18194
      ctx_param_ext_23378 = np.int32(0)
      ctx_param_ext_23379 = nm_18194
      ctx_param_ext_23380 = m_18055
      ctx_param_ext_23381 = np.int32(1)
      ctx_param_ext_23382 = nm_18194
      mem_param_23383 = mem_23375
      i_20329 = np.int32(0)
      one_25038 = np.int32(1)
      for counter_25037 in range(k2p2zq_18071):
        y_20331 = slt32(i_20329, nm_18194)
        index_certs_20332 = True
        assert y_20331, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  helpers.fut:52:16-19\n   #1  helpers.fut:72:15-33\n   #2  bfastfinal.fut:52:35-50\n   #3  bfastfinal.fut:19:1-147:20\n" % ("Index [",
                                                                                                                                                                                             i_20329,
                                                                                                                                                                                             "] out of bounds for array of shape [",
                                                                                                                                                                                             nm_18194,
                                                                                                                                                                                             "]."))
        local_memory_capacity_24198 = self.max_local_memory
        if intra_suff_and_fits_20341:
          res_ixfn_23412 = m_18055
        else:
          res_ixfn_23412 = ctx_param_ext_23380
        local_memory_capacity_24199 = self.max_local_memory
        if intra_suff_and_fits_20341:
          res_ixfn_23413 = nm_18194
        else:
          res_ixfn_23413 = ctx_param_ext_23382
        local_memory_capacity_24200 = self.max_local_memory
        if intra_suff_and_fits_20341:
          res_ixfn_23414 = m_18055
        else:
          res_ixfn_23414 = ctx_param_ext_23376
        local_memory_capacity_24201 = self.max_local_memory
        if intra_suff_and_fits_20341:
          res_ixfn_23415 = nm_18194
        else:
          res_ixfn_23415 = ctx_param_ext_23377
        local_memory_capacity_24202 = self.max_local_memory
        if intra_suff_and_fits_20341:
          res_ixfn_23416 = nm_18194
        else:
          res_ixfn_23416 = ctx_param_ext_23379
        local_memory_capacity_24203 = self.max_local_memory
        if intra_suff_and_fits_20341:
          res_ixfn_23417 = np.int32(1)
        else:
          res_ixfn_23417 = ctx_param_ext_23381
        local_memory_capacity_24204 = self.max_local_memory
        if intra_suff_and_fits_20341:
          res_ixfn_23418 = np.int32(0)
        else:
          res_ixfn_23418 = ctx_param_ext_23378
        local_memory_capacity_24232 = self.max_local_memory
        if ((sle64(np.int64(0),
                   sext_i32_i64(local_memory_capacity_24232)) and sle64(bytes_23337,
                                                                        sext_i32_i64(local_memory_capacity_24232))) and intra_suff_and_fits_20341):
          mem_23390 = opencl_alloc(self, bytes_23386, "mem_23390")
          group_sizze_24208 = self.sizes["main.group_size_24208"]
          num_groups_24209 = sdiv_up64((sext_i32_i64(m_18055) * sext_i32_i64(nm_18194)),
                                       sext_i32_i64(group_sizze_24208))
          if ((1 * (np.long(sext_i64_i32(num_groups_24209)) * np.long(group_sizze_24208))) != 0):
            self.mainzicopy_24205_var.set_args(np.int32(m_18055),
                                               np.int32(nm_18194),
                                               np.int32(ctx_param_ext_23378),
                                               np.int32(ctx_param_ext_23379),
                                               np.int32(ctx_param_ext_23381),
                                               mem_param_23383, mem_23390)
            cl.enqueue_nd_range_kernel(self.queue, self.mainzicopy_24205_var,
                                       ((np.long(sext_i64_i32(num_groups_24209)) * np.long(group_sizze_24208)),),
                                       (np.long(group_sizze_24208),))
            if synchronous:
              sync(self)
          mem_23401 = opencl_alloc(self, bytes_23371, "mem_23401")
          if ((1 * (np.long(m_18055) * np.long(nm_18194))) != 0):
            self.mainzisegmap_intragroup_19921_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(bytes_23337)),
                                                            np.int32(m_18055),
                                                            np.int32(k2p2zq_18071),
                                                            np.int32(m_18193),
                                                            np.int32(nm_18194),
                                                            np.int32(i_20329),
                                                            np.int32(ctx_param_ext_23378),
                                                            np.int32(ctx_param_ext_23379),
                                                            np.int32(ctx_param_ext_23381),
                                                            mem_param_23383,
                                                            mem_23390,
                                                            mem_23401)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegmap_intragroup_19921_var,
                                       ((np.long(m_18055) * np.long(nm_18194)),),
                                       (np.long(nm_18194),))
            if synchronous:
              sync(self)
          mem_23390 = None
          res_mem_23419 = mem_23401
        else:
          mem_23404 = opencl_alloc(self, m_19225, "mem_23404")
          if ((1 * (np.long(segmap_usable_groups_20376) * np.long(segmap_group_sizze_20373))) != 0):
            self.mainzisegmap_20150_var.set_args(self.global_failure,
                                                 np.int32(m_18055),
                                                 np.int32(i_20329),
                                                 np.int32(ctx_param_ext_23378),
                                                 np.int32(ctx_param_ext_23379),
                                                 np.int32(ctx_param_ext_23381),
                                                 mem_param_23383, mem_23404)
            cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20150_var,
                                       ((np.long(segmap_usable_groups_20376) * np.long(segmap_group_sizze_20373)),),
                                       (np.long(segmap_group_sizze_20373),))
            if synchronous:
              sync(self)
          mem_23410 = opencl_alloc(self, bytes_23371, "mem_23410")
          if ((1 * (np.long(segmap_usable_groups_20400) * np.long(segmap_group_sizze_20397))) != 0):
            self.mainzisegmap_20084_var.set_args(self.global_failure,
                                                 np.int32(m_18055),
                                                 np.int32(k2p2zq_18071),
                                                 np.int32(m_18193),
                                                 np.int32(nm_18194),
                                                 np.int32(i_20329),
                                                 np.int32(ctx_param_ext_23378),
                                                 np.int32(ctx_param_ext_23379),
                                                 np.int32(ctx_param_ext_23381),
                                                 mem_param_23383, mem_23404,
                                                 mem_23410)
            cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20084_var,
                                       ((np.long(segmap_usable_groups_20400) * np.long(segmap_group_sizze_20397)),),
                                       (np.long(segmap_group_sizze_20397),))
            if synchronous:
              sync(self)
          mem_23404 = None
          if ((1 * (np.long(segmap_usable_groups_20433) * np.long(segmap_group_sizze_20430))) != 0):
            self.mainzisegmap_20027_var.set_args(self.global_failure,
                                                 np.int32(m_18055),
                                                 np.int32(nm_18194),
                                                 np.int32(ctx_param_ext_23378),
                                                 np.int32(ctx_param_ext_23379),
                                                 np.int32(ctx_param_ext_23381),
                                                 mem_param_23383, mem_23410)
            cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20027_var,
                                       ((np.long(segmap_usable_groups_20433) * np.long(segmap_group_sizze_20430)),),
                                       (np.long(segmap_group_sizze_20430),))
            if synchronous:
              sync(self)
          mem_23410 = None
          res_mem_23419 = mem_param_23383
        ctx_param_ext_tmp_24189 = res_ixfn_23414
        ctx_param_ext_tmp_24190 = res_ixfn_23415
        ctx_param_ext_tmp_24191 = res_ixfn_23418
        ctx_param_ext_tmp_24192 = res_ixfn_23416
        ctx_param_ext_tmp_24193 = res_ixfn_23412
        ctx_param_ext_tmp_24194 = res_ixfn_23417
        ctx_param_ext_tmp_24195 = res_ixfn_23413
        mem_param_tmp_24196 = res_mem_23419
        ctx_param_ext_23376 = ctx_param_ext_tmp_24189
        ctx_param_ext_23377 = ctx_param_ext_tmp_24190
        ctx_param_ext_23378 = ctx_param_ext_tmp_24191
        ctx_param_ext_23379 = ctx_param_ext_tmp_24192
        ctx_param_ext_23380 = ctx_param_ext_tmp_24193
        ctx_param_ext_23381 = ctx_param_ext_tmp_24194
        ctx_param_ext_23382 = ctx_param_ext_tmp_24195
        mem_param_23383 = mem_param_tmp_24196
        i_20329 += one_25038
      res_r_ixfn_23420 = ctx_param_ext_23376
      res_r_ixfn_23421 = ctx_param_ext_23377
      res_r_ixfn_23422 = ctx_param_ext_23378
      res_r_ixfn_23423 = ctx_param_ext_23379
      res_r_ixfn_23424 = ctx_param_ext_23380
      res_r_ixfn_23425 = ctx_param_ext_23381
      res_r_ixfn_23426 = ctx_param_ext_23382
      res_r_mem_23427 = mem_param_23383
      mem_23375 = None
      segmap_usable_groups_64_20483 = sdiv_up64(nest_sizze_19458,
                                                segmap_group_sizze_20482)
      segmap_usable_groups_20484 = sext_i64_i32(segmap_usable_groups_64_20483)
      mem_23435 = opencl_alloc(self, bytes_23326, "mem_23435")
      if ((1 * (np.long(segmap_usable_groups_20484) * np.long(segmap_group_sizze_20481))) != 0):
        self.mainzisegmap_19798_var.set_args(self.global_failure,
                                             np.int32(m_18055),
                                             np.int32(k2p2zq_18071),
                                             np.int32(m_18193),
                                             np.int32(res_r_ixfn_23422),
                                             np.int32(res_r_ixfn_23423),
                                             np.int32(res_r_ixfn_23425),
                                             res_r_mem_23427, mem_23435)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19798_var,
                                   ((np.long(segmap_usable_groups_20484) * np.long(segmap_group_sizze_20481)),),
                                   (np.long(segmap_group_sizze_20481),))
        if synchronous:
          sync(self)
      res_r_mem_23427 = None
      res_mem_23436 = mem_23435
    res_mem_23334 = None
    suff_outer_par_20496 = (self.sizes["main.suff_outer_par_17"] <= m_18055)
    segmap_group_sizze_20520 = self.sizes["main.segmap_group_size_20501"]
    max_num_groups_24239 = self.sizes["main.segmap_num_groups_20503"]
    num_groups_20521 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(m_19225,
                                                            sext_i32_i64(segmap_group_sizze_20520)),
                                                  sext_i32_i64(max_num_groups_24239))))
    suff_outer_par_20619 = (self.sizes["main.suff_outer_par_18"] <= comparatee_19434)
    nest_sizze_20638 = (nest_sizze_19429 * n_19483)
    segred_group_sizze_20639 = self.sizes["main.segred_group_size_20571"]
    max_num_groups_24240 = self.sizes["main.segred_num_groups_20573"]
    num_groups_20640 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_20638,
                                                            sext_i32_i64(segred_group_sizze_20639)),
                                                  sext_i32_i64(max_num_groups_24240))))
    tile_sizze_22500 = self.sizes["main.tile_size_22499"]
    group_sizze_22501 = (tile_sizze_22500 * tile_sizze_22500)
    bytes_23458 = (np.int64(4) * nest_sizze_19429)
    bytes_23546 = (np.int64(4) * binop_x_23329)
    binop_x_23466 = sext_i32_i64(tile_sizze_22500)
    binop_x_23468 = (binop_x_23466 * binop_x_23466)
    bytes_23465 = (np.int64(4) * binop_x_23468)
    binop_x_23912 = (np.int64(4) * binop_x_23466)
    sizze_23914 = (binop_x_23466 * binop_x_23912)
    num_threads_23979 = (segmap_group_sizze_20520 * num_groups_20521)
    num_threads64_23981 = sext_i32_i64(num_threads_23979)
    total_sizze_23982 = (bytes_23273 * num_threads64_23981)
    local_memory_capacity_24325 = self.max_local_memory
    if (sle64(np.int64(0),
              sext_i32_i64(local_memory_capacity_24325)) and suff_outer_par_20496):
      mem_23441 = opencl_alloc(self, bytes_23214, "mem_23441")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23441, np.int32(0),
                                                  images_mem_23189, np.int32(0),
                                                  np.int32(1), N_18056, m_18055)
      mem_23462 = opencl_alloc(self, bytes_23458, "mem_23462")
      mem_23445 = opencl_alloc(self, total_sizze_23982, "mem_23445")
      if ((1 * (np.long(num_groups_20521) * np.long(segmap_group_sizze_20520))) != 0):
        self.mainzisegmap_20498_var.set_args(self.global_failure,
                                             np.int32(N_18054),
                                             np.int32(m_18055),
                                             np.int32(n_18059),
                                             np.int32(k2p2zq_18071),
                                             np.int32(num_groups_20521),
                                             binop_p_mem_23202, mem_23441,
                                             mem_23445, mem_23462)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20498_var,
                                   ((np.long(num_groups_20521) * np.long(segmap_group_sizze_20520)),),
                                   (np.long(segmap_group_sizze_20520),))
        if synchronous:
          sync(self)
      mem_23441 = None
      mem_23445 = None
      mem_23550 = opencl_alloc(self, bytes_23546, "mem_23550")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23550, np.int32(0),
                                                  mem_23462, np.int32(0),
                                                  np.int32(1), m_18055,
                                                  k2p2zq_18071)
      mem_23462 = None
      res_mem_23552 = mem_23550
    else:
      local_memory_capacity_24324 = self.max_local_memory
      if (sle64((((bytes_23465 + bytes_23465) + bytes_23465) + bytes_23465),
                sext_i32_i64(local_memory_capacity_24324)) and suff_outer_par_20619):
        num_groups_x_22502 = sdiv_up32(m_18055, tile_sizze_22500)
        num_groups_y_22503 = sdiv_up32(k2p2zq_18071, tile_sizze_22500)
        num_groups_top_22504 = (num_groups_x_22502 * num_groups_y_22503)
        num_whole_tiles_22521 = squot32(n_18059, tile_sizze_22500)
        residual_input_22671 = srem32(n_18059, tile_sizze_22500)
        cond_22672 = (residual_input_22671 == np.int32(0))
        mem_23538 = opencl_alloc(self, bytes_23546, "mem_23538")
        if ((1 * (np.long(num_groups_top_22504) * np.long(group_sizze_22501))) != 0):
          self.mainzisegmap_intragroup_22505_var.set_args(self.global_failure,
                                                          cl.LocalMemory(np.long(bytes_23465)),
                                                          cl.LocalMemory(np.long(bytes_23465)),
                                                          cl.LocalMemory(np.long(bytes_23465)),
                                                          cl.LocalMemory(np.long(bytes_23465)),
                                                          np.int32(m_18055),
                                                          np.int32(N_18056),
                                                          np.int32(n_18059),
                                                          np.int32(k2p2zq_18071),
                                                          np.int32(num_groups_y_22503),
                                                          np.int32(num_whole_tiles_22521),
                                                          np.int32(residual_input_22671),
                                                          np.byte(cond_22672),
                                                          images_mem_23189,
                                                          mem_23207, mem_23538)
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegmap_intragroup_22505_var,
                                     ((np.long(num_groups_top_22504) * np.long(group_sizze_22501)),),
                                     (np.long(group_sizze_22501),))
          if synchronous:
            sync(self)
        res_mem_23545 = mem_23538
      else:
        mem_23544 = opencl_alloc(self, bytes_23546, "mem_23544")
        if slt32((n_18059 * np.int32(2)), segred_group_sizze_20639):
          segment_sizze_nonzzero_24265 = smax32(np.int32(1), n_18059)
          num_threads_24266 = (num_groups_20640 * segred_group_sizze_20639)
          if ((1 * (np.long(num_groups_20640) * np.long(segred_group_sizze_20639))) != 0):
            self.mainzisegred_small_20577_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_20639)))),
                                                       np.int32(N_18054),
                                                       np.int32(m_18055),
                                                       np.int32(N_18056),
                                                       np.int32(n_18059),
                                                       np.int32(k2p2zq_18071),
                                                       np.int32(num_groups_20640),
                                                       images_mem_23189,
                                                       binop_p_mem_23202,
                                                       mem_23544,
                                                       np.int32(segment_sizze_nonzzero_24265))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_small_20577_var,
                                       ((np.long(num_groups_20640) * np.long(segred_group_sizze_20639)),),
                                       (np.long(segred_group_sizze_20639),))
            if synchronous:
              sync(self)
        else:
          groups_per_segment_24285 = sdiv_up32(num_groups_20640,
                                               smax32(np.int32(1),
                                                      (m_18055 * k2p2zq_18071)))
          elements_per_thread_24286 = sdiv_up32(n_18059,
                                                (segred_group_sizze_20639 * groups_per_segment_24285))
          virt_num_groups_24287 = (groups_per_segment_24285 * (m_18055 * k2p2zq_18071))
          num_threads_24288 = (num_groups_20640 * segred_group_sizze_20639)
          threads_per_segment_24289 = (groups_per_segment_24285 * segred_group_sizze_20639)
          group_res_arr_mem_24290 = opencl_alloc(self,
                                                 (np.int32(4) * (sext_i32_i64(segred_group_sizze_20639) * sext_i32_i64(virt_num_groups_24287))),
                                                 "group_res_arr_mem_24290")
          mainzicounter_mem_24292 = self.mainzicounter_mem_24292
          if ((1 * (np.long(num_groups_20640) * np.long(segred_group_sizze_20639))) != 0):
            self.mainzisegred_large_20577_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long(np.int32(1))),
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_20639)))),
                                                       np.int32(N_18054),
                                                       np.int32(N_18056),
                                                       np.int32(n_18059),
                                                       np.int32(k2p2zq_18071),
                                                       np.int32(num_groups_20640),
                                                       images_mem_23189,
                                                       binop_p_mem_23202,
                                                       mem_23544,
                                                       np.int32(groups_per_segment_24285),
                                                       np.int32(elements_per_thread_24286),
                                                       np.int32(virt_num_groups_24287),
                                                       np.int32(threads_per_segment_24289),
                                                       group_res_arr_mem_24290,
                                                       mainzicounter_mem_24292)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_large_20577_var,
                                       ((np.long(num_groups_20640) * np.long(segred_group_sizze_20639)),),
                                       (np.long(segred_group_sizze_20639),))
            if synchronous:
              sync(self)
        res_mem_23545 = mem_23544
      res_mem_23552 = res_mem_23545
    binop_p_mem_23202 = None
    mem_23207 = None
    suff_outer_par_20656 = (self.sizes["main.suff_outer_par_19"] <= m_18055)
    segmap_group_sizze_20679 = self.sizes["main.segmap_group_size_20661"]
    max_num_groups_24326 = self.sizes["main.segmap_num_groups_20663"]
    num_groups_20680 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(m_19225,
                                                            sext_i32_i64(segmap_group_sizze_20679)),
                                                  sext_i32_i64(max_num_groups_24326))))
    segmap_group_sizze_20767 = self.sizes["main.segmap_group_size_20701"]
    segmap_group_sizze_20768 = sext_i32_i64(segmap_group_sizze_20767)
    suff_outer_par_20773 = (self.sizes["main.suff_outer_par_20"] <= comparatee_19434)
    nest_sizze_20790 = (nest_sizze_19429 * binop_x_23192)
    segred_group_sizze_20791 = self.sizes["main.segred_group_size_20727"]
    max_num_groups_24327 = self.sizes["main.segred_num_groups_20729"]
    num_groups_20792 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_20790,
                                                            sext_i32_i64(segred_group_sizze_20791)),
                                                  sext_i32_i64(max_num_groups_24327))))
    num_threads_23989 = (segmap_group_sizze_20679 * num_groups_20680)
    num_threads64_23991 = sext_i32_i64(num_threads_23989)
    total_sizze_23992 = (bytes_23273 * num_threads64_23991)
    local_memory_capacity_24405 = self.max_local_memory
    if (sle64(np.int64(0),
              sext_i32_i64(local_memory_capacity_24405)) and suff_outer_par_20656):
      mem_23559 = opencl_alloc(self, bytes_23265, "mem_23559")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23559, np.int32(0),
                                                  res_mem_23436, np.int32(0),
                                                  np.int32(1),
                                                  (k2p2zq_18071 * k2p2zq_18071),
                                                  m_18055)
      mem_23564 = opencl_alloc(self, bytes_23458, "mem_23564")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23564, np.int32(0),
                                                  res_mem_23552, np.int32(0),
                                                  np.int32(1), k2p2zq_18071,
                                                  m_18055)
      mem_23585 = opencl_alloc(self, bytes_23458, "mem_23585")
      mem_23568 = opencl_alloc(self, total_sizze_23992, "mem_23568")
      if ((1 * (np.long(num_groups_20680) * np.long(segmap_group_sizze_20679))) != 0):
        self.mainzisegmap_20658_var.set_args(self.global_failure,
                                             np.int32(m_18055),
                                             np.int32(k2p2zq_18071),
                                             np.int32(num_groups_20680),
                                             mem_23559, mem_23564, mem_23568,
                                             mem_23585)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20658_var,
                                   ((np.long(num_groups_20680) * np.long(segmap_group_sizze_20679)),),
                                   (np.long(segmap_group_sizze_20679),))
        if synchronous:
          sync(self)
      mem_23559 = None
      mem_23564 = None
      mem_23568 = None
      mem_23610 = opencl_alloc(self, bytes_23546, "mem_23610")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23610, np.int32(0),
                                                  mem_23585, np.int32(0),
                                                  np.int32(1), m_18055,
                                                  k2p2zq_18071)
      mem_23585 = None
      res_mem_23612 = mem_23610
    else:
      segmap_usable_groups_64_20769 = sdiv_up64(nest_sizze_19429,
                                                segmap_group_sizze_20768)
      segmap_usable_groups_20770 = sext_i64_i32(segmap_usable_groups_64_20769)
      local_memory_capacity_24404 = self.max_local_memory
      if (sle64(np.int64(0),
                sext_i32_i64(local_memory_capacity_24404)) and suff_outer_par_20773):
        mem_23592 = opencl_alloc(self, bytes_23288, "mem_23592")
        self.futhark_builtinzhgpu_map_transpose_f32(mem_23592, np.int32(0),
                                                    res_mem_23436, np.int32(0),
                                                    np.int32(1), k2p2zq_18071,
                                                    (m_18055 * k2p2zq_18071))
        mem_23598 = opencl_alloc(self, bytes_23546, "mem_23598")
        if ((1 * (np.long(segmap_usable_groups_20770) * np.long(segmap_group_sizze_20767))) != 0):
          self.mainzisegmap_20696_var.set_args(self.global_failure,
                                               np.int32(m_18055),
                                               np.int32(k2p2zq_18071),
                                               res_mem_23552, mem_23592,
                                               mem_23598)
          cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20696_var,
                                     ((np.long(segmap_usable_groups_20770) * np.long(segmap_group_sizze_20767)),),
                                     (np.long(segmap_group_sizze_20767),))
          if synchronous:
            sync(self)
        mem_23592 = None
        res_mem_23605 = mem_23598
      else:
        mem_23604 = opencl_alloc(self, bytes_23546, "mem_23604")
        if slt32((k2p2zq_18071 * np.int32(2)), segred_group_sizze_20791):
          segment_sizze_nonzzero_24345 = smax32(np.int32(1), k2p2zq_18071)
          num_threads_24346 = (num_groups_20792 * segred_group_sizze_20791)
          if ((1 * (np.long(num_groups_20792) * np.long(segred_group_sizze_20791))) != 0):
            self.mainzisegred_small_20733_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_20791)))),
                                                       np.int32(m_18055),
                                                       np.int32(k2p2zq_18071),
                                                       np.int32(num_groups_20792),
                                                       res_mem_23436,
                                                       res_mem_23552, mem_23604,
                                                       np.int32(segment_sizze_nonzzero_24345))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_small_20733_var,
                                       ((np.long(num_groups_20792) * np.long(segred_group_sizze_20791)),),
                                       (np.long(segred_group_sizze_20791),))
            if synchronous:
              sync(self)
        else:
          groups_per_segment_24365 = sdiv_up32(num_groups_20792,
                                               smax32(np.int32(1),
                                                      (m_18055 * k2p2zq_18071)))
          elements_per_thread_24366 = sdiv_up32(k2p2zq_18071,
                                                (segred_group_sizze_20791 * groups_per_segment_24365))
          virt_num_groups_24367 = (groups_per_segment_24365 * (m_18055 * k2p2zq_18071))
          num_threads_24368 = (num_groups_20792 * segred_group_sizze_20791)
          threads_per_segment_24369 = (groups_per_segment_24365 * segred_group_sizze_20791)
          group_res_arr_mem_24370 = opencl_alloc(self,
                                                 (np.int32(4) * (sext_i32_i64(segred_group_sizze_20791) * sext_i32_i64(virt_num_groups_24367))),
                                                 "group_res_arr_mem_24370")
          mainzicounter_mem_24372 = self.mainzicounter_mem_24372
          if ((1 * (np.long(num_groups_20792) * np.long(segred_group_sizze_20791))) != 0):
            self.mainzisegred_large_20733_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long(np.int32(1))),
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_20791)))),
                                                       np.int32(k2p2zq_18071),
                                                       np.int32(num_groups_20792),
                                                       res_mem_23436,
                                                       res_mem_23552, mem_23604,
                                                       np.int32(groups_per_segment_24365),
                                                       np.int32(elements_per_thread_24366),
                                                       np.int32(virt_num_groups_24367),
                                                       np.int32(threads_per_segment_24369),
                                                       group_res_arr_mem_24370,
                                                       mainzicounter_mem_24372)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_large_20733_var,
                                       ((np.long(num_groups_20792) * np.long(segred_group_sizze_20791)),),
                                       (np.long(segred_group_sizze_20791),))
            if synchronous:
              sync(self)
        res_mem_23605 = mem_23604
      res_mem_23612 = res_mem_23605
    res_mem_23436 = None
    res_mem_23552 = None
    suff_outer_par_20807 = (self.sizes["main.suff_outer_par_21"] <= m_18055)
    segmap_group_sizze_20829 = self.sizes["main.segmap_group_size_20812"]
    max_num_groups_24406 = self.sizes["main.segmap_num_groups_20814"]
    num_groups_20830 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(m_19225,
                                                            sext_i32_i64(segmap_group_sizze_20829)),
                                                  sext_i32_i64(max_num_groups_24406))))
    comparatee_20919 = (N_18054 * m_18055)
    suff_outer_par_20920 = (self.sizes["main.suff_outer_par_22"] <= comparatee_20919)
    y_20936 = (m_19225 * binop_y_23193)
    nest_sizze_20937 = (y_20936 * binop_x_23192)
    segred_group_sizze_20938 = self.sizes["main.segred_group_size_20876"]
    max_num_groups_24407 = self.sizes["main.segred_num_groups_20878"]
    num_groups_20939 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_20937,
                                                            sext_i32_i64(segred_group_sizze_20938)),
                                                  sext_i32_i64(max_num_groups_24407))))
    tile_sizze_22822 = self.sizes["main.tile_size_22821"]
    group_sizze_22823 = (tile_sizze_22822 * tile_sizze_22822)
    bytes_23634 = (np.int64(4) * y_20936)
    bytes_23619 = (np.int64(4) * binop_y_23193)
    binop_x_23730 = (m_19225 * binop_y_23193)
    bytes_23727 = (np.int64(4) * binop_x_23730)
    binop_x_23647 = sext_i32_i64(tile_sizze_22822)
    binop_x_23649 = (binop_x_23647 * binop_x_23647)
    bytes_23646 = (np.int64(4) * binop_x_23649)
    binop_x_23926 = (np.int64(4) * binop_x_23647)
    sizze_23928 = (binop_x_23647 * binop_x_23926)
    num_threads_23999 = (segmap_group_sizze_20829 * num_groups_20830)
    num_threads64_24001 = sext_i32_i64(num_threads_23999)
    total_sizze_24002 = (bytes_23619 * num_threads64_24001)
    local_memory_capacity_24492 = self.max_local_memory
    if (sle64(np.int64(0),
              sext_i32_i64(local_memory_capacity_24492)) and suff_outer_par_20807):
      mem_23617 = opencl_alloc(self, bytes_23458, "mem_23617")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23617, np.int32(0),
                                                  res_mem_23612, np.int32(0),
                                                  np.int32(1), k2p2zq_18071,
                                                  m_18055)
      mem_23638 = opencl_alloc(self, bytes_23634, "mem_23638")
      mem_23621 = opencl_alloc(self, total_sizze_24002, "mem_23621")
      if ((1 * (np.long(num_groups_20830) * np.long(segmap_group_sizze_20829))) != 0):
        self.mainzisegmap_20809_var.set_args(self.global_failure,
                                             np.int32(N_18054),
                                             np.int32(m_18055),
                                             np.int32(k2p2zq_18071),
                                             np.int32(num_groups_20830),
                                             mem_23213, mem_23617, mem_23621,
                                             mem_23638)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20809_var,
                                   ((np.long(num_groups_20830) * np.long(segmap_group_sizze_20829)),),
                                   (np.long(segmap_group_sizze_20829),))
        if synchronous:
          sync(self)
      mem_23617 = None
      mem_23621 = None
      mem_23731 = opencl_alloc(self, bytes_23727, "mem_23731")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23731, np.int32(0),
                                                  mem_23638, np.int32(0),
                                                  np.int32(1), m_18055, N_18054)
      mem_23638 = None
      res_mem_23733 = mem_23731
    else:
      local_memory_capacity_24491 = self.max_local_memory
      if (sle64((((bytes_23646 + bytes_23646) + bytes_23646) + bytes_23646),
                sext_i32_i64(local_memory_capacity_24491)) and suff_outer_par_20920):
        mem_23643 = opencl_alloc(self, bytes_23191, "mem_23643")
        self.futhark_builtinzhgpu_map_transpose_f32(mem_23643, np.int32(0),
                                                    mem_23213, np.int32(0),
                                                    np.int32(1), k2p2zq_18071,
                                                    N_18054)
        num_groups_x_22824 = sdiv_up32(m_18055, tile_sizze_22822)
        num_groups_y_22825 = sdiv_up32(N_18054, tile_sizze_22822)
        num_groups_top_22826 = (num_groups_x_22824 * num_groups_y_22825)
        num_whole_tiles_22843 = squot32(k2p2zq_18071, tile_sizze_22822)
        residual_input_22987 = srem32(k2p2zq_18071, tile_sizze_22822)
        cond_22988 = (residual_input_22987 == np.int32(0))
        mem_23719 = opencl_alloc(self, bytes_23727, "mem_23719")
        if ((1 * (np.long(num_groups_top_22826) * np.long(group_sizze_22823))) != 0):
          self.mainzisegmap_intragroup_22827_var.set_args(self.global_failure,
                                                          cl.LocalMemory(np.long(bytes_23646)),
                                                          cl.LocalMemory(np.long(bytes_23646)),
                                                          cl.LocalMemory(np.long(bytes_23646)),
                                                          cl.LocalMemory(np.long(bytes_23646)),
                                                          np.int32(N_18054),
                                                          np.int32(m_18055),
                                                          np.int32(k2p2zq_18071),
                                                          np.int32(num_groups_y_22825),
                                                          np.int32(num_whole_tiles_22843),
                                                          np.int32(residual_input_22987),
                                                          np.byte(cond_22988),
                                                          res_mem_23612,
                                                          mem_23643, mem_23719)
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegmap_intragroup_22827_var,
                                     ((np.long(num_groups_top_22826) * np.long(group_sizze_22823)),),
                                     (np.long(group_sizze_22823),))
          if synchronous:
            sync(self)
        mem_23643 = None
        res_mem_23726 = mem_23719
      else:
        mem_23725 = opencl_alloc(self, bytes_23727, "mem_23725")
        if slt32((k2p2zq_18071 * np.int32(2)), segred_group_sizze_20938):
          segment_sizze_nonzzero_24432 = smax32(np.int32(1), k2p2zq_18071)
          num_threads_24433 = (num_groups_20939 * segred_group_sizze_20938)
          if ((1 * (np.long(num_groups_20939) * np.long(segred_group_sizze_20938))) != 0):
            self.mainzisegred_small_20882_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_20938)))),
                                                       np.int32(N_18054),
                                                       np.int32(m_18055),
                                                       np.int32(k2p2zq_18071),
                                                       np.int32(num_groups_20939),
                                                       mem_23213, res_mem_23612,
                                                       mem_23725,
                                                       np.int32(segment_sizze_nonzzero_24432))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_small_20882_var,
                                       ((np.long(num_groups_20939) * np.long(segred_group_sizze_20938)),),
                                       (np.long(segred_group_sizze_20938),))
            if synchronous:
              sync(self)
        else:
          groups_per_segment_24452 = sdiv_up32(num_groups_20939,
                                               smax32(np.int32(1),
                                                      (m_18055 * N_18054)))
          elements_per_thread_24453 = sdiv_up32(k2p2zq_18071,
                                                (segred_group_sizze_20938 * groups_per_segment_24452))
          virt_num_groups_24454 = (groups_per_segment_24452 * (m_18055 * N_18054))
          num_threads_24455 = (num_groups_20939 * segred_group_sizze_20938)
          threads_per_segment_24456 = (groups_per_segment_24452 * segred_group_sizze_20938)
          group_res_arr_mem_24457 = opencl_alloc(self,
                                                 (np.int32(4) * (sext_i32_i64(segred_group_sizze_20938) * sext_i32_i64(virt_num_groups_24454))),
                                                 "group_res_arr_mem_24457")
          mainzicounter_mem_24459 = self.mainzicounter_mem_24459
          if ((1 * (np.long(num_groups_20939) * np.long(segred_group_sizze_20938))) != 0):
            self.mainzisegred_large_20882_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long(np.int32(1))),
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_20938)))),
                                                       np.int32(N_18054),
                                                       np.int32(k2p2zq_18071),
                                                       np.int32(num_groups_20939),
                                                       mem_23213, res_mem_23612,
                                                       mem_23725,
                                                       np.int32(groups_per_segment_24452),
                                                       np.int32(elements_per_thread_24453),
                                                       np.int32(virt_num_groups_24454),
                                                       np.int32(threads_per_segment_24456),
                                                       group_res_arr_mem_24457,
                                                       mainzicounter_mem_24459)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_large_20882_var,
                                       ((np.long(num_groups_20939) * np.long(segred_group_sizze_20938)),),
                                       (np.long(segred_group_sizze_20938),))
            if synchronous:
              sync(self)
        res_mem_23726 = mem_23725
      res_mem_23733 = res_mem_23726
    mem_23213 = None
    res_mem_23612 = None
    i_18298 = (N_18054 - np.int32(1))
    x_18299 = sle32(np.int32(0), i_18298)
    y_18300 = slt32(i_18298, N_18054)
    bounds_check_18301 = (x_18299 and y_18300)
    index_certs_18302 = True
    assert bounds_check_18301, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:18:29-34\n   #1  helpers.fut:14:13-20\n   #2  bfastfinal.fut:77:30-91\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  bfastfinal.fut:73:5-80:25\n   #6  bfastfinal.fut:19:1-147:20\n" % ("Index [",
                                                                                                                                                                                                                                                                                                                    i_18298,
                                                                                                                                                                                                                                                                                                                    "] out of bounds for array of shape [",
                                                                                                                                                                                                                                                                                                                    N_18054,
                                                                                                                                                                                                                                                                                                                    "]."))
    fits_21038 = sle32(N_18054, max_group_sizze_19687)
    suff_intra_par_21036 = (self.sizes["main.suff_intra_par_24"] <= N_18054)
    intra_suff_and_fits_21039 = (suff_intra_par_21036 and fits_21038)
    segscan_group_sizze_21221 = self.sizes["main.segscan_group_size_21196"]
    max_num_groups_24493 = self.sizes["main.segscan_num_groups_21198"]
    num_groups_21222 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(y_20936,
                                                            sext_i32_i64(segscan_group_sizze_21221)),
                                                  sext_i32_i64(max_num_groups_24493))))
    mem_23738 = opencl_alloc(self, bytes_23727, "mem_23738")
    self.futhark_builtinzhreplicate_f32(mem_23738, (m_18055 * N_18054), np.nan)
    mem_23743 = opencl_alloc(self, bytes_23727, "mem_23743")
    self.futhark_builtinzhreplicate_i32(mem_23743, (m_18055 * N_18054),
                                        np.int32(0))
    segmap_group_sizze_21297 = self.sizes["main.segmap_group_size_21082"]
    segmap_group_sizze_21298 = sext_i32_i64(segmap_group_sizze_21297)
    bytes_23759 = (np.int64(4) * m_19225)
    local_memory_capacity_24592 = self.max_local_memory
    if (sle64((((bytes_23619 + bytes_23619) + bytes_23619) + bytes_23619),
              sext_i32_i64(local_memory_capacity_24592)) and intra_suff_and_fits_21039):
      mem_23761 = opencl_alloc(self, bytes_23759, "mem_23761")
      mem_23766 = opencl_alloc(self, bytes_23727, "mem_23766")
      mem_23771 = opencl_alloc(self, bytes_23727, "mem_23771")
      if ((1 * (np.long(m_18055) * np.long(N_18054))) != 0):
        self.mainzisegmap_intragroup_20961_var.set_args(self.global_failure,
                                                        cl.LocalMemory(np.long(bytes_23619)),
                                                        cl.LocalMemory(np.long(bytes_23619)),
                                                        cl.LocalMemory(np.long(bytes_23619)),
                                                        cl.LocalMemory(np.long(bytes_23619)),
                                                        np.int32(N_18054),
                                                        np.int32(N_18056),
                                                        np.int32(i_18298),
                                                        images_mem_23189,
                                                        res_mem_23733,
                                                        mem_23761, mem_23766,
                                                        mem_23771)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegmap_intragroup_20961_var,
                                   ((np.long(m_18055) * np.long(N_18054)),),
                                   (np.long(N_18054),))
        if synchronous:
          sync(self)
      res_mem_23787 = mem_23761
      res_mem_23788 = mem_23766
      res_mem_23789 = mem_23771
    else:
      mem_23777 = opencl_alloc(self, bytes_23727, "mem_23777")
      mem_23782 = opencl_alloc(self, bytes_23727, "mem_23782")
      if slt32(np.int32(0), (m_18055 * N_18054)):
        stage1_max_num_groups_24528 = self.max_group_size
        stage1_num_groups_24529 = smin32(stage1_max_num_groups_24528,
                                         num_groups_21222)
        num_threads_24530 = (stage1_num_groups_24529 * segscan_group_sizze_21221)
        if ((1 * (np.long(stage1_num_groups_24529) * np.long(segscan_group_sizze_21221))) != 0):
          self.mainziscan_stage1_21202_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * sext_i32_i64(segscan_group_sizze_21221))))),
                                                    np.int32(N_18054),
                                                    np.int32(m_18055),
                                                    np.int32(N_18056),
                                                    images_mem_23189,
                                                    res_mem_23733, mem_23777,
                                                    mem_23782,
                                                    np.int32(num_threads_24530))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_21202_var,
                                     ((np.long(stage1_num_groups_24529) * np.long(segscan_group_sizze_21221)),),
                                     (np.long(segscan_group_sizze_21221),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_24529))) != 0):
          self.mainziscan_stage2_21202_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * sext_i32_i64(stage1_num_groups_24529))))),
                                                    np.int32(N_18054),
                                                    np.int32(m_18055),
                                                    mem_23777,
                                                    np.int32(stage1_num_groups_24529),
                                                    np.int32(num_threads_24530))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_21202_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_24529)),),
                                     (np.long(stage1_num_groups_24529),))
          if synchronous:
            sync(self)
        required_groups_24570 = sdiv_up32((m_18055 * N_18054),
                                          segscan_group_sizze_21221)
        if ((1 * (np.long(num_groups_21222) * np.long(segscan_group_sizze_21221))) != 0):
          self.mainziscan_stage3_21202_var.set_args(self.global_failure,
                                                    np.int32(N_18054),
                                                    np.int32(m_18055),
                                                    np.int32(num_groups_21222),
                                                    mem_23777,
                                                    np.int32(num_threads_24530),
                                                    np.int32(required_groups_24570))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_21202_var,
                                     ((np.long(num_groups_21222) * np.long(segscan_group_sizze_21221)),),
                                     (np.long(segscan_group_sizze_21221),))
          if synchronous:
            sync(self)
      mem_23785 = opencl_alloc(self, bytes_23759, "mem_23785")
      group_sizze_24585 = self.sizes["main.group_size_24585"]
      num_groups_24586 = sdiv_up64(sext_i32_i64(m_18055),
                                   sext_i32_i64(group_sizze_24585))
      if ((1 * (np.long(sext_i64_i32(num_groups_24586)) * np.long(group_sizze_24585))) != 0):
        self.mainzicopy_24582_var.set_args(np.int32(N_18054), np.int32(m_18055),
                                           np.int32(i_18298), mem_23777,
                                           mem_23785)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzicopy_24582_var,
                                   ((np.long(sext_i64_i32(num_groups_24586)) * np.long(group_sizze_24585)),),
                                   (np.long(group_sizze_24585),))
        if synchronous:
          sync(self)
      segmap_usable_groups_64_21299 = sdiv_up64(y_20936,
                                                segmap_group_sizze_21298)
      segmap_usable_groups_21300 = sext_i64_i32(segmap_usable_groups_64_21299)
      if ((1 * (np.long(segmap_usable_groups_21300) * np.long(segmap_group_sizze_21297))) != 0):
        self.mainzisegmap_21077_var.set_args(self.global_failure,
                                             np.int32(N_18054),
                                             np.int32(m_18055), mem_23738,
                                             mem_23743, mem_23777, mem_23782)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_21077_var,
                                   ((np.long(segmap_usable_groups_21300) * np.long(segmap_group_sizze_21297)),),
                                   (np.long(segmap_group_sizze_21297),))
        if synchronous:
          sync(self)
      mem_23777 = None
      mem_23782 = None
      res_mem_23787 = mem_23785
      res_mem_23788 = mem_23738
      res_mem_23789 = mem_23743
    res_mem_23733 = None
    mem_23738 = None
    mem_23743 = None
    suff_outer_par_21318 = (self.sizes["main.suff_outer_par_27"] <= m_18055)
    fits_21400 = sle32(n_18059, max_group_sizze_19687)
    suff_intra_par_21398 = (self.sizes["main.suff_intra_par_28"] <= n_18059)
    intra_suff_and_fits_21401 = (suff_intra_par_21398 and fits_21400)
    segmap_group_sizze_21364 = self.sizes["main.segmap_group_size_21331"]
    segmap_group_sizze_21365 = sext_i32_i64(segmap_group_sizze_21364)
    nest_sizze_21495 = (m_19225 * n_19483)
    segred_group_sizze_21496 = self.sizes["main.segred_group_size_21477"]
    max_num_groups_24593 = self.sizes["main.segred_num_groups_21479"]
    num_groups_21497 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_21495,
                                                            sext_i32_i64(segred_group_sizze_21496)),
                                                  sext_i32_i64(max_num_groups_24593))))
    segred_group_sizze_21512 = self.sizes["main.segred_group_size_21455"]
    max_num_groups_24594 = self.sizes["main.segred_num_groups_21457"]
    num_groups_21513 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_21495,
                                                            sext_i32_i64(segred_group_sizze_21512)),
                                                  sext_i32_i64(max_num_groups_24594))))
    segmap_group_sizze_21528 = self.sizes["main.segmap_group_size_21436"]
    segmap_group_sizze_21529 = sext_i32_i64(segmap_group_sizze_21528)
    local_memory_capacity_24740 = self.max_local_memory
    if (sle64(np.int64(0),
              sext_i32_i64(local_memory_capacity_24740)) and suff_outer_par_21318):
      segmap_usable_groups_64_21366 = sdiv_up64(m_19225,
                                                segmap_group_sizze_21365)
      segmap_usable_groups_21367 = sext_i64_i32(segmap_usable_groups_64_21366)
      mem_23794 = opencl_alloc(self, bytes_23214, "mem_23794")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23794, np.int32(0),
                                                  images_mem_23189, np.int32(0),
                                                  np.int32(1), N_18056, m_18055)
      mem_23799 = opencl_alloc(self, bytes_23634, "mem_23799")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23799, np.int32(0),
                                                  res_mem_23788, np.int32(0),
                                                  np.int32(1), N_18054, m_18055)
      mem_23803 = opencl_alloc(self, bytes_23759, "mem_23803")
      mem_23806 = opencl_alloc(self, bytes_23759, "mem_23806")
      mem_23809 = opencl_alloc(self, bytes_23759, "mem_23809")
      if ((1 * (np.long(segmap_usable_groups_21367) * np.long(segmap_group_sizze_21364))) != 0):
        self.mainzisegmap_21328_var.set_args(self.global_failure,
                                             np.int32(m_18055),
                                             np.int32(n_18059),
                                             np.float32(hfrac_18061),
                                             np.int32(k2p2_18069), mem_23794,
                                             mem_23799, mem_23803, mem_23806,
                                             mem_23809)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_21328_var,
                                   ((np.long(segmap_usable_groups_21367) * np.long(segmap_group_sizze_21364)),),
                                   (np.long(segmap_group_sizze_21364),))
        if synchronous:
          sync(self)
      mem_23794 = None
      mem_23799 = None
      res_mem_23840 = mem_23803
      res_mem_23841 = mem_23806
      res_mem_23842 = mem_23809
    else:
      local_memory_capacity_24739 = self.max_local_memory
      if (sle64(((np.int32(4) * sext_i32_i64(n_18059)) + (np.int32(4) * sext_i32_i64(n_18059))),
                sext_i32_i64(local_memory_capacity_24739)) and intra_suff_and_fits_21401):
        mem_23815 = opencl_alloc(self, bytes_23759, "mem_23815")
        mem_23818 = opencl_alloc(self, bytes_23759, "mem_23818")
        mem_23821 = opencl_alloc(self, bytes_23759, "mem_23821")
        if ((1 * (np.long(m_18055) * np.long(n_18059))) != 0):
          self.mainzisegmap_intragroup_21326_var.set_args(self.global_failure,
                                                          cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(n_18059)))),
                                                          cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(n_18059)))),
                                                          np.int32(N_18054),
                                                          np.int32(N_18056),
                                                          np.int32(n_18059),
                                                          np.float32(hfrac_18061),
                                                          np.int32(k2p2_18069),
                                                          images_mem_23189,
                                                          res_mem_23788,
                                                          mem_23815, mem_23818,
                                                          mem_23821)
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegmap_intragroup_21326_var,
                                     ((np.long(m_18055) * np.long(n_18059)),),
                                     (np.long(n_18059),))
          if synchronous:
            sync(self)
        res_mem_23837 = mem_23815
        res_mem_23838 = mem_23818
        res_mem_23839 = mem_23821
      else:
        mem_23825 = opencl_alloc(self, bytes_23759, "mem_23825")
        if slt32((n_18059 * np.int32(2)), segred_group_sizze_21496):
          segment_sizze_nonzzero_24616 = smax32(np.int32(1), n_18059)
          num_threads_24617 = (num_groups_21497 * segred_group_sizze_21496)
          if ((1 * (np.long(num_groups_21497) * np.long(segred_group_sizze_21496))) != 0):
            self.mainzisegred_small_21483_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21496)))),
                                                       np.int32(m_18055),
                                                       np.int32(N_18056),
                                                       np.int32(n_18059),
                                                       np.int32(num_groups_21497),
                                                       images_mem_23189,
                                                       mem_23825,
                                                       np.int32(segment_sizze_nonzzero_24616))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_small_21483_var,
                                       ((np.long(num_groups_21497) * np.long(segred_group_sizze_21496)),),
                                       (np.long(segred_group_sizze_21496),))
            if synchronous:
              sync(self)
        else:
          groups_per_segment_24636 = sdiv_up32(num_groups_21497,
                                               smax32(np.int32(1), m_18055))
          elements_per_thread_24637 = sdiv_up32(n_18059,
                                                (segred_group_sizze_21496 * groups_per_segment_24636))
          virt_num_groups_24638 = (groups_per_segment_24636 * m_18055)
          num_threads_24639 = (num_groups_21497 * segred_group_sizze_21496)
          threads_per_segment_24640 = (groups_per_segment_24636 * segred_group_sizze_21496)
          group_res_arr_mem_24641 = opencl_alloc(self,
                                                 (np.int32(4) * (sext_i32_i64(segred_group_sizze_21496) * sext_i32_i64(virt_num_groups_24638))),
                                                 "group_res_arr_mem_24641")
          mainzicounter_mem_24643 = self.mainzicounter_mem_24643
          if ((1 * (np.long(num_groups_21497) * np.long(segred_group_sizze_21496))) != 0):
            self.mainzisegred_large_21483_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long(np.int32(1))),
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21496)))),
                                                       np.int32(N_18056),
                                                       np.int32(n_18059),
                                                       np.int32(num_groups_21497),
                                                       images_mem_23189,
                                                       mem_23825,
                                                       np.int32(groups_per_segment_24636),
                                                       np.int32(elements_per_thread_24637),
                                                       np.int32(virt_num_groups_24638),
                                                       np.int32(threads_per_segment_24640),
                                                       group_res_arr_mem_24641,
                                                       mainzicounter_mem_24643)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_large_21483_var,
                                       ((np.long(num_groups_21497) * np.long(segred_group_sizze_21496)),),
                                       (np.long(segred_group_sizze_21496),))
            if synchronous:
              sync(self)
        mem_23829 = opencl_alloc(self, bytes_23759, "mem_23829")
        if slt32((n_18059 * np.int32(2)), segred_group_sizze_21512):
          segment_sizze_nonzzero_24675 = smax32(np.int32(1), n_18059)
          num_threads_24676 = (num_groups_21513 * segred_group_sizze_21512)
          if ((1 * (np.long(num_groups_21513) * np.long(segred_group_sizze_21512))) != 0):
            self.mainzisegred_small_21461_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21512)))),
                                                       np.int32(N_18054),
                                                       np.int32(m_18055),
                                                       np.int32(n_18059),
                                                       np.int32(num_groups_21513),
                                                       res_mem_23788, mem_23825,
                                                       mem_23829,
                                                       np.int32(segment_sizze_nonzzero_24675))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_small_21461_var,
                                       ((np.long(num_groups_21513) * np.long(segred_group_sizze_21512)),),
                                       (np.long(segred_group_sizze_21512),))
            if synchronous:
              sync(self)
        else:
          groups_per_segment_24695 = sdiv_up32(num_groups_21513,
                                               smax32(np.int32(1), m_18055))
          elements_per_thread_24696 = sdiv_up32(n_18059,
                                                (segred_group_sizze_21512 * groups_per_segment_24695))
          virt_num_groups_24697 = (groups_per_segment_24695 * m_18055)
          num_threads_24698 = (num_groups_21513 * segred_group_sizze_21512)
          threads_per_segment_24699 = (groups_per_segment_24695 * segred_group_sizze_21512)
          group_res_arr_mem_24700 = opencl_alloc(self,
                                                 (np.int32(4) * (sext_i32_i64(segred_group_sizze_21512) * sext_i32_i64(virt_num_groups_24697))),
                                                 "group_res_arr_mem_24700")
          mainzicounter_mem_24702 = self.mainzicounter_mem_24702
          if ((1 * (np.long(num_groups_21513) * np.long(segred_group_sizze_21512))) != 0):
            self.mainzisegred_large_21461_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long(np.int32(1))),
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21512)))),
                                                       np.int32(N_18054),
                                                       np.int32(n_18059),
                                                       np.int32(num_groups_21513),
                                                       res_mem_23788, mem_23825,
                                                       mem_23829,
                                                       np.int32(groups_per_segment_24695),
                                                       np.int32(elements_per_thread_24696),
                                                       np.int32(virt_num_groups_24697),
                                                       np.int32(threads_per_segment_24699),
                                                       group_res_arr_mem_24700,
                                                       mainzicounter_mem_24702)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_large_21461_var,
                                       ((np.long(num_groups_21513) * np.long(segred_group_sizze_21512)),),
                                       (np.long(segred_group_sizze_21512),))
            if synchronous:
              sync(self)
        segmap_usable_groups_64_21530 = sdiv_up64(m_19225,
                                                  segmap_group_sizze_21529)
        segmap_usable_groups_21531 = sext_i64_i32(segmap_usable_groups_64_21530)
        mem_23833 = opencl_alloc(self, bytes_23759, "mem_23833")
        mem_23836 = opencl_alloc(self, bytes_23759, "mem_23836")
        if ((1 * (np.long(segmap_usable_groups_21531) * np.long(segmap_group_sizze_21528))) != 0):
          self.mainzisegmap_21433_var.set_args(self.global_failure,
                                               np.int32(m_18055),
                                               np.float32(hfrac_18061),
                                               np.int32(k2p2_18069), mem_23825,
                                               mem_23829, mem_23833, mem_23836)
          cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_21433_var,
                                     ((np.long(segmap_usable_groups_21531) * np.long(segmap_group_sizze_21528)),),
                                     (np.long(segmap_group_sizze_21528),))
          if synchronous:
            sync(self)
        mem_23829 = None
        res_mem_23837 = mem_23833
        res_mem_23838 = mem_23825
        res_mem_23839 = mem_23836
      res_mem_23840 = res_mem_23837
      res_mem_23841 = res_mem_23838
      res_mem_23842 = res_mem_23839
    segred_group_sizze_21552 = self.sizes["main.segred_group_size_21551"]
    max_num_groups_24741 = self.sizes["main.segred_num_groups_21553"]
    num_groups_21554 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(m_19225,
                                                            sext_i32_i64(segred_group_sizze_21552)),
                                                  sext_i32_i64(max_num_groups_24741))))
    mem_23845 = opencl_alloc(self, np.int64(4), "mem_23845")
    mainzicounter_mem_24742 = self.mainzicounter_mem_24742
    group_res_arr_mem_24744 = opencl_alloc(self,
                                           (np.int32(4) * (sext_i32_i64(segred_group_sizze_21552) * sext_i32_i64(num_groups_21554))),
                                           "group_res_arr_mem_24744")
    num_threads_24746 = (num_groups_21554 * segred_group_sizze_21552)
    if ((1 * (np.long(num_groups_21554) * np.long(segred_group_sizze_21552))) != 0):
      self.mainzisegred_nonseg_21559_var.set_args(self.global_failure,
                                                  cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21552)))),
                                                  cl.LocalMemory(np.long(np.int32(1))),
                                                  np.int32(m_18055),
                                                  np.int32(num_groups_21554),
                                                  res_mem_23840, mem_23845,
                                                  mainzicounter_mem_24742,
                                                  group_res_arr_mem_24744,
                                                  np.int32(num_threads_24746))
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegred_nonseg_21559_var,
                                 ((np.long(num_groups_21554) * np.long(segred_group_sizze_21552)),),
                                 (np.long(segred_group_sizze_21552),))
      if synchronous:
        sync(self)
    read_res_25045 = np.empty(1, dtype=ct.c_int32)
    cl.enqueue_copy(self.queue, read_res_25045, mem_23845,
                    device_offset=(np.long(np.int64(0)) * 4),
                    is_blocking=synchronous)
    sync(self)
    res_18381 = read_res_25045[0]
    mem_23845 = None
    suff_outer_par_21561 = (self.sizes["main.suff_outer_par_29"] <= m_18055)
    segmap_group_sizze_21587 = self.sizes["main.segmap_group_size_21566"]
    segmap_group_sizze_21588 = sext_i32_i64(segmap_group_sizze_21587)
    res_21631 = sext_i32_i64(res_18381)
    nest_sizze_21634 = (m_19225 * res_21631)
    segred_group_sizze_21635 = self.sizes["main.segred_group_size_21611"]
    max_num_groups_24772 = self.sizes["main.segred_num_groups_21613"]
    num_groups_21636 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_21634,
                                                            sext_i32_i64(segred_group_sizze_21635)),
                                                  sext_i32_i64(max_num_groups_24772))))
    local_memory_capacity_24838 = self.max_local_memory
    if (sle64(np.int64(0),
              sext_i32_i64(local_memory_capacity_24838)) and suff_outer_par_21561):
      segmap_usable_groups_64_21589 = sdiv_up64(m_19225,
                                                segmap_group_sizze_21588)
      segmap_usable_groups_21590 = sext_i64_i32(segmap_usable_groups_64_21589)
      mem_23849 = opencl_alloc(self, bytes_23759, "mem_23849")
      if ((1 * (np.long(segmap_usable_groups_21590) * np.long(segmap_group_sizze_21587))) != 0):
        self.mainzisegmap_21563_var.set_args(self.global_failure,
                                             np.int32(N_18054),
                                             np.int32(m_18055),
                                             np.int32(res_18381), res_mem_23788,
                                             res_mem_23840, res_mem_23841,
                                             mem_23849)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_21563_var,
                                   ((np.long(segmap_usable_groups_21590) * np.long(segmap_group_sizze_21587)),),
                                   (np.long(segmap_group_sizze_21587),))
        if synchronous:
          sync(self)
      res_mem_23854 = mem_23849
    else:
      mem_23853 = opencl_alloc(self, bytes_23759, "mem_23853")
      if slt32((res_18381 * np.int32(2)), segred_group_sizze_21635):
        segment_sizze_nonzzero_24779 = smax32(np.int32(1), res_18381)
        num_threads_24780 = (num_groups_21636 * segred_group_sizze_21635)
        if ((1 * (np.long(num_groups_21636) * np.long(segred_group_sizze_21635))) != 0):
          self.mainzisegred_small_21617_var.set_args(self.global_failure,
                                                     cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21635)))),
                                                     np.int32(N_18054),
                                                     np.int32(m_18055),
                                                     np.int32(res_18381),
                                                     np.int32(num_groups_21636),
                                                     res_mem_23788,
                                                     res_mem_23840,
                                                     res_mem_23841, mem_23853,
                                                     np.int32(segment_sizze_nonzzero_24779))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegred_small_21617_var,
                                     ((np.long(num_groups_21636) * np.long(segred_group_sizze_21635)),),
                                     (np.long(segred_group_sizze_21635),))
          if synchronous:
            sync(self)
      else:
        groups_per_segment_24799 = sdiv_up32(num_groups_21636,
                                             smax32(np.int32(1), m_18055))
        elements_per_thread_24800 = sdiv_up32(res_18381,
                                              (segred_group_sizze_21635 * groups_per_segment_24799))
        virt_num_groups_24801 = (groups_per_segment_24799 * m_18055)
        num_threads_24802 = (num_groups_21636 * segred_group_sizze_21635)
        threads_per_segment_24803 = (groups_per_segment_24799 * segred_group_sizze_21635)
        group_res_arr_mem_24804 = opencl_alloc(self,
                                               (np.int32(4) * (sext_i32_i64(segred_group_sizze_21635) * sext_i32_i64(virt_num_groups_24801))),
                                               "group_res_arr_mem_24804")
        mainzicounter_mem_24806 = self.mainzicounter_mem_24806
        if ((1 * (np.long(num_groups_21636) * np.long(segred_group_sizze_21635))) != 0):
          self.mainzisegred_large_21617_var.set_args(self.global_failure,
                                                     cl.LocalMemory(np.long(np.int32(1))),
                                                     cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21635)))),
                                                     np.int32(N_18054),
                                                     np.int32(res_18381),
                                                     np.int32(num_groups_21636),
                                                     res_mem_23788,
                                                     res_mem_23840,
                                                     res_mem_23841, mem_23853,
                                                     np.int32(groups_per_segment_24799),
                                                     np.int32(elements_per_thread_24800),
                                                     np.int32(virt_num_groups_24801),
                                                     np.int32(threads_per_segment_24803),
                                                     group_res_arr_mem_24804,
                                                     mainzicounter_mem_24806)
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegred_large_21617_var,
                                     ((np.long(num_groups_21636) * np.long(segred_group_sizze_21635)),),
                                     (np.long(segred_group_sizze_21635),))
          if synchronous:
            sync(self)
      res_mem_23854 = mem_23853
    iota_arg_18403 = (N_18054 - n_18059)
    bounds_invalid_upwards_18404 = slt32(iota_arg_18403, np.int32(0))
    valid_18405 = not(bounds_invalid_upwards_18404)
    range_valid_c_18406 = True
    assert valid_18405, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  bfastfinal.fut:110:22-31\n   #2  bfastfinal.fut:19:1-147:20\n" % ("Range ",
                                                                                                                                                                           np.int32(0),
                                                                                                                                                                           "..",
                                                                                                                                                                           np.int32(1),
                                                                                                                                                                           "..<",
                                                                                                                                                                           iota_arg_18403,
                                                                                                                                                                           " is invalid."))
    x_18408 = (np.int32(1) + n_18059)
    index_certs_18409 = True
    assert bounds_check_18301, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  bfastfinal.fut:108:63-81\n   #1  bfastfinal.fut:106:15-110:32\n   #2  bfastfinal.fut:19:1-147:20\n" % ("Index [",
                                                                                                                                                                                i_18298,
                                                                                                                                                                                "] out of bounds for array of shape [",
                                                                                                                                                                                N_18054,
                                                                                                                                                                                "]."))
    read_res_25047 = np.empty(1, dtype=ct.c_int32)
    cl.enqueue_copy(self.queue, read_res_25047, mappingindices_mem_23188,
                    device_offset=(np.long(sext_i32_i64(i_18298)) * 4),
                    is_blocking=synchronous)
    sync(self)
    r32_arg_18410 = read_res_25047[0]
    res_18411 = sitofp_i32_f32(r32_arg_18410)
    iota_arg_21713 = sext_i32_i64(iota_arg_18403)
    segmap_group_sizze_21715 = self.sizes["main.segmap_group_size_21697"]
    segmap_group_sizze_21716 = sext_i32_i64(segmap_group_sizze_21715)
    segmap_usable_groups_64_21717 = sdiv_up64(iota_arg_21713,
                                              segmap_group_sizze_21716)
    segmap_usable_groups_21718 = sext_i64_i32(segmap_usable_groups_64_21717)
    bytes_23856 = (np.int64(4) * iota_arg_21713)
    mem_23858 = opencl_alloc(self, bytes_23856, "mem_23858")
    if ((1 * (np.long(segmap_usable_groups_21718) * np.long(segmap_group_sizze_21715))) != 0):
      self.mainzisegmap_21694_var.set_args(self.global_failure,
                                           np.float32(lam_18062),
                                           np.int32(iota_arg_18403),
                                           np.int32(x_18408),
                                           np.float32(res_18411),
                                           mappingindices_mem_23188, mem_23858)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_21694_var,
                                 ((np.long(segmap_usable_groups_21718) * np.long(segmap_group_sizze_21715)),),
                                 (np.long(segmap_group_sizze_21715),))
      if synchronous:
        sync(self)
    range_valid_c_18424 = True
    assert valid_18405, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  bfastfinal.fut:121:29-38\n   #2  /prelude/functional.fut:9:42-44\n   #3  bfastfinal.fut:115:40-145:9\n   #4  bfastfinal.fut:19:1-147:20\n" % ("Range ",
                                                                                                                                                                                                                                                       np.int32(0),
                                                                                                                                                                                                                                                       "..",
                                                                                                                                                                                                                                                       np.int32(1),
                                                                                                                                                                                                                                                       "..<",
                                                                                                                                                                                                                                                       iota_arg_18403,
                                                                                                                                                                                                                                                       " is invalid."))
    fits_21949 = sle32(iota_arg_18403, max_group_sizze_19687)
    suff_intra_par_21947 = (self.sizes["main.suff_intra_par_32"] <= iota_arg_18403)
    intra_suff_and_fits_21950 = (suff_intra_par_21947 and fits_21949)
    segmap_group_sizze_22224 = self.sizes["main.segmap_group_size_22203"]
    max_num_groups_24844 = self.sizes["main.segmap_num_groups_22205"]
    num_groups_22225 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(m_19225,
                                                            sext_i32_i64(segmap_group_sizze_22224)),
                                                  sext_i32_i64(max_num_groups_24844))))
    nest_sizze_22248 = (m_19225 * iota_arg_21713)
    segscan_group_sizze_22249 = self.sizes["main.segscan_group_size_22161"]
    max_num_groups_24845 = self.sizes["main.segscan_num_groups_22163"]
    num_groups_22250 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_22248,
                                                            sext_i32_i64(segscan_group_sizze_22249)),
                                                  sext_i32_i64(max_num_groups_24845))))
    segred_group_sizze_22291 = self.sizes["main.segred_group_size_22102"]
    max_num_groups_24846 = self.sizes["main.segred_num_groups_22104"]
    num_groups_22292 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_22248,
                                                            sext_i32_i64(segred_group_sizze_22291)),
                                                  sext_i32_i64(max_num_groups_24846))))
    segmap_group_sizze_22331 = self.sizes["main.segmap_group_size_22056"]
    segmap_group_sizze_22332 = sext_i32_i64(segmap_group_sizze_22331)
    bytes_23879 = (np.int64(4) * nest_sizze_22248)
    local_memory_capacity_25035 = self.max_local_memory
    if (sle64((((bytes_23856 + (np.int32(1) * sext_i32_i64(iota_arg_18403))) + (np.int32(4) * sext_i32_i64(iota_arg_18403))) + (np.int32(4) * sext_i32_i64(iota_arg_18403))),
              sext_i32_i64(local_memory_capacity_25035)) and intra_suff_and_fits_21950):
      mem_23867 = opencl_alloc(self, bytes_23759, "mem_23867")
      mem_23870 = opencl_alloc(self, bytes_23759, "mem_23870")
      if ((1 * (np.long(m_18055) * np.long(iota_arg_18403))) != 0):
        self.mainzisegmap_intragroup_21740_var.set_args(self.global_failure,
                                                        cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(iota_arg_18403)))),
                                                        cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(iota_arg_18403)))),
                                                        cl.LocalMemory(np.long((np.int32(1) * sext_i32_i64(iota_arg_18403)))),
                                                        cl.LocalMemory(np.long(bytes_23856)),
                                                        np.int32(N_18054),
                                                        np.int32(n_18059),
                                                        np.int32(iota_arg_18403),
                                                        res_mem_23787,
                                                        res_mem_23788,
                                                        res_mem_23789,
                                                        res_mem_23840,
                                                        res_mem_23841,
                                                        res_mem_23842,
                                                        res_mem_23854,
                                                        mem_23858, mem_23867,
                                                        mem_23870)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegmap_intragroup_21740_var,
                                   ((np.long(m_18055) * np.long(iota_arg_18403)),),
                                   (np.long(iota_arg_18403),))
        if synchronous:
          sync(self)
      res_mem_23900 = mem_23867
      res_mem_23901 = mem_23870
    else:
      mem_23874 = opencl_alloc(self, bytes_23759, "mem_23874")
      mem_23877 = opencl_alloc(self, bytes_23759, "mem_23877")
      if ((1 * (np.long(num_groups_22225) * np.long(segmap_group_sizze_22224))) != 0):
        self.mainzisegmap_22200_var.set_args(self.global_failure,
                                             np.int32(m_18055),
                                             np.int32(num_groups_22225),
                                             res_mem_23787, res_mem_23841,
                                             res_mem_23842, mem_23874,
                                             mem_23877)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_22200_var,
                                   ((np.long(num_groups_22225) * np.long(segmap_group_sizze_22224)),),
                                   (np.long(segmap_group_sizze_22224),))
        if synchronous:
          sync(self)
      mem_23883 = opencl_alloc(self, bytes_23879, "mem_23883")
      if slt32(np.int32(0), (m_18055 * iota_arg_18403)):
        stage1_max_num_groups_24879 = self.max_group_size
        stage1_num_groups_24880 = smin32(stage1_max_num_groups_24879,
                                         num_groups_22250)
        num_threads_24881 = (stage1_num_groups_24880 * segscan_group_sizze_22249)
        if ((1 * (np.long(stage1_num_groups_24880) * np.long(segscan_group_sizze_22249))) != 0):
          self.mainziscan_stage1_22167_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * sext_i32_i64(segscan_group_sizze_22249))))),
                                                    np.int32(N_18054),
                                                    np.int32(m_18055),
                                                    np.int32(iota_arg_18403),
                                                    res_mem_23788,
                                                    res_mem_23840,
                                                    res_mem_23841,
                                                    res_mem_23854, mem_23877,
                                                    mem_23883,
                                                    np.int32(num_threads_24881))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_22167_var,
                                     ((np.long(stage1_num_groups_24880) * np.long(segscan_group_sizze_22249)),),
                                     (np.long(segscan_group_sizze_22249),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_24880))) != 0):
          self.mainziscan_stage2_22167_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * sext_i32_i64(stage1_num_groups_24880))))),
                                                    np.int32(m_18055),
                                                    np.int32(iota_arg_18403),
                                                    mem_23883,
                                                    np.int32(stage1_num_groups_24880),
                                                    np.int32(num_threads_24881))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_22167_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_24880)),),
                                     (np.long(stage1_num_groups_24880),))
          if synchronous:
            sync(self)
        required_groups_24921 = sdiv_up32((m_18055 * iota_arg_18403),
                                          segscan_group_sizze_22249)
        if ((1 * (np.long(num_groups_22250) * np.long(segscan_group_sizze_22249))) != 0):
          self.mainziscan_stage3_22167_var.set_args(self.global_failure,
                                                    np.int32(m_18055),
                                                    np.int32(iota_arg_18403),
                                                    np.int32(num_groups_22250),
                                                    mem_23883,
                                                    np.int32(num_threads_24881),
                                                    np.int32(required_groups_24921))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_22167_var,
                                     ((np.long(num_groups_22250) * np.long(segscan_group_sizze_22249)),),
                                     (np.long(segscan_group_sizze_22249),))
          if synchronous:
            sync(self)
      mem_23886 = opencl_alloc(self, m_19225, "mem_23886")
      mem_23889 = opencl_alloc(self, bytes_23759, "mem_23889")
      mem_23892 = opencl_alloc(self, bytes_23759, "mem_23892")
      if slt32((iota_arg_18403 * np.int32(2)), segred_group_sizze_22291):
        segment_sizze_nonzzero_24933 = smax32(np.int32(1), iota_arg_18403)
        num_threads_24934 = (num_groups_22292 * segred_group_sizze_22291)
        if ((1 * (np.long(num_groups_22292) * np.long(segred_group_sizze_22291))) != 0):
          self.mainzisegred_small_22108_var.set_args(self.global_failure,
                                                     cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_22291)))),
                                                     cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_22291)))),
                                                     cl.LocalMemory(np.long((np.int32(1) * sext_i32_i64(segred_group_sizze_22291)))),
                                                     np.int32(m_18055),
                                                     np.int32(iota_arg_18403),
                                                     np.int32(num_groups_22292),
                                                     mem_23858, mem_23874,
                                                     mem_23877, mem_23883,
                                                     mem_23886, mem_23889,
                                                     mem_23892,
                                                     np.int32(segment_sizze_nonzzero_24933))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegred_small_22108_var,
                                     ((np.long(num_groups_22292) * np.long(segred_group_sizze_22291)),),
                                     (np.long(segred_group_sizze_22291),))
          if synchronous:
            sync(self)
      else:
        groups_per_segment_24968 = sdiv_up32(num_groups_22292,
                                             smax32(np.int32(1), m_18055))
        elements_per_thread_24969 = sdiv_up32(iota_arg_18403,
                                              (segred_group_sizze_22291 * groups_per_segment_24968))
        virt_num_groups_24970 = (groups_per_segment_24968 * m_18055)
        num_threads_24971 = (num_groups_22292 * segred_group_sizze_22291)
        threads_per_segment_24972 = (groups_per_segment_24968 * segred_group_sizze_22291)
        group_res_arr_mem_24973 = opencl_alloc(self,
                                               (np.int32(1) * (sext_i32_i64(segred_group_sizze_22291) * sext_i32_i64(virt_num_groups_24970))),
                                               "group_res_arr_mem_24973")
        group_res_arr_mem_24975 = opencl_alloc(self,
                                               (np.int32(4) * (sext_i32_i64(segred_group_sizze_22291) * sext_i32_i64(virt_num_groups_24970))),
                                               "group_res_arr_mem_24975")
        group_res_arr_mem_24977 = opencl_alloc(self,
                                               (np.int32(4) * (sext_i32_i64(segred_group_sizze_22291) * sext_i32_i64(virt_num_groups_24970))),
                                               "group_res_arr_mem_24977")
        mainzicounter_mem_24979 = self.mainzicounter_mem_24979
        if ((1 * (np.long(num_groups_22292) * np.long(segred_group_sizze_22291))) != 0):
          self.mainzisegred_large_22108_var.set_args(self.global_failure,
                                                     cl.LocalMemory(np.long(np.int32(1))),
                                                     cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_22291)))),
                                                     cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_22291)))),
                                                     cl.LocalMemory(np.long((np.int32(1) * sext_i32_i64(segred_group_sizze_22291)))),
                                                     np.int32(iota_arg_18403),
                                                     np.int32(num_groups_22292),
                                                     mem_23858, mem_23874,
                                                     mem_23877, mem_23883,
                                                     mem_23886, mem_23889,
                                                     mem_23892,
                                                     np.int32(groups_per_segment_24968),
                                                     np.int32(elements_per_thread_24969),
                                                     np.int32(virt_num_groups_24970),
                                                     group_res_arr_mem_24973,
                                                     group_res_arr_mem_24975,
                                                     group_res_arr_mem_24977,
                                                     mainzicounter_mem_24979)
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegred_large_22108_var,
                                     ((np.long(num_groups_22292) * np.long(segred_group_sizze_22291)),),
                                     (np.long(segred_group_sizze_22291),))
          if synchronous:
            sync(self)
      mem_23874 = None
      mem_23883 = None
      segmap_usable_groups_64_22333 = sdiv_up64(m_19225,
                                                segmap_group_sizze_22332)
      segmap_usable_groups_22334 = sext_i64_i32(segmap_usable_groups_64_22333)
      mem_23895 = opencl_alloc(self, bytes_23759, "mem_23895")
      if ((sext_i32_i64(m_18055) * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_23895, mem_23892,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((sext_i32_i64(m_18055) * np.int32(4))))
      if synchronous:
        sync(self)
      mem_23892 = None
      mem_23899 = opencl_alloc(self, bytes_23759, "mem_23899")
      if ((1 * (np.long(segmap_usable_groups_22334) * np.long(segmap_group_sizze_22331))) != 0):
        self.mainzisegmap_22053_var.set_args(self.global_failure,
                                             np.int32(N_18054),
                                             np.int32(m_18055),
                                             np.int32(n_18059), res_mem_23789,
                                             res_mem_23841, mem_23877,
                                             mem_23886, mem_23889, mem_23899)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_22053_var,
                                   ((np.long(segmap_usable_groups_22334) * np.long(segmap_group_sizze_22331)),),
                                   (np.long(segmap_group_sizze_22331),))
        if synchronous:
          sync(self)
      mem_23877 = None
      mem_23886 = None
      mem_23889 = None
      res_mem_23900 = mem_23899
      res_mem_23901 = mem_23895
    res_mem_23787 = None
    res_mem_23788 = None
    res_mem_23789 = None
    res_mem_23840 = None
    res_mem_23841 = None
    res_mem_23842 = None
    res_mem_23854 = None
    mem_23858 = None
    out_arrsizze_24061 = m_18055
    out_arrsizze_24063 = m_18055
    out_mem_24060 = res_mem_23900
    out_mem_24062 = res_mem_23901
    return (out_mem_24060, out_arrsizze_24061, out_mem_24062,
            out_arrsizze_24063)
  def futhark_remove_nans(self, images_mem_23188, m_18040, n_18041, p_18042,
                          nan_value_18043):
    m_18690 = sext_i32_i64(m_18040)
    n_18691 = sext_i32_i64(n_18041)
    p_18692 = sext_i32_i64(p_18042)
    y_18694 = (n_18691 * p_18692)
    nest_sizze_18695 = (m_18690 * y_18694)
    segmap_group_sizze_18696 = self.sizes["remove_nans.segmap_group_size_18612"]
    segmap_group_sizze_18697 = sext_i32_i64(segmap_group_sizze_18696)
    segmap_usable_groups_64_18698 = sdiv_up64(nest_sizze_18695,
                                              segmap_group_sizze_18697)
    segmap_usable_groups_18699 = sext_i64_i32(segmap_usable_groups_64_18698)
    binop_x_23193 = (m_18690 * n_18691)
    binop_x_23195 = (p_18692 * binop_x_23193)
    bytes_23190 = (np.int64(4) * binop_x_23195)
    mem_23196 = opencl_alloc(self, bytes_23190, "mem_23196")
    if ((1 * (np.long(segmap_usable_groups_18699) * np.long(segmap_group_sizze_18696))) != 0):
      self.remove_nanszisegmap_18605_var.set_args(self.global_failure,
                                                  np.int32(m_18040),
                                                  np.int32(n_18041),
                                                  np.int32(p_18042),
                                                  np.int16(nan_value_18043),
                                                  images_mem_23188, mem_23196)
      cl.enqueue_nd_range_kernel(self.queue, self.remove_nanszisegmap_18605_var,
                                 ((np.long(segmap_usable_groups_18699) * np.long(segmap_group_sizze_18696)),),
                                 (np.long(segmap_group_sizze_18696),))
      if synchronous:
        sync(self)
    out_arrsizze_24061 = m_18040
    out_arrsizze_24062 = n_18041
    out_arrsizze_24063 = p_18042
    out_mem_24060 = mem_23196
    return (out_mem_24060, out_arrsizze_24061, out_arrsizze_24062,
            out_arrsizze_24063)
  def futhark_reshapeTransp(self, images_mem_23188, m_18033, n_18034, p_18035):
    flatten_to_arg_18037 = (n_18034 * p_18035)
    binop_x_23190 = sext_i32_i64(flatten_to_arg_18037)
    binop_y_23191 = sext_i32_i64(m_18033)
    binop_x_23192 = (binop_x_23190 * binop_y_23191)
    bytes_23189 = (np.int64(4) * binop_x_23192)
    mem_23193 = opencl_alloc(self, bytes_23189, "mem_23193")
    self.futhark_builtinzhgpu_map_transpose_f32(mem_23193, np.int32(0),
                                                images_mem_23188, np.int32(0),
                                                np.int32(1),
                                                flatten_to_arg_18037, m_18033)
    out_arrsizze_24061 = flatten_to_arg_18037
    out_arrsizze_24062 = m_18033
    out_mem_24060 = mem_23193
    return (out_mem_24060, out_arrsizze_24061, out_arrsizze_24062)
  def main(self, trend_18057_ext, k_18058_ext, n_18059_ext, freq_18060_ext,
           hfrac_18061_ext, lam_18062_ext, mappingindices_mem_23188_ext,
           images_mem_23189_ext):
    try:
      trend_18057 = np.int32(ct.c_int32(trend_18057_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(trend_18057_ext),
                                                                                                                            trend_18057_ext))
    try:
      k_18058 = np.int32(ct.c_int32(k_18058_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(k_18058_ext),
                                                                                                                            k_18058_ext))
    try:
      n_18059 = np.int32(ct.c_int32(n_18059_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(n_18059_ext),
                                                                                                                            n_18059_ext))
    try:
      freq_18060 = np.float32(ct.c_float(freq_18060_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(freq_18060_ext),
                                                                                                                            freq_18060_ext))
    try:
      hfrac_18061 = np.float32(ct.c_float(hfrac_18061_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(hfrac_18061_ext),
                                                                                                                            hfrac_18061_ext))
    try:
      lam_18062 = np.float32(ct.c_float(lam_18062_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(lam_18062_ext),
                                                                                                                            lam_18062_ext))
    try:
      assert ((type(mappingindices_mem_23188_ext) in [np.ndarray,
                                                      cl.array.Array]) and (mappingindices_mem_23188_ext.dtype == np.int32)), "Parameter has unexpected type"
      N_18054 = np.int32(mappingindices_mem_23188_ext.shape[0])
      if (type(mappingindices_mem_23188_ext) == cl.array.Array):
        mappingindices_mem_23188 = mappingindices_mem_23188_ext.data
      else:
        mappingindices_mem_23188 = opencl_alloc(self,
                                                np.int64(mappingindices_mem_23188_ext.nbytes),
                                                "mappingindices_mem_23188")
        if (np.int64(mappingindices_mem_23188_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, mappingindices_mem_23188,
                          normaliseArray(mappingindices_mem_23188_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]i32",
                                                                                                                            type(mappingindices_mem_23188_ext),
                                                                                                                            mappingindices_mem_23188_ext))
    try:
      assert ((type(images_mem_23189_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_23189_ext.dtype == np.float32)), "Parameter has unexpected type"
      m_18055 = np.int32(images_mem_23189_ext.shape[0])
      N_18056 = np.int32(images_mem_23189_ext.shape[1])
      if (type(images_mem_23189_ext) == cl.array.Array):
        images_mem_23189 = images_mem_23189_ext.data
      else:
        images_mem_23189 = opencl_alloc(self,
                                        np.int64(images_mem_23189_ext.nbytes),
                                        "images_mem_23189")
        if (np.int64(images_mem_23189_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_23189,
                          normaliseArray(images_mem_23189_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(images_mem_23189_ext),
                                                                                                                            images_mem_23189_ext))
    (out_mem_24060, out_arrsizze_24061, out_mem_24062,
     out_arrsizze_24063) = self.futhark_main(mappingindices_mem_23188,
                                             images_mem_23189, N_18054, m_18055,
                                             N_18056, trend_18057, k_18058,
                                             n_18059, freq_18060, hfrac_18061,
                                             lam_18062)
    sync(self)
    return (cl.array.Array(self.queue, (out_arrsizze_24061,), ct.c_int32,
                           data=out_mem_24060), cl.array.Array(self.queue,
                                                               (out_arrsizze_24063,),
                                                               ct.c_float,
                                                               data=out_mem_24062))
  def remove_nans(self, nan_value_18043_ext, images_mem_23188_ext):
    try:
      nan_value_18043 = np.int16(ct.c_int16(nan_value_18043_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i16",
                                                                                                                            type(nan_value_18043_ext),
                                                                                                                            nan_value_18043_ext))
    try:
      assert ((type(images_mem_23188_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_23188_ext.dtype == np.int16)), "Parameter has unexpected type"
      m_18040 = np.int32(images_mem_23188_ext.shape[0])
      n_18041 = np.int32(images_mem_23188_ext.shape[1])
      p_18042 = np.int32(images_mem_23188_ext.shape[2])
      if (type(images_mem_23188_ext) == cl.array.Array):
        images_mem_23188 = images_mem_23188_ext.data
      else:
        images_mem_23188 = opencl_alloc(self,
                                        np.int64(images_mem_23188_ext.nbytes),
                                        "images_mem_23188")
        if (np.int64(images_mem_23188_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_23188,
                          normaliseArray(images_mem_23188_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]i16",
                                                                                                                            type(images_mem_23188_ext),
                                                                                                                            images_mem_23188_ext))
    (out_mem_24060, out_arrsizze_24061, out_arrsizze_24062,
     out_arrsizze_24063) = self.futhark_remove_nans(images_mem_23188, m_18040,
                                                    n_18041, p_18042,
                                                    nan_value_18043)
    sync(self)
    return cl.array.Array(self.queue, (out_arrsizze_24061, out_arrsizze_24062,
                                       out_arrsizze_24063), ct.c_float,
                          data=out_mem_24060)
  def reshapeTransp(self, images_mem_23188_ext):
    try:
      assert ((type(images_mem_23188_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_23188_ext.dtype == np.float32)), "Parameter has unexpected type"
      m_18033 = np.int32(images_mem_23188_ext.shape[0])
      n_18034 = np.int32(images_mem_23188_ext.shape[1])
      p_18035 = np.int32(images_mem_23188_ext.shape[2])
      if (type(images_mem_23188_ext) == cl.array.Array):
        images_mem_23188 = images_mem_23188_ext.data
      else:
        images_mem_23188 = opencl_alloc(self,
                                        np.int64(images_mem_23188_ext.nbytes),
                                        "images_mem_23188")
        if (np.int64(images_mem_23188_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_23188,
                          normaliseArray(images_mem_23188_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]f32",
                                                                                                                            type(images_mem_23188_ext),
                                                                                                                            images_mem_23188_ext))
    (out_mem_24060, out_arrsizze_24061,
     out_arrsizze_24062) = self.futhark_reshapeTransp(images_mem_23188, m_18033,
                                                      n_18034, p_18035)
    sync(self)
    return cl.array.Array(self.queue, (out_arrsizze_24061, out_arrsizze_24062),
                          ct.c_float, data=out_mem_24060)