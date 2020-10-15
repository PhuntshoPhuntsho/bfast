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




__kernel void builtinzhreplicate_f32zireplicate_24896(__global
                                                      unsigned char *mem_24892,
                                                      int32_t num_elems_24893,
                                                      float val_24894)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_24896;
    int32_t replicate_ltid_24897;
    int32_t replicate_gid_24898;
    
    replicate_gtid_24896 = get_global_id(0);
    replicate_ltid_24897 = get_local_id(0);
    replicate_gid_24898 = get_group_id(0);
    if (slt64(replicate_gtid_24896, sext_i32_i64(num_elems_24893))) {
        ((__global float *) mem_24892)[sext_i32_i64(replicate_gtid_24896)] =
            val_24894;
    }
    
  error_0:
    return;
}
__kernel void builtinzhreplicate_i32zireplicate_24905(__global
                                                      unsigned char *mem_24901,
                                                      int32_t num_elems_24902,
                                                      int32_t val_24903)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_24905;
    int32_t replicate_ltid_24906;
    int32_t replicate_gid_24907;
    
    replicate_gtid_24905 = get_global_id(0);
    replicate_ltid_24906 = get_local_id(0);
    replicate_gid_24907 = get_group_id(0);
    if (slt64(replicate_gtid_24905, sext_i32_i64(num_elems_24902))) {
        ((__global int32_t *) mem_24901)[sext_i32_i64(replicate_gtid_24905)] =
            val_24903;
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
__kernel void mainzicopy_24603(int32_t m_18149, int32_t nm_18288,
                               int32_t ctx_param_ext_23712,
                               int32_t ctx_param_ext_23713,
                               int32_t ctx_param_ext_23715, __global
                               unsigned char *mem_param_23717, __global
                               unsigned char *mem_23724)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_24603;
    int32_t copy_ltid_24604;
    int32_t copy_gid_24605;
    
    copy_gtid_24603 = get_global_id(0);
    copy_ltid_24604 = get_local_id(0);
    copy_gid_24605 = get_group_id(0);
    if (slt32(copy_gtid_24603, sext_i64_i32(sext_i32_i64(m_18149) *
              sext_i32_i64(nm_18288)))) {
        ((__global float *) mem_23724)[sext_i32_i64(copy_gtid_24603 -
                                       squot32(copy_gtid_24603, nm_18288) *
                                       nm_18288) * sext_i32_i64(m_18149) +
                                       sext_i32_i64(squot32(copy_gtid_24603,
                                                            nm_18288))] =
            ((__global
              float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                        (sext_i32_i64(squot32(copy_gtid_24603,
                                                              nm_18288)) *
                                         sext_i32_i64(ctx_param_ext_23713) +
                                         sext_i32_i64(copy_gtid_24603 -
                                         squot32(copy_gtid_24603, nm_18288) *
                                         nm_18288) *
                                         sext_i32_i64(ctx_param_ext_23715))];
    }
    
  error_0:
    return;
}
__kernel void mainzicopy_24980(int32_t N_18148, int32_t m_18149,
                               int32_t i_18392, __global
                               unsigned char *mem_24111, __global
                               unsigned char *mem_24119)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t copy_gtid_24980;
    int32_t copy_ltid_24981;
    int32_t copy_gid_24982;
    
    copy_gtid_24980 = get_global_id(0);
    copy_ltid_24981 = get_local_id(0);
    copy_gid_24982 = get_group_id(0);
    if (slt32(copy_gtid_24980, sext_i64_i32(sext_i32_i64(m_18149)))) {
        ((__global int32_t *) mem_24119)[sext_i32_i64(copy_gtid_24980)] =
            ((__global int32_t *) mem_24111)[sext_i32_i64(i_18392) +
                                             sext_i32_i64(copy_gtid_24980) *
                                             sext_i32_i64(N_18148)];
    }
    
  error_0:
    return;
}
__kernel void mainziscan_stage1_21355(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_24934_backing_aligned_0,
                                      int32_t N_18148, int32_t m_18149,
                                      int32_t N_18150, __global
                                      unsigned char *images_mem_23523, __global
                                      unsigned char *res_mem_24067, __global
                                      unsigned char *mem_24111, __global
                                      unsigned char *mem_24116,
                                      int32_t num_threads_24928)
{
    #define segscan_group_sizze_21374 (mainzisegscan_group_sizze_21349)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_24934_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_24934_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24929;
    int32_t local_tid_24930;
    int32_t group_sizze_24933;
    int32_t wave_sizze_24932;
    int32_t group_tid_24931;
    
    global_tid_24929 = get_global_id(0);
    local_tid_24930 = get_local_id(0);
    group_sizze_24933 = get_local_size(0);
    wave_sizze_24932 = LOCKSTEP_WIDTH;
    group_tid_24931 = get_group_id(0);
    
    int32_t phys_tid_21355;
    
    phys_tid_21355 = global_tid_24929;
    
    __local char *scan_arr_mem_24934;
    
    scan_arr_mem_24934 = (__local char *) scan_arr_mem_24934_backing_0;
    
    int32_t x_21379;
    int32_t x_21380;
    
    x_21379 = 0;
    for (int32_t j_24936 = 0; j_24936 < sdiv_up32(m_18149 * N_18148,
                                                  num_threads_24928);
         j_24936++) {
        int32_t chunk_offset_24937 = segscan_group_sizze_21374 * j_24936 +
                group_tid_24931 * (segscan_group_sizze_21374 *
                                   sdiv_up32(m_18149 * N_18148,
                                             num_threads_24928));
        int32_t flat_idx_24938 = chunk_offset_24937 + local_tid_24930;
        int32_t gtid_21344 = squot32(flat_idx_24938, N_18148);
        int32_t gtid_21354 = flat_idx_24938 - squot32(flat_idx_24938, N_18148) *
                N_18148;
        
        // threads in bounds read input
        {
            if (slt32(gtid_21344, m_18149) && slt32(gtid_21354, N_18148)) {
                float x_21384 = ((__global
                                  float *) images_mem_23523)[sext_i32_i64(gtid_21344) *
                                                             sext_i32_i64(N_18150) +
                                                             sext_i32_i64(gtid_21354)];
                bool res_21386;
                
                res_21386 = futrts_isnan32(x_21384);
                
                bool cond_21387 = !res_21386;
                float res_21388;
                
                if (cond_21387) {
                    float x_21385 = ((__global
                                      float *) res_mem_24067)[sext_i32_i64(gtid_21344) *
                                                              sext_i32_i64(N_18148) +
                                                              sext_i32_i64(gtid_21354)];
                    float res_21389 = x_21384 - x_21385;
                    
                    res_21388 = res_21389;
                } else {
                    res_21388 = NAN;
                }
                
                bool res_21390;
                
                res_21390 = futrts_isnan32(res_21388);
                
                bool res_21391 = !res_21390;
                int32_t res_21392 = btoi_bool_i32(res_21391);
                
                // write to-scan values to parameters
                {
                    x_21380 = res_21392;
                }
                // write mapped values results to global memory
                {
                    ((__global float *) mem_24116)[sext_i32_i64(gtid_21344) *
                                                   sext_i32_i64(N_18148) +
                                                   sext_i32_i64(gtid_21354)] =
                        res_21388;
                }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_21344, m_18149) && slt32(gtid_21354,
                                                          N_18148))) {
                    x_21380 = 0;
                }
            }
            // combine with carry and write to local memory
            {
                int32_t res_21381 = add32(x_21379, x_21380);
                
                ((__local
                  int32_t *) scan_arr_mem_24934)[sext_i32_i64(local_tid_24930)] =
                    res_21381;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t x_24939;
            int32_t x_24940;
            int32_t x_24942;
            int32_t x_24943;
            int32_t skip_threads_24945;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_24930, segscan_group_sizze_21374)) {
                    x_24940 = ((volatile __local
                                int32_t *) scan_arr_mem_24934)[sext_i32_i64(local_tid_24930)];
                    if ((local_tid_24930 - squot32(local_tid_24930, 32) * 32) ==
                        0) {
                        x_24939 = x_24940;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_24945 = 1;
                while (slt32(skip_threads_24945, 32)) {
                    if (sle32(skip_threads_24945, local_tid_24930 -
                              squot32(local_tid_24930, 32) * 32) &&
                        slt32(local_tid_24930, segscan_group_sizze_21374)) {
                        // read operands
                        {
                            x_24939 = ((volatile __local
                                        int32_t *) scan_arr_mem_24934)[sext_i32_i64(local_tid_24930 -
                                                                       skip_threads_24945)];
                        }
                        // perform operation
                        {
                            bool inactive_24946 = slt32(srem32(local_tid_24930 +
                                                               chunk_offset_24937,
                                                               N_18148),
                                                        local_tid_24930 +
                                                        chunk_offset_24937 -
                                                        (local_tid_24930 -
                                                         skip_threads_24945 +
                                                         chunk_offset_24937));
                            
                            if (inactive_24946) {
                                x_24939 = x_24940;
                            }
                            if (!inactive_24946) {
                                int32_t res_24941 = add32(x_24939, x_24940);
                                
                                x_24939 = res_24941;
                            }
                        }
                    }
                    if (sle32(wave_sizze_24932, skip_threads_24945)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_24945, local_tid_24930 -
                              squot32(local_tid_24930, 32) * 32) &&
                        slt32(local_tid_24930, segscan_group_sizze_21374)) {
                        // write result
                        {
                            ((volatile __local
                              int32_t *) scan_arr_mem_24934)[sext_i32_i64(local_tid_24930)] =
                                x_24939;
                            x_24940 = x_24939;
                        }
                    }
                    if (sle32(wave_sizze_24932, skip_threads_24945)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_24945 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_24930 - squot32(local_tid_24930, 32) * 32) ==
                    31 && slt32(local_tid_24930, segscan_group_sizze_21374)) {
                    ((volatile __local
                      int32_t *) scan_arr_mem_24934)[sext_i32_i64(squot32(local_tid_24930,
                                                                          32))] =
                        x_24939;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_24947;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_24930, 32) == 0 &&
                        slt32(local_tid_24930, segscan_group_sizze_21374)) {
                        x_24943 = ((volatile __local
                                    int32_t *) scan_arr_mem_24934)[sext_i32_i64(local_tid_24930)];
                        if ((local_tid_24930 - squot32(local_tid_24930, 32) *
                             32) == 0) {
                            x_24942 = x_24943;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24947 = 1;
                    while (slt32(skip_threads_24947, 32)) {
                        if (sle32(skip_threads_24947, local_tid_24930 -
                                  squot32(local_tid_24930, 32) * 32) &&
                            (squot32(local_tid_24930, 32) == 0 &&
                             slt32(local_tid_24930,
                                   segscan_group_sizze_21374))) {
                            // read operands
                            {
                                x_24942 = ((volatile __local
                                            int32_t *) scan_arr_mem_24934)[sext_i32_i64(local_tid_24930 -
                                                                           skip_threads_24947)];
                            }
                            // perform operation
                            {
                                bool inactive_24948 =
                                     slt32(srem32(local_tid_24930 * 32 + 32 -
                                                  1 + chunk_offset_24937,
                                                  N_18148), local_tid_24930 *
                                           32 + 32 - 1 + chunk_offset_24937 -
                                           ((local_tid_24930 -
                                             skip_threads_24947) * 32 + 32 - 1 +
                                            chunk_offset_24937));
                                
                                if (inactive_24948) {
                                    x_24942 = x_24943;
                                }
                                if (!inactive_24948) {
                                    int32_t res_24944 = add32(x_24942, x_24943);
                                    
                                    x_24942 = res_24944;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24932, skip_threads_24947)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24947, local_tid_24930 -
                                  squot32(local_tid_24930, 32) * 32) &&
                            (squot32(local_tid_24930, 32) == 0 &&
                             slt32(local_tid_24930,
                                   segscan_group_sizze_21374))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) scan_arr_mem_24934)[sext_i32_i64(local_tid_24930)] =
                                    x_24942;
                                x_24943 = x_24942;
                            }
                        }
                        if (sle32(wave_sizze_24932, skip_threads_24947)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24947 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_24930, 32) == 0 ||
                      !slt32(local_tid_24930, segscan_group_sizze_21374))) {
                    // read operands
                    {
                        x_24940 = x_24939;
                        x_24939 = ((__local
                                    int32_t *) scan_arr_mem_24934)[sext_i32_i64(squot32(local_tid_24930,
                                                                                        32) -
                                                                   1)];
                    }
                    // perform operation
                    {
                        bool inactive_24949 = slt32(srem32(local_tid_24930 +
                                                           chunk_offset_24937,
                                                           N_18148),
                                                    local_tid_24930 +
                                                    chunk_offset_24937 -
                                                    (squot32(local_tid_24930,
                                                             32) * 32 - 1 +
                                                     chunk_offset_24937));
                        
                        if (inactive_24949) {
                            x_24939 = x_24940;
                        }
                        if (!inactive_24949) {
                            int32_t res_24941 = add32(x_24939, x_24940);
                            
                            x_24939 = res_24941;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          int32_t *) scan_arr_mem_24934)[sext_i32_i64(local_tid_24930)] =
                            x_24939;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_24930, 32) == 0) {
                    ((__local
                      int32_t *) scan_arr_mem_24934)[sext_i32_i64(local_tid_24930)] =
                        x_24940;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_21344, m_18149) && slt32(gtid_21354, N_18148)) {
                    ((__global int32_t *) mem_24111)[sext_i32_i64(gtid_21344) *
                                                     sext_i32_i64(N_18148) +
                                                     sext_i32_i64(gtid_21354)] =
                        ((__local
                          int32_t *) scan_arr_mem_24934)[sext_i32_i64(local_tid_24930)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_24950 = slt32(srem32(chunk_offset_24937 +
                                                          segscan_group_sizze_21374,
                                                          N_18148),
                                                   chunk_offset_24937 +
                                                   segscan_group_sizze_21374 -
                                                   (chunk_offset_24937 +
                                                    segscan_group_sizze_21374 -
                                                    1));
                bool should_load_carry_24951 = local_tid_24930 == 0 &&
                     !crosses_segment_24950;
                
                if (should_load_carry_24951) {
                    x_21379 = ((__local
                                int32_t *) scan_arr_mem_24934)[sext_i32_i64(segscan_group_sizze_21374 -
                                                               1)];
                }
                if (!should_load_carry_24951) {
                    x_21379 = 0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_21374
}
__kernel void mainziscan_stage1_22438(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_25285_backing_aligned_0,
                                      int32_t N_18148, int32_t m_18149,
                                      int32_t iota_arg_18497, __global
                                      unsigned char *res_mem_24122, __global
                                      unsigned char *res_mem_24174, __global
                                      unsigned char *res_mem_24175, __global
                                      unsigned char *res_mem_24188, __global
                                      unsigned char *mem_24233, __global
                                      unsigned char *mem_24239,
                                      int32_t num_threads_25279)
{
    #define segscan_group_sizze_22520 (mainzisegscan_group_sizze_22432)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_25285_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_25285_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25280;
    int32_t local_tid_25281;
    int32_t group_sizze_25284;
    int32_t wave_sizze_25283;
    int32_t group_tid_25282;
    
    global_tid_25280 = get_global_id(0);
    local_tid_25281 = get_local_id(0);
    group_sizze_25284 = get_local_size(0);
    wave_sizze_25283 = LOCKSTEP_WIDTH;
    group_tid_25282 = get_group_id(0);
    
    int32_t phys_tid_22438;
    
    phys_tid_22438 = global_tid_25280;
    
    __local char *scan_arr_mem_25285;
    
    scan_arr_mem_25285 = (__local char *) scan_arr_mem_25285_backing_0;
    
    float x_22524;
    float x_22525;
    
    x_22524 = 0.0F;
    for (int32_t j_25287 = 0; j_25287 < sdiv_up32(m_18149 * iota_arg_18497,
                                                  num_threads_25279);
         j_25287++) {
        int32_t chunk_offset_25288 = segscan_group_sizze_22520 * j_25287 +
                group_tid_25282 * (segscan_group_sizze_22520 *
                                   sdiv_up32(m_18149 * iota_arg_18497,
                                             num_threads_25279));
        int32_t flat_idx_25289 = chunk_offset_25288 + local_tid_25281;
        int32_t gtid_22427 = squot32(flat_idx_25289, iota_arg_18497);
        int32_t gtid_22437 = flat_idx_25289 - squot32(flat_idx_25289,
                                                      iota_arg_18497) *
                iota_arg_18497;
        
        // threads in bounds read input
        {
            if (slt32(gtid_22427, m_18149) && slt32(gtid_22437,
                                                    iota_arg_18497)) {
                int32_t y_22531 = ((__global
                                    int32_t *) mem_24233)[sext_i32_i64(gtid_22427)];
                bool cond_22534 = sle32(y_22531, gtid_22437);
                float res_22535;
                
                if (cond_22534) {
                    res_22535 = 0.0F;
                } else {
                    int32_t x_22527 = ((__global
                                        int32_t *) res_mem_24175)[sext_i32_i64(gtid_22427)];
                    int32_t x_22528 = ((__global
                                        int32_t *) res_mem_24174)[sext_i32_i64(gtid_22427)];
                    float x_22529 = ((__global
                                      float *) res_mem_24188)[sext_i32_i64(gtid_22427)];
                    bool cond_22536 = gtid_22437 == 0;
                    float res_22537;
                    
                    if (cond_22536) {
                        res_22537 = x_22529;
                    } else {
                        int32_t x_22538 = sub32(x_22527, x_22528);
                        int32_t i_22539 = add32(gtid_22437, x_22538);
                        float negate_arg_22540 = ((__global
                                                   float *) res_mem_24122)[sext_i32_i64(gtid_22427) *
                                                                           sext_i32_i64(N_18148) +
                                                                           sext_i32_i64(i_22539)];
                        float x_22541 = 0.0F - negate_arg_22540;
                        int32_t i_22542 = add32(gtid_22437, x_22527);
                        float y_22543 = ((__global
                                          float *) res_mem_24122)[sext_i32_i64(gtid_22427) *
                                                                  sext_i32_i64(N_18148) +
                                                                  sext_i32_i64(i_22542)];
                        float res_22544 = x_22541 + y_22543;
                        
                        res_22537 = res_22544;
                    }
                    res_22535 = res_22537;
                }
                // write to-scan values to parameters
                {
                    x_22525 = res_22535;
                }
                // write mapped values results to global memory
                { }
            }
        }
        // do one intra-group scan operation
        {
            // maybe restore some to-scan values to parameters, or read neutral
            {
                if (!(slt32(gtid_22427, m_18149) && slt32(gtid_22437,
                                                          iota_arg_18497))) {
                    x_22525 = 0.0F;
                }
            }
            // combine with carry and write to local memory
            {
                float res_22526 = x_22524 + x_22525;
                
                ((__local
                  float *) scan_arr_mem_25285)[sext_i32_i64(local_tid_25281)] =
                    res_22526;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            float x_25290;
            float x_25291;
            float x_25293;
            float x_25294;
            int32_t skip_threads_25296;
            
            // read input for in-block scan
            {
                if (slt32(local_tid_25281, segscan_group_sizze_22520)) {
                    x_25291 = ((volatile __local
                                float *) scan_arr_mem_25285)[sext_i32_i64(local_tid_25281)];
                    if ((local_tid_25281 - squot32(local_tid_25281, 32) * 32) ==
                        0) {
                        x_25290 = x_25291;
                    }
                }
            }
            // in-block scan (hopefully no barriers needed)
            {
                skip_threads_25296 = 1;
                while (slt32(skip_threads_25296, 32)) {
                    if (sle32(skip_threads_25296, local_tid_25281 -
                              squot32(local_tid_25281, 32) * 32) &&
                        slt32(local_tid_25281, segscan_group_sizze_22520)) {
                        // read operands
                        {
                            x_25290 = ((volatile __local
                                        float *) scan_arr_mem_25285)[sext_i32_i64(local_tid_25281 -
                                                                     skip_threads_25296)];
                        }
                        // perform operation
                        {
                            bool inactive_25297 = slt32(srem32(local_tid_25281 +
                                                               chunk_offset_25288,
                                                               iota_arg_18497),
                                                        local_tid_25281 +
                                                        chunk_offset_25288 -
                                                        (local_tid_25281 -
                                                         skip_threads_25296 +
                                                         chunk_offset_25288));
                            
                            if (inactive_25297) {
                                x_25290 = x_25291;
                            }
                            if (!inactive_25297) {
                                float res_25292 = x_25290 + x_25291;
                                
                                x_25290 = res_25292;
                            }
                        }
                    }
                    if (sle32(wave_sizze_25283, skip_threads_25296)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    if (sle32(skip_threads_25296, local_tid_25281 -
                              squot32(local_tid_25281, 32) * 32) &&
                        slt32(local_tid_25281, segscan_group_sizze_22520)) {
                        // write result
                        {
                            ((volatile __local
                              float *) scan_arr_mem_25285)[sext_i32_i64(local_tid_25281)] =
                                x_25290;
                            x_25291 = x_25290;
                        }
                    }
                    if (sle32(wave_sizze_25283, skip_threads_25296)) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                    skip_threads_25296 *= 2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // last thread of block 'i' writes its result to offset 'i'
            {
                if ((local_tid_25281 - squot32(local_tid_25281, 32) * 32) ==
                    31 && slt32(local_tid_25281, segscan_group_sizze_22520)) {
                    ((volatile __local
                      float *) scan_arr_mem_25285)[sext_i32_i64(squot32(local_tid_25281,
                                                                        32))] =
                        x_25290;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
            {
                int32_t skip_threads_25298;
                
                // read input for in-block scan
                {
                    if (squot32(local_tid_25281, 32) == 0 &&
                        slt32(local_tid_25281, segscan_group_sizze_22520)) {
                        x_25294 = ((volatile __local
                                    float *) scan_arr_mem_25285)[sext_i32_i64(local_tid_25281)];
                        if ((local_tid_25281 - squot32(local_tid_25281, 32) *
                             32) == 0) {
                            x_25293 = x_25294;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_25298 = 1;
                    while (slt32(skip_threads_25298, 32)) {
                        if (sle32(skip_threads_25298, local_tid_25281 -
                                  squot32(local_tid_25281, 32) * 32) &&
                            (squot32(local_tid_25281, 32) == 0 &&
                             slt32(local_tid_25281,
                                   segscan_group_sizze_22520))) {
                            // read operands
                            {
                                x_25293 = ((volatile __local
                                            float *) scan_arr_mem_25285)[sext_i32_i64(local_tid_25281 -
                                                                         skip_threads_25298)];
                            }
                            // perform operation
                            {
                                bool inactive_25299 =
                                     slt32(srem32(local_tid_25281 * 32 + 32 -
                                                  1 + chunk_offset_25288,
                                                  iota_arg_18497),
                                           local_tid_25281 * 32 + 32 - 1 +
                                           chunk_offset_25288 -
                                           ((local_tid_25281 -
                                             skip_threads_25298) * 32 + 32 - 1 +
                                            chunk_offset_25288));
                                
                                if (inactive_25299) {
                                    x_25293 = x_25294;
                                }
                                if (!inactive_25299) {
                                    float res_25295 = x_25293 + x_25294;
                                    
                                    x_25293 = res_25295;
                                }
                            }
                        }
                        if (sle32(wave_sizze_25283, skip_threads_25298)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_25298, local_tid_25281 -
                                  squot32(local_tid_25281, 32) * 32) &&
                            (squot32(local_tid_25281, 32) == 0 &&
                             slt32(local_tid_25281,
                                   segscan_group_sizze_22520))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) scan_arr_mem_25285)[sext_i32_i64(local_tid_25281)] =
                                    x_25293;
                                x_25294 = x_25293;
                            }
                        }
                        if (sle32(wave_sizze_25283, skip_threads_25298)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_25298 *= 2;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // carry-in for every block except the first
            {
                if (!(squot32(local_tid_25281, 32) == 0 ||
                      !slt32(local_tid_25281, segscan_group_sizze_22520))) {
                    // read operands
                    {
                        x_25291 = x_25290;
                        x_25290 = ((__local
                                    float *) scan_arr_mem_25285)[sext_i32_i64(squot32(local_tid_25281,
                                                                                      32) -
                                                                 1)];
                    }
                    // perform operation
                    {
                        bool inactive_25300 = slt32(srem32(local_tid_25281 +
                                                           chunk_offset_25288,
                                                           iota_arg_18497),
                                                    local_tid_25281 +
                                                    chunk_offset_25288 -
                                                    (squot32(local_tid_25281,
                                                             32) * 32 - 1 +
                                                     chunk_offset_25288));
                        
                        if (inactive_25300) {
                            x_25290 = x_25291;
                        }
                        if (!inactive_25300) {
                            float res_25292 = x_25290 + x_25291;
                            
                            x_25290 = res_25292;
                        }
                    }
                    // write final result
                    {
                        ((__local
                          float *) scan_arr_mem_25285)[sext_i32_i64(local_tid_25281)] =
                            x_25290;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // restore correct values for first block
            {
                if (squot32(local_tid_25281, 32) == 0) {
                    ((__local
                      float *) scan_arr_mem_25285)[sext_i32_i64(local_tid_25281)] =
                        x_25291;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // threads in bounds write partial scan result
            {
                if (slt32(gtid_22427, m_18149) && slt32(gtid_22437,
                                                        iota_arg_18497)) {
                    ((__global float *) mem_24239)[sext_i32_i64(gtid_22427) *
                                                   sext_i32_i64(iota_arg_18497) +
                                                   sext_i32_i64(gtid_22437)] =
                        ((__local
                          float *) scan_arr_mem_25285)[sext_i32_i64(local_tid_25281)];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread reads last element as carry-in for next iteration
            {
                bool crosses_segment_25301 = slt32(srem32(chunk_offset_25288 +
                                                          segscan_group_sizze_22520,
                                                          iota_arg_18497),
                                                   chunk_offset_25288 +
                                                   segscan_group_sizze_22520 -
                                                   (chunk_offset_25288 +
                                                    segscan_group_sizze_22520 -
                                                    1));
                bool should_load_carry_25302 = local_tid_25281 == 0 &&
                     !crosses_segment_25301;
                
                if (should_load_carry_25302) {
                    x_22524 = ((__local
                                float *) scan_arr_mem_25285)[sext_i32_i64(segscan_group_sizze_22520 -
                                                             1)];
                }
                if (!should_load_carry_25302) {
                    x_22524 = 0.0F;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    
  error_1:
    return;
    #undef segscan_group_sizze_22520
}
__kernel void mainziscan_stage2_21355(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_24957_backing_aligned_0,
                                      int32_t N_18148, int32_t m_18149, __global
                                      unsigned char *mem_24111,
                                      int32_t stage1_num_groups_24927,
                                      int32_t num_threads_24928)
{
    #define segscan_group_sizze_21374 (mainzisegscan_group_sizze_21349)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_24957_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_24957_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24952;
    int32_t local_tid_24953;
    int32_t group_sizze_24956;
    int32_t wave_sizze_24955;
    int32_t group_tid_24954;
    
    global_tid_24952 = get_global_id(0);
    local_tid_24953 = get_local_id(0);
    group_sizze_24956 = get_local_size(0);
    wave_sizze_24955 = LOCKSTEP_WIDTH;
    group_tid_24954 = get_group_id(0);
    
    int32_t phys_tid_21355;
    
    phys_tid_21355 = global_tid_24952;
    
    __local char *scan_arr_mem_24957;
    
    scan_arr_mem_24957 = (__local char *) scan_arr_mem_24957_backing_0;
    
    int32_t flat_idx_24959;
    
    flat_idx_24959 = (local_tid_24953 + 1) * (segscan_group_sizze_21374 *
                                              sdiv_up32(m_18149 * N_18148,
                                                        num_threads_24928)) - 1;
    
    int32_t gtid_21344;
    
    gtid_21344 = squot32(flat_idx_24959, N_18148);
    
    int32_t gtid_21354;
    
    gtid_21354 = flat_idx_24959 - squot32(flat_idx_24959, N_18148) * N_18148;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_21344, m_18149) && slt32(gtid_21354, N_18148)) {
            ((__local
              int32_t *) scan_arr_mem_24957)[sext_i32_i64(local_tid_24953)] =
                ((__global int32_t *) mem_24111)[sext_i32_i64(gtid_21344) *
                                                 sext_i32_i64(N_18148) +
                                                 sext_i32_i64(gtid_21354)];
        } else {
            ((__local
              int32_t *) scan_arr_mem_24957)[sext_i32_i64(local_tid_24953)] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t x_21379;
    int32_t x_21380;
    int32_t x_24960;
    int32_t x_24961;
    int32_t skip_threads_24963;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_24953, stage1_num_groups_24927)) {
            x_21380 = ((volatile __local
                        int32_t *) scan_arr_mem_24957)[sext_i32_i64(local_tid_24953)];
            if ((local_tid_24953 - squot32(local_tid_24953, 32) * 32) == 0) {
                x_21379 = x_21380;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_24963 = 1;
        while (slt32(skip_threads_24963, 32)) {
            if (sle32(skip_threads_24963, local_tid_24953 -
                      squot32(local_tid_24953, 32) * 32) &&
                slt32(local_tid_24953, stage1_num_groups_24927)) {
                // read operands
                {
                    x_21379 = ((volatile __local
                                int32_t *) scan_arr_mem_24957)[sext_i32_i64(local_tid_24953 -
                                                               skip_threads_24963)];
                }
                // perform operation
                {
                    bool inactive_24964 = slt32(srem32((local_tid_24953 + 1) *
                                                       (segscan_group_sizze_21374 *
                                                        sdiv_up32(m_18149 *
                                                                  N_18148,
                                                                  num_threads_24928)) -
                                                       1, N_18148),
                                                (local_tid_24953 + 1) *
                                                (segscan_group_sizze_21374 *
                                                 sdiv_up32(m_18149 * N_18148,
                                                           num_threads_24928)) -
                                                1 - ((local_tid_24953 -
                                                      skip_threads_24963 + 1) *
                                                     (segscan_group_sizze_21374 *
                                                      sdiv_up32(m_18149 *
                                                                N_18148,
                                                                num_threads_24928)) -
                                                     1));
                    
                    if (inactive_24964) {
                        x_21379 = x_21380;
                    }
                    if (!inactive_24964) {
                        int32_t res_21381 = add32(x_21379, x_21380);
                        
                        x_21379 = res_21381;
                    }
                }
            }
            if (sle32(wave_sizze_24955, skip_threads_24963)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_24963, local_tid_24953 -
                      squot32(local_tid_24953, 32) * 32) &&
                slt32(local_tid_24953, stage1_num_groups_24927)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) scan_arr_mem_24957)[sext_i32_i64(local_tid_24953)] =
                        x_21379;
                    x_21380 = x_21379;
                }
            }
            if (sle32(wave_sizze_24955, skip_threads_24963)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_24963 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_24953 - squot32(local_tid_24953, 32) * 32) == 31 &&
            slt32(local_tid_24953, stage1_num_groups_24927)) {
            ((volatile __local
              int32_t *) scan_arr_mem_24957)[sext_i32_i64(squot32(local_tid_24953,
                                                                  32))] =
                x_21379;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_24965;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_24953, 32) == 0 && slt32(local_tid_24953,
                                                           stage1_num_groups_24927)) {
                x_24961 = ((volatile __local
                            int32_t *) scan_arr_mem_24957)[sext_i32_i64(local_tid_24953)];
                if ((local_tid_24953 - squot32(local_tid_24953, 32) * 32) ==
                    0) {
                    x_24960 = x_24961;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_24965 = 1;
            while (slt32(skip_threads_24965, 32)) {
                if (sle32(skip_threads_24965, local_tid_24953 -
                          squot32(local_tid_24953, 32) * 32) &&
                    (squot32(local_tid_24953, 32) == 0 && slt32(local_tid_24953,
                                                                stage1_num_groups_24927))) {
                    // read operands
                    {
                        x_24960 = ((volatile __local
                                    int32_t *) scan_arr_mem_24957)[sext_i32_i64(local_tid_24953 -
                                                                   skip_threads_24965)];
                    }
                    // perform operation
                    {
                        bool inactive_24966 = slt32(srem32((local_tid_24953 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_21374 *
                                                            sdiv_up32(m_18149 *
                                                                      N_18148,
                                                                      num_threads_24928)) -
                                                           1, N_18148),
                                                    (local_tid_24953 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_21374 *
                                                     sdiv_up32(m_18149 *
                                                               N_18148,
                                                               num_threads_24928)) -
                                                    1 - (((local_tid_24953 -
                                                           skip_threads_24965) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_21374 *
                                                          sdiv_up32(m_18149 *
                                                                    N_18148,
                                                                    num_threads_24928)) -
                                                         1));
                        
                        if (inactive_24966) {
                            x_24960 = x_24961;
                        }
                        if (!inactive_24966) {
                            int32_t res_24962 = add32(x_24960, x_24961);
                            
                            x_24960 = res_24962;
                        }
                    }
                }
                if (sle32(wave_sizze_24955, skip_threads_24965)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_24965, local_tid_24953 -
                          squot32(local_tid_24953, 32) * 32) &&
                    (squot32(local_tid_24953, 32) == 0 && slt32(local_tid_24953,
                                                                stage1_num_groups_24927))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) scan_arr_mem_24957)[sext_i32_i64(local_tid_24953)] =
                            x_24960;
                        x_24961 = x_24960;
                    }
                }
                if (sle32(wave_sizze_24955, skip_threads_24965)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_24965 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_24953, 32) == 0 || !slt32(local_tid_24953,
                                                          stage1_num_groups_24927))) {
            // read operands
            {
                x_21380 = x_21379;
                x_21379 = ((__local
                            int32_t *) scan_arr_mem_24957)[sext_i32_i64(squot32(local_tid_24953,
                                                                                32) -
                                                           1)];
            }
            // perform operation
            {
                bool inactive_24967 = slt32(srem32((local_tid_24953 + 1) *
                                                   (segscan_group_sizze_21374 *
                                                    sdiv_up32(m_18149 * N_18148,
                                                              num_threads_24928)) -
                                                   1, N_18148),
                                            (local_tid_24953 + 1) *
                                            (segscan_group_sizze_21374 *
                                             sdiv_up32(m_18149 * N_18148,
                                                       num_threads_24928)) - 1 -
                                            ((squot32(local_tid_24953, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_21374 *
                                              sdiv_up32(m_18149 * N_18148,
                                                        num_threads_24928)) -
                                             1));
                
                if (inactive_24967) {
                    x_21379 = x_21380;
                }
                if (!inactive_24967) {
                    int32_t res_21381 = add32(x_21379, x_21380);
                    
                    x_21379 = res_21381;
                }
            }
            // write final result
            {
                ((__local
                  int32_t *) scan_arr_mem_24957)[sext_i32_i64(local_tid_24953)] =
                    x_21379;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_24953, 32) == 0) {
            ((__local
              int32_t *) scan_arr_mem_24957)[sext_i32_i64(local_tid_24953)] =
                x_21380;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_21344, m_18149) && slt32(gtid_21354, N_18148)) {
            ((__global int32_t *) mem_24111)[sext_i32_i64(gtid_21344) *
                                             sext_i32_i64(N_18148) +
                                             sext_i32_i64(gtid_21354)] =
                ((__local
                  int32_t *) scan_arr_mem_24957)[sext_i32_i64(local_tid_24953)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_21374
}
__kernel void mainziscan_stage2_22438(__global int *global_failure,
                                      __local volatile
                                      int64_t *scan_arr_mem_25308_backing_aligned_0,
                                      int32_t m_18149, int32_t iota_arg_18497,
                                      __global unsigned char *mem_24239,
                                      int32_t stage1_num_groups_25278,
                                      int32_t num_threads_25279)
{
    #define segscan_group_sizze_22520 (mainzisegscan_group_sizze_22432)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict scan_arr_mem_25308_backing_0 =
                          (__local volatile
                           char *) scan_arr_mem_25308_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25303;
    int32_t local_tid_25304;
    int32_t group_sizze_25307;
    int32_t wave_sizze_25306;
    int32_t group_tid_25305;
    
    global_tid_25303 = get_global_id(0);
    local_tid_25304 = get_local_id(0);
    group_sizze_25307 = get_local_size(0);
    wave_sizze_25306 = LOCKSTEP_WIDTH;
    group_tid_25305 = get_group_id(0);
    
    int32_t phys_tid_22438;
    
    phys_tid_22438 = global_tid_25303;
    
    __local char *scan_arr_mem_25308;
    
    scan_arr_mem_25308 = (__local char *) scan_arr_mem_25308_backing_0;
    
    int32_t flat_idx_25310;
    
    flat_idx_25310 = (local_tid_25304 + 1) * (segscan_group_sizze_22520 *
                                              sdiv_up32(m_18149 *
                                                        iota_arg_18497,
                                                        num_threads_25279)) - 1;
    
    int32_t gtid_22427;
    
    gtid_22427 = squot32(flat_idx_25310, iota_arg_18497);
    
    int32_t gtid_22437;
    
    gtid_22437 = flat_idx_25310 - squot32(flat_idx_25310, iota_arg_18497) *
        iota_arg_18497;
    // threads in bound read carries; others get neutral element
    {
        if (slt32(gtid_22427, m_18149) && slt32(gtid_22437, iota_arg_18497)) {
            ((__local
              float *) scan_arr_mem_25308)[sext_i32_i64(local_tid_25304)] =
                ((__global float *) mem_24239)[sext_i32_i64(gtid_22427) *
                                               sext_i32_i64(iota_arg_18497) +
                                               sext_i32_i64(gtid_22437)];
        } else {
            ((__local
              float *) scan_arr_mem_25308)[sext_i32_i64(local_tid_25304)] =
                0.0F;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float x_22524;
    float x_22525;
    float x_25311;
    float x_25312;
    int32_t skip_threads_25314;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_25304, stage1_num_groups_25278)) {
            x_22525 = ((volatile __local
                        float *) scan_arr_mem_25308)[sext_i32_i64(local_tid_25304)];
            if ((local_tid_25304 - squot32(local_tid_25304, 32) * 32) == 0) {
                x_22524 = x_22525;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_25314 = 1;
        while (slt32(skip_threads_25314, 32)) {
            if (sle32(skip_threads_25314, local_tid_25304 -
                      squot32(local_tid_25304, 32) * 32) &&
                slt32(local_tid_25304, stage1_num_groups_25278)) {
                // read operands
                {
                    x_22524 = ((volatile __local
                                float *) scan_arr_mem_25308)[sext_i32_i64(local_tid_25304 -
                                                             skip_threads_25314)];
                }
                // perform operation
                {
                    bool inactive_25315 = slt32(srem32((local_tid_25304 + 1) *
                                                       (segscan_group_sizze_22520 *
                                                        sdiv_up32(m_18149 *
                                                                  iota_arg_18497,
                                                                  num_threads_25279)) -
                                                       1, iota_arg_18497),
                                                (local_tid_25304 + 1) *
                                                (segscan_group_sizze_22520 *
                                                 sdiv_up32(m_18149 *
                                                           iota_arg_18497,
                                                           num_threads_25279)) -
                                                1 - ((local_tid_25304 -
                                                      skip_threads_25314 + 1) *
                                                     (segscan_group_sizze_22520 *
                                                      sdiv_up32(m_18149 *
                                                                iota_arg_18497,
                                                                num_threads_25279)) -
                                                     1));
                    
                    if (inactive_25315) {
                        x_22524 = x_22525;
                    }
                    if (!inactive_25315) {
                        float res_22526 = x_22524 + x_22525;
                        
                        x_22524 = res_22526;
                    }
                }
            }
            if (sle32(wave_sizze_25306, skip_threads_25314)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_25314, local_tid_25304 -
                      squot32(local_tid_25304, 32) * 32) &&
                slt32(local_tid_25304, stage1_num_groups_25278)) {
                // write result
                {
                    ((volatile __local
                      float *) scan_arr_mem_25308)[sext_i32_i64(local_tid_25304)] =
                        x_22524;
                    x_22525 = x_22524;
                }
            }
            if (sle32(wave_sizze_25306, skip_threads_25314)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_25314 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_25304 - squot32(local_tid_25304, 32) * 32) == 31 &&
            slt32(local_tid_25304, stage1_num_groups_25278)) {
            ((volatile __local
              float *) scan_arr_mem_25308)[sext_i32_i64(squot32(local_tid_25304,
                                                                32))] = x_22524;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_25316;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_25304, 32) == 0 && slt32(local_tid_25304,
                                                           stage1_num_groups_25278)) {
                x_25312 = ((volatile __local
                            float *) scan_arr_mem_25308)[sext_i32_i64(local_tid_25304)];
                if ((local_tid_25304 - squot32(local_tid_25304, 32) * 32) ==
                    0) {
                    x_25311 = x_25312;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_25316 = 1;
            while (slt32(skip_threads_25316, 32)) {
                if (sle32(skip_threads_25316, local_tid_25304 -
                          squot32(local_tid_25304, 32) * 32) &&
                    (squot32(local_tid_25304, 32) == 0 && slt32(local_tid_25304,
                                                                stage1_num_groups_25278))) {
                    // read operands
                    {
                        x_25311 = ((volatile __local
                                    float *) scan_arr_mem_25308)[sext_i32_i64(local_tid_25304 -
                                                                 skip_threads_25316)];
                    }
                    // perform operation
                    {
                        bool inactive_25317 = slt32(srem32((local_tid_25304 *
                                                            32 + 32 - 1 + 1) *
                                                           (segscan_group_sizze_22520 *
                                                            sdiv_up32(m_18149 *
                                                                      iota_arg_18497,
                                                                      num_threads_25279)) -
                                                           1, iota_arg_18497),
                                                    (local_tid_25304 * 32 + 32 -
                                                     1 + 1) *
                                                    (segscan_group_sizze_22520 *
                                                     sdiv_up32(m_18149 *
                                                               iota_arg_18497,
                                                               num_threads_25279)) -
                                                    1 - (((local_tid_25304 -
                                                           skip_threads_25316) *
                                                          32 + 32 - 1 + 1) *
                                                         (segscan_group_sizze_22520 *
                                                          sdiv_up32(m_18149 *
                                                                    iota_arg_18497,
                                                                    num_threads_25279)) -
                                                         1));
                        
                        if (inactive_25317) {
                            x_25311 = x_25312;
                        }
                        if (!inactive_25317) {
                            float res_25313 = x_25311 + x_25312;
                            
                            x_25311 = res_25313;
                        }
                    }
                }
                if (sle32(wave_sizze_25306, skip_threads_25316)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_25316, local_tid_25304 -
                          squot32(local_tid_25304, 32) * 32) &&
                    (squot32(local_tid_25304, 32) == 0 && slt32(local_tid_25304,
                                                                stage1_num_groups_25278))) {
                    // write result
                    {
                        ((volatile __local
                          float *) scan_arr_mem_25308)[sext_i32_i64(local_tid_25304)] =
                            x_25311;
                        x_25312 = x_25311;
                    }
                }
                if (sle32(wave_sizze_25306, skip_threads_25316)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_25316 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_25304, 32) == 0 || !slt32(local_tid_25304,
                                                          stage1_num_groups_25278))) {
            // read operands
            {
                x_22525 = x_22524;
                x_22524 = ((__local
                            float *) scan_arr_mem_25308)[sext_i32_i64(squot32(local_tid_25304,
                                                                              32) -
                                                         1)];
            }
            // perform operation
            {
                bool inactive_25318 = slt32(srem32((local_tid_25304 + 1) *
                                                   (segscan_group_sizze_22520 *
                                                    sdiv_up32(m_18149 *
                                                              iota_arg_18497,
                                                              num_threads_25279)) -
                                                   1, iota_arg_18497),
                                            (local_tid_25304 + 1) *
                                            (segscan_group_sizze_22520 *
                                             sdiv_up32(m_18149 * iota_arg_18497,
                                                       num_threads_25279)) - 1 -
                                            ((squot32(local_tid_25304, 32) *
                                              32 - 1 + 1) *
                                             (segscan_group_sizze_22520 *
                                              sdiv_up32(m_18149 *
                                                        iota_arg_18497,
                                                        num_threads_25279)) -
                                             1));
                
                if (inactive_25318) {
                    x_22524 = x_22525;
                }
                if (!inactive_25318) {
                    float res_22526 = x_22524 + x_22525;
                    
                    x_22524 = res_22526;
                }
            }
            // write final result
            {
                ((__local
                  float *) scan_arr_mem_25308)[sext_i32_i64(local_tid_25304)] =
                    x_22524;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_25304, 32) == 0) {
            ((__local
              float *) scan_arr_mem_25308)[sext_i32_i64(local_tid_25304)] =
                x_22525;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // threads in bounds write scanned carries
    {
        if (slt32(gtid_22427, m_18149) && slt32(gtid_22437, iota_arg_18497)) {
            ((__global float *) mem_24239)[sext_i32_i64(gtid_22427) *
                                           sext_i32_i64(iota_arg_18497) +
                                           sext_i32_i64(gtid_22437)] = ((__local
                                                                         float *) scan_arr_mem_25308)[sext_i32_i64(local_tid_25304)];
        }
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_22520
}
__kernel void mainziscan_stage3_21355(__global int *global_failure,
                                      int32_t N_18148, int32_t m_18149,
                                      int32_t num_groups_21375, __global
                                      unsigned char *mem_24111,
                                      int32_t num_threads_24928,
                                      int32_t required_groups_24968)
{
    #define segscan_group_sizze_21374 (mainzisegscan_group_sizze_21349)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24969;
    int32_t local_tid_24970;
    int32_t group_sizze_24973;
    int32_t wave_sizze_24972;
    int32_t group_tid_24971;
    
    global_tid_24969 = get_global_id(0);
    local_tid_24970 = get_local_id(0);
    group_sizze_24973 = get_local_size(0);
    wave_sizze_24972 = LOCKSTEP_WIDTH;
    group_tid_24971 = get_group_id(0);
    
    int32_t phys_tid_21355;
    
    phys_tid_21355 = global_tid_24969;
    
    int32_t phys_group_id_24974;
    
    phys_group_id_24974 = get_group_id(0);
    for (int32_t i_24975 = 0; i_24975 < sdiv_up32(required_groups_24968 -
                                                  phys_group_id_24974,
                                                  num_groups_21375);
         i_24975++) {
        int32_t virt_group_id_24976 = phys_group_id_24974 + i_24975 *
                num_groups_21375;
        int32_t flat_idx_24977 = virt_group_id_24976 *
                segscan_group_sizze_21374 + local_tid_24970;
        int32_t gtid_21344 = squot32(flat_idx_24977, N_18148);
        int32_t gtid_21354 = flat_idx_24977 - squot32(flat_idx_24977, N_18148) *
                N_18148;
        int32_t orig_group_24978 = squot32(flat_idx_24977,
                                           segscan_group_sizze_21374 *
                                           sdiv_up32(m_18149 * N_18148,
                                                     num_threads_24928));
        int32_t carry_in_flat_idx_24979 = orig_group_24978 *
                (segscan_group_sizze_21374 * sdiv_up32(m_18149 * N_18148,
                                                       num_threads_24928)) - 1;
        
        if (slt32(gtid_21344, m_18149) && slt32(gtid_21354, N_18148)) {
            if (!(orig_group_24978 == 0 || (flat_idx_24977 ==
                                            (orig_group_24978 + 1) *
                                            (segscan_group_sizze_21374 *
                                             sdiv_up32(m_18149 * N_18148,
                                                       num_threads_24928)) -
                                            1 || slt32(srem32(flat_idx_24977,
                                                              N_18148),
                                                       flat_idx_24977 -
                                                       carry_in_flat_idx_24979)))) {
                int32_t x_21379;
                int32_t x_21380;
                
                x_21379 = ((__global
                            int32_t *) mem_24111)[sext_i32_i64(squot32(carry_in_flat_idx_24979,
                                                                       N_18148)) *
                                                  sext_i32_i64(N_18148) +
                                                  sext_i32_i64(carry_in_flat_idx_24979 -
                                                  squot32(carry_in_flat_idx_24979,
                                                          N_18148) * N_18148)];
                x_21380 = ((__global
                            int32_t *) mem_24111)[sext_i32_i64(gtid_21344) *
                                                  sext_i32_i64(N_18148) +
                                                  sext_i32_i64(gtid_21354)];
                
                int32_t res_21381;
                
                res_21381 = add32(x_21379, x_21380);
                x_21379 = res_21381;
                ((__global int32_t *) mem_24111)[sext_i32_i64(gtid_21344) *
                                                 sext_i32_i64(N_18148) +
                                                 sext_i32_i64(gtid_21354)] =
                    x_21379;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_21374
}
__kernel void mainziscan_stage3_22438(__global int *global_failure,
                                      int32_t m_18149, int32_t iota_arg_18497,
                                      int32_t num_groups_22521, __global
                                      unsigned char *mem_24239,
                                      int32_t num_threads_25279,
                                      int32_t required_groups_25319)
{
    #define segscan_group_sizze_22520 (mainzisegscan_group_sizze_22432)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25320;
    int32_t local_tid_25321;
    int32_t group_sizze_25324;
    int32_t wave_sizze_25323;
    int32_t group_tid_25322;
    
    global_tid_25320 = get_global_id(0);
    local_tid_25321 = get_local_id(0);
    group_sizze_25324 = get_local_size(0);
    wave_sizze_25323 = LOCKSTEP_WIDTH;
    group_tid_25322 = get_group_id(0);
    
    int32_t phys_tid_22438;
    
    phys_tid_22438 = global_tid_25320;
    
    int32_t phys_group_id_25325;
    
    phys_group_id_25325 = get_group_id(0);
    for (int32_t i_25326 = 0; i_25326 < sdiv_up32(required_groups_25319 -
                                                  phys_group_id_25325,
                                                  num_groups_22521);
         i_25326++) {
        int32_t virt_group_id_25327 = phys_group_id_25325 + i_25326 *
                num_groups_22521;
        int32_t flat_idx_25328 = virt_group_id_25327 *
                segscan_group_sizze_22520 + local_tid_25321;
        int32_t gtid_22427 = squot32(flat_idx_25328, iota_arg_18497);
        int32_t gtid_22437 = flat_idx_25328 - squot32(flat_idx_25328,
                                                      iota_arg_18497) *
                iota_arg_18497;
        int32_t orig_group_25329 = squot32(flat_idx_25328,
                                           segscan_group_sizze_22520 *
                                           sdiv_up32(m_18149 * iota_arg_18497,
                                                     num_threads_25279));
        int32_t carry_in_flat_idx_25330 = orig_group_25329 *
                (segscan_group_sizze_22520 * sdiv_up32(m_18149 * iota_arg_18497,
                                                       num_threads_25279)) - 1;
        
        if (slt32(gtid_22427, m_18149) && slt32(gtid_22437, iota_arg_18497)) {
            if (!(orig_group_25329 == 0 || (flat_idx_25328 ==
                                            (orig_group_25329 + 1) *
                                            (segscan_group_sizze_22520 *
                                             sdiv_up32(m_18149 * iota_arg_18497,
                                                       num_threads_25279)) -
                                            1 || slt32(srem32(flat_idx_25328,
                                                              iota_arg_18497),
                                                       flat_idx_25328 -
                                                       carry_in_flat_idx_25330)))) {
                float x_22524;
                float x_22525;
                
                x_22524 = ((__global
                            float *) mem_24239)[sext_i32_i64(squot32(carry_in_flat_idx_25330,
                                                                     iota_arg_18497)) *
                                                sext_i32_i64(iota_arg_18497) +
                                                sext_i32_i64(carry_in_flat_idx_25330 -
                                                squot32(carry_in_flat_idx_25330,
                                                        iota_arg_18497) *
                                                iota_arg_18497)];
                x_22525 = ((__global
                            float *) mem_24239)[sext_i32_i64(gtid_22427) *
                                                sext_i32_i64(iota_arg_18497) +
                                                sext_i32_i64(gtid_22437)];
                
                float res_22526;
                
                res_22526 = x_22524 + x_22525;
                x_22524 = res_22526;
                ((__global float *) mem_24239)[sext_i32_i64(gtid_22427) *
                                               sext_i32_i64(iota_arg_18497) +
                                               sext_i32_i64(gtid_22437)] =
                    x_22524;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segscan_group_sizze_22520
}
__kernel void mainzisegmap_18952(__global int *global_failure, int32_t N_18148,
                                 float freq_18154, int32_t k2p2zq_18165,
                                 __global
                                 unsigned char *mappingindices_mem_23522,
                                 __global unsigned char *mem_23529)
{
    #define segmap_group_sizze_19042 (mainzisegmap_group_sizze_18957)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24462;
    int32_t local_tid_24463;
    int32_t group_sizze_24466;
    int32_t wave_sizze_24465;
    int32_t group_tid_24464;
    
    global_tid_24462 = get_global_id(0);
    local_tid_24463 = get_local_id(0);
    group_sizze_24466 = get_local_size(0);
    wave_sizze_24465 = LOCKSTEP_WIDTH;
    group_tid_24464 = get_group_id(0);
    
    int32_t phys_tid_18952;
    
    phys_tid_18952 = global_tid_24462;
    
    int32_t gtid_18950;
    
    gtid_18950 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24464) *
                                      sext_i32_i64(segmap_group_sizze_19042) +
                                      sext_i32_i64(local_tid_24463),
                                      sext_i32_i64(N_18148)));
    
    int32_t gtid_18951;
    
    gtid_18951 = sext_i64_i32(sext_i32_i64(group_tid_24464) *
        sext_i32_i64(segmap_group_sizze_19042) + sext_i32_i64(local_tid_24463) -
        squot64(sext_i32_i64(group_tid_24464) *
                sext_i32_i64(segmap_group_sizze_19042) +
                sext_i32_i64(local_tid_24463), sext_i32_i64(N_18148)) *
        sext_i32_i64(N_18148));
    if (slt32(gtid_18950, k2p2zq_18165) && slt32(gtid_18951, N_18148)) {
        bool index_primexp_22703 = gtid_18950 == 0;
        float res_19050;
        
        if (index_primexp_22703) {
            res_19050 = 1.0F;
        } else {
            int32_t x_19049 = ((__global
                                int32_t *) mappingindices_mem_23522)[sext_i32_i64(gtid_18951)];
            bool cond_19051 = gtid_18950 == 1;
            float res_19052;
            
            if (cond_19051) {
                float res_19053 = sitofp_i32_f32(x_19049);
                
                res_19052 = res_19053;
            } else {
                int32_t r32_arg_19054 = sdiv32(gtid_18950, 2);
                float res_19055 = sitofp_i32_f32(r32_arg_19054);
                float res_19056 = sitofp_i32_f32(x_19049);
                float x_19057 = 6.2831855F * res_19055;
                float x_19058 = res_19056 * x_19057;
                float angle_19059 = x_19058 / freq_18154;
                int32_t x_19060 = smod32(gtid_18950, 2);
                bool cond_19061 = x_19060 == 0;
                float res_19062;
                
                if (cond_19061) {
                    float res_19063;
                    
                    res_19063 = futrts_sin32(angle_19059);
                    res_19062 = res_19063;
                } else {
                    float res_19064;
                    
                    res_19064 = futrts_cos32(angle_19059);
                    res_19062 = res_19064;
                }
                res_19052 = res_19062;
            }
            res_19050 = res_19052;
        }
        ((__global float *) mem_23529)[sext_i32_i64(gtid_18950) *
                                       sext_i32_i64(N_18148) +
                                       sext_i32_i64(gtid_18951)] = res_19050;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19042
}
__kernel void mainzisegmap_19152(__global int *global_failure, int32_t N_18148,
                                 float freq_18154, int32_t k2p2zq_18165,
                                 __global
                                 unsigned char *mappingindices_mem_23522,
                                 __global unsigned char *mem_23535)
{
    #define segmap_group_sizze_19238 (mainzisegmap_group_sizze_19157)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24467;
    int32_t local_tid_24468;
    int32_t group_sizze_24471;
    int32_t wave_sizze_24470;
    int32_t group_tid_24469;
    
    global_tid_24467 = get_global_id(0);
    local_tid_24468 = get_local_id(0);
    group_sizze_24471 = get_local_size(0);
    wave_sizze_24470 = LOCKSTEP_WIDTH;
    group_tid_24469 = get_group_id(0);
    
    int32_t phys_tid_19152;
    
    phys_tid_19152 = global_tid_24467;
    
    int32_t gtid_19150;
    
    gtid_19150 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24469) *
                                      sext_i32_i64(segmap_group_sizze_19238) +
                                      sext_i32_i64(local_tid_24468),
                                      sext_i32_i64(N_18148)));
    
    int32_t gtid_19151;
    
    gtid_19151 = sext_i64_i32(sext_i32_i64(group_tid_24469) *
        sext_i32_i64(segmap_group_sizze_19238) + sext_i32_i64(local_tid_24468) -
        squot64(sext_i32_i64(group_tid_24469) *
                sext_i32_i64(segmap_group_sizze_19238) +
                sext_i32_i64(local_tid_24468), sext_i32_i64(N_18148)) *
        sext_i32_i64(N_18148));
    if (slt32(gtid_19150, k2p2zq_18165) && slt32(gtid_19151, N_18148)) {
        bool index_primexp_22710 = gtid_19150 == 0;
        float res_19246;
        
        if (index_primexp_22710) {
            res_19246 = 1.0F;
        } else {
            int32_t x_19245 = ((__global
                                int32_t *) mappingindices_mem_23522)[sext_i32_i64(gtid_19151)];
            int32_t i_19247 = add32(1, gtid_19150);
            int32_t r32_arg_19248 = sdiv32(i_19247, 2);
            float res_19249 = sitofp_i32_f32(r32_arg_19248);
            float res_19250 = sitofp_i32_f32(x_19245);
            float x_19251 = 6.2831855F * res_19249;
            float x_19252 = res_19250 * x_19251;
            float angle_19253 = x_19252 / freq_18154;
            int32_t x_19254 = smod32(i_19247, 2);
            bool cond_19255 = x_19254 == 0;
            float res_19256;
            
            if (cond_19255) {
                float res_19257;
                
                res_19257 = futrts_sin32(angle_19253);
                res_19256 = res_19257;
            } else {
                float res_19258;
                
                res_19258 = futrts_cos32(angle_19253);
                res_19256 = res_19258;
            }
            res_19246 = res_19256;
        }
        ((__global float *) mem_23535)[sext_i32_i64(gtid_19150) *
                                       sext_i32_i64(N_18148) +
                                       sext_i32_i64(gtid_19151)] = res_19246;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19238
}
__kernel void mainzisegmap_19305(__global int *global_failure, int32_t N_18148,
                                 int32_t k2p2zq_18165, float res_18228, __global
                                 unsigned char *mem_23541, __global
                                 unsigned char *mem_23547)
{
    #define segmap_group_sizze_19343 (mainzisegmap_group_sizze_19310)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24472;
    int32_t local_tid_24473;
    int32_t group_sizze_24476;
    int32_t wave_sizze_24475;
    int32_t group_tid_24474;
    
    global_tid_24472 = get_global_id(0);
    local_tid_24473 = get_local_id(0);
    group_sizze_24476 = get_local_size(0);
    wave_sizze_24475 = LOCKSTEP_WIDTH;
    group_tid_24474 = get_group_id(0);
    
    int32_t phys_tid_19305;
    
    phys_tid_19305 = global_tid_24472;
    
    int32_t gtid_19303;
    
    gtid_19303 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24474) *
                                      sext_i32_i64(segmap_group_sizze_19343) +
                                      sext_i32_i64(local_tid_24473),
                                      sext_i32_i64(k2p2zq_18165)));
    
    int32_t gtid_19304;
    
    gtid_19304 = sext_i64_i32(sext_i32_i64(group_tid_24474) *
        sext_i32_i64(segmap_group_sizze_19343) + sext_i32_i64(local_tid_24473) -
        squot64(sext_i32_i64(group_tid_24474) *
                sext_i32_i64(segmap_group_sizze_19343) +
                sext_i32_i64(local_tid_24473), sext_i32_i64(k2p2zq_18165)) *
        sext_i32_i64(k2p2zq_18165));
    if (slt32(gtid_19303, N_18148) && slt32(gtid_19304, k2p2zq_18165)) {
        float x_19348 = ((__global
                          float *) mem_23541)[sext_i32_i64(gtid_19303) *
                                              sext_i32_i64(k2p2zq_18165) +
                                              sext_i32_i64(gtid_19304)];
        float res_19349 = res_18228 + x_19348;
        
        ((__global float *) mem_23547)[sext_i32_i64(gtid_19303) *
                                       sext_i32_i64(k2p2zq_18165) +
                                       sext_i32_i64(gtid_19304)] = res_19349;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19343
}
__kernel void mainzisegmap_19354(__global int *global_failure, int32_t N_18148,
                                 int32_t m_18149, int32_t n_18153,
                                 int32_t k2p2zq_18165, int32_t num_groups_19381,
                                 __global unsigned char *binop_p_mem_23536,
                                 __global unsigned char *mem_23547, __global
                                 unsigned char *mem_23552, __global
                                 unsigned char *mem_23558, __global
                                 unsigned char *mem_23605)
{
    #define segmap_group_sizze_19380 (mainzisegmap_group_sizze_19357)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24480;
    int32_t local_tid_24481;
    int32_t group_sizze_24484;
    int32_t wave_sizze_24483;
    int32_t group_tid_24482;
    
    global_tid_24480 = get_global_id(0);
    local_tid_24481 = get_local_id(0);
    group_sizze_24484 = get_local_size(0);
    wave_sizze_24483 = LOCKSTEP_WIDTH;
    group_tid_24482 = get_group_id(0);
    
    int32_t phys_tid_19354;
    
    phys_tid_19354 = global_tid_24480;
    
    int32_t phys_group_id_24485;
    
    phys_group_id_24485 = get_group_id(0);
    for (int32_t i_24486 = 0; i_24486 < sdiv_up32(sdiv_up32(m_18149,
                                                            segmap_group_sizze_19380) -
                                                  phys_group_id_24485,
                                                  num_groups_19381);
         i_24486++) {
        int32_t virt_group_id_24487 = phys_group_id_24485 + i_24486 *
                num_groups_19381;
        int32_t gtid_19353 = sext_i64_i32(sext_i32_i64(virt_group_id_24487) *
                sext_i32_i64(segmap_group_sizze_19380) +
                sext_i32_i64(local_tid_24481));
        
        if (slt32(gtid_19353, m_18149)) {
            for (int32_t i_23467 = 0; i_23467 < k2p2zq_18165; i_23467++) {
                for (int32_t i_23471 = 0; i_23471 < k2p2zq_18165; i_23471++) {
                    float res_19389;
                    float redout_23473 = 0.0F;
                    
                    for (int32_t i_23474 = 0; i_23474 < n_18153; i_23474++) {
                        float x_19393 = ((__global
                                          float *) mem_23552)[sext_i32_i64(i_23474) *
                                                              sext_i32_i64(m_18149) +
                                                              sext_i32_i64(gtid_19353)];
                        float x_19394 = ((__global
                                          float *) binop_p_mem_23536)[sext_i32_i64(i_23467) *
                                                                      sext_i32_i64(N_18148) +
                                                                      sext_i32_i64(i_23474)];
                        float x_19395 = ((__global
                                          float *) mem_23547)[sext_i32_i64(i_23474) *
                                                              sext_i32_i64(k2p2zq_18165) +
                                                              sext_i32_i64(i_23471)];
                        float x_19396 = x_19394 * x_19395;
                        bool res_19397;
                        
                        res_19397 = futrts_isnan32(x_19393);
                        
                        float y_19398;
                        
                        if (res_19397) {
                            y_19398 = 0.0F;
                        } else {
                            y_19398 = 1.0F;
                        }
                        
                        float res_19399 = x_19396 * y_19398;
                        float res_19392 = res_19399 + redout_23473;
                        float redout_tmp_24490 = res_19392;
                        
                        redout_23473 = redout_tmp_24490;
                    }
                    res_19389 = redout_23473;
                    ((__global
                      float *) mem_23558)[sext_i32_i64(phys_tid_19354) +
                                          (sext_i32_i64(i_23467) *
                                           sext_i32_i64(num_groups_19381 *
                                           segmap_group_sizze_19380 *
                                           k2p2zq_18165) +
                                           sext_i32_i64(i_23471) *
                                           sext_i32_i64(num_groups_19381 *
                                           segmap_group_sizze_19380))] =
                        res_19389;
                }
            }
            for (int32_t i_24491 = 0; i_24491 < k2p2zq_18165; i_24491++) {
                for (int32_t i_24492 = 0; i_24492 < k2p2zq_18165; i_24492++) {
                    ((__global float *) mem_23605)[sext_i32_i64(i_24491) *
                                                   sext_i32_i64(m_18149 *
                                                   k2p2zq_18165) +
                                                   sext_i32_i64(i_24492) *
                                                   sext_i32_i64(m_18149) +
                                                   sext_i32_i64(gtid_19353)] =
                        ((__global
                          float *) mem_23558)[sext_i32_i64(phys_tid_19354) +
                                              (sext_i32_i64(i_24491) *
                                               sext_i32_i64(num_groups_19381 *
                                               segmap_group_sizze_19380 *
                                               k2p2zq_18165) +
                                               sext_i32_i64(i_24492) *
                                               sext_i32_i64(num_groups_19381 *
                                               segmap_group_sizze_19380))];
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19380
}
__kernel void mainzisegmap_19402(__global int *global_failure, int32_t m_18149,
                                 int32_t N_18150, int32_t n_18153,
                                 int32_t k2p2zq_18165, int32_t num_groups_19584,
                                 __global unsigned char *images_mem_23523,
                                 __global unsigned char *mem_23541, __global
                                 unsigned char *mem_23547, __global
                                 unsigned char *mem_23609, __global
                                 unsigned char *mem_23628)
{
    #define segmap_group_sizze_19583 (mainzisegmap_group_sizze_19407)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24493;
    int32_t local_tid_24494;
    int32_t group_sizze_24497;
    int32_t wave_sizze_24496;
    int32_t group_tid_24495;
    
    global_tid_24493 = get_global_id(0);
    local_tid_24494 = get_local_id(0);
    group_sizze_24497 = get_local_size(0);
    wave_sizze_24496 = LOCKSTEP_WIDTH;
    group_tid_24495 = get_group_id(0);
    
    int32_t phys_tid_19402;
    
    phys_tid_19402 = global_tid_24493;
    
    int32_t phys_group_id_24498;
    
    phys_group_id_24498 = get_group_id(0);
    for (int32_t i_24499 = 0; i_24499 < sdiv_up32(sdiv_up32(m_18149 *
                                                            k2p2zq_18165,
                                                            segmap_group_sizze_19583) -
                                                  phys_group_id_24498,
                                                  num_groups_19584);
         i_24499++) {
        int32_t virt_group_id_24500 = phys_group_id_24498 + i_24499 *
                num_groups_19584;
        int32_t gtid_19400 =
                sext_i64_i32(squot64(sext_i32_i64(virt_group_id_24500) *
                                     sext_i32_i64(segmap_group_sizze_19583) +
                                     sext_i32_i64(local_tid_24494),
                                     sext_i32_i64(k2p2zq_18165)));
        int32_t gtid_19401 = sext_i64_i32(sext_i32_i64(virt_group_id_24500) *
                sext_i32_i64(segmap_group_sizze_19583) +
                sext_i32_i64(local_tid_24494) -
                squot64(sext_i32_i64(virt_group_id_24500) *
                        sext_i32_i64(segmap_group_sizze_19583) +
                        sext_i32_i64(local_tid_24494),
                        sext_i32_i64(k2p2zq_18165)) *
                sext_i32_i64(k2p2zq_18165));
        
        if (slt32(gtid_19400, m_18149) && slt32(gtid_19401, k2p2zq_18165)) {
            for (int32_t i_23477 = 0; i_23477 < k2p2zq_18165; i_23477++) {
                float res_19595;
                float redout_23479 = 0.0F;
                
                for (int32_t i_23480 = 0; i_23480 < n_18153; i_23480++) {
                    float x_19599 = ((__global
                                      float *) images_mem_23523)[sext_i32_i64(gtid_19400) *
                                                                 sext_i32_i64(N_18150) +
                                                                 sext_i32_i64(i_23480)];
                    float x_19600 = ((__global
                                      float *) mem_23541)[sext_i32_i64(i_23480) *
                                                          sext_i32_i64(k2p2zq_18165) +
                                                          sext_i32_i64(gtid_19401)];
                    float x_19601 = ((__global
                                      float *) mem_23547)[sext_i32_i64(i_23480) *
                                                          sext_i32_i64(k2p2zq_18165) +
                                                          sext_i32_i64(i_23477)];
                    float x_19602 = x_19600 * x_19601;
                    bool res_19603;
                    
                    res_19603 = futrts_isnan32(x_19599);
                    
                    float y_19604;
                    
                    if (res_19603) {
                        y_19604 = 0.0F;
                    } else {
                        y_19604 = 1.0F;
                    }
                    
                    float res_19605 = x_19602 * y_19604;
                    float res_19598 = res_19605 + redout_23479;
                    float redout_tmp_24502 = res_19598;
                    
                    redout_23479 = redout_tmp_24502;
                }
                res_19595 = redout_23479;
                ((__global float *) mem_23609)[sext_i32_i64(phys_tid_19402) +
                                               sext_i32_i64(i_23477) *
                                               sext_i32_i64(num_groups_19584 *
                                               segmap_group_sizze_19583)] =
                    res_19595;
            }
            for (int32_t i_24503 = 0; i_24503 < k2p2zq_18165; i_24503++) {
                ((__global float *) mem_23628)[sext_i32_i64(i_24503) *
                                               sext_i32_i64(k2p2zq_18165 *
                                               m_18149) +
                                               sext_i32_i64(gtid_19400) *
                                               sext_i32_i64(k2p2zq_18165) +
                                               sext_i32_i64(gtid_19401)] =
                    ((__global
                      float *) mem_23609)[sext_i32_i64(phys_tid_19402) +
                                          sext_i32_i64(i_24503) *
                                          sext_i32_i64(num_groups_19584 *
                                          segmap_group_sizze_19583)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19583
}
__kernel void mainzisegmap_19434(__global int *global_failure, int32_t m_18149,
                                 int32_t N_18150, int32_t n_18153,
                                 int32_t k2p2zq_18165, __global
                                 unsigned char *images_mem_23523, __global
                                 unsigned char *mem_23541, __global
                                 unsigned char *mem_23547, __global
                                 unsigned char *mem_23636)
{
    #define segmap_group_sizze_19612 (mainzisegmap_group_sizze_19441)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24504;
    int32_t local_tid_24505;
    int32_t group_sizze_24508;
    int32_t wave_sizze_24507;
    int32_t group_tid_24506;
    
    global_tid_24504 = get_global_id(0);
    local_tid_24505 = get_local_id(0);
    group_sizze_24508 = get_local_size(0);
    wave_sizze_24507 = LOCKSTEP_WIDTH;
    group_tid_24506 = get_group_id(0);
    
    int32_t phys_tid_19434;
    
    phys_tid_19434 = global_tid_24504;
    
    int32_t gtid_19431;
    
    gtid_19431 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24506) *
                                      sext_i32_i64(segmap_group_sizze_19612) +
                                      sext_i32_i64(local_tid_24505),
                                      sext_i32_i64(k2p2zq_18165) *
                                      sext_i32_i64(k2p2zq_18165)));
    
    int32_t gtid_19432;
    
    gtid_19432 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24506) *
                                      sext_i32_i64(segmap_group_sizze_19612) +
                                      sext_i32_i64(local_tid_24505) -
                                      squot64(sext_i32_i64(group_tid_24506) *
                                              sext_i32_i64(segmap_group_sizze_19612) +
                                              sext_i32_i64(local_tid_24505),
                                              sext_i32_i64(k2p2zq_18165) *
                                              sext_i32_i64(k2p2zq_18165)) *
                                      (sext_i32_i64(k2p2zq_18165) *
                                       sext_i32_i64(k2p2zq_18165)),
                                      sext_i32_i64(k2p2zq_18165)));
    
    int32_t gtid_19433;
    
    gtid_19433 = sext_i64_i32(sext_i32_i64(group_tid_24506) *
        sext_i32_i64(segmap_group_sizze_19612) + sext_i32_i64(local_tid_24505) -
        squot64(sext_i32_i64(group_tid_24506) *
                sext_i32_i64(segmap_group_sizze_19612) +
                sext_i32_i64(local_tid_24505), sext_i32_i64(k2p2zq_18165) *
                sext_i32_i64(k2p2zq_18165)) * (sext_i32_i64(k2p2zq_18165) *
                                               sext_i32_i64(k2p2zq_18165)) -
        squot64(sext_i32_i64(group_tid_24506) *
                sext_i32_i64(segmap_group_sizze_19612) +
                sext_i32_i64(local_tid_24505) -
                squot64(sext_i32_i64(group_tid_24506) *
                        sext_i32_i64(segmap_group_sizze_19612) +
                        sext_i32_i64(local_tid_24505),
                        sext_i32_i64(k2p2zq_18165) *
                        sext_i32_i64(k2p2zq_18165)) *
                (sext_i32_i64(k2p2zq_18165) * sext_i32_i64(k2p2zq_18165)),
                sext_i32_i64(k2p2zq_18165)) * sext_i32_i64(k2p2zq_18165));
    if ((slt32(gtid_19431, m_18149) && slt32(gtid_19432, k2p2zq_18165)) &&
        slt32(gtid_19433, k2p2zq_18165)) {
        float res_19625;
        float redout_23481 = 0.0F;
        
        for (int32_t i_23482 = 0; i_23482 < n_18153; i_23482++) {
            float x_19629 = ((__global
                              float *) images_mem_23523)[sext_i32_i64(gtid_19431) *
                                                         sext_i32_i64(N_18150) +
                                                         sext_i32_i64(i_23482)];
            float x_19630 = ((__global
                              float *) mem_23541)[sext_i32_i64(i_23482) *
                                                  sext_i32_i64(k2p2zq_18165) +
                                                  sext_i32_i64(gtid_19432)];
            float x_19631 = ((__global
                              float *) mem_23547)[sext_i32_i64(i_23482) *
                                                  sext_i32_i64(k2p2zq_18165) +
                                                  sext_i32_i64(gtid_19433)];
            float x_19632 = x_19630 * x_19631;
            bool res_19633;
            
            res_19633 = futrts_isnan32(x_19629);
            
            float y_19634;
            
            if (res_19633) {
                y_19634 = 0.0F;
            } else {
                y_19634 = 1.0F;
            }
            
            float res_19635 = x_19632 * y_19634;
            float res_19628 = res_19635 + redout_23481;
            float redout_tmp_24509 = res_19628;
            
            redout_23481 = redout_tmp_24509;
        }
        res_19625 = redout_23481;
        ((__global float *) mem_23636)[sext_i32_i64(gtid_19431) *
                                       sext_i32_i64(k2p2zq_18165 *
                                       k2p2zq_18165) +
                                       sext_i32_i64(gtid_19432) *
                                       sext_i32_i64(k2p2zq_18165) +
                                       sext_i32_i64(gtid_19433)] = res_19625;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_19612
}
__kernel void mainzisegmap_19951(__global int *global_failure, int32_t m_18149,
                                 int32_t k2p2zq_18165, int32_t m_18287,
                                 int32_t res_r_ixfn_23756,
                                 int32_t res_r_ixfn_23757,
                                 int32_t res_r_ixfn_23759, __global
                                 unsigned char *res_r_mem_23761, __global
                                 unsigned char *mem_23769)
{
    #define segmap_group_sizze_20634 (mainzisegmap_group_sizze_19958)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24631;
    int32_t local_tid_24632;
    int32_t group_sizze_24635;
    int32_t wave_sizze_24634;
    int32_t group_tid_24633;
    
    global_tid_24631 = get_global_id(0);
    local_tid_24632 = get_local_id(0);
    group_sizze_24635 = get_local_size(0);
    wave_sizze_24634 = LOCKSTEP_WIDTH;
    group_tid_24633 = get_group_id(0);
    
    int32_t phys_tid_19951;
    
    phys_tid_19951 = global_tid_24631;
    
    int32_t gtid_19948;
    
    gtid_19948 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24633) *
                                      sext_i32_i64(segmap_group_sizze_20634) +
                                      sext_i32_i64(local_tid_24632),
                                      sext_i32_i64(k2p2zq_18165) *
                                      sext_i32_i64(k2p2zq_18165)));
    
    int32_t gtid_19949;
    
    gtid_19949 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24633) *
                                      sext_i32_i64(segmap_group_sizze_20634) +
                                      sext_i32_i64(local_tid_24632) -
                                      squot64(sext_i32_i64(group_tid_24633) *
                                              sext_i32_i64(segmap_group_sizze_20634) +
                                              sext_i32_i64(local_tid_24632),
                                              sext_i32_i64(k2p2zq_18165) *
                                              sext_i32_i64(k2p2zq_18165)) *
                                      (sext_i32_i64(k2p2zq_18165) *
                                       sext_i32_i64(k2p2zq_18165)),
                                      sext_i32_i64(k2p2zq_18165)));
    
    int32_t gtid_19950;
    
    gtid_19950 = sext_i64_i32(sext_i32_i64(group_tid_24633) *
        sext_i32_i64(segmap_group_sizze_20634) + sext_i32_i64(local_tid_24632) -
        squot64(sext_i32_i64(group_tid_24633) *
                sext_i32_i64(segmap_group_sizze_20634) +
                sext_i32_i64(local_tid_24632), sext_i32_i64(k2p2zq_18165) *
                sext_i32_i64(k2p2zq_18165)) * (sext_i32_i64(k2p2zq_18165) *
                                               sext_i32_i64(k2p2zq_18165)) -
        squot64(sext_i32_i64(group_tid_24633) *
                sext_i32_i64(segmap_group_sizze_20634) +
                sext_i32_i64(local_tid_24632) -
                squot64(sext_i32_i64(group_tid_24633) *
                        sext_i32_i64(segmap_group_sizze_20634) +
                        sext_i32_i64(local_tid_24632),
                        sext_i32_i64(k2p2zq_18165) *
                        sext_i32_i64(k2p2zq_18165)) *
                (sext_i32_i64(k2p2zq_18165) * sext_i32_i64(k2p2zq_18165)),
                sext_i32_i64(k2p2zq_18165)) * sext_i32_i64(k2p2zq_18165));
    if ((slt32(gtid_19948, m_18149) && slt32(gtid_19949, k2p2zq_18165)) &&
        slt32(gtid_19950, k2p2zq_18165)) {
        int32_t index_primexp_22742 = m_18287 * gtid_19949;
        int32_t i_20642 = add32(k2p2zq_18165, gtid_19950);
        int32_t new_index_20643 = i_20642 + index_primexp_22742;
        float res_20644 = ((__global
                            float *) res_r_mem_23761)[sext_i32_i64(res_r_ixfn_23756) +
                                                      (sext_i32_i64(gtid_19948) *
                                                       sext_i32_i64(res_r_ixfn_23757) +
                                                       sext_i32_i64(new_index_20643) *
                                                       sext_i32_i64(res_r_ixfn_23759))];
        
        ((__global float *) mem_23769)[sext_i32_i64(gtid_19948) *
                                       sext_i32_i64(k2p2zq_18165 *
                                       k2p2zq_18165) +
                                       sext_i32_i64(gtid_19949) *
                                       sext_i32_i64(k2p2zq_18165) +
                                       sext_i32_i64(gtid_19950)] = res_20644;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20634
}
__kernel void mainzisegmap_20180(__global int *global_failure, int32_t m_18149,
                                 int32_t nm_18288, int32_t ctx_param_ext_23712,
                                 int32_t ctx_param_ext_23713,
                                 int32_t ctx_param_ext_23715, __global
                                 unsigned char *mem_param_23717, __global
                                 unsigned char *mem_23744)
{
    #define segmap_group_sizze_20583 (mainzisegmap_group_sizze_20185)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24625;
    int32_t local_tid_24626;
    int32_t group_sizze_24629;
    int32_t wave_sizze_24628;
    int32_t group_tid_24627;
    
    global_tid_24625 = get_global_id(0);
    local_tid_24626 = get_local_id(0);
    group_sizze_24629 = get_local_size(0);
    wave_sizze_24628 = LOCKSTEP_WIDTH;
    group_tid_24627 = get_group_id(0);
    
    int32_t phys_tid_20180;
    
    phys_tid_20180 = global_tid_24625;
    
    int32_t gtid_20178;
    
    gtid_20178 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24627) *
                                      sext_i32_i64(segmap_group_sizze_20583) +
                                      sext_i32_i64(local_tid_24626),
                                      sext_i32_i64(nm_18288)));
    
    int32_t gtid_20179;
    
    gtid_20179 = sext_i64_i32(sext_i32_i64(group_tid_24627) *
        sext_i32_i64(segmap_group_sizze_20583) + sext_i32_i64(local_tid_24626) -
        squot64(sext_i32_i64(group_tid_24627) *
                sext_i32_i64(segmap_group_sizze_20583) +
                sext_i32_i64(local_tid_24626), sext_i32_i64(nm_18288)) *
        sext_i32_i64(nm_18288));
    if (slt32(gtid_20178, m_18149) && slt32(gtid_20179, nm_18288)) {
        float write_value_20591 = ((__global
                                    float *) mem_23744)[sext_i32_i64(gtid_20178) *
                                                        sext_i32_i64(nm_18288) +
                                                        sext_i32_i64(gtid_20179)];
        
        if ((sle32(0, gtid_20178) && slt32(gtid_20178, m_18149)) && (sle32(0,
                                                                           gtid_20179) &&
                                                                     slt32(gtid_20179,
                                                                           nm_18288))) {
            ((__global
              float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                        (sext_i32_i64(gtid_20178) *
                                         sext_i32_i64(ctx_param_ext_23713) +
                                         sext_i32_i64(gtid_20179) *
                                         sext_i32_i64(ctx_param_ext_23715))] =
                write_value_20591;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20583
}
__kernel void mainzisegmap_20237(__global int *global_failure, int32_t m_18149,
                                 int32_t k2p2zq_18165, int32_t m_18287,
                                 int32_t nm_18288, int32_t i_20482,
                                 int32_t ctx_param_ext_23712,
                                 int32_t ctx_param_ext_23713,
                                 int32_t ctx_param_ext_23715, __global
                                 unsigned char *mem_param_23717, __global
                                 unsigned char *mem_23738, __global
                                 unsigned char *mem_23744)
{
    #define segmap_group_sizze_20550 (mainzisegmap_group_sizze_20242)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24620;
    int32_t local_tid_24621;
    int32_t group_sizze_24624;
    int32_t wave_sizze_24623;
    int32_t group_tid_24622;
    
    global_tid_24620 = get_global_id(0);
    local_tid_24621 = get_local_id(0);
    group_sizze_24624 = get_local_size(0);
    wave_sizze_24623 = LOCKSTEP_WIDTH;
    group_tid_24622 = get_group_id(0);
    
    int32_t phys_tid_20237;
    
    phys_tid_20237 = global_tid_24620;
    
    int32_t gtid_20235;
    
    gtid_20235 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24622) *
                                      sext_i32_i64(segmap_group_sizze_20550) +
                                      sext_i32_i64(local_tid_24621),
                                      sext_i32_i64(nm_18288)));
    
    int32_t gtid_20236;
    
    gtid_20236 = sext_i64_i32(sext_i32_i64(group_tid_24622) *
        sext_i32_i64(segmap_group_sizze_20550) + sext_i32_i64(local_tid_24621) -
        squot64(sext_i32_i64(group_tid_24622) *
                sext_i32_i64(segmap_group_sizze_20550) +
                sext_i32_i64(local_tid_24621), sext_i32_i64(nm_18288)) *
        sext_i32_i64(nm_18288));
    if (slt32(gtid_20235, m_18149) && slt32(gtid_20236, nm_18288)) {
        bool cond_20557 = ((__global
                            bool *) mem_23738)[sext_i32_i64(gtid_20235)];
        int32_t res_20559 = sdiv32(gtid_20236, m_18287);
        int32_t res_20560 = smod32(gtid_20236, m_18287);
        float res_20561;
        
        if (cond_20557) {
            int32_t x_20562 = mul32(m_18287, res_20559);
            int32_t i_20563 = add32(res_20560, x_20562);
            float res_20564 = ((__global
                                float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                                          (sext_i32_i64(gtid_20235) *
                                                           sext_i32_i64(ctx_param_ext_23713) +
                                                           sext_i32_i64(i_20563) *
                                                           sext_i32_i64(ctx_param_ext_23715))];
            
            res_20561 = res_20564;
        } else {
            float v1_20556 = ((__global
                               float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                                         (sext_i32_i64(gtid_20235) *
                                                          sext_i32_i64(ctx_param_ext_23713) +
                                                          sext_i32_i64(i_20482) *
                                                          sext_i32_i64(ctx_param_ext_23715))];
            float x_20565 = ((__global
                              float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                                        (sext_i32_i64(gtid_20235) *
                                                         sext_i32_i64(ctx_param_ext_23713) +
                                                         sext_i32_i64(res_20560) *
                                                         sext_i32_i64(ctx_param_ext_23715))];
            float x_20566 = x_20565 / v1_20556;
            int32_t y_20567 = sub32(k2p2zq_18165, 1);
            bool cond_20568 = slt32(res_20559, y_20567);
            float res_20569;
            
            if (cond_20568) {
                int32_t x_20570 = add32(1, res_20559);
                int32_t x_20571 = mul32(m_18287, x_20570);
                int32_t i_20572 = add32(res_20560, x_20571);
                float x_20573 = ((__global
                                  float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                                            (sext_i32_i64(gtid_20235) *
                                                             sext_i32_i64(ctx_param_ext_23713) +
                                                             sext_i32_i64(i_20572) *
                                                             sext_i32_i64(ctx_param_ext_23715))];
                int32_t i_20574 = add32(i_20482, x_20571);
                float x_20575 = ((__global
                                  float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                                            (sext_i32_i64(gtid_20235) *
                                                             sext_i32_i64(ctx_param_ext_23713) +
                                                             sext_i32_i64(i_20574) *
                                                             sext_i32_i64(ctx_param_ext_23715))];
                float y_20576 = x_20566 * x_20575;
                float res_20577 = x_20573 - y_20576;
                
                res_20569 = res_20577;
            } else {
                res_20569 = x_20566;
            }
            res_20561 = res_20569;
        }
        ((__global float *) mem_23744)[sext_i32_i64(gtid_20235) *
                                       sext_i32_i64(nm_18288) +
                                       sext_i32_i64(gtid_20236)] = res_20561;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20550
}
__kernel void mainzisegmap_20303(__global int *global_failure, int32_t m_18149,
                                 int32_t i_20482, int32_t ctx_param_ext_23712,
                                 int32_t ctx_param_ext_23713,
                                 int32_t ctx_param_ext_23715, __global
                                 unsigned char *mem_param_23717, __global
                                 unsigned char *mem_23738)
{
    #define segmap_group_sizze_20526 (mainzisegmap_group_sizze_20306)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24615;
    int32_t local_tid_24616;
    int32_t group_sizze_24619;
    int32_t wave_sizze_24618;
    int32_t group_tid_24617;
    
    global_tid_24615 = get_global_id(0);
    local_tid_24616 = get_local_id(0);
    group_sizze_24619 = get_local_size(0);
    wave_sizze_24618 = LOCKSTEP_WIDTH;
    group_tid_24617 = get_group_id(0);
    
    int32_t phys_tid_20303;
    
    phys_tid_20303 = global_tid_24615;
    
    int32_t gtid_20302;
    
    gtid_20302 = sext_i64_i32(sext_i32_i64(group_tid_24617) *
        sext_i32_i64(segmap_group_sizze_20526) + sext_i32_i64(local_tid_24616));
    if (slt32(gtid_20302, m_18149)) {
        float v1_20533 = ((__global
                           float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                                     (sext_i32_i64(gtid_20302) *
                                                      sext_i32_i64(ctx_param_ext_23713) +
                                                      sext_i32_i64(i_20482) *
                                                      sext_i32_i64(ctx_param_ext_23715))];
        bool cond_20534 = v1_20533 == 0.0F;
        
        ((__global bool *) mem_23738)[sext_i32_i64(gtid_20302)] = cond_20534;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20526
}
__kernel void mainzisegmap_20411(__global int *global_failure, int32_t m_18149,
                                 int32_t k2p2zq_18165, int32_t m_18287,
                                 int32_t nm_18288, __global
                                 unsigned char *res_mem_23668, __global
                                 unsigned char *mem_23709)
{
    #define segmap_group_sizze_20465 (mainzisegmap_group_sizze_20416)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24582;
    int32_t local_tid_24583;
    int32_t group_sizze_24586;
    int32_t wave_sizze_24585;
    int32_t group_tid_24584;
    
    global_tid_24582 = get_global_id(0);
    local_tid_24583 = get_local_id(0);
    group_sizze_24586 = get_local_size(0);
    wave_sizze_24585 = LOCKSTEP_WIDTH;
    group_tid_24584 = get_group_id(0);
    
    int32_t phys_tid_20411;
    
    phys_tid_20411 = global_tid_24582;
    
    int32_t gtid_20409;
    
    gtid_20409 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24584) *
                                      sext_i32_i64(segmap_group_sizze_20465) +
                                      sext_i32_i64(local_tid_24583),
                                      sext_i32_i64(nm_18288)));
    
    int32_t gtid_20410;
    
    gtid_20410 = sext_i64_i32(sext_i32_i64(group_tid_24584) *
        sext_i32_i64(segmap_group_sizze_20465) + sext_i32_i64(local_tid_24583) -
        squot64(sext_i32_i64(group_tid_24584) *
                sext_i32_i64(segmap_group_sizze_20465) +
                sext_i32_i64(local_tid_24583), sext_i32_i64(nm_18288)) *
        sext_i32_i64(nm_18288));
    if (slt32(gtid_20409, m_18149) && slt32(gtid_20410, nm_18288)) {
        int32_t res_20472 = sdiv32(gtid_20410, m_18287);
        int32_t res_20473 = smod32(gtid_20410, m_18287);
        bool cond_20474 = slt32(res_20473, k2p2zq_18165);
        float res_20475;
        
        if (cond_20474) {
            float res_20476 = ((__global
                                float *) res_mem_23668)[sext_i32_i64(gtid_20409) *
                                                        sext_i32_i64(k2p2zq_18165 *
                                                        k2p2zq_18165) +
                                                        sext_i32_i64(res_20472) *
                                                        sext_i32_i64(k2p2zq_18165) +
                                                        sext_i32_i64(res_20473)];
            
            res_20475 = res_20476;
        } else {
            int32_t y_20477 = add32(k2p2zq_18165, res_20472);
            bool cond_20478 = res_20473 == y_20477;
            float res_20479;
            
            if (cond_20478) {
                res_20479 = 1.0F;
            } else {
                res_20479 = 0.0F;
            }
            res_20475 = res_20479;
        }
        ((__global float *) mem_23709)[sext_i32_i64(gtid_20409) *
                                       sext_i32_i64(nm_18288) +
                                       sext_i32_i64(gtid_20410)] = res_20475;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20465
}
__kernel void mainzisegmap_20651(__global int *global_failure, int32_t N_18148,
                                 int32_t m_18149, int32_t n_18153,
                                 int32_t k2p2zq_18165, int32_t num_groups_20674,
                                 __global unsigned char *binop_p_mem_23536,
                                 __global unsigned char *mem_23775, __global
                                 unsigned char *mem_23779, __global
                                 unsigned char *mem_23796)
{
    #define segmap_group_sizze_20673 (mainzisegmap_group_sizze_20654)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24639;
    int32_t local_tid_24640;
    int32_t group_sizze_24643;
    int32_t wave_sizze_24642;
    int32_t group_tid_24641;
    
    global_tid_24639 = get_global_id(0);
    local_tid_24640 = get_local_id(0);
    group_sizze_24643 = get_local_size(0);
    wave_sizze_24642 = LOCKSTEP_WIDTH;
    group_tid_24641 = get_group_id(0);
    
    int32_t phys_tid_20651;
    
    phys_tid_20651 = global_tid_24639;
    
    int32_t phys_group_id_24644;
    
    phys_group_id_24644 = get_group_id(0);
    for (int32_t i_24645 = 0; i_24645 < sdiv_up32(sdiv_up32(m_18149,
                                                            segmap_group_sizze_20673) -
                                                  phys_group_id_24644,
                                                  num_groups_20674);
         i_24645++) {
        int32_t virt_group_id_24646 = phys_group_id_24644 + i_24645 *
                num_groups_20674;
        int32_t gtid_20650 = sext_i64_i32(sext_i32_i64(virt_group_id_24646) *
                sext_i32_i64(segmap_group_sizze_20673) +
                sext_i32_i64(local_tid_24640));
        
        if (slt32(gtid_20650, m_18149)) {
            for (int32_t i_23485 = 0; i_23485 < k2p2zq_18165; i_23485++) {
                float res_20680;
                float redout_23487 = 0.0F;
                
                for (int32_t i_23488 = 0; i_23488 < n_18153; i_23488++) {
                    float x_20685 = ((__global
                                      float *) mem_23775)[sext_i32_i64(i_23488) *
                                                          sext_i32_i64(m_18149) +
                                                          sext_i32_i64(gtid_20650)];
                    bool res_20686;
                    
                    res_20686 = futrts_isnan32(x_20685);
                    
                    float res_20687;
                    
                    if (res_20686) {
                        res_20687 = 0.0F;
                    } else {
                        float x_20684 = ((__global
                                          float *) binop_p_mem_23536)[sext_i32_i64(i_23485) *
                                                                      sext_i32_i64(N_18148) +
                                                                      sext_i32_i64(i_23488)];
                        float res_20688 = x_20684 * x_20685;
                        
                        res_20687 = res_20688;
                    }
                    
                    float res_20683 = res_20687 + redout_23487;
                    float redout_tmp_24648 = res_20683;
                    
                    redout_23487 = redout_tmp_24648;
                }
                res_20680 = redout_23487;
                ((__global float *) mem_23779)[sext_i32_i64(phys_tid_20651) +
                                               sext_i32_i64(i_23485) *
                                               sext_i32_i64(num_groups_20674 *
                                               segmap_group_sizze_20673)] =
                    res_20680;
            }
            for (int32_t i_24649 = 0; i_24649 < k2p2zq_18165; i_24649++) {
                ((__global float *) mem_23796)[sext_i32_i64(i_24649) *
                                               sext_i32_i64(m_18149) +
                                               sext_i32_i64(gtid_20650)] =
                    ((__global
                      float *) mem_23779)[sext_i32_i64(phys_tid_20651) +
                                          sext_i32_i64(i_24649) *
                                          sext_i32_i64(num_groups_20674 *
                                          segmap_group_sizze_20673)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20673
}
__kernel void mainzisegmap_20811(__global int *global_failure, int32_t m_18149,
                                 int32_t k2p2zq_18165, int32_t num_groups_20833,
                                 __global unsigned char *mem_23893, __global
                                 unsigned char *mem_23898, __global
                                 unsigned char *mem_23902, __global
                                 unsigned char *mem_23919)
{
    #define segmap_group_sizze_20832 (mainzisegmap_group_sizze_20814)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24726;
    int32_t local_tid_24727;
    int32_t group_sizze_24730;
    int32_t wave_sizze_24729;
    int32_t group_tid_24728;
    
    global_tid_24726 = get_global_id(0);
    local_tid_24727 = get_local_id(0);
    group_sizze_24730 = get_local_size(0);
    wave_sizze_24729 = LOCKSTEP_WIDTH;
    group_tid_24728 = get_group_id(0);
    
    int32_t phys_tid_20811;
    
    phys_tid_20811 = global_tid_24726;
    
    int32_t phys_group_id_24731;
    
    phys_group_id_24731 = get_group_id(0);
    for (int32_t i_24732 = 0; i_24732 < sdiv_up32(sdiv_up32(m_18149,
                                                            segmap_group_sizze_20832) -
                                                  phys_group_id_24731,
                                                  num_groups_20833);
         i_24732++) {
        int32_t virt_group_id_24733 = phys_group_id_24731 + i_24732 *
                num_groups_20833;
        int32_t gtid_20810 = sext_i64_i32(sext_i32_i64(virt_group_id_24733) *
                sext_i32_i64(segmap_group_sizze_20832) +
                sext_i32_i64(local_tid_24727));
        
        if (slt32(gtid_20810, m_18149)) {
            for (int32_t i_23495 = 0; i_23495 < k2p2zq_18165; i_23495++) {
                float res_20840;
                float redout_23497 = 0.0F;
                
                for (int32_t i_23498 = 0; i_23498 < k2p2zq_18165; i_23498++) {
                    float x_20844 = ((__global
                                      float *) mem_23898)[sext_i32_i64(i_23498) *
                                                          sext_i32_i64(m_18149) +
                                                          sext_i32_i64(gtid_20810)];
                    float x_20845 = ((__global
                                      float *) mem_23893)[sext_i32_i64(i_23495) *
                                                          sext_i32_i64(m_18149 *
                                                          k2p2zq_18165) +
                                                          sext_i32_i64(i_23498) *
                                                          sext_i32_i64(m_18149) +
                                                          sext_i32_i64(gtid_20810)];
                    float res_20846 = x_20844 * x_20845;
                    float res_20843 = res_20846 + redout_23497;
                    float redout_tmp_24735 = res_20843;
                    
                    redout_23497 = redout_tmp_24735;
                }
                res_20840 = redout_23497;
                ((__global float *) mem_23902)[sext_i32_i64(phys_tid_20811) +
                                               sext_i32_i64(i_23495) *
                                               sext_i32_i64(num_groups_20833 *
                                               segmap_group_sizze_20832)] =
                    res_20840;
            }
            for (int32_t i_24736 = 0; i_24736 < k2p2zq_18165; i_24736++) {
                ((__global float *) mem_23919)[sext_i32_i64(i_24736) *
                                               sext_i32_i64(m_18149) +
                                               sext_i32_i64(gtid_20810)] =
                    ((__global
                      float *) mem_23902)[sext_i32_i64(phys_tid_20811) +
                                          sext_i32_i64(i_24736) *
                                          sext_i32_i64(num_groups_20833 *
                                          segmap_group_sizze_20832)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20832
}
__kernel void mainzisegmap_20849(__global int *global_failure, int32_t m_18149,
                                 int32_t k2p2zq_18165, __global
                                 unsigned char *res_mem_23886, __global
                                 unsigned char *mem_23926, __global
                                 unsigned char *mem_23932)
{
    #define segmap_group_sizze_20920 (mainzisegmap_group_sizze_20854)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24737;
    int32_t local_tid_24738;
    int32_t group_sizze_24741;
    int32_t wave_sizze_24740;
    int32_t group_tid_24739;
    
    global_tid_24737 = get_global_id(0);
    local_tid_24738 = get_local_id(0);
    group_sizze_24741 = get_local_size(0);
    wave_sizze_24740 = LOCKSTEP_WIDTH;
    group_tid_24739 = get_group_id(0);
    
    int32_t phys_tid_20849;
    
    phys_tid_20849 = global_tid_24737;
    
    int32_t gtid_20847;
    
    gtid_20847 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24739) *
                                      sext_i32_i64(segmap_group_sizze_20920) +
                                      sext_i32_i64(local_tid_24738),
                                      sext_i32_i64(k2p2zq_18165)));
    
    int32_t gtid_20848;
    
    gtid_20848 = sext_i64_i32(sext_i32_i64(group_tid_24739) *
        sext_i32_i64(segmap_group_sizze_20920) + sext_i32_i64(local_tid_24738) -
        squot64(sext_i32_i64(group_tid_24739) *
                sext_i32_i64(segmap_group_sizze_20920) +
                sext_i32_i64(local_tid_24738), sext_i32_i64(k2p2zq_18165)) *
        sext_i32_i64(k2p2zq_18165));
    if (slt32(gtid_20847, m_18149) && slt32(gtid_20848, k2p2zq_18165)) {
        float res_20931;
        float redout_23499 = 0.0F;
        
        for (int32_t i_23500 = 0; i_23500 < k2p2zq_18165; i_23500++) {
            float x_20935 = ((__global
                              float *) res_mem_23886)[sext_i32_i64(gtid_20847) *
                                                      sext_i32_i64(k2p2zq_18165) +
                                                      sext_i32_i64(i_23500)];
            float x_20936 = ((__global
                              float *) mem_23926)[sext_i32_i64(i_23500) *
                                                  sext_i32_i64(k2p2zq_18165 *
                                                  m_18149) +
                                                  sext_i32_i64(gtid_20847) *
                                                  sext_i32_i64(k2p2zq_18165) +
                                                  sext_i32_i64(gtid_20848)];
            float res_20937 = x_20935 * x_20936;
            float res_20934 = res_20937 + redout_23499;
            float redout_tmp_24742 = res_20934;
            
            redout_23499 = redout_tmp_24742;
        }
        res_20931 = redout_23499;
        ((__global float *) mem_23932)[sext_i32_i64(gtid_20847) *
                                       sext_i32_i64(k2p2zq_18165) +
                                       sext_i32_i64(gtid_20848)] = res_20931;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20920
}
__kernel void mainzisegmap_20962(__global int *global_failure, int32_t N_18148,
                                 int32_t m_18149, int32_t k2p2zq_18165,
                                 int32_t num_groups_20983, __global
                                 unsigned char *mem_23547, __global
                                 unsigned char *mem_23951, __global
                                 unsigned char *mem_23955, __global
                                 unsigned char *mem_23972)
{
    #define segmap_group_sizze_20982 (mainzisegmap_group_sizze_20965)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24806;
    int32_t local_tid_24807;
    int32_t group_sizze_24810;
    int32_t wave_sizze_24809;
    int32_t group_tid_24808;
    
    global_tid_24806 = get_global_id(0);
    local_tid_24807 = get_local_id(0);
    group_sizze_24810 = get_local_size(0);
    wave_sizze_24809 = LOCKSTEP_WIDTH;
    group_tid_24808 = get_group_id(0);
    
    int32_t phys_tid_20962;
    
    phys_tid_20962 = global_tid_24806;
    
    int32_t phys_group_id_24811;
    
    phys_group_id_24811 = get_group_id(0);
    for (int32_t i_24812 = 0; i_24812 < sdiv_up32(sdiv_up32(m_18149,
                                                            segmap_group_sizze_20982) -
                                                  phys_group_id_24811,
                                                  num_groups_20983);
         i_24812++) {
        int32_t virt_group_id_24813 = phys_group_id_24811 + i_24812 *
                num_groups_20983;
        int32_t gtid_20961 = sext_i64_i32(sext_i32_i64(virt_group_id_24813) *
                sext_i32_i64(segmap_group_sizze_20982) +
                sext_i32_i64(local_tid_24807));
        
        if (slt32(gtid_20961, m_18149)) {
            for (int32_t i_23503 = 0; i_23503 < N_18148; i_23503++) {
                float res_20989;
                float redout_23505 = 0.0F;
                
                for (int32_t i_23506 = 0; i_23506 < k2p2zq_18165; i_23506++) {
                    float x_20993 = ((__global
                                      float *) mem_23951)[sext_i32_i64(i_23506) *
                                                          sext_i32_i64(m_18149) +
                                                          sext_i32_i64(gtid_20961)];
                    float x_20994 = ((__global
                                      float *) mem_23547)[sext_i32_i64(i_23503) *
                                                          sext_i32_i64(k2p2zq_18165) +
                                                          sext_i32_i64(i_23506)];
                    float res_20995 = x_20993 * x_20994;
                    float res_20992 = res_20995 + redout_23505;
                    float redout_tmp_24815 = res_20992;
                    
                    redout_23505 = redout_tmp_24815;
                }
                res_20989 = redout_23505;
                ((__global float *) mem_23955)[sext_i32_i64(phys_tid_20962) +
                                               sext_i32_i64(i_23503) *
                                               sext_i32_i64(num_groups_20983 *
                                               segmap_group_sizze_20982)] =
                    res_20989;
            }
            for (int32_t i_24816 = 0; i_24816 < N_18148; i_24816++) {
                ((__global float *) mem_23972)[sext_i32_i64(i_24816) *
                                               sext_i32_i64(m_18149) +
                                               sext_i32_i64(gtid_20961)] =
                    ((__global
                      float *) mem_23955)[sext_i32_i64(phys_tid_20962) +
                                          sext_i32_i64(i_24816) *
                                          sext_i32_i64(num_groups_20983 *
                                          segmap_group_sizze_20982)];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_20982
}
__kernel void mainzisegmap_21230(__global int *global_failure, int32_t N_18148,
                                 int32_t m_18149, __global
                                 unsigned char *mem_24072, __global
                                 unsigned char *mem_24077, __global
                                 unsigned char *mem_24111, __global
                                 unsigned char *mem_24116)
{
    #define segmap_group_sizze_21450 (mainzisegmap_group_sizze_21235)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24985;
    int32_t local_tid_24986;
    int32_t group_sizze_24989;
    int32_t wave_sizze_24988;
    int32_t group_tid_24987;
    
    global_tid_24985 = get_global_id(0);
    local_tid_24986 = get_local_id(0);
    group_sizze_24989 = get_local_size(0);
    wave_sizze_24988 = LOCKSTEP_WIDTH;
    group_tid_24987 = get_group_id(0);
    
    int32_t phys_tid_21230;
    
    phys_tid_21230 = global_tid_24985;
    
    int32_t gtid_21228;
    
    gtid_21228 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24987) *
                                      sext_i32_i64(segmap_group_sizze_21450) +
                                      sext_i32_i64(local_tid_24986),
                                      sext_i32_i64(N_18148)));
    
    int32_t gtid_21229;
    
    gtid_21229 = sext_i64_i32(sext_i32_i64(group_tid_24987) *
        sext_i32_i64(segmap_group_sizze_21450) + sext_i32_i64(local_tid_24986) -
        squot64(sext_i32_i64(group_tid_24987) *
                sext_i32_i64(segmap_group_sizze_21450) +
                sext_i32_i64(local_tid_24986), sext_i32_i64(N_18148)) *
        sext_i32_i64(N_18148));
    if (slt32(gtid_21228, m_18149) && slt32(gtid_21229, N_18148)) {
        float x_21460 = ((__global
                          float *) mem_24116)[sext_i32_i64(gtid_21228) *
                                              sext_i32_i64(N_18148) +
                                              sext_i32_i64(gtid_21229)];
        bool res_21463;
        
        res_21463 = futrts_isnan32(x_21460);
        
        bool res_21464 = !res_21463;
        int32_t res_21465;
        
        if (res_21464) {
            int32_t x_21461 = ((__global
                                int32_t *) mem_24111)[sext_i32_i64(gtid_21228) *
                                                      sext_i32_i64(N_18148) +
                                                      sext_i32_i64(gtid_21229)];
            int32_t res_21466 = sub32(x_21461, 1);
            
            res_21465 = res_21466;
        } else {
            res_21465 = -1;
        }
        if ((sle32(0, gtid_21228) && slt32(gtid_21228, m_18149)) && (sle32(0,
                                                                           res_21465) &&
                                                                     slt32(res_21465,
                                                                           N_18148))) {
            ((__global int32_t *) mem_24077)[sext_i32_i64(gtid_21228) *
                                             sext_i32_i64(N_18148) +
                                             sext_i32_i64(res_21465)] =
                gtid_21229;
        }
        if ((sle32(0, gtid_21228) && slt32(gtid_21228, m_18149)) && (sle32(0,
                                                                           res_21465) &&
                                                                     slt32(res_21465,
                                                                           N_18148))) {
            ((__global float *) mem_24072)[sext_i32_i64(gtid_21228) *
                                           sext_i32_i64(N_18148) +
                                           sext_i32_i64(res_21465)] = x_21460;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_21450
}
__kernel void mainzisegmap_21481(__global int *global_failure, int32_t m_18149,
                                 int32_t n_18153, float hfrac_18155,
                                 int32_t k2p2_18163, __global
                                 unsigned char *mem_24128, __global
                                 unsigned char *mem_24133, __global
                                 unsigned char *mem_24137, __global
                                 unsigned char *mem_24140, __global
                                 unsigned char *mem_24143)
{
    #define segmap_group_sizze_21517 (mainzisegmap_group_sizze_21484)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24993;
    int32_t local_tid_24994;
    int32_t group_sizze_24997;
    int32_t wave_sizze_24996;
    int32_t group_tid_24995;
    
    global_tid_24993 = get_global_id(0);
    local_tid_24994 = get_local_id(0);
    group_sizze_24997 = get_local_size(0);
    wave_sizze_24996 = LOCKSTEP_WIDTH;
    group_tid_24995 = get_group_id(0);
    
    int32_t phys_tid_21481;
    
    phys_tid_21481 = global_tid_24993;
    
    int32_t gtid_21480;
    
    gtid_21480 = sext_i64_i32(sext_i32_i64(group_tid_24995) *
        sext_i32_i64(segmap_group_sizze_21517) + sext_i32_i64(local_tid_24994));
    if (slt32(gtid_21480, m_18149)) {
        int32_t res_21526;
        int32_t redout_23511 = 0;
        
        for (int32_t i_23512 = 0; i_23512 < n_18153; i_23512++) {
            float x_21530 = ((__global
                              float *) mem_24128)[sext_i32_i64(i_23512) *
                                                  sext_i32_i64(m_18149) +
                                                  sext_i32_i64(gtid_21480)];
            bool res_21531;
            
            res_21531 = futrts_isnan32(x_21530);
            
            bool cond_21532 = !res_21531;
            int32_t res_21533 = btoi_bool_i32(cond_21532);
            int32_t res_21529 = add32(res_21533, redout_23511);
            int32_t redout_tmp_24998 = res_21529;
            
            redout_23511 = redout_tmp_24998;
        }
        res_21526 = redout_23511;
        
        float res_21534;
        float redout_23513 = 0.0F;
        
        for (int32_t i_23514 = 0; i_23514 < n_18153; i_23514++) {
            bool cond_21540 = slt32(i_23514, res_21526);
            float res_21541;
            
            if (cond_21540) {
                float x_elem_21539 = ((__global
                                       float *) mem_24133)[sext_i32_i64(i_23514) *
                                                           sext_i32_i64(m_18149) +
                                                           sext_i32_i64(gtid_21480)];
                
                res_21541 = x_elem_21539;
            } else {
                res_21541 = 0.0F;
            }
            
            float res_21542 = res_21541 * res_21541;
            float res_21537 = res_21542 + redout_23513;
            float redout_tmp_24999 = res_21537;
            
            redout_23513 = redout_tmp_24999;
        }
        res_21534 = redout_23513;
        
        int32_t r32_arg_21543 = sub32(res_21526, k2p2_18163);
        float res_21544 = sitofp_i32_f32(r32_arg_21543);
        float sqrt_arg_21545 = res_21534 / res_21544;
        float res_21546;
        
        res_21546 = futrts_sqrt32(sqrt_arg_21545);
        
        float res_21547 = sitofp_i32_f32(res_21526);
        float t32_arg_21548 = hfrac_18155 * res_21547;
        int32_t res_21549 = fptosi_f32_i32(t32_arg_21548);
        
        ((__global int32_t *) mem_24137)[sext_i32_i64(gtid_21480)] = res_21549;
        ((__global int32_t *) mem_24140)[sext_i32_i64(gtid_21480)] = res_21526;
        ((__global float *) mem_24143)[sext_i32_i64(gtid_21480)] = res_21546;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_21517
}
__kernel void mainzisegmap_21586(__global int *global_failure, int32_t m_18149,
                                 float hfrac_18155, int32_t k2p2_18163, __global
                                 unsigned char *mem_24159, __global
                                 unsigned char *mem_24163, __global
                                 unsigned char *mem_24167, __global
                                 unsigned char *mem_24170)
{
    #define segmap_group_sizze_21681 (mainzisegmap_group_sizze_21589)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25132;
    int32_t local_tid_25133;
    int32_t group_sizze_25136;
    int32_t wave_sizze_25135;
    int32_t group_tid_25134;
    
    global_tid_25132 = get_global_id(0);
    local_tid_25133 = get_local_id(0);
    group_sizze_25136 = get_local_size(0);
    wave_sizze_25135 = LOCKSTEP_WIDTH;
    group_tid_25134 = get_group_id(0);
    
    int32_t phys_tid_21586;
    
    phys_tid_21586 = global_tid_25132;
    
    int32_t gtid_21585;
    
    gtid_21585 = sext_i64_i32(sext_i32_i64(group_tid_25134) *
        sext_i32_i64(segmap_group_sizze_21681) + sext_i32_i64(local_tid_25133));
    if (slt32(gtid_21585, m_18149)) {
        int32_t res_21687 = ((__global
                              int32_t *) mem_24159)[sext_i32_i64(gtid_21585)];
        float res_21688 = ((__global
                            float *) mem_24163)[sext_i32_i64(gtid_21585)];
        int32_t r32_arg_21689 = sub32(res_21687, k2p2_18163);
        float res_21690 = sitofp_i32_f32(r32_arg_21689);
        float sqrt_arg_21691 = res_21688 / res_21690;
        float res_21692;
        
        res_21692 = futrts_sqrt32(sqrt_arg_21691);
        
        float res_21693 = sitofp_i32_f32(res_21687);
        float t32_arg_21694 = hfrac_18155 * res_21693;
        int32_t res_21695 = fptosi_f32_i32(t32_arg_21694);
        
        ((__global int32_t *) mem_24167)[sext_i32_i64(gtid_21585)] = res_21695;
        ((__global float *) mem_24170)[sext_i32_i64(gtid_21585)] = res_21692;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_21681
}
__kernel void mainzisegmap_21716(__global int *global_failure, int32_t N_18148,
                                 int32_t m_18149, int32_t res_18475, __global
                                 unsigned char *res_mem_24122, __global
                                 unsigned char *res_mem_24174, __global
                                 unsigned char *res_mem_24175, __global
                                 unsigned char *mem_24183)
{
    #define segmap_group_sizze_21740 (mainzisegmap_group_sizze_21719)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25171;
    int32_t local_tid_25172;
    int32_t group_sizze_25175;
    int32_t wave_sizze_25174;
    int32_t group_tid_25173;
    
    global_tid_25171 = get_global_id(0);
    local_tid_25172 = get_local_id(0);
    group_sizze_25175 = get_local_size(0);
    wave_sizze_25174 = LOCKSTEP_WIDTH;
    group_tid_25173 = get_group_id(0);
    
    int32_t phys_tid_21716;
    
    phys_tid_21716 = global_tid_25171;
    
    int32_t gtid_21715;
    
    gtid_21715 = sext_i64_i32(sext_i32_i64(group_tid_25173) *
        sext_i32_i64(segmap_group_sizze_21740) + sext_i32_i64(local_tid_25172));
    if (slt32(gtid_21715, m_18149)) {
        int32_t x_21746 = ((__global
                            int32_t *) res_mem_24175)[sext_i32_i64(gtid_21715)];
        int32_t x_21747 = ((__global
                            int32_t *) res_mem_24174)[sext_i32_i64(gtid_21715)];
        float res_21748;
        float redout_22752 = 0.0F;
        
        for (int32_t i_22753 = 0; i_22753 < res_18475; i_22753++) {
            bool cond_21753 = slt32(i_22753, x_21747);
            float res_21754;
            
            if (cond_21753) {
                int32_t x_21755 = add32(x_21746, i_22753);
                int32_t x_21756 = sub32(x_21755, x_21747);
                int32_t i_21757 = add32(1, x_21756);
                float res_21758 = ((__global
                                    float *) res_mem_24122)[sext_i32_i64(gtid_21715) *
                                                            sext_i32_i64(N_18148) +
                                                            sext_i32_i64(i_21757)];
                
                res_21754 = res_21758;
            } else {
                res_21754 = 0.0F;
            }
            
            float res_21751 = res_21754 + redout_22752;
            float redout_tmp_25176 = res_21751;
            
            redout_22752 = redout_tmp_25176;
        }
        res_21748 = redout_22752;
        ((__global float *) mem_24183)[sext_i32_i64(gtid_21715)] = res_21748;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_21740
}
__kernel void mainzisegmap_21847(__global int *global_failure, float lam_18156,
                                 int32_t iota_arg_18497, int32_t x_18502,
                                 float res_18505, __global
                                 unsigned char *mappingindices_mem_23522,
                                 __global unsigned char *mem_24192)
{
    #define segmap_group_sizze_21868 (mainzisegmap_group_sizze_21850)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25237;
    int32_t local_tid_25238;
    int32_t group_sizze_25241;
    int32_t wave_sizze_25240;
    int32_t group_tid_25239;
    
    global_tid_25237 = get_global_id(0);
    local_tid_25238 = get_local_id(0);
    group_sizze_25241 = get_local_size(0);
    wave_sizze_25240 = LOCKSTEP_WIDTH;
    group_tid_25239 = get_group_id(0);
    
    int32_t phys_tid_21847;
    
    phys_tid_21847 = global_tid_25237;
    
    int32_t gtid_21846;
    
    gtid_21846 = sext_i64_i32(sext_i32_i64(group_tid_25239) *
        sext_i32_i64(segmap_group_sizze_21868) + sext_i32_i64(local_tid_25238));
    if (slt32(gtid_21846, iota_arg_18497)) {
        int32_t t_21874 = add32(x_18502, gtid_21846);
        int32_t i_21875 = sub32(t_21874, 1);
        int32_t time_21876 = ((__global
                               int32_t *) mappingindices_mem_23522)[sext_i32_i64(i_21875)];
        float res_21877 = sitofp_i32_f32(time_21876);
        float logplus_arg_21878 = res_21877 / res_18505;
        bool cond_21879 = 2.7182817F < logplus_arg_21878;
        float res_21880;
        
        if (cond_21879) {
            float res_21881;
            
            res_21881 = futrts_log32(logplus_arg_21878);
            res_21880 = res_21881;
        } else {
            res_21880 = 1.0F;
        }
        
        float res_21882;
        
        res_21882 = futrts_sqrt32(res_21880);
        
        float res_21883 = lam_18156 * res_21882;
        
        ((__global float *) mem_24192)[sext_i32_i64(gtid_21846)] = res_21883;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_21868
}
__kernel void mainzisegmap_22242(__global int *global_failure, int32_t N_18148,
                                 int32_t m_18149, int32_t n_18153,
                                 int32_t iota_arg_18497, __global
                                 unsigned char *res_mem_24123, __global
                                 unsigned char *res_mem_24175, __global
                                 unsigned char *mem_24197, __global
                                 unsigned char *mem_24233, __global
                                 unsigned char *mem_24253)
{
    #define segmap_group_sizze_22678 (mainzisegmap_group_sizze_22247)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25438;
    int32_t local_tid_25439;
    int32_t group_sizze_25442;
    int32_t wave_sizze_25441;
    int32_t group_tid_25440;
    
    global_tid_25438 = get_global_id(0);
    local_tid_25439 = get_local_id(0);
    group_sizze_25442 = get_local_size(0);
    wave_sizze_25441 = LOCKSTEP_WIDTH;
    group_tid_25440 = get_group_id(0);
    
    int32_t phys_tid_22242;
    
    phys_tid_22242 = global_tid_25438;
    
    int32_t gtid_22240;
    
    gtid_22240 = sext_i64_i32(squot64(sext_i32_i64(group_tid_25440) *
                                      sext_i32_i64(segmap_group_sizze_22678) +
                                      sext_i32_i64(local_tid_25439),
                                      sext_i32_i64(iota_arg_18497)));
    
    int32_t gtid_22241;
    
    gtid_22241 = sext_i64_i32(sext_i32_i64(group_tid_25440) *
        sext_i32_i64(segmap_group_sizze_22678) + sext_i32_i64(local_tid_25439) -
        squot64(sext_i32_i64(group_tid_25440) *
                sext_i32_i64(segmap_group_sizze_22678) +
                sext_i32_i64(local_tid_25439), sext_i32_i64(iota_arg_18497)) *
        sext_i32_i64(iota_arg_18497));
    if (slt32(gtid_22240, m_18149) && slt32(gtid_22241, iota_arg_18497)) {
        int32_t y_22685 = ((__global
                            int32_t *) mem_24233)[sext_i32_i64(gtid_22240)];
        float write_value_22689 = ((__global
                                    float *) mem_24253)[sext_i32_i64(gtid_22240) *
                                                        sext_i32_i64(iota_arg_18497) +
                                                        sext_i32_i64(gtid_22241)];
        bool cond_22690 = slt32(gtid_22241, y_22685);
        int32_t res_22691;
        
        if (cond_22690) {
            int32_t x_22683 = ((__global
                                int32_t *) res_mem_24175)[sext_i32_i64(gtid_22240)];
            int32_t i_22692 = add32(gtid_22241, x_22683);
            int32_t x_22693 = ((__global
                                int32_t *) res_mem_24123)[sext_i32_i64(gtid_22240) *
                                                          sext_i32_i64(N_18148) +
                                                          sext_i32_i64(i_22692)];
            int32_t res_22694 = sub32(x_22693, n_18153);
            
            res_22691 = res_22694;
        } else {
            res_22691 = -1;
        }
        if ((sle32(0, gtid_22240) && slt32(gtid_22240, m_18149)) && (sle32(0,
                                                                           res_22691) &&
                                                                     slt32(res_22691,
                                                                           iota_arg_18497))) {
            ((__global float *) mem_24197)[sext_i32_i64(gtid_22240) *
                                           sext_i32_i64(iota_arg_18497) +
                                           sext_i32_i64(res_22691)] =
                write_value_22689;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_22678
}
__kernel void mainzisegmap_22308(__global int *global_failure, int32_t N_18148,
                                 int32_t m_18149, int32_t n_18153, __global
                                 unsigned char *res_mem_24123, __global
                                 unsigned char *res_mem_24175, __global
                                 unsigned char *mem_24233, __global
                                 unsigned char *mem_24259, __global
                                 unsigned char *mem_24262, __global
                                 unsigned char *mem_24266)
{
    #define segmap_group_sizze_22629 (mainzisegmap_group_sizze_22311)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25433;
    int32_t local_tid_25434;
    int32_t group_sizze_25437;
    int32_t wave_sizze_25436;
    int32_t group_tid_25435;
    
    global_tid_25433 = get_global_id(0);
    local_tid_25434 = get_local_id(0);
    group_sizze_25437 = get_local_size(0);
    wave_sizze_25436 = LOCKSTEP_WIDTH;
    group_tid_25435 = get_group_id(0);
    
    int32_t phys_tid_22308;
    
    phys_tid_22308 = global_tid_25433;
    
    int32_t gtid_22307;
    
    gtid_22307 = sext_i64_i32(sext_i32_i64(group_tid_25435) *
        sext_i32_i64(segmap_group_sizze_22629) + sext_i32_i64(local_tid_25434));
    if (slt32(gtid_22307, m_18149)) {
        int32_t x_22634 = ((__global
                            int32_t *) res_mem_24175)[sext_i32_i64(gtid_22307)];
        int32_t y_22636 = ((__global
                            int32_t *) mem_24233)[sext_i32_i64(gtid_22307)];
        bool res_22637 = ((__global
                           bool *) mem_24259)[sext_i32_i64(gtid_22307)];
        bool cond_22639 = !res_22637;
        int32_t fst_breakzq_22640;
        
        if (cond_22639) {
            fst_breakzq_22640 = -1;
        } else {
            int32_t res_22638 = ((__global
                                  int32_t *) mem_24262)[sext_i32_i64(gtid_22307)];
            bool cond_22641 = slt32(res_22638, y_22636);
            int32_t res_22642;
            
            if (cond_22641) {
                int32_t i_22643 = add32(x_22634, res_22638);
                int32_t x_22644 = ((__global
                                    int32_t *) res_mem_24123)[sext_i32_i64(gtid_22307) *
                                                              sext_i32_i64(N_18148) +
                                                              sext_i32_i64(i_22643)];
                int32_t res_22645 = sub32(x_22644, n_18153);
                
                res_22642 = res_22645;
            } else {
                res_22642 = -1;
            }
            fst_breakzq_22640 = res_22642;
        }
        
        bool cond_22646 = sle32(x_22634, 5);
        bool res_22647 = sle32(y_22636, 5);
        bool x_22648 = !cond_22646;
        bool y_22649 = res_22647 && x_22648;
        bool cond_22650 = cond_22646 || y_22649;
        int32_t fst_breakzq_22651;
        
        if (cond_22650) {
            fst_breakzq_22651 = -2;
        } else {
            fst_breakzq_22651 = fst_breakzq_22640;
        }
        ((__global int32_t *) mem_24266)[sext_i32_i64(gtid_22307)] =
            fst_breakzq_22651;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_22629
}
__kernel void mainzisegmap_22340(__global int *global_failure, int32_t m_18149,
                                 __global unsigned char *mem_24242, __global
                                 unsigned char *mem_24245, __global
                                 unsigned char *mem_24259, __global
                                 unsigned char *mem_24262)
{
    #define segmap_group_sizze_22603 (mainzisegmap_group_sizze_22343)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25428;
    int32_t local_tid_25429;
    int32_t group_sizze_25432;
    int32_t wave_sizze_25431;
    int32_t group_tid_25430;
    
    global_tid_25428 = get_global_id(0);
    local_tid_25429 = get_local_id(0);
    group_sizze_25432 = get_local_size(0);
    wave_sizze_25431 = LOCKSTEP_WIDTH;
    group_tid_25430 = get_group_id(0);
    
    int32_t phys_tid_22340;
    
    phys_tid_22340 = global_tid_25428;
    
    int32_t gtid_22339;
    
    gtid_22339 = sext_i64_i32(sext_i32_i64(group_tid_25430) *
        sext_i32_i64(segmap_group_sizze_22603) + sext_i32_i64(local_tid_25429));
    if (slt32(gtid_22339, m_18149)) {
        bool acc0_22611 = ((__global
                            bool *) mem_24242)[sext_i32_i64(gtid_22339)];
        bool x_22616 = acc0_22611 && acc0_22611;
        int32_t res_22620;
        
        if (acc0_22611) {
            int32_t acc0_22612 = ((__global
                                   int32_t *) mem_24245)[sext_i32_i64(gtid_22339)];
            
            res_22620 = acc0_22612;
        } else {
            res_22620 = -1;
        }
        ((__global bool *) mem_24259)[sext_i32_i64(gtid_22339)] = x_22616;
        ((__global int32_t *) mem_24262)[sext_i32_i64(gtid_22339)] = res_22620;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_22603
}
__kernel void mainzisegmap_22471(__global int *global_failure, int32_t m_18149,
                                 int32_t num_groups_22496, __global
                                 unsigned char *res_mem_24121, __global
                                 unsigned char *res_mem_24175, __global
                                 unsigned char *res_mem_24176, __global
                                 unsigned char *mem_24230, __global
                                 unsigned char *mem_24233)
{
    #define segmap_group_sizze_22495 (mainzisegmap_group_sizze_22474)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25269;
    int32_t local_tid_25270;
    int32_t group_sizze_25273;
    int32_t wave_sizze_25272;
    int32_t group_tid_25271;
    
    global_tid_25269 = get_global_id(0);
    local_tid_25270 = get_local_id(0);
    group_sizze_25273 = get_local_size(0);
    wave_sizze_25272 = LOCKSTEP_WIDTH;
    group_tid_25271 = get_group_id(0);
    
    int32_t phys_tid_22471;
    
    phys_tid_22471 = global_tid_25269;
    
    int32_t phys_group_id_25274;
    
    phys_group_id_25274 = get_group_id(0);
    for (int32_t i_25275 = 0; i_25275 < sdiv_up32(sdiv_up32(m_18149,
                                                            segmap_group_sizze_22495) -
                                                  phys_group_id_25274,
                                                  num_groups_22496);
         i_25275++) {
        int32_t virt_group_id_25276 = phys_group_id_25274 + i_25275 *
                num_groups_22496;
        int32_t gtid_22470 = sext_i64_i32(sext_i32_i64(virt_group_id_25276) *
                sext_i32_i64(segmap_group_sizze_22495) +
                sext_i32_i64(local_tid_25270));
        
        if (slt32(gtid_22470, m_18149)) {
            int32_t x_22502 = ((__global
                                int32_t *) res_mem_24121)[sext_i32_i64(gtid_22470)];
            int32_t x_22503 = ((__global
                                int32_t *) res_mem_24175)[sext_i32_i64(gtid_22470)];
            float x_22504 = ((__global
                              float *) res_mem_24176)[sext_i32_i64(gtid_22470)];
            int32_t y_22505 = sub32(x_22502, x_22503);
            float res_22506 = sitofp_i32_f32(x_22503);
            float res_22507;
            
            res_22507 = futrts_sqrt32(res_22506);
            
            float y_22508 = x_22504 * res_22507;
            
            ((__global float *) mem_24230)[sext_i32_i64(gtid_22470)] = y_22508;
            ((__global int32_t *) mem_24233)[sext_i32_i64(gtid_22470)] =
                y_22505;
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_22495
}
__kernel void mainzisegmap_intragroup_19722(__global int *global_failure,
                                            int failure_is_an_option, __global
                                            int *global_failure_args,
                                            __local volatile
                                            int64_t *mem_23696_backing_aligned_0,
                                            __local volatile
                                            int64_t *mem_23684_backing_aligned_1,
                                            __local volatile
                                            int64_t *mem_23673_backing_aligned_2,
                                            int32_t k2p2zq_18165,
                                            int32_t m_18287, int32_t nm_18288,
                                            int32_t computed_group_sizze_19668,
                                            __global
                                            unsigned char *res_mem_23668,
                                            __global unsigned char *mem_23703)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_23696_backing_2 = (__local volatile
                                                           char *) mem_23696_backing_aligned_0;
    __local volatile char *restrict mem_23684_backing_1 = (__local volatile
                                                           char *) mem_23684_backing_aligned_1;
    __local volatile char *restrict mem_23673_backing_0 = (__local volatile
                                                           char *) mem_23673_backing_aligned_2;
    volatile __local bool local_failure;
    
    if (failure_is_an_option) {
        int failed = *global_failure >= 0;
        
        if (failed)
            return;
    }
    local_failure = false;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t global_tid_24572;
    int32_t local_tid_24573;
    int32_t group_sizze_24576;
    int32_t wave_sizze_24575;
    int32_t group_tid_24574;
    
    global_tid_24572 = get_global_id(0);
    local_tid_24573 = get_local_id(0);
    group_sizze_24576 = get_local_size(0);
    wave_sizze_24575 = LOCKSTEP_WIDTH;
    group_tid_24574 = get_group_id(0);
    
    int32_t phys_tid_19722;
    
    phys_tid_19722 = group_tid_24574;
    
    int32_t ltid_pre_24577;
    
    ltid_pre_24577 = squot32(local_tid_24573, k2p2zq_18165);
    
    int32_t ltid_pre_24578;
    
    ltid_pre_24578 = local_tid_24573 - squot32(local_tid_24573, k2p2zq_18165) *
        k2p2zq_18165;
    
    int32_t ltid_pre_24579;
    
    ltid_pre_24579 = local_tid_24573;
    
    int32_t gtid_19666;
    
    gtid_19666 = group_tid_24574;
    
    __local char *mem_23673;
    
    mem_23673 = (__local char *) mem_23673_backing_0;
    
    int32_t gtid_19669 = ltid_pre_24579;
    int32_t phys_tid_19670 = local_tid_24573;
    
    if (slt32(gtid_19669, nm_18288)) {
        int32_t res_19847 = sdiv32(gtid_19669, m_18287);
        int32_t res_19848 = smod32(gtid_19669, m_18287);
        bool cond_19849 = slt32(res_19848, k2p2zq_18165);
        float res_19850;
        
        if (cond_19849) {
            float res_19851 = ((__global
                                float *) res_mem_23668)[sext_i32_i64(gtid_19666) *
                                                        sext_i32_i64(k2p2zq_18165 *
                                                        k2p2zq_18165) +
                                                        sext_i32_i64(res_19847) *
                                                        sext_i32_i64(k2p2zq_18165) +
                                                        sext_i32_i64(res_19848)];
            
            res_19850 = res_19851;
        } else {
            int32_t y_19852 = add32(k2p2zq_18165, res_19847);
            bool cond_19853 = res_19848 == y_19852;
            float res_19854;
            
            if (cond_19853) {
                res_19854 = 1.0F;
            } else {
                res_19854 = 0.0F;
            }
            res_19850 = res_19854;
        }
        ((__local float *) mem_23673)[sext_i32_i64(gtid_19669)] = res_19850;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_23684;
    
    mem_23684 = (__local char *) mem_23684_backing_1;
    for (int32_t i_19856 = 0; i_19856 < k2p2zq_18165; i_19856++) {
        bool y_19858 = slt32(i_19856, nm_18288);
        bool index_certs_19859;
        
        if (!y_19858) {
            {
                if (atomic_cmpxchg_i32_global(global_failure, -1, 0) == -1) {
                    global_failure_args[0] = i_19856;
                    global_failure_args[1] = nm_18288;
                    ;
                }
                local_failure = true;
                goto error_1;
            }
        }
        
        float v1_19860 = ((__local float *) mem_23673)[sext_i32_i64(i_19856)];
        bool cond_19861 = v1_19860 == 0.0F;
        int32_t gtid_19680 = ltid_pre_24579;
        int32_t phys_tid_19681 = local_tid_24573;
        
        if (slt32(gtid_19680, nm_18288)) {
            int32_t res_19864 = sdiv32(gtid_19680, m_18287);
            int32_t res_19865 = smod32(gtid_19680, m_18287);
            float res_19866;
            
            if (cond_19861) {
                int32_t x_19867 = mul32(m_18287, res_19864);
                int32_t i_19868 = add32(res_19865, x_19867);
                float res_19869 = ((__local
                                    float *) mem_23673)[sext_i32_i64(i_19868)];
                
                res_19866 = res_19869;
            } else {
                float x_19870 = ((__local
                                  float *) mem_23673)[sext_i32_i64(res_19865)];
                float x_19871 = x_19870 / v1_19860;
                int32_t y_19872 = sub32(k2p2zq_18165, 1);
                bool cond_19873 = slt32(res_19864, y_19872);
                float res_19874;
                
                if (cond_19873) {
                    int32_t x_19875 = add32(1, res_19864);
                    int32_t x_19876 = mul32(m_18287, x_19875);
                    int32_t i_19877 = add32(res_19865, x_19876);
                    float x_19878 = ((__local
                                      float *) mem_23673)[sext_i32_i64(i_19877)];
                    int32_t i_19879 = add32(i_19856, x_19876);
                    float x_19880 = ((__local
                                      float *) mem_23673)[sext_i32_i64(i_19879)];
                    float y_19881 = x_19871 * x_19880;
                    float res_19882 = x_19878 - y_19881;
                    
                    res_19874 = res_19882;
                } else {
                    res_19874 = x_19871;
                }
                res_19866 = res_19874;
            }
            ((__local float *) mem_23684)[sext_i32_i64(gtid_19680)] = res_19866;
        }
        
      error_1:
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_failure)
            return;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t write_i_19702 = ltid_pre_24579;
        int32_t phys_tid_19703 = local_tid_24573;
        
        if (slt32(write_i_19702, nm_18288)) {
            float write_value_19885 = ((__local
                                        float *) mem_23684)[sext_i32_i64(write_i_19702)];
            
            if (sle32(0, write_i_19702) && slt32(write_i_19702, nm_18288)) {
                ((__local float *) mem_23673)[sext_i32_i64(write_i_19702)] =
                    write_value_19885;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    __local char *mem_23696;
    
    mem_23696 = (__local char *) mem_23696_backing_2;
    
    int32_t gtid_19705 = ltid_pre_24577;
    int32_t gtid_19706 = ltid_pre_24578;
    int32_t phys_tid_19707 = local_tid_24573;
    
    if (slt32(gtid_19705, k2p2zq_18165) && slt32(gtid_19706, k2p2zq_18165)) {
        int32_t index_primexp_22725 = m_18287 * gtid_19705;
        int32_t i_19892 = add32(k2p2zq_18165, gtid_19706);
        int32_t new_index_19893 = i_19892 + index_primexp_22725;
        float res_19894 = ((__local
                            float *) mem_23673)[sext_i32_i64(new_index_19893)];
        
        ((__local float *) mem_23696)[sext_i32_i64(gtid_19705) *
                                      sext_i32_i64(k2p2zq_18165) +
                                      sext_i32_i64(gtid_19706)] = res_19894;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int32_t i_24581 = 0; i_24581 < sdiv_up32(k2p2zq_18165 * k2p2zq_18165 -
                                                  local_tid_24573,
                                                  computed_group_sizze_19668);
         i_24581++) {
        ((__global float *) mem_23703)[sext_i32_i64(gtid_19666) *
                                       sext_i32_i64(k2p2zq_18165 *
                                       k2p2zq_18165) +
                                       sext_i32_i64(squot32(i_24581 *
                                                            computed_group_sizze_19668 +
                                                            local_tid_24573,
                                                            k2p2zq_18165)) *
                                       sext_i32_i64(k2p2zq_18165) +
                                       sext_i32_i64(i_24581 *
                                       computed_group_sizze_19668 +
                                       local_tid_24573 - squot32(i_24581 *
                                                                 computed_group_sizze_19668 +
                                                                 local_tid_24573,
                                                                 k2p2zq_18165) *
                                       k2p2zq_18165)] = ((__local
                                                          float *) mem_23696)[sext_i32_i64(squot32(i_24581 *
                                                                                                   computed_group_sizze_19668 +
                                                                                                   local_tid_24573,
                                                                                                   k2p2zq_18165)) *
                                                                              sext_i32_i64(k2p2zq_18165) +
                                                                              sext_i32_i64(i_24581 *
                                                                              computed_group_sizze_19668 +
                                                                              local_tid_24573 -
                                                                              squot32(i_24581 *
                                                                                      computed_group_sizze_19668 +
                                                                                      local_tid_24573,
                                                                                      k2p2zq_18165) *
                                                                              k2p2zq_18165)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
  error_4:
    return;
}
__kernel void mainzisegmap_intragroup_20074(__global int *global_failure,
                                            __local volatile
                                            int64_t *mem_23729_backing_aligned_0,
                                            int32_t m_18149,
                                            int32_t k2p2zq_18165,
                                            int32_t m_18287, int32_t nm_18288,
                                            int32_t i_20482,
                                            int32_t ctx_param_ext_23712,
                                            int32_t ctx_param_ext_23713,
                                            int32_t ctx_param_ext_23715,
                                            __global
                                            unsigned char *mem_param_23717,
                                            __global unsigned char *mem_23724,
                                            __global unsigned char *mem_23735)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_23729_backing_0 = (__local volatile
                                                           char *) mem_23729_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24608;
    int32_t local_tid_24609;
    int32_t group_sizze_24612;
    int32_t wave_sizze_24611;
    int32_t group_tid_24610;
    
    global_tid_24608 = get_global_id(0);
    local_tid_24609 = get_local_id(0);
    group_sizze_24612 = get_local_size(0);
    wave_sizze_24611 = LOCKSTEP_WIDTH;
    group_tid_24610 = get_group_id(0);
    
    int32_t phys_tid_20074;
    
    phys_tid_20074 = group_tid_24610;
    
    int32_t ltid_pre_24613;
    
    ltid_pre_24613 = local_tid_24609;
    
    int32_t gtid_20047;
    
    gtid_20047 = group_tid_24610;
    
    float v1_20498 = ((__global
                       float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                                 (sext_i32_i64(gtid_20047) *
                                                  sext_i32_i64(ctx_param_ext_23713) +
                                                  sext_i32_i64(i_20482) *
                                                  sext_i32_i64(ctx_param_ext_23715))];
    bool cond_20499 = v1_20498 == 0.0F;
    __local char *mem_23729;
    
    mem_23729 = (__local char *) mem_23729_backing_0;
    
    int32_t gtid_20050 = ltid_pre_24613;
    int32_t phys_tid_20051 = local_tid_24609;
    
    if (slt32(gtid_20050, nm_18288)) {
        int32_t res_20502 = sdiv32(gtid_20050, m_18287);
        int32_t res_20503 = smod32(gtid_20050, m_18287);
        float res_20504;
        
        if (cond_20499) {
            int32_t x_20505 = mul32(m_18287, res_20502);
            int32_t i_20506 = add32(res_20503, x_20505);
            float res_20507 = ((__global
                                float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                                          (sext_i32_i64(gtid_20047) *
                                                           sext_i32_i64(ctx_param_ext_23713) +
                                                           sext_i32_i64(i_20506) *
                                                           sext_i32_i64(ctx_param_ext_23715))];
            
            res_20504 = res_20507;
        } else {
            float x_20508 = ((__global
                              float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                                        (sext_i32_i64(gtid_20047) *
                                                         sext_i32_i64(ctx_param_ext_23713) +
                                                         sext_i32_i64(res_20503) *
                                                         sext_i32_i64(ctx_param_ext_23715))];
            float x_20509 = x_20508 / v1_20498;
            int32_t y_20510 = sub32(k2p2zq_18165, 1);
            bool cond_20511 = slt32(res_20502, y_20510);
            float res_20512;
            
            if (cond_20511) {
                int32_t x_20513 = add32(1, res_20502);
                int32_t x_20514 = mul32(m_18287, x_20513);
                int32_t i_20515 = add32(res_20503, x_20514);
                float x_20516 = ((__global
                                  float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                                            (sext_i32_i64(gtid_20047) *
                                                             sext_i32_i64(ctx_param_ext_23713) +
                                                             sext_i32_i64(i_20515) *
                                                             sext_i32_i64(ctx_param_ext_23715))];
                int32_t i_20517 = add32(i_20482, x_20514);
                float x_20518 = ((__global
                                  float *) mem_param_23717)[sext_i32_i64(ctx_param_ext_23712) +
                                                            (sext_i32_i64(gtid_20047) *
                                                             sext_i32_i64(ctx_param_ext_23713) +
                                                             sext_i32_i64(i_20517) *
                                                             sext_i32_i64(ctx_param_ext_23715))];
                float y_20519 = x_20509 * x_20518;
                float res_20520 = x_20516 - y_20519;
                
                res_20512 = res_20520;
            } else {
                res_20512 = x_20509;
            }
            res_20504 = res_20512;
        }
        ((__local float *) mem_23729)[sext_i32_i64(gtid_20050)] = res_20504;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t write_i_20072 = ltid_pre_24613;
    int32_t phys_tid_20073 = local_tid_24609;
    
    if (slt32(write_i_20072, nm_18288)) {
        float write_value_20523 = ((__local
                                    float *) mem_23729)[sext_i32_i64(write_i_20072)];
        
        if (sle32(0, write_i_20072) && slt32(write_i_20072, nm_18288)) {
            ((__global float *) mem_23724)[sext_i32_i64(gtid_20047) +
                                           sext_i32_i64(write_i_20072) *
                                           sext_i32_i64(m_18149)] =
                write_value_20523;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_tid_24609 == 0) {
        for (int32_t i_24614 = 0; i_24614 < nm_18288; i_24614++) {
            ((__global float *) mem_23735)[sext_i32_i64(gtid_20047) *
                                           sext_i32_i64(nm_18288) +
                                           sext_i32_i64(i_24614)] = ((__global
                                                                      float *) mem_23724)[sext_i32_i64(gtid_20047) +
                                                                                          sext_i32_i64(i_24614) *
                                                                                          sext_i32_i64(m_18149)];
        }
    }
    
  error_2:
    return;
}
__kernel void mainzisegmap_intragroup_21114(__global int *global_failure,
                                            __local volatile
                                            int64_t *mem_24091_backing_aligned_0,
                                            __local volatile
                                            int64_t *mem_24088_backing_aligned_1,
                                            __local volatile
                                            int64_t *mem_24085_backing_aligned_2,
                                            __local volatile
                                            int64_t *mem_24082_backing_aligned_3,
                                            int32_t N_18148, int32_t N_18150,
                                            int32_t i_18392, __global
                                            unsigned char *images_mem_23523,
                                            __global
                                            unsigned char *res_mem_24067,
                                            __global unsigned char *mem_24095,
                                            __global unsigned char *mem_24100,
                                            __global unsigned char *mem_24105)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_24091_backing_3 = (__local volatile
                                                           char *) mem_24091_backing_aligned_0;
    __local volatile char *restrict mem_24088_backing_2 = (__local volatile
                                                           char *) mem_24088_backing_aligned_1;
    __local volatile char *restrict mem_24085_backing_1 = (__local volatile
                                                           char *) mem_24085_backing_aligned_2;
    __local volatile char *restrict mem_24082_backing_0 = (__local volatile
                                                           char *) mem_24082_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24910;
    int32_t local_tid_24911;
    int32_t group_sizze_24914;
    int32_t wave_sizze_24913;
    int32_t group_tid_24912;
    
    global_tid_24910 = get_global_id(0);
    local_tid_24911 = get_local_id(0);
    group_sizze_24914 = get_local_size(0);
    wave_sizze_24913 = LOCKSTEP_WIDTH;
    group_tid_24912 = get_group_id(0);
    
    int32_t phys_tid_21114;
    
    phys_tid_21114 = group_tid_24912;
    
    int32_t ltid_pre_24915;
    
    ltid_pre_24915 = local_tid_24911;
    
    int32_t gtid_21107;
    
    gtid_21107 = group_tid_24912;
    
    __local char *mem_24082;
    
    mem_24082 = (__local char *) mem_24082_backing_0;
    
    __local char *mem_24085;
    
    mem_24085 = (__local char *) mem_24085_backing_1;
    
    int32_t gtid_21110 = ltid_pre_24915;
    int32_t phys_tid_21111 = local_tid_24911;
    
    if (slt32(gtid_21110, N_18148)) {
        float x_21203 = ((__global
                          float *) images_mem_23523)[sext_i32_i64(gtid_21107) *
                                                     sext_i32_i64(N_18150) +
                                                     sext_i32_i64(gtid_21110)];
        bool res_21205;
        
        res_21205 = futrts_isnan32(x_21203);
        
        bool cond_21206 = !res_21205;
        float res_21207;
        
        if (cond_21206) {
            float x_21204 = ((__global
                              float *) res_mem_24067)[sext_i32_i64(gtid_21107) *
                                                      sext_i32_i64(N_18148) +
                                                      sext_i32_i64(gtid_21110)];
            float res_21208 = x_21203 - x_21204;
            
            res_21207 = res_21208;
        } else {
            res_21207 = NAN;
        }
        
        bool res_21209;
        
        res_21209 = futrts_isnan32(res_21207);
        
        bool res_21210 = !res_21209;
        int32_t res_21211 = btoi_bool_i32(res_21210);
        
        ((__local int32_t *) mem_24082)[sext_i32_i64(gtid_21110)] = res_21211;
        ((__local float *) mem_24085)[sext_i32_i64(gtid_21110)] = res_21207;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_24916;
    
    dims_flat_24916 = N_18148;
    
    int32_t x_21200;
    int32_t x_21201;
    int32_t x_24918;
    int32_t x_24919;
    int32_t skip_threads_24921;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_24911, N_18148)) {
            x_21201 = ((volatile __local
                        int32_t *) mem_24082)[sext_i32_i64(local_tid_24911)];
            if ((local_tid_24911 - squot32(local_tid_24911, 32) * 32) == 0) {
                x_21200 = x_21201;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_24921 = 1;
        while (slt32(skip_threads_24921, 32)) {
            if (sle32(skip_threads_24921, local_tid_24911 -
                      squot32(local_tid_24911, 32) * 32) &&
                slt32(local_tid_24911, N_18148)) {
                // read operands
                {
                    x_21200 = ((volatile __local
                                int32_t *) mem_24082)[sext_i32_i64(local_tid_24911 -
                                                      skip_threads_24921)];
                }
                // perform operation
                {
                    bool inactive_24922 = slt32(srem32(local_tid_24911,
                                                       N_18148),
                                                local_tid_24911 -
                                                (local_tid_24911 -
                                                 skip_threads_24921));
                    
                    if (inactive_24922) {
                        x_21200 = x_21201;
                    }
                    if (!inactive_24922) {
                        int32_t res_21202 = add32(x_21200, x_21201);
                        
                        x_21200 = res_21202;
                    }
                }
            }
            if (sle32(wave_sizze_24913, skip_threads_24921)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_24921, local_tid_24911 -
                      squot32(local_tid_24911, 32) * 32) &&
                slt32(local_tid_24911, N_18148)) {
                // write result
                {
                    ((volatile __local
                      int32_t *) mem_24082)[sext_i32_i64(local_tid_24911)] =
                        x_21200;
                    x_21201 = x_21200;
                }
            }
            if (sle32(wave_sizze_24913, skip_threads_24921)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_24921 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_24911 - squot32(local_tid_24911, 32) * 32) == 31 &&
            slt32(local_tid_24911, N_18148)) {
            ((volatile __local
              int32_t *) mem_24082)[sext_i32_i64(squot32(local_tid_24911,
                                                         32))] = x_21200;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_24923;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_24911, 32) == 0 && slt32(local_tid_24911,
                                                           N_18148)) {
                x_24919 = ((volatile __local
                            int32_t *) mem_24082)[sext_i32_i64(local_tid_24911)];
                if ((local_tid_24911 - squot32(local_tid_24911, 32) * 32) ==
                    0) {
                    x_24918 = x_24919;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_24923 = 1;
            while (slt32(skip_threads_24923, 32)) {
                if (sle32(skip_threads_24923, local_tid_24911 -
                          squot32(local_tid_24911, 32) * 32) &&
                    (squot32(local_tid_24911, 32) == 0 && slt32(local_tid_24911,
                                                                N_18148))) {
                    // read operands
                    {
                        x_24918 = ((volatile __local
                                    int32_t *) mem_24082)[sext_i32_i64(local_tid_24911 -
                                                          skip_threads_24923)];
                    }
                    // perform operation
                    {
                        bool inactive_24924 = slt32(srem32(local_tid_24911 *
                                                           32 + 32 - 1,
                                                           N_18148),
                                                    local_tid_24911 * 32 + 32 -
                                                    1 - ((local_tid_24911 -
                                                          skip_threads_24923) *
                                                         32 + 32 - 1));
                        
                        if (inactive_24924) {
                            x_24918 = x_24919;
                        }
                        if (!inactive_24924) {
                            int32_t res_24920 = add32(x_24918, x_24919);
                            
                            x_24918 = res_24920;
                        }
                    }
                }
                if (sle32(wave_sizze_24913, skip_threads_24923)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_24923, local_tid_24911 -
                          squot32(local_tid_24911, 32) * 32) &&
                    (squot32(local_tid_24911, 32) == 0 && slt32(local_tid_24911,
                                                                N_18148))) {
                    // write result
                    {
                        ((volatile __local
                          int32_t *) mem_24082)[sext_i32_i64(local_tid_24911)] =
                            x_24918;
                        x_24919 = x_24918;
                    }
                }
                if (sle32(wave_sizze_24913, skip_threads_24923)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_24923 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_24911, 32) == 0 || !slt32(local_tid_24911,
                                                          N_18148))) {
            // read operands
            {
                x_21201 = x_21200;
                x_21200 = ((__local
                            int32_t *) mem_24082)[sext_i32_i64(squot32(local_tid_24911,
                                                                       32) -
                                                  1)];
            }
            // perform operation
            {
                bool inactive_24925 = slt32(srem32(local_tid_24911, N_18148),
                                            local_tid_24911 -
                                            (squot32(local_tid_24911, 32) * 32 -
                                             1));
                
                if (inactive_24925) {
                    x_21200 = x_21201;
                }
                if (!inactive_24925) {
                    int32_t res_21202 = add32(x_21200, x_21201);
                    
                    x_21200 = res_21202;
                }
            }
            // write final result
            {
                ((__local int32_t *) mem_24082)[sext_i32_i64(local_tid_24911)] =
                    x_21200;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_24911, 32) == 0) {
            ((__local int32_t *) mem_24082)[sext_i32_i64(local_tid_24911)] =
                x_21201;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t res_21212 = ((__local int32_t *) mem_24082)[sext_i32_i64(i_18392)];
    __local char *mem_24088;
    
    mem_24088 = (__local char *) mem_24088_backing_2;
    ((__local float *) mem_24088)[sext_i32_i64(local_tid_24911)] = NAN;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_24091;
    
    mem_24091 = (__local char *) mem_24091_backing_3;
    ((__local int32_t *) mem_24091)[sext_i32_i64(local_tid_24911)] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t write_i_21112 = ltid_pre_24915;
    int32_t phys_tid_21113 = local_tid_24911;
    
    if (slt32(write_i_21112, N_18148)) {
        float x_21217 = ((__local
                          float *) mem_24085)[sext_i32_i64(write_i_21112)];
        bool res_21220;
        
        res_21220 = futrts_isnan32(x_21217);
        
        bool res_21221 = !res_21220;
        int32_t res_21222;
        
        if (res_21221) {
            int32_t x_21218 = ((__local
                                int32_t *) mem_24082)[sext_i32_i64(write_i_21112)];
            int32_t res_21223 = sub32(x_21218, 1);
            
            res_21222 = res_21223;
        } else {
            res_21222 = -1;
        }
        if (sle32(0, res_21222) && slt32(res_21222, N_18148)) {
            ((__local int32_t *) mem_24091)[sext_i32_i64(res_21222)] =
                write_i_21112;
        }
        if (sle32(0, res_21222) && slt32(res_21222, N_18148)) {
            ((__local float *) mem_24088)[sext_i32_i64(res_21222)] = x_21217;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_tid_24911 == 0) {
        ((__global int32_t *) mem_24095)[sext_i32_i64(gtid_21107)] = res_21212;
    }
    ((__global float *) mem_24100)[sext_i32_i64(gtid_21107) *
                                   sext_i32_i64(N_18148) +
                                   sext_i32_i64(local_tid_24911)] = ((__local
                                                                      float *) mem_24088)[sext_i32_i64(local_tid_24911)];
    barrier(CLK_LOCAL_MEM_FENCE);
    ((__global int32_t *) mem_24105)[sext_i32_i64(gtid_21107) *
                                     sext_i32_i64(N_18148) +
                                     sext_i32_i64(local_tid_24911)] = ((__local
                                                                        int32_t *) mem_24091)[sext_i32_i64(local_tid_24911)];
    barrier(CLK_LOCAL_MEM_FENCE);
    
  error_2:
    return;
}
__kernel void mainzisegmap_intragroup_21479(__global int *global_failure,
                                            __local volatile
                                            int64_t *red_arr_mem_25010_backing_aligned_0,
                                            __local volatile
                                            int64_t *red_arr_mem_25006_backing_aligned_1,
                                            int32_t N_18148, int32_t N_18150,
                                            int32_t n_18153, float hfrac_18155,
                                            int32_t k2p2_18163, __global
                                            unsigned char *images_mem_23523,
                                            __global
                                            unsigned char *res_mem_24122,
                                            __global unsigned char *mem_24149,
                                            __global unsigned char *mem_24152,
                                            __global unsigned char *mem_24155)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_25010_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_25010_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_25006_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_25006_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25000;
    int32_t local_tid_25001;
    int32_t group_sizze_25004;
    int32_t wave_sizze_25003;
    int32_t group_tid_25002;
    
    global_tid_25000 = get_global_id(0);
    local_tid_25001 = get_local_id(0);
    group_sizze_25004 = get_local_size(0);
    wave_sizze_25003 = LOCKSTEP_WIDTH;
    group_tid_25002 = get_group_id(0);
    
    int32_t phys_tid_21479;
    
    phys_tid_21479 = group_tid_25002;
    
    int32_t ltid_pre_25005;
    
    ltid_pre_25005 = local_tid_25001;
    
    int32_t gtid_21472;
    
    gtid_21472 = group_tid_25002;
    
    int32_t res_21560;
    int32_t gtid_21475 = ltid_pre_25005;
    int32_t phys_tid_21476 = local_tid_25001;
    __local char *red_arr_mem_25006;
    
    red_arr_mem_25006 = (__local char *) red_arr_mem_25006_backing_0;
    if (slt32(gtid_21475, n_18153)) {
        float x_21564 = ((__global
                          float *) images_mem_23523)[sext_i32_i64(gtid_21472) *
                                                     sext_i32_i64(N_18150) +
                                                     sext_i32_i64(gtid_21475)];
        bool res_21565;
        
        res_21565 = futrts_isnan32(x_21564);
        
        bool cond_21566 = !res_21565;
        int32_t res_21567 = btoi_bool_i32(cond_21566);
        
        ((__local int32_t *) red_arr_mem_25006)[sext_i32_i64(gtid_21475)] =
            res_21567;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_25008;
    int32_t skip_waves_25009;
    int32_t x_21561;
    int32_t x_21562;
    
    offset_25008 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_25001, n_18153)) {
            x_21561 = ((__local
                        int32_t *) red_arr_mem_25006)[sext_i32_i64(local_tid_25001 +
                                                      offset_25008)];
        }
    }
    offset_25008 = 1;
    while (slt32(offset_25008, wave_sizze_25003)) {
        if (slt32(local_tid_25001 + offset_25008, n_18153) &&
            ((local_tid_25001 - squot32(local_tid_25001, wave_sizze_25003) *
              wave_sizze_25003) & (2 * offset_25008 - 1)) == 0) {
            // read array element
            {
                x_21562 = ((volatile __local
                            int32_t *) red_arr_mem_25006)[sext_i32_i64(local_tid_25001 +
                                                          offset_25008)];
            }
            // apply reduction operation
            {
                int32_t res_21563 = add32(x_21561, x_21562);
                
                x_21561 = res_21563;
            }
            // write result of operation
            {
                ((volatile __local
                  int32_t *) red_arr_mem_25006)[sext_i32_i64(local_tid_25001)] =
                    x_21561;
            }
        }
        offset_25008 *= 2;
    }
    skip_waves_25009 = 1;
    while (slt32(skip_waves_25009, squot32(n_18153 + wave_sizze_25003 - 1,
                                           wave_sizze_25003))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_25008 = skip_waves_25009 * wave_sizze_25003;
        if (slt32(local_tid_25001 + offset_25008, n_18153) &&
            ((local_tid_25001 - squot32(local_tid_25001, wave_sizze_25003) *
              wave_sizze_25003) == 0 && (squot32(local_tid_25001,
                                                 wave_sizze_25003) & (2 *
                                                                      skip_waves_25009 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_21562 = ((__local
                            int32_t *) red_arr_mem_25006)[sext_i32_i64(local_tid_25001 +
                                                          offset_25008)];
            }
            // apply reduction operation
            {
                int32_t res_21563 = add32(x_21561, x_21562);
                
                x_21561 = res_21563;
            }
            // write result of operation
            {
                ((__local
                  int32_t *) red_arr_mem_25006)[sext_i32_i64(local_tid_25001)] =
                    x_21561;
            }
        }
        skip_waves_25009 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_21560 = ((__local int32_t *) red_arr_mem_25006)[0];
    
    float res_21568;
    int32_t gtid_21477 = ltid_pre_25005;
    int32_t phys_tid_21478 = local_tid_25001;
    __local char *red_arr_mem_25010;
    
    red_arr_mem_25010 = (__local char *) red_arr_mem_25010_backing_1;
    if (slt32(gtid_21477, n_18153)) {
        bool cond_21574 = slt32(gtid_21477, res_21560);
        float res_21575;
        
        if (cond_21574) {
            float x_elem_21573 = ((__global
                                   float *) res_mem_24122)[sext_i32_i64(gtid_21472) *
                                                           sext_i32_i64(N_18148) +
                                                           sext_i32_i64(gtid_21477)];
            
            res_21575 = x_elem_21573;
        } else {
            res_21575 = 0.0F;
        }
        
        float res_21576 = res_21575 * res_21575;
        
        ((__local float *) red_arr_mem_25010)[sext_i32_i64(gtid_21477)] =
            res_21576;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_25012;
    int32_t skip_waves_25013;
    float x_21569;
    float x_21570;
    
    offset_25012 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_25001, n_18153)) {
            x_21569 = ((__local
                        float *) red_arr_mem_25010)[sext_i32_i64(local_tid_25001 +
                                                    offset_25012)];
        }
    }
    offset_25012 = 1;
    while (slt32(offset_25012, wave_sizze_25003)) {
        if (slt32(local_tid_25001 + offset_25012, n_18153) &&
            ((local_tid_25001 - squot32(local_tid_25001, wave_sizze_25003) *
              wave_sizze_25003) & (2 * offset_25012 - 1)) == 0) {
            // read array element
            {
                x_21570 = ((volatile __local
                            float *) red_arr_mem_25010)[sext_i32_i64(local_tid_25001 +
                                                        offset_25012)];
            }
            // apply reduction operation
            {
                float res_21571 = x_21569 + x_21570;
                
                x_21569 = res_21571;
            }
            // write result of operation
            {
                ((volatile __local
                  float *) red_arr_mem_25010)[sext_i32_i64(local_tid_25001)] =
                    x_21569;
            }
        }
        offset_25012 *= 2;
    }
    skip_waves_25013 = 1;
    while (slt32(skip_waves_25013, squot32(n_18153 + wave_sizze_25003 - 1,
                                           wave_sizze_25003))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_25012 = skip_waves_25013 * wave_sizze_25003;
        if (slt32(local_tid_25001 + offset_25012, n_18153) &&
            ((local_tid_25001 - squot32(local_tid_25001, wave_sizze_25003) *
              wave_sizze_25003) == 0 && (squot32(local_tid_25001,
                                                 wave_sizze_25003) & (2 *
                                                                      skip_waves_25013 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_21570 = ((__local
                            float *) red_arr_mem_25010)[sext_i32_i64(local_tid_25001 +
                                                        offset_25012)];
            }
            // apply reduction operation
            {
                float res_21571 = x_21569 + x_21570;
                
                x_21569 = res_21571;
            }
            // write result of operation
            {
                ((__local
                  float *) red_arr_mem_25010)[sext_i32_i64(local_tid_25001)] =
                    x_21569;
            }
        }
        skip_waves_25013 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res_21568 = ((__local float *) red_arr_mem_25010)[0];
    
    int32_t r32_arg_21577 = sub32(res_21560, k2p2_18163);
    float res_21578 = sitofp_i32_f32(r32_arg_21577);
    float sqrt_arg_21579 = res_21568 / res_21578;
    float res_21580;
    
    res_21580 = futrts_sqrt32(sqrt_arg_21579);
    
    float res_21581 = sitofp_i32_f32(res_21560);
    float t32_arg_21582 = hfrac_18155 * res_21581;
    int32_t res_21583 = fptosi_f32_i32(t32_arg_21582);
    
    if (local_tid_25001 == 0) {
        ((__global int32_t *) mem_24149)[sext_i32_i64(gtid_21472)] = res_21583;
    }
    if (local_tid_25001 == 0) {
        ((__global int32_t *) mem_24152)[sext_i32_i64(gtid_21472)] = res_21560;
    }
    if (local_tid_25001 == 0) {
        ((__global float *) mem_24155)[sext_i32_i64(gtid_21472)] = res_21580;
    }
    
  error_4:
    return;
}
__kernel void mainzisegmap_intragroup_21895(__global int *global_failure,
                                            __local volatile
                                            int64_t *mem_24209_backing_aligned_0,
                                            __local volatile
                                            int64_t *red_arr_mem_25265_backing_aligned_1,
                                            __local volatile
                                            int64_t *red_arr_mem_25263_backing_aligned_2,
                                            __local volatile
                                            int64_t *red_arr_mem_25261_backing_aligned_3,
                                            __local volatile
                                            int64_t *mem_24206_backing_aligned_4,
                                            __local volatile
                                            int64_t *mem_24202_backing_aligned_5,
                                            int32_t N_18148, int32_t n_18153,
                                            int32_t iota_arg_18497, __global
                                            unsigned char *res_mem_24121,
                                            __global
                                            unsigned char *res_mem_24122,
                                            __global
                                            unsigned char *res_mem_24123,
                                            __global
                                            unsigned char *res_mem_24174,
                                            __global
                                            unsigned char *res_mem_24175,
                                            __global
                                            unsigned char *res_mem_24176,
                                            __global
                                            unsigned char *res_mem_24188,
                                            __global unsigned char *mem_24192,
                                            __global unsigned char *mem_24215,
                                            __global unsigned char *mem_24220,
                                            __global unsigned char *mem_24223,
                                            __global unsigned char *mem_24226)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_24209_backing_5 = (__local volatile
                                                           char *) mem_24209_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_25265_backing_4 =
                          (__local volatile
                           char *) red_arr_mem_25265_backing_aligned_1;
    __local volatile char *restrict red_arr_mem_25263_backing_3 =
                          (__local volatile
                           char *) red_arr_mem_25263_backing_aligned_2;
    __local volatile char *restrict red_arr_mem_25261_backing_2 =
                          (__local volatile
                           char *) red_arr_mem_25261_backing_aligned_3;
    __local volatile char *restrict mem_24206_backing_1 = (__local volatile
                                                           char *) mem_24206_backing_aligned_4;
    __local volatile char *restrict mem_24202_backing_0 = (__local volatile
                                                           char *) mem_24202_backing_aligned_5;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25245;
    int32_t local_tid_25246;
    int32_t group_sizze_25249;
    int32_t wave_sizze_25248;
    int32_t group_tid_25247;
    
    global_tid_25245 = get_global_id(0);
    local_tid_25246 = get_local_id(0);
    group_sizze_25249 = get_local_size(0);
    wave_sizze_25248 = LOCKSTEP_WIDTH;
    group_tid_25247 = get_group_id(0);
    
    int32_t phys_tid_21895;
    
    phys_tid_21895 = group_tid_25247;
    
    int32_t ltid_pre_25250;
    
    ltid_pre_25250 = local_tid_25246;
    
    int32_t gtid_21886;
    
    gtid_21886 = group_tid_25247;
    
    int32_t x_22133;
    
    x_22133 = ((__global int32_t *) res_mem_24121)[sext_i32_i64(gtid_21886)];
    
    int32_t x_22134 = ((__global
                        int32_t *) res_mem_24175)[sext_i32_i64(gtid_21886)];
    float x_22135 = ((__global
                      float *) res_mem_24176)[sext_i32_i64(gtid_21886)];
    int32_t x_22136 = ((__global
                        int32_t *) res_mem_24174)[sext_i32_i64(gtid_21886)];
    float x_22137 = ((__global
                      float *) res_mem_24188)[sext_i32_i64(gtid_21886)];
    int32_t y_22140 = sub32(x_22133, x_22134);
    float res_22141 = sitofp_i32_f32(x_22134);
    float res_22142;
    
    res_22142 = futrts_sqrt32(res_22141);
    
    float y_22143 = x_22135 * res_22142;
    __local char *mem_24202;
    
    mem_24202 = (__local char *) mem_24202_backing_0;
    
    int32_t gtid_21889 = ltid_pre_25250;
    int32_t phys_tid_21890 = local_tid_25246;
    
    if (slt32(gtid_21889, iota_arg_18497)) {
        bool cond_22156 = sle32(y_22140, gtid_21889);
        float res_22157;
        
        if (cond_22156) {
            res_22157 = 0.0F;
        } else {
            bool cond_22158 = gtid_21889 == 0;
            float res_22159;
            
            if (cond_22158) {
                res_22159 = x_22137;
            } else {
                int32_t x_22160 = sub32(x_22134, x_22136);
                int32_t i_22161 = add32(gtid_21889, x_22160);
                float negate_arg_22162 = ((__global
                                           float *) res_mem_24122)[sext_i32_i64(gtid_21886) *
                                                                   sext_i32_i64(N_18148) +
                                                                   sext_i32_i64(i_22161)];
                float x_22163 = 0.0F - negate_arg_22162;
                int32_t i_22164 = add32(gtid_21889, x_22134);
                float y_22165 = ((__global
                                  float *) res_mem_24122)[sext_i32_i64(gtid_21886) *
                                                          sext_i32_i64(N_18148) +
                                                          sext_i32_i64(i_22164)];
                float res_22166 = x_22163 + y_22165;
                
                res_22159 = res_22166;
            }
            res_22157 = res_22159;
        }
        ((__local float *) mem_24202)[sext_i32_i64(gtid_21889)] = res_22157;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t dims_flat_25251;
    
    dims_flat_25251 = iota_arg_18497;
    
    float x_22152;
    float x_22153;
    float x_25253;
    float x_25254;
    int32_t skip_threads_25256;
    
    // read input for in-block scan
    {
        if (slt32(local_tid_25246, iota_arg_18497)) {
            x_22153 = ((volatile __local
                        float *) mem_24202)[sext_i32_i64(local_tid_25246)];
            if ((local_tid_25246 - squot32(local_tid_25246, 32) * 32) == 0) {
                x_22152 = x_22153;
            }
        }
    }
    // in-block scan (hopefully no barriers needed)
    {
        skip_threads_25256 = 1;
        while (slt32(skip_threads_25256, 32)) {
            if (sle32(skip_threads_25256, local_tid_25246 -
                      squot32(local_tid_25246, 32) * 32) &&
                slt32(local_tid_25246, iota_arg_18497)) {
                // read operands
                {
                    x_22152 = ((volatile __local
                                float *) mem_24202)[sext_i32_i64(local_tid_25246 -
                                                    skip_threads_25256)];
                }
                // perform operation
                {
                    bool inactive_25257 = slt32(srem32(local_tid_25246,
                                                       iota_arg_18497),
                                                local_tid_25246 -
                                                (local_tid_25246 -
                                                 skip_threads_25256));
                    
                    if (inactive_25257) {
                        x_22152 = x_22153;
                    }
                    if (!inactive_25257) {
                        float res_22154 = x_22152 + x_22153;
                        
                        x_22152 = res_22154;
                    }
                }
            }
            if (sle32(wave_sizze_25248, skip_threads_25256)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (sle32(skip_threads_25256, local_tid_25246 -
                      squot32(local_tid_25246, 32) * 32) &&
                slt32(local_tid_25246, iota_arg_18497)) {
                // write result
                {
                    ((volatile __local
                      float *) mem_24202)[sext_i32_i64(local_tid_25246)] =
                        x_22152;
                    x_22153 = x_22152;
                }
            }
            if (sle32(wave_sizze_25248, skip_threads_25256)) {
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            skip_threads_25256 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of block 'i' writes its result to offset 'i'
    {
        if ((local_tid_25246 - squot32(local_tid_25246, 32) * 32) == 31 &&
            slt32(local_tid_25246, iota_arg_18497)) {
            ((volatile __local
              float *) mem_24202)[sext_i32_i64(squot32(local_tid_25246, 32))] =
                x_22152;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
    {
        int32_t skip_threads_25258;
        
        // read input for in-block scan
        {
            if (squot32(local_tid_25246, 32) == 0 && slt32(local_tid_25246,
                                                           iota_arg_18497)) {
                x_25254 = ((volatile __local
                            float *) mem_24202)[sext_i32_i64(local_tid_25246)];
                if ((local_tid_25246 - squot32(local_tid_25246, 32) * 32) ==
                    0) {
                    x_25253 = x_25254;
                }
            }
        }
        // in-block scan (hopefully no barriers needed)
        {
            skip_threads_25258 = 1;
            while (slt32(skip_threads_25258, 32)) {
                if (sle32(skip_threads_25258, local_tid_25246 -
                          squot32(local_tid_25246, 32) * 32) &&
                    (squot32(local_tid_25246, 32) == 0 && slt32(local_tid_25246,
                                                                iota_arg_18497))) {
                    // read operands
                    {
                        x_25253 = ((volatile __local
                                    float *) mem_24202)[sext_i32_i64(local_tid_25246 -
                                                        skip_threads_25258)];
                    }
                    // perform operation
                    {
                        bool inactive_25259 = slt32(srem32(local_tid_25246 *
                                                           32 + 32 - 1,
                                                           iota_arg_18497),
                                                    local_tid_25246 * 32 + 32 -
                                                    1 - ((local_tid_25246 -
                                                          skip_threads_25258) *
                                                         32 + 32 - 1));
                        
                        if (inactive_25259) {
                            x_25253 = x_25254;
                        }
                        if (!inactive_25259) {
                            float res_25255 = x_25253 + x_25254;
                            
                            x_25253 = res_25255;
                        }
                    }
                }
                if (sle32(wave_sizze_25248, skip_threads_25258)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (sle32(skip_threads_25258, local_tid_25246 -
                          squot32(local_tid_25246, 32) * 32) &&
                    (squot32(local_tid_25246, 32) == 0 && slt32(local_tid_25246,
                                                                iota_arg_18497))) {
                    // write result
                    {
                        ((volatile __local
                          float *) mem_24202)[sext_i32_i64(local_tid_25246)] =
                            x_25253;
                        x_25254 = x_25253;
                    }
                }
                if (sle32(wave_sizze_25248, skip_threads_25258)) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                skip_threads_25258 *= 2;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every block except the first
    {
        if (!(squot32(local_tid_25246, 32) == 0 || !slt32(local_tid_25246,
                                                          iota_arg_18497))) {
            // read operands
            {
                x_22153 = x_22152;
                x_22152 = ((__local
                            float *) mem_24202)[sext_i32_i64(squot32(local_tid_25246,
                                                                     32) - 1)];
            }
            // perform operation
            {
                bool inactive_25260 = slt32(srem32(local_tid_25246,
                                                   iota_arg_18497),
                                            local_tid_25246 -
                                            (squot32(local_tid_25246, 32) * 32 -
                                             1));
                
                if (inactive_25260) {
                    x_22152 = x_22153;
                }
                if (!inactive_25260) {
                    float res_22154 = x_22152 + x_22153;
                    
                    x_22152 = res_22154;
                }
            }
            // write final result
            {
                ((__local float *) mem_24202)[sext_i32_i64(local_tid_25246)] =
                    x_22152;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // restore correct values for first block
    {
        if (squot32(local_tid_25246, 32) == 0) {
            ((__local float *) mem_24202)[sext_i32_i64(local_tid_25246)] =
                x_22153;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    __local char *mem_24206;
    
    mem_24206 = (__local char *) mem_24206_backing_1;
    
    bool acc0_22172;
    int32_t acc0_22173;
    float acc0_22174;
    int32_t gtid_21891 = ltid_pre_25250;
    int32_t phys_tid_21892 = local_tid_25246;
    __local char *red_arr_mem_25261;
    
    red_arr_mem_25261 = (__local char *) red_arr_mem_25261_backing_2;
    
    __local char *red_arr_mem_25263;
    
    red_arr_mem_25263 = (__local char *) red_arr_mem_25263_backing_3;
    
    __local char *red_arr_mem_25265;
    
    red_arr_mem_25265 = (__local char *) red_arr_mem_25265_backing_4;
    if (slt32(gtid_21891, iota_arg_18497)) {
        float x_22190 = ((__local float *) mem_24202)[sext_i32_i64(gtid_21891)];
        float x_22191 = ((__global
                          float *) mem_24192)[sext_i32_i64(gtid_21891)];
        float res_22194 = x_22190 / y_22143;
        bool cond_22195 = slt32(gtid_21891, y_22140);
        bool res_22196;
        
        res_22196 = futrts_isnan32(res_22194);
        
        bool res_22197 = !res_22196;
        bool x_22198 = cond_22195 && res_22197;
        float res_22199 = (float) fabs(res_22194);
        bool res_22200 = x_22191 < res_22199;
        bool x_22201 = x_22198 && res_22200;
        float res_22202;
        
        if (cond_22195) {
            res_22202 = res_22194;
        } else {
            res_22202 = 0.0F;
        }
        ((__local bool *) red_arr_mem_25261)[sext_i32_i64(gtid_21891)] =
            x_22201;
        ((__local int32_t *) red_arr_mem_25263)[sext_i32_i64(gtid_21891)] =
            gtid_21891;
        ((__local float *) red_arr_mem_25265)[sext_i32_i64(gtid_21891)] =
            res_22202;
        ((__local float *) mem_24206)[sext_i32_i64(gtid_21891)] = res_22194;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_25267;
    int32_t skip_waves_25268;
    bool x_22176;
    int32_t x_22177;
    float x_22178;
    bool x_22179;
    int32_t x_22180;
    float x_22181;
    
    offset_25267 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_25246, iota_arg_18497)) {
            x_22176 = ((__local
                        bool *) red_arr_mem_25261)[sext_i32_i64(local_tid_25246 +
                                                   offset_25267)];
            x_22177 = ((__local
                        int32_t *) red_arr_mem_25263)[sext_i32_i64(local_tid_25246 +
                                                      offset_25267)];
            x_22178 = ((__local
                        float *) red_arr_mem_25265)[sext_i32_i64(local_tid_25246 +
                                                    offset_25267)];
        }
    }
    offset_25267 = 1;
    while (slt32(offset_25267, wave_sizze_25248)) {
        if (slt32(local_tid_25246 + offset_25267, iota_arg_18497) &&
            ((local_tid_25246 - squot32(local_tid_25246, wave_sizze_25248) *
              wave_sizze_25248) & (2 * offset_25267 - 1)) == 0) {
            // read array element
            {
                x_22179 = ((volatile __local
                            bool *) red_arr_mem_25261)[sext_i32_i64(local_tid_25246 +
                                                       offset_25267)];
                x_22180 = ((volatile __local
                            int32_t *) red_arr_mem_25263)[sext_i32_i64(local_tid_25246 +
                                                          offset_25267)];
                x_22181 = ((volatile __local
                            float *) red_arr_mem_25265)[sext_i32_i64(local_tid_25246 +
                                                        offset_25267)];
            }
            // apply reduction operation
            {
                bool res_22182;
                int32_t res_22183;
                
                if (x_22176) {
                    res_22182 = x_22176;
                    res_22183 = x_22177;
                } else {
                    bool x_22184 = x_22179 && x_22179;
                    bool x_22185 = !x_22179;
                    bool y_22186 = x_22176 && x_22185;
                    bool res_22187 = x_22184 || y_22186;
                    int32_t res_22188;
                    
                    if (x_22179) {
                        res_22188 = x_22180;
                    } else {
                        res_22188 = x_22177;
                    }
                    res_22182 = res_22187;
                    res_22183 = res_22188;
                }
                
                float res_22189 = x_22178 + x_22181;
                
                x_22176 = res_22182;
                x_22177 = res_22183;
                x_22178 = res_22189;
            }
            // write result of operation
            {
                ((volatile __local
                  bool *) red_arr_mem_25261)[sext_i32_i64(local_tid_25246)] =
                    x_22176;
                ((volatile __local
                  int32_t *) red_arr_mem_25263)[sext_i32_i64(local_tid_25246)] =
                    x_22177;
                ((volatile __local
                  float *) red_arr_mem_25265)[sext_i32_i64(local_tid_25246)] =
                    x_22178;
            }
        }
        offset_25267 *= 2;
    }
    skip_waves_25268 = 1;
    while (slt32(skip_waves_25268, squot32(iota_arg_18497 + wave_sizze_25248 -
                                           1, wave_sizze_25248))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_25267 = skip_waves_25268 * wave_sizze_25248;
        if (slt32(local_tid_25246 + offset_25267, iota_arg_18497) &&
            ((local_tid_25246 - squot32(local_tid_25246, wave_sizze_25248) *
              wave_sizze_25248) == 0 && (squot32(local_tid_25246,
                                                 wave_sizze_25248) & (2 *
                                                                      skip_waves_25268 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_22179 = ((__local
                            bool *) red_arr_mem_25261)[sext_i32_i64(local_tid_25246 +
                                                       offset_25267)];
                x_22180 = ((__local
                            int32_t *) red_arr_mem_25263)[sext_i32_i64(local_tid_25246 +
                                                          offset_25267)];
                x_22181 = ((__local
                            float *) red_arr_mem_25265)[sext_i32_i64(local_tid_25246 +
                                                        offset_25267)];
            }
            // apply reduction operation
            {
                bool res_22182;
                int32_t res_22183;
                
                if (x_22176) {
                    res_22182 = x_22176;
                    res_22183 = x_22177;
                } else {
                    bool x_22184 = x_22179 && x_22179;
                    bool x_22185 = !x_22179;
                    bool y_22186 = x_22176 && x_22185;
                    bool res_22187 = x_22184 || y_22186;
                    int32_t res_22188;
                    
                    if (x_22179) {
                        res_22188 = x_22180;
                    } else {
                        res_22188 = x_22177;
                    }
                    res_22182 = res_22187;
                    res_22183 = res_22188;
                }
                
                float res_22189 = x_22178 + x_22181;
                
                x_22176 = res_22182;
                x_22177 = res_22183;
                x_22178 = res_22189;
            }
            // write result of operation
            {
                ((__local
                  bool *) red_arr_mem_25261)[sext_i32_i64(local_tid_25246)] =
                    x_22176;
                ((__local
                  int32_t *) red_arr_mem_25263)[sext_i32_i64(local_tid_25246)] =
                    x_22177;
                ((__local
                  float *) red_arr_mem_25265)[sext_i32_i64(local_tid_25246)] =
                    x_22178;
            }
        }
        skip_waves_25268 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    acc0_22172 = ((__local bool *) red_arr_mem_25261)[0];
    acc0_22173 = ((__local int32_t *) red_arr_mem_25263)[0];
    acc0_22174 = ((__local float *) red_arr_mem_25265)[0];
    
    bool x_22205 = acc0_22172 && acc0_22172;
    int32_t res_22209;
    
    if (acc0_22172) {
        res_22209 = acc0_22173;
    } else {
        res_22209 = -1;
    }
    
    bool cond_22216 = !x_22205;
    int32_t fst_breakzq_22217;
    
    if (cond_22216) {
        fst_breakzq_22217 = -1;
    } else {
        bool cond_22218 = slt32(res_22209, y_22140);
        int32_t res_22219;
        
        if (cond_22218) {
            int32_t i_22220 = add32(x_22134, res_22209);
            int32_t x_22221 = ((__global
                                int32_t *) res_mem_24123)[sext_i32_i64(gtid_21886) *
                                                          sext_i32_i64(N_18148) +
                                                          sext_i32_i64(i_22220)];
            int32_t res_22222 = sub32(x_22221, n_18153);
            
            res_22219 = res_22222;
        } else {
            res_22219 = -1;
        }
        fst_breakzq_22217 = res_22219;
    }
    
    bool cond_22223 = sle32(x_22134, 5);
    bool res_22224 = sle32(y_22140, 5);
    bool x_22225 = !cond_22223;
    bool y_22226 = res_22224 && x_22225;
    bool cond_22227 = cond_22223 || y_22226;
    int32_t fst_breakzq_22228;
    
    if (cond_22227) {
        fst_breakzq_22228 = -2;
    } else {
        fst_breakzq_22228 = fst_breakzq_22217;
    }
    
    __local char *mem_24209;
    
    mem_24209 = (__local char *) mem_24209_backing_5;
    ((__local float *) mem_24209)[sext_i32_i64(local_tid_25246)] = NAN;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t write_i_21893 = ltid_pre_25250;
    int32_t phys_tid_21894 = local_tid_25246;
    
    if (slt32(write_i_21893, iota_arg_18497)) {
        float write_value_22232 = ((__local
                                    float *) mem_24206)[sext_i32_i64(write_i_21893)];
        bool cond_22233 = slt32(write_i_21893, y_22140);
        int32_t res_22234;
        
        if (cond_22233) {
            int32_t i_22235 = add32(write_i_21893, x_22134);
            int32_t x_22236 = ((__global
                                int32_t *) res_mem_24123)[sext_i32_i64(gtid_21886) *
                                                          sext_i32_i64(N_18148) +
                                                          sext_i32_i64(i_22235)];
            int32_t res_22237 = sub32(x_22236, n_18153);
            
            res_22234 = res_22237;
        } else {
            res_22234 = -1;
        }
        if (sle32(0, res_22234) && slt32(res_22234, iota_arg_18497)) {
            ((__local float *) mem_24209)[sext_i32_i64(res_22234)] =
                write_value_22232;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ((__global float *) mem_24215)[sext_i32_i64(gtid_21886) *
                                   sext_i32_i64(iota_arg_18497) +
                                   sext_i32_i64(local_tid_25246)] = ((__local
                                                                      float *) mem_24209)[sext_i32_i64(local_tid_25246)];
    barrier(CLK_LOCAL_MEM_FENCE);
    ((__global float *) mem_24220)[sext_i32_i64(gtid_21886) *
                                   sext_i32_i64(iota_arg_18497) +
                                   sext_i32_i64(local_tid_25246)] = ((__local
                                                                      float *) mem_24206)[sext_i32_i64(local_tid_25246)];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_tid_25246 == 0) {
        ((__global int32_t *) mem_24223)[sext_i32_i64(gtid_21886)] =
            fst_breakzq_22228;
    }
    if (local_tid_25246 == 0) {
        ((__global float *) mem_24226)[sext_i32_i64(gtid_21886)] = acc0_22174;
    }
    
  error_4:
    return;
}
__kernel void mainzisegmap_intragroup_22839(__global int *global_failure,
                                            __local volatile
                                            int64_t *mem_23853_backing_aligned_0,
                                            __local volatile
                                            int64_t *mem_23848_backing_aligned_1,
                                            __local volatile
                                            int64_t *mem_23825_backing_aligned_2,
                                            __local volatile
                                            int64_t *mem_23820_backing_aligned_3,
                                            int32_t m_18149, int32_t N_18150,
                                            int32_t n_18153,
                                            int32_t k2p2zq_18165,
                                            int32_t num_groups_y_22837,
                                            int32_t num_whole_tiles_22855,
                                            int32_t residual_input_23005,
                                            unsigned char cond_23006, __global
                                            unsigned char *images_mem_23523,
                                            __global unsigned char *mem_23541,
                                            __global unsigned char *mem_23872)
{
    #define tile_sizze_22834 (mainzitile_sizze_22833)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_23853_backing_7 = (__local volatile
                                                           char *) mem_23853_backing_aligned_0;
    __local volatile char *restrict mem_23848_backing_6 = (__local volatile
                                                           char *) mem_23848_backing_aligned_1;
    __local volatile char *restrict mem_23825_backing_1 = (__local volatile
                                                           char *) mem_23825_backing_aligned_2;
    __local volatile char *restrict mem_23820_backing_0 = (__local volatile
                                                           char *) mem_23820_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24650;
    int32_t local_tid_24651;
    int32_t group_sizze_24654;
    int32_t wave_sizze_24653;
    int32_t group_tid_24652;
    
    global_tid_24650 = get_global_id(0);
    local_tid_24651 = get_local_id(0);
    group_sizze_24654 = get_local_size(0);
    wave_sizze_24653 = LOCKSTEP_WIDTH;
    group_tid_24652 = get_group_id(0);
    
    int32_t gid_flat_22839;
    
    gid_flat_22839 = group_tid_24652;
    
    int32_t ltid_pre_24655;
    
    ltid_pre_24655 = squot32(local_tid_24651, tile_sizze_22834);
    
    int32_t ltid_pre_24656;
    
    ltid_pre_24656 = local_tid_24651 - squot32(local_tid_24651,
                                               tile_sizze_22834) *
        tile_sizze_22834;
    
    int32_t gid_x_22831;
    
    gid_x_22831 = squot32(group_tid_24652, num_groups_y_22837);
    
    int32_t gid_y_22832;
    
    gid_y_22832 = group_tid_24652 - squot32(group_tid_24652,
                                            num_groups_y_22837) *
        num_groups_y_22837;
    
    float mem_23803[1];
    int32_t ltid_x_22856 = ltid_pre_24655;
    int32_t ltid_y_22857 = ltid_pre_24656;
    int32_t ltid_flat_22858 = local_tid_24651;
    
    if (slt32(ltid_x_22856, tile_sizze_22834) && slt32(ltid_y_22857,
                                                       tile_sizze_22834)) {
        mem_23803[0] = 0.0F;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_22955 = gid_x_22831 * tile_sizze_22834;
    int32_t binop_x_22957 = gid_y_22832 * tile_sizze_22834;
    __local char *mem_23820;
    
    mem_23820 = (__local char *) mem_23820_backing_0;
    
    __local char *mem_23825;
    
    mem_23825 = (__local char *) mem_23825_backing_1;
    
    float accs_mem_23842[1];
    float mem_param_23811[1];
    
    for (int32_t i_2 = 0; i_2 < 1; i_2++)
        mem_param_23811[i_2] = mem_23803[i_2];
    for (int32_t tile_id_22867 = 0; tile_id_22867 < num_whole_tiles_22855;
         tile_id_22867++) {
        int32_t binop_x_22951 = tile_sizze_22834 * tile_id_22867;
        int32_t ltid_x_22868 = ltid_pre_24655;
        int32_t ltid_y_22869 = ltid_pre_24656;
        int32_t ltid_flat_22870 = local_tid_24651;
        int32_t i_22952 = ltid_x_22868 + binop_x_22951;
        int32_t j_22954 = ltid_y_22869 + binop_x_22951;
        int32_t gtid_22956 = ltid_x_22868 + binop_x_22955;
        int32_t gtid_22958 = ltid_y_22869 + binop_x_22957;
        bool binop_x_22961 = slt32(i_22952, n_18153);
        bool binop_y_22962 = slt32(gtid_22958, k2p2zq_18165);
        bool cond_22963 = binop_x_22961 && binop_y_22962;
        float pre_22964;
        
        if (cond_22963) {
            float x_22965 = ((__global
                              float *) mem_23541)[sext_i32_i64(i_22952) *
                                                  sext_i32_i64(k2p2zq_18165) +
                                                  sext_i32_i64(gtid_22958)];
            
            pre_22964 = x_22965;
        } else {
            pre_22964 = 0.0F;
        }
        
        bool binop_x_22967 = slt32(j_22954, n_18153);
        bool binop_y_22968 = slt32(gtid_22956, m_18149);
        bool cond_22969 = binop_x_22967 && binop_y_22968;
        float pre_22970;
        
        if (cond_22969) {
            float x_22971 = ((__global
                              float *) images_mem_23523)[sext_i32_i64(gtid_22956) *
                                                         sext_i32_i64(N_18150) +
                                                         sext_i32_i64(j_22954)];
            
            pre_22970 = x_22971;
        } else {
            pre_22970 = 0.0F;
        }
        ((__local float *) mem_23820)[sext_i32_i64(ltid_x_22868) *
                                      sext_i32_i64(tile_sizze_22834) +
                                      sext_i32_i64(ltid_y_22869)] = pre_22964;
        ((__local float *) mem_23825)[sext_i32_i64(ltid_x_22868) *
                                      sext_i32_i64(tile_sizze_22834) +
                                      sext_i32_i64(ltid_y_22869)] = pre_22970;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float mem_23831[1];
        int32_t ltid_x_22913 = ltid_pre_24655;
        int32_t ltid_y_22914 = ltid_pre_24656;
        int32_t ltid_flat_22915 = local_tid_24651;
        int32_t gtid_22975 = ltid_x_22913 + binop_x_22955;
        int32_t gtid_22977 = ltid_y_22914 + binop_x_22957;
        float acc_22980 = mem_param_23811[0];
        bool binop_x_22983 = slt32(gtid_22975, m_18149);
        bool binop_y_22984 = slt32(gtid_22977, k2p2zq_18165);
        bool cond_22985 = binop_x_22983 && binop_y_22984;
        float acc_22986;
        
        if (cond_22985) {
            float x_22987;
            float redout_23489 = acc_22980;
            
            for (int32_t i_23490 = 0; i_23490 < tile_sizze_22834; i_23490++) {
                float x_22992 = ((__local
                                  float *) mem_23825)[sext_i32_i64(ltid_x_22913) *
                                                      sext_i32_i64(tile_sizze_22834) +
                                                      sext_i32_i64(i_23490)];
                bool res_22993;
                
                res_22993 = futrts_isnan32(x_22992);
                
                float res_22994;
                
                if (res_22993) {
                    res_22994 = 0.0F;
                } else {
                    float x_22991 = ((__local
                                      float *) mem_23820)[sext_i32_i64(i_23490) *
                                                          sext_i32_i64(tile_sizze_22834) +
                                                          sext_i32_i64(ltid_y_22914)];
                    float res_22995 = x_22991 * x_22992;
                    
                    res_22994 = res_22995;
                }
                
                float res_22990 = res_22994 + redout_23489;
                float redout_tmp_24659 = res_22990;
                
                redout_23489 = redout_tmp_24659;
            }
            x_22987 = redout_23489;
            acc_22986 = x_22987;
        } else {
            acc_22986 = acc_22980;
        }
        mem_23831[0] = acc_22986;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float mem_param_tmp_24657[1];
        
        for (int32_t i_3 = 0; i_3 < 1; i_3++)
            mem_param_tmp_24657[i_3] = mem_23831[i_3];
        for (int32_t i_4 = 0; i_4 < 1; i_4++)
            mem_param_23811[i_4] = mem_param_tmp_24657[i_4];
    }
    for (int32_t i_5 = 0; i_5 < 1; i_5++)
        accs_mem_23842[i_5] = mem_param_23811[i_5];
    
    __local char *mem_23848;
    
    mem_23848 = (__local char *) mem_23848_backing_6;
    
    __local char *mem_23853;
    
    mem_23853 = (__local char *) mem_23853_backing_7;
    
    float mem_23859[1];
    float mem_24285[1];
    
    if (cond_23006) {
        mem_24285[0] = accs_mem_23842[0];
    } else {
        int32_t binop_x_23092 = tile_sizze_22834 * num_whole_tiles_22855;
        int32_t ltid_x_23007 = ltid_pre_24655;
        int32_t ltid_y_23008 = ltid_pre_24656;
        int32_t ltid_flat_23009 = local_tid_24651;
        int32_t i_23093 = ltid_x_23007 + binop_x_23092;
        int32_t j_23095 = ltid_y_23008 + binop_x_23092;
        int32_t gtid_23097 = binop_x_22955 + ltid_x_23007;
        int32_t gtid_23099 = binop_x_22957 + ltid_y_23008;
        bool binop_x_23102 = slt32(i_23093, n_18153);
        bool binop_y_23103 = slt32(gtid_23099, k2p2zq_18165);
        bool cond_23104 = binop_x_23102 && binop_y_23103;
        float pre_23105;
        
        if (cond_23104) {
            float x_23106 = ((__global
                              float *) mem_23541)[sext_i32_i64(i_23093) *
                                                  sext_i32_i64(k2p2zq_18165) +
                                                  sext_i32_i64(gtid_23099)];
            
            pre_23105 = x_23106;
        } else {
            pre_23105 = 0.0F;
        }
        
        bool binop_x_23108 = slt32(j_23095, n_18153);
        bool binop_y_23109 = slt32(gtid_23097, m_18149);
        bool cond_23110 = binop_x_23108 && binop_y_23109;
        float pre_23111;
        
        if (cond_23110) {
            float x_23112 = ((__global
                              float *) images_mem_23523)[sext_i32_i64(gtid_23097) *
                                                         sext_i32_i64(N_18150) +
                                                         sext_i32_i64(j_23095)];
            
            pre_23111 = x_23112;
        } else {
            pre_23111 = 0.0F;
        }
        ((__local float *) mem_23848)[sext_i32_i64(ltid_x_23007) *
                                      sext_i32_i64(tile_sizze_22834) +
                                      sext_i32_i64(ltid_y_23008)] = pre_23105;
        ((__local float *) mem_23853)[sext_i32_i64(ltid_x_23007) *
                                      sext_i32_i64(tile_sizze_22834) +
                                      sext_i32_i64(ltid_y_23008)] = pre_23111;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_23054 = ltid_pre_24655;
        int32_t ltid_y_23055 = ltid_pre_24656;
        int32_t ltid_flat_23056 = local_tid_24651;
        int32_t gtid_23118 = binop_x_22955 + ltid_x_23054;
        int32_t gtid_23120 = binop_x_22957 + ltid_y_23055;
        float acc_23123 = accs_mem_23842[0];
        bool binop_x_23126 = slt32(gtid_23118, m_18149);
        bool binop_y_23127 = slt32(gtid_23120, k2p2zq_18165);
        bool cond_23128 = binop_x_23126 && binop_y_23127;
        float acc_23129;
        
        if (cond_23128) {
            float x_23130;
            float redout_23491 = acc_23123;
            
            for (int32_t i_23492 = 0; i_23492 < residual_input_23005;
                 i_23492++) {
                float x_23135 = ((__local
                                  float *) mem_23853)[sext_i32_i64(ltid_x_23054) *
                                                      sext_i32_i64(tile_sizze_22834) +
                                                      sext_i32_i64(i_23492)];
                bool res_23136;
                
                res_23136 = futrts_isnan32(x_23135);
                
                float res_23137;
                
                if (res_23136) {
                    res_23137 = 0.0F;
                } else {
                    float x_23134 = ((__local
                                      float *) mem_23848)[sext_i32_i64(i_23492) *
                                                          sext_i32_i64(tile_sizze_22834) +
                                                          sext_i32_i64(ltid_y_23055)];
                    float res_23138 = x_23134 * x_23135;
                    
                    res_23137 = res_23138;
                }
                
                float res_23133 = res_23137 + redout_23491;
                float redout_tmp_24660 = res_23133;
                
                redout_23491 = redout_tmp_24660;
            }
            x_23130 = redout_23491;
            acc_23129 = x_23130;
        } else {
            acc_23129 = acc_23123;
        }
        mem_23859[0] = acc_23129;
        barrier(CLK_LOCAL_MEM_FENCE);
        mem_24285[0] = mem_23859[0];
    }
    
    int32_t thread_out_index_24661 = gid_x_22831 * tile_sizze_22834 +
            ltid_pre_24655;
    int32_t thread_out_index_24662 = gid_y_22832 * tile_sizze_22834 +
            ltid_pre_24656;
    
    if (slt32(thread_out_index_24661, m_18149) && slt32(thread_out_index_24662,
                                                        k2p2zq_18165)) {
        ((__global float *) mem_23872)[sext_i32_i64(thread_out_index_24661) *
                                       sext_i32_i64(k2p2zq_18165) +
                                       sext_i32_i64(thread_out_index_24662)] =
            mem_24285[0];
    }
    
  error_5:
    return;
    #undef tile_sizze_22834
}
__kernel void mainzisegmap_intragroup_23161(__global int *global_failure,
                                            __local volatile
                                            int64_t *mem_24034_backing_aligned_0,
                                            __local volatile
                                            int64_t *mem_24029_backing_aligned_1,
                                            __local volatile
                                            int64_t *mem_24006_backing_aligned_2,
                                            __local volatile
                                            int64_t *mem_24001_backing_aligned_3,
                                            int32_t N_18148, int32_t m_18149,
                                            int32_t k2p2zq_18165,
                                            int32_t num_groups_y_23159,
                                            int32_t num_whole_tiles_23177,
                                            int32_t residual_input_23321,
                                            unsigned char cond_23322, __global
                                            unsigned char *res_mem_23946,
                                            __global unsigned char *mem_23977,
                                            __global unsigned char *mem_24053)
{
    #define tile_sizze_23156 (mainzitile_sizze_23155)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_24034_backing_7 = (__local volatile
                                                           char *) mem_24034_backing_aligned_0;
    __local volatile char *restrict mem_24029_backing_6 = (__local volatile
                                                           char *) mem_24029_backing_aligned_1;
    __local volatile char *restrict mem_24006_backing_1 = (__local volatile
                                                           char *) mem_24006_backing_aligned_2;
    __local volatile char *restrict mem_24001_backing_0 = (__local volatile
                                                           char *) mem_24001_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24817;
    int32_t local_tid_24818;
    int32_t group_sizze_24821;
    int32_t wave_sizze_24820;
    int32_t group_tid_24819;
    
    global_tid_24817 = get_global_id(0);
    local_tid_24818 = get_local_id(0);
    group_sizze_24821 = get_local_size(0);
    wave_sizze_24820 = LOCKSTEP_WIDTH;
    group_tid_24819 = get_group_id(0);
    
    int32_t gid_flat_23161;
    
    gid_flat_23161 = group_tid_24819;
    
    int32_t ltid_pre_24822;
    
    ltid_pre_24822 = squot32(local_tid_24818, tile_sizze_23156);
    
    int32_t ltid_pre_24823;
    
    ltid_pre_24823 = local_tid_24818 - squot32(local_tid_24818,
                                               tile_sizze_23156) *
        tile_sizze_23156;
    
    int32_t gid_x_23153;
    
    gid_x_23153 = squot32(group_tid_24819, num_groups_y_23159);
    
    int32_t gid_y_23154;
    
    gid_y_23154 = group_tid_24819 - squot32(group_tid_24819,
                                            num_groups_y_23159) *
        num_groups_y_23159;
    
    float mem_23984[1];
    int32_t ltid_x_23178 = ltid_pre_24822;
    int32_t ltid_y_23179 = ltid_pre_24823;
    int32_t ltid_flat_23180 = local_tid_24818;
    
    if (slt32(ltid_x_23178, tile_sizze_23156) && slt32(ltid_y_23179,
                                                       tile_sizze_23156)) {
        mem_23984[0] = 0.0F;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_23275 = gid_x_23153 * tile_sizze_23156;
    int32_t binop_x_23277 = gid_y_23154 * tile_sizze_23156;
    __local char *mem_24001;
    
    mem_24001 = (__local char *) mem_24001_backing_0;
    
    __local char *mem_24006;
    
    mem_24006 = (__local char *) mem_24006_backing_1;
    
    float accs_mem_24023[1];
    float mem_param_23992[1];
    
    for (int32_t i_2 = 0; i_2 < 1; i_2++)
        mem_param_23992[i_2] = mem_23984[i_2];
    for (int32_t tile_id_23189 = 0; tile_id_23189 < num_whole_tiles_23177;
         tile_id_23189++) {
        int32_t binop_x_23271 = tile_sizze_23156 * tile_id_23189;
        int32_t ltid_x_23190 = ltid_pre_24822;
        int32_t ltid_y_23191 = ltid_pre_24823;
        int32_t ltid_flat_23192 = local_tid_24818;
        int32_t i_23272 = ltid_x_23190 + binop_x_23271;
        int32_t j_23274 = ltid_y_23191 + binop_x_23271;
        int32_t gtid_23276 = ltid_x_23190 + binop_x_23275;
        int32_t gtid_23278 = ltid_y_23191 + binop_x_23277;
        bool binop_x_23281 = slt32(j_23274, k2p2zq_18165);
        bool binop_y_23282 = slt32(gtid_23276, m_18149);
        bool cond_23283 = binop_x_23281 && binop_y_23282;
        float pre_23284;
        
        if (cond_23283) {
            float x_23285 = ((__global
                              float *) res_mem_23946)[sext_i32_i64(gtid_23276) *
                                                      sext_i32_i64(k2p2zq_18165) +
                                                      sext_i32_i64(j_23274)];
            
            pre_23284 = x_23285;
        } else {
            pre_23284 = 0.0F;
        }
        
        bool binop_x_23287 = slt32(i_23272, k2p2zq_18165);
        bool binop_y_23288 = slt32(gtid_23278, N_18148);
        bool cond_23289 = binop_x_23287 && binop_y_23288;
        float pre_23290;
        
        if (cond_23289) {
            float x_23291 = ((__global
                              float *) mem_23977)[sext_i32_i64(i_23272) *
                                                  sext_i32_i64(N_18148) +
                                                  sext_i32_i64(gtid_23278)];
            
            pre_23290 = x_23291;
        } else {
            pre_23290 = 0.0F;
        }
        ((__local float *) mem_24001)[sext_i32_i64(ltid_x_23190) *
                                      sext_i32_i64(tile_sizze_23156) +
                                      sext_i32_i64(ltid_y_23191)] = pre_23284;
        ((__local float *) mem_24006)[sext_i32_i64(ltid_x_23190) *
                                      sext_i32_i64(tile_sizze_23156) +
                                      sext_i32_i64(ltid_y_23191)] = pre_23290;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float mem_24012[1];
        int32_t ltid_x_23235 = ltid_pre_24822;
        int32_t ltid_y_23236 = ltid_pre_24823;
        int32_t ltid_flat_23237 = local_tid_24818;
        int32_t gtid_23295 = ltid_x_23235 + binop_x_23275;
        int32_t gtid_23297 = ltid_y_23236 + binop_x_23277;
        float acc_23300 = mem_param_23992[0];
        bool binop_x_23303 = slt32(gtid_23295, m_18149);
        bool binop_y_23304 = slt32(gtid_23297, N_18148);
        bool cond_23305 = binop_x_23303 && binop_y_23304;
        float acc_23306;
        
        if (cond_23305) {
            float x_23307;
            float redout_23507 = acc_23300;
            
            for (int32_t i_23508 = 0; i_23508 < tile_sizze_23156; i_23508++) {
                float x_23311 = ((__local
                                  float *) mem_24001)[sext_i32_i64(ltid_x_23235) *
                                                      sext_i32_i64(tile_sizze_23156) +
                                                      sext_i32_i64(i_23508)];
                float x_23312 = ((__local
                                  float *) mem_24006)[sext_i32_i64(i_23508) *
                                                      sext_i32_i64(tile_sizze_23156) +
                                                      sext_i32_i64(ltid_y_23236)];
                float res_23313 = x_23311 * x_23312;
                float res_23310 = res_23313 + redout_23507;
                float redout_tmp_24826 = res_23310;
                
                redout_23507 = redout_tmp_24826;
            }
            x_23307 = redout_23507;
            acc_23306 = x_23307;
        } else {
            acc_23306 = acc_23300;
        }
        mem_24012[0] = acc_23306;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        float mem_param_tmp_24824[1];
        
        for (int32_t i_3 = 0; i_3 < 1; i_3++)
            mem_param_tmp_24824[i_3] = mem_24012[i_3];
        for (int32_t i_4 = 0; i_4 < 1; i_4++)
            mem_param_23992[i_4] = mem_param_tmp_24824[i_4];
    }
    for (int32_t i_5 = 0; i_5 < 1; i_5++)
        accs_mem_24023[i_5] = mem_param_23992[i_5];
    
    __local char *mem_24029;
    
    mem_24029 = (__local char *) mem_24029_backing_6;
    
    __local char *mem_24034;
    
    mem_24034 = (__local char *) mem_24034_backing_7;
    
    float mem_24040[1];
    float mem_24299[1];
    
    if (cond_23322) {
        mem_24299[0] = accs_mem_24023[0];
    } else {
        int32_t binop_x_23406 = tile_sizze_23156 * num_whole_tiles_23177;
        int32_t ltid_x_23323 = ltid_pre_24822;
        int32_t ltid_y_23324 = ltid_pre_24823;
        int32_t ltid_flat_23325 = local_tid_24818;
        int32_t i_23407 = ltid_x_23323 + binop_x_23406;
        int32_t j_23409 = ltid_y_23324 + binop_x_23406;
        int32_t gtid_23411 = binop_x_23275 + ltid_x_23323;
        int32_t gtid_23413 = binop_x_23277 + ltid_y_23324;
        bool binop_x_23416 = slt32(j_23409, k2p2zq_18165);
        bool binop_y_23417 = slt32(gtid_23411, m_18149);
        bool cond_23418 = binop_x_23416 && binop_y_23417;
        float pre_23419;
        
        if (cond_23418) {
            float x_23420 = ((__global
                              float *) res_mem_23946)[sext_i32_i64(gtid_23411) *
                                                      sext_i32_i64(k2p2zq_18165) +
                                                      sext_i32_i64(j_23409)];
            
            pre_23419 = x_23420;
        } else {
            pre_23419 = 0.0F;
        }
        
        bool binop_x_23422 = slt32(i_23407, k2p2zq_18165);
        bool binop_y_23423 = slt32(gtid_23413, N_18148);
        bool cond_23424 = binop_x_23422 && binop_y_23423;
        float pre_23425;
        
        if (cond_23424) {
            float x_23426 = ((__global
                              float *) mem_23977)[sext_i32_i64(i_23407) *
                                                  sext_i32_i64(N_18148) +
                                                  sext_i32_i64(gtid_23413)];
            
            pre_23425 = x_23426;
        } else {
            pre_23425 = 0.0F;
        }
        ((__local float *) mem_24029)[sext_i32_i64(ltid_x_23323) *
                                      sext_i32_i64(tile_sizze_23156) +
                                      sext_i32_i64(ltid_y_23324)] = pre_23419;
        ((__local float *) mem_24034)[sext_i32_i64(ltid_x_23323) *
                                      sext_i32_i64(tile_sizze_23156) +
                                      sext_i32_i64(ltid_y_23324)] = pre_23425;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_23370 = ltid_pre_24822;
        int32_t ltid_y_23371 = ltid_pre_24823;
        int32_t ltid_flat_23372 = local_tid_24818;
        int32_t gtid_23432 = binop_x_23275 + ltid_x_23370;
        int32_t gtid_23434 = binop_x_23277 + ltid_y_23371;
        float acc_23437 = accs_mem_24023[0];
        bool binop_x_23440 = slt32(gtid_23432, m_18149);
        bool binop_y_23441 = slt32(gtid_23434, N_18148);
        bool cond_23442 = binop_x_23440 && binop_y_23441;
        float acc_23443;
        
        if (cond_23442) {
            float x_23444;
            float redout_23509 = acc_23437;
            
            for (int32_t i_23510 = 0; i_23510 < residual_input_23321;
                 i_23510++) {
                float x_23448 = ((__local
                                  float *) mem_24029)[sext_i32_i64(ltid_x_23370) *
                                                      sext_i32_i64(tile_sizze_23156) +
                                                      sext_i32_i64(i_23510)];
                float x_23449 = ((__local
                                  float *) mem_24034)[sext_i32_i64(i_23510) *
                                                      sext_i32_i64(tile_sizze_23156) +
                                                      sext_i32_i64(ltid_y_23371)];
                float res_23450 = x_23448 * x_23449;
                float res_23447 = res_23450 + redout_23509;
                float redout_tmp_24827 = res_23447;
                
                redout_23509 = redout_tmp_24827;
            }
            x_23444 = redout_23509;
            acc_23443 = x_23444;
        } else {
            acc_23443 = acc_23437;
        }
        mem_24040[0] = acc_23443;
        barrier(CLK_LOCAL_MEM_FENCE);
        mem_24299[0] = mem_24040[0];
    }
    
    int32_t thread_out_index_24828 = gid_x_23153 * tile_sizze_23156 +
            ltid_pre_24822;
    int32_t thread_out_index_24829 = gid_y_23154 * tile_sizze_23156 +
            ltid_pre_24823;
    
    if (slt32(thread_out_index_24828, m_18149) && slt32(thread_out_index_24829,
                                                        N_18148)) {
        ((__global float *) mem_24053)[sext_i32_i64(thread_out_index_24828) *
                                       sext_i32_i64(N_18148) +
                                       sext_i32_i64(thread_out_index_24829)] =
            mem_24299[0];
    }
    
  error_5:
    return;
    #undef tile_sizze_23156
}
__kernel void mainzisegred_large_19482(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_24546_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24544_backing_aligned_1,
                                       int32_t N_18148, int32_t N_18150,
                                       int32_t n_18153, int32_t k2p2zq_18165,
                                       int32_t num_groups_19645, __global
                                       unsigned char *images_mem_23523, __global
                                       unsigned char *binop_p_mem_23536,
                                       __global unsigned char *mem_23641,
                                       __global unsigned char *mem_23649,
                                       int32_t groups_per_segment_24530,
                                       int32_t elements_per_thread_24531,
                                       int32_t virt_num_groups_24532,
                                       int32_t threads_per_segment_24534,
                                       __global
                                       unsigned char *group_res_arr_mem_24535,
                                       __global
                                       unsigned char *mainzicounter_mem_24537)
{
    #define segred_group_sizze_19644 (mainzisegred_group_sizze_19476)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_24546_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_24546_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24544_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24544_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24539;
    int32_t local_tid_24540;
    int32_t group_sizze_24543;
    int32_t wave_sizze_24542;
    int32_t group_tid_24541;
    
    global_tid_24539 = get_global_id(0);
    local_tid_24540 = get_local_id(0);
    group_sizze_24543 = get_local_size(0);
    wave_sizze_24542 = LOCKSTEP_WIDTH;
    group_tid_24541 = get_group_id(0);
    
    int32_t phys_tid_19482;
    
    phys_tid_19482 = global_tid_24539;
    
    __local char *red_arr_mem_24544;
    
    red_arr_mem_24544 = (__local char *) red_arr_mem_24544_backing_0;
    
    __local char *sync_arr_mem_24546;
    
    sync_arr_mem_24546 = (__local char *) sync_arr_mem_24546_backing_1;
    
    int32_t phys_group_id_24548;
    
    phys_group_id_24548 = get_group_id(0);
    for (int32_t i_24549 = 0; i_24549 < sdiv_up32(virt_num_groups_24532 -
                                                  phys_group_id_24548,
                                                  num_groups_19645);
         i_24549++) {
        int32_t virt_group_id_24550 = phys_group_id_24548 + i_24549 *
                num_groups_19645;
        int32_t flat_segment_id_24551 = squot32(virt_group_id_24550,
                                                groups_per_segment_24530);
        int32_t global_tid_24552 = srem32(virt_group_id_24550 *
                                          segred_group_sizze_19644 +
                                          local_tid_24540,
                                          segred_group_sizze_19644 *
                                          groups_per_segment_24530);
        int32_t gtid_19465 = squot32(flat_segment_id_24551, k2p2zq_18165 *
                                     k2p2zq_18165);
        int32_t gtid_19466 = squot32(flat_segment_id_24551 -
                                     squot32(flat_segment_id_24551,
                                             k2p2zq_18165 * k2p2zq_18165) *
                                     (k2p2zq_18165 * k2p2zq_18165),
                                     k2p2zq_18165);
        int32_t gtid_19467 = flat_segment_id_24551 -
                squot32(flat_segment_id_24551, k2p2zq_18165 * k2p2zq_18165) *
                (k2p2zq_18165 * k2p2zq_18165) - squot32(flat_segment_id_24551 -
                                                        squot32(flat_segment_id_24551,
                                                                k2p2zq_18165 *
                                                                k2p2zq_18165) *
                                                        (k2p2zq_18165 *
                                                         k2p2zq_18165),
                                                        k2p2zq_18165) *
                k2p2zq_18165;
        int32_t gtid_19481;
        float x_acc_24553;
        int32_t chunk_sizze_24554;
        
        chunk_sizze_24554 = smin32(elements_per_thread_24531,
                                   sdiv_up32(n_18153 - global_tid_24552,
                                             threads_per_segment_24534));
        
        float x_19648;
        float x_19649;
        
        // neutral-initialise the accumulators
        {
            x_acc_24553 = 0.0F;
        }
        for (int32_t i_24558 = 0; i_24558 < chunk_sizze_24554; i_24558++) {
            gtid_19481 = global_tid_24552 + threads_per_segment_24534 * i_24558;
            // apply map function
            {
                float x_19654 = ((__global
                                  float *) images_mem_23523)[sext_i32_i64(gtid_19465) *
                                                             sext_i32_i64(N_18150) +
                                                             sext_i32_i64(gtid_19481)];
                float x_19655 = ((__global
                                  float *) binop_p_mem_23536)[sext_i32_i64(gtid_19466) *
                                                              sext_i32_i64(N_18148) +
                                                              sext_i32_i64(gtid_19481)];
                float x_19656 = ((__global
                                  float *) mem_23641)[sext_i32_i64(gtid_19467) *
                                                      sext_i32_i64(N_18148) +
                                                      sext_i32_i64(gtid_19481)];
                float x_19657 = x_19655 * x_19656;
                bool res_19658;
                
                res_19658 = futrts_isnan32(x_19654);
                
                float y_19659;
                
                if (res_19658) {
                    y_19659 = 0.0F;
                } else {
                    y_19659 = 1.0F;
                }
                
                float res_19660 = x_19657 * y_19659;
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_19648 = x_acc_24553;
                }
                // load new values
                {
                    x_19649 = res_19660;
                }
                // apply reduction operator
                {
                    float res_19650 = x_19648 + x_19649;
                    
                    // store in accumulator
                    {
                        x_acc_24553 = res_19650;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_19648 = x_acc_24553;
            ((__local
              float *) red_arr_mem_24544)[sext_i32_i64(local_tid_24540)] =
                x_19648;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_24559;
        int32_t skip_waves_24560;
        float x_24555;
        float x_24556;
        
        offset_24559 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_24540, segred_group_sizze_19644)) {
                x_24555 = ((__local
                            float *) red_arr_mem_24544)[sext_i32_i64(local_tid_24540 +
                                                        offset_24559)];
            }
        }
        offset_24559 = 1;
        while (slt32(offset_24559, wave_sizze_24542)) {
            if (slt32(local_tid_24540 + offset_24559,
                      segred_group_sizze_19644) && ((local_tid_24540 -
                                                     squot32(local_tid_24540,
                                                             wave_sizze_24542) *
                                                     wave_sizze_24542) & (2 *
                                                                          offset_24559 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_24556 = ((volatile __local
                                float *) red_arr_mem_24544)[sext_i32_i64(local_tid_24540 +
                                                            offset_24559)];
                }
                // apply reduction operation
                {
                    float res_24557 = x_24555 + x_24556;
                    
                    x_24555 = res_24557;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_24544)[sext_i32_i64(local_tid_24540)] =
                        x_24555;
                }
            }
            offset_24559 *= 2;
        }
        skip_waves_24560 = 1;
        while (slt32(skip_waves_24560, squot32(segred_group_sizze_19644 +
                                               wave_sizze_24542 - 1,
                                               wave_sizze_24542))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_24559 = skip_waves_24560 * wave_sizze_24542;
            if (slt32(local_tid_24540 + offset_24559,
                      segred_group_sizze_19644) && ((local_tid_24540 -
                                                     squot32(local_tid_24540,
                                                             wave_sizze_24542) *
                                                     wave_sizze_24542) == 0 &&
                                                    (squot32(local_tid_24540,
                                                             wave_sizze_24542) &
                                                     (2 * skip_waves_24560 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_24556 = ((__local
                                float *) red_arr_mem_24544)[sext_i32_i64(local_tid_24540 +
                                                            offset_24559)];
                }
                // apply reduction operation
                {
                    float res_24557 = x_24555 + x_24556;
                    
                    x_24555 = res_24557;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_24544)[sext_i32_i64(local_tid_24540)] =
                        x_24555;
                }
            }
            skip_waves_24560 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_24540 == 0) {
                x_acc_24553 = x_24555;
            }
        }
        if (groups_per_segment_24530 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_24540 == 0) {
                    ((__global float *) mem_23649)[sext_i32_i64(gtid_19465) *
                                                   sext_i32_i64(k2p2zq_18165 *
                                                   k2p2zq_18165) +
                                                   sext_i32_i64(gtid_19466) *
                                                   sext_i32_i64(k2p2zq_18165) +
                                                   sext_i32_i64(gtid_19467)] =
                        x_acc_24553;
                }
            }
        } else {
            int32_t old_counter_24561;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_24540 == 0) {
                    ((__global
                      float *) group_res_arr_mem_24535)[sext_i32_i64(virt_group_id_24550) *
                                                        sext_i32_i64(segred_group_sizze_19644)] =
                        x_acc_24553;
                    mem_fence_global();
                    old_counter_24561 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24537)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24551,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_24546)[0] =
                        old_counter_24561 == groups_per_segment_24530 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_24562;
            
            is_last_group_24562 = ((__local bool *) sync_arr_mem_24546)[0];
            if (is_last_group_24562) {
                if (local_tid_24540 == 0) {
                    old_counter_24561 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24537)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24551,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_24530));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_24563 =
                            sdiv_up32(groups_per_segment_24530,
                                      segred_group_sizze_19644);
                    
                    x_19648 = 0.0F;
                    for (int32_t i_24564 = 0; i_24564 < read_per_thread_24563;
                         i_24564++) {
                        int32_t group_res_id_24565 = local_tid_24540 *
                                read_per_thread_24563 + i_24564;
                        int32_t index_of_group_res_24566 =
                                flat_segment_id_24551 *
                                groups_per_segment_24530 + group_res_id_24565;
                        
                        if (slt32(group_res_id_24565,
                                  groups_per_segment_24530)) {
                            x_19649 = ((__global
                                        float *) group_res_arr_mem_24535)[sext_i32_i64(index_of_group_res_24566) *
                                                                          sext_i32_i64(segred_group_sizze_19644)];
                            
                            float res_19650;
                            
                            res_19650 = x_19648 + x_19649;
                            x_19648 = res_19650;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_24544)[sext_i32_i64(local_tid_24540)] =
                    x_19648;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_24567;
                    int32_t skip_waves_24568;
                    float x_24555;
                    float x_24556;
                    
                    offset_24567 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_24540, segred_group_sizze_19644)) {
                            x_24555 = ((__local
                                        float *) red_arr_mem_24544)[sext_i32_i64(local_tid_24540 +
                                                                    offset_24567)];
                        }
                    }
                    offset_24567 = 1;
                    while (slt32(offset_24567, wave_sizze_24542)) {
                        if (slt32(local_tid_24540 + offset_24567,
                                  segred_group_sizze_19644) &&
                            ((local_tid_24540 - squot32(local_tid_24540,
                                                        wave_sizze_24542) *
                              wave_sizze_24542) & (2 * offset_24567 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_24556 = ((volatile __local
                                            float *) red_arr_mem_24544)[sext_i32_i64(local_tid_24540 +
                                                                        offset_24567)];
                            }
                            // apply reduction operation
                            {
                                float res_24557 = x_24555 + x_24556;
                                
                                x_24555 = res_24557;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24544)[sext_i32_i64(local_tid_24540)] =
                                    x_24555;
                            }
                        }
                        offset_24567 *= 2;
                    }
                    skip_waves_24568 = 1;
                    while (slt32(skip_waves_24568,
                                 squot32(segred_group_sizze_19644 +
                                         wave_sizze_24542 - 1,
                                         wave_sizze_24542))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_24567 = skip_waves_24568 * wave_sizze_24542;
                        if (slt32(local_tid_24540 + offset_24567,
                                  segred_group_sizze_19644) &&
                            ((local_tid_24540 - squot32(local_tid_24540,
                                                        wave_sizze_24542) *
                              wave_sizze_24542) == 0 &&
                             (squot32(local_tid_24540, wave_sizze_24542) & (2 *
                                                                            skip_waves_24568 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_24556 = ((__local
                                            float *) red_arr_mem_24544)[sext_i32_i64(local_tid_24540 +
                                                                        offset_24567)];
                            }
                            // apply reduction operation
                            {
                                float res_24557 = x_24555 + x_24556;
                                
                                x_24555 = res_24557;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_24544)[sext_i32_i64(local_tid_24540)] =
                                    x_24555;
                            }
                        }
                        skip_waves_24568 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_24540 == 0) {
                            ((__global
                              float *) mem_23649)[sext_i32_i64(gtid_19465) *
                                                  sext_i32_i64(k2p2zq_18165 *
                                                  k2p2zq_18165) +
                                                  sext_i32_i64(gtid_19466) *
                                                  sext_i32_i64(k2p2zq_18165) +
                                                  sext_i32_i64(gtid_19467)] =
                                x_24555;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_19644
}
__kernel void mainzisegred_large_20730(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_24699_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24697_backing_aligned_1,
                                       int32_t N_18148, int32_t N_18150,
                                       int32_t n_18153, int32_t k2p2zq_18165,
                                       int32_t num_groups_20793, __global
                                       unsigned char *images_mem_23523, __global
                                       unsigned char *binop_p_mem_23536,
                                       __global unsigned char *mem_23878,
                                       int32_t groups_per_segment_24683,
                                       int32_t elements_per_thread_24684,
                                       int32_t virt_num_groups_24685,
                                       int32_t threads_per_segment_24687,
                                       __global
                                       unsigned char *group_res_arr_mem_24688,
                                       __global
                                       unsigned char *mainzicounter_mem_24690)
{
    #define segred_group_sizze_20792 (mainzisegred_group_sizze_20724)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_24699_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_24699_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24697_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24697_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24692;
    int32_t local_tid_24693;
    int32_t group_sizze_24696;
    int32_t wave_sizze_24695;
    int32_t group_tid_24694;
    
    global_tid_24692 = get_global_id(0);
    local_tid_24693 = get_local_id(0);
    group_sizze_24696 = get_local_size(0);
    wave_sizze_24695 = LOCKSTEP_WIDTH;
    group_tid_24694 = get_group_id(0);
    
    int32_t phys_tid_20730;
    
    phys_tid_20730 = global_tid_24692;
    
    __local char *red_arr_mem_24697;
    
    red_arr_mem_24697 = (__local char *) red_arr_mem_24697_backing_0;
    
    __local char *sync_arr_mem_24699;
    
    sync_arr_mem_24699 = (__local char *) sync_arr_mem_24699_backing_1;
    
    int32_t phys_group_id_24701;
    
    phys_group_id_24701 = get_group_id(0);
    for (int32_t i_24702 = 0; i_24702 < sdiv_up32(virt_num_groups_24685 -
                                                  phys_group_id_24701,
                                                  num_groups_20793);
         i_24702++) {
        int32_t virt_group_id_24703 = phys_group_id_24701 + i_24702 *
                num_groups_20793;
        int32_t flat_segment_id_24704 = squot32(virt_group_id_24703,
                                                groups_per_segment_24683);
        int32_t global_tid_24705 = srem32(virt_group_id_24703 *
                                          segred_group_sizze_20792 +
                                          local_tid_24693,
                                          segred_group_sizze_20792 *
                                          groups_per_segment_24683);
        int32_t gtid_20716 = squot32(flat_segment_id_24704, k2p2zq_18165);
        int32_t gtid_20717 = flat_segment_id_24704 -
                squot32(flat_segment_id_24704, k2p2zq_18165) * k2p2zq_18165;
        int32_t gtid_20729;
        float x_acc_24706;
        int32_t chunk_sizze_24707;
        
        chunk_sizze_24707 = smin32(elements_per_thread_24684,
                                   sdiv_up32(n_18153 - global_tid_24705,
                                             threads_per_segment_24687));
        
        float x_20796;
        float x_20797;
        
        // neutral-initialise the accumulators
        {
            x_acc_24706 = 0.0F;
        }
        for (int32_t i_24711 = 0; i_24711 < chunk_sizze_24707; i_24711++) {
            gtid_20729 = global_tid_24705 + threads_per_segment_24687 * i_24711;
            // apply map function
            {
                float x_20802 = ((__global
                                  float *) images_mem_23523)[sext_i32_i64(gtid_20716) *
                                                             sext_i32_i64(N_18150) +
                                                             sext_i32_i64(gtid_20729)];
                bool res_20803;
                
                res_20803 = futrts_isnan32(x_20802);
                
                float res_20804;
                
                if (res_20803) {
                    res_20804 = 0.0F;
                } else {
                    float x_20801 = ((__global
                                      float *) binop_p_mem_23536)[sext_i32_i64(gtid_20717) *
                                                                  sext_i32_i64(N_18148) +
                                                                  sext_i32_i64(gtid_20729)];
                    float res_20805 = x_20801 * x_20802;
                    
                    res_20804 = res_20805;
                }
                // save map-out results
                { }
                // load accumulator
                {
                    x_20796 = x_acc_24706;
                }
                // load new values
                {
                    x_20797 = res_20804;
                }
                // apply reduction operator
                {
                    float res_20798 = x_20796 + x_20797;
                    
                    // store in accumulator
                    {
                        x_acc_24706 = res_20798;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_20796 = x_acc_24706;
            ((__local
              float *) red_arr_mem_24697)[sext_i32_i64(local_tid_24693)] =
                x_20796;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_24712;
        int32_t skip_waves_24713;
        float x_24708;
        float x_24709;
        
        offset_24712 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_24693, segred_group_sizze_20792)) {
                x_24708 = ((__local
                            float *) red_arr_mem_24697)[sext_i32_i64(local_tid_24693 +
                                                        offset_24712)];
            }
        }
        offset_24712 = 1;
        while (slt32(offset_24712, wave_sizze_24695)) {
            if (slt32(local_tid_24693 + offset_24712,
                      segred_group_sizze_20792) && ((local_tid_24693 -
                                                     squot32(local_tid_24693,
                                                             wave_sizze_24695) *
                                                     wave_sizze_24695) & (2 *
                                                                          offset_24712 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_24709 = ((volatile __local
                                float *) red_arr_mem_24697)[sext_i32_i64(local_tid_24693 +
                                                            offset_24712)];
                }
                // apply reduction operation
                {
                    float res_24710 = x_24708 + x_24709;
                    
                    x_24708 = res_24710;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_24697)[sext_i32_i64(local_tid_24693)] =
                        x_24708;
                }
            }
            offset_24712 *= 2;
        }
        skip_waves_24713 = 1;
        while (slt32(skip_waves_24713, squot32(segred_group_sizze_20792 +
                                               wave_sizze_24695 - 1,
                                               wave_sizze_24695))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_24712 = skip_waves_24713 * wave_sizze_24695;
            if (slt32(local_tid_24693 + offset_24712,
                      segred_group_sizze_20792) && ((local_tid_24693 -
                                                     squot32(local_tid_24693,
                                                             wave_sizze_24695) *
                                                     wave_sizze_24695) == 0 &&
                                                    (squot32(local_tid_24693,
                                                             wave_sizze_24695) &
                                                     (2 * skip_waves_24713 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_24709 = ((__local
                                float *) red_arr_mem_24697)[sext_i32_i64(local_tid_24693 +
                                                            offset_24712)];
                }
                // apply reduction operation
                {
                    float res_24710 = x_24708 + x_24709;
                    
                    x_24708 = res_24710;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_24697)[sext_i32_i64(local_tid_24693)] =
                        x_24708;
                }
            }
            skip_waves_24713 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_24693 == 0) {
                x_acc_24706 = x_24708;
            }
        }
        if (groups_per_segment_24683 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_24693 == 0) {
                    ((__global float *) mem_23878)[sext_i32_i64(gtid_20716) *
                                                   sext_i32_i64(k2p2zq_18165) +
                                                   sext_i32_i64(gtid_20717)] =
                        x_acc_24706;
                }
            }
        } else {
            int32_t old_counter_24714;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_24693 == 0) {
                    ((__global
                      float *) group_res_arr_mem_24688)[sext_i32_i64(virt_group_id_24703) *
                                                        sext_i32_i64(segred_group_sizze_20792)] =
                        x_acc_24706;
                    mem_fence_global();
                    old_counter_24714 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24690)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24704,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_24699)[0] =
                        old_counter_24714 == groups_per_segment_24683 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_24715;
            
            is_last_group_24715 = ((__local bool *) sync_arr_mem_24699)[0];
            if (is_last_group_24715) {
                if (local_tid_24693 == 0) {
                    old_counter_24714 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24690)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24704,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_24683));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_24716 =
                            sdiv_up32(groups_per_segment_24683,
                                      segred_group_sizze_20792);
                    
                    x_20796 = 0.0F;
                    for (int32_t i_24717 = 0; i_24717 < read_per_thread_24716;
                         i_24717++) {
                        int32_t group_res_id_24718 = local_tid_24693 *
                                read_per_thread_24716 + i_24717;
                        int32_t index_of_group_res_24719 =
                                flat_segment_id_24704 *
                                groups_per_segment_24683 + group_res_id_24718;
                        
                        if (slt32(group_res_id_24718,
                                  groups_per_segment_24683)) {
                            x_20797 = ((__global
                                        float *) group_res_arr_mem_24688)[sext_i32_i64(index_of_group_res_24719) *
                                                                          sext_i32_i64(segred_group_sizze_20792)];
                            
                            float res_20798;
                            
                            res_20798 = x_20796 + x_20797;
                            x_20796 = res_20798;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_24697)[sext_i32_i64(local_tid_24693)] =
                    x_20796;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_24720;
                    int32_t skip_waves_24721;
                    float x_24708;
                    float x_24709;
                    
                    offset_24720 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_24693, segred_group_sizze_20792)) {
                            x_24708 = ((__local
                                        float *) red_arr_mem_24697)[sext_i32_i64(local_tid_24693 +
                                                                    offset_24720)];
                        }
                    }
                    offset_24720 = 1;
                    while (slt32(offset_24720, wave_sizze_24695)) {
                        if (slt32(local_tid_24693 + offset_24720,
                                  segred_group_sizze_20792) &&
                            ((local_tid_24693 - squot32(local_tid_24693,
                                                        wave_sizze_24695) *
                              wave_sizze_24695) & (2 * offset_24720 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_24709 = ((volatile __local
                                            float *) red_arr_mem_24697)[sext_i32_i64(local_tid_24693 +
                                                                        offset_24720)];
                            }
                            // apply reduction operation
                            {
                                float res_24710 = x_24708 + x_24709;
                                
                                x_24708 = res_24710;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24697)[sext_i32_i64(local_tid_24693)] =
                                    x_24708;
                            }
                        }
                        offset_24720 *= 2;
                    }
                    skip_waves_24721 = 1;
                    while (slt32(skip_waves_24721,
                                 squot32(segred_group_sizze_20792 +
                                         wave_sizze_24695 - 1,
                                         wave_sizze_24695))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_24720 = skip_waves_24721 * wave_sizze_24695;
                        if (slt32(local_tid_24693 + offset_24720,
                                  segred_group_sizze_20792) &&
                            ((local_tid_24693 - squot32(local_tid_24693,
                                                        wave_sizze_24695) *
                              wave_sizze_24695) == 0 &&
                             (squot32(local_tid_24693, wave_sizze_24695) & (2 *
                                                                            skip_waves_24721 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_24709 = ((__local
                                            float *) red_arr_mem_24697)[sext_i32_i64(local_tid_24693 +
                                                                        offset_24720)];
                            }
                            // apply reduction operation
                            {
                                float res_24710 = x_24708 + x_24709;
                                
                                x_24708 = res_24710;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_24697)[sext_i32_i64(local_tid_24693)] =
                                    x_24708;
                            }
                        }
                        skip_waves_24721 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_24693 == 0) {
                            ((__global
                              float *) mem_23878)[sext_i32_i64(gtid_20716) *
                                                  sext_i32_i64(k2p2zq_18165) +
                                                  sext_i32_i64(gtid_20717)] =
                                x_24708;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_20792
}
__kernel void mainzisegred_large_20886(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_24779_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24777_backing_aligned_1,
                                       int32_t k2p2zq_18165,
                                       int32_t num_groups_20945, __global
                                       unsigned char *res_mem_23770, __global
                                       unsigned char *res_mem_23886, __global
                                       unsigned char *mem_23938,
                                       int32_t groups_per_segment_24763,
                                       int32_t elements_per_thread_24764,
                                       int32_t virt_num_groups_24765,
                                       int32_t threads_per_segment_24767,
                                       __global
                                       unsigned char *group_res_arr_mem_24768,
                                       __global
                                       unsigned char *mainzicounter_mem_24770)
{
    #define segred_group_sizze_20944 (mainzisegred_group_sizze_20880)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_24779_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_24779_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24777_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24777_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24772;
    int32_t local_tid_24773;
    int32_t group_sizze_24776;
    int32_t wave_sizze_24775;
    int32_t group_tid_24774;
    
    global_tid_24772 = get_global_id(0);
    local_tid_24773 = get_local_id(0);
    group_sizze_24776 = get_local_size(0);
    wave_sizze_24775 = LOCKSTEP_WIDTH;
    group_tid_24774 = get_group_id(0);
    
    int32_t phys_tid_20886;
    
    phys_tid_20886 = global_tid_24772;
    
    __local char *red_arr_mem_24777;
    
    red_arr_mem_24777 = (__local char *) red_arr_mem_24777_backing_0;
    
    __local char *sync_arr_mem_24779;
    
    sync_arr_mem_24779 = (__local char *) sync_arr_mem_24779_backing_1;
    
    int32_t phys_group_id_24781;
    
    phys_group_id_24781 = get_group_id(0);
    for (int32_t i_24782 = 0; i_24782 < sdiv_up32(virt_num_groups_24765 -
                                                  phys_group_id_24781,
                                                  num_groups_20945);
         i_24782++) {
        int32_t virt_group_id_24783 = phys_group_id_24781 + i_24782 *
                num_groups_20945;
        int32_t flat_segment_id_24784 = squot32(virt_group_id_24783,
                                                groups_per_segment_24763);
        int32_t global_tid_24785 = srem32(virt_group_id_24783 *
                                          segred_group_sizze_20944 +
                                          local_tid_24773,
                                          segred_group_sizze_20944 *
                                          groups_per_segment_24763);
        int32_t gtid_20872 = squot32(flat_segment_id_24784, k2p2zq_18165);
        int32_t gtid_20873 = flat_segment_id_24784 -
                squot32(flat_segment_id_24784, k2p2zq_18165) * k2p2zq_18165;
        int32_t gtid_20885;
        float x_acc_24786;
        int32_t chunk_sizze_24787;
        
        chunk_sizze_24787 = smin32(elements_per_thread_24764,
                                   sdiv_up32(k2p2zq_18165 - global_tid_24785,
                                             threads_per_segment_24767));
        
        float x_20948;
        float x_20949;
        
        // neutral-initialise the accumulators
        {
            x_acc_24786 = 0.0F;
        }
        for (int32_t i_24791 = 0; i_24791 < chunk_sizze_24787; i_24791++) {
            gtid_20885 = global_tid_24785 + threads_per_segment_24767 * i_24791;
            // apply map function
            {
                float x_20954 = ((__global
                                  float *) res_mem_23886)[sext_i32_i64(gtid_20872) *
                                                          sext_i32_i64(k2p2zq_18165) +
                                                          sext_i32_i64(gtid_20885)];
                float x_20955 = ((__global
                                  float *) res_mem_23770)[sext_i32_i64(gtid_20872) *
                                                          sext_i32_i64(k2p2zq_18165 *
                                                          k2p2zq_18165) +
                                                          sext_i32_i64(gtid_20873) *
                                                          sext_i32_i64(k2p2zq_18165) +
                                                          sext_i32_i64(gtid_20885)];
                float res_20956 = x_20954 * x_20955;
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_20948 = x_acc_24786;
                }
                // load new values
                {
                    x_20949 = res_20956;
                }
                // apply reduction operator
                {
                    float res_20950 = x_20948 + x_20949;
                    
                    // store in accumulator
                    {
                        x_acc_24786 = res_20950;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_20948 = x_acc_24786;
            ((__local
              float *) red_arr_mem_24777)[sext_i32_i64(local_tid_24773)] =
                x_20948;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_24792;
        int32_t skip_waves_24793;
        float x_24788;
        float x_24789;
        
        offset_24792 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_24773, segred_group_sizze_20944)) {
                x_24788 = ((__local
                            float *) red_arr_mem_24777)[sext_i32_i64(local_tid_24773 +
                                                        offset_24792)];
            }
        }
        offset_24792 = 1;
        while (slt32(offset_24792, wave_sizze_24775)) {
            if (slt32(local_tid_24773 + offset_24792,
                      segred_group_sizze_20944) && ((local_tid_24773 -
                                                     squot32(local_tid_24773,
                                                             wave_sizze_24775) *
                                                     wave_sizze_24775) & (2 *
                                                                          offset_24792 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_24789 = ((volatile __local
                                float *) red_arr_mem_24777)[sext_i32_i64(local_tid_24773 +
                                                            offset_24792)];
                }
                // apply reduction operation
                {
                    float res_24790 = x_24788 + x_24789;
                    
                    x_24788 = res_24790;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_24777)[sext_i32_i64(local_tid_24773)] =
                        x_24788;
                }
            }
            offset_24792 *= 2;
        }
        skip_waves_24793 = 1;
        while (slt32(skip_waves_24793, squot32(segred_group_sizze_20944 +
                                               wave_sizze_24775 - 1,
                                               wave_sizze_24775))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_24792 = skip_waves_24793 * wave_sizze_24775;
            if (slt32(local_tid_24773 + offset_24792,
                      segred_group_sizze_20944) && ((local_tid_24773 -
                                                     squot32(local_tid_24773,
                                                             wave_sizze_24775) *
                                                     wave_sizze_24775) == 0 &&
                                                    (squot32(local_tid_24773,
                                                             wave_sizze_24775) &
                                                     (2 * skip_waves_24793 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_24789 = ((__local
                                float *) red_arr_mem_24777)[sext_i32_i64(local_tid_24773 +
                                                            offset_24792)];
                }
                // apply reduction operation
                {
                    float res_24790 = x_24788 + x_24789;
                    
                    x_24788 = res_24790;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_24777)[sext_i32_i64(local_tid_24773)] =
                        x_24788;
                }
            }
            skip_waves_24793 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_24773 == 0) {
                x_acc_24786 = x_24788;
            }
        }
        if (groups_per_segment_24763 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_24773 == 0) {
                    ((__global float *) mem_23938)[sext_i32_i64(gtid_20872) *
                                                   sext_i32_i64(k2p2zq_18165) +
                                                   sext_i32_i64(gtid_20873)] =
                        x_acc_24786;
                }
            }
        } else {
            int32_t old_counter_24794;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_24773 == 0) {
                    ((__global
                      float *) group_res_arr_mem_24768)[sext_i32_i64(virt_group_id_24783) *
                                                        sext_i32_i64(segred_group_sizze_20944)] =
                        x_acc_24786;
                    mem_fence_global();
                    old_counter_24794 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24770)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24784,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_24779)[0] =
                        old_counter_24794 == groups_per_segment_24763 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_24795;
            
            is_last_group_24795 = ((__local bool *) sync_arr_mem_24779)[0];
            if (is_last_group_24795) {
                if (local_tid_24773 == 0) {
                    old_counter_24794 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24770)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24784,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_24763));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_24796 =
                            sdiv_up32(groups_per_segment_24763,
                                      segred_group_sizze_20944);
                    
                    x_20948 = 0.0F;
                    for (int32_t i_24797 = 0; i_24797 < read_per_thread_24796;
                         i_24797++) {
                        int32_t group_res_id_24798 = local_tid_24773 *
                                read_per_thread_24796 + i_24797;
                        int32_t index_of_group_res_24799 =
                                flat_segment_id_24784 *
                                groups_per_segment_24763 + group_res_id_24798;
                        
                        if (slt32(group_res_id_24798,
                                  groups_per_segment_24763)) {
                            x_20949 = ((__global
                                        float *) group_res_arr_mem_24768)[sext_i32_i64(index_of_group_res_24799) *
                                                                          sext_i32_i64(segred_group_sizze_20944)];
                            
                            float res_20950;
                            
                            res_20950 = x_20948 + x_20949;
                            x_20948 = res_20950;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_24777)[sext_i32_i64(local_tid_24773)] =
                    x_20948;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_24800;
                    int32_t skip_waves_24801;
                    float x_24788;
                    float x_24789;
                    
                    offset_24800 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_24773, segred_group_sizze_20944)) {
                            x_24788 = ((__local
                                        float *) red_arr_mem_24777)[sext_i32_i64(local_tid_24773 +
                                                                    offset_24800)];
                        }
                    }
                    offset_24800 = 1;
                    while (slt32(offset_24800, wave_sizze_24775)) {
                        if (slt32(local_tid_24773 + offset_24800,
                                  segred_group_sizze_20944) &&
                            ((local_tid_24773 - squot32(local_tid_24773,
                                                        wave_sizze_24775) *
                              wave_sizze_24775) & (2 * offset_24800 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_24789 = ((volatile __local
                                            float *) red_arr_mem_24777)[sext_i32_i64(local_tid_24773 +
                                                                        offset_24800)];
                            }
                            // apply reduction operation
                            {
                                float res_24790 = x_24788 + x_24789;
                                
                                x_24788 = res_24790;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24777)[sext_i32_i64(local_tid_24773)] =
                                    x_24788;
                            }
                        }
                        offset_24800 *= 2;
                    }
                    skip_waves_24801 = 1;
                    while (slt32(skip_waves_24801,
                                 squot32(segred_group_sizze_20944 +
                                         wave_sizze_24775 - 1,
                                         wave_sizze_24775))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_24800 = skip_waves_24801 * wave_sizze_24775;
                        if (slt32(local_tid_24773 + offset_24800,
                                  segred_group_sizze_20944) &&
                            ((local_tid_24773 - squot32(local_tid_24773,
                                                        wave_sizze_24775) *
                              wave_sizze_24775) == 0 &&
                             (squot32(local_tid_24773, wave_sizze_24775) & (2 *
                                                                            skip_waves_24801 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_24789 = ((__local
                                            float *) red_arr_mem_24777)[sext_i32_i64(local_tid_24773 +
                                                                        offset_24800)];
                            }
                            // apply reduction operation
                            {
                                float res_24790 = x_24788 + x_24789;
                                
                                x_24788 = res_24790;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_24777)[sext_i32_i64(local_tid_24773)] =
                                    x_24788;
                            }
                        }
                        skip_waves_24801 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_24773 == 0) {
                            ((__global
                              float *) mem_23938)[sext_i32_i64(gtid_20872) *
                                                  sext_i32_i64(k2p2zq_18165) +
                                                  sext_i32_i64(gtid_20873)] =
                                x_24788;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_20944
}
__kernel void mainzisegred_large_21035(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_24866_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_24864_backing_aligned_1,
                                       int32_t N_18148, int32_t k2p2zq_18165,
                                       int32_t num_groups_21092, __global
                                       unsigned char *mem_23547, __global
                                       unsigned char *res_mem_23946, __global
                                       unsigned char *mem_24059,
                                       int32_t groups_per_segment_24850,
                                       int32_t elements_per_thread_24851,
                                       int32_t virt_num_groups_24852,
                                       int32_t threads_per_segment_24854,
                                       __global
                                       unsigned char *group_res_arr_mem_24855,
                                       __global
                                       unsigned char *mainzicounter_mem_24857)
{
    #define segred_group_sizze_21091 (mainzisegred_group_sizze_21029)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_24866_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_24866_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_24864_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24864_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24859;
    int32_t local_tid_24860;
    int32_t group_sizze_24863;
    int32_t wave_sizze_24862;
    int32_t group_tid_24861;
    
    global_tid_24859 = get_global_id(0);
    local_tid_24860 = get_local_id(0);
    group_sizze_24863 = get_local_size(0);
    wave_sizze_24862 = LOCKSTEP_WIDTH;
    group_tid_24861 = get_group_id(0);
    
    int32_t phys_tid_21035;
    
    phys_tid_21035 = global_tid_24859;
    
    __local char *red_arr_mem_24864;
    
    red_arr_mem_24864 = (__local char *) red_arr_mem_24864_backing_0;
    
    __local char *sync_arr_mem_24866;
    
    sync_arr_mem_24866 = (__local char *) sync_arr_mem_24866_backing_1;
    
    int32_t phys_group_id_24868;
    
    phys_group_id_24868 = get_group_id(0);
    for (int32_t i_24869 = 0; i_24869 < sdiv_up32(virt_num_groups_24852 -
                                                  phys_group_id_24868,
                                                  num_groups_21092);
         i_24869++) {
        int32_t virt_group_id_24870 = phys_group_id_24868 + i_24869 *
                num_groups_21092;
        int32_t flat_segment_id_24871 = squot32(virt_group_id_24870,
                                                groups_per_segment_24850);
        int32_t global_tid_24872 = srem32(virt_group_id_24870 *
                                          segred_group_sizze_21091 +
                                          local_tid_24860,
                                          segred_group_sizze_21091 *
                                          groups_per_segment_24850);
        int32_t gtid_21021 = squot32(flat_segment_id_24871, N_18148);
        int32_t gtid_21022 = flat_segment_id_24871 -
                squot32(flat_segment_id_24871, N_18148) * N_18148;
        int32_t gtid_21034;
        float x_acc_24873;
        int32_t chunk_sizze_24874;
        
        chunk_sizze_24874 = smin32(elements_per_thread_24851,
                                   sdiv_up32(k2p2zq_18165 - global_tid_24872,
                                             threads_per_segment_24854));
        
        float x_21095;
        float x_21096;
        
        // neutral-initialise the accumulators
        {
            x_acc_24873 = 0.0F;
        }
        for (int32_t i_24878 = 0; i_24878 < chunk_sizze_24874; i_24878++) {
            gtid_21034 = global_tid_24872 + threads_per_segment_24854 * i_24878;
            // apply map function
            {
                float x_21100 = ((__global
                                  float *) res_mem_23946)[sext_i32_i64(gtid_21021) *
                                                          sext_i32_i64(k2p2zq_18165) +
                                                          sext_i32_i64(gtid_21034)];
                float x_21101 = ((__global
                                  float *) mem_23547)[sext_i32_i64(gtid_21022) *
                                                      sext_i32_i64(k2p2zq_18165) +
                                                      sext_i32_i64(gtid_21034)];
                float res_21102 = x_21100 * x_21101;
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_21095 = x_acc_24873;
                }
                // load new values
                {
                    x_21096 = res_21102;
                }
                // apply reduction operator
                {
                    float res_21097 = x_21095 + x_21096;
                    
                    // store in accumulator
                    {
                        x_acc_24873 = res_21097;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_21095 = x_acc_24873;
            ((__local
              float *) red_arr_mem_24864)[sext_i32_i64(local_tid_24860)] =
                x_21095;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_24879;
        int32_t skip_waves_24880;
        float x_24875;
        float x_24876;
        
        offset_24879 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_24860, segred_group_sizze_21091)) {
                x_24875 = ((__local
                            float *) red_arr_mem_24864)[sext_i32_i64(local_tid_24860 +
                                                        offset_24879)];
            }
        }
        offset_24879 = 1;
        while (slt32(offset_24879, wave_sizze_24862)) {
            if (slt32(local_tid_24860 + offset_24879,
                      segred_group_sizze_21091) && ((local_tid_24860 -
                                                     squot32(local_tid_24860,
                                                             wave_sizze_24862) *
                                                     wave_sizze_24862) & (2 *
                                                                          offset_24879 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_24876 = ((volatile __local
                                float *) red_arr_mem_24864)[sext_i32_i64(local_tid_24860 +
                                                            offset_24879)];
                }
                // apply reduction operation
                {
                    float res_24877 = x_24875 + x_24876;
                    
                    x_24875 = res_24877;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_24864)[sext_i32_i64(local_tid_24860)] =
                        x_24875;
                }
            }
            offset_24879 *= 2;
        }
        skip_waves_24880 = 1;
        while (slt32(skip_waves_24880, squot32(segred_group_sizze_21091 +
                                               wave_sizze_24862 - 1,
                                               wave_sizze_24862))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_24879 = skip_waves_24880 * wave_sizze_24862;
            if (slt32(local_tid_24860 + offset_24879,
                      segred_group_sizze_21091) && ((local_tid_24860 -
                                                     squot32(local_tid_24860,
                                                             wave_sizze_24862) *
                                                     wave_sizze_24862) == 0 &&
                                                    (squot32(local_tid_24860,
                                                             wave_sizze_24862) &
                                                     (2 * skip_waves_24880 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_24876 = ((__local
                                float *) red_arr_mem_24864)[sext_i32_i64(local_tid_24860 +
                                                            offset_24879)];
                }
                // apply reduction operation
                {
                    float res_24877 = x_24875 + x_24876;
                    
                    x_24875 = res_24877;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_24864)[sext_i32_i64(local_tid_24860)] =
                        x_24875;
                }
            }
            skip_waves_24880 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_24860 == 0) {
                x_acc_24873 = x_24875;
            }
        }
        if (groups_per_segment_24850 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_24860 == 0) {
                    ((__global float *) mem_24059)[sext_i32_i64(gtid_21021) *
                                                   sext_i32_i64(N_18148) +
                                                   sext_i32_i64(gtid_21022)] =
                        x_acc_24873;
                }
            }
        } else {
            int32_t old_counter_24881;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_24860 == 0) {
                    ((__global
                      float *) group_res_arr_mem_24855)[sext_i32_i64(virt_group_id_24870) *
                                                        sext_i32_i64(segred_group_sizze_21091)] =
                        x_acc_24873;
                    mem_fence_global();
                    old_counter_24881 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24857)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24871,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_24866)[0] =
                        old_counter_24881 == groups_per_segment_24850 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_24882;
            
            is_last_group_24882 = ((__local bool *) sync_arr_mem_24866)[0];
            if (is_last_group_24882) {
                if (local_tid_24860 == 0) {
                    old_counter_24881 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_24857)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_24871,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_24850));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_24883 =
                            sdiv_up32(groups_per_segment_24850,
                                      segred_group_sizze_21091);
                    
                    x_21095 = 0.0F;
                    for (int32_t i_24884 = 0; i_24884 < read_per_thread_24883;
                         i_24884++) {
                        int32_t group_res_id_24885 = local_tid_24860 *
                                read_per_thread_24883 + i_24884;
                        int32_t index_of_group_res_24886 =
                                flat_segment_id_24871 *
                                groups_per_segment_24850 + group_res_id_24885;
                        
                        if (slt32(group_res_id_24885,
                                  groups_per_segment_24850)) {
                            x_21096 = ((__global
                                        float *) group_res_arr_mem_24855)[sext_i32_i64(index_of_group_res_24886) *
                                                                          sext_i32_i64(segred_group_sizze_21091)];
                            
                            float res_21097;
                            
                            res_21097 = x_21095 + x_21096;
                            x_21095 = res_21097;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_24864)[sext_i32_i64(local_tid_24860)] =
                    x_21095;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_24887;
                    int32_t skip_waves_24888;
                    float x_24875;
                    float x_24876;
                    
                    offset_24887 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_24860, segred_group_sizze_21091)) {
                            x_24875 = ((__local
                                        float *) red_arr_mem_24864)[sext_i32_i64(local_tid_24860 +
                                                                    offset_24887)];
                        }
                    }
                    offset_24887 = 1;
                    while (slt32(offset_24887, wave_sizze_24862)) {
                        if (slt32(local_tid_24860 + offset_24887,
                                  segred_group_sizze_21091) &&
                            ((local_tid_24860 - squot32(local_tid_24860,
                                                        wave_sizze_24862) *
                              wave_sizze_24862) & (2 * offset_24887 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_24876 = ((volatile __local
                                            float *) red_arr_mem_24864)[sext_i32_i64(local_tid_24860 +
                                                                        offset_24887)];
                            }
                            // apply reduction operation
                            {
                                float res_24877 = x_24875 + x_24876;
                                
                                x_24875 = res_24877;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24864)[sext_i32_i64(local_tid_24860)] =
                                    x_24875;
                            }
                        }
                        offset_24887 *= 2;
                    }
                    skip_waves_24888 = 1;
                    while (slt32(skip_waves_24888,
                                 squot32(segred_group_sizze_21091 +
                                         wave_sizze_24862 - 1,
                                         wave_sizze_24862))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_24887 = skip_waves_24888 * wave_sizze_24862;
                        if (slt32(local_tid_24860 + offset_24887,
                                  segred_group_sizze_21091) &&
                            ((local_tid_24860 - squot32(local_tid_24860,
                                                        wave_sizze_24862) *
                              wave_sizze_24862) == 0 &&
                             (squot32(local_tid_24860, wave_sizze_24862) & (2 *
                                                                            skip_waves_24888 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_24876 = ((__local
                                            float *) red_arr_mem_24864)[sext_i32_i64(local_tid_24860 +
                                                                        offset_24887)];
                            }
                            // apply reduction operation
                            {
                                float res_24877 = x_24875 + x_24876;
                                
                                x_24875 = res_24877;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_24864)[sext_i32_i64(local_tid_24860)] =
                                    x_24875;
                            }
                        }
                        skip_waves_24888 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_24860 == 0) {
                            ((__global
                              float *) mem_24059)[sext_i32_i64(gtid_21021) *
                                                  sext_i32_i64(N_18148) +
                                                  sext_i32_i64(gtid_21022)] =
                                x_24875;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21091
}
__kernel void mainzisegred_large_21614(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_25109_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_25107_backing_aligned_1,
                                       int32_t N_18148, int32_t n_18153,
                                       int32_t num_groups_21666, __global
                                       unsigned char *res_mem_24122, __global
                                       unsigned char *mem_24159, __global
                                       unsigned char *mem_24163,
                                       int32_t groups_per_segment_25093,
                                       int32_t elements_per_thread_25094,
                                       int32_t virt_num_groups_25095,
                                       int32_t threads_per_segment_25097,
                                       __global
                                       unsigned char *group_res_arr_mem_25098,
                                       __global
                                       unsigned char *mainzicounter_mem_25100)
{
    #define segred_group_sizze_21665 (mainzisegred_group_sizze_21608)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_25109_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_25109_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_25107_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_25107_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25102;
    int32_t local_tid_25103;
    int32_t group_sizze_25106;
    int32_t wave_sizze_25105;
    int32_t group_tid_25104;
    
    global_tid_25102 = get_global_id(0);
    local_tid_25103 = get_local_id(0);
    group_sizze_25106 = get_local_size(0);
    wave_sizze_25105 = LOCKSTEP_WIDTH;
    group_tid_25104 = get_group_id(0);
    
    int32_t phys_tid_21614;
    
    phys_tid_21614 = global_tid_25102;
    
    __local char *red_arr_mem_25107;
    
    red_arr_mem_25107 = (__local char *) red_arr_mem_25107_backing_0;
    
    __local char *sync_arr_mem_25109;
    
    sync_arr_mem_25109 = (__local char *) sync_arr_mem_25109_backing_1;
    
    int32_t phys_group_id_25111;
    
    phys_group_id_25111 = get_group_id(0);
    for (int32_t i_25112 = 0; i_25112 < sdiv_up32(virt_num_groups_25095 -
                                                  phys_group_id_25111,
                                                  num_groups_21666);
         i_25112++) {
        int32_t virt_group_id_25113 = phys_group_id_25111 + i_25112 *
                num_groups_21666;
        int32_t flat_segment_id_25114 = squot32(virt_group_id_25113,
                                                groups_per_segment_25093);
        int32_t global_tid_25115 = srem32(virt_group_id_25113 *
                                          segred_group_sizze_21665 +
                                          local_tid_25103,
                                          segred_group_sizze_21665 *
                                          groups_per_segment_25093);
        int32_t gtid_21603 = flat_segment_id_25114;
        int32_t gtid_21613;
        float x_acc_25116;
        int32_t chunk_sizze_25117;
        
        chunk_sizze_25117 = smin32(elements_per_thread_25094,
                                   sdiv_up32(n_18153 - global_tid_25115,
                                             threads_per_segment_25097));
        
        float x_21669;
        float x_21670;
        
        // neutral-initialise the accumulators
        {
            x_acc_25116 = 0.0F;
        }
        for (int32_t i_25121 = 0; i_25121 < chunk_sizze_25117; i_25121++) {
            gtid_21613 = global_tid_25115 + threads_per_segment_25097 * i_25121;
            // apply map function
            {
                int32_t res_21673 = ((__global
                                      int32_t *) mem_24159)[sext_i32_i64(gtid_21603)];
                bool cond_21676 = slt32(gtid_21613, res_21673);
                float res_21677;
                
                if (cond_21676) {
                    float x_elem_21675 = ((__global
                                           float *) res_mem_24122)[sext_i32_i64(gtid_21603) *
                                                                   sext_i32_i64(N_18148) +
                                                                   sext_i32_i64(gtid_21613)];
                    
                    res_21677 = x_elem_21675;
                } else {
                    res_21677 = 0.0F;
                }
                
                float res_21678 = res_21677 * res_21677;
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_21669 = x_acc_25116;
                }
                // load new values
                {
                    x_21670 = res_21678;
                }
                // apply reduction operator
                {
                    float res_21671 = x_21669 + x_21670;
                    
                    // store in accumulator
                    {
                        x_acc_25116 = res_21671;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_21669 = x_acc_25116;
            ((__local
              float *) red_arr_mem_25107)[sext_i32_i64(local_tid_25103)] =
                x_21669;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_25122;
        int32_t skip_waves_25123;
        float x_25118;
        float x_25119;
        
        offset_25122 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_25103, segred_group_sizze_21665)) {
                x_25118 = ((__local
                            float *) red_arr_mem_25107)[sext_i32_i64(local_tid_25103 +
                                                        offset_25122)];
            }
        }
        offset_25122 = 1;
        while (slt32(offset_25122, wave_sizze_25105)) {
            if (slt32(local_tid_25103 + offset_25122,
                      segred_group_sizze_21665) && ((local_tid_25103 -
                                                     squot32(local_tid_25103,
                                                             wave_sizze_25105) *
                                                     wave_sizze_25105) & (2 *
                                                                          offset_25122 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_25119 = ((volatile __local
                                float *) red_arr_mem_25107)[sext_i32_i64(local_tid_25103 +
                                                            offset_25122)];
                }
                // apply reduction operation
                {
                    float res_25120 = x_25118 + x_25119;
                    
                    x_25118 = res_25120;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_25107)[sext_i32_i64(local_tid_25103)] =
                        x_25118;
                }
            }
            offset_25122 *= 2;
        }
        skip_waves_25123 = 1;
        while (slt32(skip_waves_25123, squot32(segred_group_sizze_21665 +
                                               wave_sizze_25105 - 1,
                                               wave_sizze_25105))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_25122 = skip_waves_25123 * wave_sizze_25105;
            if (slt32(local_tid_25103 + offset_25122,
                      segred_group_sizze_21665) && ((local_tid_25103 -
                                                     squot32(local_tid_25103,
                                                             wave_sizze_25105) *
                                                     wave_sizze_25105) == 0 &&
                                                    (squot32(local_tid_25103,
                                                             wave_sizze_25105) &
                                                     (2 * skip_waves_25123 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_25119 = ((__local
                                float *) red_arr_mem_25107)[sext_i32_i64(local_tid_25103 +
                                                            offset_25122)];
                }
                // apply reduction operation
                {
                    float res_25120 = x_25118 + x_25119;
                    
                    x_25118 = res_25120;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_25107)[sext_i32_i64(local_tid_25103)] =
                        x_25118;
                }
            }
            skip_waves_25123 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_25103 == 0) {
                x_acc_25116 = x_25118;
            }
        }
        if (groups_per_segment_25093 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_25103 == 0) {
                    ((__global float *) mem_24163)[sext_i32_i64(gtid_21603)] =
                        x_acc_25116;
                }
            }
        } else {
            int32_t old_counter_25124;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_25103 == 0) {
                    ((__global
                      float *) group_res_arr_mem_25098)[sext_i32_i64(virt_group_id_25113) *
                                                        sext_i32_i64(segred_group_sizze_21665)] =
                        x_acc_25116;
                    mem_fence_global();
                    old_counter_25124 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_25100)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_25114,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_25109)[0] =
                        old_counter_25124 == groups_per_segment_25093 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_25125;
            
            is_last_group_25125 = ((__local bool *) sync_arr_mem_25109)[0];
            if (is_last_group_25125) {
                if (local_tid_25103 == 0) {
                    old_counter_25124 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_25100)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_25114,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_25093));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_25126 =
                            sdiv_up32(groups_per_segment_25093,
                                      segred_group_sizze_21665);
                    
                    x_21669 = 0.0F;
                    for (int32_t i_25127 = 0; i_25127 < read_per_thread_25126;
                         i_25127++) {
                        int32_t group_res_id_25128 = local_tid_25103 *
                                read_per_thread_25126 + i_25127;
                        int32_t index_of_group_res_25129 =
                                flat_segment_id_25114 *
                                groups_per_segment_25093 + group_res_id_25128;
                        
                        if (slt32(group_res_id_25128,
                                  groups_per_segment_25093)) {
                            x_21670 = ((__global
                                        float *) group_res_arr_mem_25098)[sext_i32_i64(index_of_group_res_25129) *
                                                                          sext_i32_i64(segred_group_sizze_21665)];
                            
                            float res_21671;
                            
                            res_21671 = x_21669 + x_21670;
                            x_21669 = res_21671;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_25107)[sext_i32_i64(local_tid_25103)] =
                    x_21669;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_25130;
                    int32_t skip_waves_25131;
                    float x_25118;
                    float x_25119;
                    
                    offset_25130 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_25103, segred_group_sizze_21665)) {
                            x_25118 = ((__local
                                        float *) red_arr_mem_25107)[sext_i32_i64(local_tid_25103 +
                                                                    offset_25130)];
                        }
                    }
                    offset_25130 = 1;
                    while (slt32(offset_25130, wave_sizze_25105)) {
                        if (slt32(local_tid_25103 + offset_25130,
                                  segred_group_sizze_21665) &&
                            ((local_tid_25103 - squot32(local_tid_25103,
                                                        wave_sizze_25105) *
                              wave_sizze_25105) & (2 * offset_25130 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_25119 = ((volatile __local
                                            float *) red_arr_mem_25107)[sext_i32_i64(local_tid_25103 +
                                                                        offset_25130)];
                            }
                            // apply reduction operation
                            {
                                float res_25120 = x_25118 + x_25119;
                                
                                x_25118 = res_25120;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_25107)[sext_i32_i64(local_tid_25103)] =
                                    x_25118;
                            }
                        }
                        offset_25130 *= 2;
                    }
                    skip_waves_25131 = 1;
                    while (slt32(skip_waves_25131,
                                 squot32(segred_group_sizze_21665 +
                                         wave_sizze_25105 - 1,
                                         wave_sizze_25105))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_25130 = skip_waves_25131 * wave_sizze_25105;
                        if (slt32(local_tid_25103 + offset_25130,
                                  segred_group_sizze_21665) &&
                            ((local_tid_25103 - squot32(local_tid_25103,
                                                        wave_sizze_25105) *
                              wave_sizze_25105) == 0 &&
                             (squot32(local_tid_25103, wave_sizze_25105) & (2 *
                                                                            skip_waves_25131 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_25119 = ((__local
                                            float *) red_arr_mem_25107)[sext_i32_i64(local_tid_25103 +
                                                                        offset_25130)];
                            }
                            // apply reduction operation
                            {
                                float res_25120 = x_25118 + x_25119;
                                
                                x_25118 = res_25120;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_25107)[sext_i32_i64(local_tid_25103)] =
                                    x_25118;
                            }
                        }
                        skip_waves_25131 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_25103 == 0) {
                            ((__global
                              float *) mem_24163)[sext_i32_i64(gtid_21603)] =
                                x_25118;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21665
}
__kernel void mainzisegred_large_21636(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_25050_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_25048_backing_aligned_1,
                                       int32_t N_18150, int32_t n_18153,
                                       int32_t num_groups_21650, __global
                                       unsigned char *images_mem_23523, __global
                                       unsigned char *mem_24159,
                                       int32_t groups_per_segment_25034,
                                       int32_t elements_per_thread_25035,
                                       int32_t virt_num_groups_25036,
                                       int32_t threads_per_segment_25038,
                                       __global
                                       unsigned char *group_res_arr_mem_25039,
                                       __global
                                       unsigned char *mainzicounter_mem_25041)
{
    #define segred_group_sizze_21649 (mainzisegred_group_sizze_21630)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_25050_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_25050_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_25048_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_25048_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25043;
    int32_t local_tid_25044;
    int32_t group_sizze_25047;
    int32_t wave_sizze_25046;
    int32_t group_tid_25045;
    
    global_tid_25043 = get_global_id(0);
    local_tid_25044 = get_local_id(0);
    group_sizze_25047 = get_local_size(0);
    wave_sizze_25046 = LOCKSTEP_WIDTH;
    group_tid_25045 = get_group_id(0);
    
    int32_t phys_tid_21636;
    
    phys_tid_21636 = global_tid_25043;
    
    __local char *red_arr_mem_25048;
    
    red_arr_mem_25048 = (__local char *) red_arr_mem_25048_backing_0;
    
    __local char *sync_arr_mem_25050;
    
    sync_arr_mem_25050 = (__local char *) sync_arr_mem_25050_backing_1;
    
    int32_t phys_group_id_25052;
    
    phys_group_id_25052 = get_group_id(0);
    for (int32_t i_25053 = 0; i_25053 < sdiv_up32(virt_num_groups_25036 -
                                                  phys_group_id_25052,
                                                  num_groups_21650);
         i_25053++) {
        int32_t virt_group_id_25054 = phys_group_id_25052 + i_25053 *
                num_groups_21650;
        int32_t flat_segment_id_25055 = squot32(virt_group_id_25054,
                                                groups_per_segment_25034);
        int32_t global_tid_25056 = srem32(virt_group_id_25054 *
                                          segred_group_sizze_21649 +
                                          local_tid_25044,
                                          segred_group_sizze_21649 *
                                          groups_per_segment_25034);
        int32_t gtid_21625 = flat_segment_id_25055;
        int32_t gtid_21635;
        int32_t x_acc_25057;
        int32_t chunk_sizze_25058;
        
        chunk_sizze_25058 = smin32(elements_per_thread_25035,
                                   sdiv_up32(n_18153 - global_tid_25056,
                                             threads_per_segment_25038));
        
        int32_t x_21653;
        int32_t x_21654;
        
        // neutral-initialise the accumulators
        {
            x_acc_25057 = 0;
        }
        for (int32_t i_25062 = 0; i_25062 < chunk_sizze_25058; i_25062++) {
            gtid_21635 = global_tid_25056 + threads_per_segment_25038 * i_25062;
            // apply map function
            {
                float x_21657 = ((__global
                                  float *) images_mem_23523)[sext_i32_i64(gtid_21625) *
                                                             sext_i32_i64(N_18150) +
                                                             sext_i32_i64(gtid_21635)];
                bool res_21658;
                
                res_21658 = futrts_isnan32(x_21657);
                
                bool cond_21659 = !res_21658;
                int32_t res_21660 = btoi_bool_i32(cond_21659);
                
                // save map-out results
                { }
                // load accumulator
                {
                    x_21653 = x_acc_25057;
                }
                // load new values
                {
                    x_21654 = res_21660;
                }
                // apply reduction operator
                {
                    int32_t res_21655 = add32(x_21653, x_21654);
                    
                    // store in accumulator
                    {
                        x_acc_25057 = res_21655;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_21653 = x_acc_25057;
            ((__local
              int32_t *) red_arr_mem_25048)[sext_i32_i64(local_tid_25044)] =
                x_21653;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_25063;
        int32_t skip_waves_25064;
        int32_t x_25059;
        int32_t x_25060;
        
        offset_25063 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_25044, segred_group_sizze_21649)) {
                x_25059 = ((__local
                            int32_t *) red_arr_mem_25048)[sext_i32_i64(local_tid_25044 +
                                                          offset_25063)];
            }
        }
        offset_25063 = 1;
        while (slt32(offset_25063, wave_sizze_25046)) {
            if (slt32(local_tid_25044 + offset_25063,
                      segred_group_sizze_21649) && ((local_tid_25044 -
                                                     squot32(local_tid_25044,
                                                             wave_sizze_25046) *
                                                     wave_sizze_25046) & (2 *
                                                                          offset_25063 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_25060 = ((volatile __local
                                int32_t *) red_arr_mem_25048)[sext_i32_i64(local_tid_25044 +
                                                              offset_25063)];
                }
                // apply reduction operation
                {
                    int32_t res_25061 = add32(x_25059, x_25060);
                    
                    x_25059 = res_25061;
                }
                // write result of operation
                {
                    ((volatile __local
                      int32_t *) red_arr_mem_25048)[sext_i32_i64(local_tid_25044)] =
                        x_25059;
                }
            }
            offset_25063 *= 2;
        }
        skip_waves_25064 = 1;
        while (slt32(skip_waves_25064, squot32(segred_group_sizze_21649 +
                                               wave_sizze_25046 - 1,
                                               wave_sizze_25046))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_25063 = skip_waves_25064 * wave_sizze_25046;
            if (slt32(local_tid_25044 + offset_25063,
                      segred_group_sizze_21649) && ((local_tid_25044 -
                                                     squot32(local_tid_25044,
                                                             wave_sizze_25046) *
                                                     wave_sizze_25046) == 0 &&
                                                    (squot32(local_tid_25044,
                                                             wave_sizze_25046) &
                                                     (2 * skip_waves_25064 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_25060 = ((__local
                                int32_t *) red_arr_mem_25048)[sext_i32_i64(local_tid_25044 +
                                                              offset_25063)];
                }
                // apply reduction operation
                {
                    int32_t res_25061 = add32(x_25059, x_25060);
                    
                    x_25059 = res_25061;
                }
                // write result of operation
                {
                    ((__local
                      int32_t *) red_arr_mem_25048)[sext_i32_i64(local_tid_25044)] =
                        x_25059;
                }
            }
            skip_waves_25064 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_25044 == 0) {
                x_acc_25057 = x_25059;
            }
        }
        if (groups_per_segment_25034 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_25044 == 0) {
                    ((__global int32_t *) mem_24159)[sext_i32_i64(gtid_21625)] =
                        x_acc_25057;
                }
            }
        } else {
            int32_t old_counter_25065;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_25044 == 0) {
                    ((__global
                      int32_t *) group_res_arr_mem_25039)[sext_i32_i64(virt_group_id_25054) *
                                                          sext_i32_i64(segred_group_sizze_21649)] =
                        x_acc_25057;
                    mem_fence_global();
                    old_counter_25065 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_25041)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_25055,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_25050)[0] =
                        old_counter_25065 == groups_per_segment_25034 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_25066;
            
            is_last_group_25066 = ((__local bool *) sync_arr_mem_25050)[0];
            if (is_last_group_25066) {
                if (local_tid_25044 == 0) {
                    old_counter_25065 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_25041)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_25055,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_25034));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_25067 =
                            sdiv_up32(groups_per_segment_25034,
                                      segred_group_sizze_21649);
                    
                    x_21653 = 0;
                    for (int32_t i_25068 = 0; i_25068 < read_per_thread_25067;
                         i_25068++) {
                        int32_t group_res_id_25069 = local_tid_25044 *
                                read_per_thread_25067 + i_25068;
                        int32_t index_of_group_res_25070 =
                                flat_segment_id_25055 *
                                groups_per_segment_25034 + group_res_id_25069;
                        
                        if (slt32(group_res_id_25069,
                                  groups_per_segment_25034)) {
                            x_21654 = ((__global
                                        int32_t *) group_res_arr_mem_25039)[sext_i32_i64(index_of_group_res_25070) *
                                                                            sext_i32_i64(segred_group_sizze_21649)];
                            
                            int32_t res_21655;
                            
                            res_21655 = add32(x_21653, x_21654);
                            x_21653 = res_21655;
                        }
                    }
                }
                ((__local
                  int32_t *) red_arr_mem_25048)[sext_i32_i64(local_tid_25044)] =
                    x_21653;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_25071;
                    int32_t skip_waves_25072;
                    int32_t x_25059;
                    int32_t x_25060;
                    
                    offset_25071 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_25044, segred_group_sizze_21649)) {
                            x_25059 = ((__local
                                        int32_t *) red_arr_mem_25048)[sext_i32_i64(local_tid_25044 +
                                                                      offset_25071)];
                        }
                    }
                    offset_25071 = 1;
                    while (slt32(offset_25071, wave_sizze_25046)) {
                        if (slt32(local_tid_25044 + offset_25071,
                                  segred_group_sizze_21649) &&
                            ((local_tid_25044 - squot32(local_tid_25044,
                                                        wave_sizze_25046) *
                              wave_sizze_25046) & (2 * offset_25071 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_25060 = ((volatile __local
                                            int32_t *) red_arr_mem_25048)[sext_i32_i64(local_tid_25044 +
                                                                          offset_25071)];
                            }
                            // apply reduction operation
                            {
                                int32_t res_25061 = add32(x_25059, x_25060);
                                
                                x_25059 = res_25061;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  int32_t *) red_arr_mem_25048)[sext_i32_i64(local_tid_25044)] =
                                    x_25059;
                            }
                        }
                        offset_25071 *= 2;
                    }
                    skip_waves_25072 = 1;
                    while (slt32(skip_waves_25072,
                                 squot32(segred_group_sizze_21649 +
                                         wave_sizze_25046 - 1,
                                         wave_sizze_25046))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_25071 = skip_waves_25072 * wave_sizze_25046;
                        if (slt32(local_tid_25044 + offset_25071,
                                  segred_group_sizze_21649) &&
                            ((local_tid_25044 - squot32(local_tid_25044,
                                                        wave_sizze_25046) *
                              wave_sizze_25046) == 0 &&
                             (squot32(local_tid_25044, wave_sizze_25046) & (2 *
                                                                            skip_waves_25072 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_25060 = ((__local
                                            int32_t *) red_arr_mem_25048)[sext_i32_i64(local_tid_25044 +
                                                                          offset_25071)];
                            }
                            // apply reduction operation
                            {
                                int32_t res_25061 = add32(x_25059, x_25060);
                                
                                x_25059 = res_25061;
                            }
                            // write result of operation
                            {
                                ((__local
                                  int32_t *) red_arr_mem_25048)[sext_i32_i64(local_tid_25044)] =
                                    x_25059;
                            }
                        }
                        skip_waves_25072 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_25044 == 0) {
                            ((__global
                              int32_t *) mem_24159)[sext_i32_i64(gtid_21625)] =
                                x_25059;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21649
}
__kernel void mainzisegred_large_21770(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_25213_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_25211_backing_aligned_1,
                                       int32_t N_18148, int32_t res_18475,
                                       int32_t num_groups_21789, __global
                                       unsigned char *res_mem_24122, __global
                                       unsigned char *res_mem_24174, __global
                                       unsigned char *res_mem_24175, __global
                                       unsigned char *mem_24187,
                                       int32_t groups_per_segment_25197,
                                       int32_t elements_per_thread_25198,
                                       int32_t virt_num_groups_25199,
                                       int32_t threads_per_segment_25201,
                                       __global
                                       unsigned char *group_res_arr_mem_25202,
                                       __global
                                       unsigned char *mainzicounter_mem_25204)
{
    #define segred_group_sizze_21788 (mainzisegred_group_sizze_21764)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_25213_backing_1 =
                          (__local volatile
                           char *) sync_arr_mem_25213_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_25211_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_25211_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25206;
    int32_t local_tid_25207;
    int32_t group_sizze_25210;
    int32_t wave_sizze_25209;
    int32_t group_tid_25208;
    
    global_tid_25206 = get_global_id(0);
    local_tid_25207 = get_local_id(0);
    group_sizze_25210 = get_local_size(0);
    wave_sizze_25209 = LOCKSTEP_WIDTH;
    group_tid_25208 = get_group_id(0);
    
    int32_t phys_tid_21770;
    
    phys_tid_21770 = global_tid_25206;
    
    __local char *red_arr_mem_25211;
    
    red_arr_mem_25211 = (__local char *) red_arr_mem_25211_backing_0;
    
    __local char *sync_arr_mem_25213;
    
    sync_arr_mem_25213 = (__local char *) sync_arr_mem_25213_backing_1;
    
    int32_t phys_group_id_25215;
    
    phys_group_id_25215 = get_group_id(0);
    for (int32_t i_25216 = 0; i_25216 < sdiv_up32(virt_num_groups_25199 -
                                                  phys_group_id_25215,
                                                  num_groups_21789);
         i_25216++) {
        int32_t virt_group_id_25217 = phys_group_id_25215 + i_25216 *
                num_groups_21789;
        int32_t flat_segment_id_25218 = squot32(virt_group_id_25217,
                                                groups_per_segment_25197);
        int32_t global_tid_25219 = srem32(virt_group_id_25217 *
                                          segred_group_sizze_21788 +
                                          local_tid_25207,
                                          segred_group_sizze_21788 *
                                          groups_per_segment_25197);
        int32_t gtid_21759 = flat_segment_id_25218;
        int32_t gtid_21769;
        float x_acc_25220;
        int32_t chunk_sizze_25221;
        
        chunk_sizze_25221 = smin32(elements_per_thread_25198,
                                   sdiv_up32(res_18475 - global_tid_25219,
                                             threads_per_segment_25201));
        
        float x_21792;
        float x_21793;
        
        // neutral-initialise the accumulators
        {
            x_acc_25220 = 0.0F;
        }
        for (int32_t i_25225 = 0; i_25225 < chunk_sizze_25221; i_25225++) {
            gtid_21769 = global_tid_25219 + threads_per_segment_25201 * i_25225;
            // apply map function
            {
                int32_t x_21797 = ((__global
                                    int32_t *) res_mem_24174)[sext_i32_i64(gtid_21759)];
                bool cond_21799 = slt32(gtid_21769, x_21797);
                float res_21800;
                
                if (cond_21799) {
                    int32_t x_21796 = ((__global
                                        int32_t *) res_mem_24175)[sext_i32_i64(gtid_21759)];
                    int32_t x_21801 = add32(gtid_21769, x_21796);
                    int32_t x_21802 = sub32(x_21801, x_21797);
                    int32_t i_21803 = add32(1, x_21802);
                    float res_21804 = ((__global
                                        float *) res_mem_24122)[sext_i32_i64(gtid_21759) *
                                                                sext_i32_i64(N_18148) +
                                                                sext_i32_i64(i_21803)];
                    
                    res_21800 = res_21804;
                } else {
                    res_21800 = 0.0F;
                }
                // save map-out results
                { }
                // load accumulator
                {
                    x_21792 = x_acc_25220;
                }
                // load new values
                {
                    x_21793 = res_21800;
                }
                // apply reduction operator
                {
                    float res_21794 = x_21792 + x_21793;
                    
                    // store in accumulator
                    {
                        x_acc_25220 = res_21794;
                    }
                }
            }
        }
        // to reduce current chunk, first store our result in memory
        {
            x_21792 = x_acc_25220;
            ((__local
              float *) red_arr_mem_25211)[sext_i32_i64(local_tid_25207)] =
                x_21792;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t offset_25226;
        int32_t skip_waves_25227;
        float x_25222;
        float x_25223;
        
        offset_25226 = 0;
        // participating threads read initial accumulator
        {
            if (slt32(local_tid_25207, segred_group_sizze_21788)) {
                x_25222 = ((__local
                            float *) red_arr_mem_25211)[sext_i32_i64(local_tid_25207 +
                                                        offset_25226)];
            }
        }
        offset_25226 = 1;
        while (slt32(offset_25226, wave_sizze_25209)) {
            if (slt32(local_tid_25207 + offset_25226,
                      segred_group_sizze_21788) && ((local_tid_25207 -
                                                     squot32(local_tid_25207,
                                                             wave_sizze_25209) *
                                                     wave_sizze_25209) & (2 *
                                                                          offset_25226 -
                                                                          1)) ==
                0) {
                // read array element
                {
                    x_25223 = ((volatile __local
                                float *) red_arr_mem_25211)[sext_i32_i64(local_tid_25207 +
                                                            offset_25226)];
                }
                // apply reduction operation
                {
                    float res_25224 = x_25222 + x_25223;
                    
                    x_25222 = res_25224;
                }
                // write result of operation
                {
                    ((volatile __local
                      float *) red_arr_mem_25211)[sext_i32_i64(local_tid_25207)] =
                        x_25222;
                }
            }
            offset_25226 *= 2;
        }
        skip_waves_25227 = 1;
        while (slt32(skip_waves_25227, squot32(segred_group_sizze_21788 +
                                               wave_sizze_25209 - 1,
                                               wave_sizze_25209))) {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset_25226 = skip_waves_25227 * wave_sizze_25209;
            if (slt32(local_tid_25207 + offset_25226,
                      segred_group_sizze_21788) && ((local_tid_25207 -
                                                     squot32(local_tid_25207,
                                                             wave_sizze_25209) *
                                                     wave_sizze_25209) == 0 &&
                                                    (squot32(local_tid_25207,
                                                             wave_sizze_25209) &
                                                     (2 * skip_waves_25227 -
                                                      1)) == 0)) {
                // read array element
                {
                    x_25223 = ((__local
                                float *) red_arr_mem_25211)[sext_i32_i64(local_tid_25207 +
                                                            offset_25226)];
                }
                // apply reduction operation
                {
                    float res_25224 = x_25222 + x_25223;
                    
                    x_25222 = res_25224;
                }
                // write result of operation
                {
                    ((__local
                      float *) red_arr_mem_25211)[sext_i32_i64(local_tid_25207)] =
                        x_25222;
                }
            }
            skip_waves_25227 *= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // first thread saves the result in accumulator
        {
            if (local_tid_25207 == 0) {
                x_acc_25220 = x_25222;
            }
        }
        if (groups_per_segment_25197 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_25207 == 0) {
                    ((__global float *) mem_24187)[sext_i32_i64(gtid_21759)] =
                        x_acc_25220;
                }
            }
        } else {
            int32_t old_counter_25228;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_25207 == 0) {
                    ((__global
                      float *) group_res_arr_mem_25202)[sext_i32_i64(virt_group_id_25217) *
                                                        sext_i32_i64(segred_group_sizze_21788)] =
                        x_acc_25220;
                    mem_fence_global();
                    old_counter_25228 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_25204)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_25218,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_25213)[0] =
                        old_counter_25228 == groups_per_segment_25197 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_25229;
            
            is_last_group_25229 = ((__local bool *) sync_arr_mem_25213)[0];
            if (is_last_group_25229) {
                if (local_tid_25207 == 0) {
                    old_counter_25228 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_25204)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_25218,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_25197));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_25230 =
                            sdiv_up32(groups_per_segment_25197,
                                      segred_group_sizze_21788);
                    
                    x_21792 = 0.0F;
                    for (int32_t i_25231 = 0; i_25231 < read_per_thread_25230;
                         i_25231++) {
                        int32_t group_res_id_25232 = local_tid_25207 *
                                read_per_thread_25230 + i_25231;
                        int32_t index_of_group_res_25233 =
                                flat_segment_id_25218 *
                                groups_per_segment_25197 + group_res_id_25232;
                        
                        if (slt32(group_res_id_25232,
                                  groups_per_segment_25197)) {
                            x_21793 = ((__global
                                        float *) group_res_arr_mem_25202)[sext_i32_i64(index_of_group_res_25233) *
                                                                          sext_i32_i64(segred_group_sizze_21788)];
                            
                            float res_21794;
                            
                            res_21794 = x_21792 + x_21793;
                            x_21792 = res_21794;
                        }
                    }
                }
                ((__local
                  float *) red_arr_mem_25211)[sext_i32_i64(local_tid_25207)] =
                    x_21792;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_25234;
                    int32_t skip_waves_25235;
                    float x_25222;
                    float x_25223;
                    
                    offset_25234 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_25207, segred_group_sizze_21788)) {
                            x_25222 = ((__local
                                        float *) red_arr_mem_25211)[sext_i32_i64(local_tid_25207 +
                                                                    offset_25234)];
                        }
                    }
                    offset_25234 = 1;
                    while (slt32(offset_25234, wave_sizze_25209)) {
                        if (slt32(local_tid_25207 + offset_25234,
                                  segred_group_sizze_21788) &&
                            ((local_tid_25207 - squot32(local_tid_25207,
                                                        wave_sizze_25209) *
                              wave_sizze_25209) & (2 * offset_25234 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_25223 = ((volatile __local
                                            float *) red_arr_mem_25211)[sext_i32_i64(local_tid_25207 +
                                                                        offset_25234)];
                            }
                            // apply reduction operation
                            {
                                float res_25224 = x_25222 + x_25223;
                                
                                x_25222 = res_25224;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  float *) red_arr_mem_25211)[sext_i32_i64(local_tid_25207)] =
                                    x_25222;
                            }
                        }
                        offset_25234 *= 2;
                    }
                    skip_waves_25235 = 1;
                    while (slt32(skip_waves_25235,
                                 squot32(segred_group_sizze_21788 +
                                         wave_sizze_25209 - 1,
                                         wave_sizze_25209))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_25234 = skip_waves_25235 * wave_sizze_25209;
                        if (slt32(local_tid_25207 + offset_25234,
                                  segred_group_sizze_21788) &&
                            ((local_tid_25207 - squot32(local_tid_25207,
                                                        wave_sizze_25209) *
                              wave_sizze_25209) == 0 &&
                             (squot32(local_tid_25207, wave_sizze_25209) & (2 *
                                                                            skip_waves_25235 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_25223 = ((__local
                                            float *) red_arr_mem_25211)[sext_i32_i64(local_tid_25207 +
                                                                        offset_25234)];
                            }
                            // apply reduction operation
                            {
                                float res_25224 = x_25222 + x_25223;
                                
                                x_25222 = res_25224;
                            }
                            // write result of operation
                            {
                                ((__local
                                  float *) red_arr_mem_25211)[sext_i32_i64(local_tid_25207)] =
                                    x_25222;
                            }
                        }
                        skip_waves_25235 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_25207 == 0) {
                            ((__global
                              float *) mem_24187)[sext_i32_i64(gtid_21759)] =
                                x_25222;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21788
}
__kernel void mainzisegred_large_22379(__global int *global_failure,
                                       __local volatile
                                       int64_t *sync_arr_mem_25390_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_25388_backing_aligned_1,
                                       __local volatile
                                       int64_t *red_arr_mem_25386_backing_aligned_2,
                                       __local volatile
                                       int64_t *red_arr_mem_25384_backing_aligned_3,
                                       int32_t iota_arg_18497,
                                       int32_t num_groups_22563, __global
                                       unsigned char *mem_24192, __global
                                       unsigned char *mem_24230, __global
                                       unsigned char *mem_24233, __global
                                       unsigned char *mem_24239, __global
                                       unsigned char *mem_24242, __global
                                       unsigned char *mem_24245, __global
                                       unsigned char *mem_24248, __global
                                       unsigned char *mem_24253,
                                       int32_t groups_per_segment_25366,
                                       int32_t elements_per_thread_25367,
                                       int32_t virt_num_groups_25368, __global
                                       unsigned char *group_res_arr_mem_25371,
                                       __global
                                       unsigned char *group_res_arr_mem_25373,
                                       __global
                                       unsigned char *group_res_arr_mem_25375,
                                       __global
                                       unsigned char *mainzicounter_mem_25377)
{
    #define segred_group_sizze_22562 (mainzisegred_group_sizze_22373)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict sync_arr_mem_25390_backing_3 =
                          (__local volatile
                           char *) sync_arr_mem_25390_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_25388_backing_2 =
                          (__local volatile
                           char *) red_arr_mem_25388_backing_aligned_1;
    __local volatile char *restrict red_arr_mem_25386_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_25386_backing_aligned_2;
    __local volatile char *restrict red_arr_mem_25384_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_25384_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25379;
    int32_t local_tid_25380;
    int32_t group_sizze_25383;
    int32_t wave_sizze_25382;
    int32_t group_tid_25381;
    
    global_tid_25379 = get_global_id(0);
    local_tid_25380 = get_local_id(0);
    group_sizze_25383 = get_local_size(0);
    wave_sizze_25382 = LOCKSTEP_WIDTH;
    group_tid_25381 = get_group_id(0);
    
    int32_t phys_tid_22379;
    
    phys_tid_22379 = global_tid_25379;
    
    __local char *red_arr_mem_25384;
    
    red_arr_mem_25384 = (__local char *) red_arr_mem_25384_backing_0;
    
    __local char *red_arr_mem_25386;
    
    red_arr_mem_25386 = (__local char *) red_arr_mem_25386_backing_1;
    
    __local char *red_arr_mem_25388;
    
    red_arr_mem_25388 = (__local char *) red_arr_mem_25388_backing_2;
    
    __local char *sync_arr_mem_25390;
    
    sync_arr_mem_25390 = (__local char *) sync_arr_mem_25390_backing_3;
    
    int32_t phys_group_id_25392;
    
    phys_group_id_25392 = get_group_id(0);
    for (int32_t i_25393 = 0; i_25393 < sdiv_up32(virt_num_groups_25368 -
                                                  phys_group_id_25392,
                                                  num_groups_22563);
         i_25393++) {
        int32_t virt_group_id_25394 = phys_group_id_25392 + i_25393 *
                num_groups_22563;
        int32_t flat_segment_id_25395 = squot32(virt_group_id_25394,
                                                groups_per_segment_25366);
        int32_t global_tid_25396 = srem32(virt_group_id_25394 *
                                          segred_group_sizze_22562 +
                                          local_tid_25380,
                                          segred_group_sizze_22562 *
                                          groups_per_segment_25366);
        int32_t gtid_22368 = flat_segment_id_25395;
        int32_t gtid_22378;
        bool x_acc_25397;
        int32_t x_acc_25398;
        float x_acc_25399;
        int32_t chunk_sizze_25400;
        int32_t starting_point_25401;
        
        starting_point_25401 = global_tid_25396 * elements_per_thread_25367;
        
        int32_t remaining_elements_25402;
        
        remaining_elements_25402 = iota_arg_18497 - starting_point_25401;
        if (sle32(remaining_elements_25402, 0) || sle32(iota_arg_18497,
                                                        starting_point_25401)) {
            chunk_sizze_25400 = 0;
        } else {
            if (slt32(iota_arg_18497, (global_tid_25396 + 1) *
                      elements_per_thread_25367)) {
                chunk_sizze_25400 = iota_arg_18497 - global_tid_25396 *
                    elements_per_thread_25367;
            } else {
                chunk_sizze_25400 = elements_per_thread_25367;
            }
        }
        
        bool x_22569;
        int32_t x_22570;
        float x_22571;
        bool x_22572;
        int32_t x_22573;
        float x_22574;
        
        // neutral-initialise the accumulators
        {
            x_acc_25397 = 0;
            x_acc_25398 = -1;
            x_acc_25399 = 0.0F;
        }
        for (int32_t i_25417 = 0; i_25417 < elements_per_thread_25367;
             i_25417++) {
            gtid_22378 = local_tid_25380 + (squot32(global_tid_25396,
                                                    segred_group_sizze_22562) *
                                            elements_per_thread_25367 +
                                            i_25417) * segred_group_sizze_22562;
            if (slt32(gtid_22378, iota_arg_18497)) {
                // apply map function
                {
                    int32_t y_22583 = ((__global
                                        int32_t *) mem_24233)[sext_i32_i64(gtid_22368)];
                    float y_22584 = ((__global
                                      float *) mem_24230)[sext_i32_i64(gtid_22368)];
                    float x_22588 = ((__global
                                      float *) mem_24239)[sext_i32_i64(gtid_22368) *
                                                          sext_i32_i64(iota_arg_18497) +
                                                          sext_i32_i64(gtid_22378)];
                    float x_22589 = ((__global
                                      float *) mem_24192)[sext_i32_i64(gtid_22378)];
                    float res_22592 = x_22588 / y_22584;
                    bool cond_22593 = slt32(gtid_22378, y_22583);
                    bool res_22594;
                    
                    res_22594 = futrts_isnan32(res_22592);
                    
                    bool res_22595 = !res_22594;
                    bool x_22596 = cond_22593 && res_22595;
                    float res_22597 = (float) fabs(res_22592);
                    bool res_22598 = x_22589 < res_22597;
                    bool x_22599 = x_22596 && res_22598;
                    float res_22600;
                    
                    if (cond_22593) {
                        res_22600 = res_22592;
                    } else {
                        res_22600 = 0.0F;
                    }
                    // save map-out results
                    {
                        ((__global
                          float *) mem_24253)[sext_i32_i64(gtid_22368) *
                                              sext_i32_i64(iota_arg_18497) +
                                              sext_i32_i64(gtid_22378)] =
                            res_22592;
                    }
                    // load accumulator
                    {
                        x_22569 = x_acc_25397;
                        x_22570 = x_acc_25398;
                        x_22571 = x_acc_25399;
                    }
                    // load new values
                    {
                        x_22572 = x_22599;
                        x_22573 = gtid_22378;
                        x_22574 = res_22600;
                    }
                    // apply reduction operator
                    {
                        bool res_22575;
                        int32_t res_22576;
                        
                        if (x_22569) {
                            res_22575 = x_22569;
                            res_22576 = x_22570;
                        } else {
                            bool x_22577 = x_22572 && x_22572;
                            bool x_22578 = !x_22572;
                            bool y_22579 = x_22569 && x_22578;
                            bool res_22580 = x_22577 || y_22579;
                            int32_t res_22581;
                            
                            if (x_22572) {
                                res_22581 = x_22573;
                            } else {
                                res_22581 = x_22570;
                            }
                            res_22575 = res_22580;
                            res_22576 = res_22581;
                        }
                        
                        float res_22582 = x_22571 + x_22574;
                        
                        // store in accumulator
                        {
                            x_acc_25397 = res_22575;
                            x_acc_25398 = res_22576;
                            x_acc_25399 = res_22582;
                        }
                    }
                }
            }
            // to reduce current chunk, first store our result in memory
            {
                x_22569 = x_acc_25397;
                x_22570 = x_acc_25398;
                x_22571 = x_acc_25399;
                ((__local
                  bool *) red_arr_mem_25384)[sext_i32_i64(local_tid_25380)] =
                    x_22569;
                ((__local
                  int32_t *) red_arr_mem_25386)[sext_i32_i64(local_tid_25380)] =
                    x_22570;
                ((__local
                  float *) red_arr_mem_25388)[sext_i32_i64(local_tid_25380)] =
                    x_22571;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            int32_t offset_25418;
            int32_t skip_waves_25419;
            bool x_25403;
            int32_t x_25404;
            float x_25405;
            bool x_25406;
            int32_t x_25407;
            float x_25408;
            
            offset_25418 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_25380, segred_group_sizze_22562)) {
                    x_25403 = ((__local
                                bool *) red_arr_mem_25384)[sext_i32_i64(local_tid_25380 +
                                                           offset_25418)];
                    x_25404 = ((__local
                                int32_t *) red_arr_mem_25386)[sext_i32_i64(local_tid_25380 +
                                                              offset_25418)];
                    x_25405 = ((__local
                                float *) red_arr_mem_25388)[sext_i32_i64(local_tid_25380 +
                                                            offset_25418)];
                }
            }
            offset_25418 = 1;
            while (slt32(offset_25418, wave_sizze_25382)) {
                if (slt32(local_tid_25380 + offset_25418,
                          segred_group_sizze_22562) && ((local_tid_25380 -
                                                         squot32(local_tid_25380,
                                                                 wave_sizze_25382) *
                                                         wave_sizze_25382) &
                                                        (2 * offset_25418 -
                                                         1)) == 0) {
                    // read array element
                    {
                        x_25406 = ((volatile __local
                                    bool *) red_arr_mem_25384)[sext_i32_i64(local_tid_25380 +
                                                               offset_25418)];
                        x_25407 = ((volatile __local
                                    int32_t *) red_arr_mem_25386)[sext_i32_i64(local_tid_25380 +
                                                                  offset_25418)];
                        x_25408 = ((volatile __local
                                    float *) red_arr_mem_25388)[sext_i32_i64(local_tid_25380 +
                                                                offset_25418)];
                    }
                    // apply reduction operation
                    {
                        bool res_25409;
                        int32_t res_25410;
                        
                        if (x_25403) {
                            res_25409 = x_25403;
                            res_25410 = x_25404;
                        } else {
                            bool x_25411 = x_25406 && x_25406;
                            bool x_25412 = !x_25406;
                            bool y_25413 = x_25403 && x_25412;
                            bool res_25414 = x_25411 || y_25413;
                            int32_t res_25415;
                            
                            if (x_25406) {
                                res_25415 = x_25407;
                            } else {
                                res_25415 = x_25404;
                            }
                            res_25409 = res_25414;
                            res_25410 = res_25415;
                        }
                        
                        float res_25416 = x_25405 + x_25408;
                        
                        x_25403 = res_25409;
                        x_25404 = res_25410;
                        x_25405 = res_25416;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          bool *) red_arr_mem_25384)[sext_i32_i64(local_tid_25380)] =
                            x_25403;
                        ((volatile __local
                          int32_t *) red_arr_mem_25386)[sext_i32_i64(local_tid_25380)] =
                            x_25404;
                        ((volatile __local
                          float *) red_arr_mem_25388)[sext_i32_i64(local_tid_25380)] =
                            x_25405;
                    }
                }
                offset_25418 *= 2;
            }
            skip_waves_25419 = 1;
            while (slt32(skip_waves_25419, squot32(segred_group_sizze_22562 +
                                                   wave_sizze_25382 - 1,
                                                   wave_sizze_25382))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_25418 = skip_waves_25419 * wave_sizze_25382;
                if (slt32(local_tid_25380 + offset_25418,
                          segred_group_sizze_22562) && ((local_tid_25380 -
                                                         squot32(local_tid_25380,
                                                                 wave_sizze_25382) *
                                                         wave_sizze_25382) ==
                                                        0 &&
                                                        (squot32(local_tid_25380,
                                                                 wave_sizze_25382) &
                                                         (2 * skip_waves_25419 -
                                                          1)) == 0)) {
                    // read array element
                    {
                        x_25406 = ((__local
                                    bool *) red_arr_mem_25384)[sext_i32_i64(local_tid_25380 +
                                                               offset_25418)];
                        x_25407 = ((__local
                                    int32_t *) red_arr_mem_25386)[sext_i32_i64(local_tid_25380 +
                                                                  offset_25418)];
                        x_25408 = ((__local
                                    float *) red_arr_mem_25388)[sext_i32_i64(local_tid_25380 +
                                                                offset_25418)];
                    }
                    // apply reduction operation
                    {
                        bool res_25409;
                        int32_t res_25410;
                        
                        if (x_25403) {
                            res_25409 = x_25403;
                            res_25410 = x_25404;
                        } else {
                            bool x_25411 = x_25406 && x_25406;
                            bool x_25412 = !x_25406;
                            bool y_25413 = x_25403 && x_25412;
                            bool res_25414 = x_25411 || y_25413;
                            int32_t res_25415;
                            
                            if (x_25406) {
                                res_25415 = x_25407;
                            } else {
                                res_25415 = x_25404;
                            }
                            res_25409 = res_25414;
                            res_25410 = res_25415;
                        }
                        
                        float res_25416 = x_25405 + x_25408;
                        
                        x_25403 = res_25409;
                        x_25404 = res_25410;
                        x_25405 = res_25416;
                    }
                    // write result of operation
                    {
                        ((__local
                          bool *) red_arr_mem_25384)[sext_i32_i64(local_tid_25380)] =
                            x_25403;
                        ((__local
                          int32_t *) red_arr_mem_25386)[sext_i32_i64(local_tid_25380)] =
                            x_25404;
                        ((__local
                          float *) red_arr_mem_25388)[sext_i32_i64(local_tid_25380)] =
                            x_25405;
                    }
                }
                skip_waves_25419 *= 2;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            // first thread saves the result in accumulator
            {
                if (local_tid_25380 == 0) {
                    x_acc_25397 = x_25403;
                    x_acc_25398 = x_25404;
                    x_acc_25399 = x_25405;
                }
            }
            // first thread keeps accumulator; others reset to neutral element
            {
                if (!(local_tid_25380 == 0)) {
                    x_acc_25397 = 0;
                    x_acc_25398 = -1;
                    x_acc_25399 = 0.0F;
                }
            }
        }
        x_22569 = x_acc_25397;
        x_22570 = x_acc_25398;
        x_22571 = x_acc_25399;
        if (groups_per_segment_25366 == 1) {
            // first thread in group saves final result to memory
            {
                if (local_tid_25380 == 0) {
                    ((__global bool *) mem_24242)[sext_i32_i64(gtid_22368)] =
                        x_acc_25397;
                    ((__global int32_t *) mem_24245)[sext_i32_i64(gtid_22368)] =
                        x_acc_25398;
                    ((__global float *) mem_24248)[sext_i32_i64(gtid_22368)] =
                        x_acc_25399;
                }
            }
        } else {
            int32_t old_counter_25420;
            
            // first thread in group saves group result to global memory
            {
                if (local_tid_25380 == 0) {
                    ((__global
                      bool *) group_res_arr_mem_25371)[sext_i32_i64(virt_group_id_25394) *
                                                       sext_i32_i64(segred_group_sizze_22562)] =
                        x_acc_25397;
                    ((__global
                      int32_t *) group_res_arr_mem_25373)[sext_i32_i64(virt_group_id_25394) *
                                                          sext_i32_i64(segred_group_sizze_22562)] =
                        x_acc_25398;
                    ((__global
                      float *) group_res_arr_mem_25375)[sext_i32_i64(virt_group_id_25394) *
                                                        sext_i32_i64(segred_group_sizze_22562)] =
                        x_acc_25399;
                    mem_fence_global();
                    old_counter_25420 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_25377)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_25395,
                                                                                                                  10240)))],
                                              (int) 1);
                    ((__local bool *) sync_arr_mem_25390)[0] =
                        old_counter_25420 == groups_per_segment_25366 - 1;
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
            
            bool is_last_group_25421;
            
            is_last_group_25421 = ((__local bool *) sync_arr_mem_25390)[0];
            if (is_last_group_25421) {
                if (local_tid_25380 == 0) {
                    old_counter_25420 =
                        atomic_add_i32_global(&((volatile __global
                                                 int *) mainzicounter_mem_25377)[sext_i64_i32(sext_i32_i64(srem32(flat_segment_id_25395,
                                                                                                                  10240)))],
                                              (int) (0 -
                                                     groups_per_segment_25366));
                }
                // read in the per-group-results
                {
                    int32_t read_per_thread_25422 =
                            sdiv_up32(groups_per_segment_25366,
                                      segred_group_sizze_22562);
                    
                    x_22569 = 0;
                    x_22570 = -1;
                    x_22571 = 0.0F;
                    for (int32_t i_25423 = 0; i_25423 < read_per_thread_25422;
                         i_25423++) {
                        int32_t group_res_id_25424 = local_tid_25380 *
                                read_per_thread_25422 + i_25423;
                        int32_t index_of_group_res_25425 =
                                flat_segment_id_25395 *
                                groups_per_segment_25366 + group_res_id_25424;
                        
                        if (slt32(group_res_id_25424,
                                  groups_per_segment_25366)) {
                            x_22572 = ((__global
                                        bool *) group_res_arr_mem_25371)[sext_i32_i64(index_of_group_res_25425) *
                                                                         sext_i32_i64(segred_group_sizze_22562)];
                            x_22573 = ((__global
                                        int32_t *) group_res_arr_mem_25373)[sext_i32_i64(index_of_group_res_25425) *
                                                                            sext_i32_i64(segred_group_sizze_22562)];
                            x_22574 = ((__global
                                        float *) group_res_arr_mem_25375)[sext_i32_i64(index_of_group_res_25425) *
                                                                          sext_i32_i64(segred_group_sizze_22562)];
                            
                            bool res_22575;
                            int32_t res_22576;
                            
                            if (x_22569) {
                                res_22575 = x_22569;
                                res_22576 = x_22570;
                            } else {
                                bool x_22577 = x_22572 && x_22572;
                                bool x_22578 = !x_22572;
                                bool y_22579 = x_22569 && x_22578;
                                bool res_22580 = x_22577 || y_22579;
                                int32_t res_22581;
                                
                                if (x_22572) {
                                    res_22581 = x_22573;
                                } else {
                                    res_22581 = x_22570;
                                }
                                res_22575 = res_22580;
                                res_22576 = res_22581;
                            }
                            
                            float res_22582 = x_22571 + x_22574;
                            
                            x_22569 = res_22575;
                            x_22570 = res_22576;
                            x_22571 = res_22582;
                        }
                    }
                }
                ((__local
                  bool *) red_arr_mem_25384)[sext_i32_i64(local_tid_25380)] =
                    x_22569;
                ((__local
                  int32_t *) red_arr_mem_25386)[sext_i32_i64(local_tid_25380)] =
                    x_22570;
                ((__local
                  float *) red_arr_mem_25388)[sext_i32_i64(local_tid_25380)] =
                    x_22571;
                barrier(CLK_LOCAL_MEM_FENCE);
                // reduce the per-group results
                {
                    int32_t offset_25426;
                    int32_t skip_waves_25427;
                    bool x_25403;
                    int32_t x_25404;
                    float x_25405;
                    bool x_25406;
                    int32_t x_25407;
                    float x_25408;
                    
                    offset_25426 = 0;
                    // participating threads read initial accumulator
                    {
                        if (slt32(local_tid_25380, segred_group_sizze_22562)) {
                            x_25403 = ((__local
                                        bool *) red_arr_mem_25384)[sext_i32_i64(local_tid_25380 +
                                                                   offset_25426)];
                            x_25404 = ((__local
                                        int32_t *) red_arr_mem_25386)[sext_i32_i64(local_tid_25380 +
                                                                      offset_25426)];
                            x_25405 = ((__local
                                        float *) red_arr_mem_25388)[sext_i32_i64(local_tid_25380 +
                                                                    offset_25426)];
                        }
                    }
                    offset_25426 = 1;
                    while (slt32(offset_25426, wave_sizze_25382)) {
                        if (slt32(local_tid_25380 + offset_25426,
                                  segred_group_sizze_22562) &&
                            ((local_tid_25380 - squot32(local_tid_25380,
                                                        wave_sizze_25382) *
                              wave_sizze_25382) & (2 * offset_25426 - 1)) ==
                            0) {
                            // read array element
                            {
                                x_25406 = ((volatile __local
                                            bool *) red_arr_mem_25384)[sext_i32_i64(local_tid_25380 +
                                                                       offset_25426)];
                                x_25407 = ((volatile __local
                                            int32_t *) red_arr_mem_25386)[sext_i32_i64(local_tid_25380 +
                                                                          offset_25426)];
                                x_25408 = ((volatile __local
                                            float *) red_arr_mem_25388)[sext_i32_i64(local_tid_25380 +
                                                                        offset_25426)];
                            }
                            // apply reduction operation
                            {
                                bool res_25409;
                                int32_t res_25410;
                                
                                if (x_25403) {
                                    res_25409 = x_25403;
                                    res_25410 = x_25404;
                                } else {
                                    bool x_25411 = x_25406 && x_25406;
                                    bool x_25412 = !x_25406;
                                    bool y_25413 = x_25403 && x_25412;
                                    bool res_25414 = x_25411 || y_25413;
                                    int32_t res_25415;
                                    
                                    if (x_25406) {
                                        res_25415 = x_25407;
                                    } else {
                                        res_25415 = x_25404;
                                    }
                                    res_25409 = res_25414;
                                    res_25410 = res_25415;
                                }
                                
                                float res_25416 = x_25405 + x_25408;
                                
                                x_25403 = res_25409;
                                x_25404 = res_25410;
                                x_25405 = res_25416;
                            }
                            // write result of operation
                            {
                                ((volatile __local
                                  bool *) red_arr_mem_25384)[sext_i32_i64(local_tid_25380)] =
                                    x_25403;
                                ((volatile __local
                                  int32_t *) red_arr_mem_25386)[sext_i32_i64(local_tid_25380)] =
                                    x_25404;
                                ((volatile __local
                                  float *) red_arr_mem_25388)[sext_i32_i64(local_tid_25380)] =
                                    x_25405;
                            }
                        }
                        offset_25426 *= 2;
                    }
                    skip_waves_25427 = 1;
                    while (slt32(skip_waves_25427,
                                 squot32(segred_group_sizze_22562 +
                                         wave_sizze_25382 - 1,
                                         wave_sizze_25382))) {
                        barrier(CLK_LOCAL_MEM_FENCE);
                        offset_25426 = skip_waves_25427 * wave_sizze_25382;
                        if (slt32(local_tid_25380 + offset_25426,
                                  segred_group_sizze_22562) &&
                            ((local_tid_25380 - squot32(local_tid_25380,
                                                        wave_sizze_25382) *
                              wave_sizze_25382) == 0 &&
                             (squot32(local_tid_25380, wave_sizze_25382) & (2 *
                                                                            skip_waves_25427 -
                                                                            1)) ==
                             0)) {
                            // read array element
                            {
                                x_25406 = ((__local
                                            bool *) red_arr_mem_25384)[sext_i32_i64(local_tid_25380 +
                                                                       offset_25426)];
                                x_25407 = ((__local
                                            int32_t *) red_arr_mem_25386)[sext_i32_i64(local_tid_25380 +
                                                                          offset_25426)];
                                x_25408 = ((__local
                                            float *) red_arr_mem_25388)[sext_i32_i64(local_tid_25380 +
                                                                        offset_25426)];
                            }
                            // apply reduction operation
                            {
                                bool res_25409;
                                int32_t res_25410;
                                
                                if (x_25403) {
                                    res_25409 = x_25403;
                                    res_25410 = x_25404;
                                } else {
                                    bool x_25411 = x_25406 && x_25406;
                                    bool x_25412 = !x_25406;
                                    bool y_25413 = x_25403 && x_25412;
                                    bool res_25414 = x_25411 || y_25413;
                                    int32_t res_25415;
                                    
                                    if (x_25406) {
                                        res_25415 = x_25407;
                                    } else {
                                        res_25415 = x_25404;
                                    }
                                    res_25409 = res_25414;
                                    res_25410 = res_25415;
                                }
                                
                                float res_25416 = x_25405 + x_25408;
                                
                                x_25403 = res_25409;
                                x_25404 = res_25410;
                                x_25405 = res_25416;
                            }
                            // write result of operation
                            {
                                ((__local
                                  bool *) red_arr_mem_25384)[sext_i32_i64(local_tid_25380)] =
                                    x_25403;
                                ((__local
                                  int32_t *) red_arr_mem_25386)[sext_i32_i64(local_tid_25380)] =
                                    x_25404;
                                ((__local
                                  float *) red_arr_mem_25388)[sext_i32_i64(local_tid_25380)] =
                                    x_25405;
                            }
                        }
                        skip_waves_25427 *= 2;
                    }
                    // and back to memory with the final result
                    {
                        if (local_tid_25380 == 0) {
                            ((__global
                              bool *) mem_24242)[sext_i32_i64(gtid_22368)] =
                                x_25403;
                            ((__global
                              int32_t *) mem_24245)[sext_i32_i64(gtid_22368)] =
                                x_25404;
                            ((__global
                              float *) mem_24248)[sext_i32_i64(gtid_22368)] =
                                x_25405;
                        }
                    }
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_22562
}
__kernel void mainzisegred_nonseg_21712(__global int *global_failure,
                                        __local volatile
                                        int64_t *red_arr_mem_25152_backing_aligned_0,
                                        __local volatile
                                        int64_t *sync_arr_mem_25150_backing_aligned_1,
                                        int32_t m_18149,
                                        int32_t num_groups_21707, __global
                                        unsigned char *res_mem_24174, __global
                                        unsigned char *mem_24179, __global
                                        unsigned char *mainzicounter_mem_25140,
                                        __global
                                        unsigned char *group_res_arr_mem_25142,
                                        int32_t num_threads_25144)
{
    #define segred_group_sizze_21705 (mainzisegred_group_sizze_21704)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_25152_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_25152_backing_aligned_0;
    __local volatile char *restrict sync_arr_mem_25150_backing_0 =
                          (__local volatile
                           char *) sync_arr_mem_25150_backing_aligned_1;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25145;
    int32_t local_tid_25146;
    int32_t group_sizze_25149;
    int32_t wave_sizze_25148;
    int32_t group_tid_25147;
    
    global_tid_25145 = get_global_id(0);
    local_tid_25146 = get_local_id(0);
    group_sizze_25149 = get_local_size(0);
    wave_sizze_25148 = LOCKSTEP_WIDTH;
    group_tid_25147 = get_group_id(0);
    
    int32_t phys_tid_21712;
    
    phys_tid_21712 = global_tid_25145;
    
    __local char *sync_arr_mem_25150;
    
    sync_arr_mem_25150 = (__local char *) sync_arr_mem_25150_backing_0;
    
    __local char *red_arr_mem_25152;
    
    red_arr_mem_25152 = (__local char *) red_arr_mem_25152_backing_1;
    
    int32_t dummy_21710;
    
    dummy_21710 = 0;
    
    int32_t gtid_21711;
    
    gtid_21711 = 0;
    
    int32_t x_acc_25154;
    int32_t chunk_sizze_25155;
    
    chunk_sizze_25155 = smin32(sdiv_up32(m_18149, segred_group_sizze_21705 *
                                         num_groups_21707), sdiv_up32(m_18149 -
                                                                      phys_tid_21712,
                                                                      num_threads_25144));
    
    int32_t x_18476;
    int32_t x_18477;
    
    // neutral-initialise the accumulators
    {
        x_acc_25154 = 0;
    }
    for (int32_t i_25159 = 0; i_25159 < chunk_sizze_25155; i_25159++) {
        gtid_21711 = phys_tid_21712 + num_threads_25144 * i_25159;
        // apply map function
        {
            int32_t x_18479 = ((__global
                                int32_t *) res_mem_24174)[sext_i32_i64(gtid_21711)];
            
            // save map-out results
            { }
            // load accumulator
            {
                x_18476 = x_acc_25154;
            }
            // load new values
            {
                x_18477 = x_18479;
            }
            // apply reduction operator
            {
                int32_t res_18478 = smax32(x_18476, x_18477);
                
                // store in accumulator
                {
                    x_acc_25154 = res_18478;
                }
            }
        }
    }
    // to reduce current chunk, first store our result in memory
    {
        x_18476 = x_acc_25154;
        ((__local int32_t *) red_arr_mem_25152)[sext_i32_i64(local_tid_25146)] =
            x_18476;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t offset_25160;
    int32_t skip_waves_25161;
    int32_t x_25156;
    int32_t x_25157;
    
    offset_25160 = 0;
    // participating threads read initial accumulator
    {
        if (slt32(local_tid_25146, segred_group_sizze_21705)) {
            x_25156 = ((__local
                        int32_t *) red_arr_mem_25152)[sext_i32_i64(local_tid_25146 +
                                                      offset_25160)];
        }
    }
    offset_25160 = 1;
    while (slt32(offset_25160, wave_sizze_25148)) {
        if (slt32(local_tid_25146 + offset_25160, segred_group_sizze_21705) &&
            ((local_tid_25146 - squot32(local_tid_25146, wave_sizze_25148) *
              wave_sizze_25148) & (2 * offset_25160 - 1)) == 0) {
            // read array element
            {
                x_25157 = ((volatile __local
                            int32_t *) red_arr_mem_25152)[sext_i32_i64(local_tid_25146 +
                                                          offset_25160)];
            }
            // apply reduction operation
            {
                int32_t res_25158 = smax32(x_25156, x_25157);
                
                x_25156 = res_25158;
            }
            // write result of operation
            {
                ((volatile __local
                  int32_t *) red_arr_mem_25152)[sext_i32_i64(local_tid_25146)] =
                    x_25156;
            }
        }
        offset_25160 *= 2;
    }
    skip_waves_25161 = 1;
    while (slt32(skip_waves_25161, squot32(segred_group_sizze_21705 +
                                           wave_sizze_25148 - 1,
                                           wave_sizze_25148))) {
        barrier(CLK_LOCAL_MEM_FENCE);
        offset_25160 = skip_waves_25161 * wave_sizze_25148;
        if (slt32(local_tid_25146 + offset_25160, segred_group_sizze_21705) &&
            ((local_tid_25146 - squot32(local_tid_25146, wave_sizze_25148) *
              wave_sizze_25148) == 0 && (squot32(local_tid_25146,
                                                 wave_sizze_25148) & (2 *
                                                                      skip_waves_25161 -
                                                                      1)) ==
             0)) {
            // read array element
            {
                x_25157 = ((__local
                            int32_t *) red_arr_mem_25152)[sext_i32_i64(local_tid_25146 +
                                                          offset_25160)];
            }
            // apply reduction operation
            {
                int32_t res_25158 = smax32(x_25156, x_25157);
                
                x_25156 = res_25158;
            }
            // write result of operation
            {
                ((__local
                  int32_t *) red_arr_mem_25152)[sext_i32_i64(local_tid_25146)] =
                    x_25156;
            }
        }
        skip_waves_25161 *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // first thread saves the result in accumulator
    {
        if (local_tid_25146 == 0) {
            x_acc_25154 = x_25156;
        }
    }
    
    int32_t old_counter_25162;
    
    // first thread in group saves group result to global memory
    {
        if (local_tid_25146 == 0) {
            ((__global
              int32_t *) group_res_arr_mem_25142)[sext_i32_i64(group_tid_25147) *
                                                  sext_i32_i64(segred_group_sizze_21705)] =
                x_acc_25154;
            mem_fence_global();
            old_counter_25162 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_25140)[0],
                                                      (int) 1);
            ((__local bool *) sync_arr_mem_25150)[0] = old_counter_25162 ==
                num_groups_21707 - 1;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    
    bool is_last_group_25163;
    
    is_last_group_25163 = ((__local bool *) sync_arr_mem_25150)[0];
    if (is_last_group_25163) {
        if (local_tid_25146 == 0) {
            old_counter_25162 = atomic_add_i32_global(&((volatile __global
                                                         int *) mainzicounter_mem_25140)[0],
                                                      (int) (0 -
                                                             num_groups_21707));
        }
        // read in the per-group-results
        {
            int32_t read_per_thread_25164 = sdiv_up32(num_groups_21707,
                                                      segred_group_sizze_21705);
            
            x_18476 = 0;
            for (int32_t i_25165 = 0; i_25165 < read_per_thread_25164;
                 i_25165++) {
                int32_t group_res_id_25166 = local_tid_25146 *
                        read_per_thread_25164 + i_25165;
                int32_t index_of_group_res_25167 = group_res_id_25166;
                
                if (slt32(group_res_id_25166, num_groups_21707)) {
                    x_18477 = ((__global
                                int32_t *) group_res_arr_mem_25142)[sext_i32_i64(index_of_group_res_25167) *
                                                                    sext_i32_i64(segred_group_sizze_21705)];
                    
                    int32_t res_18478;
                    
                    res_18478 = smax32(x_18476, x_18477);
                    x_18476 = res_18478;
                }
            }
        }
        ((__local int32_t *) red_arr_mem_25152)[sext_i32_i64(local_tid_25146)] =
            x_18476;
        barrier(CLK_LOCAL_MEM_FENCE);
        // reduce the per-group results
        {
            int32_t offset_25168;
            int32_t skip_waves_25169;
            int32_t x_25156;
            int32_t x_25157;
            
            offset_25168 = 0;
            // participating threads read initial accumulator
            {
                if (slt32(local_tid_25146, segred_group_sizze_21705)) {
                    x_25156 = ((__local
                                int32_t *) red_arr_mem_25152)[sext_i32_i64(local_tid_25146 +
                                                              offset_25168)];
                }
            }
            offset_25168 = 1;
            while (slt32(offset_25168, wave_sizze_25148)) {
                if (slt32(local_tid_25146 + offset_25168,
                          segred_group_sizze_21705) && ((local_tid_25146 -
                                                         squot32(local_tid_25146,
                                                                 wave_sizze_25148) *
                                                         wave_sizze_25148) &
                                                        (2 * offset_25168 -
                                                         1)) == 0) {
                    // read array element
                    {
                        x_25157 = ((volatile __local
                                    int32_t *) red_arr_mem_25152)[sext_i32_i64(local_tid_25146 +
                                                                  offset_25168)];
                    }
                    // apply reduction operation
                    {
                        int32_t res_25158 = smax32(x_25156, x_25157);
                        
                        x_25156 = res_25158;
                    }
                    // write result of operation
                    {
                        ((volatile __local
                          int32_t *) red_arr_mem_25152)[sext_i32_i64(local_tid_25146)] =
                            x_25156;
                    }
                }
                offset_25168 *= 2;
            }
            skip_waves_25169 = 1;
            while (slt32(skip_waves_25169, squot32(segred_group_sizze_21705 +
                                                   wave_sizze_25148 - 1,
                                                   wave_sizze_25148))) {
                barrier(CLK_LOCAL_MEM_FENCE);
                offset_25168 = skip_waves_25169 * wave_sizze_25148;
                if (slt32(local_tid_25146 + offset_25168,
                          segred_group_sizze_21705) && ((local_tid_25146 -
                                                         squot32(local_tid_25146,
                                                                 wave_sizze_25148) *
                                                         wave_sizze_25148) ==
                                                        0 &&
                                                        (squot32(local_tid_25146,
                                                                 wave_sizze_25148) &
                                                         (2 * skip_waves_25169 -
                                                          1)) == 0)) {
                    // read array element
                    {
                        x_25157 = ((__local
                                    int32_t *) red_arr_mem_25152)[sext_i32_i64(local_tid_25146 +
                                                                  offset_25168)];
                    }
                    // apply reduction operation
                    {
                        int32_t res_25158 = smax32(x_25156, x_25157);
                        
                        x_25156 = res_25158;
                    }
                    // write result of operation
                    {
                        ((__local
                          int32_t *) red_arr_mem_25152)[sext_i32_i64(local_tid_25146)] =
                            x_25156;
                    }
                }
                skip_waves_25169 *= 2;
            }
            // and back to memory with the final result
            {
                if (local_tid_25146 == 0) {
                    ((__global int32_t *) mem_24179)[0] = x_25156;
                }
            }
        }
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21705
}
__kernel void mainzisegred_small_19482(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_24517_backing_aligned_0,
                                       int32_t N_18148, int32_t m_18149,
                                       int32_t N_18150, int32_t n_18153,
                                       int32_t k2p2zq_18165,
                                       int32_t num_groups_19645, __global
                                       unsigned char *images_mem_23523, __global
                                       unsigned char *binop_p_mem_23536,
                                       __global unsigned char *mem_23641,
                                       __global unsigned char *mem_23649,
                                       int32_t segment_sizze_nonzzero_24510)
{
    #define segred_group_sizze_19644 (mainzisegred_group_sizze_19476)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24517_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24517_backing_aligned_0;
    
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
    
    int32_t phys_tid_19482;
    
    phys_tid_19482 = global_tid_24512;
    
    __local char *red_arr_mem_24517;
    
    red_arr_mem_24517 = (__local char *) red_arr_mem_24517_backing_0;
    
    int32_t phys_group_id_24519;
    
    phys_group_id_24519 = get_group_id(0);
    for (int32_t i_24520 = 0; i_24520 < sdiv_up32(sdiv_up32(m_18149 *
                                                            k2p2zq_18165 *
                                                            k2p2zq_18165,
                                                            squot32(segred_group_sizze_19644,
                                                                    segment_sizze_nonzzero_24510)) -
                                                  phys_group_id_24519,
                                                  num_groups_19645);
         i_24520++) {
        int32_t virt_group_id_24521 = phys_group_id_24519 + i_24520 *
                num_groups_19645;
        int32_t gtid_19465 = squot32(squot32(local_tid_24513,
                                             segment_sizze_nonzzero_24510) +
                                     virt_group_id_24521 *
                                     squot32(segred_group_sizze_19644,
                                             segment_sizze_nonzzero_24510),
                                     k2p2zq_18165 * k2p2zq_18165);
        int32_t gtid_19466 = squot32(squot32(local_tid_24513,
                                             segment_sizze_nonzzero_24510) +
                                     virt_group_id_24521 *
                                     squot32(segred_group_sizze_19644,
                                             segment_sizze_nonzzero_24510) -
                                     squot32(squot32(local_tid_24513,
                                                     segment_sizze_nonzzero_24510) +
                                             virt_group_id_24521 *
                                             squot32(segred_group_sizze_19644,
                                                     segment_sizze_nonzzero_24510),
                                             k2p2zq_18165 * k2p2zq_18165) *
                                     (k2p2zq_18165 * k2p2zq_18165),
                                     k2p2zq_18165);
        int32_t gtid_19467 = squot32(local_tid_24513,
                                     segment_sizze_nonzzero_24510) +
                virt_group_id_24521 * squot32(segred_group_sizze_19644,
                                              segment_sizze_nonzzero_24510) -
                squot32(squot32(local_tid_24513, segment_sizze_nonzzero_24510) +
                        virt_group_id_24521 * squot32(segred_group_sizze_19644,
                                                      segment_sizze_nonzzero_24510),
                        k2p2zq_18165 * k2p2zq_18165) * (k2p2zq_18165 *
                                                        k2p2zq_18165) -
                squot32(squot32(local_tid_24513, segment_sizze_nonzzero_24510) +
                        virt_group_id_24521 * squot32(segred_group_sizze_19644,
                                                      segment_sizze_nonzzero_24510) -
                        squot32(squot32(local_tid_24513,
                                        segment_sizze_nonzzero_24510) +
                                virt_group_id_24521 *
                                squot32(segred_group_sizze_19644,
                                        segment_sizze_nonzzero_24510),
                                k2p2zq_18165 * k2p2zq_18165) * (k2p2zq_18165 *
                                                                k2p2zq_18165),
                        k2p2zq_18165) * k2p2zq_18165;
        int32_t gtid_19481 = srem32(local_tid_24513, n_18153);
        
        // apply map function if in bounds
        {
            if (slt32(0, n_18153) && (((slt32(gtid_19465, m_18149) &&
                                        slt32(gtid_19466, k2p2zq_18165)) &&
                                       slt32(gtid_19467, k2p2zq_18165)) &&
                                      slt32(local_tid_24513, n_18153 *
                                            squot32(segred_group_sizze_19644,
                                                    segment_sizze_nonzzero_24510)))) {
                float x_19654 = ((__global
                                  float *) images_mem_23523)[sext_i32_i64(gtid_19465) *
                                                             sext_i32_i64(N_18150) +
                                                             sext_i32_i64(gtid_19481)];
                float x_19655 = ((__global
                                  float *) binop_p_mem_23536)[sext_i32_i64(gtid_19466) *
                                                              sext_i32_i64(N_18148) +
                                                              sext_i32_i64(gtid_19481)];
                float x_19656 = ((__global
                                  float *) mem_23641)[sext_i32_i64(gtid_19467) *
                                                      sext_i32_i64(N_18148) +
                                                      sext_i32_i64(gtid_19481)];
                float x_19657 = x_19655 * x_19656;
                bool res_19658;
                
                res_19658 = futrts_isnan32(x_19654);
                
                float y_19659;
                
                if (res_19658) {
                    y_19659 = 0.0F;
                } else {
                    y_19659 = 1.0F;
                }
                
                float res_19660 = x_19657 * y_19659;
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_24517)[sext_i32_i64(local_tid_24513)] =
                        res_19660;
                }
            } else {
                ((__local
                  float *) red_arr_mem_24517)[sext_i32_i64(local_tid_24513)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, n_18153)) {
            // perform segmented scan to imitate reduction
            {
                float x_19648;
                float x_19649;
                float x_24522;
                float x_24523;
                int32_t skip_threads_24525;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_24513, n_18153 *
                              squot32(segred_group_sizze_19644,
                                      segment_sizze_nonzzero_24510))) {
                        x_19649 = ((volatile __local
                                    float *) red_arr_mem_24517)[sext_i32_i64(local_tid_24513)];
                        if ((local_tid_24513 - squot32(local_tid_24513, 32) *
                             32) == 0) {
                            x_19648 = x_19649;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24525 = 1;
                    while (slt32(skip_threads_24525, 32)) {
                        if (sle32(skip_threads_24525, local_tid_24513 -
                                  squot32(local_tid_24513, 32) * 32) &&
                            slt32(local_tid_24513, n_18153 *
                                  squot32(segred_group_sizze_19644,
                                          segment_sizze_nonzzero_24510))) {
                            // read operands
                            {
                                x_19648 = ((volatile __local
                                            float *) red_arr_mem_24517)[sext_i32_i64(local_tid_24513 -
                                                                        skip_threads_24525)];
                            }
                            // perform operation
                            {
                                bool inactive_24526 =
                                     slt32(srem32(local_tid_24513, n_18153),
                                           local_tid_24513 - (local_tid_24513 -
                                                              skip_threads_24525));
                                
                                if (inactive_24526) {
                                    x_19648 = x_19649;
                                }
                                if (!inactive_24526) {
                                    float res_19650 = x_19648 + x_19649;
                                    
                                    x_19648 = res_19650;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24515, skip_threads_24525)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24525, local_tid_24513 -
                                  squot32(local_tid_24513, 32) * 32) &&
                            slt32(local_tid_24513, n_18153 *
                                  squot32(segred_group_sizze_19644,
                                          segment_sizze_nonzzero_24510))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24517)[sext_i32_i64(local_tid_24513)] =
                                    x_19648;
                                x_19649 = x_19648;
                            }
                        }
                        if (sle32(wave_sizze_24515, skip_threads_24525)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24525 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_24513 - squot32(local_tid_24513, 32) * 32) ==
                        31 && slt32(local_tid_24513, n_18153 *
                                    squot32(segred_group_sizze_19644,
                                            segment_sizze_nonzzero_24510))) {
                        ((volatile __local
                          float *) red_arr_mem_24517)[sext_i32_i64(squot32(local_tid_24513,
                                                                           32))] =
                            x_19648;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_24527;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_24513, 32) == 0 &&
                            slt32(local_tid_24513, n_18153 *
                                  squot32(segred_group_sizze_19644,
                                          segment_sizze_nonzzero_24510))) {
                            x_24523 = ((volatile __local
                                        float *) red_arr_mem_24517)[sext_i32_i64(local_tid_24513)];
                            if ((local_tid_24513 - squot32(local_tid_24513,
                                                           32) * 32) == 0) {
                                x_24522 = x_24523;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_24527 = 1;
                        while (slt32(skip_threads_24527, 32)) {
                            if (sle32(skip_threads_24527, local_tid_24513 -
                                      squot32(local_tid_24513, 32) * 32) &&
                                (squot32(local_tid_24513, 32) == 0 &&
                                 slt32(local_tid_24513, n_18153 *
                                       squot32(segred_group_sizze_19644,
                                               segment_sizze_nonzzero_24510)))) {
                                // read operands
                                {
                                    x_24522 = ((volatile __local
                                                float *) red_arr_mem_24517)[sext_i32_i64(local_tid_24513 -
                                                                            skip_threads_24527)];
                                }
                                // perform operation
                                {
                                    bool inactive_24528 =
                                         slt32(srem32(local_tid_24513 * 32 +
                                                      32 - 1, n_18153),
                                               local_tid_24513 * 32 + 32 - 1 -
                                               ((local_tid_24513 -
                                                 skip_threads_24527) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_24528) {
                                        x_24522 = x_24523;
                                    }
                                    if (!inactive_24528) {
                                        float res_24524 = x_24522 + x_24523;
                                        
                                        x_24522 = res_24524;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_24515, skip_threads_24527)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_24527, local_tid_24513 -
                                      squot32(local_tid_24513, 32) * 32) &&
                                (squot32(local_tid_24513, 32) == 0 &&
                                 slt32(local_tid_24513, n_18153 *
                                       squot32(segred_group_sizze_19644,
                                               segment_sizze_nonzzero_24510)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_24517)[sext_i32_i64(local_tid_24513)] =
                                        x_24522;
                                    x_24523 = x_24522;
                                }
                            }
                            if (sle32(wave_sizze_24515, skip_threads_24527)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_24527 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_24513, 32) == 0 ||
                          !slt32(local_tid_24513, n_18153 *
                                 squot32(segred_group_sizze_19644,
                                         segment_sizze_nonzzero_24510)))) {
                        // read operands
                        {
                            x_19649 = x_19648;
                            x_19648 = ((__local
                                        float *) red_arr_mem_24517)[sext_i32_i64(squot32(local_tid_24513,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_24529 = slt32(srem32(local_tid_24513,
                                                               n_18153),
                                                        local_tid_24513 -
                                                        (squot32(local_tid_24513,
                                                                 32) * 32 - 1));
                            
                            if (inactive_24529) {
                                x_19648 = x_19649;
                            }
                            if (!inactive_24529) {
                                float res_19650 = x_19648 + x_19649;
                                
                                x_19648 = res_19650;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_24517)[sext_i32_i64(local_tid_24513)] =
                                x_19648;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_24513, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_24517)[sext_i32_i64(local_tid_24513)] =
                            x_19649;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_24521 * squot32(segred_group_sizze_19644,
                                                    segment_sizze_nonzzero_24510) +
                      local_tid_24513, m_18149 * k2p2zq_18165 * k2p2zq_18165) &&
                slt32(local_tid_24513, squot32(segred_group_sizze_19644,
                                               segment_sizze_nonzzero_24510))) {
                ((__global
                  float *) mem_23649)[sext_i32_i64(squot32(virt_group_id_24521 *
                                                           squot32(segred_group_sizze_19644,
                                                                   segment_sizze_nonzzero_24510) +
                                                           local_tid_24513,
                                                           k2p2zq_18165 *
                                                           k2p2zq_18165)) *
                                      sext_i32_i64(k2p2zq_18165 *
                                      k2p2zq_18165) +
                                      sext_i32_i64(squot32(virt_group_id_24521 *
                                                           squot32(segred_group_sizze_19644,
                                                                   segment_sizze_nonzzero_24510) +
                                                           local_tid_24513 -
                                                           squot32(virt_group_id_24521 *
                                                                   squot32(segred_group_sizze_19644,
                                                                           segment_sizze_nonzzero_24510) +
                                                                   local_tid_24513,
                                                                   k2p2zq_18165 *
                                                                   k2p2zq_18165) *
                                                           (k2p2zq_18165 *
                                                            k2p2zq_18165),
                                                           k2p2zq_18165)) *
                                      sext_i32_i64(k2p2zq_18165) +
                                      sext_i32_i64(virt_group_id_24521 *
                                      squot32(segred_group_sizze_19644,
                                              segment_sizze_nonzzero_24510) +
                                      local_tid_24513 -
                                      squot32(virt_group_id_24521 *
                                              squot32(segred_group_sizze_19644,
                                                      segment_sizze_nonzzero_24510) +
                                              local_tid_24513, k2p2zq_18165 *
                                              k2p2zq_18165) * (k2p2zq_18165 *
                                                               k2p2zq_18165) -
                                      squot32(virt_group_id_24521 *
                                              squot32(segred_group_sizze_19644,
                                                      segment_sizze_nonzzero_24510) +
                                              local_tid_24513 -
                                              squot32(virt_group_id_24521 *
                                                      squot32(segred_group_sizze_19644,
                                                              segment_sizze_nonzzero_24510) +
                                                      local_tid_24513,
                                                      k2p2zq_18165 *
                                                      k2p2zq_18165) *
                                              (k2p2zq_18165 * k2p2zq_18165),
                                              k2p2zq_18165) * k2p2zq_18165)] =
                    ((__local
                      float *) red_arr_mem_24517)[sext_i32_i64((local_tid_24513 +
                                                                1) *
                                                  segment_sizze_nonzzero_24510 -
                                                  1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_19644
}
__kernel void mainzisegred_small_20730(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_24670_backing_aligned_0,
                                       int32_t N_18148, int32_t m_18149,
                                       int32_t N_18150, int32_t n_18153,
                                       int32_t k2p2zq_18165,
                                       int32_t num_groups_20793, __global
                                       unsigned char *images_mem_23523, __global
                                       unsigned char *binop_p_mem_23536,
                                       __global unsigned char *mem_23878,
                                       int32_t segment_sizze_nonzzero_24663)
{
    #define segred_group_sizze_20792 (mainzisegred_group_sizze_20724)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24670_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24670_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24665;
    int32_t local_tid_24666;
    int32_t group_sizze_24669;
    int32_t wave_sizze_24668;
    int32_t group_tid_24667;
    
    global_tid_24665 = get_global_id(0);
    local_tid_24666 = get_local_id(0);
    group_sizze_24669 = get_local_size(0);
    wave_sizze_24668 = LOCKSTEP_WIDTH;
    group_tid_24667 = get_group_id(0);
    
    int32_t phys_tid_20730;
    
    phys_tid_20730 = global_tid_24665;
    
    __local char *red_arr_mem_24670;
    
    red_arr_mem_24670 = (__local char *) red_arr_mem_24670_backing_0;
    
    int32_t phys_group_id_24672;
    
    phys_group_id_24672 = get_group_id(0);
    for (int32_t i_24673 = 0; i_24673 < sdiv_up32(sdiv_up32(m_18149 *
                                                            k2p2zq_18165,
                                                            squot32(segred_group_sizze_20792,
                                                                    segment_sizze_nonzzero_24663)) -
                                                  phys_group_id_24672,
                                                  num_groups_20793);
         i_24673++) {
        int32_t virt_group_id_24674 = phys_group_id_24672 + i_24673 *
                num_groups_20793;
        int32_t gtid_20716 = squot32(squot32(local_tid_24666,
                                             segment_sizze_nonzzero_24663) +
                                     virt_group_id_24674 *
                                     squot32(segred_group_sizze_20792,
                                             segment_sizze_nonzzero_24663),
                                     k2p2zq_18165);
        int32_t gtid_20717 = squot32(local_tid_24666,
                                     segment_sizze_nonzzero_24663) +
                virt_group_id_24674 * squot32(segred_group_sizze_20792,
                                              segment_sizze_nonzzero_24663) -
                squot32(squot32(local_tid_24666, segment_sizze_nonzzero_24663) +
                        virt_group_id_24674 * squot32(segred_group_sizze_20792,
                                                      segment_sizze_nonzzero_24663),
                        k2p2zq_18165) * k2p2zq_18165;
        int32_t gtid_20729 = srem32(local_tid_24666, n_18153);
        
        // apply map function if in bounds
        {
            if (slt32(0, n_18153) && ((slt32(gtid_20716, m_18149) &&
                                       slt32(gtid_20717, k2p2zq_18165)) &&
                                      slt32(local_tid_24666, n_18153 *
                                            squot32(segred_group_sizze_20792,
                                                    segment_sizze_nonzzero_24663)))) {
                float x_20802 = ((__global
                                  float *) images_mem_23523)[sext_i32_i64(gtid_20716) *
                                                             sext_i32_i64(N_18150) +
                                                             sext_i32_i64(gtid_20729)];
                bool res_20803;
                
                res_20803 = futrts_isnan32(x_20802);
                
                float res_20804;
                
                if (res_20803) {
                    res_20804 = 0.0F;
                } else {
                    float x_20801 = ((__global
                                      float *) binop_p_mem_23536)[sext_i32_i64(gtid_20717) *
                                                                  sext_i32_i64(N_18148) +
                                                                  sext_i32_i64(gtid_20729)];
                    float res_20805 = x_20801 * x_20802;
                    
                    res_20804 = res_20805;
                }
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_24670)[sext_i32_i64(local_tid_24666)] =
                        res_20804;
                }
            } else {
                ((__local
                  float *) red_arr_mem_24670)[sext_i32_i64(local_tid_24666)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, n_18153)) {
            // perform segmented scan to imitate reduction
            {
                float x_20796;
                float x_20797;
                float x_24675;
                float x_24676;
                int32_t skip_threads_24678;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_24666, n_18153 *
                              squot32(segred_group_sizze_20792,
                                      segment_sizze_nonzzero_24663))) {
                        x_20797 = ((volatile __local
                                    float *) red_arr_mem_24670)[sext_i32_i64(local_tid_24666)];
                        if ((local_tid_24666 - squot32(local_tid_24666, 32) *
                             32) == 0) {
                            x_20796 = x_20797;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24678 = 1;
                    while (slt32(skip_threads_24678, 32)) {
                        if (sle32(skip_threads_24678, local_tid_24666 -
                                  squot32(local_tid_24666, 32) * 32) &&
                            slt32(local_tid_24666, n_18153 *
                                  squot32(segred_group_sizze_20792,
                                          segment_sizze_nonzzero_24663))) {
                            // read operands
                            {
                                x_20796 = ((volatile __local
                                            float *) red_arr_mem_24670)[sext_i32_i64(local_tid_24666 -
                                                                        skip_threads_24678)];
                            }
                            // perform operation
                            {
                                bool inactive_24679 =
                                     slt32(srem32(local_tid_24666, n_18153),
                                           local_tid_24666 - (local_tid_24666 -
                                                              skip_threads_24678));
                                
                                if (inactive_24679) {
                                    x_20796 = x_20797;
                                }
                                if (!inactive_24679) {
                                    float res_20798 = x_20796 + x_20797;
                                    
                                    x_20796 = res_20798;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24668, skip_threads_24678)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24678, local_tid_24666 -
                                  squot32(local_tid_24666, 32) * 32) &&
                            slt32(local_tid_24666, n_18153 *
                                  squot32(segred_group_sizze_20792,
                                          segment_sizze_nonzzero_24663))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24670)[sext_i32_i64(local_tid_24666)] =
                                    x_20796;
                                x_20797 = x_20796;
                            }
                        }
                        if (sle32(wave_sizze_24668, skip_threads_24678)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24678 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_24666 - squot32(local_tid_24666, 32) * 32) ==
                        31 && slt32(local_tid_24666, n_18153 *
                                    squot32(segred_group_sizze_20792,
                                            segment_sizze_nonzzero_24663))) {
                        ((volatile __local
                          float *) red_arr_mem_24670)[sext_i32_i64(squot32(local_tid_24666,
                                                                           32))] =
                            x_20796;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_24680;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_24666, 32) == 0 &&
                            slt32(local_tid_24666, n_18153 *
                                  squot32(segred_group_sizze_20792,
                                          segment_sizze_nonzzero_24663))) {
                            x_24676 = ((volatile __local
                                        float *) red_arr_mem_24670)[sext_i32_i64(local_tid_24666)];
                            if ((local_tid_24666 - squot32(local_tid_24666,
                                                           32) * 32) == 0) {
                                x_24675 = x_24676;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_24680 = 1;
                        while (slt32(skip_threads_24680, 32)) {
                            if (sle32(skip_threads_24680, local_tid_24666 -
                                      squot32(local_tid_24666, 32) * 32) &&
                                (squot32(local_tid_24666, 32) == 0 &&
                                 slt32(local_tid_24666, n_18153 *
                                       squot32(segred_group_sizze_20792,
                                               segment_sizze_nonzzero_24663)))) {
                                // read operands
                                {
                                    x_24675 = ((volatile __local
                                                float *) red_arr_mem_24670)[sext_i32_i64(local_tid_24666 -
                                                                            skip_threads_24680)];
                                }
                                // perform operation
                                {
                                    bool inactive_24681 =
                                         slt32(srem32(local_tid_24666 * 32 +
                                                      32 - 1, n_18153),
                                               local_tid_24666 * 32 + 32 - 1 -
                                               ((local_tid_24666 -
                                                 skip_threads_24680) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_24681) {
                                        x_24675 = x_24676;
                                    }
                                    if (!inactive_24681) {
                                        float res_24677 = x_24675 + x_24676;
                                        
                                        x_24675 = res_24677;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_24668, skip_threads_24680)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_24680, local_tid_24666 -
                                      squot32(local_tid_24666, 32) * 32) &&
                                (squot32(local_tid_24666, 32) == 0 &&
                                 slt32(local_tid_24666, n_18153 *
                                       squot32(segred_group_sizze_20792,
                                               segment_sizze_nonzzero_24663)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_24670)[sext_i32_i64(local_tid_24666)] =
                                        x_24675;
                                    x_24676 = x_24675;
                                }
                            }
                            if (sle32(wave_sizze_24668, skip_threads_24680)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_24680 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_24666, 32) == 0 ||
                          !slt32(local_tid_24666, n_18153 *
                                 squot32(segred_group_sizze_20792,
                                         segment_sizze_nonzzero_24663)))) {
                        // read operands
                        {
                            x_20797 = x_20796;
                            x_20796 = ((__local
                                        float *) red_arr_mem_24670)[sext_i32_i64(squot32(local_tid_24666,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_24682 = slt32(srem32(local_tid_24666,
                                                               n_18153),
                                                        local_tid_24666 -
                                                        (squot32(local_tid_24666,
                                                                 32) * 32 - 1));
                            
                            if (inactive_24682) {
                                x_20796 = x_20797;
                            }
                            if (!inactive_24682) {
                                float res_20798 = x_20796 + x_20797;
                                
                                x_20796 = res_20798;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_24670)[sext_i32_i64(local_tid_24666)] =
                                x_20796;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_24666, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_24670)[sext_i32_i64(local_tid_24666)] =
                            x_20797;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_24674 * squot32(segred_group_sizze_20792,
                                                    segment_sizze_nonzzero_24663) +
                      local_tid_24666, m_18149 * k2p2zq_18165) &&
                slt32(local_tid_24666, squot32(segred_group_sizze_20792,
                                               segment_sizze_nonzzero_24663))) {
                ((__global
                  float *) mem_23878)[sext_i32_i64(squot32(virt_group_id_24674 *
                                                           squot32(segred_group_sizze_20792,
                                                                   segment_sizze_nonzzero_24663) +
                                                           local_tid_24666,
                                                           k2p2zq_18165)) *
                                      sext_i32_i64(k2p2zq_18165) +
                                      sext_i32_i64(virt_group_id_24674 *
                                      squot32(segred_group_sizze_20792,
                                              segment_sizze_nonzzero_24663) +
                                      local_tid_24666 -
                                      squot32(virt_group_id_24674 *
                                              squot32(segred_group_sizze_20792,
                                                      segment_sizze_nonzzero_24663) +
                                              local_tid_24666, k2p2zq_18165) *
                                      k2p2zq_18165)] = ((__local
                                                         float *) red_arr_mem_24670)[sext_i32_i64((local_tid_24666 +
                                                                                                   1) *
                                                                                     segment_sizze_nonzzero_24663 -
                                                                                     1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_20792
}
__kernel void mainzisegred_small_20886(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_24750_backing_aligned_0,
                                       int32_t m_18149, int32_t k2p2zq_18165,
                                       int32_t num_groups_20945, __global
                                       unsigned char *res_mem_23770, __global
                                       unsigned char *res_mem_23886, __global
                                       unsigned char *mem_23938,
                                       int32_t segment_sizze_nonzzero_24743)
{
    #define segred_group_sizze_20944 (mainzisegred_group_sizze_20880)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24750_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24750_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24745;
    int32_t local_tid_24746;
    int32_t group_sizze_24749;
    int32_t wave_sizze_24748;
    int32_t group_tid_24747;
    
    global_tid_24745 = get_global_id(0);
    local_tid_24746 = get_local_id(0);
    group_sizze_24749 = get_local_size(0);
    wave_sizze_24748 = LOCKSTEP_WIDTH;
    group_tid_24747 = get_group_id(0);
    
    int32_t phys_tid_20886;
    
    phys_tid_20886 = global_tid_24745;
    
    __local char *red_arr_mem_24750;
    
    red_arr_mem_24750 = (__local char *) red_arr_mem_24750_backing_0;
    
    int32_t phys_group_id_24752;
    
    phys_group_id_24752 = get_group_id(0);
    for (int32_t i_24753 = 0; i_24753 < sdiv_up32(sdiv_up32(m_18149 *
                                                            k2p2zq_18165,
                                                            squot32(segred_group_sizze_20944,
                                                                    segment_sizze_nonzzero_24743)) -
                                                  phys_group_id_24752,
                                                  num_groups_20945);
         i_24753++) {
        int32_t virt_group_id_24754 = phys_group_id_24752 + i_24753 *
                num_groups_20945;
        int32_t gtid_20872 = squot32(squot32(local_tid_24746,
                                             segment_sizze_nonzzero_24743) +
                                     virt_group_id_24754 *
                                     squot32(segred_group_sizze_20944,
                                             segment_sizze_nonzzero_24743),
                                     k2p2zq_18165);
        int32_t gtid_20873 = squot32(local_tid_24746,
                                     segment_sizze_nonzzero_24743) +
                virt_group_id_24754 * squot32(segred_group_sizze_20944,
                                              segment_sizze_nonzzero_24743) -
                squot32(squot32(local_tid_24746, segment_sizze_nonzzero_24743) +
                        virt_group_id_24754 * squot32(segred_group_sizze_20944,
                                                      segment_sizze_nonzzero_24743),
                        k2p2zq_18165) * k2p2zq_18165;
        int32_t gtid_20885 = srem32(local_tid_24746, k2p2zq_18165);
        
        // apply map function if in bounds
        {
            if (slt32(0, k2p2zq_18165) && ((slt32(gtid_20872, m_18149) &&
                                            slt32(gtid_20873, k2p2zq_18165)) &&
                                           slt32(local_tid_24746, k2p2zq_18165 *
                                                 squot32(segred_group_sizze_20944,
                                                         segment_sizze_nonzzero_24743)))) {
                float x_20954 = ((__global
                                  float *) res_mem_23886)[sext_i32_i64(gtid_20872) *
                                                          sext_i32_i64(k2p2zq_18165) +
                                                          sext_i32_i64(gtid_20885)];
                float x_20955 = ((__global
                                  float *) res_mem_23770)[sext_i32_i64(gtid_20872) *
                                                          sext_i32_i64(k2p2zq_18165 *
                                                          k2p2zq_18165) +
                                                          sext_i32_i64(gtid_20873) *
                                                          sext_i32_i64(k2p2zq_18165) +
                                                          sext_i32_i64(gtid_20885)];
                float res_20956 = x_20954 * x_20955;
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_24750)[sext_i32_i64(local_tid_24746)] =
                        res_20956;
                }
            } else {
                ((__local
                  float *) red_arr_mem_24750)[sext_i32_i64(local_tid_24746)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, k2p2zq_18165)) {
            // perform segmented scan to imitate reduction
            {
                float x_20948;
                float x_20949;
                float x_24755;
                float x_24756;
                int32_t skip_threads_24758;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_24746, k2p2zq_18165 *
                              squot32(segred_group_sizze_20944,
                                      segment_sizze_nonzzero_24743))) {
                        x_20949 = ((volatile __local
                                    float *) red_arr_mem_24750)[sext_i32_i64(local_tid_24746)];
                        if ((local_tid_24746 - squot32(local_tid_24746, 32) *
                             32) == 0) {
                            x_20948 = x_20949;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24758 = 1;
                    while (slt32(skip_threads_24758, 32)) {
                        if (sle32(skip_threads_24758, local_tid_24746 -
                                  squot32(local_tid_24746, 32) * 32) &&
                            slt32(local_tid_24746, k2p2zq_18165 *
                                  squot32(segred_group_sizze_20944,
                                          segment_sizze_nonzzero_24743))) {
                            // read operands
                            {
                                x_20948 = ((volatile __local
                                            float *) red_arr_mem_24750)[sext_i32_i64(local_tid_24746 -
                                                                        skip_threads_24758)];
                            }
                            // perform operation
                            {
                                bool inactive_24759 =
                                     slt32(srem32(local_tid_24746,
                                                  k2p2zq_18165),
                                           local_tid_24746 - (local_tid_24746 -
                                                              skip_threads_24758));
                                
                                if (inactive_24759) {
                                    x_20948 = x_20949;
                                }
                                if (!inactive_24759) {
                                    float res_20950 = x_20948 + x_20949;
                                    
                                    x_20948 = res_20950;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24748, skip_threads_24758)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24758, local_tid_24746 -
                                  squot32(local_tid_24746, 32) * 32) &&
                            slt32(local_tid_24746, k2p2zq_18165 *
                                  squot32(segred_group_sizze_20944,
                                          segment_sizze_nonzzero_24743))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24750)[sext_i32_i64(local_tid_24746)] =
                                    x_20948;
                                x_20949 = x_20948;
                            }
                        }
                        if (sle32(wave_sizze_24748, skip_threads_24758)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24758 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_24746 - squot32(local_tid_24746, 32) * 32) ==
                        31 && slt32(local_tid_24746, k2p2zq_18165 *
                                    squot32(segred_group_sizze_20944,
                                            segment_sizze_nonzzero_24743))) {
                        ((volatile __local
                          float *) red_arr_mem_24750)[sext_i32_i64(squot32(local_tid_24746,
                                                                           32))] =
                            x_20948;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_24760;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_24746, 32) == 0 &&
                            slt32(local_tid_24746, k2p2zq_18165 *
                                  squot32(segred_group_sizze_20944,
                                          segment_sizze_nonzzero_24743))) {
                            x_24756 = ((volatile __local
                                        float *) red_arr_mem_24750)[sext_i32_i64(local_tid_24746)];
                            if ((local_tid_24746 - squot32(local_tid_24746,
                                                           32) * 32) == 0) {
                                x_24755 = x_24756;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_24760 = 1;
                        while (slt32(skip_threads_24760, 32)) {
                            if (sle32(skip_threads_24760, local_tid_24746 -
                                      squot32(local_tid_24746, 32) * 32) &&
                                (squot32(local_tid_24746, 32) == 0 &&
                                 slt32(local_tid_24746, k2p2zq_18165 *
                                       squot32(segred_group_sizze_20944,
                                               segment_sizze_nonzzero_24743)))) {
                                // read operands
                                {
                                    x_24755 = ((volatile __local
                                                float *) red_arr_mem_24750)[sext_i32_i64(local_tid_24746 -
                                                                            skip_threads_24760)];
                                }
                                // perform operation
                                {
                                    bool inactive_24761 =
                                         slt32(srem32(local_tid_24746 * 32 +
                                                      32 - 1, k2p2zq_18165),
                                               local_tid_24746 * 32 + 32 - 1 -
                                               ((local_tid_24746 -
                                                 skip_threads_24760) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_24761) {
                                        x_24755 = x_24756;
                                    }
                                    if (!inactive_24761) {
                                        float res_24757 = x_24755 + x_24756;
                                        
                                        x_24755 = res_24757;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_24748, skip_threads_24760)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_24760, local_tid_24746 -
                                      squot32(local_tid_24746, 32) * 32) &&
                                (squot32(local_tid_24746, 32) == 0 &&
                                 slt32(local_tid_24746, k2p2zq_18165 *
                                       squot32(segred_group_sizze_20944,
                                               segment_sizze_nonzzero_24743)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_24750)[sext_i32_i64(local_tid_24746)] =
                                        x_24755;
                                    x_24756 = x_24755;
                                }
                            }
                            if (sle32(wave_sizze_24748, skip_threads_24760)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_24760 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_24746, 32) == 0 ||
                          !slt32(local_tid_24746, k2p2zq_18165 *
                                 squot32(segred_group_sizze_20944,
                                         segment_sizze_nonzzero_24743)))) {
                        // read operands
                        {
                            x_20949 = x_20948;
                            x_20948 = ((__local
                                        float *) red_arr_mem_24750)[sext_i32_i64(squot32(local_tid_24746,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_24762 = slt32(srem32(local_tid_24746,
                                                               k2p2zq_18165),
                                                        local_tid_24746 -
                                                        (squot32(local_tid_24746,
                                                                 32) * 32 - 1));
                            
                            if (inactive_24762) {
                                x_20948 = x_20949;
                            }
                            if (!inactive_24762) {
                                float res_20950 = x_20948 + x_20949;
                                
                                x_20948 = res_20950;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_24750)[sext_i32_i64(local_tid_24746)] =
                                x_20948;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_24746, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_24750)[sext_i32_i64(local_tid_24746)] =
                            x_20949;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_24754 * squot32(segred_group_sizze_20944,
                                                    segment_sizze_nonzzero_24743) +
                      local_tid_24746, m_18149 * k2p2zq_18165) &&
                slt32(local_tid_24746, squot32(segred_group_sizze_20944,
                                               segment_sizze_nonzzero_24743))) {
                ((__global
                  float *) mem_23938)[sext_i32_i64(squot32(virt_group_id_24754 *
                                                           squot32(segred_group_sizze_20944,
                                                                   segment_sizze_nonzzero_24743) +
                                                           local_tid_24746,
                                                           k2p2zq_18165)) *
                                      sext_i32_i64(k2p2zq_18165) +
                                      sext_i32_i64(virt_group_id_24754 *
                                      squot32(segred_group_sizze_20944,
                                              segment_sizze_nonzzero_24743) +
                                      local_tid_24746 -
                                      squot32(virt_group_id_24754 *
                                              squot32(segred_group_sizze_20944,
                                                      segment_sizze_nonzzero_24743) +
                                              local_tid_24746, k2p2zq_18165) *
                                      k2p2zq_18165)] = ((__local
                                                         float *) red_arr_mem_24750)[sext_i32_i64((local_tid_24746 +
                                                                                                   1) *
                                                                                     segment_sizze_nonzzero_24743 -
                                                                                     1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_20944
}
__kernel void mainzisegred_small_21035(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_24837_backing_aligned_0,
                                       int32_t N_18148, int32_t m_18149,
                                       int32_t k2p2zq_18165,
                                       int32_t num_groups_21092, __global
                                       unsigned char *mem_23547, __global
                                       unsigned char *res_mem_23946, __global
                                       unsigned char *mem_24059,
                                       int32_t segment_sizze_nonzzero_24830)
{
    #define segred_group_sizze_21091 (mainzisegred_group_sizze_21029)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_24837_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_24837_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24832;
    int32_t local_tid_24833;
    int32_t group_sizze_24836;
    int32_t wave_sizze_24835;
    int32_t group_tid_24834;
    
    global_tid_24832 = get_global_id(0);
    local_tid_24833 = get_local_id(0);
    group_sizze_24836 = get_local_size(0);
    wave_sizze_24835 = LOCKSTEP_WIDTH;
    group_tid_24834 = get_group_id(0);
    
    int32_t phys_tid_21035;
    
    phys_tid_21035 = global_tid_24832;
    
    __local char *red_arr_mem_24837;
    
    red_arr_mem_24837 = (__local char *) red_arr_mem_24837_backing_0;
    
    int32_t phys_group_id_24839;
    
    phys_group_id_24839 = get_group_id(0);
    for (int32_t i_24840 = 0; i_24840 < sdiv_up32(sdiv_up32(m_18149 * N_18148,
                                                            squot32(segred_group_sizze_21091,
                                                                    segment_sizze_nonzzero_24830)) -
                                                  phys_group_id_24839,
                                                  num_groups_21092);
         i_24840++) {
        int32_t virt_group_id_24841 = phys_group_id_24839 + i_24840 *
                num_groups_21092;
        int32_t gtid_21021 = squot32(squot32(local_tid_24833,
                                             segment_sizze_nonzzero_24830) +
                                     virt_group_id_24841 *
                                     squot32(segred_group_sizze_21091,
                                             segment_sizze_nonzzero_24830),
                                     N_18148);
        int32_t gtid_21022 = squot32(local_tid_24833,
                                     segment_sizze_nonzzero_24830) +
                virt_group_id_24841 * squot32(segred_group_sizze_21091,
                                              segment_sizze_nonzzero_24830) -
                squot32(squot32(local_tid_24833, segment_sizze_nonzzero_24830) +
                        virt_group_id_24841 * squot32(segred_group_sizze_21091,
                                                      segment_sizze_nonzzero_24830),
                        N_18148) * N_18148;
        int32_t gtid_21034 = srem32(local_tid_24833, k2p2zq_18165);
        
        // apply map function if in bounds
        {
            if (slt32(0, k2p2zq_18165) && ((slt32(gtid_21021, m_18149) &&
                                            slt32(gtid_21022, N_18148)) &&
                                           slt32(local_tid_24833, k2p2zq_18165 *
                                                 squot32(segred_group_sizze_21091,
                                                         segment_sizze_nonzzero_24830)))) {
                float x_21100 = ((__global
                                  float *) res_mem_23946)[sext_i32_i64(gtid_21021) *
                                                          sext_i32_i64(k2p2zq_18165) +
                                                          sext_i32_i64(gtid_21034)];
                float x_21101 = ((__global
                                  float *) mem_23547)[sext_i32_i64(gtid_21022) *
                                                      sext_i32_i64(k2p2zq_18165) +
                                                      sext_i32_i64(gtid_21034)];
                float res_21102 = x_21100 * x_21101;
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_24837)[sext_i32_i64(local_tid_24833)] =
                        res_21102;
                }
            } else {
                ((__local
                  float *) red_arr_mem_24837)[sext_i32_i64(local_tid_24833)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, k2p2zq_18165)) {
            // perform segmented scan to imitate reduction
            {
                float x_21095;
                float x_21096;
                float x_24842;
                float x_24843;
                int32_t skip_threads_24845;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_24833, k2p2zq_18165 *
                              squot32(segred_group_sizze_21091,
                                      segment_sizze_nonzzero_24830))) {
                        x_21096 = ((volatile __local
                                    float *) red_arr_mem_24837)[sext_i32_i64(local_tid_24833)];
                        if ((local_tid_24833 - squot32(local_tid_24833, 32) *
                             32) == 0) {
                            x_21095 = x_21096;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_24845 = 1;
                    while (slt32(skip_threads_24845, 32)) {
                        if (sle32(skip_threads_24845, local_tid_24833 -
                                  squot32(local_tid_24833, 32) * 32) &&
                            slt32(local_tid_24833, k2p2zq_18165 *
                                  squot32(segred_group_sizze_21091,
                                          segment_sizze_nonzzero_24830))) {
                            // read operands
                            {
                                x_21095 = ((volatile __local
                                            float *) red_arr_mem_24837)[sext_i32_i64(local_tid_24833 -
                                                                        skip_threads_24845)];
                            }
                            // perform operation
                            {
                                bool inactive_24846 =
                                     slt32(srem32(local_tid_24833,
                                                  k2p2zq_18165),
                                           local_tid_24833 - (local_tid_24833 -
                                                              skip_threads_24845));
                                
                                if (inactive_24846) {
                                    x_21095 = x_21096;
                                }
                                if (!inactive_24846) {
                                    float res_21097 = x_21095 + x_21096;
                                    
                                    x_21095 = res_21097;
                                }
                            }
                        }
                        if (sle32(wave_sizze_24835, skip_threads_24845)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_24845, local_tid_24833 -
                                  squot32(local_tid_24833, 32) * 32) &&
                            slt32(local_tid_24833, k2p2zq_18165 *
                                  squot32(segred_group_sizze_21091,
                                          segment_sizze_nonzzero_24830))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_24837)[sext_i32_i64(local_tid_24833)] =
                                    x_21095;
                                x_21096 = x_21095;
                            }
                        }
                        if (sle32(wave_sizze_24835, skip_threads_24845)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_24845 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_24833 - squot32(local_tid_24833, 32) * 32) ==
                        31 && slt32(local_tid_24833, k2p2zq_18165 *
                                    squot32(segred_group_sizze_21091,
                                            segment_sizze_nonzzero_24830))) {
                        ((volatile __local
                          float *) red_arr_mem_24837)[sext_i32_i64(squot32(local_tid_24833,
                                                                           32))] =
                            x_21095;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_24847;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_24833, 32) == 0 &&
                            slt32(local_tid_24833, k2p2zq_18165 *
                                  squot32(segred_group_sizze_21091,
                                          segment_sizze_nonzzero_24830))) {
                            x_24843 = ((volatile __local
                                        float *) red_arr_mem_24837)[sext_i32_i64(local_tid_24833)];
                            if ((local_tid_24833 - squot32(local_tid_24833,
                                                           32) * 32) == 0) {
                                x_24842 = x_24843;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_24847 = 1;
                        while (slt32(skip_threads_24847, 32)) {
                            if (sle32(skip_threads_24847, local_tid_24833 -
                                      squot32(local_tid_24833, 32) * 32) &&
                                (squot32(local_tid_24833, 32) == 0 &&
                                 slt32(local_tid_24833, k2p2zq_18165 *
                                       squot32(segred_group_sizze_21091,
                                               segment_sizze_nonzzero_24830)))) {
                                // read operands
                                {
                                    x_24842 = ((volatile __local
                                                float *) red_arr_mem_24837)[sext_i32_i64(local_tid_24833 -
                                                                            skip_threads_24847)];
                                }
                                // perform operation
                                {
                                    bool inactive_24848 =
                                         slt32(srem32(local_tid_24833 * 32 +
                                                      32 - 1, k2p2zq_18165),
                                               local_tid_24833 * 32 + 32 - 1 -
                                               ((local_tid_24833 -
                                                 skip_threads_24847) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_24848) {
                                        x_24842 = x_24843;
                                    }
                                    if (!inactive_24848) {
                                        float res_24844 = x_24842 + x_24843;
                                        
                                        x_24842 = res_24844;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_24835, skip_threads_24847)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_24847, local_tid_24833 -
                                      squot32(local_tid_24833, 32) * 32) &&
                                (squot32(local_tid_24833, 32) == 0 &&
                                 slt32(local_tid_24833, k2p2zq_18165 *
                                       squot32(segred_group_sizze_21091,
                                               segment_sizze_nonzzero_24830)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_24837)[sext_i32_i64(local_tid_24833)] =
                                        x_24842;
                                    x_24843 = x_24842;
                                }
                            }
                            if (sle32(wave_sizze_24835, skip_threads_24847)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_24847 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_24833, 32) == 0 ||
                          !slt32(local_tid_24833, k2p2zq_18165 *
                                 squot32(segred_group_sizze_21091,
                                         segment_sizze_nonzzero_24830)))) {
                        // read operands
                        {
                            x_21096 = x_21095;
                            x_21095 = ((__local
                                        float *) red_arr_mem_24837)[sext_i32_i64(squot32(local_tid_24833,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_24849 = slt32(srem32(local_tid_24833,
                                                               k2p2zq_18165),
                                                        local_tid_24833 -
                                                        (squot32(local_tid_24833,
                                                                 32) * 32 - 1));
                            
                            if (inactive_24849) {
                                x_21095 = x_21096;
                            }
                            if (!inactive_24849) {
                                float res_21097 = x_21095 + x_21096;
                                
                                x_21095 = res_21097;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_24837)[sext_i32_i64(local_tid_24833)] =
                                x_21095;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_24833, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_24837)[sext_i32_i64(local_tid_24833)] =
                            x_21096;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_24841 * squot32(segred_group_sizze_21091,
                                                    segment_sizze_nonzzero_24830) +
                      local_tid_24833, m_18149 * N_18148) &&
                slt32(local_tid_24833, squot32(segred_group_sizze_21091,
                                               segment_sizze_nonzzero_24830))) {
                ((__global
                  float *) mem_24059)[sext_i32_i64(squot32(virt_group_id_24841 *
                                                           squot32(segred_group_sizze_21091,
                                                                   segment_sizze_nonzzero_24830) +
                                                           local_tid_24833,
                                                           N_18148)) *
                                      sext_i32_i64(N_18148) +
                                      sext_i32_i64(virt_group_id_24841 *
                                      squot32(segred_group_sizze_21091,
                                              segment_sizze_nonzzero_24830) +
                                      local_tid_24833 -
                                      squot32(virt_group_id_24841 *
                                              squot32(segred_group_sizze_21091,
                                                      segment_sizze_nonzzero_24830) +
                                              local_tid_24833, N_18148) *
                                      N_18148)] = ((__local
                                                    float *) red_arr_mem_24837)[sext_i32_i64((local_tid_24833 +
                                                                                              1) *
                                                                                segment_sizze_nonzzero_24830 -
                                                                                1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21091
}
__kernel void mainzisegred_small_21614(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_25080_backing_aligned_0,
                                       int32_t N_18148, int32_t m_18149,
                                       int32_t n_18153,
                                       int32_t num_groups_21666, __global
                                       unsigned char *res_mem_24122, __global
                                       unsigned char *mem_24159, __global
                                       unsigned char *mem_24163,
                                       int32_t segment_sizze_nonzzero_25073)
{
    #define segred_group_sizze_21665 (mainzisegred_group_sizze_21608)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_25080_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_25080_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25075;
    int32_t local_tid_25076;
    int32_t group_sizze_25079;
    int32_t wave_sizze_25078;
    int32_t group_tid_25077;
    
    global_tid_25075 = get_global_id(0);
    local_tid_25076 = get_local_id(0);
    group_sizze_25079 = get_local_size(0);
    wave_sizze_25078 = LOCKSTEP_WIDTH;
    group_tid_25077 = get_group_id(0);
    
    int32_t phys_tid_21614;
    
    phys_tid_21614 = global_tid_25075;
    
    __local char *red_arr_mem_25080;
    
    red_arr_mem_25080 = (__local char *) red_arr_mem_25080_backing_0;
    
    int32_t phys_group_id_25082;
    
    phys_group_id_25082 = get_group_id(0);
    for (int32_t i_25083 = 0; i_25083 < sdiv_up32(sdiv_up32(m_18149,
                                                            squot32(segred_group_sizze_21665,
                                                                    segment_sizze_nonzzero_25073)) -
                                                  phys_group_id_25082,
                                                  num_groups_21666);
         i_25083++) {
        int32_t virt_group_id_25084 = phys_group_id_25082 + i_25083 *
                num_groups_21666;
        int32_t gtid_21603 = squot32(local_tid_25076,
                                     segment_sizze_nonzzero_25073) +
                virt_group_id_25084 * squot32(segred_group_sizze_21665,
                                              segment_sizze_nonzzero_25073);
        int32_t gtid_21613 = srem32(local_tid_25076, n_18153);
        
        // apply map function if in bounds
        {
            if (slt32(0, n_18153) && (slt32(gtid_21603, m_18149) &&
                                      slt32(local_tid_25076, n_18153 *
                                            squot32(segred_group_sizze_21665,
                                                    segment_sizze_nonzzero_25073)))) {
                int32_t res_21673 = ((__global
                                      int32_t *) mem_24159)[sext_i32_i64(gtid_21603)];
                bool cond_21676 = slt32(gtid_21613, res_21673);
                float res_21677;
                
                if (cond_21676) {
                    float x_elem_21675 = ((__global
                                           float *) res_mem_24122)[sext_i32_i64(gtid_21603) *
                                                                   sext_i32_i64(N_18148) +
                                                                   sext_i32_i64(gtid_21613)];
                    
                    res_21677 = x_elem_21675;
                } else {
                    res_21677 = 0.0F;
                }
                
                float res_21678 = res_21677 * res_21677;
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_25080)[sext_i32_i64(local_tid_25076)] =
                        res_21678;
                }
            } else {
                ((__local
                  float *) red_arr_mem_25080)[sext_i32_i64(local_tid_25076)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, n_18153)) {
            // perform segmented scan to imitate reduction
            {
                float x_21669;
                float x_21670;
                float x_25085;
                float x_25086;
                int32_t skip_threads_25088;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_25076, n_18153 *
                              squot32(segred_group_sizze_21665,
                                      segment_sizze_nonzzero_25073))) {
                        x_21670 = ((volatile __local
                                    float *) red_arr_mem_25080)[sext_i32_i64(local_tid_25076)];
                        if ((local_tid_25076 - squot32(local_tid_25076, 32) *
                             32) == 0) {
                            x_21669 = x_21670;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_25088 = 1;
                    while (slt32(skip_threads_25088, 32)) {
                        if (sle32(skip_threads_25088, local_tid_25076 -
                                  squot32(local_tid_25076, 32) * 32) &&
                            slt32(local_tid_25076, n_18153 *
                                  squot32(segred_group_sizze_21665,
                                          segment_sizze_nonzzero_25073))) {
                            // read operands
                            {
                                x_21669 = ((volatile __local
                                            float *) red_arr_mem_25080)[sext_i32_i64(local_tid_25076 -
                                                                        skip_threads_25088)];
                            }
                            // perform operation
                            {
                                bool inactive_25089 =
                                     slt32(srem32(local_tid_25076, n_18153),
                                           local_tid_25076 - (local_tid_25076 -
                                                              skip_threads_25088));
                                
                                if (inactive_25089) {
                                    x_21669 = x_21670;
                                }
                                if (!inactive_25089) {
                                    float res_21671 = x_21669 + x_21670;
                                    
                                    x_21669 = res_21671;
                                }
                            }
                        }
                        if (sle32(wave_sizze_25078, skip_threads_25088)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_25088, local_tid_25076 -
                                  squot32(local_tid_25076, 32) * 32) &&
                            slt32(local_tid_25076, n_18153 *
                                  squot32(segred_group_sizze_21665,
                                          segment_sizze_nonzzero_25073))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_25080)[sext_i32_i64(local_tid_25076)] =
                                    x_21669;
                                x_21670 = x_21669;
                            }
                        }
                        if (sle32(wave_sizze_25078, skip_threads_25088)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_25088 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_25076 - squot32(local_tid_25076, 32) * 32) ==
                        31 && slt32(local_tid_25076, n_18153 *
                                    squot32(segred_group_sizze_21665,
                                            segment_sizze_nonzzero_25073))) {
                        ((volatile __local
                          float *) red_arr_mem_25080)[sext_i32_i64(squot32(local_tid_25076,
                                                                           32))] =
                            x_21669;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_25090;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_25076, 32) == 0 &&
                            slt32(local_tid_25076, n_18153 *
                                  squot32(segred_group_sizze_21665,
                                          segment_sizze_nonzzero_25073))) {
                            x_25086 = ((volatile __local
                                        float *) red_arr_mem_25080)[sext_i32_i64(local_tid_25076)];
                            if ((local_tid_25076 - squot32(local_tid_25076,
                                                           32) * 32) == 0) {
                                x_25085 = x_25086;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_25090 = 1;
                        while (slt32(skip_threads_25090, 32)) {
                            if (sle32(skip_threads_25090, local_tid_25076 -
                                      squot32(local_tid_25076, 32) * 32) &&
                                (squot32(local_tid_25076, 32) == 0 &&
                                 slt32(local_tid_25076, n_18153 *
                                       squot32(segred_group_sizze_21665,
                                               segment_sizze_nonzzero_25073)))) {
                                // read operands
                                {
                                    x_25085 = ((volatile __local
                                                float *) red_arr_mem_25080)[sext_i32_i64(local_tid_25076 -
                                                                            skip_threads_25090)];
                                }
                                // perform operation
                                {
                                    bool inactive_25091 =
                                         slt32(srem32(local_tid_25076 * 32 +
                                                      32 - 1, n_18153),
                                               local_tid_25076 * 32 + 32 - 1 -
                                               ((local_tid_25076 -
                                                 skip_threads_25090) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_25091) {
                                        x_25085 = x_25086;
                                    }
                                    if (!inactive_25091) {
                                        float res_25087 = x_25085 + x_25086;
                                        
                                        x_25085 = res_25087;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_25078, skip_threads_25090)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_25090, local_tid_25076 -
                                      squot32(local_tid_25076, 32) * 32) &&
                                (squot32(local_tid_25076, 32) == 0 &&
                                 slt32(local_tid_25076, n_18153 *
                                       squot32(segred_group_sizze_21665,
                                               segment_sizze_nonzzero_25073)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_25080)[sext_i32_i64(local_tid_25076)] =
                                        x_25085;
                                    x_25086 = x_25085;
                                }
                            }
                            if (sle32(wave_sizze_25078, skip_threads_25090)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_25090 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_25076, 32) == 0 ||
                          !slt32(local_tid_25076, n_18153 *
                                 squot32(segred_group_sizze_21665,
                                         segment_sizze_nonzzero_25073)))) {
                        // read operands
                        {
                            x_21670 = x_21669;
                            x_21669 = ((__local
                                        float *) red_arr_mem_25080)[sext_i32_i64(squot32(local_tid_25076,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_25092 = slt32(srem32(local_tid_25076,
                                                               n_18153),
                                                        local_tid_25076 -
                                                        (squot32(local_tid_25076,
                                                                 32) * 32 - 1));
                            
                            if (inactive_25092) {
                                x_21669 = x_21670;
                            }
                            if (!inactive_25092) {
                                float res_21671 = x_21669 + x_21670;
                                
                                x_21669 = res_21671;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_25080)[sext_i32_i64(local_tid_25076)] =
                                x_21669;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_25076, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_25080)[sext_i32_i64(local_tid_25076)] =
                            x_21670;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_25084 * squot32(segred_group_sizze_21665,
                                                    segment_sizze_nonzzero_25073) +
                      local_tid_25076, m_18149) && slt32(local_tid_25076,
                                                         squot32(segred_group_sizze_21665,
                                                                 segment_sizze_nonzzero_25073))) {
                ((__global
                  float *) mem_24163)[sext_i32_i64(virt_group_id_25084 *
                                      squot32(segred_group_sizze_21665,
                                              segment_sizze_nonzzero_25073) +
                                      local_tid_25076)] = ((__local
                                                            float *) red_arr_mem_25080)[sext_i32_i64((local_tid_25076 +
                                                                                                      1) *
                                                                                        segment_sizze_nonzzero_25073 -
                                                                                        1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21665
}
__kernel void mainzisegred_small_21636(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_25021_backing_aligned_0,
                                       int32_t m_18149, int32_t N_18150,
                                       int32_t n_18153,
                                       int32_t num_groups_21650, __global
                                       unsigned char *images_mem_23523, __global
                                       unsigned char *mem_24159,
                                       int32_t segment_sizze_nonzzero_25014)
{
    #define segred_group_sizze_21649 (mainzisegred_group_sizze_21630)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_25021_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_25021_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25016;
    int32_t local_tid_25017;
    int32_t group_sizze_25020;
    int32_t wave_sizze_25019;
    int32_t group_tid_25018;
    
    global_tid_25016 = get_global_id(0);
    local_tid_25017 = get_local_id(0);
    group_sizze_25020 = get_local_size(0);
    wave_sizze_25019 = LOCKSTEP_WIDTH;
    group_tid_25018 = get_group_id(0);
    
    int32_t phys_tid_21636;
    
    phys_tid_21636 = global_tid_25016;
    
    __local char *red_arr_mem_25021;
    
    red_arr_mem_25021 = (__local char *) red_arr_mem_25021_backing_0;
    
    int32_t phys_group_id_25023;
    
    phys_group_id_25023 = get_group_id(0);
    for (int32_t i_25024 = 0; i_25024 < sdiv_up32(sdiv_up32(m_18149,
                                                            squot32(segred_group_sizze_21649,
                                                                    segment_sizze_nonzzero_25014)) -
                                                  phys_group_id_25023,
                                                  num_groups_21650);
         i_25024++) {
        int32_t virt_group_id_25025 = phys_group_id_25023 + i_25024 *
                num_groups_21650;
        int32_t gtid_21625 = squot32(local_tid_25017,
                                     segment_sizze_nonzzero_25014) +
                virt_group_id_25025 * squot32(segred_group_sizze_21649,
                                              segment_sizze_nonzzero_25014);
        int32_t gtid_21635 = srem32(local_tid_25017, n_18153);
        
        // apply map function if in bounds
        {
            if (slt32(0, n_18153) && (slt32(gtid_21625, m_18149) &&
                                      slt32(local_tid_25017, n_18153 *
                                            squot32(segred_group_sizze_21649,
                                                    segment_sizze_nonzzero_25014)))) {
                float x_21657 = ((__global
                                  float *) images_mem_23523)[sext_i32_i64(gtid_21625) *
                                                             sext_i32_i64(N_18150) +
                                                             sext_i32_i64(gtid_21635)];
                bool res_21658;
                
                res_21658 = futrts_isnan32(x_21657);
                
                bool cond_21659 = !res_21658;
                int32_t res_21660 = btoi_bool_i32(cond_21659);
                
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      int32_t *) red_arr_mem_25021)[sext_i32_i64(local_tid_25017)] =
                        res_21660;
                }
            } else {
                ((__local
                  int32_t *) red_arr_mem_25021)[sext_i32_i64(local_tid_25017)] =
                    0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, n_18153)) {
            // perform segmented scan to imitate reduction
            {
                int32_t x_21653;
                int32_t x_21654;
                int32_t x_25026;
                int32_t x_25027;
                int32_t skip_threads_25029;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_25017, n_18153 *
                              squot32(segred_group_sizze_21649,
                                      segment_sizze_nonzzero_25014))) {
                        x_21654 = ((volatile __local
                                    int32_t *) red_arr_mem_25021)[sext_i32_i64(local_tid_25017)];
                        if ((local_tid_25017 - squot32(local_tid_25017, 32) *
                             32) == 0) {
                            x_21653 = x_21654;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_25029 = 1;
                    while (slt32(skip_threads_25029, 32)) {
                        if (sle32(skip_threads_25029, local_tid_25017 -
                                  squot32(local_tid_25017, 32) * 32) &&
                            slt32(local_tid_25017, n_18153 *
                                  squot32(segred_group_sizze_21649,
                                          segment_sizze_nonzzero_25014))) {
                            // read operands
                            {
                                x_21653 = ((volatile __local
                                            int32_t *) red_arr_mem_25021)[sext_i32_i64(local_tid_25017 -
                                                                          skip_threads_25029)];
                            }
                            // perform operation
                            {
                                bool inactive_25030 =
                                     slt32(srem32(local_tid_25017, n_18153),
                                           local_tid_25017 - (local_tid_25017 -
                                                              skip_threads_25029));
                                
                                if (inactive_25030) {
                                    x_21653 = x_21654;
                                }
                                if (!inactive_25030) {
                                    int32_t res_21655 = add32(x_21653, x_21654);
                                    
                                    x_21653 = res_21655;
                                }
                            }
                        }
                        if (sle32(wave_sizze_25019, skip_threads_25029)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_25029, local_tid_25017 -
                                  squot32(local_tid_25017, 32) * 32) &&
                            slt32(local_tid_25017, n_18153 *
                                  squot32(segred_group_sizze_21649,
                                          segment_sizze_nonzzero_25014))) {
                            // write result
                            {
                                ((volatile __local
                                  int32_t *) red_arr_mem_25021)[sext_i32_i64(local_tid_25017)] =
                                    x_21653;
                                x_21654 = x_21653;
                            }
                        }
                        if (sle32(wave_sizze_25019, skip_threads_25029)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_25029 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_25017 - squot32(local_tid_25017, 32) * 32) ==
                        31 && slt32(local_tid_25017, n_18153 *
                                    squot32(segred_group_sizze_21649,
                                            segment_sizze_nonzzero_25014))) {
                        ((volatile __local
                          int32_t *) red_arr_mem_25021)[sext_i32_i64(squot32(local_tid_25017,
                                                                             32))] =
                            x_21653;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_25031;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_25017, 32) == 0 &&
                            slt32(local_tid_25017, n_18153 *
                                  squot32(segred_group_sizze_21649,
                                          segment_sizze_nonzzero_25014))) {
                            x_25027 = ((volatile __local
                                        int32_t *) red_arr_mem_25021)[sext_i32_i64(local_tid_25017)];
                            if ((local_tid_25017 - squot32(local_tid_25017,
                                                           32) * 32) == 0) {
                                x_25026 = x_25027;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_25031 = 1;
                        while (slt32(skip_threads_25031, 32)) {
                            if (sle32(skip_threads_25031, local_tid_25017 -
                                      squot32(local_tid_25017, 32) * 32) &&
                                (squot32(local_tid_25017, 32) == 0 &&
                                 slt32(local_tid_25017, n_18153 *
                                       squot32(segred_group_sizze_21649,
                                               segment_sizze_nonzzero_25014)))) {
                                // read operands
                                {
                                    x_25026 = ((volatile __local
                                                int32_t *) red_arr_mem_25021)[sext_i32_i64(local_tid_25017 -
                                                                              skip_threads_25031)];
                                }
                                // perform operation
                                {
                                    bool inactive_25032 =
                                         slt32(srem32(local_tid_25017 * 32 +
                                                      32 - 1, n_18153),
                                               local_tid_25017 * 32 + 32 - 1 -
                                               ((local_tid_25017 -
                                                 skip_threads_25031) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_25032) {
                                        x_25026 = x_25027;
                                    }
                                    if (!inactive_25032) {
                                        int32_t res_25028 = add32(x_25026,
                                                                  x_25027);
                                        
                                        x_25026 = res_25028;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_25019, skip_threads_25031)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_25031, local_tid_25017 -
                                      squot32(local_tid_25017, 32) * 32) &&
                                (squot32(local_tid_25017, 32) == 0 &&
                                 slt32(local_tid_25017, n_18153 *
                                       squot32(segred_group_sizze_21649,
                                               segment_sizze_nonzzero_25014)))) {
                                // write result
                                {
                                    ((volatile __local
                                      int32_t *) red_arr_mem_25021)[sext_i32_i64(local_tid_25017)] =
                                        x_25026;
                                    x_25027 = x_25026;
                                }
                            }
                            if (sle32(wave_sizze_25019, skip_threads_25031)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_25031 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_25017, 32) == 0 ||
                          !slt32(local_tid_25017, n_18153 *
                                 squot32(segred_group_sizze_21649,
                                         segment_sizze_nonzzero_25014)))) {
                        // read operands
                        {
                            x_21654 = x_21653;
                            x_21653 = ((__local
                                        int32_t *) red_arr_mem_25021)[sext_i32_i64(squot32(local_tid_25017,
                                                                                           32) -
                                                                      1)];
                        }
                        // perform operation
                        {
                            bool inactive_25033 = slt32(srem32(local_tid_25017,
                                                               n_18153),
                                                        local_tid_25017 -
                                                        (squot32(local_tid_25017,
                                                                 32) * 32 - 1));
                            
                            if (inactive_25033) {
                                x_21653 = x_21654;
                            }
                            if (!inactive_25033) {
                                int32_t res_21655 = add32(x_21653, x_21654);
                                
                                x_21653 = res_21655;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              int32_t *) red_arr_mem_25021)[sext_i32_i64(local_tid_25017)] =
                                x_21653;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_25017, 32) == 0) {
                        ((__local
                          int32_t *) red_arr_mem_25021)[sext_i32_i64(local_tid_25017)] =
                            x_21654;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_25025 * squot32(segred_group_sizze_21649,
                                                    segment_sizze_nonzzero_25014) +
                      local_tid_25017, m_18149) && slt32(local_tid_25017,
                                                         squot32(segred_group_sizze_21649,
                                                                 segment_sizze_nonzzero_25014))) {
                ((__global
                  int32_t *) mem_24159)[sext_i32_i64(virt_group_id_25025 *
                                        squot32(segred_group_sizze_21649,
                                                segment_sizze_nonzzero_25014) +
                                        local_tid_25017)] = ((__local
                                                              int32_t *) red_arr_mem_25021)[sext_i32_i64((local_tid_25017 +
                                                                                                          1) *
                                                                                            segment_sizze_nonzzero_25014 -
                                                                                            1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21649
}
__kernel void mainzisegred_small_21770(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_25184_backing_aligned_0,
                                       int32_t N_18148, int32_t m_18149,
                                       int32_t res_18475,
                                       int32_t num_groups_21789, __global
                                       unsigned char *res_mem_24122, __global
                                       unsigned char *res_mem_24174, __global
                                       unsigned char *res_mem_24175, __global
                                       unsigned char *mem_24187,
                                       int32_t segment_sizze_nonzzero_25177)
{
    #define segred_group_sizze_21788 (mainzisegred_group_sizze_21764)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_25184_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_25184_backing_aligned_0;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25179;
    int32_t local_tid_25180;
    int32_t group_sizze_25183;
    int32_t wave_sizze_25182;
    int32_t group_tid_25181;
    
    global_tid_25179 = get_global_id(0);
    local_tid_25180 = get_local_id(0);
    group_sizze_25183 = get_local_size(0);
    wave_sizze_25182 = LOCKSTEP_WIDTH;
    group_tid_25181 = get_group_id(0);
    
    int32_t phys_tid_21770;
    
    phys_tid_21770 = global_tid_25179;
    
    __local char *red_arr_mem_25184;
    
    red_arr_mem_25184 = (__local char *) red_arr_mem_25184_backing_0;
    
    int32_t phys_group_id_25186;
    
    phys_group_id_25186 = get_group_id(0);
    for (int32_t i_25187 = 0; i_25187 < sdiv_up32(sdiv_up32(m_18149,
                                                            squot32(segred_group_sizze_21788,
                                                                    segment_sizze_nonzzero_25177)) -
                                                  phys_group_id_25186,
                                                  num_groups_21789);
         i_25187++) {
        int32_t virt_group_id_25188 = phys_group_id_25186 + i_25187 *
                num_groups_21789;
        int32_t gtid_21759 = squot32(local_tid_25180,
                                     segment_sizze_nonzzero_25177) +
                virt_group_id_25188 * squot32(segred_group_sizze_21788,
                                              segment_sizze_nonzzero_25177);
        int32_t gtid_21769 = srem32(local_tid_25180, res_18475);
        
        // apply map function if in bounds
        {
            if (slt32(0, res_18475) && (slt32(gtid_21759, m_18149) &&
                                        slt32(local_tid_25180, res_18475 *
                                              squot32(segred_group_sizze_21788,
                                                      segment_sizze_nonzzero_25177)))) {
                int32_t x_21797 = ((__global
                                    int32_t *) res_mem_24174)[sext_i32_i64(gtid_21759)];
                bool cond_21799 = slt32(gtid_21769, x_21797);
                float res_21800;
                
                if (cond_21799) {
                    int32_t x_21796 = ((__global
                                        int32_t *) res_mem_24175)[sext_i32_i64(gtid_21759)];
                    int32_t x_21801 = add32(gtid_21769, x_21796);
                    int32_t x_21802 = sub32(x_21801, x_21797);
                    int32_t i_21803 = add32(1, x_21802);
                    float res_21804 = ((__global
                                        float *) res_mem_24122)[sext_i32_i64(gtid_21759) *
                                                                sext_i32_i64(N_18148) +
                                                                sext_i32_i64(i_21803)];
                    
                    res_21800 = res_21804;
                } else {
                    res_21800 = 0.0F;
                }
                // save map-out results
                { }
                // save results to be reduced
                {
                    ((__local
                      float *) red_arr_mem_25184)[sext_i32_i64(local_tid_25180)] =
                        res_21800;
                }
            } else {
                ((__local
                  float *) red_arr_mem_25184)[sext_i32_i64(local_tid_25180)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, res_18475)) {
            // perform segmented scan to imitate reduction
            {
                float x_21792;
                float x_21793;
                float x_25189;
                float x_25190;
                int32_t skip_threads_25192;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_25180, res_18475 *
                              squot32(segred_group_sizze_21788,
                                      segment_sizze_nonzzero_25177))) {
                        x_21793 = ((volatile __local
                                    float *) red_arr_mem_25184)[sext_i32_i64(local_tid_25180)];
                        if ((local_tid_25180 - squot32(local_tid_25180, 32) *
                             32) == 0) {
                            x_21792 = x_21793;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_25192 = 1;
                    while (slt32(skip_threads_25192, 32)) {
                        if (sle32(skip_threads_25192, local_tid_25180 -
                                  squot32(local_tid_25180, 32) * 32) &&
                            slt32(local_tid_25180, res_18475 *
                                  squot32(segred_group_sizze_21788,
                                          segment_sizze_nonzzero_25177))) {
                            // read operands
                            {
                                x_21792 = ((volatile __local
                                            float *) red_arr_mem_25184)[sext_i32_i64(local_tid_25180 -
                                                                        skip_threads_25192)];
                            }
                            // perform operation
                            {
                                bool inactive_25193 =
                                     slt32(srem32(local_tid_25180, res_18475),
                                           local_tid_25180 - (local_tid_25180 -
                                                              skip_threads_25192));
                                
                                if (inactive_25193) {
                                    x_21792 = x_21793;
                                }
                                if (!inactive_25193) {
                                    float res_21794 = x_21792 + x_21793;
                                    
                                    x_21792 = res_21794;
                                }
                            }
                        }
                        if (sle32(wave_sizze_25182, skip_threads_25192)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_25192, local_tid_25180 -
                                  squot32(local_tid_25180, 32) * 32) &&
                            slt32(local_tid_25180, res_18475 *
                                  squot32(segred_group_sizze_21788,
                                          segment_sizze_nonzzero_25177))) {
                            // write result
                            {
                                ((volatile __local
                                  float *) red_arr_mem_25184)[sext_i32_i64(local_tid_25180)] =
                                    x_21792;
                                x_21793 = x_21792;
                            }
                        }
                        if (sle32(wave_sizze_25182, skip_threads_25192)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_25192 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_25180 - squot32(local_tid_25180, 32) * 32) ==
                        31 && slt32(local_tid_25180, res_18475 *
                                    squot32(segred_group_sizze_21788,
                                            segment_sizze_nonzzero_25177))) {
                        ((volatile __local
                          float *) red_arr_mem_25184)[sext_i32_i64(squot32(local_tid_25180,
                                                                           32))] =
                            x_21792;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_25194;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_25180, 32) == 0 &&
                            slt32(local_tid_25180, res_18475 *
                                  squot32(segred_group_sizze_21788,
                                          segment_sizze_nonzzero_25177))) {
                            x_25190 = ((volatile __local
                                        float *) red_arr_mem_25184)[sext_i32_i64(local_tid_25180)];
                            if ((local_tid_25180 - squot32(local_tid_25180,
                                                           32) * 32) == 0) {
                                x_25189 = x_25190;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_25194 = 1;
                        while (slt32(skip_threads_25194, 32)) {
                            if (sle32(skip_threads_25194, local_tid_25180 -
                                      squot32(local_tid_25180, 32) * 32) &&
                                (squot32(local_tid_25180, 32) == 0 &&
                                 slt32(local_tid_25180, res_18475 *
                                       squot32(segred_group_sizze_21788,
                                               segment_sizze_nonzzero_25177)))) {
                                // read operands
                                {
                                    x_25189 = ((volatile __local
                                                float *) red_arr_mem_25184)[sext_i32_i64(local_tid_25180 -
                                                                            skip_threads_25194)];
                                }
                                // perform operation
                                {
                                    bool inactive_25195 =
                                         slt32(srem32(local_tid_25180 * 32 +
                                                      32 - 1, res_18475),
                                               local_tid_25180 * 32 + 32 - 1 -
                                               ((local_tid_25180 -
                                                 skip_threads_25194) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_25195) {
                                        x_25189 = x_25190;
                                    }
                                    if (!inactive_25195) {
                                        float res_25191 = x_25189 + x_25190;
                                        
                                        x_25189 = res_25191;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_25182, skip_threads_25194)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_25194, local_tid_25180 -
                                      squot32(local_tid_25180, 32) * 32) &&
                                (squot32(local_tid_25180, 32) == 0 &&
                                 slt32(local_tid_25180, res_18475 *
                                       squot32(segred_group_sizze_21788,
                                               segment_sizze_nonzzero_25177)))) {
                                // write result
                                {
                                    ((volatile __local
                                      float *) red_arr_mem_25184)[sext_i32_i64(local_tid_25180)] =
                                        x_25189;
                                    x_25190 = x_25189;
                                }
                            }
                            if (sle32(wave_sizze_25182, skip_threads_25194)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_25194 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_25180, 32) == 0 ||
                          !slt32(local_tid_25180, res_18475 *
                                 squot32(segred_group_sizze_21788,
                                         segment_sizze_nonzzero_25177)))) {
                        // read operands
                        {
                            x_21793 = x_21792;
                            x_21792 = ((__local
                                        float *) red_arr_mem_25184)[sext_i32_i64(squot32(local_tid_25180,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_25196 = slt32(srem32(local_tid_25180,
                                                               res_18475),
                                                        local_tid_25180 -
                                                        (squot32(local_tid_25180,
                                                                 32) * 32 - 1));
                            
                            if (inactive_25196) {
                                x_21792 = x_21793;
                            }
                            if (!inactive_25196) {
                                float res_21794 = x_21792 + x_21793;
                                
                                x_21792 = res_21794;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              float *) red_arr_mem_25184)[sext_i32_i64(local_tid_25180)] =
                                x_21792;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_25180, 32) == 0) {
                        ((__local
                          float *) red_arr_mem_25184)[sext_i32_i64(local_tid_25180)] =
                            x_21793;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_25188 * squot32(segred_group_sizze_21788,
                                                    segment_sizze_nonzzero_25177) +
                      local_tid_25180, m_18149) && slt32(local_tid_25180,
                                                         squot32(segred_group_sizze_21788,
                                                                 segment_sizze_nonzzero_25177))) {
                ((__global
                  float *) mem_24187)[sext_i32_i64(virt_group_id_25188 *
                                      squot32(segred_group_sizze_21788,
                                              segment_sizze_nonzzero_25177) +
                                      local_tid_25180)] = ((__local
                                                            float *) red_arr_mem_25184)[sext_i32_i64((local_tid_25180 +
                                                                                                      1) *
                                                                                        segment_sizze_nonzzero_25177 -
                                                                                        1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_21788
}
__kernel void mainzisegred_small_22379(__global int *global_failure,
                                       __local volatile
                                       int64_t *red_arr_mem_25342_backing_aligned_0,
                                       __local volatile
                                       int64_t *red_arr_mem_25340_backing_aligned_1,
                                       __local volatile
                                       int64_t *red_arr_mem_25338_backing_aligned_2,
                                       int32_t m_18149, int32_t iota_arg_18497,
                                       int32_t num_groups_22563, __global
                                       unsigned char *mem_24192, __global
                                       unsigned char *mem_24230, __global
                                       unsigned char *mem_24233, __global
                                       unsigned char *mem_24239, __global
                                       unsigned char *mem_24242, __global
                                       unsigned char *mem_24245, __global
                                       unsigned char *mem_24248, __global
                                       unsigned char *mem_24253,
                                       int32_t segment_sizze_nonzzero_25331)
{
    #define segred_group_sizze_22562 (mainzisegred_group_sizze_22373)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict red_arr_mem_25342_backing_2 =
                          (__local volatile
                           char *) red_arr_mem_25342_backing_aligned_0;
    __local volatile char *restrict red_arr_mem_25340_backing_1 =
                          (__local volatile
                           char *) red_arr_mem_25340_backing_aligned_1;
    __local volatile char *restrict red_arr_mem_25338_backing_0 =
                          (__local volatile
                           char *) red_arr_mem_25338_backing_aligned_2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_25333;
    int32_t local_tid_25334;
    int32_t group_sizze_25337;
    int32_t wave_sizze_25336;
    int32_t group_tid_25335;
    
    global_tid_25333 = get_global_id(0);
    local_tid_25334 = get_local_id(0);
    group_sizze_25337 = get_local_size(0);
    wave_sizze_25336 = LOCKSTEP_WIDTH;
    group_tid_25335 = get_group_id(0);
    
    int32_t phys_tid_22379;
    
    phys_tid_22379 = global_tid_25333;
    
    __local char *red_arr_mem_25338;
    
    red_arr_mem_25338 = (__local char *) red_arr_mem_25338_backing_0;
    
    __local char *red_arr_mem_25340;
    
    red_arr_mem_25340 = (__local char *) red_arr_mem_25340_backing_1;
    
    __local char *red_arr_mem_25342;
    
    red_arr_mem_25342 = (__local char *) red_arr_mem_25342_backing_2;
    
    int32_t phys_group_id_25344;
    
    phys_group_id_25344 = get_group_id(0);
    for (int32_t i_25345 = 0; i_25345 < sdiv_up32(sdiv_up32(m_18149,
                                                            squot32(segred_group_sizze_22562,
                                                                    segment_sizze_nonzzero_25331)) -
                                                  phys_group_id_25344,
                                                  num_groups_22563);
         i_25345++) {
        int32_t virt_group_id_25346 = phys_group_id_25344 + i_25345 *
                num_groups_22563;
        int32_t gtid_22368 = squot32(local_tid_25334,
                                     segment_sizze_nonzzero_25331) +
                virt_group_id_25346 * squot32(segred_group_sizze_22562,
                                              segment_sizze_nonzzero_25331);
        int32_t gtid_22378 = srem32(local_tid_25334, iota_arg_18497);
        
        // apply map function if in bounds
        {
            if (slt32(0, iota_arg_18497) && (slt32(gtid_22368, m_18149) &&
                                             slt32(local_tid_25334,
                                                   iota_arg_18497 *
                                                   squot32(segred_group_sizze_22562,
                                                           segment_sizze_nonzzero_25331)))) {
                int32_t y_22583 = ((__global
                                    int32_t *) mem_24233)[sext_i32_i64(gtid_22368)];
                float y_22584 = ((__global
                                  float *) mem_24230)[sext_i32_i64(gtid_22368)];
                float x_22588 = ((__global
                                  float *) mem_24239)[sext_i32_i64(gtid_22368) *
                                                      sext_i32_i64(iota_arg_18497) +
                                                      sext_i32_i64(gtid_22378)];
                float x_22589 = ((__global
                                  float *) mem_24192)[sext_i32_i64(gtid_22378)];
                float res_22592 = x_22588 / y_22584;
                bool cond_22593 = slt32(gtid_22378, y_22583);
                bool res_22594;
                
                res_22594 = futrts_isnan32(res_22592);
                
                bool res_22595 = !res_22594;
                bool x_22596 = cond_22593 && res_22595;
                float res_22597 = (float) fabs(res_22592);
                bool res_22598 = x_22589 < res_22597;
                bool x_22599 = x_22596 && res_22598;
                float res_22600;
                
                if (cond_22593) {
                    res_22600 = res_22592;
                } else {
                    res_22600 = 0.0F;
                }
                // save map-out results
                {
                    ((__global float *) mem_24253)[sext_i32_i64(gtid_22368) *
                                                   sext_i32_i64(iota_arg_18497) +
                                                   sext_i32_i64(gtid_22378)] =
                        res_22592;
                }
                // save results to be reduced
                {
                    ((__local
                      bool *) red_arr_mem_25338)[sext_i32_i64(local_tid_25334)] =
                        x_22599;
                    ((__local
                      int32_t *) red_arr_mem_25340)[sext_i32_i64(local_tid_25334)] =
                        gtid_22378;
                    ((__local
                      float *) red_arr_mem_25342)[sext_i32_i64(local_tid_25334)] =
                        res_22600;
                }
            } else {
                ((__local
                  bool *) red_arr_mem_25338)[sext_i32_i64(local_tid_25334)] = 0;
                ((__local
                  int32_t *) red_arr_mem_25340)[sext_i32_i64(local_tid_25334)] =
                    -1;
                ((__local
                  float *) red_arr_mem_25342)[sext_i32_i64(local_tid_25334)] =
                    0.0F;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (slt32(0, iota_arg_18497)) {
            // perform segmented scan to imitate reduction
            {
                bool x_22569;
                int32_t x_22570;
                float x_22571;
                bool x_22572;
                int32_t x_22573;
                float x_22574;
                bool x_25347;
                int32_t x_25348;
                float x_25349;
                bool x_25350;
                int32_t x_25351;
                float x_25352;
                int32_t skip_threads_25361;
                
                // read input for in-block scan
                {
                    if (slt32(local_tid_25334, iota_arg_18497 *
                              squot32(segred_group_sizze_22562,
                                      segment_sizze_nonzzero_25331))) {
                        x_22572 = ((volatile __local
                                    bool *) red_arr_mem_25338)[sext_i32_i64(local_tid_25334)];
                        x_22573 = ((volatile __local
                                    int32_t *) red_arr_mem_25340)[sext_i32_i64(local_tid_25334)];
                        x_22574 = ((volatile __local
                                    float *) red_arr_mem_25342)[sext_i32_i64(local_tid_25334)];
                        if ((local_tid_25334 - squot32(local_tid_25334, 32) *
                             32) == 0) {
                            x_22569 = x_22572;
                            x_22570 = x_22573;
                            x_22571 = x_22574;
                        }
                    }
                }
                // in-block scan (hopefully no barriers needed)
                {
                    skip_threads_25361 = 1;
                    while (slt32(skip_threads_25361, 32)) {
                        if (sle32(skip_threads_25361, local_tid_25334 -
                                  squot32(local_tid_25334, 32) * 32) &&
                            slt32(local_tid_25334, iota_arg_18497 *
                                  squot32(segred_group_sizze_22562,
                                          segment_sizze_nonzzero_25331))) {
                            // read operands
                            {
                                x_22569 = ((volatile __local
                                            bool *) red_arr_mem_25338)[sext_i32_i64(local_tid_25334 -
                                                                       skip_threads_25361)];
                                x_22570 = ((volatile __local
                                            int32_t *) red_arr_mem_25340)[sext_i32_i64(local_tid_25334 -
                                                                          skip_threads_25361)];
                                x_22571 = ((volatile __local
                                            float *) red_arr_mem_25342)[sext_i32_i64(local_tid_25334 -
                                                                        skip_threads_25361)];
                            }
                            // perform operation
                            {
                                bool inactive_25362 =
                                     slt32(srem32(local_tid_25334,
                                                  iota_arg_18497),
                                           local_tid_25334 - (local_tid_25334 -
                                                              skip_threads_25361));
                                
                                if (inactive_25362) {
                                    x_22569 = x_22572;
                                    x_22570 = x_22573;
                                    x_22571 = x_22574;
                                }
                                if (!inactive_25362) {
                                    bool res_22575;
                                    int32_t res_22576;
                                    
                                    if (x_22569) {
                                        res_22575 = x_22569;
                                        res_22576 = x_22570;
                                    } else {
                                        bool x_22577 = x_22572 && x_22572;
                                        bool x_22578 = !x_22572;
                                        bool y_22579 = x_22569 && x_22578;
                                        bool res_22580 = x_22577 || y_22579;
                                        int32_t res_22581;
                                        
                                        if (x_22572) {
                                            res_22581 = x_22573;
                                        } else {
                                            res_22581 = x_22570;
                                        }
                                        res_22575 = res_22580;
                                        res_22576 = res_22581;
                                    }
                                    
                                    float res_22582 = x_22571 + x_22574;
                                    
                                    x_22569 = res_22575;
                                    x_22570 = res_22576;
                                    x_22571 = res_22582;
                                }
                            }
                        }
                        if (sle32(wave_sizze_25336, skip_threads_25361)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        if (sle32(skip_threads_25361, local_tid_25334 -
                                  squot32(local_tid_25334, 32) * 32) &&
                            slt32(local_tid_25334, iota_arg_18497 *
                                  squot32(segred_group_sizze_22562,
                                          segment_sizze_nonzzero_25331))) {
                            // write result
                            {
                                ((volatile __local
                                  bool *) red_arr_mem_25338)[sext_i32_i64(local_tid_25334)] =
                                    x_22569;
                                x_22572 = x_22569;
                                ((volatile __local
                                  int32_t *) red_arr_mem_25340)[sext_i32_i64(local_tid_25334)] =
                                    x_22570;
                                x_22573 = x_22570;
                                ((volatile __local
                                  float *) red_arr_mem_25342)[sext_i32_i64(local_tid_25334)] =
                                    x_22571;
                                x_22574 = x_22571;
                            }
                        }
                        if (sle32(wave_sizze_25336, skip_threads_25361)) {
                            barrier(CLK_LOCAL_MEM_FENCE);
                        }
                        skip_threads_25361 *= 2;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // last thread of block 'i' writes its result to offset 'i'
                {
                    if ((local_tid_25334 - squot32(local_tid_25334, 32) * 32) ==
                        31 && slt32(local_tid_25334, iota_arg_18497 *
                                    squot32(segred_group_sizze_22562,
                                            segment_sizze_nonzzero_25331))) {
                        ((volatile __local
                          bool *) red_arr_mem_25338)[sext_i32_i64(squot32(local_tid_25334,
                                                                          32))] =
                            x_22569;
                        ((volatile __local
                          int32_t *) red_arr_mem_25340)[sext_i32_i64(squot32(local_tid_25334,
                                                                             32))] =
                            x_22570;
                        ((volatile __local
                          float *) red_arr_mem_25342)[sext_i32_i64(squot32(local_tid_25334,
                                                                           32))] =
                            x_22571;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'
                {
                    int32_t skip_threads_25363;
                    
                    // read input for in-block scan
                    {
                        if (squot32(local_tid_25334, 32) == 0 &&
                            slt32(local_tid_25334, iota_arg_18497 *
                                  squot32(segred_group_sizze_22562,
                                          segment_sizze_nonzzero_25331))) {
                            x_25350 = ((volatile __local
                                        bool *) red_arr_mem_25338)[sext_i32_i64(local_tid_25334)];
                            x_25351 = ((volatile __local
                                        int32_t *) red_arr_mem_25340)[sext_i32_i64(local_tid_25334)];
                            x_25352 = ((volatile __local
                                        float *) red_arr_mem_25342)[sext_i32_i64(local_tid_25334)];
                            if ((local_tid_25334 - squot32(local_tid_25334,
                                                           32) * 32) == 0) {
                                x_25347 = x_25350;
                                x_25348 = x_25351;
                                x_25349 = x_25352;
                            }
                        }
                    }
                    // in-block scan (hopefully no barriers needed)
                    {
                        skip_threads_25363 = 1;
                        while (slt32(skip_threads_25363, 32)) {
                            if (sle32(skip_threads_25363, local_tid_25334 -
                                      squot32(local_tid_25334, 32) * 32) &&
                                (squot32(local_tid_25334, 32) == 0 &&
                                 slt32(local_tid_25334, iota_arg_18497 *
                                       squot32(segred_group_sizze_22562,
                                               segment_sizze_nonzzero_25331)))) {
                                // read operands
                                {
                                    x_25347 = ((volatile __local
                                                bool *) red_arr_mem_25338)[sext_i32_i64(local_tid_25334 -
                                                                           skip_threads_25363)];
                                    x_25348 = ((volatile __local
                                                int32_t *) red_arr_mem_25340)[sext_i32_i64(local_tid_25334 -
                                                                              skip_threads_25363)];
                                    x_25349 = ((volatile __local
                                                float *) red_arr_mem_25342)[sext_i32_i64(local_tid_25334 -
                                                                            skip_threads_25363)];
                                }
                                // perform operation
                                {
                                    bool inactive_25364 =
                                         slt32(srem32(local_tid_25334 * 32 +
                                                      32 - 1, iota_arg_18497),
                                               local_tid_25334 * 32 + 32 - 1 -
                                               ((local_tid_25334 -
                                                 skip_threads_25363) * 32 + 32 -
                                                1));
                                    
                                    if (inactive_25364) {
                                        x_25347 = x_25350;
                                        x_25348 = x_25351;
                                        x_25349 = x_25352;
                                    }
                                    if (!inactive_25364) {
                                        bool res_25353;
                                        int32_t res_25354;
                                        
                                        if (x_25347) {
                                            res_25353 = x_25347;
                                            res_25354 = x_25348;
                                        } else {
                                            bool x_25355 = x_25350 && x_25350;
                                            bool x_25356 = !x_25350;
                                            bool y_25357 = x_25347 && x_25356;
                                            bool res_25358 = x_25355 || y_25357;
                                            int32_t res_25359;
                                            
                                            if (x_25350) {
                                                res_25359 = x_25351;
                                            } else {
                                                res_25359 = x_25348;
                                            }
                                            res_25353 = res_25358;
                                            res_25354 = res_25359;
                                        }
                                        
                                        float res_25360 = x_25349 + x_25352;
                                        
                                        x_25347 = res_25353;
                                        x_25348 = res_25354;
                                        x_25349 = res_25360;
                                    }
                                }
                            }
                            if (sle32(wave_sizze_25336, skip_threads_25363)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            if (sle32(skip_threads_25363, local_tid_25334 -
                                      squot32(local_tid_25334, 32) * 32) &&
                                (squot32(local_tid_25334, 32) == 0 &&
                                 slt32(local_tid_25334, iota_arg_18497 *
                                       squot32(segred_group_sizze_22562,
                                               segment_sizze_nonzzero_25331)))) {
                                // write result
                                {
                                    ((volatile __local
                                      bool *) red_arr_mem_25338)[sext_i32_i64(local_tid_25334)] =
                                        x_25347;
                                    x_25350 = x_25347;
                                    ((volatile __local
                                      int32_t *) red_arr_mem_25340)[sext_i32_i64(local_tid_25334)] =
                                        x_25348;
                                    x_25351 = x_25348;
                                    ((volatile __local
                                      float *) red_arr_mem_25342)[sext_i32_i64(local_tid_25334)] =
                                        x_25349;
                                    x_25352 = x_25349;
                                }
                            }
                            if (sle32(wave_sizze_25336, skip_threads_25363)) {
                                barrier(CLK_LOCAL_MEM_FENCE);
                            }
                            skip_threads_25363 *= 2;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // carry-in for every block except the first
                {
                    if (!(squot32(local_tid_25334, 32) == 0 ||
                          !slt32(local_tid_25334, iota_arg_18497 *
                                 squot32(segred_group_sizze_22562,
                                         segment_sizze_nonzzero_25331)))) {
                        // read operands
                        {
                            x_22572 = x_22569;
                            x_22573 = x_22570;
                            x_22574 = x_22571;
                            x_22569 = ((__local
                                        bool *) red_arr_mem_25338)[sext_i32_i64(squot32(local_tid_25334,
                                                                                        32) -
                                                                   1)];
                            x_22570 = ((__local
                                        int32_t *) red_arr_mem_25340)[sext_i32_i64(squot32(local_tid_25334,
                                                                                           32) -
                                                                      1)];
                            x_22571 = ((__local
                                        float *) red_arr_mem_25342)[sext_i32_i64(squot32(local_tid_25334,
                                                                                         32) -
                                                                    1)];
                        }
                        // perform operation
                        {
                            bool inactive_25365 = slt32(srem32(local_tid_25334,
                                                               iota_arg_18497),
                                                        local_tid_25334 -
                                                        (squot32(local_tid_25334,
                                                                 32) * 32 - 1));
                            
                            if (inactive_25365) {
                                x_22569 = x_22572;
                                x_22570 = x_22573;
                                x_22571 = x_22574;
                            }
                            if (!inactive_25365) {
                                bool res_22575;
                                int32_t res_22576;
                                
                                if (x_22569) {
                                    res_22575 = x_22569;
                                    res_22576 = x_22570;
                                } else {
                                    bool x_22577 = x_22572 && x_22572;
                                    bool x_22578 = !x_22572;
                                    bool y_22579 = x_22569 && x_22578;
                                    bool res_22580 = x_22577 || y_22579;
                                    int32_t res_22581;
                                    
                                    if (x_22572) {
                                        res_22581 = x_22573;
                                    } else {
                                        res_22581 = x_22570;
                                    }
                                    res_22575 = res_22580;
                                    res_22576 = res_22581;
                                }
                                
                                float res_22582 = x_22571 + x_22574;
                                
                                x_22569 = res_22575;
                                x_22570 = res_22576;
                                x_22571 = res_22582;
                            }
                        }
                        // write final result
                        {
                            ((__local
                              bool *) red_arr_mem_25338)[sext_i32_i64(local_tid_25334)] =
                                x_22569;
                            ((__local
                              int32_t *) red_arr_mem_25340)[sext_i32_i64(local_tid_25334)] =
                                x_22570;
                            ((__local
                              float *) red_arr_mem_25342)[sext_i32_i64(local_tid_25334)] =
                                x_22571;
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // restore correct values for first block
                {
                    if (squot32(local_tid_25334, 32) == 0) {
                        ((__local
                          bool *) red_arr_mem_25338)[sext_i32_i64(local_tid_25334)] =
                            x_22572;
                        ((__local
                          int32_t *) red_arr_mem_25340)[sext_i32_i64(local_tid_25334)] =
                            x_22573;
                        ((__local
                          float *) red_arr_mem_25342)[sext_i32_i64(local_tid_25334)] =
                            x_22574;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // save final values of segments
        {
            if (slt32(virt_group_id_25346 * squot32(segred_group_sizze_22562,
                                                    segment_sizze_nonzzero_25331) +
                      local_tid_25334, m_18149) && slt32(local_tid_25334,
                                                         squot32(segred_group_sizze_22562,
                                                                 segment_sizze_nonzzero_25331))) {
                ((__global bool *) mem_24242)[sext_i32_i64(virt_group_id_25346 *
                                              squot32(segred_group_sizze_22562,
                                                      segment_sizze_nonzzero_25331) +
                                              local_tid_25334)] = ((__local
                                                                    bool *) red_arr_mem_25338)[sext_i32_i64((local_tid_25334 +
                                                                                                             1) *
                                                                                               segment_sizze_nonzzero_25331 -
                                                                                               1)];
                ((__global
                  int32_t *) mem_24245)[sext_i32_i64(virt_group_id_25346 *
                                        squot32(segred_group_sizze_22562,
                                                segment_sizze_nonzzero_25331) +
                                        local_tid_25334)] = ((__local
                                                              int32_t *) red_arr_mem_25340)[sext_i32_i64((local_tid_25334 +
                                                                                                          1) *
                                                                                            segment_sizze_nonzzero_25331 -
                                                                                            1)];
                ((__global
                  float *) mem_24248)[sext_i32_i64(virt_group_id_25346 *
                                      squot32(segred_group_sizze_22562,
                                              segment_sizze_nonzzero_25331) +
                                      local_tid_25334)] = ((__local
                                                            float *) red_arr_mem_25342)[sext_i32_i64((local_tid_25334 +
                                                                                                      1) *
                                                                                        segment_sizze_nonzzero_25331 -
                                                                                        1)];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_1:
    return;
    #undef segred_group_sizze_22562
}
__kernel void remove_nanszisegmap_18758(__global int *global_failure,
                                        int32_t m_18134, int32_t n_18135,
                                        int32_t p_18136,
                                        int16_t nan_value_18137, __global
                                        unsigned char *images_mem_23522,
                                        __global unsigned char *mem_23530)
{
    #define segmap_group_sizze_18849 (remove_nanszisegmap_group_sizze_18765)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_24440;
    int32_t local_tid_24441;
    int32_t group_sizze_24444;
    int32_t wave_sizze_24443;
    int32_t group_tid_24442;
    
    global_tid_24440 = get_global_id(0);
    local_tid_24441 = get_local_id(0);
    group_sizze_24444 = get_local_size(0);
    wave_sizze_24443 = LOCKSTEP_WIDTH;
    group_tid_24442 = get_group_id(0);
    
    int32_t phys_tid_18758;
    
    phys_tid_18758 = global_tid_24440;
    
    int32_t gtid_18755;
    
    gtid_18755 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24442) *
                                      sext_i32_i64(segmap_group_sizze_18849) +
                                      sext_i32_i64(local_tid_24441),
                                      sext_i32_i64(n_18135) *
                                      sext_i32_i64(p_18136)));
    
    int32_t gtid_18756;
    
    gtid_18756 = sext_i64_i32(squot64(sext_i32_i64(group_tid_24442) *
                                      sext_i32_i64(segmap_group_sizze_18849) +
                                      sext_i32_i64(local_tid_24441) -
                                      squot64(sext_i32_i64(group_tid_24442) *
                                              sext_i32_i64(segmap_group_sizze_18849) +
                                              sext_i32_i64(local_tid_24441),
                                              sext_i32_i64(n_18135) *
                                              sext_i32_i64(p_18136)) *
                                      (sext_i32_i64(n_18135) *
                                       sext_i32_i64(p_18136)),
                                      sext_i32_i64(p_18136)));
    
    int32_t gtid_18757;
    
    gtid_18757 = sext_i64_i32(sext_i32_i64(group_tid_24442) *
        sext_i32_i64(segmap_group_sizze_18849) + sext_i32_i64(local_tid_24441) -
        squot64(sext_i32_i64(group_tid_24442) *
                sext_i32_i64(segmap_group_sizze_18849) +
                sext_i32_i64(local_tid_24441), sext_i32_i64(n_18135) *
                sext_i32_i64(p_18136)) * (sext_i32_i64(n_18135) *
                                          sext_i32_i64(p_18136)) -
        squot64(sext_i32_i64(group_tid_24442) *
                sext_i32_i64(segmap_group_sizze_18849) +
                sext_i32_i64(local_tid_24441) -
                squot64(sext_i32_i64(group_tid_24442) *
                        sext_i32_i64(segmap_group_sizze_18849) +
                        sext_i32_i64(local_tid_24441), sext_i32_i64(n_18135) *
                        sext_i32_i64(p_18136)) * (sext_i32_i64(n_18135) *
                                                  sext_i32_i64(p_18136)),
                sext_i32_i64(p_18136)) * sext_i32_i64(p_18136));
    if ((slt32(gtid_18755, m_18134) && slt32(gtid_18756, n_18135)) &&
        slt32(gtid_18757, p_18136)) {
        int16_t x_18854 = ((__global
                            int16_t *) images_mem_23522)[sext_i32_i64(gtid_18755) *
                                                         sext_i32_i64(p_18136 *
                                                         n_18135) +
                                                         sext_i32_i64(gtid_18756) *
                                                         sext_i32_i64(p_18136) +
                                                         sext_i32_i64(gtid_18757)];
        bool cond_18855 = x_18854 == nan_value_18137;
        float res_18856;
        
        if (cond_18855) {
            res_18856 = NAN;
        } else {
            float res_18857 = sitofp_i16_f32(x_18854);
            
            res_18856 = res_18857;
        }
        ((__global float *) mem_23530)[sext_i32_i64(gtid_18755) *
                                       sext_i32_i64(p_18136 * n_18135) +
                                       sext_i32_i64(gtid_18756) *
                                       sext_i32_i64(p_18136) +
                                       sext_i32_i64(gtid_18757)] = res_18856;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_18849
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
class bfastfinaldetailed:
  entry_points = {"main": (["i32", "i32", "i32", "f32", "f32", "f32", "[]i32",
                            "[][]f32"], ["[]f32", "[]i32", "[]i32", "[]f32",
                                         "[][]f32", "[][]f32", "[]f32", "[]i32",
                                         "[]f32", "[][]f32", "[][]f32"]),
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
    self.failure_msgs=["Index [{}] out of bounds for array of shape [{}].\n-> #0  helpers.fut:52:16-19\n   #1  helpers.fut:72:15-33\n   #2  bfastfinaldetailed.fut:52:35-50\n   #3  bfastfinaldetailed.fut:19:1-146:86\n"]
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
                                       all_sizes={"builtin#replicate_f32.group_size_24899": {"class": "group_size",
                                                                                   "value": None},
                                        "builtin#replicate_i32.group_size_24908": {"class": "group_size",
                                                                                   "value": None},
                                        "main.group_size_24606": {"class": "group_size", "value": None},
                                        "main.group_size_24983": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_18957": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19157": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19310": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19357": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19407": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19441": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_19958": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20185": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20242": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20306": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20416": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20654": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20814": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20854": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_20965": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_21235": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_21484": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_21589": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_21719": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_21850": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_22247": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_22311": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_22343": {"class": "group_size", "value": None},
                                        "main.segmap_group_size_22474": {"class": "group_size", "value": None},
                                        "main.segmap_num_groups_19359": {"class": "num_groups", "value": None},
                                        "main.segmap_num_groups_19409": {"class": "num_groups", "value": None},
                                        "main.segmap_num_groups_20656": {"class": "num_groups", "value": None},
                                        "main.segmap_num_groups_20816": {"class": "num_groups", "value": None},
                                        "main.segmap_num_groups_20967": {"class": "num_groups", "value": None},
                                        "main.segmap_num_groups_22476": {"class": "num_groups", "value": None},
                                        "main.segred_group_size_19476": {"class": "group_size", "value": None},
                                        "main.segred_group_size_20724": {"class": "group_size", "value": None},
                                        "main.segred_group_size_20880": {"class": "group_size", "value": None},
                                        "main.segred_group_size_21029": {"class": "group_size", "value": None},
                                        "main.segred_group_size_21608": {"class": "group_size", "value": None},
                                        "main.segred_group_size_21630": {"class": "group_size", "value": None},
                                        "main.segred_group_size_21704": {"class": "group_size", "value": None},
                                        "main.segred_group_size_21764": {"class": "group_size", "value": None},
                                        "main.segred_group_size_22373": {"class": "group_size", "value": None},
                                        "main.segred_num_groups_19478": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_20726": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_20882": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_21031": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_21610": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_21632": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_21706": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_21766": {"class": "num_groups", "value": None},
                                        "main.segred_num_groups_22375": {"class": "num_groups", "value": None},
                                        "main.segscan_group_size_21349": {"class": "group_size", "value": None},
                                        "main.segscan_group_size_22432": {"class": "group_size", "value": None},
                                        "main.segscan_num_groups_21351": {"class": "num_groups", "value": None},
                                        "main.segscan_num_groups_22434": {"class": "num_groups", "value": None},
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
                                        "main.tile_size_22833": {"class": "tile_size", "value": None},
                                        "main.tile_size_23155": {"class": "tile_size", "value": None},
                                        "remove_nans.segmap_group_size_18765": {"class": "group_size", "value": None}})
    self.builtinzhreplicate_f32zireplicate_24896_var = program.builtinzhreplicate_f32zireplicate_24896
    self.builtinzhreplicate_i32zireplicate_24905_var = program.builtinzhreplicate_i32zireplicate_24905
    self.gpu_map_transpose_f32_var = program.gpu_map_transpose_f32
    self.gpu_map_transpose_f32_low_height_var = program.gpu_map_transpose_f32_low_height
    self.gpu_map_transpose_f32_low_width_var = program.gpu_map_transpose_f32_low_width
    self.gpu_map_transpose_f32_small_var = program.gpu_map_transpose_f32_small
    self.mainzicopy_24603_var = program.mainzicopy_24603
    self.mainzicopy_24980_var = program.mainzicopy_24980
    self.mainziscan_stage1_21355_var = program.mainziscan_stage1_21355
    self.mainziscan_stage1_22438_var = program.mainziscan_stage1_22438
    self.mainziscan_stage2_21355_var = program.mainziscan_stage2_21355
    self.mainziscan_stage2_22438_var = program.mainziscan_stage2_22438
    self.mainziscan_stage3_21355_var = program.mainziscan_stage3_21355
    self.mainziscan_stage3_22438_var = program.mainziscan_stage3_22438
    self.mainzisegmap_18952_var = program.mainzisegmap_18952
    self.mainzisegmap_19152_var = program.mainzisegmap_19152
    self.mainzisegmap_19305_var = program.mainzisegmap_19305
    self.mainzisegmap_19354_var = program.mainzisegmap_19354
    self.mainzisegmap_19402_var = program.mainzisegmap_19402
    self.mainzisegmap_19434_var = program.mainzisegmap_19434
    self.mainzisegmap_19951_var = program.mainzisegmap_19951
    self.mainzisegmap_20180_var = program.mainzisegmap_20180
    self.mainzisegmap_20237_var = program.mainzisegmap_20237
    self.mainzisegmap_20303_var = program.mainzisegmap_20303
    self.mainzisegmap_20411_var = program.mainzisegmap_20411
    self.mainzisegmap_20651_var = program.mainzisegmap_20651
    self.mainzisegmap_20811_var = program.mainzisegmap_20811
    self.mainzisegmap_20849_var = program.mainzisegmap_20849
    self.mainzisegmap_20962_var = program.mainzisegmap_20962
    self.mainzisegmap_21230_var = program.mainzisegmap_21230
    self.mainzisegmap_21481_var = program.mainzisegmap_21481
    self.mainzisegmap_21586_var = program.mainzisegmap_21586
    self.mainzisegmap_21716_var = program.mainzisegmap_21716
    self.mainzisegmap_21847_var = program.mainzisegmap_21847
    self.mainzisegmap_22242_var = program.mainzisegmap_22242
    self.mainzisegmap_22308_var = program.mainzisegmap_22308
    self.mainzisegmap_22340_var = program.mainzisegmap_22340
    self.mainzisegmap_22471_var = program.mainzisegmap_22471
    self.mainzisegmap_intragroup_19722_var = program.mainzisegmap_intragroup_19722
    self.mainzisegmap_intragroup_20074_var = program.mainzisegmap_intragroup_20074
    self.mainzisegmap_intragroup_21114_var = program.mainzisegmap_intragroup_21114
    self.mainzisegmap_intragroup_21479_var = program.mainzisegmap_intragroup_21479
    self.mainzisegmap_intragroup_21895_var = program.mainzisegmap_intragroup_21895
    self.mainzisegmap_intragroup_22839_var = program.mainzisegmap_intragroup_22839
    self.mainzisegmap_intragroup_23161_var = program.mainzisegmap_intragroup_23161
    self.mainzisegred_large_19482_var = program.mainzisegred_large_19482
    self.mainzisegred_large_20730_var = program.mainzisegred_large_20730
    self.mainzisegred_large_20886_var = program.mainzisegred_large_20886
    self.mainzisegred_large_21035_var = program.mainzisegred_large_21035
    self.mainzisegred_large_21614_var = program.mainzisegred_large_21614
    self.mainzisegred_large_21636_var = program.mainzisegred_large_21636
    self.mainzisegred_large_21770_var = program.mainzisegred_large_21770
    self.mainzisegred_large_22379_var = program.mainzisegred_large_22379
    self.mainzisegred_nonseg_21712_var = program.mainzisegred_nonseg_21712
    self.mainzisegred_small_19482_var = program.mainzisegred_small_19482
    self.mainzisegred_small_20730_var = program.mainzisegred_small_20730
    self.mainzisegred_small_20886_var = program.mainzisegred_small_20886
    self.mainzisegred_small_21035_var = program.mainzisegred_small_21035
    self.mainzisegred_small_21614_var = program.mainzisegred_small_21614
    self.mainzisegred_small_21636_var = program.mainzisegred_small_21636
    self.mainzisegred_small_21770_var = program.mainzisegred_small_21770
    self.mainzisegred_small_22379_var = program.mainzisegred_small_22379
    self.remove_nanszisegmap_18758_var = program.remove_nanszisegmap_18758
    self.constants = {}
    mainzicounter_mem_24537 = np.zeros(10240, dtype=np.int32)
    static_mem_25444 = opencl_alloc(self, 40960, "static_mem_25444")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25444,
                      normaliseArray(mainzicounter_mem_24537),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24537 = static_mem_25444
    mainzicounter_mem_24690 = np.zeros(10240, dtype=np.int32)
    static_mem_25447 = opencl_alloc(self, 40960, "static_mem_25447")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25447,
                      normaliseArray(mainzicounter_mem_24690),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24690 = static_mem_25447
    mainzicounter_mem_24770 = np.zeros(10240, dtype=np.int32)
    static_mem_25448 = opencl_alloc(self, 40960, "static_mem_25448")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25448,
                      normaliseArray(mainzicounter_mem_24770),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24770 = static_mem_25448
    mainzicounter_mem_24857 = np.zeros(10240, dtype=np.int32)
    static_mem_25449 = opencl_alloc(self, 40960, "static_mem_25449")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25449,
                      normaliseArray(mainzicounter_mem_24857),
                      is_blocking=synchronous)
    self.mainzicounter_mem_24857 = static_mem_25449
    mainzicounter_mem_25041 = np.zeros(10240, dtype=np.int32)
    static_mem_25450 = opencl_alloc(self, 40960, "static_mem_25450")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25450,
                      normaliseArray(mainzicounter_mem_25041),
                      is_blocking=synchronous)
    self.mainzicounter_mem_25041 = static_mem_25450
    mainzicounter_mem_25100 = np.zeros(10240, dtype=np.int32)
    static_mem_25451 = opencl_alloc(self, 40960, "static_mem_25451")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25451,
                      normaliseArray(mainzicounter_mem_25100),
                      is_blocking=synchronous)
    self.mainzicounter_mem_25100 = static_mem_25451
    mainzicounter_mem_25140 = np.array([np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0), np.int32(0), np.int32(0),
                                        np.int32(0)], dtype=np.int32)
    static_mem_25452 = opencl_alloc(self, 40, "static_mem_25452")
    if (40 != 0):
      cl.enqueue_copy(self.queue, static_mem_25452,
                      normaliseArray(mainzicounter_mem_25140),
                      is_blocking=synchronous)
    self.mainzicounter_mem_25140 = static_mem_25452
    mainzicounter_mem_25204 = np.zeros(10240, dtype=np.int32)
    static_mem_25454 = opencl_alloc(self, 40960, "static_mem_25454")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25454,
                      normaliseArray(mainzicounter_mem_25204),
                      is_blocking=synchronous)
    self.mainzicounter_mem_25204 = static_mem_25454
    mainzicounter_mem_25377 = np.zeros(10240, dtype=np.int32)
    static_mem_25456 = opencl_alloc(self, 40960, "static_mem_25456")
    if (40960 != 0):
      cl.enqueue_copy(self.queue, static_mem_25456,
                      normaliseArray(mainzicounter_mem_25377),
                      is_blocking=synchronous)
    self.mainzicounter_mem_25377 = static_mem_25456
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
  def futhark_builtinzhreplicate_f32(self, mem_24892, num_elems_24893,
                                     val_24894):
    group_sizze_24899 = self.sizes["builtin#replicate_f32.group_size_24899"]
    num_groups_24900 = sdiv_up64(sext_i32_i64(num_elems_24893),
                                 sext_i32_i64(group_sizze_24899))
    if ((1 * (np.long(sext_i64_i32(num_groups_24900)) * np.long(group_sizze_24899))) != 0):
      self.builtinzhreplicate_f32zireplicate_24896_var.set_args(mem_24892,
                                                                np.int32(num_elems_24893),
                                                                np.float32(val_24894))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_f32zireplicate_24896_var,
                                 ((np.long(sext_i64_i32(num_groups_24900)) * np.long(group_sizze_24899)),),
                                 (np.long(group_sizze_24899),))
      if synchronous:
        sync(self)
    return ()
  def futhark_builtinzhreplicate_i32(self, mem_24901, num_elems_24902,
                                     val_24903):
    group_sizze_24908 = self.sizes["builtin#replicate_i32.group_size_24908"]
    num_groups_24909 = sdiv_up64(sext_i32_i64(num_elems_24902),
                                 sext_i32_i64(group_sizze_24908))
    if ((1 * (np.long(sext_i64_i32(num_groups_24909)) * np.long(group_sizze_24908))) != 0):
      self.builtinzhreplicate_i32zireplicate_24905_var.set_args(mem_24901,
                                                                np.int32(num_elems_24902),
                                                                np.int32(val_24903))
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_i32zireplicate_24905_var,
                                 ((np.long(sext_i64_i32(num_groups_24909)) * np.long(group_sizze_24908)),),
                                 (np.long(group_sizze_24908),))
      if synchronous:
        sync(self)
    return ()
  def futhark_main(self, mappingindices_mem_23522, images_mem_23523, N_18148,
                   m_18149, N_18150, trend_18151, k_18152, n_18153, freq_18154,
                   hfrac_18155, lam_18156):
    dim_match_18159 = (N_18148 == N_18150)
    empty_or_match_cert_18160 = True
    assert dim_match_18159, ("Error: %s\n\nBacktrace:\n-> #0  bfastfinaldetailed.fut:19:1-146:86\n" % ("function arguments of wrong shape",))
    x_18162 = (np.int32(2) * k_18152)
    k2p2_18163 = (np.int32(2) + x_18162)
    cond_18164 = slt32(np.int32(0), trend_18151)
    if cond_18164:
      k2p2zq_18165 = k2p2_18163
    else:
      res_18166 = (k2p2_18163 - np.int32(1))
      k2p2zq_18165 = res_18166
    binop_x_23526 = sext_i32_i64(k2p2zq_18165)
    binop_y_23527 = sext_i32_i64(N_18148)
    binop_x_23528 = (binop_x_23526 * binop_y_23527)
    bytes_23525 = (np.int64(4) * binop_x_23528)
    if cond_18164:
      bounds_invalid_upwards_18168 = slt32(k2p2zq_18165, np.int32(0))
      valid_18169 = not(bounds_invalid_upwards_18168)
      range_valid_c_18170 = True
      assert valid_18169, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  helpers.fut:31:10-18\n   #2  bfastfinaldetailed.fut:30:16-55\n   #3  bfastfinaldetailed.fut:19:1-146:86\n" % ("Range ",
                                                                                                                                                                                                                         np.int32(0),
                                                                                                                                                                                                                         "..",
                                                                                                                                                                                                                         np.int32(1),
                                                                                                                                                                                                                         "..<",
                                                                                                                                                                                                                         k2p2zq_18165,
                                                                                                                                                                                                                         " is invalid."))
      segmap_group_sizze_19042 = self.sizes["main.segmap_group_size_18957"]
      segmap_group_sizze_19043 = sext_i32_i64(segmap_group_sizze_19042)
      segmap_usable_groups_64_19044 = sdiv_up64(binop_x_23528,
                                                segmap_group_sizze_19043)
      segmap_usable_groups_19045 = sext_i64_i32(segmap_usable_groups_64_19044)
      mem_23529 = opencl_alloc(self, bytes_23525, "mem_23529")
      if ((1 * (np.long(segmap_usable_groups_19045) * np.long(segmap_group_sizze_19042))) != 0):
        self.mainzisegmap_18952_var.set_args(self.global_failure,
                                             np.int32(N_18148),
                                             np.float32(freq_18154),
                                             np.int32(k2p2zq_18165),
                                             mappingindices_mem_23522,
                                             mem_23529)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_18952_var,
                                   ((np.long(segmap_usable_groups_19045) * np.long(segmap_group_sizze_19042)),),
                                   (np.long(segmap_group_sizze_19042),))
        if synchronous:
          sync(self)
      binop_p_mem_23536 = mem_23529
    else:
      bounds_invalid_upwards_18193 = slt32(k2p2zq_18165, np.int32(0))
      valid_18194 = not(bounds_invalid_upwards_18193)
      range_valid_c_18195 = True
      assert valid_18194, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  helpers.fut:43:10-20\n   #2  bfastfinaldetailed.fut:31:10-49\n   #3  bfastfinaldetailed.fut:19:1-146:86\n" % ("Range ",
                                                                                                                                                                                                                         np.int32(0),
                                                                                                                                                                                                                         "..",
                                                                                                                                                                                                                         np.int32(1),
                                                                                                                                                                                                                         "..<",
                                                                                                                                                                                                                         k2p2zq_18165,
                                                                                                                                                                                                                         " is invalid."))
      segmap_group_sizze_19238 = self.sizes["main.segmap_group_size_19157"]
      segmap_group_sizze_19239 = sext_i32_i64(segmap_group_sizze_19238)
      segmap_usable_groups_64_19240 = sdiv_up64(binop_x_23528,
                                                segmap_group_sizze_19239)
      segmap_usable_groups_19241 = sext_i64_i32(segmap_usable_groups_64_19240)
      mem_23535 = opencl_alloc(self, bytes_23525, "mem_23535")
      if ((1 * (np.long(segmap_usable_groups_19241) * np.long(segmap_group_sizze_19238))) != 0):
        self.mainzisegmap_19152_var.set_args(self.global_failure,
                                             np.int32(N_18148),
                                             np.float32(freq_18154),
                                             np.int32(k2p2zq_18165),
                                             mappingindices_mem_23522,
                                             mem_23535)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19152_var,
                                   ((np.long(segmap_usable_groups_19241) * np.long(segmap_group_sizze_19238)),),
                                   (np.long(segmap_group_sizze_19238),))
        if synchronous:
          sync(self)
      binop_p_mem_23536 = mem_23535
    x_18217 = (N_18148 * N_18148)
    y_18218 = (np.int32(2) * N_18148)
    x_18219 = (x_18217 + y_18218)
    x_18220 = (np.int32(1) + x_18219)
    y_18221 = (np.int32(1) + N_18148)
    zzero_18222 = (y_18221 == np.int32(0))
    nonzzero_18223 = not(zzero_18222)
    nonzzero_cert_18224 = True
    assert nonzzero_18223, ("Error: %s\n\nBacktrace:\n-> #0  bfastfinaldetailed.fut:37:21-45\n   #1  bfastfinaldetailed.fut:19:1-146:86\n" % ("division by zero",))
    x_18225 = sdiv32(x_18220, y_18221)
    x_18226 = (x_18225 - N_18148)
    binop_p_18227 = (x_18226 - np.int32(1))
    res_18228 = sitofp_i32_f32(binop_p_18227)
    nest_sizze_19342 = (binop_x_23526 * binop_y_23527)
    segmap_group_sizze_19343 = self.sizes["main.segmap_group_size_19310"]
    segmap_group_sizze_19344 = sext_i32_i64(segmap_group_sizze_19343)
    segmap_usable_groups_64_19345 = sdiv_up64(nest_sizze_19342,
                                              segmap_group_sizze_19344)
    segmap_usable_groups_19346 = sext_i64_i32(segmap_usable_groups_64_19345)
    bytes_23537 = (np.int64(4) * nest_sizze_19342)
    mem_23541 = opencl_alloc(self, bytes_23537, "mem_23541")
    self.futhark_builtinzhgpu_map_transpose_f32(mem_23541, np.int32(0),
                                                binop_p_mem_23536, np.int32(0),
                                                np.int32(1), N_18148,
                                                k2p2zq_18165)
    mem_23547 = opencl_alloc(self, bytes_23537, "mem_23547")
    if ((1 * (np.long(segmap_usable_groups_19346) * np.long(segmap_group_sizze_19343))) != 0):
      self.mainzisegmap_19305_var.set_args(self.global_failure,
                                           np.int32(N_18148),
                                           np.int32(k2p2zq_18165),
                                           np.float32(res_18228), mem_23541,
                                           mem_23547)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19305_var,
                                 ((np.long(segmap_usable_groups_19346) * np.long(segmap_group_sizze_19343)),),
                                 (np.long(segmap_group_sizze_19343),))
      if synchronous:
        sync(self)
    empty_slice_18236 = (k2p2zq_18165 == np.int32(0))
    m_18237 = (k2p2zq_18165 - np.int32(1))
    zzero_leq_i_p_m_t_s_18238 = sle32(np.int32(0), m_18237)
    i_p_m_t_s_leq_w_18239 = slt32(m_18237, k2p2zq_18165)
    i_lte_j_18240 = sle32(np.int32(0), k2p2zq_18165)
    y_18241 = (zzero_leq_i_p_m_t_s_18238 and i_p_m_t_s_leq_w_18239)
    y_18242 = (i_lte_j_18240 and y_18241)
    ok_or_empty_18243 = (empty_slice_18236 or y_18242)
    empty_slice_18244 = (n_18153 == np.int32(0))
    m_18245 = (n_18153 - np.int32(1))
    zzero_leq_i_p_m_t_s_18246 = sle32(np.int32(0), m_18245)
    i_p_m_t_s_leq_w_18247 = slt32(m_18245, N_18148)
    i_lte_j_18248 = sle32(np.int32(0), n_18153)
    y_18249 = (zzero_leq_i_p_m_t_s_18246 and i_p_m_t_s_leq_w_18247)
    y_18250 = (i_lte_j_18248 and y_18249)
    ok_or_empty_18251 = (empty_slice_18244 or y_18250)
    index_ok_18252 = (ok_or_empty_18243 and ok_or_empty_18251)
    index_certs_18253 = True
    assert index_ok_18252, ("Error: %s%d%s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  bfastfinaldetailed.fut:40:15-21\n   #1  bfastfinaldetailed.fut:19:1-146:86\n" % ("Index [",
                                                                                                                                                              np.int32(0),
                                                                                                                                                              ":, :",
                                                                                                                                                              n_18153,
                                                                                                                                                              "] out of bounds for array of shape [",
                                                                                                                                                              k2p2zq_18165,
                                                                                                                                                              "][",
                                                                                                                                                              N_18148,
                                                                                                                                                              "]."))
    index_certs_18255 = True
    assert index_ok_18252, ("Error: %s%d%s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  bfastfinaldetailed.fut:41:15-22\n   #1  bfastfinaldetailed.fut:19:1-146:86\n" % ("Index [:",
                                                                                                                                                              n_18153,
                                                                                                                                                              ", ",
                                                                                                                                                              np.int32(0),
                                                                                                                                                              ":] out of bounds for array of shape [",
                                                                                                                                                              N_18148,
                                                                                                                                                              "][",
                                                                                                                                                              k2p2zq_18165,
                                                                                                                                                              "]."))
    empty_slice_18257 = (m_18149 == np.int32(0))
    m_18258 = (m_18149 - np.int32(1))
    zzero_leq_i_p_m_t_s_18259 = sle32(np.int32(0), m_18258)
    i_p_m_t_s_leq_w_18260 = slt32(m_18258, m_18149)
    i_lte_j_18261 = sle32(np.int32(0), m_18149)
    y_18262 = (zzero_leq_i_p_m_t_s_18259 and i_p_m_t_s_leq_w_18260)
    y_18263 = (i_lte_j_18261 and y_18262)
    ok_or_empty_18264 = (empty_slice_18257 or y_18263)
    index_ok_18265 = (ok_or_empty_18251 and ok_or_empty_18264)
    index_certs_18266 = True
    assert index_ok_18265, ("Error: %s%d%s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  bfastfinaldetailed.fut:42:15-26\n   #1  bfastfinaldetailed.fut:19:1-146:86\n" % ("Index [",
                                                                                                                                                              np.int32(0),
                                                                                                                                                              ":, :",
                                                                                                                                                              n_18153,
                                                                                                                                                              "] out of bounds for array of shape [",
                                                                                                                                                              m_18149,
                                                                                                                                                              "][",
                                                                                                                                                              N_18148,
                                                                                                                                                              "]."))
    suff_outer_par_19352 = (self.sizes["main.suff_outer_par_6"] <= m_18149)
    m_19378 = sext_i32_i64(m_18149)
    segmap_group_sizze_19380 = self.sizes["main.segmap_group_size_19357"]
    max_num_groups_24477 = self.sizes["main.segmap_num_groups_19359"]
    num_groups_19381 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(m_19378,
                                                            sext_i32_i64(segmap_group_sizze_19380)),
                                                  sext_i32_i64(max_num_groups_24477))))
    nest_sizze_19582 = (m_19378 * binop_x_23526)
    segmap_group_sizze_19583 = self.sizes["main.segmap_group_size_19407"]
    max_num_groups_24478 = self.sizes["main.segmap_num_groups_19409"]
    num_groups_19584 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_19582,
                                                            sext_i32_i64(segmap_group_sizze_19583)),
                                                  sext_i32_i64(max_num_groups_24478))))
    comparatee_19587 = (m_18149 * k2p2zq_18165)
    suff_outer_par_19588 = (self.sizes["main.suff_outer_par_7"] <= comparatee_19587)
    y_19610 = (binop_x_23526 * binop_x_23526)
    nest_sizze_19611 = (m_19378 * y_19610)
    segmap_group_sizze_19612 = self.sizes["main.segmap_group_size_19441"]
    segmap_group_sizze_19613 = sext_i32_i64(segmap_group_sizze_19612)
    y_19617 = (k2p2zq_18165 * k2p2zq_18165)
    comparatee_19618 = (m_18149 * y_19617)
    suff_outer_par_19619 = (self.sizes["main.suff_outer_par_8"] <= comparatee_19618)
    n_19636 = sext_i32_i64(n_18153)
    nest_sizze_19643 = (nest_sizze_19611 * n_19636)
    segred_group_sizze_19644 = self.sizes["main.segred_group_size_19476"]
    max_num_groups_24479 = self.sizes["main.segred_num_groups_19478"]
    num_groups_19645 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_19643,
                                                            sext_i32_i64(segred_group_sizze_19644)),
                                                  sext_i32_i64(max_num_groups_24479))))
    binop_x_23549 = sext_i32_i64(N_18150)
    binop_x_23551 = (m_19378 * binop_x_23549)
    bytes_23548 = (np.int64(4) * binop_x_23551)
    binop_x_23604 = (m_19378 * y_19610)
    bytes_23599 = (np.int64(4) * binop_x_23604)
    bytes_23554 = (np.int64(4) * y_19610)
    binop_x_23663 = (m_19378 * binop_x_23526)
    binop_x_23665 = (binop_x_23526 * binop_x_23663)
    bytes_23660 = (np.int64(4) * binop_x_23665)
    binop_x_23627 = (nest_sizze_19582 * binop_x_23526)
    bytes_23622 = (np.int64(4) * binop_x_23627)
    bytes_23607 = (np.int64(4) * binop_x_23526)
    num_threads_24314 = (segmap_group_sizze_19380 * num_groups_19381)
    num_threads64_24316 = sext_i32_i64(num_threads_24314)
    total_sizze_24317 = (bytes_23554 * num_threads64_24316)
    num_threads_24318 = (segmap_group_sizze_19583 * num_groups_19584)
    num_threads64_24320 = sext_i32_i64(num_threads_24318)
    total_sizze_24321 = (bytes_23607 * num_threads64_24320)
    local_memory_capacity_24571 = self.max_local_memory
    if (sle64(np.int64(0),
              sext_i32_i64(local_memory_capacity_24571)) and suff_outer_par_19352):
      mem_23552 = opencl_alloc(self, bytes_23548, "mem_23552")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23552, np.int32(0),
                                                  images_mem_23523, np.int32(0),
                                                  np.int32(1), N_18150, m_18149)
      mem_23605 = opencl_alloc(self, bytes_23599, "mem_23605")
      mem_23558 = opencl_alloc(self, total_sizze_24317, "mem_23558")
      if ((1 * (np.long(num_groups_19381) * np.long(segmap_group_sizze_19380))) != 0):
        self.mainzisegmap_19354_var.set_args(self.global_failure,
                                             np.int32(N_18148),
                                             np.int32(m_18149),
                                             np.int32(n_18153),
                                             np.int32(k2p2zq_18165),
                                             np.int32(num_groups_19381),
                                             binop_p_mem_23536, mem_23547,
                                             mem_23552, mem_23558, mem_23605)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19354_var,
                                   ((np.long(num_groups_19381) * np.long(segmap_group_sizze_19380)),),
                                   (np.long(segmap_group_sizze_19380),))
        if synchronous:
          sync(self)
      mem_23552 = None
      mem_23558 = None
      mem_23666 = opencl_alloc(self, bytes_23660, "mem_23666")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23666, np.int32(0),
                                                  mem_23605, np.int32(0),
                                                  np.int32(1), m_18149,
                                                  (k2p2zq_18165 * k2p2zq_18165))
      mem_23605 = None
      res_mem_23668 = mem_23666
    else:
      local_memory_capacity_24570 = self.max_local_memory
      if (sle64(np.int64(0),
                sext_i32_i64(local_memory_capacity_24570)) and suff_outer_par_19588):
        mem_23628 = opencl_alloc(self, bytes_23622, "mem_23628")
        mem_23609 = opencl_alloc(self, total_sizze_24321, "mem_23609")
        if ((1 * (np.long(num_groups_19584) * np.long(segmap_group_sizze_19583))) != 0):
          self.mainzisegmap_19402_var.set_args(self.global_failure,
                                               np.int32(m_18149),
                                               np.int32(N_18150),
                                               np.int32(n_18153),
                                               np.int32(k2p2zq_18165),
                                               np.int32(num_groups_19584),
                                               images_mem_23523, mem_23541,
                                               mem_23547, mem_23609, mem_23628)
          cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19402_var,
                                     ((np.long(num_groups_19584) * np.long(segmap_group_sizze_19583)),),
                                     (np.long(segmap_group_sizze_19583),))
          if synchronous:
            sync(self)
        mem_23609 = None
        mem_23657 = opencl_alloc(self, bytes_23660, "mem_23657")
        self.futhark_builtinzhgpu_map_transpose_f32(mem_23657, np.int32(0),
                                                    mem_23628, np.int32(0),
                                                    np.int32(1),
                                                    (m_18149 * k2p2zq_18165),
                                                    k2p2zq_18165)
        mem_23628 = None
        res_mem_23659 = mem_23657
      else:
        segmap_usable_groups_64_19614 = sdiv_up64(nest_sizze_19611,
                                                  segmap_group_sizze_19613)
        segmap_usable_groups_19615 = sext_i64_i32(segmap_usable_groups_64_19614)
        local_memory_capacity_24569 = self.max_local_memory
        if (sle64(np.int64(0),
                  sext_i32_i64(local_memory_capacity_24569)) and suff_outer_par_19619):
          mem_23636 = opencl_alloc(self, bytes_23660, "mem_23636")
          if ((1 * (np.long(segmap_usable_groups_19615) * np.long(segmap_group_sizze_19612))) != 0):
            self.mainzisegmap_19434_var.set_args(self.global_failure,
                                                 np.int32(m_18149),
                                                 np.int32(N_18150),
                                                 np.int32(n_18153),
                                                 np.int32(k2p2zq_18165),
                                                 images_mem_23523, mem_23541,
                                                 mem_23547, mem_23636)
            cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19434_var,
                                       ((np.long(segmap_usable_groups_19615) * np.long(segmap_group_sizze_19612)),),
                                       (np.long(segmap_group_sizze_19612),))
            if synchronous:
              sync(self)
          res_mem_23650 = mem_23636
        else:
          mem_23641 = opencl_alloc(self, bytes_23525, "mem_23641")
          self.futhark_builtinzhgpu_map_transpose_f32(mem_23641, np.int32(0),
                                                      mem_23547, np.int32(0),
                                                      np.int32(1), k2p2zq_18165,
                                                      N_18148)
          mem_23649 = opencl_alloc(self, bytes_23660, "mem_23649")
          if slt32((n_18153 * np.int32(2)), segred_group_sizze_19644):
            segment_sizze_nonzzero_24510 = smax32(np.int32(1), n_18153)
            num_threads_24511 = (num_groups_19645 * segred_group_sizze_19644)
            if ((1 * (np.long(num_groups_19645) * np.long(segred_group_sizze_19644))) != 0):
              self.mainzisegred_small_19482_var.set_args(self.global_failure,
                                                         cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_19644)))),
                                                         np.int32(N_18148),
                                                         np.int32(m_18149),
                                                         np.int32(N_18150),
                                                         np.int32(n_18153),
                                                         np.int32(k2p2zq_18165),
                                                         np.int32(num_groups_19645),
                                                         images_mem_23523,
                                                         binop_p_mem_23536,
                                                         mem_23641, mem_23649,
                                                         np.int32(segment_sizze_nonzzero_24510))
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainzisegred_small_19482_var,
                                         ((np.long(num_groups_19645) * np.long(segred_group_sizze_19644)),),
                                         (np.long(segred_group_sizze_19644),))
              if synchronous:
                sync(self)
          else:
            groups_per_segment_24530 = sdiv_up32(num_groups_19645,
                                                 smax32(np.int32(1),
                                                        ((m_18149 * k2p2zq_18165) * k2p2zq_18165)))
            elements_per_thread_24531 = sdiv_up32(n_18153,
                                                  (segred_group_sizze_19644 * groups_per_segment_24530))
            virt_num_groups_24532 = (groups_per_segment_24530 * ((m_18149 * k2p2zq_18165) * k2p2zq_18165))
            num_threads_24533 = (num_groups_19645 * segred_group_sizze_19644)
            threads_per_segment_24534 = (groups_per_segment_24530 * segred_group_sizze_19644)
            group_res_arr_mem_24535 = opencl_alloc(self,
                                                   (np.int32(4) * (sext_i32_i64(segred_group_sizze_19644) * sext_i32_i64(virt_num_groups_24532))),
                                                   "group_res_arr_mem_24535")
            mainzicounter_mem_24537 = self.mainzicounter_mem_24537
            if ((1 * (np.long(num_groups_19645) * np.long(segred_group_sizze_19644))) != 0):
              self.mainzisegred_large_19482_var.set_args(self.global_failure,
                                                         cl.LocalMemory(np.long(np.int32(1))),
                                                         cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_19644)))),
                                                         np.int32(N_18148),
                                                         np.int32(N_18150),
                                                         np.int32(n_18153),
                                                         np.int32(k2p2zq_18165),
                                                         np.int32(num_groups_19645),
                                                         images_mem_23523,
                                                         binop_p_mem_23536,
                                                         mem_23641, mem_23649,
                                                         np.int32(groups_per_segment_24530),
                                                         np.int32(elements_per_thread_24531),
                                                         np.int32(virt_num_groups_24532),
                                                         np.int32(threads_per_segment_24534),
                                                         group_res_arr_mem_24535,
                                                         mainzicounter_mem_24537)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.mainzisegred_large_19482_var,
                                         ((np.long(num_groups_19645) * np.long(segred_group_sizze_19644)),),
                                         (np.long(segred_group_sizze_19644),))
              if synchronous:
                sync(self)
          mem_23641 = None
          res_mem_23650 = mem_23649
        res_mem_23659 = res_mem_23650
      res_mem_23668 = res_mem_23659
    m_18287 = (np.int32(2) * k2p2zq_18165)
    nm_18288 = (k2p2zq_18165 * m_18287)
    bounds_invalid_upwards_18289 = slt32(nm_18288, np.int32(0))
    valid_18290 = not(bounds_invalid_upwards_18289)
    range_valid_c_18291 = True
    assert valid_18290, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  helpers.fut:71:21-27\n   #2  bfastfinaldetailed.fut:52:35-50\n   #3  bfastfinaldetailed.fut:19:1-146:86\n" % ("Range ",
                                                                                                                                                                                                                       np.int32(0),
                                                                                                                                                                                                                       "..",
                                                                                                                                                                                                                       np.int32(1),
                                                                                                                                                                                                                       "..<",
                                                                                                                                                                                                                       nm_18288,
                                                                                                                                                                                                                       " is invalid."))
    zzero_18293 = (m_18287 == np.int32(0))
    nonzzero_18294 = not(zzero_18293)
    nonzzero_cert_18295 = True
    assert nonzzero_18294, ("Error: %s\n\nBacktrace:\n-> #0  helpers.fut:66:41-47\n   #1  helpers.fut:66:14-71:28\n   #2  bfastfinaldetailed.fut:52:35-50\n   #3  bfastfinaldetailed.fut:19:1-146:86\n" % ("division by zero",))
    loop_nonempty_18296 = slt32(np.int32(0), k2p2zq_18165)
    loop_not_taken_18297 = not(loop_nonempty_18296)
    protect_assert_disj_18298 = (nonzzero_18294 or loop_not_taken_18297)
    nonzzero_cert_18299 = True
    assert protect_assert_disj_18298, ("Error: %s\n\nBacktrace:\n-> #0  helpers.fut:53:43-49\n   #1  helpers.fut:53:16-59:30\n   #2  helpers.fut:72:15-33\n   #3  bfastfinaldetailed.fut:52:35-50\n   #4  bfastfinaldetailed.fut:19:1-146:86\n" % ("division by zero",))
    y_19719 = smin32(k2p2zq_18165, nm_18288)
    intra_avail_par_19720 = smin32(y_19617, y_19719)
    y_19721 = smax32(k2p2zq_18165, nm_18288)
    computed_group_sizze_19668 = smax32(y_19617, y_19721)
    max_group_sizze_19840 = self.max_group_size
    fits_19841 = sle32(computed_group_sizze_19668, max_group_sizze_19840)
    suff_intra_par_19839 = (self.sizes["main.suff_intra_par_10"] <= intra_avail_par_19720)
    intra_suff_and_fits_19842 = (suff_intra_par_19839 and fits_19841)
    nm_20462 = sext_i32_i64(nm_18288)
    nest_sizze_20464 = (m_19378 * nm_20462)
    segmap_group_sizze_20465 = self.sizes["main.segmap_group_size_20416"]
    segmap_group_sizze_20466 = sext_i32_i64(segmap_group_sizze_20465)
    fits_20491 = sle32(nm_18288, max_group_sizze_19840)
    suff_intra_par_20493 = (self.sizes["main.suff_intra_par_14"] <= nm_18288)
    intra_suff_and_fits_20494 = (fits_20491 and suff_intra_par_20493)
    segmap_group_sizze_20526 = self.sizes["main.segmap_group_size_20306"]
    segmap_group_sizze_20527 = sext_i32_i64(segmap_group_sizze_20526)
    segmap_group_sizze_20550 = self.sizes["main.segmap_group_size_20242"]
    segmap_group_sizze_20551 = sext_i32_i64(segmap_group_sizze_20550)
    segmap_group_sizze_20583 = self.sizes["main.segmap_group_size_20185"]
    segmap_group_sizze_20584 = sext_i32_i64(segmap_group_sizze_20583)
    segmap_group_sizze_20634 = self.sizes["main.segmap_group_size_19958"]
    segmap_group_sizze_20635 = sext_i32_i64(segmap_group_sizze_20634)
    segmap_usable_groups_64_20528 = sdiv_up_safe64(m_19378,
                                                   segmap_group_sizze_20527)
    segmap_usable_groups_20529 = sext_i64_i32(segmap_usable_groups_64_20528)
    segmap_usable_groups_64_20552 = sdiv_up_safe64(nest_sizze_20464,
                                                   segmap_group_sizze_20551)
    segmap_usable_groups_20553 = sext_i64_i32(segmap_usable_groups_64_20552)
    segmap_usable_groups_64_20585 = sdiv_up_safe64(nest_sizze_20464,
                                                   segmap_group_sizze_20584)
    segmap_usable_groups_20586 = sext_i64_i32(segmap_usable_groups_64_20585)
    bytes_23671 = (np.int64(4) * nm_20462)
    bytes_23705 = (np.int64(4) * nest_sizze_20464)
    binop_x_23723 = (m_19378 * nm_20462)
    bytes_23720 = (np.int64(4) * binop_x_23723)
    local_memory_capacity_24636 = self.max_local_memory
    if (sle64(((bytes_23671 + bytes_23671) + bytes_23554),
              sext_i32_i64(local_memory_capacity_24636)) and intra_suff_and_fits_19842):
      mem_23703 = opencl_alloc(self, bytes_23660, "mem_23703")
      if ((1 * (np.long(m_18149) * np.long(computed_group_sizze_19668))) != 0):
        self.mainzisegmap_intragroup_19722_var.set_args(self.global_failure,
                                                        self.failure_is_an_option,
                                                        self.global_failure_args,
                                                        cl.LocalMemory(np.long(bytes_23554)),
                                                        cl.LocalMemory(np.long(bytes_23671)),
                                                        cl.LocalMemory(np.long(bytes_23671)),
                                                        np.int32(k2p2zq_18165),
                                                        np.int32(m_18287),
                                                        np.int32(nm_18288),
                                                        np.int32(computed_group_sizze_19668),
                                                        res_mem_23668,
                                                        mem_23703)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegmap_intragroup_19722_var,
                                   ((np.long(m_18149) * np.long(computed_group_sizze_19668)),),
                                   (np.long(computed_group_sizze_19668),))
        if synchronous:
          sync(self)
      self.failure_is_an_option = np.int32(1)
      res_mem_23770 = mem_23703
    else:
      segmap_usable_groups_64_20467 = sdiv_up64(nest_sizze_20464,
                                                segmap_group_sizze_20466)
      segmap_usable_groups_20468 = sext_i64_i32(segmap_usable_groups_64_20467)
      mem_23709 = opencl_alloc(self, bytes_23705, "mem_23709")
      if ((1 * (np.long(segmap_usable_groups_20468) * np.long(segmap_group_sizze_20465))) != 0):
        self.mainzisegmap_20411_var.set_args(self.global_failure,
                                             np.int32(m_18149),
                                             np.int32(k2p2zq_18165),
                                             np.int32(m_18287),
                                             np.int32(nm_18288), res_mem_23668,
                                             mem_23709)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20411_var,
                                   ((np.long(segmap_usable_groups_20468) * np.long(segmap_group_sizze_20465)),),
                                   (np.long(segmap_group_sizze_20465),))
        if synchronous:
          sync(self)
      ctx_param_ext_23710 = m_18149
      ctx_param_ext_23711 = nm_18288
      ctx_param_ext_23712 = np.int32(0)
      ctx_param_ext_23713 = nm_18288
      ctx_param_ext_23714 = m_18149
      ctx_param_ext_23715 = np.int32(1)
      ctx_param_ext_23716 = nm_18288
      mem_param_23717 = mem_23709
      i_20482 = np.int32(0)
      one_25446 = np.int32(1)
      for counter_25445 in range(k2p2zq_18165):
        y_20484 = slt32(i_20482, nm_18288)
        index_certs_20485 = True
        assert y_20484, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  helpers.fut:52:16-19\n   #1  helpers.fut:72:15-33\n   #2  bfastfinaldetailed.fut:52:35-50\n   #3  bfastfinaldetailed.fut:19:1-146:86\n" % ("Index [",
                                                                                                                                                                                                             i_20482,
                                                                                                                                                                                                             "] out of bounds for array of shape [",
                                                                                                                                                                                                             nm_18288,
                                                                                                                                                                                                             "]."))
        local_memory_capacity_24596 = self.max_local_memory
        if intra_suff_and_fits_20494:
          res_ixfn_23746 = m_18149
        else:
          res_ixfn_23746 = ctx_param_ext_23714
        local_memory_capacity_24597 = self.max_local_memory
        if intra_suff_and_fits_20494:
          res_ixfn_23747 = nm_18288
        else:
          res_ixfn_23747 = ctx_param_ext_23716
        local_memory_capacity_24598 = self.max_local_memory
        if intra_suff_and_fits_20494:
          res_ixfn_23748 = m_18149
        else:
          res_ixfn_23748 = ctx_param_ext_23710
        local_memory_capacity_24599 = self.max_local_memory
        if intra_suff_and_fits_20494:
          res_ixfn_23749 = nm_18288
        else:
          res_ixfn_23749 = ctx_param_ext_23711
        local_memory_capacity_24600 = self.max_local_memory
        if intra_suff_and_fits_20494:
          res_ixfn_23750 = nm_18288
        else:
          res_ixfn_23750 = ctx_param_ext_23713
        local_memory_capacity_24601 = self.max_local_memory
        if intra_suff_and_fits_20494:
          res_ixfn_23751 = np.int32(1)
        else:
          res_ixfn_23751 = ctx_param_ext_23715
        local_memory_capacity_24602 = self.max_local_memory
        if intra_suff_and_fits_20494:
          res_ixfn_23752 = np.int32(0)
        else:
          res_ixfn_23752 = ctx_param_ext_23712
        local_memory_capacity_24630 = self.max_local_memory
        if ((sle64(np.int64(0),
                   sext_i32_i64(local_memory_capacity_24630)) and sle64(bytes_23671,
                                                                        sext_i32_i64(local_memory_capacity_24630))) and intra_suff_and_fits_20494):
          mem_23724 = opencl_alloc(self, bytes_23720, "mem_23724")
          group_sizze_24606 = self.sizes["main.group_size_24606"]
          num_groups_24607 = sdiv_up64((sext_i32_i64(m_18149) * sext_i32_i64(nm_18288)),
                                       sext_i32_i64(group_sizze_24606))
          if ((1 * (np.long(sext_i64_i32(num_groups_24607)) * np.long(group_sizze_24606))) != 0):
            self.mainzicopy_24603_var.set_args(np.int32(m_18149),
                                               np.int32(nm_18288),
                                               np.int32(ctx_param_ext_23712),
                                               np.int32(ctx_param_ext_23713),
                                               np.int32(ctx_param_ext_23715),
                                               mem_param_23717, mem_23724)
            cl.enqueue_nd_range_kernel(self.queue, self.mainzicopy_24603_var,
                                       ((np.long(sext_i64_i32(num_groups_24607)) * np.long(group_sizze_24606)),),
                                       (np.long(group_sizze_24606),))
            if synchronous:
              sync(self)
          mem_23735 = opencl_alloc(self, bytes_23705, "mem_23735")
          if ((1 * (np.long(m_18149) * np.long(nm_18288))) != 0):
            self.mainzisegmap_intragroup_20074_var.set_args(self.global_failure,
                                                            cl.LocalMemory(np.long(bytes_23671)),
                                                            np.int32(m_18149),
                                                            np.int32(k2p2zq_18165),
                                                            np.int32(m_18287),
                                                            np.int32(nm_18288),
                                                            np.int32(i_20482),
                                                            np.int32(ctx_param_ext_23712),
                                                            np.int32(ctx_param_ext_23713),
                                                            np.int32(ctx_param_ext_23715),
                                                            mem_param_23717,
                                                            mem_23724,
                                                            mem_23735)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegmap_intragroup_20074_var,
                                       ((np.long(m_18149) * np.long(nm_18288)),),
                                       (np.long(nm_18288),))
            if synchronous:
              sync(self)
          mem_23724 = None
          res_mem_23753 = mem_23735
        else:
          mem_23738 = opencl_alloc(self, m_19378, "mem_23738")
          if ((1 * (np.long(segmap_usable_groups_20529) * np.long(segmap_group_sizze_20526))) != 0):
            self.mainzisegmap_20303_var.set_args(self.global_failure,
                                                 np.int32(m_18149),
                                                 np.int32(i_20482),
                                                 np.int32(ctx_param_ext_23712),
                                                 np.int32(ctx_param_ext_23713),
                                                 np.int32(ctx_param_ext_23715),
                                                 mem_param_23717, mem_23738)
            cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20303_var,
                                       ((np.long(segmap_usable_groups_20529) * np.long(segmap_group_sizze_20526)),),
                                       (np.long(segmap_group_sizze_20526),))
            if synchronous:
              sync(self)
          mem_23744 = opencl_alloc(self, bytes_23705, "mem_23744")
          if ((1 * (np.long(segmap_usable_groups_20553) * np.long(segmap_group_sizze_20550))) != 0):
            self.mainzisegmap_20237_var.set_args(self.global_failure,
                                                 np.int32(m_18149),
                                                 np.int32(k2p2zq_18165),
                                                 np.int32(m_18287),
                                                 np.int32(nm_18288),
                                                 np.int32(i_20482),
                                                 np.int32(ctx_param_ext_23712),
                                                 np.int32(ctx_param_ext_23713),
                                                 np.int32(ctx_param_ext_23715),
                                                 mem_param_23717, mem_23738,
                                                 mem_23744)
            cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20237_var,
                                       ((np.long(segmap_usable_groups_20553) * np.long(segmap_group_sizze_20550)),),
                                       (np.long(segmap_group_sizze_20550),))
            if synchronous:
              sync(self)
          mem_23738 = None
          if ((1 * (np.long(segmap_usable_groups_20586) * np.long(segmap_group_sizze_20583))) != 0):
            self.mainzisegmap_20180_var.set_args(self.global_failure,
                                                 np.int32(m_18149),
                                                 np.int32(nm_18288),
                                                 np.int32(ctx_param_ext_23712),
                                                 np.int32(ctx_param_ext_23713),
                                                 np.int32(ctx_param_ext_23715),
                                                 mem_param_23717, mem_23744)
            cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20180_var,
                                       ((np.long(segmap_usable_groups_20586) * np.long(segmap_group_sizze_20583)),),
                                       (np.long(segmap_group_sizze_20583),))
            if synchronous:
              sync(self)
          mem_23744 = None
          res_mem_23753 = mem_param_23717
        ctx_param_ext_tmp_24587 = res_ixfn_23748
        ctx_param_ext_tmp_24588 = res_ixfn_23749
        ctx_param_ext_tmp_24589 = res_ixfn_23752
        ctx_param_ext_tmp_24590 = res_ixfn_23750
        ctx_param_ext_tmp_24591 = res_ixfn_23746
        ctx_param_ext_tmp_24592 = res_ixfn_23751
        ctx_param_ext_tmp_24593 = res_ixfn_23747
        mem_param_tmp_24594 = res_mem_23753
        ctx_param_ext_23710 = ctx_param_ext_tmp_24587
        ctx_param_ext_23711 = ctx_param_ext_tmp_24588
        ctx_param_ext_23712 = ctx_param_ext_tmp_24589
        ctx_param_ext_23713 = ctx_param_ext_tmp_24590
        ctx_param_ext_23714 = ctx_param_ext_tmp_24591
        ctx_param_ext_23715 = ctx_param_ext_tmp_24592
        ctx_param_ext_23716 = ctx_param_ext_tmp_24593
        mem_param_23717 = mem_param_tmp_24594
        i_20482 += one_25446
      res_r_ixfn_23754 = ctx_param_ext_23710
      res_r_ixfn_23755 = ctx_param_ext_23711
      res_r_ixfn_23756 = ctx_param_ext_23712
      res_r_ixfn_23757 = ctx_param_ext_23713
      res_r_ixfn_23758 = ctx_param_ext_23714
      res_r_ixfn_23759 = ctx_param_ext_23715
      res_r_ixfn_23760 = ctx_param_ext_23716
      res_r_mem_23761 = mem_param_23717
      mem_23709 = None
      segmap_usable_groups_64_20636 = sdiv_up64(nest_sizze_19611,
                                                segmap_group_sizze_20635)
      segmap_usable_groups_20637 = sext_i64_i32(segmap_usable_groups_64_20636)
      mem_23769 = opencl_alloc(self, bytes_23660, "mem_23769")
      if ((1 * (np.long(segmap_usable_groups_20637) * np.long(segmap_group_sizze_20634))) != 0):
        self.mainzisegmap_19951_var.set_args(self.global_failure,
                                             np.int32(m_18149),
                                             np.int32(k2p2zq_18165),
                                             np.int32(m_18287),
                                             np.int32(res_r_ixfn_23756),
                                             np.int32(res_r_ixfn_23757),
                                             np.int32(res_r_ixfn_23759),
                                             res_r_mem_23761, mem_23769)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_19951_var,
                                   ((np.long(segmap_usable_groups_20637) * np.long(segmap_group_sizze_20634)),),
                                   (np.long(segmap_group_sizze_20634),))
        if synchronous:
          sync(self)
      res_r_mem_23761 = None
      res_mem_23770 = mem_23769
    res_mem_23668 = None
    suff_outer_par_20649 = (self.sizes["main.suff_outer_par_17"] <= m_18149)
    segmap_group_sizze_20673 = self.sizes["main.segmap_group_size_20654"]
    max_num_groups_24637 = self.sizes["main.segmap_num_groups_20656"]
    num_groups_20674 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(m_19378,
                                                            sext_i32_i64(segmap_group_sizze_20673)),
                                                  sext_i32_i64(max_num_groups_24637))))
    suff_outer_par_20772 = (self.sizes["main.suff_outer_par_18"] <= comparatee_19587)
    nest_sizze_20791 = (nest_sizze_19582 * n_19636)
    segred_group_sizze_20792 = self.sizes["main.segred_group_size_20724"]
    max_num_groups_24638 = self.sizes["main.segred_num_groups_20726"]
    num_groups_20793 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_20791,
                                                            sext_i32_i64(segred_group_sizze_20792)),
                                                  sext_i32_i64(max_num_groups_24638))))
    tile_sizze_22834 = self.sizes["main.tile_size_22833"]
    group_sizze_22835 = (tile_sizze_22834 * tile_sizze_22834)
    bytes_23792 = (np.int64(4) * nest_sizze_19582)
    bytes_23880 = (np.int64(4) * binop_x_23663)
    binop_x_23800 = sext_i32_i64(tile_sizze_22834)
    binop_x_23802 = (binop_x_23800 * binop_x_23800)
    bytes_23799 = (np.int64(4) * binop_x_23802)
    binop_x_24282 = (np.int64(4) * binop_x_23800)
    sizze_24284 = (binop_x_23800 * binop_x_24282)
    num_threads_24349 = (segmap_group_sizze_20673 * num_groups_20674)
    num_threads64_24351 = sext_i32_i64(num_threads_24349)
    total_sizze_24352 = (bytes_23607 * num_threads64_24351)
    local_memory_capacity_24723 = self.max_local_memory
    if (sle64(np.int64(0),
              sext_i32_i64(local_memory_capacity_24723)) and suff_outer_par_20649):
      mem_23775 = opencl_alloc(self, bytes_23548, "mem_23775")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23775, np.int32(0),
                                                  images_mem_23523, np.int32(0),
                                                  np.int32(1), N_18150, m_18149)
      mem_23796 = opencl_alloc(self, bytes_23792, "mem_23796")
      mem_23779 = opencl_alloc(self, total_sizze_24352, "mem_23779")
      if ((1 * (np.long(num_groups_20674) * np.long(segmap_group_sizze_20673))) != 0):
        self.mainzisegmap_20651_var.set_args(self.global_failure,
                                             np.int32(N_18148),
                                             np.int32(m_18149),
                                             np.int32(n_18153),
                                             np.int32(k2p2zq_18165),
                                             np.int32(num_groups_20674),
                                             binop_p_mem_23536, mem_23775,
                                             mem_23779, mem_23796)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20651_var,
                                   ((np.long(num_groups_20674) * np.long(segmap_group_sizze_20673)),),
                                   (np.long(segmap_group_sizze_20673),))
        if synchronous:
          sync(self)
      mem_23775 = None
      mem_23779 = None
      mem_23884 = opencl_alloc(self, bytes_23880, "mem_23884")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23884, np.int32(0),
                                                  mem_23796, np.int32(0),
                                                  np.int32(1), m_18149,
                                                  k2p2zq_18165)
      mem_23796 = None
      res_mem_23886 = mem_23884
    else:
      local_memory_capacity_24722 = self.max_local_memory
      if (sle64((((bytes_23799 + bytes_23799) + bytes_23799) + bytes_23799),
                sext_i32_i64(local_memory_capacity_24722)) and suff_outer_par_20772):
        num_groups_x_22836 = sdiv_up32(m_18149, tile_sizze_22834)
        num_groups_y_22837 = sdiv_up32(k2p2zq_18165, tile_sizze_22834)
        num_groups_top_22838 = (num_groups_x_22836 * num_groups_y_22837)
        num_whole_tiles_22855 = squot32(n_18153, tile_sizze_22834)
        residual_input_23005 = srem32(n_18153, tile_sizze_22834)
        cond_23006 = (residual_input_23005 == np.int32(0))
        mem_23872 = opencl_alloc(self, bytes_23880, "mem_23872")
        if ((1 * (np.long(num_groups_top_22838) * np.long(group_sizze_22835))) != 0):
          self.mainzisegmap_intragroup_22839_var.set_args(self.global_failure,
                                                          cl.LocalMemory(np.long(bytes_23799)),
                                                          cl.LocalMemory(np.long(bytes_23799)),
                                                          cl.LocalMemory(np.long(bytes_23799)),
                                                          cl.LocalMemory(np.long(bytes_23799)),
                                                          np.int32(m_18149),
                                                          np.int32(N_18150),
                                                          np.int32(n_18153),
                                                          np.int32(k2p2zq_18165),
                                                          np.int32(num_groups_y_22837),
                                                          np.int32(num_whole_tiles_22855),
                                                          np.int32(residual_input_23005),
                                                          np.byte(cond_23006),
                                                          images_mem_23523,
                                                          mem_23541, mem_23872)
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegmap_intragroup_22839_var,
                                     ((np.long(num_groups_top_22838) * np.long(group_sizze_22835)),),
                                     (np.long(group_sizze_22835),))
          if synchronous:
            sync(self)
        res_mem_23879 = mem_23872
      else:
        mem_23878 = opencl_alloc(self, bytes_23880, "mem_23878")
        if slt32((n_18153 * np.int32(2)), segred_group_sizze_20792):
          segment_sizze_nonzzero_24663 = smax32(np.int32(1), n_18153)
          num_threads_24664 = (num_groups_20793 * segred_group_sizze_20792)
          if ((1 * (np.long(num_groups_20793) * np.long(segred_group_sizze_20792))) != 0):
            self.mainzisegred_small_20730_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_20792)))),
                                                       np.int32(N_18148),
                                                       np.int32(m_18149),
                                                       np.int32(N_18150),
                                                       np.int32(n_18153),
                                                       np.int32(k2p2zq_18165),
                                                       np.int32(num_groups_20793),
                                                       images_mem_23523,
                                                       binop_p_mem_23536,
                                                       mem_23878,
                                                       np.int32(segment_sizze_nonzzero_24663))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_small_20730_var,
                                       ((np.long(num_groups_20793) * np.long(segred_group_sizze_20792)),),
                                       (np.long(segred_group_sizze_20792),))
            if synchronous:
              sync(self)
        else:
          groups_per_segment_24683 = sdiv_up32(num_groups_20793,
                                               smax32(np.int32(1),
                                                      (m_18149 * k2p2zq_18165)))
          elements_per_thread_24684 = sdiv_up32(n_18153,
                                                (segred_group_sizze_20792 * groups_per_segment_24683))
          virt_num_groups_24685 = (groups_per_segment_24683 * (m_18149 * k2p2zq_18165))
          num_threads_24686 = (num_groups_20793 * segred_group_sizze_20792)
          threads_per_segment_24687 = (groups_per_segment_24683 * segred_group_sizze_20792)
          group_res_arr_mem_24688 = opencl_alloc(self,
                                                 (np.int32(4) * (sext_i32_i64(segred_group_sizze_20792) * sext_i32_i64(virt_num_groups_24685))),
                                                 "group_res_arr_mem_24688")
          mainzicounter_mem_24690 = self.mainzicounter_mem_24690
          if ((1 * (np.long(num_groups_20793) * np.long(segred_group_sizze_20792))) != 0):
            self.mainzisegred_large_20730_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long(np.int32(1))),
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_20792)))),
                                                       np.int32(N_18148),
                                                       np.int32(N_18150),
                                                       np.int32(n_18153),
                                                       np.int32(k2p2zq_18165),
                                                       np.int32(num_groups_20793),
                                                       images_mem_23523,
                                                       binop_p_mem_23536,
                                                       mem_23878,
                                                       np.int32(groups_per_segment_24683),
                                                       np.int32(elements_per_thread_24684),
                                                       np.int32(virt_num_groups_24685),
                                                       np.int32(threads_per_segment_24687),
                                                       group_res_arr_mem_24688,
                                                       mainzicounter_mem_24690)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_large_20730_var,
                                       ((np.long(num_groups_20793) * np.long(segred_group_sizze_20792)),),
                                       (np.long(segred_group_sizze_20792),))
            if synchronous:
              sync(self)
        res_mem_23879 = mem_23878
      res_mem_23886 = res_mem_23879
    binop_p_mem_23536 = None
    mem_23541 = None
    suff_outer_par_20809 = (self.sizes["main.suff_outer_par_19"] <= m_18149)
    segmap_group_sizze_20832 = self.sizes["main.segmap_group_size_20814"]
    max_num_groups_24724 = self.sizes["main.segmap_num_groups_20816"]
    num_groups_20833 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(m_19378,
                                                            sext_i32_i64(segmap_group_sizze_20832)),
                                                  sext_i32_i64(max_num_groups_24724))))
    segmap_group_sizze_20920 = self.sizes["main.segmap_group_size_20854"]
    segmap_group_sizze_20921 = sext_i32_i64(segmap_group_sizze_20920)
    suff_outer_par_20926 = (self.sizes["main.suff_outer_par_20"] <= comparatee_19587)
    nest_sizze_20943 = (nest_sizze_19582 * binop_x_23526)
    segred_group_sizze_20944 = self.sizes["main.segred_group_size_20880"]
    max_num_groups_24725 = self.sizes["main.segred_num_groups_20882"]
    num_groups_20945 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_20943,
                                                            sext_i32_i64(segred_group_sizze_20944)),
                                                  sext_i32_i64(max_num_groups_24725))))
    num_threads_24359 = (segmap_group_sizze_20832 * num_groups_20833)
    num_threads64_24361 = sext_i32_i64(num_threads_24359)
    total_sizze_24362 = (bytes_23607 * num_threads64_24361)
    local_memory_capacity_24803 = self.max_local_memory
    if (sle64(np.int64(0),
              sext_i32_i64(local_memory_capacity_24803)) and suff_outer_par_20809):
      mem_23893 = opencl_alloc(self, bytes_23599, "mem_23893")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23893, np.int32(0),
                                                  res_mem_23770, np.int32(0),
                                                  np.int32(1),
                                                  (k2p2zq_18165 * k2p2zq_18165),
                                                  m_18149)
      mem_23898 = opencl_alloc(self, bytes_23792, "mem_23898")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23898, np.int32(0),
                                                  res_mem_23886, np.int32(0),
                                                  np.int32(1), k2p2zq_18165,
                                                  m_18149)
      mem_23919 = opencl_alloc(self, bytes_23792, "mem_23919")
      mem_23902 = opencl_alloc(self, total_sizze_24362, "mem_23902")
      if ((1 * (np.long(num_groups_20833) * np.long(segmap_group_sizze_20832))) != 0):
        self.mainzisegmap_20811_var.set_args(self.global_failure,
                                             np.int32(m_18149),
                                             np.int32(k2p2zq_18165),
                                             np.int32(num_groups_20833),
                                             mem_23893, mem_23898, mem_23902,
                                             mem_23919)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20811_var,
                                   ((np.long(num_groups_20833) * np.long(segmap_group_sizze_20832)),),
                                   (np.long(segmap_group_sizze_20832),))
        if synchronous:
          sync(self)
      mem_23893 = None
      mem_23898 = None
      mem_23902 = None
      mem_23944 = opencl_alloc(self, bytes_23880, "mem_23944")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23944, np.int32(0),
                                                  mem_23919, np.int32(0),
                                                  np.int32(1), m_18149,
                                                  k2p2zq_18165)
      mem_23919 = None
      res_mem_23946 = mem_23944
    else:
      segmap_usable_groups_64_20922 = sdiv_up64(nest_sizze_19582,
                                                segmap_group_sizze_20921)
      segmap_usable_groups_20923 = sext_i64_i32(segmap_usable_groups_64_20922)
      local_memory_capacity_24802 = self.max_local_memory
      if (sle64(np.int64(0),
                sext_i32_i64(local_memory_capacity_24802)) and suff_outer_par_20926):
        mem_23926 = opencl_alloc(self, bytes_23622, "mem_23926")
        self.futhark_builtinzhgpu_map_transpose_f32(mem_23926, np.int32(0),
                                                    res_mem_23770, np.int32(0),
                                                    np.int32(1), k2p2zq_18165,
                                                    (m_18149 * k2p2zq_18165))
        mem_23932 = opencl_alloc(self, bytes_23880, "mem_23932")
        if ((1 * (np.long(segmap_usable_groups_20923) * np.long(segmap_group_sizze_20920))) != 0):
          self.mainzisegmap_20849_var.set_args(self.global_failure,
                                               np.int32(m_18149),
                                               np.int32(k2p2zq_18165),
                                               res_mem_23886, mem_23926,
                                               mem_23932)
          cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20849_var,
                                     ((np.long(segmap_usable_groups_20923) * np.long(segmap_group_sizze_20920)),),
                                     (np.long(segmap_group_sizze_20920),))
          if synchronous:
            sync(self)
        mem_23926 = None
        res_mem_23939 = mem_23932
      else:
        mem_23938 = opencl_alloc(self, bytes_23880, "mem_23938")
        if slt32((k2p2zq_18165 * np.int32(2)), segred_group_sizze_20944):
          segment_sizze_nonzzero_24743 = smax32(np.int32(1), k2p2zq_18165)
          num_threads_24744 = (num_groups_20945 * segred_group_sizze_20944)
          if ((1 * (np.long(num_groups_20945) * np.long(segred_group_sizze_20944))) != 0):
            self.mainzisegred_small_20886_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_20944)))),
                                                       np.int32(m_18149),
                                                       np.int32(k2p2zq_18165),
                                                       np.int32(num_groups_20945),
                                                       res_mem_23770,
                                                       res_mem_23886, mem_23938,
                                                       np.int32(segment_sizze_nonzzero_24743))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_small_20886_var,
                                       ((np.long(num_groups_20945) * np.long(segred_group_sizze_20944)),),
                                       (np.long(segred_group_sizze_20944),))
            if synchronous:
              sync(self)
        else:
          groups_per_segment_24763 = sdiv_up32(num_groups_20945,
                                               smax32(np.int32(1),
                                                      (m_18149 * k2p2zq_18165)))
          elements_per_thread_24764 = sdiv_up32(k2p2zq_18165,
                                                (segred_group_sizze_20944 * groups_per_segment_24763))
          virt_num_groups_24765 = (groups_per_segment_24763 * (m_18149 * k2p2zq_18165))
          num_threads_24766 = (num_groups_20945 * segred_group_sizze_20944)
          threads_per_segment_24767 = (groups_per_segment_24763 * segred_group_sizze_20944)
          group_res_arr_mem_24768 = opencl_alloc(self,
                                                 (np.int32(4) * (sext_i32_i64(segred_group_sizze_20944) * sext_i32_i64(virt_num_groups_24765))),
                                                 "group_res_arr_mem_24768")
          mainzicounter_mem_24770 = self.mainzicounter_mem_24770
          if ((1 * (np.long(num_groups_20945) * np.long(segred_group_sizze_20944))) != 0):
            self.mainzisegred_large_20886_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long(np.int32(1))),
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_20944)))),
                                                       np.int32(k2p2zq_18165),
                                                       np.int32(num_groups_20945),
                                                       res_mem_23770,
                                                       res_mem_23886, mem_23938,
                                                       np.int32(groups_per_segment_24763),
                                                       np.int32(elements_per_thread_24764),
                                                       np.int32(virt_num_groups_24765),
                                                       np.int32(threads_per_segment_24767),
                                                       group_res_arr_mem_24768,
                                                       mainzicounter_mem_24770)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_large_20886_var,
                                       ((np.long(num_groups_20945) * np.long(segred_group_sizze_20944)),),
                                       (np.long(segred_group_sizze_20944),))
            if synchronous:
              sync(self)
        res_mem_23939 = mem_23938
      res_mem_23946 = res_mem_23939
    res_mem_23770 = None
    res_mem_23886 = None
    suff_outer_par_20960 = (self.sizes["main.suff_outer_par_21"] <= m_18149)
    segmap_group_sizze_20982 = self.sizes["main.segmap_group_size_20965"]
    max_num_groups_24804 = self.sizes["main.segmap_num_groups_20967"]
    num_groups_20983 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(m_19378,
                                                            sext_i32_i64(segmap_group_sizze_20982)),
                                                  sext_i32_i64(max_num_groups_24804))))
    comparatee_21072 = (N_18148 * m_18149)
    suff_outer_par_21073 = (self.sizes["main.suff_outer_par_22"] <= comparatee_21072)
    y_21089 = (m_19378 * binop_y_23527)
    nest_sizze_21090 = (y_21089 * binop_x_23526)
    segred_group_sizze_21091 = self.sizes["main.segred_group_size_21029"]
    max_num_groups_24805 = self.sizes["main.segred_num_groups_21031"]
    num_groups_21092 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_21090,
                                                            sext_i32_i64(segred_group_sizze_21091)),
                                                  sext_i32_i64(max_num_groups_24805))))
    tile_sizze_23156 = self.sizes["main.tile_size_23155"]
    group_sizze_23157 = (tile_sizze_23156 * tile_sizze_23156)
    bytes_23968 = (np.int64(4) * y_21089)
    bytes_23953 = (np.int64(4) * binop_y_23527)
    binop_x_24064 = (m_19378 * binop_y_23527)
    bytes_24061 = (np.int64(4) * binop_x_24064)
    binop_x_23981 = sext_i32_i64(tile_sizze_23156)
    binop_x_23983 = (binop_x_23981 * binop_x_23981)
    bytes_23980 = (np.int64(4) * binop_x_23983)
    binop_x_24296 = (np.int64(4) * binop_x_23981)
    sizze_24298 = (binop_x_23981 * binop_x_24296)
    num_threads_24369 = (segmap_group_sizze_20982 * num_groups_20983)
    num_threads64_24371 = sext_i32_i64(num_threads_24369)
    total_sizze_24372 = (bytes_23953 * num_threads64_24371)
    local_memory_capacity_24890 = self.max_local_memory
    if (sle64(np.int64(0),
              sext_i32_i64(local_memory_capacity_24890)) and suff_outer_par_20960):
      mem_23951 = opencl_alloc(self, bytes_23792, "mem_23951")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_23951, np.int32(0),
                                                  res_mem_23946, np.int32(0),
                                                  np.int32(1), k2p2zq_18165,
                                                  m_18149)
      mem_23972 = opencl_alloc(self, bytes_23968, "mem_23972")
      mem_23955 = opencl_alloc(self, total_sizze_24372, "mem_23955")
      if ((1 * (np.long(num_groups_20983) * np.long(segmap_group_sizze_20982))) != 0):
        self.mainzisegmap_20962_var.set_args(self.global_failure,
                                             np.int32(N_18148),
                                             np.int32(m_18149),
                                             np.int32(k2p2zq_18165),
                                             np.int32(num_groups_20983),
                                             mem_23547, mem_23951, mem_23955,
                                             mem_23972)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_20962_var,
                                   ((np.long(num_groups_20983) * np.long(segmap_group_sizze_20982)),),
                                   (np.long(segmap_group_sizze_20982),))
        if synchronous:
          sync(self)
      mem_23951 = None
      mem_23955 = None
      mem_24065 = opencl_alloc(self, bytes_24061, "mem_24065")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_24065, np.int32(0),
                                                  mem_23972, np.int32(0),
                                                  np.int32(1), m_18149, N_18148)
      mem_23972 = None
      res_mem_24067 = mem_24065
    else:
      local_memory_capacity_24889 = self.max_local_memory
      if (sle64((((bytes_23980 + bytes_23980) + bytes_23980) + bytes_23980),
                sext_i32_i64(local_memory_capacity_24889)) and suff_outer_par_21073):
        mem_23977 = opencl_alloc(self, bytes_23525, "mem_23977")
        self.futhark_builtinzhgpu_map_transpose_f32(mem_23977, np.int32(0),
                                                    mem_23547, np.int32(0),
                                                    np.int32(1), k2p2zq_18165,
                                                    N_18148)
        num_groups_x_23158 = sdiv_up32(m_18149, tile_sizze_23156)
        num_groups_y_23159 = sdiv_up32(N_18148, tile_sizze_23156)
        num_groups_top_23160 = (num_groups_x_23158 * num_groups_y_23159)
        num_whole_tiles_23177 = squot32(k2p2zq_18165, tile_sizze_23156)
        residual_input_23321 = srem32(k2p2zq_18165, tile_sizze_23156)
        cond_23322 = (residual_input_23321 == np.int32(0))
        mem_24053 = opencl_alloc(self, bytes_24061, "mem_24053")
        if ((1 * (np.long(num_groups_top_23160) * np.long(group_sizze_23157))) != 0):
          self.mainzisegmap_intragroup_23161_var.set_args(self.global_failure,
                                                          cl.LocalMemory(np.long(bytes_23980)),
                                                          cl.LocalMemory(np.long(bytes_23980)),
                                                          cl.LocalMemory(np.long(bytes_23980)),
                                                          cl.LocalMemory(np.long(bytes_23980)),
                                                          np.int32(N_18148),
                                                          np.int32(m_18149),
                                                          np.int32(k2p2zq_18165),
                                                          np.int32(num_groups_y_23159),
                                                          np.int32(num_whole_tiles_23177),
                                                          np.int32(residual_input_23321),
                                                          np.byte(cond_23322),
                                                          res_mem_23946,
                                                          mem_23977, mem_24053)
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegmap_intragroup_23161_var,
                                     ((np.long(num_groups_top_23160) * np.long(group_sizze_23157)),),
                                     (np.long(group_sizze_23157),))
          if synchronous:
            sync(self)
        mem_23977 = None
        res_mem_24060 = mem_24053
      else:
        mem_24059 = opencl_alloc(self, bytes_24061, "mem_24059")
        if slt32((k2p2zq_18165 * np.int32(2)), segred_group_sizze_21091):
          segment_sizze_nonzzero_24830 = smax32(np.int32(1), k2p2zq_18165)
          num_threads_24831 = (num_groups_21092 * segred_group_sizze_21091)
          if ((1 * (np.long(num_groups_21092) * np.long(segred_group_sizze_21091))) != 0):
            self.mainzisegred_small_21035_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21091)))),
                                                       np.int32(N_18148),
                                                       np.int32(m_18149),
                                                       np.int32(k2p2zq_18165),
                                                       np.int32(num_groups_21092),
                                                       mem_23547, res_mem_23946,
                                                       mem_24059,
                                                       np.int32(segment_sizze_nonzzero_24830))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_small_21035_var,
                                       ((np.long(num_groups_21092) * np.long(segred_group_sizze_21091)),),
                                       (np.long(segred_group_sizze_21091),))
            if synchronous:
              sync(self)
        else:
          groups_per_segment_24850 = sdiv_up32(num_groups_21092,
                                               smax32(np.int32(1),
                                                      (m_18149 * N_18148)))
          elements_per_thread_24851 = sdiv_up32(k2p2zq_18165,
                                                (segred_group_sizze_21091 * groups_per_segment_24850))
          virt_num_groups_24852 = (groups_per_segment_24850 * (m_18149 * N_18148))
          num_threads_24853 = (num_groups_21092 * segred_group_sizze_21091)
          threads_per_segment_24854 = (groups_per_segment_24850 * segred_group_sizze_21091)
          group_res_arr_mem_24855 = opencl_alloc(self,
                                                 (np.int32(4) * (sext_i32_i64(segred_group_sizze_21091) * sext_i32_i64(virt_num_groups_24852))),
                                                 "group_res_arr_mem_24855")
          mainzicounter_mem_24857 = self.mainzicounter_mem_24857
          if ((1 * (np.long(num_groups_21092) * np.long(segred_group_sizze_21091))) != 0):
            self.mainzisegred_large_21035_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long(np.int32(1))),
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21091)))),
                                                       np.int32(N_18148),
                                                       np.int32(k2p2zq_18165),
                                                       np.int32(num_groups_21092),
                                                       mem_23547, res_mem_23946,
                                                       mem_24059,
                                                       np.int32(groups_per_segment_24850),
                                                       np.int32(elements_per_thread_24851),
                                                       np.int32(virt_num_groups_24852),
                                                       np.int32(threads_per_segment_24854),
                                                       group_res_arr_mem_24855,
                                                       mainzicounter_mem_24857)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_large_21035_var,
                                       ((np.long(num_groups_21092) * np.long(segred_group_sizze_21091)),),
                                       (np.long(segred_group_sizze_21091),))
            if synchronous:
              sync(self)
        res_mem_24060 = mem_24059
      res_mem_24067 = res_mem_24060
    mem_23547 = None
    res_mem_23946 = None
    i_18392 = (N_18148 - np.int32(1))
    x_18393 = sle32(np.int32(0), i_18392)
    y_18394 = slt32(i_18392, N_18148)
    bounds_check_18395 = (x_18393 and y_18394)
    index_certs_18396 = True
    assert bounds_check_18395, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:18:29-34\n   #1  helpers.fut:14:13-20\n   #2  bfastfinaldetailed.fut:77:30-91\n   #3  /prelude/soacs.fut:56:19-23\n   #4  /prelude/soacs.fut:56:3-37\n   #5  bfastfinaldetailed.fut:73:5-80:25\n   #6  bfastfinaldetailed.fut:19:1-146:86\n" % ("Index [",
                                                                                                                                                                                                                                                                                                                                            i_18392,
                                                                                                                                                                                                                                                                                                                                            "] out of bounds for array of shape [",
                                                                                                                                                                                                                                                                                                                                            N_18148,
                                                                                                                                                                                                                                                                                                                                            "]."))
    fits_21191 = sle32(N_18148, max_group_sizze_19840)
    suff_intra_par_21189 = (self.sizes["main.suff_intra_par_24"] <= N_18148)
    intra_suff_and_fits_21192 = (suff_intra_par_21189 and fits_21191)
    segscan_group_sizze_21374 = self.sizes["main.segscan_group_size_21349"]
    max_num_groups_24891 = self.sizes["main.segscan_num_groups_21351"]
    num_groups_21375 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(y_21089,
                                                            sext_i32_i64(segscan_group_sizze_21374)),
                                                  sext_i32_i64(max_num_groups_24891))))
    mem_24072 = opencl_alloc(self, bytes_24061, "mem_24072")
    self.futhark_builtinzhreplicate_f32(mem_24072, (m_18149 * N_18148), np.nan)
    mem_24077 = opencl_alloc(self, bytes_24061, "mem_24077")
    self.futhark_builtinzhreplicate_i32(mem_24077, (m_18149 * N_18148),
                                        np.int32(0))
    segmap_group_sizze_21450 = self.sizes["main.segmap_group_size_21235"]
    segmap_group_sizze_21451 = sext_i32_i64(segmap_group_sizze_21450)
    bytes_24093 = (np.int64(4) * m_19378)
    local_memory_capacity_24990 = self.max_local_memory
    if (sle64((((bytes_23953 + bytes_23953) + bytes_23953) + bytes_23953),
              sext_i32_i64(local_memory_capacity_24990)) and intra_suff_and_fits_21192):
      mem_24095 = opencl_alloc(self, bytes_24093, "mem_24095")
      mem_24100 = opencl_alloc(self, bytes_24061, "mem_24100")
      mem_24105 = opencl_alloc(self, bytes_24061, "mem_24105")
      if ((1 * (np.long(m_18149) * np.long(N_18148))) != 0):
        self.mainzisegmap_intragroup_21114_var.set_args(self.global_failure,
                                                        cl.LocalMemory(np.long(bytes_23953)),
                                                        cl.LocalMemory(np.long(bytes_23953)),
                                                        cl.LocalMemory(np.long(bytes_23953)),
                                                        cl.LocalMemory(np.long(bytes_23953)),
                                                        np.int32(N_18148),
                                                        np.int32(N_18150),
                                                        np.int32(i_18392),
                                                        images_mem_23523,
                                                        res_mem_24067,
                                                        mem_24095, mem_24100,
                                                        mem_24105)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegmap_intragroup_21114_var,
                                   ((np.long(m_18149) * np.long(N_18148)),),
                                   (np.long(N_18148),))
        if synchronous:
          sync(self)
      res_mem_24121 = mem_24095
      res_mem_24122 = mem_24100
      res_mem_24123 = mem_24105
    else:
      mem_24111 = opencl_alloc(self, bytes_24061, "mem_24111")
      mem_24116 = opencl_alloc(self, bytes_24061, "mem_24116")
      if slt32(np.int32(0), (m_18149 * N_18148)):
        stage1_max_num_groups_24926 = self.max_group_size
        stage1_num_groups_24927 = smin32(stage1_max_num_groups_24926,
                                         num_groups_21375)
        num_threads_24928 = (stage1_num_groups_24927 * segscan_group_sizze_21374)
        if ((1 * (np.long(stage1_num_groups_24927) * np.long(segscan_group_sizze_21374))) != 0):
          self.mainziscan_stage1_21355_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * sext_i32_i64(segscan_group_sizze_21374))))),
                                                    np.int32(N_18148),
                                                    np.int32(m_18149),
                                                    np.int32(N_18150),
                                                    images_mem_23523,
                                                    res_mem_24067, mem_24111,
                                                    mem_24116,
                                                    np.int32(num_threads_24928))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_21355_var,
                                     ((np.long(stage1_num_groups_24927) * np.long(segscan_group_sizze_21374)),),
                                     (np.long(segscan_group_sizze_21374),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_24927))) != 0):
          self.mainziscan_stage2_21355_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * sext_i32_i64(stage1_num_groups_24927))))),
                                                    np.int32(N_18148),
                                                    np.int32(m_18149),
                                                    mem_24111,
                                                    np.int32(stage1_num_groups_24927),
                                                    np.int32(num_threads_24928))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_21355_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_24927)),),
                                     (np.long(stage1_num_groups_24927),))
          if synchronous:
            sync(self)
        required_groups_24968 = sdiv_up32((m_18149 * N_18148),
                                          segscan_group_sizze_21374)
        if ((1 * (np.long(num_groups_21375) * np.long(segscan_group_sizze_21374))) != 0):
          self.mainziscan_stage3_21355_var.set_args(self.global_failure,
                                                    np.int32(N_18148),
                                                    np.int32(m_18149),
                                                    np.int32(num_groups_21375),
                                                    mem_24111,
                                                    np.int32(num_threads_24928),
                                                    np.int32(required_groups_24968))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_21355_var,
                                     ((np.long(num_groups_21375) * np.long(segscan_group_sizze_21374)),),
                                     (np.long(segscan_group_sizze_21374),))
          if synchronous:
            sync(self)
      mem_24119 = opencl_alloc(self, bytes_24093, "mem_24119")
      group_sizze_24983 = self.sizes["main.group_size_24983"]
      num_groups_24984 = sdiv_up64(sext_i32_i64(m_18149),
                                   sext_i32_i64(group_sizze_24983))
      if ((1 * (np.long(sext_i64_i32(num_groups_24984)) * np.long(group_sizze_24983))) != 0):
        self.mainzicopy_24980_var.set_args(np.int32(N_18148), np.int32(m_18149),
                                           np.int32(i_18392), mem_24111,
                                           mem_24119)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzicopy_24980_var,
                                   ((np.long(sext_i64_i32(num_groups_24984)) * np.long(group_sizze_24983)),),
                                   (np.long(group_sizze_24983),))
        if synchronous:
          sync(self)
      segmap_usable_groups_64_21452 = sdiv_up64(y_21089,
                                                segmap_group_sizze_21451)
      segmap_usable_groups_21453 = sext_i64_i32(segmap_usable_groups_64_21452)
      if ((1 * (np.long(segmap_usable_groups_21453) * np.long(segmap_group_sizze_21450))) != 0):
        self.mainzisegmap_21230_var.set_args(self.global_failure,
                                             np.int32(N_18148),
                                             np.int32(m_18149), mem_24072,
                                             mem_24077, mem_24111, mem_24116)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_21230_var,
                                   ((np.long(segmap_usable_groups_21453) * np.long(segmap_group_sizze_21450)),),
                                   (np.long(segmap_group_sizze_21450),))
        if synchronous:
          sync(self)
      mem_24111 = None
      mem_24116 = None
      res_mem_24121 = mem_24119
      res_mem_24122 = mem_24072
      res_mem_24123 = mem_24077
    mem_24072 = None
    mem_24077 = None
    suff_outer_par_21471 = (self.sizes["main.suff_outer_par_27"] <= m_18149)
    fits_21553 = sle32(n_18153, max_group_sizze_19840)
    suff_intra_par_21551 = (self.sizes["main.suff_intra_par_28"] <= n_18153)
    intra_suff_and_fits_21554 = (suff_intra_par_21551 and fits_21553)
    segmap_group_sizze_21517 = self.sizes["main.segmap_group_size_21484"]
    segmap_group_sizze_21518 = sext_i32_i64(segmap_group_sizze_21517)
    nest_sizze_21648 = (m_19378 * n_19636)
    segred_group_sizze_21649 = self.sizes["main.segred_group_size_21630"]
    max_num_groups_24991 = self.sizes["main.segred_num_groups_21632"]
    num_groups_21650 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_21648,
                                                            sext_i32_i64(segred_group_sizze_21649)),
                                                  sext_i32_i64(max_num_groups_24991))))
    segred_group_sizze_21665 = self.sizes["main.segred_group_size_21608"]
    max_num_groups_24992 = self.sizes["main.segred_num_groups_21610"]
    num_groups_21666 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_21648,
                                                            sext_i32_i64(segred_group_sizze_21665)),
                                                  sext_i32_i64(max_num_groups_24992))))
    segmap_group_sizze_21681 = self.sizes["main.segmap_group_size_21589"]
    segmap_group_sizze_21682 = sext_i32_i64(segmap_group_sizze_21681)
    local_memory_capacity_25138 = self.max_local_memory
    if (sle64(np.int64(0),
              sext_i32_i64(local_memory_capacity_25138)) and suff_outer_par_21471):
      segmap_usable_groups_64_21519 = sdiv_up64(m_19378,
                                                segmap_group_sizze_21518)
      segmap_usable_groups_21520 = sext_i64_i32(segmap_usable_groups_64_21519)
      mem_24128 = opencl_alloc(self, bytes_23548, "mem_24128")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_24128, np.int32(0),
                                                  images_mem_23523, np.int32(0),
                                                  np.int32(1), N_18150, m_18149)
      mem_24133 = opencl_alloc(self, bytes_23968, "mem_24133")
      self.futhark_builtinzhgpu_map_transpose_f32(mem_24133, np.int32(0),
                                                  res_mem_24122, np.int32(0),
                                                  np.int32(1), N_18148, m_18149)
      mem_24137 = opencl_alloc(self, bytes_24093, "mem_24137")
      mem_24140 = opencl_alloc(self, bytes_24093, "mem_24140")
      mem_24143 = opencl_alloc(self, bytes_24093, "mem_24143")
      if ((1 * (np.long(segmap_usable_groups_21520) * np.long(segmap_group_sizze_21517))) != 0):
        self.mainzisegmap_21481_var.set_args(self.global_failure,
                                             np.int32(m_18149),
                                             np.int32(n_18153),
                                             np.float32(hfrac_18155),
                                             np.int32(k2p2_18163), mem_24128,
                                             mem_24133, mem_24137, mem_24140,
                                             mem_24143)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_21481_var,
                                   ((np.long(segmap_usable_groups_21520) * np.long(segmap_group_sizze_21517)),),
                                   (np.long(segmap_group_sizze_21517),))
        if synchronous:
          sync(self)
      mem_24128 = None
      mem_24133 = None
      res_mem_24174 = mem_24137
      res_mem_24175 = mem_24140
      res_mem_24176 = mem_24143
    else:
      local_memory_capacity_25137 = self.max_local_memory
      if (sle64(((np.int32(4) * sext_i32_i64(n_18153)) + (np.int32(4) * sext_i32_i64(n_18153))),
                sext_i32_i64(local_memory_capacity_25137)) and intra_suff_and_fits_21554):
        mem_24149 = opencl_alloc(self, bytes_24093, "mem_24149")
        mem_24152 = opencl_alloc(self, bytes_24093, "mem_24152")
        mem_24155 = opencl_alloc(self, bytes_24093, "mem_24155")
        if ((1 * (np.long(m_18149) * np.long(n_18153))) != 0):
          self.mainzisegmap_intragroup_21479_var.set_args(self.global_failure,
                                                          cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(n_18153)))),
                                                          cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(n_18153)))),
                                                          np.int32(N_18148),
                                                          np.int32(N_18150),
                                                          np.int32(n_18153),
                                                          np.float32(hfrac_18155),
                                                          np.int32(k2p2_18163),
                                                          images_mem_23523,
                                                          res_mem_24122,
                                                          mem_24149, mem_24152,
                                                          mem_24155)
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegmap_intragroup_21479_var,
                                     ((np.long(m_18149) * np.long(n_18153)),),
                                     (np.long(n_18153),))
          if synchronous:
            sync(self)
        res_mem_24171 = mem_24149
        res_mem_24172 = mem_24152
        res_mem_24173 = mem_24155
      else:
        mem_24159 = opencl_alloc(self, bytes_24093, "mem_24159")
        if slt32((n_18153 * np.int32(2)), segred_group_sizze_21649):
          segment_sizze_nonzzero_25014 = smax32(np.int32(1), n_18153)
          num_threads_25015 = (num_groups_21650 * segred_group_sizze_21649)
          if ((1 * (np.long(num_groups_21650) * np.long(segred_group_sizze_21649))) != 0):
            self.mainzisegred_small_21636_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21649)))),
                                                       np.int32(m_18149),
                                                       np.int32(N_18150),
                                                       np.int32(n_18153),
                                                       np.int32(num_groups_21650),
                                                       images_mem_23523,
                                                       mem_24159,
                                                       np.int32(segment_sizze_nonzzero_25014))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_small_21636_var,
                                       ((np.long(num_groups_21650) * np.long(segred_group_sizze_21649)),),
                                       (np.long(segred_group_sizze_21649),))
            if synchronous:
              sync(self)
        else:
          groups_per_segment_25034 = sdiv_up32(num_groups_21650,
                                               smax32(np.int32(1), m_18149))
          elements_per_thread_25035 = sdiv_up32(n_18153,
                                                (segred_group_sizze_21649 * groups_per_segment_25034))
          virt_num_groups_25036 = (groups_per_segment_25034 * m_18149)
          num_threads_25037 = (num_groups_21650 * segred_group_sizze_21649)
          threads_per_segment_25038 = (groups_per_segment_25034 * segred_group_sizze_21649)
          group_res_arr_mem_25039 = opencl_alloc(self,
                                                 (np.int32(4) * (sext_i32_i64(segred_group_sizze_21649) * sext_i32_i64(virt_num_groups_25036))),
                                                 "group_res_arr_mem_25039")
          mainzicounter_mem_25041 = self.mainzicounter_mem_25041
          if ((1 * (np.long(num_groups_21650) * np.long(segred_group_sizze_21649))) != 0):
            self.mainzisegred_large_21636_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long(np.int32(1))),
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21649)))),
                                                       np.int32(N_18150),
                                                       np.int32(n_18153),
                                                       np.int32(num_groups_21650),
                                                       images_mem_23523,
                                                       mem_24159,
                                                       np.int32(groups_per_segment_25034),
                                                       np.int32(elements_per_thread_25035),
                                                       np.int32(virt_num_groups_25036),
                                                       np.int32(threads_per_segment_25038),
                                                       group_res_arr_mem_25039,
                                                       mainzicounter_mem_25041)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_large_21636_var,
                                       ((np.long(num_groups_21650) * np.long(segred_group_sizze_21649)),),
                                       (np.long(segred_group_sizze_21649),))
            if synchronous:
              sync(self)
        mem_24163 = opencl_alloc(self, bytes_24093, "mem_24163")
        if slt32((n_18153 * np.int32(2)), segred_group_sizze_21665):
          segment_sizze_nonzzero_25073 = smax32(np.int32(1), n_18153)
          num_threads_25074 = (num_groups_21666 * segred_group_sizze_21665)
          if ((1 * (np.long(num_groups_21666) * np.long(segred_group_sizze_21665))) != 0):
            self.mainzisegred_small_21614_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21665)))),
                                                       np.int32(N_18148),
                                                       np.int32(m_18149),
                                                       np.int32(n_18153),
                                                       np.int32(num_groups_21666),
                                                       res_mem_24122, mem_24159,
                                                       mem_24163,
                                                       np.int32(segment_sizze_nonzzero_25073))
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_small_21614_var,
                                       ((np.long(num_groups_21666) * np.long(segred_group_sizze_21665)),),
                                       (np.long(segred_group_sizze_21665),))
            if synchronous:
              sync(self)
        else:
          groups_per_segment_25093 = sdiv_up32(num_groups_21666,
                                               smax32(np.int32(1), m_18149))
          elements_per_thread_25094 = sdiv_up32(n_18153,
                                                (segred_group_sizze_21665 * groups_per_segment_25093))
          virt_num_groups_25095 = (groups_per_segment_25093 * m_18149)
          num_threads_25096 = (num_groups_21666 * segred_group_sizze_21665)
          threads_per_segment_25097 = (groups_per_segment_25093 * segred_group_sizze_21665)
          group_res_arr_mem_25098 = opencl_alloc(self,
                                                 (np.int32(4) * (sext_i32_i64(segred_group_sizze_21665) * sext_i32_i64(virt_num_groups_25095))),
                                                 "group_res_arr_mem_25098")
          mainzicounter_mem_25100 = self.mainzicounter_mem_25100
          if ((1 * (np.long(num_groups_21666) * np.long(segred_group_sizze_21665))) != 0):
            self.mainzisegred_large_21614_var.set_args(self.global_failure,
                                                       cl.LocalMemory(np.long(np.int32(1))),
                                                       cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21665)))),
                                                       np.int32(N_18148),
                                                       np.int32(n_18153),
                                                       np.int32(num_groups_21666),
                                                       res_mem_24122, mem_24159,
                                                       mem_24163,
                                                       np.int32(groups_per_segment_25093),
                                                       np.int32(elements_per_thread_25094),
                                                       np.int32(virt_num_groups_25095),
                                                       np.int32(threads_per_segment_25097),
                                                       group_res_arr_mem_25098,
                                                       mainzicounter_mem_25100)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.mainzisegred_large_21614_var,
                                       ((np.long(num_groups_21666) * np.long(segred_group_sizze_21665)),),
                                       (np.long(segred_group_sizze_21665),))
            if synchronous:
              sync(self)
        segmap_usable_groups_64_21683 = sdiv_up64(m_19378,
                                                  segmap_group_sizze_21682)
        segmap_usable_groups_21684 = sext_i64_i32(segmap_usable_groups_64_21683)
        mem_24167 = opencl_alloc(self, bytes_24093, "mem_24167")
        mem_24170 = opencl_alloc(self, bytes_24093, "mem_24170")
        if ((1 * (np.long(segmap_usable_groups_21684) * np.long(segmap_group_sizze_21681))) != 0):
          self.mainzisegmap_21586_var.set_args(self.global_failure,
                                               np.int32(m_18149),
                                               np.float32(hfrac_18155),
                                               np.int32(k2p2_18163), mem_24159,
                                               mem_24163, mem_24167, mem_24170)
          cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_21586_var,
                                     ((np.long(segmap_usable_groups_21684) * np.long(segmap_group_sizze_21681)),),
                                     (np.long(segmap_group_sizze_21681),))
          if synchronous:
            sync(self)
        mem_24163 = None
        res_mem_24171 = mem_24167
        res_mem_24172 = mem_24159
        res_mem_24173 = mem_24170
      res_mem_24174 = res_mem_24171
      res_mem_24175 = res_mem_24172
      res_mem_24176 = res_mem_24173
    segred_group_sizze_21705 = self.sizes["main.segred_group_size_21704"]
    max_num_groups_25139 = self.sizes["main.segred_num_groups_21706"]
    num_groups_21707 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(m_19378,
                                                            sext_i32_i64(segred_group_sizze_21705)),
                                                  sext_i32_i64(max_num_groups_25139))))
    mem_24179 = opencl_alloc(self, np.int64(4), "mem_24179")
    mainzicounter_mem_25140 = self.mainzicounter_mem_25140
    group_res_arr_mem_25142 = opencl_alloc(self,
                                           (np.int32(4) * (sext_i32_i64(segred_group_sizze_21705) * sext_i32_i64(num_groups_21707))),
                                           "group_res_arr_mem_25142")
    num_threads_25144 = (num_groups_21707 * segred_group_sizze_21705)
    if ((1 * (np.long(num_groups_21707) * np.long(segred_group_sizze_21705))) != 0):
      self.mainzisegred_nonseg_21712_var.set_args(self.global_failure,
                                                  cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21705)))),
                                                  cl.LocalMemory(np.long(np.int32(1))),
                                                  np.int32(m_18149),
                                                  np.int32(num_groups_21707),
                                                  res_mem_24174, mem_24179,
                                                  mainzicounter_mem_25140,
                                                  group_res_arr_mem_25142,
                                                  np.int32(num_threads_25144))
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegred_nonseg_21712_var,
                                 ((np.long(num_groups_21707) * np.long(segred_group_sizze_21705)),),
                                 (np.long(segred_group_sizze_21705),))
      if synchronous:
        sync(self)
    read_res_25453 = np.empty(1, dtype=ct.c_int32)
    cl.enqueue_copy(self.queue, read_res_25453, mem_24179,
                    device_offset=(np.long(np.int64(0)) * 4),
                    is_blocking=synchronous)
    sync(self)
    res_18475 = read_res_25453[0]
    mem_24179 = None
    suff_outer_par_21714 = (self.sizes["main.suff_outer_par_29"] <= m_18149)
    segmap_group_sizze_21740 = self.sizes["main.segmap_group_size_21719"]
    segmap_group_sizze_21741 = sext_i32_i64(segmap_group_sizze_21740)
    res_21784 = sext_i32_i64(res_18475)
    nest_sizze_21787 = (m_19378 * res_21784)
    segred_group_sizze_21788 = self.sizes["main.segred_group_size_21764"]
    max_num_groups_25170 = self.sizes["main.segred_num_groups_21766"]
    num_groups_21789 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_21787,
                                                            sext_i32_i64(segred_group_sizze_21788)),
                                                  sext_i32_i64(max_num_groups_25170))))
    local_memory_capacity_25236 = self.max_local_memory
    if (sle64(np.int64(0),
              sext_i32_i64(local_memory_capacity_25236)) and suff_outer_par_21714):
      segmap_usable_groups_64_21742 = sdiv_up64(m_19378,
                                                segmap_group_sizze_21741)
      segmap_usable_groups_21743 = sext_i64_i32(segmap_usable_groups_64_21742)
      mem_24183 = opencl_alloc(self, bytes_24093, "mem_24183")
      if ((1 * (np.long(segmap_usable_groups_21743) * np.long(segmap_group_sizze_21740))) != 0):
        self.mainzisegmap_21716_var.set_args(self.global_failure,
                                             np.int32(N_18148),
                                             np.int32(m_18149),
                                             np.int32(res_18475), res_mem_24122,
                                             res_mem_24174, res_mem_24175,
                                             mem_24183)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_21716_var,
                                   ((np.long(segmap_usable_groups_21743) * np.long(segmap_group_sizze_21740)),),
                                   (np.long(segmap_group_sizze_21740),))
        if synchronous:
          sync(self)
      res_mem_24188 = mem_24183
    else:
      mem_24187 = opencl_alloc(self, bytes_24093, "mem_24187")
      if slt32((res_18475 * np.int32(2)), segred_group_sizze_21788):
        segment_sizze_nonzzero_25177 = smax32(np.int32(1), res_18475)
        num_threads_25178 = (num_groups_21789 * segred_group_sizze_21788)
        if ((1 * (np.long(num_groups_21789) * np.long(segred_group_sizze_21788))) != 0):
          self.mainzisegred_small_21770_var.set_args(self.global_failure,
                                                     cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21788)))),
                                                     np.int32(N_18148),
                                                     np.int32(m_18149),
                                                     np.int32(res_18475),
                                                     np.int32(num_groups_21789),
                                                     res_mem_24122,
                                                     res_mem_24174,
                                                     res_mem_24175, mem_24187,
                                                     np.int32(segment_sizze_nonzzero_25177))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegred_small_21770_var,
                                     ((np.long(num_groups_21789) * np.long(segred_group_sizze_21788)),),
                                     (np.long(segred_group_sizze_21788),))
          if synchronous:
            sync(self)
      else:
        groups_per_segment_25197 = sdiv_up32(num_groups_21789,
                                             smax32(np.int32(1), m_18149))
        elements_per_thread_25198 = sdiv_up32(res_18475,
                                              (segred_group_sizze_21788 * groups_per_segment_25197))
        virt_num_groups_25199 = (groups_per_segment_25197 * m_18149)
        num_threads_25200 = (num_groups_21789 * segred_group_sizze_21788)
        threads_per_segment_25201 = (groups_per_segment_25197 * segred_group_sizze_21788)
        group_res_arr_mem_25202 = opencl_alloc(self,
                                               (np.int32(4) * (sext_i32_i64(segred_group_sizze_21788) * sext_i32_i64(virt_num_groups_25199))),
                                               "group_res_arr_mem_25202")
        mainzicounter_mem_25204 = self.mainzicounter_mem_25204
        if ((1 * (np.long(num_groups_21789) * np.long(segred_group_sizze_21788))) != 0):
          self.mainzisegred_large_21770_var.set_args(self.global_failure,
                                                     cl.LocalMemory(np.long(np.int32(1))),
                                                     cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_21788)))),
                                                     np.int32(N_18148),
                                                     np.int32(res_18475),
                                                     np.int32(num_groups_21789),
                                                     res_mem_24122,
                                                     res_mem_24174,
                                                     res_mem_24175, mem_24187,
                                                     np.int32(groups_per_segment_25197),
                                                     np.int32(elements_per_thread_25198),
                                                     np.int32(virt_num_groups_25199),
                                                     np.int32(threads_per_segment_25201),
                                                     group_res_arr_mem_25202,
                                                     mainzicounter_mem_25204)
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegred_large_21770_var,
                                     ((np.long(num_groups_21789) * np.long(segred_group_sizze_21788)),),
                                     (np.long(segred_group_sizze_21788),))
          if synchronous:
            sync(self)
      res_mem_24188 = mem_24187
    iota_arg_18497 = (N_18148 - n_18153)
    bounds_invalid_upwards_18498 = slt32(iota_arg_18497, np.int32(0))
    valid_18499 = not(bounds_invalid_upwards_18498)
    range_valid_c_18500 = True
    assert valid_18499, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  bfastfinaldetailed.fut:110:22-31\n   #2  bfastfinaldetailed.fut:19:1-146:86\n" % ("Range ",
                                                                                                                                                                                           np.int32(0),
                                                                                                                                                                                           "..",
                                                                                                                                                                                           np.int32(1),
                                                                                                                                                                                           "..<",
                                                                                                                                                                                           iota_arg_18497,
                                                                                                                                                                                           " is invalid."))
    x_18502 = (np.int32(1) + n_18153)
    index_certs_18503 = True
    assert bounds_check_18395, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  bfastfinaldetailed.fut:108:63-81\n   #1  bfastfinaldetailed.fut:106:15-110:32\n   #2  bfastfinaldetailed.fut:19:1-146:86\n" % ("Index [",
                                                                                                                                                                                                        i_18392,
                                                                                                                                                                                                        "] out of bounds for array of shape [",
                                                                                                                                                                                                        N_18148,
                                                                                                                                                                                                        "]."))
    read_res_25455 = np.empty(1, dtype=ct.c_int32)
    cl.enqueue_copy(self.queue, read_res_25455, mappingindices_mem_23522,
                    device_offset=(np.long(sext_i32_i64(i_18392)) * 4),
                    is_blocking=synchronous)
    sync(self)
    r32_arg_18504 = read_res_25455[0]
    res_18505 = sitofp_i32_f32(r32_arg_18504)
    iota_arg_21866 = sext_i32_i64(iota_arg_18497)
    segmap_group_sizze_21868 = self.sizes["main.segmap_group_size_21850"]
    segmap_group_sizze_21869 = sext_i32_i64(segmap_group_sizze_21868)
    segmap_usable_groups_64_21870 = sdiv_up64(iota_arg_21866,
                                              segmap_group_sizze_21869)
    segmap_usable_groups_21871 = sext_i64_i32(segmap_usable_groups_64_21870)
    bytes_24190 = (np.int64(4) * iota_arg_21866)
    mem_24192 = opencl_alloc(self, bytes_24190, "mem_24192")
    if ((1 * (np.long(segmap_usable_groups_21871) * np.long(segmap_group_sizze_21868))) != 0):
      self.mainzisegmap_21847_var.set_args(self.global_failure,
                                           np.float32(lam_18156),
                                           np.int32(iota_arg_18497),
                                           np.int32(x_18502),
                                           np.float32(res_18505),
                                           mappingindices_mem_23522, mem_24192)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_21847_var,
                                 ((np.long(segmap_usable_groups_21871) * np.long(segmap_group_sizze_21868)),),
                                 (np.long(segmap_group_sizze_21868),))
      if synchronous:
        sync(self)
    range_valid_c_18518 = True
    assert valid_18499, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:60:3-10\n   #1  bfastfinaldetailed.fut:121:29-40\n   #2  /prelude/functional.fut:9:42-44\n   #3  bfastfinaldetailed.fut:115:38-144:9\n   #4  bfastfinaldetailed.fut:19:1-146:86\n" % ("Range ",
                                                                                                                                                                                                                                                                               np.int32(0),
                                                                                                                                                                                                                                                                               "..",
                                                                                                                                                                                                                                                                               np.int32(1),
                                                                                                                                                                                                                                                                               "..<",
                                                                                                                                                                                                                                                                               iota_arg_18497,
                                                                                                                                                                                                                                                                               " is invalid."))
    fits_22127 = sle32(iota_arg_18497, max_group_sizze_19840)
    suff_intra_par_22125 = (self.sizes["main.suff_intra_par_32"] <= iota_arg_18497)
    intra_suff_and_fits_22128 = (suff_intra_par_22125 and fits_22127)
    segmap_group_sizze_22495 = self.sizes["main.segmap_group_size_22474"]
    max_num_groups_25242 = self.sizes["main.segmap_num_groups_22476"]
    num_groups_22496 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(m_19378,
                                                            sext_i32_i64(segmap_group_sizze_22495)),
                                                  sext_i32_i64(max_num_groups_25242))))
    nest_sizze_22519 = (m_19378 * iota_arg_21866)
    segscan_group_sizze_22520 = self.sizes["main.segscan_group_size_22432"]
    max_num_groups_25243 = self.sizes["main.segscan_num_groups_22434"]
    num_groups_22521 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_22519,
                                                            sext_i32_i64(segscan_group_sizze_22520)),
                                                  sext_i32_i64(max_num_groups_25243))))
    segred_group_sizze_22562 = self.sizes["main.segred_group_size_22373"]
    max_num_groups_25244 = self.sizes["main.segred_num_groups_22375"]
    num_groups_22563 = sext_i64_i32(smax64(np.int64(1),
                                           smin64(sdiv_up64(nest_sizze_22519,
                                                            sext_i32_i64(segred_group_sizze_22562)),
                                                  sext_i32_i64(max_num_groups_25244))))
    segmap_group_sizze_22603 = self.sizes["main.segmap_group_size_22343"]
    segmap_group_sizze_22604 = sext_i32_i64(segmap_group_sizze_22603)
    segmap_group_sizze_22629 = self.sizes["main.segmap_group_size_22311"]
    segmap_group_sizze_22630 = sext_i32_i64(segmap_group_sizze_22629)
    bytes_24193 = (np.int64(4) * nest_sizze_22519)
    mem_24197 = opencl_alloc(self, bytes_24193, "mem_24197")
    self.futhark_builtinzhreplicate_f32(mem_24197, (m_18149 * iota_arg_18497),
                                        np.nan)
    segmap_group_sizze_22678 = self.sizes["main.segmap_group_size_22247"]
    segmap_group_sizze_22679 = sext_i32_i64(segmap_group_sizze_22678)
    local_memory_capacity_25443 = self.max_local_memory
    if (sle64((((((bytes_24190 + bytes_24190) + (np.int32(1) * sext_i32_i64(iota_arg_18497))) + (np.int32(4) * sext_i32_i64(iota_arg_18497))) + (np.int32(4) * sext_i32_i64(iota_arg_18497))) + bytes_24190),
              sext_i32_i64(local_memory_capacity_25443)) and intra_suff_and_fits_22128):
      mem_24215 = opencl_alloc(self, bytes_24193, "mem_24215")
      mem_24220 = opencl_alloc(self, bytes_24193, "mem_24220")
      mem_24223 = opencl_alloc(self, bytes_24093, "mem_24223")
      mem_24226 = opencl_alloc(self, bytes_24093, "mem_24226")
      if ((1 * (np.long(m_18149) * np.long(iota_arg_18497))) != 0):
        self.mainzisegmap_intragroup_21895_var.set_args(self.global_failure,
                                                        cl.LocalMemory(np.long(bytes_24190)),
                                                        cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(iota_arg_18497)))),
                                                        cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(iota_arg_18497)))),
                                                        cl.LocalMemory(np.long((np.int32(1) * sext_i32_i64(iota_arg_18497)))),
                                                        cl.LocalMemory(np.long(bytes_24190)),
                                                        cl.LocalMemory(np.long(bytes_24190)),
                                                        np.int32(N_18148),
                                                        np.int32(n_18153),
                                                        np.int32(iota_arg_18497),
                                                        res_mem_24121,
                                                        res_mem_24122,
                                                        res_mem_24123,
                                                        res_mem_24174,
                                                        res_mem_24175,
                                                        res_mem_24176,
                                                        res_mem_24188,
                                                        mem_24192, mem_24215,
                                                        mem_24220, mem_24223,
                                                        mem_24226)
        cl.enqueue_nd_range_kernel(self.queue,
                                   self.mainzisegmap_intragroup_21895_var,
                                   ((np.long(m_18149) * np.long(iota_arg_18497)),),
                                   (np.long(iota_arg_18497),))
        if synchronous:
          sync(self)
      res_mem_24268 = mem_24215
      res_mem_24269 = mem_24220
      res_mem_24270 = mem_24223
      res_mem_24271 = mem_24226
    else:
      mem_24230 = opencl_alloc(self, bytes_24093, "mem_24230")
      mem_24233 = opencl_alloc(self, bytes_24093, "mem_24233")
      if ((1 * (np.long(num_groups_22496) * np.long(segmap_group_sizze_22495))) != 0):
        self.mainzisegmap_22471_var.set_args(self.global_failure,
                                             np.int32(m_18149),
                                             np.int32(num_groups_22496),
                                             res_mem_24121, res_mem_24175,
                                             res_mem_24176, mem_24230,
                                             mem_24233)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_22471_var,
                                   ((np.long(num_groups_22496) * np.long(segmap_group_sizze_22495)),),
                                   (np.long(segmap_group_sizze_22495),))
        if synchronous:
          sync(self)
      mem_24239 = opencl_alloc(self, bytes_24193, "mem_24239")
      if slt32(np.int32(0), (m_18149 * iota_arg_18497)):
        stage1_max_num_groups_25277 = self.max_group_size
        stage1_num_groups_25278 = smin32(stage1_max_num_groups_25277,
                                         num_groups_22521)
        num_threads_25279 = (stage1_num_groups_25278 * segscan_group_sizze_22520)
        if ((1 * (np.long(stage1_num_groups_25278) * np.long(segscan_group_sizze_22520))) != 0):
          self.mainziscan_stage1_22438_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * sext_i32_i64(segscan_group_sizze_22520))))),
                                                    np.int32(N_18148),
                                                    np.int32(m_18149),
                                                    np.int32(iota_arg_18497),
                                                    res_mem_24122,
                                                    res_mem_24174,
                                                    res_mem_24175,
                                                    res_mem_24188, mem_24233,
                                                    mem_24239,
                                                    np.int32(num_threads_25279))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage1_22438_var,
                                     ((np.long(stage1_num_groups_25278) * np.long(segscan_group_sizze_22520)),),
                                     (np.long(segscan_group_sizze_22520),))
          if synchronous:
            sync(self)
        if ((1 * (np.long(np.int32(1)) * np.long(stage1_num_groups_25278))) != 0):
          self.mainziscan_stage2_22438_var.set_args(self.global_failure,
                                                    cl.LocalMemory(np.long(smax64(np.int64(1),
                                                                                  (np.int32(4) * sext_i32_i64(stage1_num_groups_25278))))),
                                                    np.int32(m_18149),
                                                    np.int32(iota_arg_18497),
                                                    mem_24239,
                                                    np.int32(stage1_num_groups_25278),
                                                    np.int32(num_threads_25279))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage2_22438_var,
                                     ((np.long(np.int32(1)) * np.long(stage1_num_groups_25278)),),
                                     (np.long(stage1_num_groups_25278),))
          if synchronous:
            sync(self)
        required_groups_25319 = sdiv_up32((m_18149 * iota_arg_18497),
                                          segscan_group_sizze_22520)
        if ((1 * (np.long(num_groups_22521) * np.long(segscan_group_sizze_22520))) != 0):
          self.mainziscan_stage3_22438_var.set_args(self.global_failure,
                                                    np.int32(m_18149),
                                                    np.int32(iota_arg_18497),
                                                    np.int32(num_groups_22521),
                                                    mem_24239,
                                                    np.int32(num_threads_25279),
                                                    np.int32(required_groups_25319))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainziscan_stage3_22438_var,
                                     ((np.long(num_groups_22521) * np.long(segscan_group_sizze_22520)),),
                                     (np.long(segscan_group_sizze_22520),))
          if synchronous:
            sync(self)
      mem_24242 = opencl_alloc(self, m_19378, "mem_24242")
      mem_24245 = opencl_alloc(self, bytes_24093, "mem_24245")
      mem_24248 = opencl_alloc(self, bytes_24093, "mem_24248")
      mem_24253 = opencl_alloc(self, bytes_24193, "mem_24253")
      if slt32((iota_arg_18497 * np.int32(2)), segred_group_sizze_22562):
        segment_sizze_nonzzero_25331 = smax32(np.int32(1), iota_arg_18497)
        num_threads_25332 = (num_groups_22563 * segred_group_sizze_22562)
        if ((1 * (np.long(num_groups_22563) * np.long(segred_group_sizze_22562))) != 0):
          self.mainzisegred_small_22379_var.set_args(self.global_failure,
                                                     cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_22562)))),
                                                     cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_22562)))),
                                                     cl.LocalMemory(np.long((np.int32(1) * sext_i32_i64(segred_group_sizze_22562)))),
                                                     np.int32(m_18149),
                                                     np.int32(iota_arg_18497),
                                                     np.int32(num_groups_22563),
                                                     mem_24192, mem_24230,
                                                     mem_24233, mem_24239,
                                                     mem_24242, mem_24245,
                                                     mem_24248, mem_24253,
                                                     np.int32(segment_sizze_nonzzero_25331))
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegred_small_22379_var,
                                     ((np.long(num_groups_22563) * np.long(segred_group_sizze_22562)),),
                                     (np.long(segred_group_sizze_22562),))
          if synchronous:
            sync(self)
      else:
        groups_per_segment_25366 = sdiv_up32(num_groups_22563,
                                             smax32(np.int32(1), m_18149))
        elements_per_thread_25367 = sdiv_up32(iota_arg_18497,
                                              (segred_group_sizze_22562 * groups_per_segment_25366))
        virt_num_groups_25368 = (groups_per_segment_25366 * m_18149)
        num_threads_25369 = (num_groups_22563 * segred_group_sizze_22562)
        threads_per_segment_25370 = (groups_per_segment_25366 * segred_group_sizze_22562)
        group_res_arr_mem_25371 = opencl_alloc(self,
                                               (np.int32(1) * (sext_i32_i64(segred_group_sizze_22562) * sext_i32_i64(virt_num_groups_25368))),
                                               "group_res_arr_mem_25371")
        group_res_arr_mem_25373 = opencl_alloc(self,
                                               (np.int32(4) * (sext_i32_i64(segred_group_sizze_22562) * sext_i32_i64(virt_num_groups_25368))),
                                               "group_res_arr_mem_25373")
        group_res_arr_mem_25375 = opencl_alloc(self,
                                               (np.int32(4) * (sext_i32_i64(segred_group_sizze_22562) * sext_i32_i64(virt_num_groups_25368))),
                                               "group_res_arr_mem_25375")
        mainzicounter_mem_25377 = self.mainzicounter_mem_25377
        if ((1 * (np.long(num_groups_22563) * np.long(segred_group_sizze_22562))) != 0):
          self.mainzisegred_large_22379_var.set_args(self.global_failure,
                                                     cl.LocalMemory(np.long(np.int32(1))),
                                                     cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_22562)))),
                                                     cl.LocalMemory(np.long((np.int32(4) * sext_i32_i64(segred_group_sizze_22562)))),
                                                     cl.LocalMemory(np.long((np.int32(1) * sext_i32_i64(segred_group_sizze_22562)))),
                                                     np.int32(iota_arg_18497),
                                                     np.int32(num_groups_22563),
                                                     mem_24192, mem_24230,
                                                     mem_24233, mem_24239,
                                                     mem_24242, mem_24245,
                                                     mem_24248, mem_24253,
                                                     np.int32(groups_per_segment_25366),
                                                     np.int32(elements_per_thread_25367),
                                                     np.int32(virt_num_groups_25368),
                                                     group_res_arr_mem_25371,
                                                     group_res_arr_mem_25373,
                                                     group_res_arr_mem_25375,
                                                     mainzicounter_mem_25377)
          cl.enqueue_nd_range_kernel(self.queue,
                                     self.mainzisegred_large_22379_var,
                                     ((np.long(num_groups_22563) * np.long(segred_group_sizze_22562)),),
                                     (np.long(segred_group_sizze_22562),))
          if synchronous:
            sync(self)
      mem_24230 = None
      mem_24239 = None
      segmap_usable_groups_64_22605 = sdiv_up64(m_19378,
                                                segmap_group_sizze_22604)
      segmap_usable_groups_22606 = sext_i64_i32(segmap_usable_groups_64_22605)
      mem_24256 = opencl_alloc(self, bytes_24093, "mem_24256")
      if ((sext_i32_i64(m_18149) * np.int32(4)) != 0):
        cl.enqueue_copy(self.queue, mem_24256, mem_24248,
                        dest_offset=np.long(np.int64(0)),
                        src_offset=np.long(np.int64(0)),
                        byte_count=np.long((sext_i32_i64(m_18149) * np.int32(4))))
      if synchronous:
        sync(self)
      mem_24248 = None
      mem_24259 = opencl_alloc(self, m_19378, "mem_24259")
      mem_24262 = opencl_alloc(self, bytes_24093, "mem_24262")
      if ((1 * (np.long(segmap_usable_groups_22606) * np.long(segmap_group_sizze_22603))) != 0):
        self.mainzisegmap_22340_var.set_args(self.global_failure,
                                             np.int32(m_18149), mem_24242,
                                             mem_24245, mem_24259, mem_24262)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_22340_var,
                                   ((np.long(segmap_usable_groups_22606) * np.long(segmap_group_sizze_22603)),),
                                   (np.long(segmap_group_sizze_22603),))
        if synchronous:
          sync(self)
      mem_24242 = None
      mem_24245 = None
      segmap_usable_groups_64_22631 = sdiv_up64(m_19378,
                                                segmap_group_sizze_22630)
      segmap_usable_groups_22632 = sext_i64_i32(segmap_usable_groups_64_22631)
      mem_24266 = opencl_alloc(self, bytes_24093, "mem_24266")
      if ((1 * (np.long(segmap_usable_groups_22632) * np.long(segmap_group_sizze_22629))) != 0):
        self.mainzisegmap_22308_var.set_args(self.global_failure,
                                             np.int32(N_18148),
                                             np.int32(m_18149),
                                             np.int32(n_18153), res_mem_24123,
                                             res_mem_24175, mem_24233,
                                             mem_24259, mem_24262, mem_24266)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_22308_var,
                                   ((np.long(segmap_usable_groups_22632) * np.long(segmap_group_sizze_22629)),),
                                   (np.long(segmap_group_sizze_22629),))
        if synchronous:
          sync(self)
      mem_24259 = None
      mem_24262 = None
      segmap_usable_groups_64_22680 = sdiv_up64(nest_sizze_22519,
                                                segmap_group_sizze_22679)
      segmap_usable_groups_22681 = sext_i64_i32(segmap_usable_groups_64_22680)
      if ((1 * (np.long(segmap_usable_groups_22681) * np.long(segmap_group_sizze_22678))) != 0):
        self.mainzisegmap_22242_var.set_args(self.global_failure,
                                             np.int32(N_18148),
                                             np.int32(m_18149),
                                             np.int32(n_18153),
                                             np.int32(iota_arg_18497),
                                             res_mem_24123, res_mem_24175,
                                             mem_24197, mem_24233, mem_24253)
        cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_22242_var,
                                   ((np.long(segmap_usable_groups_22681) * np.long(segmap_group_sizze_22678)),),
                                   (np.long(segmap_group_sizze_22678),))
        if synchronous:
          sync(self)
      mem_24233 = None
      res_mem_24268 = mem_24197
      res_mem_24269 = mem_24253
      res_mem_24270 = mem_24266
      res_mem_24271 = mem_24256
    res_mem_24123 = None
    res_mem_24174 = None
    mem_24197 = None
    out_arrsizze_24437 = m_18149
    out_arrsizze_24439 = m_18149
    out_arrsizze_24441 = m_18149
    out_arrsizze_24443 = m_18149
    out_arrsizze_24445 = m_18149
    out_arrsizze_24446 = iota_arg_18497
    out_arrsizze_24448 = m_18149
    out_arrsizze_24449 = iota_arg_18497
    out_arrsizze_24451 = iota_arg_18497
    out_arrsizze_24453 = m_18149
    out_arrsizze_24455 = m_18149
    out_arrsizze_24457 = m_18149
    out_arrsizze_24458 = N_18148
    out_arrsizze_24460 = m_18149
    out_arrsizze_24461 = N_18148
    out_mem_24436 = res_mem_24188
    out_mem_24438 = res_mem_24121
    out_mem_24440 = res_mem_24175
    out_mem_24442 = res_mem_24176
    out_mem_24444 = res_mem_24268
    out_mem_24447 = res_mem_24269
    out_mem_24450 = mem_24192
    out_mem_24452 = res_mem_24270
    out_mem_24454 = res_mem_24271
    out_mem_24456 = res_mem_24122
    out_mem_24459 = res_mem_24067
    return (out_mem_24436, out_arrsizze_24437, out_mem_24438,
            out_arrsizze_24439, out_mem_24440, out_arrsizze_24441,
            out_mem_24442, out_arrsizze_24443, out_mem_24444,
            out_arrsizze_24445, out_arrsizze_24446, out_mem_24447,
            out_arrsizze_24448, out_arrsizze_24449, out_mem_24450,
            out_arrsizze_24451, out_mem_24452, out_arrsizze_24453,
            out_mem_24454, out_arrsizze_24455, out_mem_24456,
            out_arrsizze_24457, out_arrsizze_24458, out_mem_24459,
            out_arrsizze_24460, out_arrsizze_24461)
  def futhark_remove_nans(self, images_mem_23522, m_18134, n_18135, p_18136,
                          nan_value_18137):
    m_18843 = sext_i32_i64(m_18134)
    n_18844 = sext_i32_i64(n_18135)
    p_18845 = sext_i32_i64(p_18136)
    y_18847 = (n_18844 * p_18845)
    nest_sizze_18848 = (m_18843 * y_18847)
    segmap_group_sizze_18849 = self.sizes["remove_nans.segmap_group_size_18765"]
    segmap_group_sizze_18850 = sext_i32_i64(segmap_group_sizze_18849)
    segmap_usable_groups_64_18851 = sdiv_up64(nest_sizze_18848,
                                              segmap_group_sizze_18850)
    segmap_usable_groups_18852 = sext_i64_i32(segmap_usable_groups_64_18851)
    binop_x_23527 = (m_18843 * n_18844)
    binop_x_23529 = (p_18845 * binop_x_23527)
    bytes_23524 = (np.int64(4) * binop_x_23529)
    mem_23530 = opencl_alloc(self, bytes_23524, "mem_23530")
    if ((1 * (np.long(segmap_usable_groups_18852) * np.long(segmap_group_sizze_18849))) != 0):
      self.remove_nanszisegmap_18758_var.set_args(self.global_failure,
                                                  np.int32(m_18134),
                                                  np.int32(n_18135),
                                                  np.int32(p_18136),
                                                  np.int16(nan_value_18137),
                                                  images_mem_23522, mem_23530)
      cl.enqueue_nd_range_kernel(self.queue, self.remove_nanszisegmap_18758_var,
                                 ((np.long(segmap_usable_groups_18852) * np.long(segmap_group_sizze_18849)),),
                                 (np.long(segmap_group_sizze_18849),))
      if synchronous:
        sync(self)
    out_arrsizze_24437 = m_18134
    out_arrsizze_24438 = n_18135
    out_arrsizze_24439 = p_18136
    out_mem_24436 = mem_23530
    return (out_mem_24436, out_arrsizze_24437, out_arrsizze_24438,
            out_arrsizze_24439)
  def futhark_reshapeTransp(self, images_mem_23522, m_18127, n_18128, p_18129):
    flatten_to_arg_18131 = (n_18128 * p_18129)
    binop_x_23524 = sext_i32_i64(flatten_to_arg_18131)
    binop_y_23525 = sext_i32_i64(m_18127)
    binop_x_23526 = (binop_x_23524 * binop_y_23525)
    bytes_23523 = (np.int64(4) * binop_x_23526)
    mem_23527 = opencl_alloc(self, bytes_23523, "mem_23527")
    self.futhark_builtinzhgpu_map_transpose_f32(mem_23527, np.int32(0),
                                                images_mem_23522, np.int32(0),
                                                np.int32(1),
                                                flatten_to_arg_18131, m_18127)
    out_arrsizze_24437 = flatten_to_arg_18131
    out_arrsizze_24438 = m_18127
    out_mem_24436 = mem_23527
    return (out_mem_24436, out_arrsizze_24437, out_arrsizze_24438)
  def main(self, trend_18151_ext, k_18152_ext, n_18153_ext, freq_18154_ext,
           hfrac_18155_ext, lam_18156_ext, mappingindices_mem_23522_ext,
           images_mem_23523_ext):
    try:
      trend_18151 = np.int32(ct.c_int32(trend_18151_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(trend_18151_ext),
                                                                                                                            trend_18151_ext))
    try:
      k_18152 = np.int32(ct.c_int32(k_18152_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(k_18152_ext),
                                                                                                                            k_18152_ext))
    try:
      n_18153 = np.int32(ct.c_int32(n_18153_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i32",
                                                                                                                            type(n_18153_ext),
                                                                                                                            n_18153_ext))
    try:
      freq_18154 = np.float32(ct.c_float(freq_18154_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(freq_18154_ext),
                                                                                                                            freq_18154_ext))
    try:
      hfrac_18155 = np.float32(ct.c_float(hfrac_18155_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(hfrac_18155_ext),
                                                                                                                            hfrac_18155_ext))
    try:
      lam_18156 = np.float32(ct.c_float(lam_18156_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(lam_18156_ext),
                                                                                                                            lam_18156_ext))
    try:
      assert ((type(mappingindices_mem_23522_ext) in [np.ndarray,
                                                      cl.array.Array]) and (mappingindices_mem_23522_ext.dtype == np.int32)), "Parameter has unexpected type"
      N_18148 = np.int32(mappingindices_mem_23522_ext.shape[0])
      if (type(mappingindices_mem_23522_ext) == cl.array.Array):
        mappingindices_mem_23522 = mappingindices_mem_23522_ext.data
      else:
        mappingindices_mem_23522 = opencl_alloc(self,
                                                np.int64(mappingindices_mem_23522_ext.nbytes),
                                                "mappingindices_mem_23522")
        if (np.int64(mappingindices_mem_23522_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, mappingindices_mem_23522,
                          normaliseArray(mappingindices_mem_23522_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]i32",
                                                                                                                            type(mappingindices_mem_23522_ext),
                                                                                                                            mappingindices_mem_23522_ext))
    try:
      assert ((type(images_mem_23523_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_23523_ext.dtype == np.float32)), "Parameter has unexpected type"
      m_18149 = np.int32(images_mem_23523_ext.shape[0])
      N_18150 = np.int32(images_mem_23523_ext.shape[1])
      if (type(images_mem_23523_ext) == cl.array.Array):
        images_mem_23523 = images_mem_23523_ext.data
      else:
        images_mem_23523 = opencl_alloc(self,
                                        np.int64(images_mem_23523_ext.nbytes),
                                        "images_mem_23523")
        if (np.int64(images_mem_23523_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_23523,
                          normaliseArray(images_mem_23523_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f32",
                                                                                                                            type(images_mem_23523_ext),
                                                                                                                            images_mem_23523_ext))
    (out_mem_24436, out_arrsizze_24437, out_mem_24438, out_arrsizze_24439,
     out_mem_24440, out_arrsizze_24441, out_mem_24442, out_arrsizze_24443,
     out_mem_24444, out_arrsizze_24445, out_arrsizze_24446, out_mem_24447,
     out_arrsizze_24448, out_arrsizze_24449, out_mem_24450, out_arrsizze_24451,
     out_mem_24452, out_arrsizze_24453, out_mem_24454, out_arrsizze_24455,
     out_mem_24456, out_arrsizze_24457, out_arrsizze_24458, out_mem_24459,
     out_arrsizze_24460,
     out_arrsizze_24461) = self.futhark_main(mappingindices_mem_23522,
                                             images_mem_23523, N_18148, m_18149,
                                             N_18150, trend_18151, k_18152,
                                             n_18153, freq_18154, hfrac_18155,
                                             lam_18156)
    sync(self)
    return (cl.array.Array(self.queue, (out_arrsizze_24437,), ct.c_float,
                           data=out_mem_24436), cl.array.Array(self.queue,
                                                               (out_arrsizze_24439,),
                                                               ct.c_int32,
                                                               data=out_mem_24438),
            cl.array.Array(self.queue, (out_arrsizze_24441,), ct.c_int32,
                           data=out_mem_24440), cl.array.Array(self.queue,
                                                               (out_arrsizze_24443,),
                                                               ct.c_float,
                                                               data=out_mem_24442),
            cl.array.Array(self.queue, (out_arrsizze_24445, out_arrsizze_24446),
                           ct.c_float, data=out_mem_24444),
            cl.array.Array(self.queue, (out_arrsizze_24448, out_arrsizze_24449),
                           ct.c_float, data=out_mem_24447),
            cl.array.Array(self.queue, (out_arrsizze_24451,), ct.c_float,
                           data=out_mem_24450), cl.array.Array(self.queue,
                                                               (out_arrsizze_24453,),
                                                               ct.c_int32,
                                                               data=out_mem_24452),
            cl.array.Array(self.queue, (out_arrsizze_24455,), ct.c_float,
                           data=out_mem_24454), cl.array.Array(self.queue,
                                                               (out_arrsizze_24457,
                                                                out_arrsizze_24458),
                                                               ct.c_float,
                                                               data=out_mem_24456),
            cl.array.Array(self.queue, (out_arrsizze_24460, out_arrsizze_24461),
                           ct.c_float, data=out_mem_24459))
  def remove_nans(self, nan_value_18137_ext, images_mem_23522_ext):
    try:
      nan_value_18137 = np.int16(ct.c_int16(nan_value_18137_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i16",
                                                                                                                            type(nan_value_18137_ext),
                                                                                                                            nan_value_18137_ext))
    try:
      assert ((type(images_mem_23522_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_23522_ext.dtype == np.int16)), "Parameter has unexpected type"
      m_18134 = np.int32(images_mem_23522_ext.shape[0])
      n_18135 = np.int32(images_mem_23522_ext.shape[1])
      p_18136 = np.int32(images_mem_23522_ext.shape[2])
      if (type(images_mem_23522_ext) == cl.array.Array):
        images_mem_23522 = images_mem_23522_ext.data
      else:
        images_mem_23522 = opencl_alloc(self,
                                        np.int64(images_mem_23522_ext.nbytes),
                                        "images_mem_23522")
        if (np.int64(images_mem_23522_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_23522,
                          normaliseArray(images_mem_23522_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]i16",
                                                                                                                            type(images_mem_23522_ext),
                                                                                                                            images_mem_23522_ext))
    (out_mem_24436, out_arrsizze_24437, out_arrsizze_24438,
     out_arrsizze_24439) = self.futhark_remove_nans(images_mem_23522, m_18134,
                                                    n_18135, p_18136,
                                                    nan_value_18137)
    sync(self)
    return cl.array.Array(self.queue, (out_arrsizze_24437, out_arrsizze_24438,
                                       out_arrsizze_24439), ct.c_float,
                          data=out_mem_24436)
  def reshapeTransp(self, images_mem_23522_ext):
    try:
      assert ((type(images_mem_23522_ext) in [np.ndarray,
                                              cl.array.Array]) and (images_mem_23522_ext.dtype == np.float32)), "Parameter has unexpected type"
      m_18127 = np.int32(images_mem_23522_ext.shape[0])
      n_18128 = np.int32(images_mem_23522_ext.shape[1])
      p_18129 = np.int32(images_mem_23522_ext.shape[2])
      if (type(images_mem_23522_ext) == cl.array.Array):
        images_mem_23522 = images_mem_23522_ext.data
      else:
        images_mem_23522 = opencl_alloc(self,
                                        np.int64(images_mem_23522_ext.nbytes),
                                        "images_mem_23522")
        if (np.int64(images_mem_23522_ext.nbytes) != 0):
          cl.enqueue_copy(self.queue, images_mem_23522,
                          normaliseArray(images_mem_23522_ext),
                          is_blocking=synchronous)
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][][]f32",
                                                                                                                            type(images_mem_23522_ext),
                                                                                                                            images_mem_23522_ext))
    (out_mem_24436, out_arrsizze_24437,
     out_arrsizze_24438) = self.futhark_reshapeTransp(images_mem_23522, m_18127,
                                                      n_18128, p_18129)
    sync(self)
    return cl.array.Array(self.queue, (out_arrsizze_24437, out_arrsizze_24438),
                          ct.c_float, data=out_mem_24436)