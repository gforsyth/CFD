from numbapro import CU
import numpy as np

def sum(tid, a, b, result):
    result[tid] = a[tid] + b[tid]

def execute_sum_on_gpu_via_CU(a, b):
    assert a.shape == b.shape
    cu = CU(target='gpu')
    result = np.zeros_like(a)
    d_result = cu.output(result)
    cu.enqueue(sum, ntid=result.size, args=(a, b, result))
    cu.wait()
    cu.close()
    return result

