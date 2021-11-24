import time
from functools import total_ordering
import torch
from actnn.ops import op_quantize
from actnn.ops import op_dequantize


class Controller:
    '''
    default_bit: the number of bits used to quantize
    swap: if turned on, swap activation memory to CPU
    prefetch: if turned on, activation of the previous layer will be prefetched. the parameter is meaningful only when swap is True
    debug: if turned on, the same tensor that is saved twice will be quantized twice, which introduces memory overhead
    verbose: print debug log
    '''

    def __init__(self, default_bit=4, swap=False, debug=False, prefetch=False):
        self.unrelated_tensors = set()
        self.default_bit = default_bit
        self.debug = debug

        self.swap = swap
        self.swap_out_stream = torch.cuda.Stream()
        self.swap_in_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.current_stream()
        self.ptr_qtensor_map = {}
        self.prefetch = prefetch
        self.layer_key_map = {}
        self.tid = 0

    def filter_tensors(self, pairs):
        for k, v in pairs:
            self.unrelated_tensors.add(v.data_ptr())

    def check_quantize(self, input_tensor):
        if input_tensor.dtype != torch.float32:
            return False
        if input_tensor.requires_grad is False:
            return False
        if (len(input_tensor.shape) != 3) and (len(input_tensor.shape) != 4):
            return False
        if input_tensor.data_ptr() in self.unrelated_tensors:
            return False
        return True

    def iterate(self):
        del self.ptr_qtensor_map
        self.ptr_qtensor_map = {}
        self.tid = 0

    def quantize(self, input):
        if not self.check_quantize(input):
            return False, input

        if self.debug:
            ret = op_quantize(input, self.default_bit)
            return True, ret, input.shape

        tid = self.tid
        input_shape = input.shape
        key = (input.data_ptr(), input.sum().item(), input._version)
        self.layer_key_map[tid] = key
        self.tid += 1
        if key not in self.ptr_qtensor_map:
            # quantize
            q_inputs = op_quantize(input, self.default_bit)
            if self.swap:
                q_input_cpu = torch.empty(
                    q_inputs[0].shape, dtype=q_inputs[0].dtype, device='cpu', pin_memory=True)
                q_input_cpu.copy_(q_inputs[0], non_blocking=True)
                q_input_gpu = q_inputs[0]
                del q_input_gpu
                q_inputs[0] = q_input_cpu
            self.ptr_qtensor_map[key] = (q_inputs, 1)
        else:
            q_inputs, ref_cnt = self.ptr_qtensor_map[key]
            self.ptr_qtensor_map[key] = (q_inputs, ref_cnt + 1)
        return True, key, input_shape, tid

    def dequantize(self, input):
        quantized = input[0]
        if not quantized:
            return input[1]

        if self.debug:
            _, q_inputs, input_shape = input
            ret = op_dequantize(q_inputs, input_shape)
            return ret

        _, key, input_shape, tid = input
        q_inputs, ref_cnt = self.ptr_qtensor_map[key]

        # swap
        if not q_inputs[0].is_cuda:
            q_inputs[0] = q_inputs[0].cuda(non_blocking=True)
            self.ptr_qtensor_map[key] = (q_inputs, ref_cnt)

        # prefetch previous layer
        if self.prefetch and self.swap:
            with torch.cuda.stream(self.swap_in_stream):
                if tid > 0:
                    previous_key = self.layer_key_map[tid-1]
                    q_previous_inputs, pre_ref_cnt = self.ptr_qtensor_map[previous_key]
                    if not q_previous_inputs[0].is_cuda:
                        q_previous_inputs[0] = q_previous_inputs[0].cuda(
                            non_blocking=True)
                        self.ptr_qtensor_map[previous_key] = (
                            q_previous_inputs, pre_ref_cnt)

        ret = op_dequantize(q_inputs, input_shape)

        ref_cnt -= 1
        if ref_cnt == 0:
            del self.ptr_qtensor_map[key]
        else:
            self.ptr_qtensor_map[key] = (q_inputs, ref_cnt)
        return ret
