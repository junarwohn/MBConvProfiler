from enum import Enum
from multiprocessing import Process, Value
import numpy as np
import time
from typing import Any, Dict, Tuple, Optional

import sys
if sys.version_info >= (3, 8):
    import multiprocessing.shared_memory as shared_memory
else:
    import shared_memory

from package.classes.model import Model
from package.classes.profiler import InferenceResult
from package.util import MsgLevel, print_msg, get_random_data


class XavierType(Enum):
    AGX = 0
    NX = 1


class PEType(Enum):
    NONE = -1
    DLA0 = 0
    DLA1 = 1
    GPU = 2

    def is_gpu(self) -> bool:
        return 'GPU' in self.name

    def is_dla(self) -> bool:
        return 'DLA' in self.name


class XavierRunner():
    def __init__(self, type: XavierType, model_name: str,
                 model_size: Optional[int] = None) -> None:
        from package.classes.device import Device, DevType

        if type == XavierType.AGX:
            dev_name = "NVIDIA Xavier AGX"
            dev_type_gpu = DevType.NVIDIA_XAVIER_AGX_TRT_GPU
            dev_type_dla = DevType.NVIDIA_XAVIER_AGX_TRT_DLA
        elif type == XavierType.NX:
            dev_name = "NVIDIA Xavier NX"
            dev_type_gpu = DevType.NVIDIA_XAVIER_NX_TRT_GPU
            dev_type_dla = DevType.NVIDIA_XAVIER_NX_TRT_DLA

        # # PE definitions
        # self.dev_dla0 = Device(dev_name, dev_type_dla, target_id=0, mode='local',
        #     url='192.168.0.116', port=9186, home_dir=f'/home/xavier/sjlee/vact',
        #     account='xavier', password='xavier')
        # self.dev_dla1 = Device(dev_name, dev_type_dla, target_id=1, mode='local',
        #     url='192.168.0.116', port=9186, home_dir=f'/home/xavier/sjlee/vact',
        #     account='xavier', password='xavier')
        # self.dev_gpu = Device(dev_name, dev_type_gpu, target_id=0, mode='local',
        #     url='192.168.0.116', port=9186, home_dir=f'/home/xavier/sjlee/vact',
        #     account='xavier', password='xavier')

        # PE definitions
        self.dev_dla0 = Device(dev_name, dev_type_dla, target_id=0, mode='local',
            url='192.168.0.122', port=9186, home_dir=f'/home/jd/',
            account='xavier', password='xavier')
        self.dev_dla1 = Device(dev_name, dev_type_dla, target_id=1, mode='local',
            url='192.168.0.122', port=9186, home_dir=f'/home/jd/',
            account='xavier', password='xavier')
        self.dev_gpu = Device(dev_name, dev_type_gpu, target_id=0, mode='local',
            url='192.168.0.122', port=9186, home_dir=f'/home/jd/',
            account='xavier', password='xavier')

        # enable or disable devices
        self.devs: Dict[Device] = {
            PEType.GPU: self.dev_gpu,
            PEType.DLA0: self.dev_dla0,
            PEType.DLA1: self.dev_dla1
        }

        self.disabled_devs = set([
        ])

        for pe_type in self.disabled_devs:
            print_msg(f"{self.devs[pe_type].get_minimal_name().upper()} has been disabled")
            self.devs.pop(pe_type)
            self.models.pop(pe_type)

        self.enabled_gpus = 0
        self.enabled_dlas = 0
        for pe_type in self.devs.keys():
            if pe_type.is_dla():
                self.enabled_dlas += 1
            elif pe_type.is_gpu():
                self.enabled_gpus += 1

        # parameters for run
        self.batch_dla = 40
        self.unit_batch_dla = 8
        self.batch_gpu = 48
        self.unit_batch_gpu = 16

        self.target_batch = self.batch_gpu + self.enabled_dlas * self.batch_dla
        self.front_batch = self.unit_batch_dla
        self.back_batch = self.target_batch
        
        # model definition for each PE
        self.model_name = model_name
        self.model_size = model_size
        self.models = {
            PEType.GPU: CNN(self.model_name, self.model_size, self.unit_batch_gpu, sliced=0),
            PEType.DLA0: CNN(self.model_name, self.model_size, self.unit_batch_dla, sliced=0),
            PEType.DLA1: CNN(self.model_name, self.model_size, self.unit_batch_dla, sliced=0),
        }
        self.back_model = CNN(self.model_name, self.model_size, self.target_batch, sliced=1)

        # shape declarations
        self.front_fmap_shape = self.models[PEType.DLA0].front_fmap_shape
        self.back_fmap_shape = self.back_model.back_fmap_shape
        self.output_fmap_shape = self.back_model.output_fmap_shape
        self.gpu_input_shape = (self.unit_batch_gpu,) + self.front_fmap_shape
        self.gpu_output_shape = (self.unit_batch_gpu,) + self.back_fmap_shape
        self.dla_input_shape = (self.unit_batch_dla,) + self.front_fmap_shape
        self.dla_output_shape = (self.unit_batch_dla,) + self.back_fmap_shape
        self.back_input_shape = (self.back_batch,) + self.back_fmap_shape
        self.output_shape = (self.back_batch,)

        # shared values
        self.output_shms = {}
        self.input_status = {}
        self.output_status = {}
        self.common_input_shm = None
        self.common_output_shm = None
        self.common_input_offset = Value('i', 0)
        self.common_output_offset = Value('i', 0)
        self.finished_front_batches = Value('i', 0)
        self.finished_batches = Value('i', 0)
        self.back_output_shm = None
        self.barrier_status = Value('i', 0)
        self.barrier_threshold = len(self.devs) + 1
        self.dev_status = {}
        self.procs = {}

        # for logging
        self.elapsed_time = None
        self.data_overhead = None

        # initialization routines
        self.arrange_input()
        self.init_PEs()
    
    def arrange_input(self) -> None:
        whole_input = get_random_data((self.target_batch,) + self.front_fmap_shape, dtype=np.uint8)
        self.common_input_shm = shared_memory.SharedMemory(create=True, size=whole_input.nbytes)
        self.get_np_buffer(self.common_input_shm.buf, dtype=np.uint8,
                           input_data=whole_input, shape=whole_input.shape)

    def init_PEs(self) -> None:
        for pe_type, dev in self.devs.items():
            dev.id = pe_type
            dev.model = self.models[pe_type]
        #     dev.model.load()
        # self.back_model.load()

        back_input_arr = np.zeros(self.back_input_shape, dtype=np.float32)
        output_arr = np.zeros(self.output_shape, dtype=np.uint8)

        self.common_output_shm = shared_memory.SharedMemory(
                create=True, size=back_input_arr.nbytes)

        for pe_type, dev in self.devs.items():
            if pe_type.is_gpu():
                self.output_shms[pe_type] = shared_memory.SharedMemory(
                    create=True, size=output_arr.nbytes)
            else:
                self.output_shms[pe_type] = self.common_output_shm

            self.output_status[pe_type] = Value('i', 0)
            self.dev_status[pe_type] = Value('i', 0)

            proc = Process(target=self.PE_routine,
                           args=(dev, pe_type, self.models[pe_type], self.dev_status[pe_type]))
            self.procs[pe_type] = proc

    def setup_modules(self) -> None:
        # spawn processes
        for proc in self.procs.values():
            proc.start()

        # wait for warmup
        print_msg("Waiting for device initialization...", MsgLevel.INFO)
        self.wait_status(self.barrier_status, self.barrier_threshold)
        # self.change_status(self.barrier_status, 0)

    def wait_status(self, target, value = 0) -> None:
        while True:
            if target.value == value:
                return

    def change_status(self, target, value) -> None:
        with target.get_lock():
            target.value = value

    def get_byte_size(self, shape: Tuple[int], dtype = None) -> int:
        if shape is None:
            return 0
        if dtype is None:
            size = 1
        else:
            size = 32 if dtype == np.float32 else 8
        for dim in shape:
            size *= dim
        return size

    def get_np_buffer(self, buffer, dtype = np.float32, input_data = None, shape = None, offset = 0) -> Any:
        try:
            np_buffer = np.frombuffer(buffer=buffer, dtype=dtype, count=-1)

            if input_data is not None:
                size = self.get_byte_size(input_data.shape)
                np_buffer = np_buffer[offset:offset + size]
                np_buffer[:] = input_data.flatten()[:]
                if shape is not None:
                    np_buffer = np_buffer.reshape(shape)

            elif shape is not None:
                size = self.get_byte_size(shape)
                np_buffer = np_buffer[offset:offset + size].reshape(shape)

            else:
                print_msg("Source data or target shape should be given.", MsgLevel.ERROR)
                exit(0)

            return np_buffer

        except TypeError as e:
            print_msg(f"{e}", MsgLevel.ERROR)
            print_msg(f"Buffer size {len(buffer)} cannot match {input_data.nbytes}", MsgLevel.DEBUG)
            exit(0)
    
    def get_front_input_from_shm(self, shape: Tuple[int]) -> Any:
        batch_offset = 0
        with self.common_input_offset.get_lock():
            batch_offset = self.common_input_offset.value
            self.common_input_offset.value += shape[0]
        offset = self.get_byte_size((batch_offset,) + self.front_fmap_shape)

        input_buffer = self.get_np_buffer(self.common_input_shm.buf, np.uint8,
                                          shape=shape, offset=offset)
        return input_buffer
    
    def set_front_output_in_shm(self, output_data) -> None:
        batch_offset = 0
        shape = output_data.shape
        with self.common_output_offset.get_lock():
            batch_offset = self.common_output_offset.value
            self.common_output_offset.value += shape[0]
        offset = self.get_byte_size((batch_offset,) + self.back_fmap_shape)

        self.get_np_buffer(self.common_output_shm.buf, output_data.dtype,
                           input_data=output_data, shape=shape, offset=offset)
        return
    
    def run_full_model(self, dev, input_data = None) -> InferenceResult:
        infer_result = dev.exec_local(input_data=input_data)
        return infer_result

    def run_PEs(self) -> None:
        self.setup_modules()

        execute_loop = True

        # start inference
        print()
        print_msg("Start inference!", MsgLevel.INFO)
        elapsed_time = time.time()

        while execute_loop:
            # input data will be achieved by each device itself
            # just check the output data is available from the last device (GPU)
            if self.output_status[PEType.GPU].value == 1:
                output_data = self.get_np_buffer(self.output_shms[PEType.GPU].buf,
                                                 dtype=np.uint8, shape=self.output_shape)
                self.change_status(self.output_status[PEType.GPU], 0)
                if self.finished_batches.value >= self.target_batch:
                    execute_loop = False
                    break

        elapsed_time = (time.time() - elapsed_time) * 1000
        print()
        print_msg("Inference Complete!", MsgLevel.INFO)
        print_msg(f"Elapsed time: {elapsed_time:.3f} ms", MsgLevel.INFO)
        print_msg(f"Finished batches: {self.finished_batches.value}", MsgLevel.INFO)

        self.wrap_up_modules()
        print_msg("Parallel execution has been succeeded!")

        self.elapsed_time = elapsed_time

    def PE_routine(self, dev, pe_type: PEType, model: Model,
                   dev_status: Value) -> None:
        import numpy as np
        from package.util import get_random_data

        input_cnt = self.batch_gpu if pe_type.is_gpu() else self.batch_dla

        # test run
        # gpu should load both modules for front and back part of the model
        if pe_type.is_gpu():
            dev.model = self.back_model
            dev.run_setup(self.back_model)
            back_input_data = get_random_data(self.back_model.input_shape)
            self.run_full_model(dev, back_input_data)
            with self.barrier_status.get_lock():
                self.barrier_status.value += 1

        dev.model = model
        dev.run_setup(model)
        input_data = get_random_data(self.models[pe_type].input_shape)
        self.run_full_model(dev, input_data)
        with self.barrier_status.get_lock():
            self.barrier_status.value += 1
        self.wait_status(self.barrier_status, self.barrier_threshold)

        print_msg(f"Module has been loaded for {pe_type.name}!")

        # start inference
        while dev_status.value != -1:
            if input_cnt > 0:
                input_shape = self.gpu_input_shape if pe_type.is_gpu() \
                    else self.dla_input_shape
                batch_size = input_shape[0]
                input_data = self.get_front_input_from_shm(input_shape)
                input_cnt -= batch_size
                
                infer_result = self.run_full_model(dev, input_data)
                output_data = infer_result.output_data
                self.set_front_output_in_shm(output_data)
                
                with self.finished_front_batches.get_lock():
                    self.finished_front_batches.value += batch_size

            else:
                print_msg(f"{pe_type.name} has exited the loop!", MsgLevel.DEBUG)
                break

        if pe_type.is_gpu():
            self.wait_status(self.finished_front_batches, self.target_batch)

            input_data = self.get_np_buffer(self.common_output_shm.buf, dtype=np.float32,
                                            shape=self.back_input_shape)
            dev.model = self.back_model
            infer_result = self.run_full_model(dev, input_data)
            
            output_data = np.max(infer_result.output_data, axis=1)
            batch_size = len(output_data)

            self.get_np_buffer(self.output_shms[PEType.GPU].buf,
                               dtype=np.uint8, input_data=output_data)
            with self.finished_batches.get_lock():
                self.finished_batches.value += batch_size
            
            self.change_status(self.output_status[PEType.GPU], 1)
            
            self.data_overhead = infer_result.set_time
    
    def wrap_up_modules(self) -> None:
        for pe_type, dev in self.devs.items():
            self.change_status(self.dev_status[pe_type], -1)
        
        for pe_type, proc in self.procs.items():
            proc.join()
    
    def clean_shms(self) -> None:
        for shms in [self.output_shms,
                     self.common_input_shm, self.common_output_shm]:
            if shms is None:
                continue

            try:
                if isinstance(shms, dict):
                    for shm in shms.values():
                        if shm is not None:
                            shm.close()
                            shm.unlink()
                else:
                    shms.close()
                    shms.unlink()

            except FileNotFoundError as e:
                print_msg(e, MsgLevel.WARNING)
                continue


if __name__ == '__main__':
    import argparse
    from package.classes.model import CNN

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['mobilenet', 'resnet'])
    parser.add_argument('--size', type=int, required=False)

    args = parser.parse_args()
    model_name = args.model
    model_size = args.size

    repeat = 1
    elapsed_times = []
    data_times = []

    for _ in range(repeat):
        try:
            # runner = XavierRunner(XavierType.NX, model_name, model_size)
            runner = XavierRunner(XavierType.AGX, model_name, model_size)
            runner.run_PEs()
            
            elapsed_times.append(runner.elapsed_time)
            data_times.append(runner.data_overhead if runner.data_overhead is not None else 0)

        # except(e):
            # print(e)
        finally:
            runner.clean_shms()
            pass

    print()
    print_msg(f"Average elapsed time: {sum(elapsed_times)/repeat:.3f} ms")

    if sum(data_times) > 0:
        print_msg(f"Average data overhead time: {sum(data_times)/repeat:.3f} ms")
