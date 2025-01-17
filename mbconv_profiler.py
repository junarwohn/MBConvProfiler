import tvm
import numpy as np
import os, json
from relay_modules import *

class MBProfiler():
    def __init__(self, 
            device, 
            target='cpu', 
            opt_level=3, 
            tuning_log=None,
            experiment=1,
            warmup=1, 
            repeat=12):
        from package.classes.device import Device, DevType 
        assert device in ['firefly', 'snapdragon', 'nano', 'xavier']
        assert target in ['cpu','mali','adreno','cuda','trt', 'dla']
        assert opt_level in [0,1,2,3]
        assert repeat > warmup
        
        self.device = device
        self.target = target
        self.opt_level = opt_level
        self.experiment = experiment
        self.warmup = warmup
        self.repeat = repeat

        dev_name = "NVIDIA Xavier AGX"
        dev_type_dla = DevType.NVIDIA_XAVIER_AGX_TRT_DLA
        self.dla_dev = None

        if self.target == 'cpu':
            self.tgt = tvm.target.create('llvm -mtriple=aarch64-linux-gnu')
            self.ctx = tvm.cpu()
        elif self.target == 'mali':
            self.tgt = tvm.target.create('opencl -device=mali')
            self.ctx = tvm.cl()
        elif self.target == 'adreno':
            self.tgt = tvm.target.create('opencl')
            self.ctx = tvm.cl()
        elif self.target in ['cuda', 'trt']:
            self.tgt = tvm.target.cuda()
            self.ctx = tvm.gpu()
        elif self.target in ['dla']:
            self.dla_dev = Device(dev_name, dev_type_dla, target_id=0, mode='local',
                url='192.168.0.122', port=9186, home_dir=f'/home/jd/',
                account='xavier', password='xavier')

            #self.tgt = tvm.target.cuda()
            #self.ctx = tvm.gpu()
            self.ctx = self.dla_dev.tvm_dev



        self.block_name = 'mbconv_tmp'

        self.tuning_log = tuning_log

    

    ###Execution Profiling
    ###
    def build_runtime_module(self, config_key: tuple, act='ReLU6'): #(in_s, k, s, in_c, exp_c, out_c), 'ReLU6'
        members = ['mod', 'params', 'module', 'exec_record', 'input_shape', 
                   'output_shape', 'h2d_record', 'd2h_record']
        self.clear_runtime_module(members)

        self.compute_workload(config_key)
        self.input_shape = (1, self.in_c, self.in_s, self.in_s)

        self.mod, self.params = get_mbconv(self.input_shape, self.k, self.s, 
                                             self.in_c, self.exp_c, self.out_c, act,
                                             self.block_name)
        if self.tuning_log != None:
            self.module = get_runtime_module_tuned(self.mod, self.params, self.tuning_log, self.opt_level, self.target)
        else:
            self.module = get_runtime_module(self.mod, self.params, self.opt_level, self.target, self.dla_dev)


    def build_mod_params(self, config_key: tuple, act='ReLU6'): #(in_s, k, s, in_c, exp_c, out_c), 'ReLU6'
        members = ['mod', 'params','input_shape']
        self.clear_runtime_module(members)

        self.compute_workload(config_key)
        self.input_shape = (1, self.in_c, self.in_s, self.in_s)

        self.mod, self.params = get_mbconv(self.input_shape, self.k, self.s, 
                                             self.in_c, self.exp_c, self.out_c, act,
                                             self.block_name)
    

    def profile_execution(self):
        self.exec_record = dict()
        records = []
        #if self.device == 'xavier':
        #    record_execution = dla_record_execution
        for i in range(self.experiment): 
            result = record_execution(self.module, self.input_shape, self.ctx, 
                                      self.repeat, self.warmup)
            records += result
        records.sort()
        self.exec_record['mean'] = np.mean(records)
        self.exec_record['median'] = np.median(records)
        self.exec_record['mins'] = np.median(records[:self.experiment])
    

    def do_exec(self, config_key: tuple, act='ReLU6'):
        self.build_runtime_module(config_key, act)
        #self.profile_execution()


    def clear_runtime_module(self, members):
        for member in members:
            if hasattr(self, member):
                del self.__dict__[member]


    def compute_workload(self, config_key: tuple):
        self.in_s = config_key[0]
        self.k = config_key[1]
        self.s = config_key[2]
        self.in_c = config_key[3]
        self.exp_c = config_key[4]
        self.out_c = config_key[5]
        self.out_s = self.in_s // self.s

        #compute size of params
        param_pw1 = self.in_c * self.exp_c
        param_dw = self.k**2 * self.exp_c
        param_pw2 = self.exp_c * self.out_c
        self.param_size = param_pw1 + param_dw + param_pw2

        #compute size of MAC(Multiply-Add)
        mac_pw1 = self.in_s**2 * self.in_c * self.exp_c
        self.exp_s = (self.in_s + 2*(self.k//2) - self.k)//self.s + 1
        mac_dw = self.exp_s**2 * self.k**2 * self.exp_c
        mac_pw2 = self.exp_s**2 * self.exp_c * self.out_c
        self.mac_size = mac_pw1 + mac_dw + mac_pw2
        
        print(" ")
        print(config_key)
        print("Params #: ", self.param_size)
        print("MACs: ", self.mac_size)
        print(" ")



    ###Other blocks(conv, linear) Profiling
    ###
    def build_runtime_module(self, config_key: tuple, act='ReLU6'): #(in_s, k, s, in_c, exp_c, out_c), 'ReLU6'
        members = ['mod', 'params', 'module', 'exec_record', 'input_shape', 
                   'output_shape', 'h2d_record', 'd2h_record']
        self.clear_runtime_module(members)

        self.compute_workload(config_key)
        self.input_shape = (1, self.in_c, self.in_s, self.in_s)

        self.mod, self.params = get_mbconv(self.input_shape, self.k, self.s, 
                                             self.in_c, self.exp_c, self.out_c, act,
                                             self.block_name)
        print(self.mod)
        return
        ################
        if self.tuning_log != None:
            self.module = get_runtime_module_tuned(self.mod, self.params, self.tuning_log, self.opt_level, self.target)
        else:
            self.module = get_runtime_module(self.mod, self.params, self.opt_level, self.target, self.dla_dev)

    def profile_execution(self):
        self.exec_record = dict()
        records = []
        for i in range(self.experiment): 
            result = record_execution(self.module, self.input_shape, self.ctx, 
                                      self.repeat, self.warmup)
            records += result
        records.sort()
        self.exec_record['mean'] = np.mean(records)
        self.exec_record['median'] = np.median(records)
        self.exec_record['mins'] = np.median(records[:self.experiment])
    

