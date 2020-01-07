import os
import sys
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import numpy as np
import time

from resnet1d.resnet1d import ResNet1D

import ray.experimental.serve as serve
from store_data import StorePatientData
from patient_prediction import PytorchPredictorECG
from ray.experimental.serve import BackendConfig

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='test', help='start system with test config')
args = parser.parse_args()

def generate_bagging_system(list_of_models, system_constraint):
    #this function will be used to generate system based on Bagging strategy with specific system constraint 
    print('start generating system based on bagging strategy')
    print('list of models length: {}'.format(len(list_of_models)))
    #print('system_constraint: {}'.format(system_constraint))
    print('generating gpus for constraint: {}'.format(system_constraint["gpu"]))
    init_system(list_of_models, system_constraint["gpu"])

def init_system(list_of_models, gpus):
    serve.init(blocking=True)

    # create data point service for hospital
    serve.create_endpoint("hospital", route="/hospital")
    
    for i,model in enumerate(list_of_models):
        p = Path("Resnet1d_ray_serve.jsonl")
        p.touch()
        os.environ["SERVE_PROFILE_PATH"] = str(p.resolve())
        print("generate list no: ECG".format(i))
        init_prediction_service("ECG", 
        PytorchPredictorECG, model)
        print('generating patient clients: {}'.format(system_constraint["npatient"]))
        generate_dummy_client(system_constraint["npatient"])


def init_prediction_service(service_name, backend_class, model):
    # create service
    serve.create_endpoint("{}".format(service_name))

    # create backend for service
    b_config = BackendConfig(num_replicas=1)
    serve.create_backend(backend_class, "Predict{}".format(service_name),
                        model, False, backend_config=b_config)
    # link service and backend
    serve.link("{}".format(service_name), "Predict{}".format(service_name))
    handle = serve.get_handle(service_name)

    # prepare args for StorePatientData backend.
    service_handles_dict = {service_name: handle}
    # do prediction after every 3750 queries.
    num_queries_dict = {service_name: 3750}
    # Always keep num_replicas as 1 as this is a stateful Backend
    # This backend will store all the patient's data and transfer
    # the prediction to respective Backend (ECG handle in this case)
    b_config_hospital = BackendConfig(num_replicas=1)
    store_data_name = "StoreData{}".format(service_name)
    serve.create_backend(StorePatientData, store_data_name,
                        service_handles_dict, num_queries_dict,
                        backend_config=b_config_hospital)
    serve.link("hospital", store_data_name)

def generate_dummy_client(npatient):
    # fire client
    procs = []
    for patient_id in range(npatient):
        ls_output = subprocess.Popen(["go", "run", "patient_client.go", "-patientId", str(patient_id)])
        procs.append(ls_output)
    for p in procs:
        p.wait()


if __name__ == '__main__':
    
    list_of_models = []
    system_constraint = {"gpu":2, "npatient":10}
    print("config: {}".format(args.config))
    if(args.config == 'test'):
        #Test model for ECG example data
        n_channel = 1
        base_filters = 64
        kernel_size = 16
        n_classes = 2
        n_block = 2
        pytorch_model_1 = ResNet1D(in_channels=n_channel,
                    base_filters=base_filters,
                    kernel_size=kernel_size,
                    stride=2,
                    n_block=n_block,
                    groups=base_filters,
                    n_classes=n_classes,
                    downsample_gap=max(n_block//8, 1),
                    increasefilter_gap=max(n_block//4, 1),
                    verbose=False)
        list_of_models.append(pytorch_model_1)
        pytorch_model_2 = ResNet1D(in_channels=n_channel,
                    base_filters=base_filters,
                    kernel_size=kernel_size,
                    stride=2,
                    n_block=n_block,
                    groups=base_filters,
                    n_classes=n_classes,
                    downsample_gap=max(n_block//8, 1),
                    increasefilter_gap=max(n_block//4, 1),
                    verbose=False)
        list_of_models.append(pytorch_model_2)
    else:
        print("no config parameter")

    generate_bagging_system(list_of_models, system_constraint) 
