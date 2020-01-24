import subprocess
from pathlib import Path
import os
from ray.experimental.serve import BackendConfig
import ray.experimental.serve as serve
import ray

from ensemble_profiler.constants import (SERVICE_STORE_ECG_DATA,
                                         MODEL_SERVICE_ECG_PREFIX,
                                         AGGREGATE_PREDICTIONS,
                                         BACKEND_PREFIX,
                                         ROUTE_ADDRESS)
from ensemble_profiler.store_data import StorePatientData
from ensemble_profiler.patient_prediction import PytorchPredictorECG
from ensemble_profiler.ensemble_predictions import Aggregate
from ensemble_profiler.ensemble_pipeline import EnsemblePipeline
from ensemble_profiler.server import HTTPActor
import time
import torch

package_directory = os.path.dirname(os.path.abspath(__file__))

total_data_request = 3750

def _create_services(model_list):
    all_services = []
    # create relevant services
    serve.create_endpoint(SERVICE_STORE_ECG_DATA)
    all_services.append(SERVICE_STORE_ECG_DATA)
    model_services = []
    for i in range(len(model_list)):
        model_service_name = MODEL_SERVICE_ECG_PREFIX + "::" + str(i)
        model_services.append(model_service_name)
        serve.create_endpoint(model_service_name)
    all_services += model_services
    serve.create_endpoint(AGGREGATE_PREDICTIONS)
    all_services.append(AGGREGATE_PREDICTIONS)

    # create backends
    num_queries_dict = {"ECG": total_data_request}
    b_config_store_data = BackendConfig(num_replicas=1, enable_predicate=True)
    serve.create_backend(
        StorePatientData, BACKEND_PREFIX+SERVICE_STORE_ECG_DATA, num_queries_dict,
        backend_config=b_config_store_data)
    for service, model in zip(model_services, model_list):
        b_config = BackendConfig(num_replicas=1, num_gpus=1)
        serve.create_backend(PytorchPredictorECG, BACKEND_PREFIX+service,
                             model, True, backend_config=b_config)
    serve.create_backend(Aggregate, BACKEND_PREFIX+AGGREGATE_PREDICTIONS)

    # link services to backends
    for service in all_services:
        serve.link(service, BACKEND_PREFIX+service)

    # get handles
    service_handles = {}
    for service in all_services:
        service_handles[service] = serve.get_handle(service)

    pipeline = EnsemblePipeline(model_services, service_handles)
    return pipeline


def calculate_throughput(model_list, num_queries=300):
    serve.init(blocking=True)
    pipeline = _create_services(model_list)
    future_list = []

    # dummy request
    info = {
        "patient_name": "adam",
        "value": 1.0,
        "vtype": "ECG"
    }
    start_time = time.time()
    for _ in range(num_queries):
        fut = pipeline.remote(info=info)
        future_list.append(fut)
    ray.get(future_list)
    end_time = time.time()
    serve.shutdown()
    return end_time - start_time, num_queries

def profile_ensemble(model_list, file_path, system_constraint):
    for constraint in system_constraint:
        print(constraint, '->', system_constraint[constraint])
    
    serve.init(blocking=True)
    if not os.path.exists(str(file_path.resolve())):
        file_path.touch()
    file_name = str(file_path.resolve())

    # create the pipeline
    pipeline = _create_services(model_list)

    # start the http server
    http_actor_handle = HTTPActor.remote(ROUTE_ADDRESS, pipeline, file_name)
    http_actor_handle.run.remote()
    # wait for http actor to get started
    time.sleep(2)
    warmup_gpu(service_handles, warmup = 200)
    print("start generating client")
    generate_dummy_client(system_constraint['npatient'])
    print("finish generating client and request")
    serve.shutdown()

def warmup_gpu(service_handles, warmup):
    print("warmup GPU")
    for handle_name in service_handles:
        if handle_name != SERVICE_STORE_ECG_DATA:
            for e in range(warmup):
                # print("warming up handle {} epoch {}".format(handle_name,e))
                ObjectID = serve.get_handle(handle_name).remote(
                    data=torch.zeros(1,1,total_data_request)
                )
                ray.get(ObjectID)
    print("finish warming up GPU by firing torch zero {} times".format(warmup))

def generate_dummy_client(npatient):
    # fire client
    client_path = os.path.join(package_directory, "patient_client.go")
    procs = []
    for patient_id in range(npatient):
        print(patient_id)
        ls_output = subprocess.Popen(["go", "run", client_path, "-nreq", str(total_data_request), "-patientId", str(patient_id)])
        procs.append(ls_output)
    for p in procs:
        p.wait()
        
