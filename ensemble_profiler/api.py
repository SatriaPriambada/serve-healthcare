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
def profile_ensemble(model_list, file_path, system_constraint):
    for constraint in system_constraint:
        print(constraint, '->', system_constraint[constraint])
    serve.init(blocking=True)
    if not os.path.exists(str(file_path.resolve())):
        file_path.touch()
    file_name = str(file_path.resolve())
    all_services = []
    # create relevant services
    serve.create_endpoint(SERVICE_STORE_ECG_DATA)
    all_services.append(SERVICE_STORE_ECG_DATA)
    model_services = []
    for i in range(len(model_list)):
        model_service_name =  MODEL_SERVICE_ECG_PREFIX + "::" + str(i)
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
        b_config = BackendConfig(num_replicas=1)
        serve.create_backend(PytorchPredictorECG, BACKEND_PREFIX+service,
                            model, False, backend_config=b_config)
    serve.create_backend(Aggregate, BACKEND_PREFIX+AGGREGATE_PREDICTIONS)
    
    # link services to backends
    for service in all_services:
        serve.link(service,BACKEND_PREFIX+service)
    
    # get handles
    service_handles = {}
    for service in all_services:
        service_handles[service] = serve.get_handle(service)
    
    pipeline = EnsemblePipeline(model_services, service_handles)
    # start the http server
    http_actor_handle = HTTPActor.remote(ROUTE_ADDRESS, pipeline, file_name)
    print("[Tio] start http server")
    http_actor_handle.run.remote()
    # wait for http actor to get started
    time.sleep(2)
    print("warmup GPU")
    warmup = 200
    for handle_name in service_handles:
        if handle_name != "ECGStoreData":
            for e in range(warmup):
                print("warming up handle {} epoch {}".format(handle_name,e))
                ObjectID = serve.get_handle(handle_name).remote(
                    data=torch.zeros(total_data_request)
                )

    print("start generating client")
    generate_dummy_client(system_constraint['npatient'])
    print("finish generating client and request")
    serve.shutdown()

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
        
