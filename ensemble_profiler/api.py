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
package_directory = os.path.dirname(os.path.abspath(__file__))
def profile_ensemble(model_list, filename):
    serve.init(blocking=True)
    os.environ["ENSEMBLE_PROFILE_PATH"] = filename
    
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
    num_queries_dict = {"ECG": 3750}
    b_config_store_data = BackendConfig(num_replicas=1, enable_predicate=True)
    serve.create_backend(
        StorePatientData, BACKEND_PREFIX+SERVICE_STORE_ECG_DATA, {"ECG": 3750},
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
    http_actor_handle = HTTPActor.remote(ROUTE_ADDRESS, pipeline)
    http_actor_handle.run.remote()
    # wait for http actor to get started
    time.sleep(2)

    # fire client
    client_path = os.path.join(package_directory, "patient_client.go")
    procs = []
    for _ in range(1):
        ls_output = subprocess.Popen(["go", "run", client_path])
        procs.append(ls_output)
    for p in procs:
        p.wait()
        
    serve.shutdown()
