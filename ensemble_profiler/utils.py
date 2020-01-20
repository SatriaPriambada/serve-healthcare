import subprocess
from pathlib import Path
import os
from ray.experimental.serve import BackendConfig
import ray.experimental.serve as serve
import ray

from ensemble_profiler.constants import (MODEL_SERVICE_ECG_PREFIX,
                                         AGGREGATE_PREDICTIONS,
                                         BACKEND_PREFIX,
                                         ROUTE_ADDRESS,
                                         PATIENT_NAME_PREFIX)
from ensemble_profiler.store_data_actor import StatefulPatientActor
from ensemble_profiler.patient_prediction import PytorchPredictorECG
from ensemble_profiler.ensemble_predictions import Aggregate
from ensemble_profiler.ensemble_pipeline import EnsemblePipeline
from ensemble_profiler.server import HTTPActor
import time
package_directory = os.path.dirname(os.path.abspath(__file__))


def create_services(model_list):
    all_services = []
    # create relevant services
    model_services = []
    for i in range(len(model_list)):
        model_service_name = MODEL_SERVICE_ECG_PREFIX + "::" + str(i)
        model_services.append(model_service_name)
        serve.create_endpoint(model_service_name)
    all_services += model_services
    serve.create_endpoint(AGGREGATE_PREDICTIONS)
    all_services.append(AGGREGATE_PREDICTIONS)

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


def start_patient_actors(num_patients, pipeline, periodic_interval=3750):
    # start actor for collecting patients_data
    actor_handles = {}
    for patient_id in range(num_patients):
        patient_name = PATIENT_NAME_PREFIX + str(patient_id)
        handle = StatefulPatientActor.remote(
            patient_name=patient_name,
            pipeline=pipeline,
            periodic_interval=periodic_interval
        )
        actor_handles[patient_name] = handle
    return actor_handles
