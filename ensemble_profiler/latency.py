from ray.experimental import serve
import os
import ray
from ensemble_profiler.utils import *
import time
import torch
from ensemble_profiler.server import HTTPActor
import subprocess
from ensemble_profiler.constants import ROUTE_ADDRESS
import time
from threading import Event

package_directory = os.path.dirname(os.path.abspath(__file__))


def profile_ensemble(model_list, file_path, num_patients=1,
                     http_host="0.0.0.0", fire_clients=True):
    if not ray.is_initialized():
        serve.init(blocking=True, http_port=5000)
        nursery_handle = start_nursery()
        if not os.path.exists(str(file_path.resolve())):
            file_path.touch()
        file_name = str(file_path.resolve())

        # create the pipeline
        pipeline, service_handles = create_services(model_list)

        # create patient handles
        actor_handles = start_patient_actors(num_patients=num_patients,
                                             nursery_handle=nursery_handle,
                                             pipeline=pipeline)

        # start the http server
        obj_id = nursery_handle.start_actor.remote(HTTPActor,
                                                   "HEALTH_HTTP_SERVER",
                                                   init_args=[ROUTE_ADDRESS,
                                                              actor_handles,
                                                              file_name])
        http_actor_handle = ray.get(obj_id)[0]
        http_actor_handle.run.remote(host=http_host, port=8000)
        # wait for http actor to get started
        time.sleep(2)
        warmup_gpu(service_handles, warmup = 200)
        generate_dummy_client(fire_clients, actor_handles)
        Event().wait()
        serve.shutdown()

def warmup_gpu(service_handles, warmup):
    print("warmup GPU")
    total_data_request = 3750
    for handle_name in service_handles:
        for e in range(warmup):
            # print("warming up handle {} epoch {}".format(handle_name,e))
            ObjectID = serve.get_handle(handle_name).remote(
                    data=torch.zeros(1,1,total_data_request)
            )
            ray.get(ObjectID)
    print("finish warming up GPU by firing torch zero {} times".format(warmup))

def generate_dummy_client(fire_clients, actor_handles):
    # fire client
    if fire_clients:
        client_path = os.path.join(package_directory, "patient_client.go")
        procs = []
        for patient_name in actor_handles.keys():
            ls_output = subprocess.Popen(
                ["go", "run", client_path, patient_name])
            procs.append(ls_output)
        for p in procs:
            p.wait()
