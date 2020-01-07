import ray
from collections import defaultdict
from ray.experimental import serve
import torch


class StorePatientData:
    """
    A Ray Serve Backend class which stores the data of every patient.
    It also passes on the data if each patients ECG values gets filled
    upto 3750 values.

    Args:
        service_handles_dict(dict[value_type, Handles]): A dictionary of 
            different service handles.
        num_queries_dict(dict[value_type,int]): A dictionary of # of queries 
            to wait for a value type. 
    """

    def __init__(self, service_handles_dict, num_queries_dict,
                 supported_vtypes=["ECG"]):
        self.service_handles_dict = service_handles_dict
        self.num_queries_dict = num_queries_dict
        # store every patient data in a dictionary
        # patient_id -> { value_type: [values, ...]}
        self.patient_data = defaultdict(lambda: defaultdict(list))
        # value_type: ECG (supported right now), vitals etc.
        self.supported_vtypes = supported_vtypes

    def __call__(self, flask_request, info={}):
        result = ""
        # when client requests via web context
        if serve.context.web:
            patient_id = flask_request.args.get("patient_id")
            value = float(flask_request.args.get("value"))
            value_type = flask_request.args.get("vtype")
        else:
            # for profiling via kwargs
            patient_id = info["patient_id"]
            value = info["value"]
            value_type = info["vtype"]
        if value_type in self.supported_vtypes:
            # append the data point to the patient's stored data structure
            patient_val_list = self.patient_data[patient_id][value_type]
            patient_val_list.append(torch.tensor([[value]]))
            # check for prediction
            if (len(patient_val_list) ==
                    self.num_queries_dict[value_type]):
                # prepare data for prediction
                data = torch.cat(patient_val_list, dim=1)
                data = torch.stack([data])
                # submit prediction task
                ObjectID = self.service_handles_dict[value_type].remote(
                    data=data
                )
                # wait for prediction
                result = ray.get(ObjectID)
                # clear the data for next queries to be recorded
                patient_val_list.clear()
            else:
                result = "Data recorded"
        return result
