import grpc
from concurrent import futures
from inference_pb2 import PredictResponse, PredictRequest, HeartbeatRequest, HeartbeatResponse
from inference_pb2_grpc import (
    HeadServiceServicer,
    add_HeadServiceServicer_to_server,
)
import onnxruntime as ort
import numpy as np
from run_onnx_utils import load_combined_head, load_torch_combined_head
import argparse
import timeit
import time
import torch
import sys
sys.path.insert(1, "3rdparty/pytorch-image-models")
from ensemble_efficient_net_b0 import (
    EnsembleEfficientNet,
    get_multiexit_efficientnet_b0,
)

class HeadInferenceService(HeadServiceServicer):
    def __init__(self, model_name):
        # self.head_sess = load_combined_head(model_name=model_name)
        # self.head_sess = load_torch_combined_head(model_name=model_name)
        self.model = EnsembleEfficientNet(num_classes=608, cut_point=5)
        # self.model.load_state_dict(torch.load(f"models/{model_name}/model_best.pth.tar", map_location="cuda"))
        self.model.to("cuda")
        self.model.eval()
        self.head_sess = self.model.classifier_comb

        self.requests = {}
        self.enc_network_time = {}
        self.enc_service_time = {}

    def Predict(self, request, context):
        recv_time = time.time()
        if self.requests.get(request.request_id) is None:
            self.requests[request.request_id] = request.input
            self.enc_network_time[request.request_id] = recv_time - request.enc_send_time
            self.enc_service_time[request.request_id] = request.enc_service_time
            return PredictResponse(
                output=b"",
                shape=[],
                full_model=True,
                has_result=False,
            )
        else:
            enc1_output = self.requests[request.request_id]
            enc1_output = np.frombuffer(enc1_output, dtype=np.float32).reshape(
                request.shape
            )
            enc1_output = torch.from_numpy(enc1_output).to("cuda")
            enc1_network_time = self.enc_network_time[request.request_id]
            enc1_service_time = self.enc_service_time[request.request_id]
            
            enc2_network_time = recv_time - request.enc_send_time
            enc2_service_time = request.enc_service_time

            del self.enc_network_time[request.request_id]
            del self.enc_service_time[request.request_id]
            del self.requests[request.request_id]
            enc2_output = np.frombuffer(request.input, dtype=np.float32).reshape(
                request.shape
            )
            enc2_output = torch.from_numpy(enc2_output).to("cuda")

        # Run inference (encoder then classifer)
        start_time = timeit.default_timer()

        with torch.no_grad():
            result = self.head_sess(enc1_output, enc2_output)
            service_time = timeit.default_timer() - start_time
        
        return PredictResponse(
            output=result.cpu().numpy().tobytes(),
            shape=list(result.shape),
            full_model=True,
            has_result=True,
            service_time=service_time + enc1_service_time + enc2_service_time,
            network_time=enc1_network_time + enc2_network_time,
        )

    def Heartbeat(self, request, context):
        return HeartbeatResponse()


def serve():
    parser = argparse.ArgumentParser(description="ONNX Single Location Inference")
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="EnsembleEfficientNet_C3",
        help="Model Name we follow the same scheme for now.",
    )

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8185,
        help="Port number for the server.",
    )
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_HeadServiceServicer_to_server(
        HeadInferenceService(
            model_name=args.model_name,
        ),
        server,
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"Server running on port {args.port}...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
