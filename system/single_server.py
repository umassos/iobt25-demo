import grpc
from concurrent import futures
from inference_pb2 import PredictResponse, PredictRequest, HeartbeatRequest, HeartbeatResponse
from inference_pb2_grpc import (
    EncoderServiceServicer,
    add_EncoderServiceServicer_to_server,
    HeadServiceStub,EncoderServiceStub
)
import onnxruntime as ort
import numpy as np
from run_onnx_utils import load_torch_encoder, load_torch_classifier, load_torch_original,load_torch_split
import argparse
import timeit
import time
from onnx2torch import convert
import torch
import sys
sys.path.insert(1, "3rdparty/pytorch-image-models")
from ensemble_efficient_net_b0 import (
    EnsembleEfficientNet,
    get_multiexit_efficientnet_b0,
)
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class InferenceService(EncoderServiceServicer):
    def __init__(self, model_name, encoder_num, head_server, split, original = False):
        if original: 
            self.original_sess = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).to("cuda")
        else: 
            self.model = EnsembleEfficientNet(num_classes=608, cut_point=5)
            # self.model.load_state_dict(torch.load(f"models/{model_name}/model_best.pth.tar", map_location="cuda"))
            self.model.to("cuda")
            self.model.eval()

            if encoder_num == 1:
                self.enc_sess = self.model.encoder1.encoder
                self.class_sess = self.model.encoder1.classifier
            elif encoder_num == 2:
                self.enc_sess = self.model.encoder2.encoder
                self.class_sess = self.model.encoder2.classifier
            else:
                raise ValueError(f"Encoder number {encoder_num} not supported")
            self.head_stub = HeadServiceStub(grpc.insecure_channel(head_server))
            self.encoder_num = encoder_num  

    def Predict(self, request, context):
        np_input = np.frombuffer(request.input, dtype=np.float32).reshape(
            request.shape
        )
        input_tensor = torch.from_numpy(np_input).to("cuda")
        service_time = 0.0
        start_time = timeit.default_timer()
        # enc1_output = self.enc_sess.run([f"enc{self.encoder_num}_output"], {"input": np_input})[0]
        with torch.no_grad():
            enc1_output = self.enc_sess(input_tensor)
            result = self.class_sess(enc1_output)
            service_time = timeit.default_timer() - start_time
        return PredictResponse(
            output=result.cpu().numpy().tobytes(), shape=list(result.shape), full_model=False, has_result=True,
            service_time=service_time)

    def PredictFull(self, request, context):
        np_input = np.frombuffer(request.input, dtype=np.float32).reshape(
            request.shape
        )
        start_time = timeit.default_timer()
        result = self.single_sess.run(["output"], {"input": np_input})[0]
        end_time = timeit.default_timer()
        return PredictResponse(
            output=result.tobytes(), shape=list(result.shape), full_model=True, has_result=True,
            service_time=end_time - start_time,
        )

    def PredictOriginal(self, request, context):
        np_input = np.frombuffer(request.input, dtype=np.float32).reshape(
            request.shape
        )
        input_tensor = torch.from_numpy(np_input).to("cuda")
        service_time = 0.0
        start_time = timeit.default_timer()
        with torch.no_grad():
            result = self.original_sess(input_tensor)
            service_time = timeit.default_timer() - start_time
        return PredictResponse(
            output=result.cpu().numpy().tobytes(), shape=list(result.shape), full_model=True, has_result=True,
            service_time=service_time,
        )

    def PredictForward(self, request, context):
        np_input = np.frombuffer(request.input, dtype=np.float32).reshape(
            request.shape
        )
        input_tensor = torch.from_numpy(np_input).to("cuda")
        service_time = 0.0
        start_time = timeit.default_timer()
        with torch.no_grad():
            enc1_output = self.enc_sess(input_tensor)
            service_time = timeit.default_timer() - start_time
        try:
            request = PredictRequest(
                request_id=request.request_id,
                input=enc1_output.cpu().numpy().tobytes(),
                shape=enc1_output.shape,
                enc_service_time=service_time,
                enc_send_time=time.time(),
            )
            response = self.head_stub.Predict(request)
            print(f"Response service time: {response.service_time}")
            response.service_time = response.service_time + service_time
            return response
        except Exception as e:
            print(f"Failure due to {e}")
            pass

        start_time = timeit.default_timer()
        with torch.no_grad():
            result = self.class_sess(enc1_output)
            service_time += timeit.default_timer() - start_time
        return PredictResponse(
            output=result.cpu().numpy().tobytes(), shape=list(result.shape), full_model=False, has_result=True,
            service_time=service_time
        )
        
    def PredictSplit(self, request, context):
        np_input = np.frombuffer(request.input, dtype=np.float32).reshape(
            request.shape
        )
        start_time = timeit.default_timer()
        output = self.split_sess.run([f"output"], {"input": np_input})[0]
        end_time = timeit.default_timer()
        if "C" in self.split:
            return PredictResponse(
                output=output.tobytes(), shape=list(output.shape), full_model=True, has_result=True,
                service_time=end_time - start_time
            )
        else:
            request = PredictRequest(
                request_id=request.request_id,
                input=output.tobytes(),
                shape=output.shape,
            )
            response = self.single_stub.PredictSplit(request)
            response.service_time = response.service_time + (end_time - start_time)
            return response

    def Heartbeat(self, request, context):
        return HeartbeatResponse()


def serve():
    parser = argparse.ArgumentParser(description="ONNX Single Location Inference")
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="EENetB0_1_FC",
        help="Model Name we follow the same scheme for now.",
    )
    parser.add_argument(
        "-n",
        "--encoder_num",
        type=int,
        default=1,
        help="Encoder Number to load.",
    )

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8180,
        help="Port number for the server.",
    )
    parser.add_argument(
        "-s",
        "--head-server",
        type=str,
        default="localhost:8180",
        help="Head server URL.",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="1-5",
        help="Split",
    )

    parser.add_argument(
        "--original",
        action="store_true",
        help="Use original model.",
    )
    
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    add_EncoderServiceServicer_to_server(
        InferenceService(
            model_name=args.model_name,
            encoder_num=args.encoder_num,
            head_server=args.head_server,
            split=args.split,
            original=args.original,
        ),
        server,
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"Server running on port {args.port}...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
