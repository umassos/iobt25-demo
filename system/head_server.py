import grpc
from concurrent import futures
from inference_pb2 import PredictResponse, PredictRequest
from inference_pb2_grpc import (
    HeadServiceServicer,
    add_HeadServiceServicer_to_server,
)
import onnxruntime as ort
import numpy as np
from run_onnx_utils import load_combined_head
import argparse


class HeadInferenceService(HeadServiceServicer):
    def __init__(self, model_name):
        self.head_sess = load_combined_head(model_name=model_name)
        self.requests = {}

    def Predict(self, request, context):
        if self.requests.get(request.request_id) is None:
            self.requests[request.request_id] = request.input
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
            del self.requests[request.request_id]
            enc2_output = np.frombuffer(request.input, dtype=np.float32).reshape(
                request.shape
            )

        # Run inference (encoder then classifer)
        result = self.head_sess.run(
            ["head_output"],
            {
                "enc1_output": enc1_output,
                "enc2_output": enc2_output,
            },
        )[0]
        return PredictResponse(
            output=result.tobytes(),
            shape=list(result.shape),
            full_model=True,
            has_result=True,
        )


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
