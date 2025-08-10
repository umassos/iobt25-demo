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
from run_onnx_utils import load_encoder, load_classifier, load_single, load_original,load_split
import argparse
import timeit

class InferenceService(EncoderServiceServicer):
    def __init__(self, model_name, encoder_num, head_server, split):
        self.single_sess = load_single(model_name=model_name)
        self.enc_sess = load_encoder(model_name=model_name, encoder_num=encoder_num)
        self.head_stub = HeadServiceStub(grpc.insecure_channel(head_server))
        self.single_stub = EncoderServiceStub(grpc.insecure_channel(head_server))

        self.class_sess = load_classifier(
            model_name=model_name, classifier_num=encoder_num
        )
        self.original_sess = load_original(model_name=model_name)
        self.encoder_num = encoder_num
        
        # self.split_sess = load_split(model_name=model_name, split=split)
        self.split = split

    def Predict(self, request, context):
        start_time = timeit.default_timer()
        np_input = np.frombuffer(request.input, dtype=np.float32).reshape(
            request.shape
        )
        enc1_output = self.enc_sess.run([f"enc{self.encoder_num}_output"], {"input": np_input})[0]
        result = self.class_sess.run([f"cl{self.encoder_num}_output"], {f"enc{self.encoder_num}_output": enc1_output})[0]
        end_time = timeit.default_timer()
        return PredictResponse(
            output=result.tobytes(), shape=list(result.shape), full_model=False, has_result=True,
            service_time=end_time - start_time,
        )

    def PredictFull(self, request, context):
        start_time = timeit.default_timer()
        np_input = np.frombuffer(request.input, dtype=np.float32).reshape(
            request.shape
        )
        result = self.single_sess.run(["output"], {"input": np_input})[0]
        end_time = timeit.default_timer()
        return PredictResponse(
            output=result.tobytes(), shape=list(result.shape), full_model=True, has_result=True,
            service_time=end_time - start_time,
        )

    def PredictOriginal(self, request, context):
        start_time = timeit.default_timer()
        np_input = np.frombuffer(request.input, dtype=np.float32).reshape(
            request.shape
        )
        result = self.original_sess.run(["output"], {"input": np_input})[0]
        end_time = timeit.default_timer()
        return PredictResponse(
            output=result.tobytes(), shape=list(result.shape), full_model=True, has_result=True,
            service_time=end_time - start_time,
        )

    def PredictForward(self, request, context):
        start_time = timeit.default_timer()
        np_input = np.frombuffer(request.input, dtype=np.float32).reshape(
            request.shape
        )
        enc1_output = self.enc_sess.run([f"enc{self.encoder_num}_output"], {"input": np_input})[0]
        try:
            request = PredictRequest(
                request_id=request.request_id,
                input=enc1_output.tobytes(),
                shape=enc1_output.shape,
            )
            response = self.head_stub.Predict(request)
            end_time = timeit.default_timer()


            ## TODO: Add service time to the response
            return response
        except Exception as e:
            print(f"Failure due to {e}")
            pass

        result = self.class_sess.run([f"cl{self.encoder_num}_output"], {f"enc{self.encoder_num}_output": enc1_output})[0]
        return PredictResponse(
            output=result.tobytes(), shape=list(result.shape), full_model=False, has_result=True,
        )
        
    def PredictSplit(self, request, context):
        np_input = np.frombuffer(request.input, dtype=np.float32).reshape(
            request.shape
        )
        output = self.split_sess.run([f"output"], {"input": np_input})[0]
        if "C" in self.split:
            return PredictResponse(
                output=output.tobytes(), shape=list(output.shape), full_model=True, has_result=True,
            )
        else:
            request = PredictRequest(
                request_id=request.request_id,
                input=output.tobytes(),
                shape=output.shape,
            )
            response = self.single_stub.PredictSplit(request)
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
    
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    add_EncoderServiceServicer_to_server(
        InferenceService(
            model_name=args.model_name,
            encoder_num=args.encoder_num,
            head_server=args.head_server,
            split=args.split,
        ),
        server,
    )
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"Server running on port {args.port}...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
