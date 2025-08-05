import numpy as np
import grpc
import zlib
import asyncio
import timeit
import logging
import argparse
import pandas as pd
import socket
from inference_pb2_grpc import EncoderServiceStub
from inference_pb2 import PredictRequest

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s]:%(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def remote_request(input, request_id, stub, function):
    try:
        #compressed_input = zlib.compress(input.tobytes())
        request = PredictRequest(
            request_id=request_id,
            input=input.tobytes(),
            shape=input.shape,
        )
        start = timeit.default_timer()
        if function == "Predict":
            response = stub.Predict(request)
        elif function == "PredictForward":
            response = stub.PredictForward(request)
        elif function == "PredictOriginal":
            response = stub.PredictOriginal(request)
        elif function == "PredictFull":
            response = stub.PredictFull(request)
        elif function == "PredictSplit":
            response = stub.PredictSplit(request)
        else:   
            raise ValueError("Invalid function specified")
        end = timeit.default_timer()
        total_time = end - start
        if request_id % 50 == 0:
            logger.info(
                f"Request ID: {request_id}, Time taken: {total_time:.4f} seconds, {response.full_model}-{response.shape}"
            )
        return total_time
    except Exception as e:
        print("Failure")
        return 0


async def main():
    parser = argparse.ArgumentParser(description="ONNX multi-server client.")
    parser.add_argument(
        "--server1",
        type=str,
        default="localhost:8180",
        help="Server 1 URL",
    )
    parser.add_argument(
        "--server2",
        type=str,
        default="localhost:8181",
        help="Server 2 URL",
    )
    parser.add_argument(
        "-i", "--requests", type=int, default=5, help="Number of requests to send"
    )
    
    parser.add_argument(
        "-m", "--model", type=str, default="", help="Model Name"
    )
    
    parser.add_argument(
        "-f",
        "--function",
        type=str,
        choices=["Predict", "PredictFull", "PredictForward", "PredictOriginal", "PredictSplit"],
        default="PredictOriginal",
        help="Function to call on the server",
    )

    args = parser.parse_args()
    logger.info("Starting ONNX multi-server client.")
    logger.info(f"Server 1 URL: {args.server1}")
    if args.function == "PredictForward":
        logger.info(f"Server 2 URL: {args.server2}")
    logger.info("Starting inference...")
    # Example usage
    input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    if "EDeepSp" in args.model:
        input = np.random.rand(1, 1, 128, 31).astype(np.float32)
    print(input.nbytes / 1024)

    results = []
    # Schedule both tasks to run concurrently
    channel1 = grpc.insecure_channel(args.server1)
    stub1 = EncoderServiceStub(channel1)

    channel2 = grpc.insecure_channel(args.server2)
    stub2 = EncoderServiceStub(channel2)

    for i in range(args.requests):
        gather_time = timeit.default_timer()
        if args.function == "PredictForward":
            times = await asyncio.gather(
                remote_request(
                    input=input, request_id=i, function=args.function, stub=stub1
                ),
                remote_request(
                    input=input, request_id=i, function=args.function, stub=stub2
                ),
            )
        else:
            time = await remote_request(
                    input=input, request_id=i, function=args.function, stub=stub1
                )
            times = [time]
        gather_time = timeit.default_timer() - gather_time
        # logger.info(f"All Gather time: {gather_time:.4f} seconds")
        results.append(times)
        # Run both requests concurrently
    if len(results[0]) == 1:
        df = pd.DataFrame(results, columns=["Time"])
    else:
        df = pd.DataFrame(results, columns=["Time1", "Time2"])
    print("Average Time taken for both requests:", np.mean(results, axis=0))
    hostname = socket.gethostname()
    df.to_csv(
        f"system/rpc_results/{hostname.split('.')[0]}_client_{args.function}_{args.model}_results.csv",
        index=False,
    )


if __name__ == "__main__":
    asyncio.run(main())
