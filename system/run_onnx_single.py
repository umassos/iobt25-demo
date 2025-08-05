import onnx
import onnxruntime as rt
import argparse
import numpy as np
import timeit
import logging
import pandas as pd
import socket
from run_onnx_utils import (
    load_classifier,
    load_encoder,
    load_combined_head,
    load_single,
    load_original,
)
from run_onnx_utils import EXECUTION_PROVIDER

# rt.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s]:%(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ONNX Single Location Inference")
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="EENetB0_2_FC",
        help="Model Name we follow the same scheme for now.",
    )
    parser.add_argument(
        "-s",
        "--scope",
        type=str,
        choices=["single", "forward", "full", "original"],
        default="single",
        help="How much of the model to run.",
    )

    parser.add_argument("save_results", help="Save Results", action="store_true")
    parser.add_argument(
        "-i", "--iterations", type=int, default=1000, help="Number of iterations to run"
    )

    args = parser.parse_args()
    logger.info(f"Running ONNX Inference for {args.model_name} with {args.scope}")

    input = np.random.random((1, 3, 224, 224))
    if "EDeepSp" in args.model_name:
        input = np.random.random((1, 1, 128, 31))
    input = input.astype(np.float32)
        
    if args.scope == "single":
        # Load Two Parts of the model
        logger.info("Running Encoder 1")
        enc_sess = load_encoder(model_name=args.model_name, encoder_num=1)
        class_sess = load_classifier(model_name=args.model_name, classifier_num=1)
        # Run inference (encoder then classifer)
        enc1_output = enc_sess.run(["enc1_output"], {"input": input})
        logger.info(f"Encoder 1 Output: {enc1_output[0].shape}")
        classification = class_sess.run(["cl1_output"], {"enc1_output": enc1_output[0]})
        logger.info(f"ClassificationOutput: {classification[0].shape}")

        results = []
        for i in range(args.iterations):
            start = timeit.default_timer()
            enc1_output = enc_sess.run(["enc1_output"], {"input": input})
            classification = class_sess.run(
                ["cl1_output"], {"enc1_output": enc1_output[0]}
            )
            end = timeit.default_timer()
            results.append(end - start)
        logger.info(f"Average Time: {np.mean(results)*1000:.2f} ms")
    elif args.scope == "forward":
        logger.info("Running Full Model as sub models")
        enc1_sess = load_encoder(model_name=args.model_name, encoder_num=1)
        enc2_sess = load_encoder(model_name=args.model_name, encoder_num=2)
        head_sess = load_combined_head(model_name=args.model_name)
        # Run inference (two encoders then classifer)
        enc1_output = enc1_sess.run(["enc1_output"], {"input": input})
        enc2_output = enc2_sess.run(["enc2_output"], {"input": input})
        logger.info(f"Encoder 1 Output: {enc1_output[0].shape}")
        logger.info(f"Encoder 2 Output: {enc2_output[0].shape}")
        classification = head_sess.run(
            ["head_output"],
            {"enc1_output": enc1_output[0], "enc2_output": enc2_output[0]},
        )
        logger.info(f"ClassificationOutput: {classification[0].shape}")
        results = []
        for i in range(args.iterations):
            start = timeit.default_timer()
            # Run inference (two encoders then classifer)
            enc1_output = enc1_sess.run(["enc1_output"], {"input": input})
            enc2_output = enc2_sess.run(["enc2_output"], {"input": input})
            classification = head_sess.run(
                ["head_output"],
                {"enc1_output": enc1_output[0], "enc2_output": enc2_output[0]},
            )
            end = timeit.default_timer()
            results.append(end - start)
        logger.info(f"Average Time: {np.mean(results)*1000:.2f} ms")
    elif args.scope == "full":
        logger.info("Running a full single model")
        single_sess = load_single(model_name=args.model_name)
        # Prepare the input
        # Run inference (two encoders then classifer)
        classification = single_sess.run(["output"], {"input": input})
        logger.info(f"ClassificationOutput: {classification[0].shape}")
        results = []
        for i in range(args.iterations):
            start = timeit.default_timer()
            # Run inference (two encoders then classifer)
            classification = single_sess.run(["output"], {"input": input})
            end = timeit.default_timer()
            results.append(end - start)
        logger.info(f"Average Time: {np.mean(results)*1000:.2f} ms")
    elif args.scope == "original":
        logger.info("Running Original Model")
        or_sess = load_original(model_name=args.model_name)
        # Run inference (two encoders then classifer)
        classification = or_sess.run(["output"], {"input": input})
        logger.info(f"ClassificationOutput: {classification[0].shape}")
        results = []
        for i in range(args.iterations):
            start = timeit.default_timer()
            # Run inference (two encoders then classifer)
            classification = or_sess.run(["output"], {"input": input})
            end = timeit.default_timer()
            results.append(end - start)
        logger.info(f"Average Time: {np.mean(results)*1000:.2f} ms")
    else:
        logger.error("Invalid Scope")
        return

    if args.save_results:
        logger.info("Saving Results")
        df = pd.DataFrame(results, columns=["Time(s)"])
        hostname = socket.gethostname()
        df.to_csv(
            f"system/local_results/{hostname.split('.')[0]}_{args.model_name}_{args.scope}_{EXECUTION_PROVIDER}.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
