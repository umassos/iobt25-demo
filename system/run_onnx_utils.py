import onnx
import onnxruntime as rt
import logging

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s]:%(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)
#rt.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None)
EXECUTION_PROVIDER = "CUDAExecutionProvider"
#EXECUTION_PROVIDER = "CPUExecutionProvider"

def load_encoder(model_name, encoder_num):
    # Load the ONNX model
    encoder = onnx.load(f"models/{model_name}/encoder{encoder_num}.onnx")
    # Check the model
    onnx.checker.check_model(encoder)
    encoder_sess = rt.InferenceSession(
        encoder.SerializeToString(),
        providers=[EXECUTION_PROVIDER],
    )
    encoder_input = encoder_sess.get_inputs()[0]
    encoder_output = encoder_sess.get_outputs()[0]
    logger.info(
        f"nn{encoder_num}_encoder Input Name: {encoder_input.name} and Shape: {encoder_input.shape}"
    )
    logger.info(
        f"nn{encoder_num}_encoder Output Name: {encoder_output.name} and Shape: {encoder_output.shape}"
    )

    return encoder_sess


def load_classifier(model_name, classifier_num):
    # Load the ONNX model
    classifier = onnx.load(f"models/{model_name}/classifier{classifier_num}.onnx")
    # Check the model
    onnx.checker.check_model(classifier)

    classifier_sess = rt.InferenceSession(
        classifier.SerializeToString(),
        providers=[EXECUTION_PROVIDER],
    )
    classifier_input = classifier_sess.get_inputs()[0]
    classifier_output = classifier_sess.get_outputs()[0]
    logger.info(
        f"nn1_classifier Input Name: {classifier_input.name} and Shape: {classifier_input.shape}"
    )
    logger.info(
        f"nn1_classifier Output Name: {classifier_output.name} and Shape: {classifier_output.shape}"
    )

    return classifier_sess


def load_combined_head(model_name):
    combined_head = onnx.load(f"models/{model_name}/head.onnx")
    # Check the model
    onnx.checker.check_model(combined_head)

    combined_head_sess = rt.InferenceSession(
        combined_head.SerializeToString(),
        providers=[EXECUTION_PROVIDER],
    )
    head_input1 = combined_head_sess.get_inputs()[0]
    head_input2 = combined_head_sess.get_inputs()[1]
    classifier_output = combined_head_sess.get_outputs()[0]
    logger.info(
        f"Combined Head Input 1 Name: {head_input1.name} and Shape: {head_input1.shape}"
    )
    logger.info(
        f"Combined Head Input 2 Name: {head_input2.name} and Shape: {head_input2.shape}"
    )
    logger.info(
        f"Combined Head Output Name: {classifier_output.name} and Shape: {classifier_output.shape}"
    )

    return combined_head_sess

def load_single(model_name):
    single_model = onnx.load(f"models/{model_name}/single.onnx")
    # Check the model
    onnx.checker.check_model(single_model)

    single_sess = rt.InferenceSession(
        single_model.SerializeToString(),
        providers=[EXECUTION_PROVIDER],
    )
    input = single_sess.get_inputs()[0]
    classifier_output = single_sess.get_outputs()[0]
    logger.info(
        f"Single Model Input Name: {input.name} and Shape: {input.shape}"
    )
    logger.info(
        f"Single Model Input Name: {classifier_output.name} and Shape: {classifier_output.shape}"
    )

    return single_sess

def load_original(model_name):
    original_model = onnx.load(f"models/original.onnx")
    # Check the model
    onnx.checker.check_model(original_model)

    original_sess = rt.InferenceSession(
        original_model.SerializeToString(),
        providers=[EXECUTION_PROVIDER],
    )
    # original_sess = rt.InferenceSession(    
    input = original_sess.get_inputs()[0]
    classifier_output = original_sess.get_outputs()[0]
    logger.info(
        f"Original Model Input Name: {input.name} and Shape: {input.shape}"
    )
    logger.info(
        f"Original Model Input Name: {classifier_output.name} and Shape: {classifier_output.shape}"
    )

    return original_sess


def load_split(model_name, split):
    split_model = onnx.load(f"models/{model_name}/split_{split}.onnx")
    # Check the model
    onnx.checker.check_model(split_model)

    split_sess = rt.InferenceSession(
        split_model.SerializeToString(),
        providers=[EXECUTION_PROVIDER],
    )
    # original_sess = rt.InferenceSession(    
    input = split_sess.get_inputs()[0]
    split_output = split_sess.get_outputs()[0]
    logger.info(
        f"Split Model {model_name}_{split} Input Name: {input.name} and Shape: {input.shape}"
    )
    logger.info(
        f"Split Model {model_name}_{split} Input Name: {split_output.name} and Shape: {split_output.shape}"
    )

    return split_sess