import mlrun
import time
import numpy as np
from PIL import Image
import argparse
from torchinfo import summary

config_parser = parser = argparse.ArgumentParser(description='Inference Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Model Inference')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')

parser.add_argument('--model', metavar='NAME', default='',
                    help='model type + name ("<type>/<name>")')

parser.add_argument('--model-dir', metavar='DIR', default='',
                    help='path to model (root dir)')




# group.add_argument('--train-split', metavar='NAME', default='train',
#                    help='dataset train split (default: train)')
# group.add_argument('--val-split', metavar='NAME', default='validation',
#                    help='dataset validation split (default: validation)')
# group.add_argument('--dataset-download', action='store_true', default=False,
#                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
# group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
#                    help='path to class to idx mapping file (default: "")')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


                   
def load_model_and_test_inference(model_path, test_image_path):
    # Load the model from the given MLRun path
    model = mlrun.load_model(model_path)
    
    # Load and preprocess the test image
    image = Image.open(test_image_path)
    image = image.resize((224, 224))  # Assuming the model expects 224x224 input
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Test inference time
    start_time = time.time()
    predictions = model.predict(image_array)
    end_time = time.time()
    
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")
    
    return predictions

# Example usage
# model_path = "path/to/your/model"
# test_image_path = "path/to/your/test/image.jpg"
# predictions = load_model_and_test_inference(model_path, test_image_path)

def main():
    args, args_text = _parse_args()
    model_stat = summary(model)


if __name__ = "__main__":
    main()