import os
import yaml
import argparse
from model import encoderModel, DualStreamDecoder, Lstmstream

from train import train_H
from dataset import get_train_dataloader, get_test_dataloader

parser = argparse.ArgumentParser(description='nc')

task_idx = 0
task_name = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM', 'REST']

parser.add_argument('--train_data_dir', type=str, default="/mnt/dataset1/anonymous/dataset/HCP/%s/train" % task_name[task_idx], help='data_dir')
parser.add_argument('--test_data_dir', type=str, default="/mnt/dataset1/anonymous/dataset/HCP/%s/test" % task_name[task_idx], help='data_dir')

parser.add_argument('--model_save_dir', type=str, default="/mnt/dataset1/anonymous/model/HCP/%s/full_model/r_10" % task_name[task_idx], help='model save dir')
parser.add_argument('--pre_model_path', type=str, default="/mnt/dataset1/anonymous/model/HCP/%s/pre_train_model/r_10" % task_name[task_idx], help='pre_model_path')
args = parser.parse_args()
print(args)

os.makedirs(args.model_save_dir, exist_ok=True)

project_root = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(project_root, "new1_HCP", "config", "base.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    keys = ["train", "encoder", "decoder", "pred_model"]
    output = "\n".join(f"{key}: {config.get(key, '')}" for key in keys)
    print(output)

save_config_path = os.path.join(args.model_save_dir, "config_used.yaml")
with open(save_config_path, "w") as f:
    yaml.dump(config, f)
print(f"config save to: {save_config_path}")

train_dataloader = get_train_dataloader(data_dir=args.train_data_dir, time_chunk_length = config['train']['time_split_len'], batch_size = config['train']['batch_size'])
test_dataloader = get_test_dataloader(data_dir=args.test_data_dir, time_chunk_length = config['train']['time_split_len'], batch_size = config['train']['batch_size'])

pre_model_path = os.path.join(args.pre_model_path, 'model_epoch_200.pt')

train_H(
    encoderModel=encoderModel,
    pred_model=Lstmstream,
    decoderModel=DualStreamDecoder,
    config=config,
    pre_model_path=pre_model_path,
    model_save_dir = args.model_save_dir,
    train_dataloader = train_dataloader,
    test_dataloader = test_dataloader,
)


