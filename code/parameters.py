import argparse

parser = argparse.ArgumentParser()

YML_PATH = {
    "mit-states": './config/troika/mit-states.yml',
    "ut-zappos": './config/troika/ut-zappos.yml',
    "cgqa": './config/troika/cgqa.yml'
}

# model config
parser.add_argument("--extra_info", default=None, help="extra info to add to the log file")
parser.add_argument("--model_name", help="model name", type=str)
parser.add_argument("--lr", help="learning rate", type=float, default=5e-05)
parser.add_argument("--dataset", help="name of the dataset", type=str, default='mit-states')
parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-05)
parser.add_argument("--clip_model", help="clip model type", type=str, default="ViT-L/14")
parser.add_argument("--epochs", help="number of epochs", default=10, type=int)
parser.add_argument("--epoch_start", help="start epoch", default=0, type=int)
parser.add_argument("--train_batch_size", help="train batch size", default=32, type=int)
parser.add_argument("--eval_batch_size", help="eval batch size", default=16, type=int)
parser.add_argument("--num_workers", help="number of workers", default=10, type=int)
parser.add_argument("--context_length", help="sets the context length of the clip model", default=8, type=int)
parser.add_argument("--attr_dropout", help="add dropout to attributes", type=float, default=0.3)
parser.add_argument("--yml_path", help="yml path", type=str)
parser.add_argument("--clip_arch", help="clip path", default="clip_modules/ViT-L-14.pt", type=str)
parser.add_argument("--dataset_path", help="dataset path", type=str)
parser.add_argument("--save_path", help="save path", type=str)
parser.add_argument("--save_every_n", default=8, type=int, help="saves the model every n epochs")
parser.add_argument("--save_final_model", help="indicate if you want to save the model state dict()", action="store_true")
parser.add_argument("--load_model", default=None, help="load the trained model")
parser.add_argument("--seed", help="seed value", default=0, type=int)
parser.add_argument("--gradient_accumulation_steps", help="number of gradient accumulation steps", default=1, type=int)
parser.add_argument("--same_prim_sample", help="if sample same prim samples", action="store_true")

parser.add_argument("--disable_comp", type=bool, default=False, help="禁用组合")
parser.add_argument("--disable_attr", type=bool, default=False, help="禁用属性")
parser.add_argument("--disable_obj", type=bool, default=False, help="禁用对象")

parser.add_argument("--open_world", help="evaluate on open world setup", default=False)
parser.add_argument("--bias", help="eval bias", type=float, default=1e3)
parser.add_argument("--topk", help="eval topk", type=int, default=1)
parser.add_argument("--text_encoder_batch_size", help="batch size of the text encoder", default=16, type=int)
parser.add_argument('--threshold', type=float, default=None, help="optional threshold")
parser.add_argument('--threshold_trials', type=int, default=50, help="how many threshold values to try")

parser.add_argument("--adapter_dim", help="middle dimension of Adapter", type=int, default=64)
parser.add_argument("--init_lamda", help="lamda initialization value", type=float, default=0.1)
parser.add_argument("--cmt_layers", help="Number of layers in cross-attention", type=int, default=2)
