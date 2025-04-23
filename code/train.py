#
import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model.model_factory import get_model
from parameters import parser, YML_PATH

# from test import *
import test as test
from dataset import CompositionDataset
from utils import *

import datetime
from logger_utils import setup_logger, log_section

def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset, logger):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    model.train()
    best_metric = 0
    best_loss = 1e5
    best_epoch = 0
    final_model_state = None
    
    val_results = []
    
    scheduler = get_scheduler(optimizer, config, len(train_dataloader))
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()
                                
    train_losses = []

    log_section("训练初始化")
    logger.info(f"开始训练，总共 {config.epochs} 轮")

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):

            predict = model(batch, train_pairs)

            loss = model.loss_calu(predict, batch)

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            loss.backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

        progress_bar.close()

        epoch_loss = np.mean(epoch_train_losses)
        progress_bar.write(f"epoch {i+1} train loss {epoch_loss}")
        logger.info(f"Epoch {i+1}/{config.epochs} - 训练损失: {epoch_loss:.6f}")
        train_losses.append(epoch_loss)

        if (i + 1) % config.save_every_n == 0:
            save_path = os.path.join(config.save_path, f"epoch_{i}.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"模型保存至: {save_path}")

        print("Evaluating val dataset:")
        log_section(f"Epoch {i+1} 验证集评估")
        logger.info("评估验证集...")
        val_result = evaluate(model, val_dataset, config, logger)
        val_results.append(val_result)
        print("Loss average on val dataset: {}".format(val_result['loss']))
        logger.info(f"验证集平均损失: {val_result['loss']:.6f}")

        if config.val_metric == 'best_loss':
            if val_result['loss'] < best_loss:
                best_loss = val_result['loss']
                logger.info(f"发现新的最佳模型 (损失: {best_loss:.6f})")
                best_epoch = i
                best_model_path = os.path.join(config.save_path, f"best.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"最佳模型保存至: {best_model_path}")

        else:
            if val_result[config.val_metric] > best_metric:
                best_metric = val_result[config.val_metric]
                best_epoch = i
                logger.info(f"发现新的最佳模型 ({config.val_metric}: {best_metric:.4f})")
                best_model_path = os.path.join(config.save_path, f"best.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"最佳模型保存至: {best_model_path}")

        final_model_state = model.state_dict()
        if i + 1 == config.epochs:
            print("--- Evaluating test dataset on Closed World ---")
            logger.info("评估封闭世界下的测试集")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "best.pt"
            )))
            evaluate(model, test_dataset, config, logger)


    final_model_path = os.path.join(config.save_path, f'final_model.pt')
    torch.save(final_model_state, final_model_path)
    logger.info(f"最终模型保存至: {final_model_path}")


def evaluate(model, dataset, config, logger):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
            model, dataset, config)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    test_saved_results = dict()
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
    print(result)
    logger.info(f"评估结果: {result}")  
    test_saved_results['loss'] = loss_avg
    return test_saved_results



if __name__ == "__main__":
    config = parser.parse_args()
    load_args(YML_PATH[config.dataset], config)
    print(config)

    # 模型保存路径
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    path_components = [
        config.dataset,
        config.clip_model,
        timestamp
    ]
    path_components = [comp for comp in path_components if comp]
    model_dir_name = "_".join(path_components)
    custom_save_dir = f"saved_models/{model_dir_name}"
    config.save_path = custom_save_dir

    # 声明logger
    logger = setup_logger(config, mode="train")
    logger.info("开始训练过程")
    logger.info(f"配置参数:\n{pprint.pformat(vars(config))}")

    # set the seed value
    set_seed(config.seed)
    logger.info(f"随机种子设置为: {config.seed}")

    dataset_path = config.dataset_path
    logger.info(f"数据集路径: {dataset_path}")

    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural',
                                       same_prim_sample=config.same_prim_sample)

    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural')

    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural')

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)
    logger.info(f"属性数量: {len(attributes)}，类别数量: {len(classes)}")

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    optimizer = get_optimizer(model, config)

    os.makedirs(config.save_path, exist_ok=True)
    logger.info(f"模型将保存到: {config.save_path}")

    train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset, logger)

    # 更新最近模型保存路径， 便于test读取
    os.makedirs("saved_models", exist_ok=True)
    with open(f"saved_models/{config.dataset}_latest_model.txt", "w") as f: 
        f.write(os.path.join(config.save_path, f"best.pt"))
        logger.info(f"已更新最新模型路径记录")

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    write_json(os.path.join(config.save_path, "config.json"), vars(config))
    print("done!")
    logger.info("训练完成！配置已保存。")
