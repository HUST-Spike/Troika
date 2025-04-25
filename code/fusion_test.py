# fusion_test.py
import argparse
import copy
import json
import os
import sys
import time
import gc

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from utils import *
from parameters import parser, YML_PATH
from dataset import CompositionDataset
from model.model_factory import get_model
from logger_utils import setup_logger, log_section

# 使用原有的test.py中的Evaluator和threshold_with_feasibility函数
from test import Evaluator, threshold_with_feasibility, test
from model.dfsp import DFSP

cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"


def clear_gpu_memory():
    """清理GPU显存"""
    torch.cuda.empty_cache()
    gc.collect()


def load_dfsp_model(dfsp_config, config, attributes, classes, offset, logger):
    """
    加载DFSP模型
    Returns:
        DFSP模型对象
    """
    print(f"加载DFSP模型: {config.dfsp_model_path}")
    logger.info(f"加载DFSP模型: {config.dfsp_model_path}")
    
    # 添加DFSP模型路径到系统路径
    dfsp_dir = os.path.dirname(config.dfsp_model_path)
    if dfsp_dir not in sys.path:
        sys.path.append(dfsp_dir)
    
    # 创建DFSP模型
    dfsp_model = DFSP(dfsp_config, attributes=attributes, classes=classes, offset=offset).cuda()
    
    # 加载预训练权重
    dfsp_model.load_state_dict(torch.load(config.dfsp_model_path))
    
    # 将模型移至GPU
    dfsp_model = dfsp_model.to(device)
    dfsp_model.eval()
    
    return dfsp_model


def predict_logits_dfsp(model, dataset, config):
    model.eval()
    all_attr_gt, all_obj_gt, all_pair_gt = [], [], []
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                          for attr, obj in pairs_dataset]).cuda()
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False)
    all_logits = torch.Tensor()
    
    with torch.no_grad():
        for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="获取DFSP预测"
        ):
            batch_img = data[0].cuda()
            predict = model(batch_img, pairs)
            logits = predict[0]
            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
            logits = logits.cpu()
            all_logits = torch.cat([all_logits, logits], dim=0)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

    all_attr_gt = torch.cat(all_attr_gt).to("cpu")
    all_obj_gt = torch.cat(all_obj_gt).to("cpu")
    all_pair_gt = torch.cat(all_pair_gt).to("cpu")

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt


def predict_logits_troika(model, dataset, config):
    """
    获取Troika模型的预测结果
    """
    model.eval()
    all_attr_gt, all_obj_gt, all_pair_gt = [], [], []
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                          for attr, obj in pairs_dataset]).cuda()
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers)
    all_logits = torch.Tensor()
    loss = 0
    
    with torch.no_grad():
        # 如果使用text_first方式
        if hasattr(config, 'text_first') and config.text_first:
            text_feats = [[], [], []]
            num_text_batch = pairs.shape[0] // config.text_encoder_batch_size
            for i_text_batch in range(num_text_batch):
                cur_pair = pairs[i_text_batch*config.text_encoder_batch_size:(i_text_batch+1)*config.text_encoder_batch_size, :]
                cur_text_feats = model.encode_text_for_open(cur_pair)
                for i_item in range(len(text_feats)):
                    text_feats[i_item].append(cur_text_feats[i_item])
            if pairs.shape[0] % config.text_encoder_batch_size != 0:
                cur_pair = pairs[num_text_batch*config.text_encoder_batch_size:, :]
                cur_text_feats = model.encode_text_for_open(cur_pair)
                for i_item in range(len(text_feats)):
                    text_feats[i_item].append(cur_text_feats[i_item])
            for i_item in range(len(text_feats)):
                text_feats[i_item] = torch.cat(text_feats[i_item], dim=0)
                
            for idx, data in tqdm(
                enumerate(dataloader), total=len(dataloader), desc="获取Troika预测"
            ):
                predict = model.forward_for_open(data, text_feats)
                logits = model.logit_infer(predict, pairs)
                loss += model.loss_calu(predict, data).item()
                attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
                logits = logits.cpu()
                all_logits = torch.cat([all_logits, logits], dim=0)
                all_attr_gt.append(attr_truth)
                all_obj_gt.append(obj_truth)
                all_pair_gt.append(pair_truth)
        else:
            # 常规方式
            for idx, data in tqdm(
                enumerate(dataloader), total=len(dataloader), desc="获取Troika预测"
            ):
                predict = model(data, pairs)
                logits = model.logit_infer(predict, pairs)
                loss += model.loss_calu(predict, data).item()
                attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
                logits = logits.cpu()
                all_logits = torch.cat([all_logits, logits], dim=0)
                all_attr_gt.append(attr_truth)
                all_obj_gt.append(obj_truth)
                all_pair_gt.append(pair_truth)

    all_attr_gt = torch.cat(all_attr_gt).to("cpu")
    all_obj_gt = torch.cat(all_obj_gt).to("cpu")
    all_pair_gt = torch.cat(all_pair_gt).to("cpu")

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss / len(dataloader)


def fuse_predictions(dfsp_logits, troika_logits, weight):
    """
    融合DFSP和Troika的预测结果
    
    Args:
        dfsp_logits (torch.Tensor): DFSP的预测logits
        troika_logits (torch.Tensor): Troika的预测logits
        weight (float): Troika预测的权重 (0~1之间)
    
    Returns:
        torch.Tensor: 融合后的预测logits
    """
    # 确保两个预测的形状一致
    if dfsp_logits.shape != troika_logits.shape:
        raise ValueError(f"DFSP预测形状 {dfsp_logits.shape} 与Troika预测形状 {troika_logits.shape} 不匹配")
    
    # 对DFSP和Troika的预测进行加权融合
    # weight是Troika的权重，(1-weight)是DFSP的权重
    fused_logits = weight * troika_logits + (1 - weight) * dfsp_logits
    
    return fused_logits


def find_best_weight(val_dataset, evaluator, dfsp_logits, troika_logits, 
                     all_attr_gt, all_obj_gt, all_pair_gt, config, logger,
                     unseen_scores=None, best_th=None, step=0.01):
    """
    在验证集上寻找最佳的融合权重
    """
    # 生成0到1之间，步长为0.01的权重值
    weights = np.arange(0, 1.01, step)
    num_weights = len(weights)
    best_hm = 0.0
    best_weight = 0.0
    best_stats = None
    
    print(f"在验证集上寻找最佳融合权重 (以{step}为步长，共尝试 {num_weights} 个权重值)")
    logger.info(f"在验证集上寻找最佳融合权重 (以{step}为步长，共尝试 {num_weights} 个权重值)")
    
    logger.info(f"配置信息: 开放世界={config.open_world}, 应用阈值={best_th is not None}")

    for weight in weights:
        # 融合预测
        fused_logits = fuse_predictions(dfsp_logits, troika_logits, weight)
        
        # 如果是开放世界设置且有阈值，应用阈值
        if config.open_world and best_th is not None:
            fused_logits = threshold_with_feasibility(
                fused_logits, 
                val_dataset.seen_mask, 
                threshold=best_th, 
                feasiblity=unseen_scores
            )
        
        # 评估融合结果
        results = test(
            val_dataset,
            evaluator,
            fused_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
        
        hm = results['best_hm']

        logger.info(f"权重 {weight:.2f} 的详细评估结果:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # 更新最佳权重
        if hm > best_hm:
            best_hm = hm
            best_weight = weight
            best_stats = copy.deepcopy(results)
            print(f"新的最佳权重: {best_weight:.4f}, HM: {best_hm:.4f}, AUC: {best_stats['AUC']:.4f}")
            logger.info(f"新的最佳权重: {best_weight:.4f}, HM: {best_hm:.4f}, AUC: {best_stats['AUC']:.4f}")
    
    return best_weight, best_stats


def main():
    # 添加融合模型的参数
    parser.add_argument('--dfsp_model_path', type=str, required=True, help='DFSP模型路径')
    parser.add_argument('--Troika_model_path', type=str, required=True, help='Troika模型路径')
    parser.add_argument('--fusion_weight', type=float, default=None, help='融合权重 (0~1之间，表示Troika的权重)')
    parser.add_argument('--weight_step', type=float, default=0.01, help='寻找最佳权重时的步长')
    
    config = parser.parse_args()
    load_args(YML_PATH[config.dataset], config)
    
    # 设置日志记录器
    logger = setup_logger(config, mode="fusion_test")
    log_section("融合模型评估详情")
    logger.info(f"数据集: {config.dataset}")

    # 加载DFSP模型配置
    dfsp_parser = argparse.ArgumentParser(description='DFSP模型参数')
    dfsp_config = dfsp_parser.parse_args([])  # 创建一个空的命名空间对象    
    dfsp_config_path = os.path.join(DIR_PATH, f'config/DFSP/{config.dataset}.yml')
    if os.path.exists(dfsp_config_path):
        # 使用与主配置相同的加载函数加载DFSP配置
        load_args(dfsp_config_path, dfsp_config)
        logger.info(f"已加载DFSP专用配置: {dfsp_config_path}")
        print(f"已加载DFSP专用配置: {dfsp_config_path}")
    else:
        # 如果找不到DFSP配置，复制主配置的属性
        for key, value in vars(config).items():
            setattr(dfsp_config, key, value)
        logger.warning(f"未找到DFSP专用配置文件: {dfsp_config_path}，使用通用配置")
        print(f"未找到DFSP专用配置文件: {dfsp_config_path}，使用通用配置")

    
    # 检查DFSP模型路径
    if not os.path.exists(config.dfsp_model_path):
        raise FileNotFoundError(f"DFSP模型文件不存在: {config.dfsp_model_path}")
    # 检查Troika模型路径
    if not os.path.exists(config.Troika_model_path):
        raise FileNotFoundError(f"Troika模型文件不存在: {config.Troika_model_path}")
    
    dataset_path = config.dataset_path
    
    # 加载验证数据集
    print('加载验证数据集')
    logger.info('加载验证数据集')
    val_dataset = CompositionDataset(dataset_path,
                                    phase='val',
                                    split='compositional-split-natural',
                                    open_world=config.open_world)
    allattrs = val_dataset.attrs
    allobj = val_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    # 加载DFSP模型
    print('加载DFSP模型')
    logger.info('加载DFSP模型')
    dfsp_model = load_dfsp_model(dfsp_config, config, attributes, classes, offset, logger)
    
    # 获取DFSP在验证集上的预测
    print('获取DFSP在验证集上的预测')
    logger.info('获取DFSP在验证集上的预测')
    dfsp_val_logits, val_attr_gt, val_obj_gt, val_pair_gt = predict_logits_dfsp(
        dfsp_model, val_dataset, dfsp_config)
    
    # 释放DFSP模型占用的显存
    print('释放DFSP模型占用的显存')
    logger.info('释放DFSP模型占用的显存')
    dfsp_model.cpu()  # 先将模型移至CPU
    del dfsp_model    # 删除模型
    clear_gpu_memory()  # 清理显存

    # 加载Troika模型
    print('加载Troika模型')
    logger.info('加载Troika模型')
    
    troika_model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    troika_model.load_state_dict(torch.load(config.Troika_model_path))
    
    # 获取Troika在验证集上的预测
    print('获取Troika在验证集上的预测')
    logger.info('获取Troika在验证集上的预测')
    troika_val_logits, _, _, _, _ = predict_logits_troika(
        troika_model, val_dataset, config)
    
    # 释放Troika模型占用的显存
    print('释放Troika模型占用的显存')
    logger.info('释放Troika模型占用的显存')
    troika_model.cpu()
    del troika_model
    clear_gpu_memory()
    
    # 初始化评估器
    evaluator = Evaluator(val_dataset, model=None)
    
    # 处理开放世界设置下的阈值
    best_th = None
    unseen_scores = None
    
    if config.open_world:
        if config.threshold is None:
            # 加载可行性分数
            feasibility_path = os.path.join(DIR_PATH, f'data/feasibility_{config.dataset}.pt')
            unseen_scores = torch.load(feasibility_path, map_location='cpu')['feasibility']
            
            # 寻找最佳阈值 (使用Troika的预测)
            seen_mask = val_dataset.seen_mask.to('cpu')
            min_feasibility = (unseen_scores + seen_mask * 10.).min()
            max_feasibility = (unseen_scores - seen_mask * 10.).max()
            thresholds = np.linspace(min_feasibility, max_feasibility, num=config.threshold_trials)
            
            best_auc = 0.
            best_th = -10
            
            print('寻找最佳阈值...')
            logger.info('寻找最佳阈值...')
            
            for th in thresholds:
                temp_logits = threshold_with_feasibility(
                    troika_val_logits, val_dataset.seen_mask, threshold=th, feasiblity=unseen_scores)
                results = test(
                    val_dataset,
                    evaluator,
                    temp_logits,
                    val_attr_gt,
                    val_obj_gt,
                    val_pair_gt,
                    config
                )
                auc = results['AUC']
                if auc > best_auc:
                    best_auc = auc
                    best_th = th
                    print(f'新的最佳阈值: {best_th:.4f}, AUC: {best_auc:.4f}')
                    logger.info(f'新的最佳阈值: {best_th:.4f}, AUC: {best_auc:.4f}')

        else:
            best_th = config.threshold
            print(f'使用指定的阈值: {best_th}')
            logger.info(f'使用指定的阈值: {best_th}')
            
            # 加载可行性分数
            feasibility_path = os.path.join(DIR_PATH, f'data/feasibility_{config.dataset}.pt')
            unseen_scores = torch.load(feasibility_path, map_location='cpu')['feasibility']
    
    # 寻找最佳融合权重或使用指定的权重
    if config.fusion_weight is None:
        print('在验证集上寻找最佳融合权重')
        logger.info('在验证集上寻找最佳融合权重')
        
        best_weight, val_stats = find_best_weight(
            val_dataset, evaluator, dfsp_val_logits, troika_val_logits,
            val_attr_gt, val_obj_gt, val_pair_gt, config, logger,
            unseen_scores, best_th, config.weight_step
        )
        
        print(f'最佳融合权重: {best_weight:.4f}')
        logger.info(f'最佳融合权重: {best_weight:.4f}')
    else:
        best_weight = config.fusion_weight
        print(f'使用指定的融合权重: {best_weight}')
        logger.info(f'使用指定的融合权重: {best_weight}')
        
        # 使用指定权重评估验证集
        fused_val_logits = fuse_predictions(dfsp_val_logits, troika_val_logits, best_weight)
        
        if config.open_world and best_th is not None:
            fused_val_logits = threshold_with_feasibility(
                fused_val_logits, val_dataset.seen_mask, threshold=best_th, feasiblity=unseen_scores)
        
        val_stats = test(
            val_dataset,
            evaluator,
            fused_val_logits,
            val_attr_gt,
            val_obj_gt,
            val_pair_gt,
            config
        )
    
    # 输出验证集结果
    log_section("验证集融合评估结果")
    result = ""
    for key in val_stats:
        result = result + key + "  " + str(round(val_stats[key], 4)) + "| "
        logger.info(f"{key}: {round(val_stats[key], 4)}")
    print(result)
    
    # 加载测试数据集
    log_section("加载测试数据集")
    print('加载测试数据集')
    logger.info('加载测试数据集')
    test_dataset = CompositionDataset(dataset_path,
                                      phase='test',
                                      split='compositional-split-natural',
                                      open_world=config.open_world)
    # 在测试集上评估
    print('在测试集上评估融合模型')
    log_section('在测试集上评估融合模型')
    test_evaluator = Evaluator(test_dataset, model=None)


    # 获取DFSP在测试集上的预测
    print('获取DFSP在测试集上的预测')
    logger.info('获取DFSP在测试集上的预测')
    dfsp_model = load_dfsp_model(dfsp_config, config, attributes, classes, offset, logger)  # 重新加载DFSP模型
    dfsp_test_logits, test_attr_gt, test_obj_gt, test_pair_gt = predict_logits_dfsp(
        dfsp_model, test_dataset, dfsp_config)
    
    # 在测试集上评估DFSP模型
    log_section("DFSP模型在测试集上的评估结果")
    print('在测试集上评估DFSP模型')
    logger.info('在测试集上评估DFSP模型')

    # 如果是开放世界设置且有阈值，应用阈值到DFSP预测
    dfsp_test_logits_eval = dfsp_test_logits.clone()
    if config.open_world and best_th is not None:
        dfsp_test_logits_eval = threshold_with_feasibility(
            dfsp_test_logits_eval,
            test_dataset.seen_mask,
            threshold=best_th,
            feasiblity=unseen_scores)

    dfsp_test_stats = test(
        test_dataset,
        test_evaluator,
        dfsp_test_logits_eval,
        test_attr_gt,
        test_obj_gt,
        test_pair_gt,
        config
    )

    # 输出DFSP测试结果
    print("\nDFSP模型测试结果:")
    logger.info("DFSP模型测试结果:")
    for key, value in dfsp_test_stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
            logger.info(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
            logger.info(f"  {key}: {value}")

    # 释放DFSP模型占用的显存
    print('释放DFSP模型占用的显存')
    logger.info('释放DFSP模型占用的显存')
    dfsp_model.cpu()
    del dfsp_model
    clear_gpu_memory()

    troika_model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    troika_model.load_state_dict(torch.load(config.Troika_model_path))

    # 获取Troika在测试集上的预测
    print('获取Troika在测试集上的预测')
    logger.info('获取Troika在测试集上的预测')
    troika_test_logits, _, _, _, _ = predict_logits_troika(
        troika_model, test_dataset, config)
    
    # 在测试集上评估Troika模型
    log_section("Troika模型在测试集上的评估结果")
    print('在测试集上评估Troika模型')
    logger.info('在测试集上评估Troika模型')

    # 如果是开放世界设置且有阈值，应用阈值到Troika预测
    troika_test_logits_eval = troika_test_logits.clone()
    if config.open_world and best_th is not None:
        troika_test_logits_eval = threshold_with_feasibility(
            troika_test_logits_eval,
            test_dataset.seen_mask,
            threshold=best_th,
            feasiblity=unseen_scores)
    
    troika_test_stats = test(
        test_dataset,
        test_evaluator,
        troika_test_logits_eval,
        test_attr_gt,
        test_obj_gt,
        test_pair_gt,
        config
    )

    # 输出Troika测试结果
    print("\nTroika模型测试结果:")
    logger.info("Troika模型测试结果:")
    for key, value in troika_test_stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
            logger.info(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
            logger.info(f"  {key}: {value}")

    # 释放Troika模型占用的显存
    print('释放Troika模型占用的显存')
    logger.info('释放Troika模型占用的显存')
    troika_model.cpu()
    del troika_model
    clear_gpu_memory()
    
    # 使用最佳权重融合测试集预测
    print(f'使用权重 {best_weight:.4f} 融合测试集预测')
    logger.info(f'使用权重 {best_weight:.4f} 融合测试集预测')
    fused_test_logits = fuse_predictions(dfsp_test_logits, troika_test_logits, best_weight)
    
    
    if config.open_world and best_th is not None:
        print(f'使用阈值: {best_th}')
        logger.info(f'使用阈值: {best_th}')
        fused_test_logits = threshold_with_feasibility(
            fused_test_logits,
            test_dataset.seen_mask,
            threshold=best_th,
            feasiblity=unseen_scores)
    
    fusion_test_stats = test(
        test_dataset,
        test_evaluator,
        fused_test_logits,
        test_attr_gt,
        test_obj_gt,
        test_pair_gt,
        config
    )
    
    # 输出融合模型测试结果
    print(f"\n融合模型测试结果 (Troika权重: {best_weight:.4f}):")
    logger.info(f"融合模型测试结果 (Troika权重: {best_weight:.4f}):")
    for key, value in fusion_test_stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
            logger.info(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
            logger.info(f"  {key}: {value}")
    
    # 计算并展示提升百分比
    log_section("融合模型提升百分比")
    print("\n融合模型相对于基础模型的提升百分比:")
    logger.info("融合模型相对于基础模型的提升百分比:")

    metrics_to_compare = ['best_hm', 'best_seen', 'best_unseen', 'AUC']
    
    # 创建一个表格格式的输出
    header = f"{'指标':<15} | {'DFSP':<10} | {'Troika':<10} | {'融合':<10} | {'vs DFSP':<10} | {'vs Troika':<10}"
    separator = "-" * len(header)
    
    print(separator)
    print(header)
    print(separator)
    
    logger.info(separator)
    logger.info(header)
    logger.info(separator)
    
    for metric in metrics_to_compare:
        if metric in fusion_test_stats and metric in dfsp_test_stats and metric in troika_test_stats:
            dfsp_value = dfsp_test_stats[metric]
            troika_value = troika_test_stats[metric]
            fusion_value = fusion_test_stats[metric]
            
            # 计算提升百分比
            vs_dfsp = (fusion_value - dfsp_value) / dfsp_value * 100 if dfsp_value != 0 else float('inf')
            vs_troika = (fusion_value - troika_value) / troika_value * 100 if troika_value != 0 else float('inf')
            
            # 格式化输出
            row = f"{metric:<15} | {dfsp_value:<10.4f} | {troika_value:<10.4f} | {fusion_value:<10.4f} | {vs_dfsp:+<10.2f}% | {vs_troika:+<10.2f}%"
            print(row)
            logger.info(row)
    
    print(separator)
    logger.info(separator)
    
    # 输出总结
    best_base_model = "DFSP" if dfsp_test_stats['best_hm'] > troika_test_stats['best_hm'] else "Troika"
    best_base_hm = max(dfsp_test_stats['best_hm'], troika_test_stats['best_hm'])
    hm_improvement = (fusion_test_stats['best_hm'] - best_base_hm) / best_base_hm * 100
    
    summary = f"总结: 融合模型在最佳基础模型({best_base_model})上提升了HM指标 {hm_improvement:.2f}%"
    print(f"\n{summary}")
    logger.info(f"\n{summary}")
    
    # 汇总所有结果
    results = {
        'val': val_stats,
        'test': {
            'dfsp': dfsp_test_stats,
            'troika': troika_test_stats,
            'fusion': fusion_test_stats
        },
        'fusion_weight': float(best_weight),
        'improvement': {
            'vs_dfsp': {metric: (fusion_test_stats[metric] - dfsp_test_stats[metric]) / dfsp_test_stats[metric] * 100 
                       for metric in metrics_to_compare if metric in fusion_test_stats and metric in dfsp_test_stats and dfsp_test_stats[metric] != 0},
            'vs_troika': {metric: (fusion_test_stats[metric] - troika_test_stats[metric]) / troika_test_stats[metric] * 100 
                         for metric in metrics_to_compare if metric in fusion_test_stats and metric in troika_test_stats and troika_test_stats[metric] != 0}
        }
    }
    
    if best_th is not None:
        results['best_threshold'] = float(best_th)
    
    # 保存详细结果到JSON文件
    results_path = os.path.join(config.exp_dir, f'fusion_results_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"详细结果已保存到: {results_path}")
    logger.info(f"详细结果已保存到: {results_path}")
    
    print("评估完成!")
    log_section("评估完成")


if __name__ == "__main__":
    main()