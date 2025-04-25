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


def clear_gpu_memory(model=None, model_name=None, logger=None):
    """清理GPU显存
    Args:
        model: 要清理的模型对象，如果为None则只清理通用显存
        model_name: 模型名称，用于日志记录
        logger: 日志记录器对象
    """
    if model is not None:
        if model_name and logger:
            message = f'释放{model_name}模型占用的显存'
            print(message)
            logger.info(message)
        # 先将模型移至CPU
        model.cpu()
        # 删除模型
        del model
    # 清理通用显存
    torch.cuda.empty_cache()
    gc.collect()


def load_dataset_and_prepare_metadata(config, phase, logger):
    """加载数据集并准备元数据
    Returns:
        dataset: 加载的数据集
        attributes: 属性列表
        classes: 类别列表
        offset: 属性数量
    """
    print(f'加载{phase}数据集')
    logger.info(f'加载{phase}数据集')
    
    dataset = CompositionDataset(config.dataset_path,
                                phase=phase,
                                split='compositional-split-natural',
                                open_world=config.open_world)
    
    allattrs = dataset.attrs
    allobj = dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)
    
    return dataset, attributes, classes, offset


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


def setup_open_world_threshold(config, val_dataset, val_logits, val_attr_gt, 
                              val_obj_gt, val_pair_gt, evaluator, logger):
    """设置开放世界模式的阈值和不可行性分数
    
    Args:
        config: 配置参数
        val_dataset: 验证数据集
        val_logits: 模型在验证集上的预测结果 通常使用Troika的预测 
        val_attr_gt, val_obj_gt, val_pair_gt: 验证集的ground truth
        evaluator: 评估器对象
        logger: 日志记录器
        
    Returns:
        tuple: (最佳阈值, 不可行性分数)
    """
    best_th = None
    unseen_scores = None
    
    if not config.open_world:
        return best_th, unseen_scores
    
    # 加载可行性分数
    feasibility_path = os.path.join(DIR_PATH, f'data/feasibility_{config.dataset}.pt')
    unseen_scores = torch.load(feasibility_path, map_location='cpu')['feasibility']
    
    # 如果已指定阈值，直接使用
    if config.threshold is not None:
        best_th = config.threshold
        print(f'使用指定的阈值: {best_th}')
        logger.info(f'使用指定的阈值: {best_th}')
        return best_th, unseen_scores
    
    # 寻找最佳阈值
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
            val_logits, val_dataset.seen_mask, threshold=th, feasiblity=unseen_scores)
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
    
    return best_th, unseen_scores


def evaluate_and_print_results(model_name, dataset, evaluator, logits, 
                              attr_gt, obj_gt, pair_gt, config, best_th=None, 
                              unseen_scores=None, logger=None):
    """评估模型性能并打印结果
    
    Args:
        model_name: 模型名称 (用于日志输出)
        dataset: 数据集对象
        evaluator: 评估器对象
        logits: 模型预测的logits
        attr_gt, obj_gt, pair_gt: 真实标签
        config: 配置对象
        best_th: 最佳阈值 (如果为None则不应用阈值)
        unseen_scores: 不可行性分数 (在开放世界设置中使用)
        logger: 日志记录器
        
    Returns:
        dict: 评估结果统计信息
    """
    # 记录评估开始
    log_section(f"{model_name}模型评估结果")
    print(f'评估{model_name}模型')
    if logger:
        logger.info(f'评估{model_name}模型')
    
    # 克隆logits以避免修改原始数据
    logits_eval = logits.clone()
    
    # 如果是开放世界设置且有阈值，应用阈值
    if config.open_world and best_th is not None:
        logits_eval = threshold_with_feasibility(
            logits_eval,
            dataset.seen_mask,
            threshold=best_th,
            feasiblity=unseen_scores)
    
    # 执行评估
    stats = test(
        dataset,
        evaluator,
        logits_eval,
        attr_gt,
        obj_gt,
        pair_gt,
        config
    )
    
    # 打印结果
    print(f"\n{model_name}模型评估结果:")
    if logger:
        logger.info(f"{model_name}模型评估结果:")
        
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
            if logger:
                logger.info(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
            if logger:
                logger.info(f"  {key}: {value}")
                
    return stats


def calculate_and_print_improvement(dfsp_stats, troika_stats, fusion_stats, best_weight, metrics_to_compare, logger):
    """计算并打印融合模型相对于基础模型的性能提升
    
    Args:
        dfsp_stats: DFSP模型的评估结果
        troika_stats: Troika模型的评估结果
        fusion_stats: 融合模型的评估结果
        best_weight: 最佳融合权重
        metrics_to_compare: 需要比较的指标列表
        logger: 日志记录器
        
    Returns:
        dict: 包含所有比较结果的字典
    """
    log_section("融合模型提升百分比")
    print("\n融合模型相对于基础模型的提升百分比:")
    logger.info("融合模型相对于基础模型的提升百分比:")

    # 创建一个表格格式的输出
    header = f"{'指标':<15} | {'DFSP':<10} | {'Troika':<10} | {'融合':<10} | {'vs DFSP':<10} | {'vs Troika':<10}"
    separator = "-" * len(header)
    
    print(separator)
    print(header)
    print(separator)
    
    logger.info(separator)
    logger.info(header)
    logger.info(separator)
    
    # 存储改进百分比
    improvements = {
        'vs_dfsp': {},
        'vs_troika': {}
    }
    
    for metric in metrics_to_compare:
        if metric in fusion_stats and metric in dfsp_stats and metric in troika_stats:
            dfsp_value = dfsp_stats[metric]
            troika_value = troika_stats[metric]
            fusion_value = fusion_stats[metric]
            
            # 计算提升百分比
            vs_dfsp = (fusion_value - dfsp_value) / dfsp_value * 100 if dfsp_value != 0 else float('inf')
            vs_troika = (fusion_value - troika_value) / troika_value * 100 if troika_value != 0 else float('inf')
            
            # 存储改进
            if dfsp_value != 0:
                improvements['vs_dfsp'][metric] = vs_dfsp
            if troika_value != 0:
                improvements['vs_troika'][metric] = vs_troika
            
            # 格式化输出
            row = f"{metric:<15} | {dfsp_value:<10.4f} | {troika_value:<10.4f} | {fusion_value:<10.4f} | {vs_dfsp:+<10.2f}% | {vs_troika:+<10.2f}%"
            print(row)
            logger.info(row)
    
    print(separator)
    logger.info(separator)
    
    # 输出总结
    best_base_model = "DFSP" if dfsp_stats['best_hm'] > troika_stats['best_hm'] else "Troika"
    best_base_hm = max(dfsp_stats['best_hm'], troika_stats['best_hm'])
    hm_improvement = (fusion_stats['best_hm'] - best_base_hm) / best_base_hm * 100
    
    summary = f"总结: 融合模型在最佳基础模型({best_base_model})上提升了HM指标 {hm_improvement:.2f}%"
    print(f"\n{summary}")
    logger.info(f"\n{summary}")
    
    # 构建结果字典
    results = {
        'test': {
            'dfsp': dfsp_stats,
            'troika': troika_stats,
            'fusion': fusion_stats
        },
        'fusion_weight': float(best_weight),
        'improvement': improvements,
        'best_base_model': best_base_model,
        'hm_improvement': hm_improvement
    }
    
    return results


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
    
    val_dataset, attributes, classes, offset = load_dataset_and_prepare_metadata(config, 'val', logger)

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
    clear_gpu_memory(dfsp_model, "DFSP", logger)

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
    clear_gpu_memory(troika_model, "Troika", logger)
    
    # 初始化评估器
    evaluator = Evaluator(val_dataset, model=None)
    
    # 处理开放世界设置下的阈值
    best_th, unseen_scores = setup_open_world_threshold(
        config, val_dataset, troika_val_logits, 
        val_attr_gt, val_obj_gt, val_pair_gt, 
        evaluator, logger
    )
    
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
    
    test_dataset, _, _, _ = load_dataset_and_prepare_metadata(config, 'test', logger)
    
    
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
    
    # 在测试集上评估DFSP模型并打印相关信息
    dfsp_test_stats = evaluate_and_print_results(
        "DFSP", test_dataset, test_evaluator, dfsp_test_logits,
        test_attr_gt, test_obj_gt, test_pair_gt, config,
        best_th, unseen_scores, logger
    )

    # 释放DFSP模型占用的显存
    clear_gpu_memory(dfsp_model, "DFSP", logger)

    troika_model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    troika_model.load_state_dict(torch.load(config.Troika_model_path))

    # 获取Troika在测试集上的预测
    print('获取Troika在测试集上的预测')
    logger.info('获取Troika在测试集上的预测')
    troika_test_logits, _, _, _, _ = predict_logits_troika(
        troika_model, test_dataset, config)
    
    # 在测试集上评估Troika模型并打印相关信息
    troika_test_stats = evaluate_and_print_results(
        "Troika", test_dataset, test_evaluator, troika_test_logits,
        test_attr_gt, test_obj_gt, test_pair_gt, config,
        best_th, unseen_scores, logger
    )

    # 释放Troika模型占用的显存
    clear_gpu_memory(troika_model, "Troika", logger)
    
    # 使用最佳权重融合测试集预测
    print(f'使用权重 {best_weight:.4f} 融合测试集预测')
    logger.info(f'使用权重 {best_weight:.4f} 融合测试集预测')
    fused_test_logits = fuse_predictions(dfsp_test_logits, troika_test_logits, best_weight)
    
    # 然后评估融合模型（注意这里我们自定义了模型名称以包含权重信息）
    fusion_test_stats = evaluate_and_print_results(
        f"融合模型 (Troika权重: {best_weight:.4f})", test_dataset, test_evaluator, fused_test_logits,
        test_attr_gt, test_obj_gt, test_pair_gt, config,
        best_th, unseen_scores, logger
    )
    
 
    # 计算并展示提升百分比
    metrics_to_compare = ['best_hm', 'best_seen', 'best_unseen', 'AUC']
    improvement_results = calculate_and_print_improvement(
        dfsp_test_stats, troika_test_stats, fusion_test_stats, 
        best_weight, metrics_to_compare, logger
    )

    # 构建完整结果
    results = {
        'val': val_stats,
        **improvement_results
    }
    
    if best_th is not None:
        results['best_threshold'] = float(best_th)
    
    # 保存详细结果到JSON文件
    fusion_log_dir = f'fusion_log/{config.dataset}{"_open_" if config.open_world else "_closed_"}'
    os.makedirs(fusion_log_dir, exist_ok=True)

    results_path = os.path.join(fusion_log_dir, f'{time.strftime("%m%d_%H%M")}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"详细结果已保存到: {results_path}")
    logger.info(f"详细结果已保存到: {results_path}")
    
    print("评估完成!")
    log_section("评估完成")


if __name__ == "__main__":
    main()