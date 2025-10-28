#!/usr/bin/env python3
"""
轴承故障诊断 - 集成预测测试脚本
【【【已修改】】】: 仅支持 (spec, env) 双模态，移除了 time_stats

支持两种集成策略:
1. 同源5折集成 (best_trial的5个fold投票)
2. 异源5折集成 (全局Top-5最佳fold投票)
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from collections import Counter
from tqdm import tqdm
import re

# ==================== 配置参数 ====================
# 数据集路径
DATASET_PATH = "optimized_features_dataset.pkl"

# 优化结果路径（运行train_optimized_features.py后生成的目录）
# 【【【请修改为你的优化结果目录】】】
# 例如: /AUGMENTED_SpecEnv_FoldGAN_opt_20251028_152319
OPTIMIZATION_RESULT_DIR = "/AUGMENTED_SpecEnv_FoldGAN_opt_20251028_152319"
# 提示：运行 python quick_test_config.py 可以自动找到最新的优化结果目录

# 输出路径
OUTPUT_DIR = f"{OPTIMIZATION_RESULT_DIR}/test_predictions"


# ==================== 注意力机制模块（与训练脚本一致）====================
class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation Block for 1D"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 4), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class CBAM1D(nn.Module):
    """Convolutional Block Attention Module for 1D"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(x)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_att(sa_input)
        x = x * sa
        return x


class SelfAttention1D(nn.Module):
    """Self-Attention for 1D sequences"""

    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv1d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, L = x.size()
        Q = self.query(x).permute(0, 2, 1)
        K = self.key(x)
        V = self.value(x)
        attn = torch.bmm(Q, K)
        attn = F.softmax(attn / (K.size(1) ** 0.5), dim=-1)
        out = torch.bmm(V, attn.permute(0, 2, 1))
        out = self.gamma * out + x
        return out


class FlexibleBranch1D(nn.Module):
    """灵活的1D CNN分支"""

    def __init__(self, in_channels, hidden_dim, attention_type='none', dropout=0.3):
        super().__init__()
        self.attention_type = attention_type

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if attention_type == 'se':
            self.attention = SEBlock1D(hidden_dim)
        elif attention_type == 'cbam':
            self.attention = CBAM1D(hidden_dim)
        elif attention_type == 'self':
            self.attention = SelfAttention1D(hidden_dim)
        else:
            self.attention = None

        self.pool = nn.AdaptiveAvgPool1d(8)

    def forward(self, x):
        x = self.conv_layers(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.pool(x)
        return x


class MultiFeatureFusionModel(nn.Module):
    """
    多特征融合模型
    【【【已修改】】】: 移除了 time_stats 分支
    """

    def __init__(self, n_spec_channels, n_env_channels,
                 spec_hidden=32, env_hidden=32, num_classes=6,
                 spec_attention='none', env_attention='none',
                 fusion_method='concat', dropout=0.3):
        super().__init__()

        self.fusion_method = fusion_method

        self.spec_branch = FlexibleBranch1D(
            n_spec_channels, spec_hidden, spec_attention, dropout
        )
        self.env_branch = FlexibleBranch1D(
            n_env_channels, env_hidden, env_attention, dropout
        )

        # 【【【已修改】】】: 移除了 time_branch

        if fusion_method == 'concat':
            # 【【【已修改】】】: 融合维度不再包含 time_branch
            fusion_dim = spec_hidden * 8 + env_hidden * 8
        else:
            # 默认为 concat
            fusion_dim = spec_hidden * 8 + env_hidden * 8

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, spec, env, return_features=False):
        """【【【已修改】】】: 移除了 time_feat 输入"""
        spec_out = self.spec_branch(spec).flatten(1)
        env_out = self.env_branch(env).flatten(1)

        # 【【【已修改】】】: 移除了 time_out

        if self.fusion_method == 'concat':
            fused = torch.cat([spec_out, env_out], dim=1)
        else:
            fused = torch.cat([spec_out, env_out], dim=1)

        features = self.fusion(fused)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


# ==================== 加载模型集合 ====================
def load_ensemble_models(model_paths, config, device):
    """
    加载多个模型用于集成
    【【【已修改】】】: 移除 time_feat_dim
    """
    models = []

    # 从数据集中动态获取通道数（或在此处硬编码，如果确定的话）
    # 这里我们根据训练脚本的推断，硬编码为 7 和 5
    N_SPEC_CHANNELS = 7
    N_ENV_CHANNELS = 5

    for model_path in model_paths:
        # 创建模型
        model = MultiFeatureFusionModel(
            n_spec_channels=N_SPEC_CHANNELS,
            n_env_channels=N_ENV_CHANNELS,
            # 【【【已修改】】】: 移除了 time_feat_dim
            spec_hidden=config['spec_hidden'],
            env_hidden=config['env_hidden'],
            num_classes=6,  # 假设为6类别
            spec_attention=config['spec_attention'],
            env_attention=config['env_attention'],
            fusion_method=config['fusion_method'],
            dropout=config['dropout']
        ).to(device)

        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    return models


# ==================== 集成预测 ====================
def ensemble_predict_detailed(models, spec, env, device):
    """
    集成预测：多个模型投票（返回详细信息）
    【【【已修改】】】: 移除了 time_feat
    """
    all_predictions = []  # 每个模型的预测类别
    all_probabilities = []  # 每个模型的softmax概率

    with torch.no_grad():
        for model in models:
            # 【【【已修改】】】: 模型调用不再传入 time_feat
            outputs = model(spec, env)

            # 获取预测类别
            pred = torch.argmax(outputs, dim=1)
            all_predictions.append(pred.cpu().numpy())

            # 获取softmax概率
            probs = F.softmax(outputs, dim=1)
            all_probabilities.append(probs.cpu().numpy())

    # 转换为数组
    # all_predictions: (n_models, n_samples)
    # all_probabilities: (n_models, n_samples, n_classes)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # 对每个样本进行投票和统计
    n_samples = all_predictions.shape[1]
    final_predictions = []
    decision_details = []

    for i in range(n_samples):
        votes = all_predictions[:, i]  # 5个模型的预测
        probs = all_probabilities[:, i, :]  # (5, 6) 每个模型的6类概率

        # 统计投票
        vote_counts = Counter(votes)
        final_pred = vote_counts.most_common(1)[0][0]

        # 计算置信度（投票比例）
        n_models = len(votes)
        confidence = vote_counts[final_pred] / n_models

        # 平均概率（作为参考）
        avg_probs = probs.mean(axis=0)

        # 记录详细决策信息
        detail = {
            'sample_idx': i,
            'final_prediction': int(final_pred),
            'vote_counts': dict(vote_counts),
            'confidence': confidence,
            'individual_predictions': votes.tolist(),
            'avg_probabilities': avg_probs.tolist(),
            'max_avg_prob': float(avg_probs[final_pred])
        }

        decision_details.append(detail)
        final_predictions.append(final_pred)

    return np.array(final_predictions), decision_details


def ensemble_predict(models, spec, env, device):
    """
    集成预测：多个模型投票（简化版）
    【【【已修改】】】: 移除了 time_feat
    """
    final_preds, _ = ensemble_predict_detailed(models, spec, env, device)
    return final_preds


# ==================== 测试函数 ====================
def test_with_ensemble(ensemble_type, optimization_dir, dataset_path, output_txt_path):
    """
    使用集成模型测试
    ensemble_type: 'homogeneous' (同源5折) 或 'heterogeneous' (异源5折)
    【【【已修改】】】: 移除了 time_stats
    """
    print(f"\n{'=' * 60}")
    print(f"集成预测测试 - {ensemble_type}")
    print(f"{'=' * 60}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载数据集
    print(f"\n加载数据集: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # 提取测试集数据
    x_test = dataset['x_test']
    test_sample_names = dataset['test_sample_names']
    id_to_label = dataset['id_to_label']

    spec_test = x_test['spec']
    env_test = x_test['env']
    # 【【【已修改】】】: 移除了 time_test

    print(f"测试集样本数: {len(test_sample_names)}")
    # 【【【已修改】】】: 更新了打印信息
    print(f"特征维度: spec={spec_test.shape}, env={env_test.shape}")

    # 2. 根据集成类型加载模型
    if ensemble_type == 'homogeneous':
        # 同源5折集成：加载最佳试验的5个fold模型
        print(f"\n策略：同源5折集成（最佳试验的5个fold）")

        # 读取最佳试验信息
        with open(f"{optimization_dir}/best_trial_all_folds_info.json", 'r') as f:
            best_trial_info = json.load(f)

        trial_num = best_trial_info['trial_number']
        config = best_trial_info['config']

        print(f"  最佳试验: #{trial_num}")
        print(f"  平均准确率: {best_trial_info['mean_accuracy']:.2f}%")
        print(f"  标准差: {best_trial_info['std_accuracy']:.2f}%")

        # 构建模型路径列表
        model_paths = []
        print(f"\n  加载5个fold模型:")
        for fold_detail in best_trial_info['fold_details']:
            model_path = fold_detail['model_path']
            model_paths.append(model_path)
            print(f"    Fold {fold_detail['fold']}: {fold_detail['accuracy']:.2f}% - {os.path.basename(model_path)}")

    else:  # heterogeneous
        # 异源5折集成：加载全局Top-5最佳验证准确率的fold模型
        print(f"\n策略：异源5折集成（全局Top-5最佳验证准确率fold）")

        # 读取全局Top-5 fold信息
        with open(f"{optimization_dir}/top_k_best_folds_info.json", 'r') as f:
            top_folds_info = json.load(f)

        print(f"  全局Top-5最佳fold:")
        model_paths = []
        configs = []

        for fold_info in top_folds_info:
            trial_num = fold_info['trial_number']
            fold_num = fold_info['fold_number']
            acc = fold_info['accuracy']

            # 构建模型路径
            model_path = f"{optimization_dir}/top_fold{fold_info['rank']}_trial{trial_num}_fold{fold_num}_acc{acc:.2f}.pth"
            model_paths.append(model_path)

            # 读取对应的trial配置
            with open(f"{optimization_dir}/trial_{trial_num}_info.json", 'r') as f:
                trial_info = json.load(f)
                configs.append(trial_info['config'])

            print(
                f"    {fold_info['rank']}. {acc:.2f}% (Trial #{trial_num}, Fold {fold_num}) - {os.path.basename(model_path)}")

        # 异源集成需要用第一个fold的配置（假设配置大致相同，主要差异在超参数）
        # 实际上每个模型可能有不同配置，需要分别加载
        config = configs[0]  # 暂时用第一个配置，仅用于日志

    # 3. 加载模型集合
    print(f"\n加载模型...")
    # 假设通道数
    N_SPEC_CHANNELS = 7
    N_ENV_CHANNELS = 5

    if ensemble_type == 'homogeneous':
        # 同源：所有模型配置相同
        models = load_ensemble_models(model_paths, config, device)
    else:
        # 异源：每个模型配置可能不同，需要分别加载
        models = []
        for model_path, model_config in zip(model_paths, configs):
            model = MultiFeatureFusionModel(
                n_spec_channels=N_SPEC_CHANNELS,
                n_env_channels=N_ENV_CHANNELS,
                # 【【【已修改】】】: 移除了 time_feat_dim
                spec_hidden=model_config['spec_hidden'],
                env_hidden=model_config['env_hidden'],
                num_classes=6,  # 假设为6类别
                spec_attention=model_config['spec_attention'],
                env_attention=model_config['env_attention'],
                fusion_method=model_config['fusion_method'],
                dropout=model_config['dropout']
            ).to(device)

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)

    print(f"  成功加载 {len(models)} 个模型")

    # 4. 准备测试数据（转换为tensor）
    print(f"\n准备测试数据...")
    spec_test_tensor = torch.FloatTensor(spec_test).to(device)
    env_test_tensor = torch.FloatTensor(env_test).to(device)
    # 【【【已修改】】】: 移除了 time_test_tensor

    # 5. 集成预测（获取详细信息）
    print(f"\n开始集成预测（{len(models)}个模型投票）...")
    # 【【【已修改】】】: 更新了函数调用
    final_predictions, decision_details = ensemble_predict_detailed(
        models, spec_test_tensor, env_test_tensor, device
    )

    # 6. 生成精简版结果（两列）
    print(f"\n生成预测结果...")
    simple_results = ["测试集名称\t故障类型"]

    for sample_name, pred_id in zip(test_sample_names, final_predictions):
        pred_label = id_to_label[int(pred_id)]
        simple_results.append(f"{sample_name}\t{pred_label}")

    # 7. 生成详细版结果（包含决策过程）
    detailed_results = [
        "测试集名称\t最终预测\t置信度\t模型1预测\t模型2预测\t模型3预测\t模型4预测\t模型5预测\t投票分布\t平均概率"
    ]

    for sample_name, pred_id, detail in zip(test_sample_names, final_predictions, decision_details):
        pred_label = id_to_label[int(pred_id)]

        # 每个模型的预测
        individual_preds = [id_to_label[int(p)] for p in detail['individual_predictions']]

        # 投票分布（格式：类别:票数）
        vote_str = "; ".join([
            f"{id_to_label[int(k)]}:{v}票"
            for k, v in sorted(detail['vote_counts'].items(), key=lambda x: -x[1])
        ])

        # 平均概率（格式：类别:概率，只显示>10%的）
        avg_probs = detail['avg_probabilities']
        prob_strs = []
        for class_id, prob in enumerate(avg_probs):
            if prob > 0.1:  # 只显示概率>10%的类别
                prob_strs.append(f"{id_to_label[class_id]}:{prob:.1%}")
        prob_str = "; ".join(prob_strs)

        # 置信度（投票比例）
        confidence = f"{detail['confidence']:.1%}"

        # 拼接行
        line = (f"{sample_name}\t{pred_label}\t{confidence}\t"
                f"{individual_preds[0]}\t{individual_preds[1]}\t{individual_preds[2]}\t"
                f"{individual_preds[3]}\t{individual_preds[4]}\t{vote_str}\t{prob_str}")

        detailed_results.append(line)

    # 8. 保存结果（两个文件）
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存精简版
    with open(output_txt_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(simple_results))

    # 保存详细版（在文件名中加入_detailed）
    detailed_txt_path = output_txt_path.replace('.txt', '_detailed.txt')
    with open(detailed_txt_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(detailed_results))

    print(f"\n✅ 预测完成!")
    print(f"  测试样本数: {len(test_sample_names)}")
    print(f"  精简版结果: {output_txt_path}")
    print(f"  详细版结果: {detailed_txt_path}")

    # 9. 显示预测分布
    pred_counts = Counter(final_predictions)
    print(f"\n预测类别分布:")
    for label_id in sorted(pred_counts.keys()):
        label_name = id_to_label[int(label_id)]
        count = pred_counts[label_id]
        percentage = 100 * count / len(test_sample_names)
        print(f"  {label_name}: {count} 样本 ({percentage:.1f}%)")

    # 10. 置信度统计
    confidences = [detail['confidence'] for detail in decision_details]
    print(f"\n置信度统计:")
    print(
        f"  5/5一致（100%）: {sum(1 for c in confidences if c == 1.0)} 样本 ({100 * sum(1 for c in confidences if c == 1.0) / len(confidences):.1f}%)")
    print(
        f"  4/5一致（80%）: {sum(1 for c in confidences if c == 0.8)} 样本 ({100 * sum(1 for c in confidences if c == 0.8) / len(confidences):.1f}%)")
    print(
        f"  3/5一致（60%）: {sum(1 for c in confidences if c == 0.6)} 样本 ({100 * sum(1 for c in confidences if c == 0.6) / len(confidences):.1f}%)")
    print(f"  平均置信度: {np.mean(confidences):.1%}")

    # 11. 显示低置信度样本（可能需要人工检查）
    low_confidence_samples = [
        (test_sample_names[i], id_to_label[int(final_predictions[i])], detail)
        for i, detail in enumerate(decision_details)
        if detail['confidence'] <= 0.6
    ]

    if len(low_confidence_samples) > 0:
        print(f"\n⚠️ 低置信度样本（3/5投票）: {len(low_confidence_samples)} 个")
        if len(low_confidence_samples) <= 10:
            print(f"  {'样本名称':<15} {'预测':<20} {'投票分布':<40}")
            print(f"  {'-' * 80}")
            for name, pred, detail in low_confidence_samples:
                vote_str = ", ".join([f"{id_to_label[int(k)]}:{v}" for k, v in detail['vote_counts'].items()])
                print(f"  {name:<15} {pred:<20} {vote_str}")
        else:
            print(f"  （数量较多，详见详细版txt文件）")

    return simple_results, decision_details


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("轴承故障诊断 - 集成预测测试 (仅 Spec+Env)")
    print("=" * 60)

    # 检查路径
    if not os.path.exists(DATASET_PATH):
        print(f"\n错误: 数据集文件不存在")
        print(f"路径: {DATASET_PATH}")
        return

    if not os.path.exists(OPTIMIZATION_RESULT_DIR):
        print(f"\n错误: 优化结果目录不存在")
        print(f"路径: {OPTIMIZATION_RESULT_DIR}")
        print(f"请确保第 23 行的 OPTIMIZATION_RESULT_DIR 变量已修改为你的训练结果目录")
        return

    # ========== 策略1: 同源5折集成 ==========
    print(f"\n{'=' * 60}")
    print("策略1: 同源5折集成")
    print(f"{'=' * 60}")

    output_path_homo = f"{OUTPUT_DIR}/predictions_homogeneous_5fold.txt"
    results_homo, details_homo = test_with_ensemble(
        ensemble_type='homogeneous',
        optimization_dir=OPTIMIZATION_RESULT_DIR,
        dataset_path=DATASET_PATH,
        output_txt_path=output_path_homo
    )

    # ========== 策略2: 异源5折集成 ==========
    print(f"\n{'=' * 60}")
    print("策略2: 异源5折集成")
    print(f"{'=' * 60}")

    output_path_hetero = f"{OUTPUT_DIR}/predictions_heterogeneous_top5folds.txt"
    results_hetero, details_hetero = test_with_ensemble(
        ensemble_type='heterogeneous',
        optimization_dir=OPTIMIZATION_RESULT_DIR,
        dataset_path=DATASET_PATH,
        output_txt_path=output_path_hetero
    )

    # ========== 对比两种策略的预测差异 ==========
    print(f"\n{'=' * 60}")
    print("两种策略预测对比")
    print(f"{'=' * 60}")

    # 提取预测结果（跳过header）
    preds_homo = [line.split('\t')[1] for line in results_homo[1:]]
    preds_hetero = [line.split('\t')[1] for line in results_hetero[1:]]

    # 计算差异
    diff_count = sum(1 for p1, p2 in zip(preds_homo, preds_hetero) if p1 != p2)
    diff_percentage = 100 * diff_count / len(preds_homo)

    print(f"\n预测差异:")
    print(f"  总样本数: {len(preds_homo)}")
    print(f"  预测不同的样本数: {diff_count}")
    print(f"  差异率: {diff_percentage:.2f}%")
    print(f"  一致率: {100 - diff_percentage:.2f}%")

    # 显示预测不同的样本（附加置信度信息）
    if diff_count > 0 and diff_count <= 20:
        print(f"\n预测不同的样本（附置信度）:")
        print(f"  {'样本名称':<15} {'同源预测':<18} {'同源置信度':<12} {'异源预测':<18} {'异源置信度':<12}")
        print(f"  {'-' * 80}")

        for i, (name, p1, p2) in enumerate(zip(results_homo[1:], preds_homo, preds_hetero)):
            if p1 != p2:
                sample_name = name.split('\t')[0]
                conf_homo = details_homo[i]['confidence']
                conf_hetero = details_hetero[i]['confidence']
                print(f"  {sample_name:<15} {p1:<18} {conf_homo:.1%}         {p2:<18} {conf_hetero:.1%}")

    # 对比两种策略的置信度
    conf_homo_avg = np.mean([d['confidence'] for d in details_homo])
    conf_hetero_avg = np.mean([d['confidence'] for d in details_hetero])

    print(f"\n置信度对比:")
    print(f"  同源5折平均置信度: {conf_homo_avg:.1%}")
    print(f"  异源5折平均置信度: {conf_hetero_avg:.1%}")

    if conf_hetero_avg > conf_homo_avg:
        print(f"  → 异源5折置信度更高 (+{(conf_hetero_avg - conf_homo_avg) * 100:.1f}%)")

    print(f"\n{'=' * 60}")
    print("✅ 测试完成!")
    print(f"{'=' * 60}")
    print(f"\n结果文件:")
    print(f"  同源5折集成:")
    print(f"    - 精简版: predictions_homogeneous_5fold.txt")
    print(f"    - 详细版: predictions_homogeneous_5fold_detailed.txt")
    print(f"\n  异源5折集成:")
    print(f"    - 精简版: predictions_heterogeneous_top5folds.txt")
    print(f"    - 详细版: predictions_heterogeneous_top5folds_detailed.txt")

    print(f"\n推荐:")
    print(f"  - 提交结果: 使用精简版txt文件")
    print(f"  - 分析决策: 查看详细版txt文件")
    print(f"  - 如果一致率>95%: 两种策略都可靠，优先用异源")
    print(f"  - 如果一致率<95%: 优先用置信度更高的那个")

    return (results_homo, details_homo), (results_hetero, details_hetero)


if __name__ == "__main__":
    homo_data, hetero_data = main()