#!/usr/bin/env python3
"""
对比实验脚本
同时运行原项目和修改后的项目，收集指标并生成对比报告
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from threading import Thread
from queue import Queue

# 项目路径
ORIGINAL_PROJECT = "/home/ubuntu/lzy/Neural-SLAM"
MODIFIED_PROJECT = "/home/ubuntu/lzy/ANS/Neural-SLAM"

class ExperimentRunner:
    def __init__(self, original_path, modified_path, output_dir="comparison_results"):
        self.original_path = Path(original_path).resolve()
        self.modified_path = Path(modified_path).resolve()
        # 使用修改项目的路径作为基准（因为脚本在修改项目中运行）
        base_path = self.modified_path
        self.output_dir = (base_path / output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = (self.output_dir / f"results_{self.timestamp}").resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[调试] 结果根目录: {self.results_dir}")
        print(f"[调试] 结果根目录存在: {self.results_dir.exists()}")
        
    def run_experiment(self, project_path, exp_name, args_list, timeout=3600):
        """运行单个实验"""
        print(f"\n{'='*60}")
        print(f"运行实验: {exp_name}")
        print(f"项目路径: {project_path}")
        print(f"参数: {' '.join(args_list)}")
        print(f"{'='*60}\n")
        
        # 使用绝对路径，避免工作目录切换导致的问题
        result_dir = (self.results_dir / exp_name).resolve()
        print(f"[调试] 准备创建结果目录: {result_dir}")
        print(f"[调试] 父目录存在: {result_dir.parent.exists()}")
        
        # 确保目录创建成功（线程安全）
        try:
            # 先确保父目录存在
            result_dir.parent.mkdir(parents=True, exist_ok=True)
            # 再创建结果目录
            result_dir.mkdir(parents=True, exist_ok=True)
            # 验证目录确实创建了
            if not result_dir.exists():
                raise OSError(f"无法创建结果目录: {result_dir}")
            print(f"  ✓ 结果目录已创建: {result_dir}")
            print(f"  ✓ 结果目录存在: {result_dir.exists()}")
        except Exception as e:
            print(f"  ✗ 创建结果目录失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 添加输出重定向（使用绝对路径）
        log_file = result_dir / "run.log"
        output_file = result_dir / "output.txt"
        
        # 确保目录存在
        log_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存当前工作目录
        original_cwd = os.getcwd()
        
        # 修改工作目录到项目目录
        project_path_abs = Path(project_path).resolve()
        os.chdir(str(project_path_abs))
        
        # 构建命令
        cmd = ["python", "main.py"] + args_list
        
        start_time = time.time()
        
        try:
            # 使用绝对路径打开文件（确保路径正确）
            log_file_str = str(log_file.resolve())
            output_file_str = str(output_file.resolve())
            
            # 再次确保目录存在（在打开文件前）
            Path(log_file_str).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file_str).parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file_str, 'w') as log, open(output_file_str, 'w') as out:
                process = subprocess.Popen(
                    cmd,
                    stdout=out,
                    stderr=log,
                    cwd=str(project_path_abs),
                    env=os.environ.copy()
                )
                
                # 等待完成或超时
                try:
                    return_code = process.wait(timeout=timeout)
                    elapsed_time = time.time() - start_time
                    
                    if return_code == 0:
                        print(f"✓ 实验完成 ({elapsed_time:.1f}秒)")
                    else:
                        print(f"✗ 实验失败，返回码: {return_code} ({elapsed_time:.1f}秒)")
                    
                    # 验证结果目录确实存在
                    result_dir_abs = result_dir.resolve()
                    print(f"[调试] 实验完成，检查结果目录: {result_dir_abs}")
                    print(f"[调试] 结果目录存在: {result_dir_abs.exists()}")
                    
                    if not result_dir_abs.exists():
                        print(f"  ⚠ 警告: 结果目录不存在，尝试重新创建: {result_dir_abs}")
                        try:
                            result_dir_abs.mkdir(parents=True, exist_ok=True)
                            print(f"  ✓ 已重新创建目录")
                        except Exception as e:
                            print(f"  ✗ 重新创建失败: {e}")
                    
                    # 再次验证
                    if not result_dir_abs.exists():
                        print(f"  ✗ 严重错误: 结果目录仍然不存在: {result_dir_abs}")
                        # 尝试使用备用路径
                        backup_dir = self.results_dir / exp_name
                        backup_dir.mkdir(parents=True, exist_ok=True)
                        print(f"  ✓ 已创建备用目录: {backup_dir}")
                        result_dir_abs = backup_dir.resolve()
                    
                    result_dict = {
                        'success': return_code == 0,
                        'elapsed_time': elapsed_time,
                        'return_code': return_code,
                        'result_dir': str(result_dir_abs)  # 使用绝对路径
                    }
                    print(f"  最终结果目录: {result_dict['result_dir']}")
                    print(f"  最终结果目录存在: {Path(result_dict['result_dir']).exists()}")
                    return result_dict
                except subprocess.TimeoutExpired:
                    process.kill()
                    print(f"✗ 实验超时 ({timeout}秒)")
                    return {
                        'success': False,
                        'elapsed_time': timeout,
                        'return_code': -1,
                        'result_dir': str(result_dir),
                        'error': 'timeout'
                    }
        except Exception as e:
            print(f"✗ 运行出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'elapsed_time': time.time() - start_time,
                'return_code': -1,
                'result_dir': str(result_dir),
                'error': str(e)
            }
        finally:
            # 恢复工作目录
            os.chdir(original_cwd)
    
    def extract_metrics(self, result_dir):
        """从实验结果中提取指标"""
        metrics = {}
        
        # 从日志文件中提取指标
        result_path = Path(result_dir)
        log_file = result_path / "run.log"
        output_file = result_path / "output.txt"
        
        # 尝试从输出文件中提取指标
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # 提取探索覆盖率 - 改进的提取逻辑（支持跨行）
                    if 'Final Exp Ratio' in content:
                        import re
                        # 查找 "Final Exp Ratio:" 后面可能跨行的所有数字
                        # 使用 DOTALL 模式匹配换行符
                        ratio_pattern = r'Final Exp Ratio:\s*\n?([\d\s.,]+)'
                        match = re.search(ratio_pattern, content, re.MULTILINE | re.DOTALL)
                        if match:
                            # 提取所有数字
                            ratio_str = match.group(1)
                            ratios = re.findall(r'(\d+\.\d+)', ratio_str)
                            if ratios:
                                # 转换为浮点数列表
                                ratio_values = [float(r) for r in ratios]
                                metrics['exploration_ratios'] = ratio_values
                                metrics['final_exploration_ratio'] = ratio_values[-1] if ratio_values else 0
                                metrics['max_exploration_ratio'] = max(ratio_values) if ratio_values else 0
                                metrics['mean_exploration_ratio'] = sum(ratio_values) / len(ratio_values) if ratio_values else 0
                                print(f"  ✓ 从output.txt提取到 {len(ratio_values)} 个探索覆盖率数据点")
                                print(f"    最终覆盖率: {metrics['final_exploration_ratio']:.4f}")
                        else:
                            print(f"  ⚠ 未找到Final Exp Ratio数据")
                    
                    # 提取探索面积
                    if 'Final Exp Area' in content:
                        import re
                        # 查找 "Final Exp Area:" 后面可能跨行的所有数字
                        area_pattern = r'Final Exp Area:\s*\n?([\d\s.,]+)'
                        match = re.search(area_pattern, content, re.MULTILINE | re.DOTALL)
                        if match:
                            area_str = match.group(1)
                            areas = re.findall(r'(\d+\.\d+)', area_str)
                            if areas:
                                area_values = [float(a) for a in areas]
                                metrics['explored_areas'] = area_values
                                metrics['final_explored_area'] = area_values[-1] if area_values else 0
                                metrics['max_explored_area'] = max(area_values) if area_values else 0
                                print(f"  ✓ 从output.txt提取到 {len(area_values)} 个探索面积数据点")
                                print(f"    最终面积: {metrics['final_explored_area']:.2f}")
                        else:
                            print(f"  ⚠ 未找到Final Exp Area数据")
                    
                    # 提取奖励信息
                    if 'Global eps mean' in content:
                        import re
                        match = re.search(r'Global eps mean.*?rew:\s*([\d.]+)', content)
                        if match:
                            metrics['mean_episode_reward'] = float(match.group(1))
            except Exception as e:
                print(f"  警告: 读取output.txt失败: {e}")
        
        # 从dump目录中提取数据
        # 尝试多个可能的dump目录路径
        result_path = Path(result_dir)
        possible_dump_dirs = [
            result_path.parent.parent / "tmp" / result_path.name,
            Path(self.original_path) / "tmp" / result_path.name if "original" in str(result_dir) else Path(self.modified_path) / "tmp" / result_path.name,
        ]
        
        dump_dir = None
        for possible_dir in possible_dump_dirs:
            if possible_dir.exists():
                dump_dir = possible_dir
                break
        
        if dump_dir and dump_dir.exists():
            # 查找explored_ratio.txt
            ratio_file = dump_dir / "explored_ratio.txt"
            if ratio_file.exists():
                try:
                    with open(ratio_file, 'r') as f:
                        ratios = []
                        for line in f:
                            line = line.strip()
                            if line.startswith('['):
                                # 解析numpy数组格式
                                import ast
                                arr = ast.literal_eval(line)
                                if isinstance(arr, list) and len(arr) > 0:
                                    ratios.extend(arr)
                        if ratios:
                            metrics['exploration_ratios'] = ratios
                            metrics['final_exploration_ratio'] = ratios[-1] if ratios else 0
                            metrics['max_exploration_ratio'] = max(ratios) if ratios else 0
                            metrics['mean_exploration_ratio'] = np.mean(ratios) if ratios else 0
                except Exception as e:
                    print(f"  警告: 无法解析explored_ratio.txt: {e}")
            
            # 查找explored_area.txt
            area_file = dump_dir / "explored_area.txt"
            if area_file.exists():
                try:
                    with open(area_file, 'r') as f:
                        areas = []
                        for line in f:
                            line = line.strip()
                            if line.startswith('['):
                                import ast
                                arr = ast.literal_eval(line)
                                if isinstance(arr, list) and len(arr) > 0:
                                    areas.extend(arr)
                        if areas:
                            metrics['explored_areas'] = areas
                            metrics['final_explored_area'] = areas[-1] if areas else 0
                            metrics['max_explored_area'] = max(areas) if areas else 0
                except Exception as e:
                    print(f"  警告: 无法解析explored_area.txt: {e}")
        
        return metrics
    
    def compare_results(self, original_metrics, modified_metrics):
        """对比两个实验的结果"""
        comparison = {
            'timestamp': self.timestamp,
            'original': original_metrics,
            'modified': modified_metrics,
            'improvements': {}
        }
        
        # 计算改进百分比
        for key in ['final_exploration_ratio', 'mean_exploration_ratio', 'max_exploration_ratio']:
            if key in original_metrics and key in modified_metrics:
                orig_val = original_metrics[key]
                mod_val = modified_metrics[key]
                if orig_val > 0:
                    improvement = ((mod_val - orig_val) / orig_val) * 100
                    comparison['improvements'][key] = {
                        'original': orig_val,
                        'modified': mod_val,
                        'improvement_percent': improvement,
                        'absolute_improvement': mod_val - orig_val
                    }
        
        return comparison
    
    def generate_report(self, comparison):
        """生成对比报告"""
        report_file = self.results_dir / "comparison_report.md"
        
        # 确保目录存在
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(str(report_file.resolve()), 'w', encoding='utf-8') as f:
                f.write("# 实验对比报告\n\n")
                f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## 实验配置\n\n")
                f.write(f"- 原项目路径: `{self.original_path}`\n")
                f.write(f"- 修改项目路径: `{self.modified_path}`\n")
                f.write(f"- 结果目录: `{self.results_dir}`\n\n")
                
                f.write("## 实验结果对比\n\n")
                if comparison.get('improvements'):
                    f.write("### 探索覆盖率指标\n\n")
                    f.write("| 指标 | 原项目 | 修改项目 | 改进 | 改进百分比 |\n")
                    f.write("|------|--------|----------|------|------------|\n")
                    
                    for key, data in comparison['improvements'].items():
                        metric_name = key.replace('_', ' ').title()
                        orig = f"{data['original']:.4f}"
                        mod = f"{data['modified']:.4f}"
                        abs_imp = f"{data['absolute_improvement']:+.4f}"
                        pct_imp = f"{data['improvement_percent']:+.2f}%"
                        f.write(f"| {metric_name} | {orig} | {mod} | {abs_imp} | {pct_imp} |\n")
                else:
                    f.write("### 探索覆盖率指标\n\n")
                    f.write("⚠ 没有可对比的改进数据\n\n")
                
                f.write("\n### 详细指标\n\n")
                f.write("#### 原项目指标\n\n")
                f.write("```json\n")
                f.write(json.dumps(comparison['original'], indent=2, default=str))
                f.write("\n```\n\n")
                
                f.write("#### 修改项目指标\n\n")
                f.write("```json\n")
                f.write(json.dumps(comparison['modified'], indent=2, default=str))
                f.write("\n```\n\n")
        except Exception as e:
            print(f"✗ 写入报告文件失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        print(f"\n✓ 对比报告已生成: {report_file.resolve()}")
        return report_file
    
    def plot_comparison(self, comparison):
        """绘制对比图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('实验对比结果', fontsize=16)
        
        # 1. 探索覆盖率对比
        if 'exploration_ratios' in comparison['original'] and 'exploration_ratios' in comparison['modified']:
            ax = axes[0, 0]
            orig_ratios = comparison['original']['exploration_ratios']
            mod_ratios = comparison['modified']['exploration_ratios']
            
            steps_orig = np.arange(len(orig_ratios))
            steps_mod = np.arange(len(mod_ratios))
            
            ax.plot(steps_orig, orig_ratios, label='原项目', linewidth=2)
            ax.plot(steps_mod, mod_ratios, label='修改项目', linewidth=2)
            ax.set_xlabel('步数')
            ax.set_ylabel('探索覆盖率')
            ax.set_title('探索覆盖率对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. 最终指标对比
        ax = axes[0, 1]
        metrics_to_compare = ['final_exploration_ratio', 'mean_exploration_ratio', 'max_exploration_ratio']
        metric_labels = ['最终覆盖率', '平均覆盖率', '最大覆盖率']
        
        orig_values = []
        mod_values = []
        for metric in metrics_to_compare:
            if metric in comparison['original']:
                orig_values.append(comparison['original'][metric])
            else:
                orig_values.append(0)
            if metric in comparison['modified']:
                mod_values.append(comparison['modified'][metric])
            else:
                mod_values.append(0)
        
        x = np.arange(len(metric_labels))
        width = 0.35
        ax.bar(x - width/2, orig_values, width, label='原项目', alpha=0.8)
        ax.bar(x + width/2, mod_values, width, label='修改项目', alpha=0.8)
        ax.set_ylabel('覆盖率')
        ax.set_title('关键指标对比')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. 改进百分比
        ax = axes[1, 0]
        improvements = []
        labels = []
        for key, data in comparison['improvements'].items():
            improvements.append(data['improvement_percent'])
            labels.append(key.replace('_', ' ').title())
        
        if improvements:
            colors = ['green' if x > 0 else 'red' for x in improvements]
            ax.barh(labels, improvements, color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('改进百分比 (%)')
            ax.set_title('改进百分比对比')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 4. 探索面积对比
        if 'explored_areas' in comparison['original'] and 'explored_areas' in comparison['modified']:
            ax = axes[1, 1]
            orig_areas = comparison['original']['explored_areas']
            mod_areas = comparison['modified']['explored_areas']
            
            steps_orig = np.arange(len(orig_areas))
            steps_mod = np.arange(len(mod_areas))
            
            ax.plot(steps_orig, orig_areas, label='原项目', linewidth=2)
            ax.plot(steps_mod, mod_areas, label='修改项目', linewidth=2)
            ax.set_xlabel('步数')
            ax.set_ylabel('探索面积 (m²)')
            ax.set_title('探索面积对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.results_dir / "comparison_plots.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ 对比图表已保存: {plot_file}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='对比实验脚本')
    parser.add_argument('--original', type=str, default=ORIGINAL_PROJECT,
                        help='原项目路径')
    parser.add_argument('--modified', type=str, default=MODIFIED_PROJECT,
                        help='修改项目路径')
    parser.add_argument('--output', type=str, default='comparison_results',
                        help='输出目录')
    parser.add_argument('--scene', type=str, default=None,
                        help='指定场景（如：Cantwell）')
    parser.add_argument('--episodes', type=int, default=1,
                        help='运行episode数量')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='单个实验超时时间（秒）')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.original, args.modified, args.output)
    
    # 构建实验参数
    base_args = [
        '--split', 'val',
        '--eval', '1',
        '--train_global', '0',
        '--train_local', '0',
        '--train_slam', '0',
        '--load_global', 'pretrained_models/model_best.global',
        '--load_local', 'pretrained_models/model_best.local',
        '--load_slam', 'pretrained_models/model_best.slam',
        '--num_processes', '1',
        '--auto_gpu_config', '0',
        '--num_episodes', str(args.episodes),
    ]
    
    if args.scene:
        base_args.extend(['--priority_scene', args.scene])
    
    # 原项目参数（不使用语义）
    # 注意：原项目可能不支持 --priority_scene 参数，需要检查
    original_exp_name = f'original_{args.scene or "default"}'
    original_args = base_args.copy()
    
    # 原项目可能不支持priority_scene，先移除
    if '--priority_scene' in original_args:
        idx = original_args.index('--priority_scene')
        original_args.pop(idx)  # 移除 --priority_scene
        original_args.pop(idx)  # 移除场景名
    
    original_args.extend([
        '--exp_name', original_exp_name,
        '-v', '1',  # 开启可视化
    ])
    
    # 修改项目参数（使用语义）
    modified_exp_name = f'modified_{args.scene or "default"}'
    modified_args = base_args + [
        '--use_semantic',
        '--semantic_use_all_classes',
        '--semantic_conf_thresh', '0.1',
        '--semantic_interval', '1',
        '--semantic_reward_coeff', '0.12',
        '--exp_name', modified_exp_name,
        '-v', '1',  # 开启可视化
    ]
    
    print("="*60)
    print("开始对比实验（并行运行）")
    print("="*60)
    print(f"配置: 场景={args.scene or '默认'}, Episodes={args.episodes}, 可视化=开启, 进程数=1")
    print("="*60)
    print("\n注意: 两个项目将同时运行，会显示两个可视化窗口")
    print("="*60)
    
    # 使用线程并行运行两个项目
    results_queue = Queue()
    
    def run_original():
        """运行原项目的线程函数"""
        print("\n[并行] 启动原项目（无语义检测）...")
        result = runner.run_experiment(
            args.original,
            'original',
            original_args,
            timeout=args.timeout
        )
        results_queue.put(('original', result))
        if result.get('success'):
            print("\n✓ 原项目运行完成")
        else:
            print("\n✗ 原项目运行失败")
    
    def run_modified():
        """运行修改项目的线程函数"""
        print("\n[并行] 启动修改项目（带语义检测）...")
        try:
            result = runner.run_experiment(
                args.modified,
                'modified',
                modified_args,
                timeout=args.timeout
            )
            print(f"[调试] 修改项目返回结果: {result}")
            results_queue.put(('modified', result))
            if result.get('success'):
                print("\n✓ 修改项目运行完成")
                result_dir = result.get('result_dir', 'N/A')
                if result_dir and result_dir != 'N/A':
                    if Path(result_dir).exists():
                        print(f"  结果目录已存在: {result_dir}")
                    else:
                        print(f"  ⚠ 警告: 结果目录不存在: {result_dir}")
            else:
                print("\n✗ 修改项目运行失败")
                print(f"  错误信息: {result.get('error', '未知错误')}")
        except Exception as e:
            print(f"\n✗ 修改项目运行异常: {e}")
            import traceback
            traceback.print_exc()
            results_queue.put(('modified', {
                'success': False,
                'result_dir': None,
                'error': str(e)
            }))
    
    # 创建并启动两个线程
    thread_original = Thread(target=run_original, daemon=False)
    thread_modified = Thread(target=run_modified, daemon=False)
    
    print("\n[并行] 同时启动两个项目...")
    thread_original.start()
    time.sleep(2)  # 稍微延迟启动第二个，避免资源竞争
    thread_modified.start()
    
    # 等待两个线程完成
    print("\n[并行] 等待两个项目运行完成...")
    thread_original.join()
    thread_modified.join()
    
    # 从队列中获取结果
    original_result = None
    modified_result = None
    
    # 等待队列中有结果（最多等待10秒）
    import queue
    timeout_count = 0
    while (original_result is None or modified_result is None) and timeout_count < 50:
        try:
            name, result = results_queue.get(timeout=0.2)
            if name == 'original':
                original_result = result
                result_dir = result.get('result_dir', 'N/A')
                print(f"\n✓ 收到原项目结果: {result_dir}")
                if result_dir and result_dir != 'N/A':
                    exists = Path(result_dir).exists()
                    print(f"  原项目结果目录存在: {exists}")
            elif name == 'modified':
                modified_result = result
                result_dir = result.get('result_dir', 'N/A')
                print(f"\n✓ 收到修改项目结果: {result_dir}")
                if result_dir and result_dir != 'N/A':
                    exists = Path(result_dir).exists()
                    print(f"  修改项目结果目录存在: {exists}")
                    if not exists:
                        print(f"  ⚠ 警告: 修改项目结果目录不存在，尝试创建...")
                        try:
                            Path(result_dir).mkdir(parents=True, exist_ok=True)
                            print(f"  ✓ 已创建目录: {result_dir}")
                        except Exception as e:
                            print(f"  ✗ 创建失败: {e}")
        except queue.Empty:
            timeout_count += 1
            if timeout_count % 10 == 0:
                print(f"  等待结果... ({timeout_count * 0.2:.1f}秒)")
    
    # 如果结果为空，设置默认值
    if original_result is None:
        print("\n⚠ 警告: 未获取到原项目结果")
        original_result = {'success': False, 'result_dir': None}
    
    if modified_result is None:
        print("\n⚠ 警告: 未获取到修改项目结果")
        modified_result = {'success': False, 'result_dir': None}
    
    # 打印详细的结果信息
    print(f"\n{'='*60}")
    print("结果验证")
    print(f"{'='*60}")
    print(f"原项目运行状态: {'成功' if original_result.get('success') else '失败'}")
    if original_result.get('result_dir'):
        orig_dir = Path(original_result['result_dir'])
        print(f"原项目结果目录: {original_result['result_dir']}")
        print(f"  目录存在: {orig_dir.exists()}")
        if orig_dir.exists():
            files = list(orig_dir.glob('*'))
            print(f"  文件数: {len(files)}")
    
    print(f"\n修改项目运行状态: {'成功' if modified_result.get('success') else '失败'}")
    if modified_result.get('result_dir'):
        mod_dir = Path(modified_result['result_dir'])
        print(f"修改项目结果目录: {modified_result['result_dir']}")
        print(f"  目录存在: {mod_dir.exists()}")
        if not mod_dir.exists():
            print(f"  ⚠ 警告: 修改项目结果目录不存在！")
            # 尝试从runner的results_dir创建
            expected_dir = runner.results_dir / 'modified'
            print(f"  尝试创建期望目录: {expected_dir}")
            try:
                expected_dir.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ 已创建目录: {expected_dir}")
                modified_result['result_dir'] = str(expected_dir)
            except Exception as e:
                print(f"  ✗ 创建失败: {e}")
    else:
        print(f"  ⚠ 警告: 修改项目结果目录为空！")
        # 尝试创建默认目录
        expected_dir = runner.results_dir / 'modified'
        print(f"  尝试创建默认目录: {expected_dir}")
        try:
            expected_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ 已创建默认目录: {expected_dir}")
            modified_result['result_dir'] = str(expected_dir)
        except Exception as e:
            print(f"  ✗ 创建失败: {e}")
    print(f"{'='*60}\n")
    
    # 提取指标
    print("\n[3/3] 提取和对比指标...")
    
    # 尝试从多个位置提取指标
    original_metrics = {}
    modified_metrics = {}
    
    # 从结果目录提取
    if original_result.get('result_dir'):
        original_metrics = runner.extract_metrics(original_result['result_dir'])
    
    # 从原项目的tmp目录提取
    original_dump_dir = Path(args.original) / "tmp" / original_exp_name
    if original_dump_dir.exists():
        print(f"  从原项目dump目录提取: {original_dump_dir}")
        # 尝试从dump目录直接读取
        ratio_file = original_dump_dir / "explored_ratio.txt"
        if ratio_file.exists():
            try:
                with open(ratio_file, 'r') as f:
                    ratios = []
                    for line in f:
                        line = line.strip()
                        if line.startswith('['):
                            import ast
                            arr = ast.literal_eval(line)
                            if isinstance(arr, list) and len(arr) > 0:
                                ratios.extend(arr)
                    if ratios:
                        original_metrics['exploration_ratios'] = ratios
                        original_metrics['final_exploration_ratio'] = ratios[-1] if ratios else 0
                        original_metrics['mean_exploration_ratio'] = np.mean(ratios) if ratios else 0
            except Exception as e:
                print(f"  警告: 无法读取原项目数据: {e}")
    
    print("\n提取修改项目指标...")
    if modified_result.get('result_dir'):
        print(f"  从结果目录: {modified_result['result_dir']}")
        modified_metrics = runner.extract_metrics(modified_result['result_dir'])
    else:
        print("  警告: 修改项目结果目录为空")
        modified_metrics = {}
    
    # 从修改项目的tmp目录提取
    modified_dump_dir = Path(args.modified) / "tmp" / modified_exp_name
    print(f"  检查修改项目dump目录: {modified_dump_dir}")
    if modified_dump_dir.exists():
        print(f"  从修改项目dump目录提取: {modified_dump_dir}")
        ratio_file = modified_dump_dir / "explored_ratio.txt"
        if ratio_file.exists():
            try:
                with open(ratio_file, 'r') as f:
                    ratios = []
                    for line in f:
                        line = line.strip()
                        if line.startswith('['):
                            import ast
                            arr = ast.literal_eval(line)
                            if isinstance(arr, list) and len(arr) > 0:
                                ratios.extend(arr)
                    if ratios:
                        modified_metrics['exploration_ratios'] = ratios
                        modified_metrics['final_exploration_ratio'] = ratios[-1] if ratios else 0
                        modified_metrics['mean_exploration_ratio'] = np.mean(ratios) if ratios else 0
            except Exception as e:
                print(f"  警告: 无法读取修改项目数据: {e}")
    
    # 打印调试信息
    print(f"\n原项目结果: {original_result}")
    print(f"修改项目结果: {modified_result}")
    print(f"原项目指标: {len(original_metrics)} 个")
    print(f"修改项目指标: {len(modified_metrics)} 个")
    
    # 对比结果
    comparison = runner.compare_results(original_metrics, modified_metrics)
    
    # 保存对比结果
    comparison_file = runner.results_dir / "comparison.json"
    comparison_file.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    try:
        comparison_file_abs = comparison_file.resolve()
        with open(str(comparison_file_abs), 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, default=str, ensure_ascii=False)
        print(f"✓ 对比结果已保存: {comparison_file_abs}")
    except Exception as e:
        print(f"✗ 保存对比结果失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成报告
    try:
        runner.generate_report(comparison)
    except Exception as e:
        print(f"✗ 生成报告失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 绘制图表
    if comparison.get('improvements'):
        try:
            runner.plot_comparison(comparison)
        except Exception as e:
            print(f"✗ 绘制图表失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠ 没有改进数据，跳过图表绘制")
    
    print("\n" + "="*60)
    print("对比实验完成！")
    print(f"结果目录: {runner.results_dir.resolve()}")
    print(f"原项目结果: {original_result.get('result_dir', 'N/A')}")
    print(f"修改项目结果: {modified_result.get('result_dir', 'N/A')}")
    print("="*60)

if __name__ == "__main__":
    main()

