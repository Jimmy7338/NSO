"""Summarize and check provenance without rerunning simulation or fitting."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = {
    'raw_visibility_m2': '原可见性预测 V',
    'raw_acquisition_m2': '原重建预测 A',
    'scaled_acquisition_m2': 'A 单系数缩放',
    'corrected_acquisition_m2': 'A 按观测支持距离修正',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def verify_resource(resource):
    path = Path(resource['path'])
    assert sha(path) == resource['sha256'], path
    return path


def lock_source_paths():
    protocol = read(HERE / 'protocol.json')
    fit = HERE / 'calibrated_coefficients.json'
    coeff = read(fit)
    summary = HERE / 'heldout_summary.json'
    validation = read(summary)
    freeze = HERE / 'freeze.json'
    for path, expected in read(freeze)['source_sha256'].items():
        assert sha(path)==expected, path
    assert coeff['freeze_sha256'] == validation['freeze_sha256'] == sha(freeze)
    assert validation['fit_sha256'] == sha(fit)
    assert len(protocol['layouts']) == len({row['id'] for row in protocol['layouts']}) == 8
    calibrations = [row['id'] for row in protocol['layouts'] if row['split']=='calibration']
    assert coeff['layout_ids'] == calibrations
    assert 0 <= coeff['beta_far'] <= coeff['beta_near'] <= 1
    for source, expected in coeff['calibration_sources_sha256'].items():
        assert Path(source) == HERE / 'calibration_labels.json' and sha(source)==expected
    check = {}
    for entry in protocol['layouts']:
        layout = entry['id']
        split = entry['split']
        res_dir = HERE / 'runs' / split / layout
        run = res_dir / 'run_receipt.json'
        if not run.exists():
            check[layout] = dict(status='failed_or_missing')
            continue
        assert freeze.stat().st_mtime_ns < run.stat().st_mtime_ns
        data = read(run)
        assert data['freeze_sha256']==sha(freeze)
        predictions = verify_resource(data['predictions'])
        prior = read(predictions)
        assert prior['recorded_before_actual_branches']
        if split == 'heldout':
            assert fit.stat().st_mtime_ns < predictions.stat().st_mtime_ns
            assert prior['correction_coefficients_sha256']==sha(fit)
        else:
            assert run.stat().st_mtime_ns < fit.stat().st_mtime_ns
            assert prior['correction_coefficients_sha256'] is None
        verify_resource(data['before_mesh'])
        for forecast in prior['predictions']:
            assert np.isclose(forecast['support_near_m2'] + forecast['support_far_m2'],
                              forecast['raw_acquisition_m2'], atol=1.e-10)
        for branch in data['branches']:
            endpoint = verify_resource(branch['execution'])
            assert predictions.stat().st_mtime_ns < endpoint.stat().st_mtime_ns
            verify_resource(branch['after_mesh'])
        check[layout] = dict(status='hashes_verified', endpoints=sum(
            bool(read(branch['execution']['path'])['endpoint_reached']) for branch in data['branches']),
            prediction_inputs=len(prior['initial_history_packet_sha256']), modeled=len(prior['modeled_support_tracks']),
            run=receipt(run))
    touched = [HERE / f'{name}.json' for name in ('calibration_labels', 'heldout_labels',
               'calibration_summary', 'heldout_summary', 'calibrated_coefficients', 'freeze', 'protocol')]
    return coeff, validation, dict(layout_checks=check, evidence={p.name: receipt(p) for p in touched})


def main():
    fit, report, audit = lock_source_paths()
    metric_names = [
        'mean_layout_mae_m2','mean_layout_bias_m2','concordant_pairs','comparable_pairs','forecast_ties',
        'mean_layout_actual_selected_gross_m2','mean_layout_actual_selected_net_m2','mean_layout_selection_regret_m2',
    ]
    with (HERE/'heldout_metrics.csv').open('x', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', *metric_names])
        for method in METHODS:
            w.writerow([method, *(report['metrics'][method][key] for key in metric_names)])
    all_rows = [row for l in report['layouts'] for row in l['rows']]
    with (HERE/'heldout_routes.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    def n(value):
        return '—' if value is None else f'{value:.4f}'
    lines = [
        '# 家具观察收益的独立校准与留出验证',
        '',
        '每个程序化房间包含一个新构造的家具。4 个校准布局拟合折扣，4 个留出布局只作评价。'
        '每布局从同一可见前缀执行三条等动作数相机路线。所有预测共用原模型的拟合几何、全后验及 TSDF。',
        '',
        f'距原有实例表面 0.25 m 以内和以外的新增面积分别乘 {fit["beta_near"]:.2f} 和 '
        f'{fit["beta_far"]:.2f}。单系数参照为 {fit["beta_global"]:.2f}。预测之前锁定了留出布局、估计规则和评价口径。',
        '',
        '| 留出预测 | 面积 MAE（m²） | 有符号偏差（m²） | 正确排序对数 | 选中路线实际毛增量（m²） | 选中路线实际净增量（m²） | 距最佳毛面积（m²） |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for method, label in METHODS.items():
        s = report['metrics'][method]
        lines.append(f'| {label} | {n(s["mean_layout_mae_m2"])} | {n(s["mean_layout_bias_m2"])} | '
                     f'{s["concordant_pairs"]}/{s["comparable_pairs"]} | '
                     f'{n(s["mean_layout_actual_selected_gross_m2"])} | {n(s["mean_layout_actual_selected_net_m2"])} | '
                     f'{n(s["mean_layout_selection_regret_m2"])} |')
    lines += ['', '各布局等权。排序只计实际毛增量不同的路线对，预测并列不记为正确。面积评价用独立的真值家具表面和 5 cm 距离阈值，早停或碰撞保留在逐路线记录。',
              '', '该实验使用精确位姿和理想可见实例信息。路线由脚本预先固定，选点比较为同三条已执行分支上的反事实比较，覆盖效率、完整闭环收益和实车表现仍没有由这些数据衡量。']
    with (HERE/'RESULTS.md').open('x') as f:
        f.write('\n'.join(lines)+'\n')
    audit['summary_artifacts'] = {name: receipt(HERE/name) for name in ('RESULTS.md','heldout_metrics.csv','heldout_routes.csv')}
    with (HERE/'final_receipt.json').open('x') as f:
        json.dump(audit,f,indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
