"""논문용 평가 결과 표 생성 스크립트"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# 평가 결과 데이터 (로그에서 추출)
# Temperature search 후 best temperature 결과 사용

results = {
    'HumanEval': {
        'n_problems': 80,
        'n_samples': 20,
        'eval_method': 'Function execution',
        'temperature': 0.8,
        'Pure': {'pass@1': 2.94, 'pass@5': 13.17, 'pass@10': 23.30, 'pass@20': 37.50},
        'Baseline': {'pass@1': 2.56, 'pass@5': 11.85, 'pass@10': 21.45, 'pass@20': 35.00},
        'Verifiable': {'pass@1': 3.25, 'pass@5': 14.27, 'pass@10': 24.79, 'pass@20': 40.00},
    },
    'MBPP': {
        'n_problems': 100,
        'n_samples': 20,
        'eval_method': 'Assert execution',
        'temperature': 0.2,
        'Pure': {'pass@1': 17.00, 'pass@5': 24.58, 'pass@10': 27.99, 'pass@20': 31.00},
        'Baseline': {'pass@1': 17.50, 'pass@5': 23.55, 'pass@10': 26.28, 'pass@20': 30.00},
        'Verifiable': {'pass@1': 17.80, 'pass@5': 26.26, 'pass@10': 30.29, 'pass@20': 35.00},
    },
    'GSM8K': {
        'n_problems': 100,
        'n_samples': 20,
        'eval_method': 'Exact match',
        'temperature': 0.2,
        'Pure': {'pass@1': 2.95, 'pass@5': 7.34, 'pass@10': 10.40, 'pass@20': 14.00},
        'Baseline': {'pass@1': 3.20, 'pass@5': 8.80, 'pass@10': 11.34, 'pass@20': 13.00},
        'Verifiable': {'pass@1': 3.00, 'pass@5': 8.45, 'pass@10': 12.16, 'pass@20': 17.00},
    },
    'CodeContests': {
        'n_problems': 20,
        'n_samples': 20,
        'eval_method': 'stdin/stdout (10 tests)',
        'temperature': 0.2,
        'Pure': {'pass@1': 0.25, 'pass@5': 1.25, 'pass@10': 2.50, 'pass@20': 5.00},
        'Baseline': {'pass@1': 1.00, 'pass@5': 4.25, 'pass@10': 6.97, 'pass@20': 10.00},
        'Verifiable': {'pass@1': 0.50, 'pass@5': 2.50, 'pass@10': 5.00, 'pass@20': 10.00},
    },
}

def create_results_table():
    """논문용 평가 결과 표 생성"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # 제목
    fig.suptitle('Table 1: Pass@K Evaluation Results on Code Generation Benchmarks',
                 fontsize=14, fontweight='bold', y=0.95)

    # 서브타이틀
    ax.text(0.5, 0.92,
            'Models: Meta-LLaMA MTP 7B (4-head). Training: LoRA (rank=64) on CodeContests.\n'
            'Pure = Pretrained MTP, Baseline = MTP + SFT, Verifiable = MTP + WMTP (GAE-weighted)',
            ha='center', va='top', fontsize=9, style='italic',
            transform=ax.transAxes)

    # 테이블 데이터 구성
    col_labels = ['Dataset', 'Problems', 'Samples', 'Eval Method', 'Temp',
                  'Model', 'Pass@1', 'Pass@5', 'Pass@10', 'Pass@20']

    cell_data = []
    cell_colors = []

    row_colors = {
        'Pure': '#E8F4FD',      # 연한 파랑
        'Baseline': '#FFF3E0',   # 연한 주황
        'Verifiable': '#E8F5E9', # 연한 초록
    }

    for dataset, data in results.items():
        for i, model in enumerate(['Pure', 'Baseline', 'Verifiable']):
            if i == 0:
                row = [dataset, str(data['n_problems']), str(data['n_samples']),
                       data['eval_method'], str(data['temperature'])]
            else:
                row = ['', '', '', '', '']

            scores = data[model]
            row.extend([
                model,
                f"{scores['pass@1']:.2f}%",
                f"{scores['pass@5']:.2f}%",
                f"{scores['pass@10']:.2f}%",
                f"{scores['pass@20']:.2f}%",
            ])
            cell_data.append(row)
            cell_colors.append([row_colors[model]] * len(col_labels))

    # 테이블 생성
    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=['#D0D0D0'] * len(col_labels),
        cellLoc='center',
        loc='center',
        bbox=[0.02, 0.08, 0.96, 0.80]
    )

    # 스타일 조정
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # 헤더 볼드
    for j in range(len(col_labels)):
        table[(0, j)].set_text_props(fontweight='bold')

    # 데이터셋 이름 볼드
    for i in range(1, len(cell_data) + 1):
        if cell_data[i-1][0]:  # 데이터셋 이름이 있는 행
            table[(i, 0)].set_text_props(fontweight='bold')

    # Best 결과 하이라이트 (각 데이터셋별 Pass@1 최고값)
    dataset_start_rows = [1, 4, 7, 10]  # 각 데이터셋 시작 행
    for start_row in dataset_start_rows:
        # Pass@1 값 추출 (인덱스 6)
        values = []
        for i in range(3):
            val_str = cell_data[start_row - 1 + i][6]
            values.append(float(val_str.replace('%', '')))

        # 최고값 찾기
        max_idx = np.argmax(values)
        best_row = start_row + max_idx

        # Pass@1, 5, 10, 20 모두 볼드 처리
        for col in [6, 7, 8, 9]:
            table[(best_row, col)].set_text_props(fontweight='bold')

    # 범례
    legend_elements = [
        mpatches.Patch(facecolor='#E8F4FD', edgecolor='gray', label='Pure (Pretrained MTP)'),
        mpatches.Patch(facecolor='#FFF3E0', edgecolor='gray', label='Baseline (MTP + SFT)'),
        mpatches.Patch(facecolor='#E8F5E9', edgecolor='gray', label='Verifiable (MTP + WMTP)'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3,
              bbox_to_anchor=(0.5, -0.02), fontsize=9)

    # 하단 노트
    note_text = (
        "Notes: Pass@K computed using unbiased estimator (Chen et al., 2021). "
        "Temperature selected via search (0.2, 0.8). "
        "Best results per dataset in bold."
    )
    ax.text(0.5, 0.02, note_text, ha='center', va='bottom', fontsize=8,
            style='italic', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('/Users/wesley/Desktop/wooshikwon/weighted_mtp/docs/figures/evaluation_results_table.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/wesley/Desktop/wooshikwon/weighted_mtp/docs/figures/evaluation_results_table.pdf',
                bbox_inches='tight', facecolor='white')
    print("저장 완료: evaluation_results_table.png, evaluation_results_table.pdf")
    plt.close()


def create_comparison_chart():
    """Pass@K 비교 바 차트 생성"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Figure 1: Pass@K Performance Comparison Across Benchmarks',
                 fontsize=14, fontweight='bold')

    datasets = ['HumanEval', 'MBPP', 'GSM8K', 'CodeContests']
    models = ['Pure', 'Baseline', 'Verifiable']
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    k_values = ['pass@1', 'pass@5', 'pass@10', 'pass@20']

    for idx, (ax, dataset) in enumerate(zip(axes.flat, datasets)):
        data = results[dataset]
        x = np.arange(len(k_values))
        width = 0.25

        for i, model in enumerate(models):
            values = [data[model][k] for k in k_values]
            bars = ax.bar(x + i * width, values, width, label=model, color=colors[i], alpha=0.8)

            # 값 레이블
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Metric')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_title(f'{dataset} (n={data["n_problems"]}, T={data["temperature"]})')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['@1', '@5', '@10', '@20'])
        ax.legend(loc='upper left', fontsize=8)
        ax.set_ylim(0, max(max(data[m][k] for k in k_values for m in models) * 1.3, 5))
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/wesley/Desktop/wooshikwon/weighted_mtp/docs/figures/evaluation_comparison_chart.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/wesley/Desktop/wooshikwon/weighted_mtp/docs/figures/evaluation_comparison_chart.pdf',
                bbox_inches='tight', facecolor='white')
    print("저장 완료: evaluation_comparison_chart.png, evaluation_comparison_chart.pdf")
    plt.close()


def create_improvement_table():
    """Verifiable vs Baseline 개선율 표"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    fig.suptitle('Table 2: Improvement of Verifiable (WMTP) over Baseline (SFT)',
                 fontsize=12, fontweight='bold', y=0.95)

    col_labels = ['Dataset', 'Δ Pass@1', 'Δ Pass@5', 'Δ Pass@10', 'Δ Pass@20', 'Avg Δ']

    cell_data = []
    cell_colors = []

    for dataset, data in results.items():
        baseline = data['Baseline']
        verifiable = data['Verifiable']

        deltas = []
        for k in ['pass@1', 'pass@5', 'pass@10', 'pass@20']:
            delta = verifiable[k] - baseline[k]
            deltas.append(delta)

        avg_delta = np.mean(deltas)

        row = [dataset]
        row_color = []

        for delta in deltas:
            if delta > 0:
                row.append(f'+{delta:.2f}%p')
                row_color.append('#C8E6C9')  # 초록
            elif delta < 0:
                row.append(f'{delta:.2f}%p')
                row_color.append('#FFCDD2')  # 빨강
            else:
                row.append(f'{delta:.2f}%p')
                row_color.append('#FFFFFF')

        if avg_delta > 0:
            row.append(f'+{avg_delta:.2f}%p')
            row_color.append('#A5D6A7')
        else:
            row.append(f'{avg_delta:.2f}%p')
            row_color.append('#EF9A9A')

        cell_data.append(row)
        cell_colors.append(['#F5F5F5'] + row_color)

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=['#BDBDBD'] * len(col_labels),
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.2, 0.8, 0.65]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    for j in range(len(col_labels)):
        table[(0, j)].set_text_props(fontweight='bold')

    # 범례
    ax.text(0.5, 0.12,
            'Green: Verifiable outperforms Baseline | Red: Baseline outperforms Verifiable',
            ha='center', fontsize=9, style='italic', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('/Users/wesley/Desktop/wooshikwon/weighted_mtp/docs/figures/improvement_table.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/wesley/Desktop/wooshikwon/weighted_mtp/docs/figures/improvement_table.pdf',
                bbox_inches='tight', facecolor='white')
    print("저장 완료: improvement_table.png, improvement_table.pdf")
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('/Users/wesley/Desktop/wooshikwon/weighted_mtp/docs/figures', exist_ok=True)

    create_results_table()
    create_comparison_chart()
    create_improvement_table()
    print("\n모든 그림 생성 완료!")
