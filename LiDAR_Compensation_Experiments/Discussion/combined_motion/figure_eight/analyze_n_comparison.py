from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).parent / 'experiment_results' / 'make__figure8_static'
OUT = Path(__file__).parent / 'n_comparison_analysis'
OUT.mkdir(exist_ok=True)
sources = {10:'N=10',20:'20261022_201445_272536',30:'N=30',40:'N=40',50:'20261023_130922_127636'}
rows=[]
all_trials={}
for n, folder in sources.items():
    d=BASE/folder
    t=pd.read_csv(d/'trials.csv').query("status == 'completed'")
    s=pd.read_csv(d/'scans.csv')
    s=s[s.trial.isin(t.trial)]
    assert s.timing_history_samples_N.eq(n).all()
    all_trials[n]=t
    # Preserve the experiment's equal-trial weighting, including timing.
    metrics=['raw_processing_ms','ours_processing_ms','extra_processing_vs_raw_ms','raw_observation_to_control_estimated_ms','ours_observation_to_control_estimated_ms']
    means=s.groupby('trial')[metrics].mean().mean()
    rawxy=(t.raw_x_mae_m+t.raw_y_mae_m)/2
    xy=(t.corrected_x_mae_m+t.corrected_y_mae_m)/2
    delta=t.raw_bearing_mae_deg-t.ours_bearing_mae_deg
    r=dict(N=n,source=folder,trials=len(t),scans=len(s),raw_x=t.raw_x_mae_m.mean(),x=t.corrected_x_mae_m.mean(),raw_y=t.raw_y_mae_m.mean(),y=t.corrected_y_mae_m.mean(),raw_xy=rawxy.mean(),xy=xy.mean(),xy_reduction_pct=100*(1-xy.mean()/rawxy.mean()),raw_bearing=t.raw_bearing_mae_deg.mean(),bearing=t.ours_bearing_mae_deg.mean(),bearing_reduction_pct=100*(1-t.ours_bearing_mae_deg.mean()/t.raw_bearing_mae_deg.mean()),bearing_delta=delta.mean(),bearing_delta_ci95_halfwidth=1.984*delta.std(ddof=1)/np.sqrt(len(t)),bearing_improved_trials=int((delta>0).sum()),processing_median_ms=s.ours_processing_ms.median(),processing_p95_ms=s.ours_processing_ms.quantile(.95),processing_p99_ms=s.ours_processing_ms.quantile(.99),processing_over_50ms_pct=100*s.ours_processing_ms.gt(50).mean(),**means.to_dict())
    rows.append(r)
result=pd.DataFrame(rows)
result.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
t=all_trials[10]
split=[]
for label, sub in [('1-48',t[t.trial<=48]),('49-100',t[t.trial>=49])]:
    split.append(dict(trials=label,count=len(sub),scans_mean=sub.scans.mean(),raw_bearing=sub.raw_bearing_mae_deg.mean(),bearing=sub.ours_bearing_mae_deg.mean(),reduction_pct=100*(1-sub.ours_bearing_mae_deg.mean()/sub.raw_bearing_mae_deg.mean())))
print(result.round(5).to_string(index=False))
print('N10 SPLIT',split)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(15,4.6))
axs[0].plot(result.N,result.raw_bearing,'o--',color='#8796a5',label='Raw (within each run)')
axs[0].plot(result.N,result.bearing,'o-',color='#167c80',label='Corrected')
axs[0].set(xlabel='History samples N',ylabel='Bearing MAE (degrees)',title='Accuracy: equal weight per trial')
axs[0].legend(fontsize=9)
axs[1].plot(result.N,result.ours_processing_ms,'o-',color='#167c80',label='Corrected mean')
axs[1].plot(result.N,result.raw_processing_ms,'o--',color='#8796a5',label='Raw mean')
axs[1].set(xlabel='History samples N',ylabel='Processing time (ms / scan)',title='Measured processing cost',ylim=(0,3))
axs[1].legend(fontsize=9)
axs[2].plot(t.trial,t.raw_bearing_mae_deg,color='#8796a5',label='Raw')
axs[2].plot(t.trial,t.ours_bearing_mae_deg,color='#167c80',label='Corrected')
axs[2].axvline(48.5,color='#d16a45',ls='--')
axs[2].set(xlabel='Trial (N=10)',ylabel='Bearing MAE (degrees)',title='N=10: regime change at trial 49')
axs[2].legend(fontsize=9)
for ax in axs: ax.grid(alpha=.18)
fig.tight_layout()
fig.savefig(OUT/'comparison.png',dpi=180)
fig.savefig(OUT/'comparison.svg')
lines=['# N=10~50 실험 결과 분석','', '완료 시행에 동일 가중치를 적용했다. 각 시행 안에서는 스캔별 절대오차를 평균했다. 감소율 = 100 × (raw 평균 − 보정 평균) / raw 평균. XY 평균은 (X MAE + Y MAE)/2이며 유클리드 거리 오차가 아니다. 처리 시간도 시행별 평균의 평균을 사용했다. p95/p99는 전체 스캔 기준이다.','', '| N | 완료 시행 | Raw XY (m) | 보정 XY (m) | 감소율 | Raw 방위각 (°) | 보정 방위각 (°) | 감소율 | Raw 처리(ms) | 보정 처리(ms) | 추가(ms) |','|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    lines.append(f"| {r['N']} | {r['trials']} | {r['raw_xy']:.4f} | {r['xy']:.4f} | {r['xy_reduction_pct']:.2f}% | {r['raw_bearing']:.4f} | {r['bearing']:.4f} | {r['bearing_reduction_pct']:.2f}% | {r['raw_processing_ms']:.4f} | {r['ours_processing_ms']:.4f} | {r['extra_processing_vs_raw_ms']:.4f} |")
lines+=['','## 해석','', 'N=20이 본 비교에서 가장 낮은 보정 XY 및 방위각 평균 오차를 보였다. N=30~50에서는 N을 늘려도 오차가 더 줄지 않았다. N=20을 현재 조건의 우선 후보로 볼 수 있으나, N=10의 중간 변화 때문에 N=10 대비 우월성을 확정하려면 동일 조건 재실험이 필요하다.','', 'N=20~50의 처리 시간은 약 2.3 ms로 거의 일정하다. 이 기록은 경과 시간이며 FLOPs/CPU 사용량/메모리나 이론적 복잡도를 직접 측정한 값은 아니다. 원시 처리 시간이 매우 짧아 배수보다 추가 ms가 해석에 유용하다. 지연 추정값에는 설정된 스캔 시간 100 ms 및 지연 대기 약 210 ms가 포함되며, 처리 시간과 구분해야 한다. 처리 시간의 평균만으로 매번 제어 주기 50 ms를 만족한다고 단정할 수 없다.','', '긴 이력이 비정상 운동의 오래된 정보를 포함하여 2차 다항식의 국소 근사를 악화시켰을 가능성이 있다. 이는 결과와 부합하는 가설이며, 코드 및 시간별 운동과 오차의 연계 검증 전에는 원인으로 확정할 수 없다.','', '## N=10의 시행 중 변화','', '| 시행 | 개수 | 평균 스캔 수 | Raw 방위각 | 보정 방위각 | 감소율 |','|---|---|---|---|---|---|']
for r in split: lines.append(f"| {r['trials']} | {r['count']} | {r['scans_mean']:.2f} | {r['raw_bearing']:.4f} | {r['bearing']:.4f} | {r['reduction_pct']:.2f}% |")
lines+=['','1~48회와 49~100회의 처리·관측 조건이 달라졌음을 시사한다. 원인은 로그만으로 확정할 수 없다. 전체 N=10 결과를 유지하고, 위 구분은 진단 목적으로만 제시했다. 전반부만 골라 N 효과를 확정하지 않았다.','', '## 처리 시간 꼬리 및 지연','', '| N | 중앙값(ms) | p95(ms) | p99(ms) | 50ms 초과 비율 | raw 관측→제어 추정(ms) | 보정 관측→제어 추정(ms) |','|---|---|---|---|---|---|---|']
for r in rows: lines.append(f"| {r['N']} | {r['processing_median_ms']:.3f} | {r['processing_p95_ms']:.3f} | {r['processing_p99_ms']:.3f} | {r['processing_over_50ms_pct']:.3f}% | {r['raw_observation_to_control_estimated_ms']:.3f} | {r['ours_observation_to_control_estimated_ms']:.3f} |")
lines+=['','## 출처 및 비교 조건','', '공통 설정: latency=0.2 s, scan_time=0.1 s, polynomial degree=2, history_wait_interval=0.02 s, control_dt=0.05 s, figure-eight=90×45 m. history_wait_interval은 설정된 대기 간격이며 실제 샘플 타임스탬프 간격으로 간주하지 않았다. raw는 별도 무보정 비행이 아니라 각 실험의 동일 스캔에서 계산한 기준이다.']
for n,folder in sources.items():
    lines.append(f"- N={n}: `{folder}` — settings.csv, trials.csv, scans.csv, summary.csv. scans.csv의 timing_history_samples_N도 설정과 일치함.")
lines+=['', '`N=20` 폴더는 64회 완료 후 65회에서 ValueError로 중단돼 주 비교에서 제외했다. 대신 동일 N=20의 100회 완료 폴더를 사용했다. 중단 실험의 방위각 보정 평균 1.17225°는 완료 실험의 1.16404°와 유사했다. N=30은 명시적 N=30 폴더를 사용하고 다른 날짜 폴더와 합치지 않았다. 원본 파일은 변경하지 않았다.','', '![비교](comparison.png)']
(OUT/'analysis_ko.md').write_text('\n'.join(lines),encoding='utf-8')
