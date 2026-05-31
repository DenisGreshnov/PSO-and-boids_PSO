# plot_results.py (обновлённый)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------- настройка стиля ----------
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ---------- загрузка данных ----------
df = pd.read_csv("benchmark_results.csv")
df = df.dropna(subset=['time_sec', 'best_value'])
df[['alpha','beta','gamma','r_neigh']] = df[['alpha','beta','gamma','r_neigh']].fillna(-1)

# ---------- усреднение по seed ----------
group_keys = ['exe','type','dim','func','particles','iter','alpha','beta','gamma','r_neigh']
df_agg = df.groupby(group_keys, as_index=False).agg(
    time_sec_mean=('time_sec', 'mean'),
    time_sec_std=('time_sec', 'std'),
    best_value_mean=('best_value', 'mean'),
    best_value_std=('best_value', 'std')
)

# ---------- ФИЛЬТРАЦИЯ GLOBAL по лучшим гиперпараметрам ----------
BEST_GLOBAL_BETA = 0.01
BEST_GLOBAL_GAMMA = 0.2
# Для global оставляем только указанные beta/gamma (если они есть в данных)
mask_global = (df_agg['type'] == 'global')
# Если параметры не -1, то проверяем; иначе global без гиперпараметров 
if not df_agg[mask_global].empty:
    # Определим, какие значения beta/gamma реально есть
    global_betas = df_agg.loc[mask_global, 'beta'].unique()
    global_gammas = df_agg.loc[mask_global, 'gamma'].unique()
    # Оставляем только строки, где beta/gamma близки к лучшим (или -1, если параметры не поддерживаются)
    cond = (mask_global) & (
        ((df_agg['beta'].isin([BEST_GLOBAL_BETA, -1])) & (df_agg['gamma'].isin([BEST_GLOBAL_GAMMA, -1])))
    )
    df_agg = df_agg[~mask_global | cond]  # удаляем global-строки, не удовлетворяющие условию

# ---------- вспомогательная функция для speedup ----------
def get_speedup(data, type_str):
    cpu = data[data['exe'].str.contains('cpu', case=False) & (data['type'] == type_str)].copy()
    gpu = data[data['exe'].str.contains('cuda', case=False) & (data['type'] == type_str)].copy()
    if cpu.empty or gpu.empty:
        return pd.DataFrame()
    merge_keys = ['dim','func','particles','iter','alpha','beta','gamma','r_neigh']
    merged = pd.merge(cpu, gpu, on=merge_keys, suffixes=('_cpu','_gpu'))
    merged['speedup'] = merged['time_sec_mean_cpu'] / merged['time_sec_mean_gpu']
    return merged[['dim','func','particles','speedup','time_sec_mean_cpu','time_sec_mean_gpu']]

# ---------- 1. Ускорение GPU vs CPU (тепловые карты) ----------
def plot_speedup_heatmaps(df_agg):
    out_dir = Path("plots/speedup")
    out_dir.mkdir(parents=True, exist_ok=True)
    for algo_type in ['pso', 'boids', 'global']:
        sp = get_speedup(df_agg, algo_type)
        if sp.empty:
            continue
        for func in sp['func'].unique():
            sp_func = sp[sp['func'] == func]
            if sp_func.empty:
                continue
            pivot = sp_func.pivot_table(values='speedup', index='dim', columns='particles', aggfunc='mean')
            plt.figure()
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Ускорение (CPU/GPU)'})
            plt.title(f"Ускорение – {algo_type} – {func}")
            plt.xlabel("Число частиц")
            plt.ylabel("Размерность")
            plt.tight_layout()
            plt.savefig(out_dir / f"speedup_{algo_type}_{func}.png")
            plt.close()

# ---------- 2. Сравнение времени CPU и GPU ----------
def plot_time_comparison(df_agg):
    out_dir = Path("plots/time_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    for func in df_agg['func'].unique():
        func_data = df_agg[df_agg['func'] == func]
        for dim in [10, 30, 100]:
            sub = func_data[func_data['dim'] == dim]
            if sub.empty:
                continue
            plt.figure()
            for algo_type in ['pso', 'boids', 'global']:
                cpu_line = sub[(sub['type'] == algo_type) & (sub['exe'].str.contains('cpu', case=False))]
                gpu_line = sub[(sub['type'] == algo_type) & (sub['exe'].str.contains('cuda', case=False))]
                if not cpu_line.empty:
                    plt.plot(cpu_line['particles'], cpu_line['time_sec_mean'], 'o--', label=f'{algo_type} CPU')
                if not gpu_line.empty:
                    plt.plot(gpu_line['particles'], gpu_line['time_sec_mean'], 's-', label=f'{algo_type} GPU')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Число частиц')
            plt.ylabel('Время (с)')
            plt.title(f'Время выполнения – {func} – dim={dim}')
            plt.legend()
            plt.grid(True, which='both', ls='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(out_dir / f"time_{func}_dim{dim}.png")
            plt.close()

# ---------- 3. Качество оптимизации (barplot) ----------
def plot_best_values(df_agg):
    out_dir = Path("plots/quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = (df_agg['dim'] == 30) & (df_agg['particles'] == 500)
    sub = df_agg[mask].copy()
    if sub.empty:
        return
    sub['algo_label'] = sub['type'] + '_' + sub['exe'].apply(lambda x: 'CPU' if 'cpu' in x else 'GPU')
    for func in sub['func'].unique():
        plt.figure()
        func_sub = sub[sub['func'] == func]
        sns.barplot(data=func_sub, x='algo_label', y='best_value_mean', capsize=.1)
        plt.xticks(rotation=45)
        plt.ylabel('Среднее лучшее значение')
        plt.title(f'Качество оптимизации – {func} (dim=30, particles=500)')
        plt.tight_layout()
        plt.savefig(out_dir / f"quality_{func}_dim30_p500.png")
        plt.close()

# ---------- 4. Тепловая карта гиперпараметров global (если ещё не отфильтрована) ----------
def plot_global_hyperparams(df_agg):
    out_dir = Path("plots/hyperparams_global")
    out_dir.mkdir(parents=True, exist_ok=True)
    global_data = df_agg[df_agg['type'] == 'global'].copy()
    if global_data.empty:
        return
    for func in global_data['func'].unique():
        func_global = global_data[global_data['func'] == func]
        heat = func_global.groupby(['beta','gamma'])['best_value_mean'].mean().reset_index()
        if heat.shape[0] < 4:
            continue
        pivot = heat.pivot(index='gamma', columns='beta', values='best_value_mean')
        plt.figure()
        sns.heatmap(pivot, annot=True, fmt=".3g", cmap="viridis_r")
        plt.title(f"Качество (best value) – global – {func}")
        plt.xlabel("beta")
        plt.ylabel("gamma")
        plt.tight_layout()
        plt.savefig(out_dir / f"global_hyper_{func}.png")
        plt.close()

# ---------- сравнение PSO vs Global по размерности ----------
def plot_pso_vs_global(df_agg):
    out_dir = Path("plots/pso_vs_global")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Выбираем только pso и global (отфильтрованные), убираем alpha,beta,gamma из группировки
    compare = df_agg[df_agg['type'].isin(['pso','global'])].copy()
    if compare.empty:
        return

    # Для одинаковых частиц и итераций
    particles_fixed = 500
    sub = compare[compare['particles'] == particles_fixed]
    if sub.empty:
        return

    # Группируем для каждого типа, dim, func (усредняем по seed, параметры глобал уже фиксированы)
    grouped = sub.groupby(['type','dim','func','exe'], as_index=False).agg(
        time_sec=('time_sec_mean', 'mean'),
        best_val=('best_value_mean', 'mean')
    )

    # Разделяем на CPU и GPU
    for func in grouped['func'].unique():
        func_data = grouped[grouped['func'] == func]
        for metric, ylabel, fname_prefix in [
            ('best_val', 'Среднее лучшее значение', 'quality'),
            ('time_sec', 'Время (с)', 'time')
        ]:
            plt.figure()
            # Отдельные линии для pso CPU/GPU и global CPU/GPU
            for algo_type in ['pso','global']:
                cpu_data = func_data[(func_data['type'] == algo_type) & (func_data['exe'].str.contains('cpu'))]
                gpu_data = func_data[(func_data['type'] == algo_type) & (func_data['exe'].str.contains('cuda'))]
                if not cpu_data.empty:
                    plt.plot(cpu_data['dim'], cpu_data[metric], 'o--', label=f'{algo_type} CPU')
                if not gpu_data.empty:
                    plt.plot(gpu_data['dim'], gpu_data[metric], 's-', label=f'{algo_type} GPU')
            plt.xlabel('Размерность')
            plt.ylabel(ylabel)
            plt.title(f'{func} – {ylabel} от размерности (particles={particles_fixed})')
            plt.legend()
            plt.grid(True, which='both', ls='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(out_dir / f"{fname_prefix}_{func}_dim_scaling.png")
            plt.close()

    # Дополнительно: нормированное улучшение global над pso для GPU (качество)
    for func in grouped['func'].unique():
        func_data = grouped[grouped['func'] == func]
        # Берём только GPU версии
        pso_gpu = func_data[(func_data['type'] == 'pso') & (func_data['exe'].str.contains('cuda'))].set_index('dim')
        global_gpu = func_data[(func_data['type'] == 'global') & (func_data['exe'].str.contains('cuda'))].set_index('dim')
        if pso_gpu.empty or global_gpu.empty:
            continue
        common_dims = pso_gpu.index.intersection(global_gpu.index)
        if len(common_dims) < 2:
            continue
        improvement = (pso_gpu.loc[common_dims, 'best_val'] - global_gpu.loc[common_dims, 'best_val']) / pso_gpu.loc[common_dims, 'best_val'] * 100
        plt.figure()
        plt.plot(common_dims, improvement, 'D-', color='green')
        plt.xlabel('Размерность')
        plt.ylabel('Улучшение качества (%)')
        plt.title(f'{func} – улучшение global над PSO (GPU, particles={particles_fixed})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"improvement_{func}_GPU.png")
        plt.close()

# ---------- Запуск ----------
if __name__ == "__main__":
    print("Строим графики...")
    plot_speedup_heatmaps(df_agg)
    plot_time_comparison(df_agg)
    plot_best_values(df_agg)
    plot_global_hyperparams(df_agg)      # теперь покажет только одну точку? лучше убрать, если параметры зафиксированы
    plot_pso_vs_global(df_agg)
    print("Готово. Результаты в папке plots/")