import subprocess
import os
import csv
import itertools
import time
import re

# ----------------------------------------------------------------------
#  Конфигурация
# ----------------------------------------------------------------------

# Базовые параметры, общие для всех программ
DIM_VALUES = [10, 30, 50, 100, 200]                  #размерности
FUNC_VALUES = ["rastrigin", "rosenbrock", "ackley"]
SEEDS = list(range(33, 33*10+1, 33))                 #повторные запуски для усреднения

# Определяем программы и их особенности
# Каждая программа описывается:
#   - exe: имя исполняемого файла
#   - types: "pso" | "boids" | "global"
#   - can_alpha: True/False (принимает ли -alpha)
#   - can_rneigh: True/False (принимает ли -r_neigh)
#   - speed: "fast" | "slow" (для подбора числа частиц и итераций)

PROGRAMS = [
    {"exe": "pso_cpu_bench.exe",       "type": "pso",    "alpha": False, "rneigh": False, "speed": "slow"},
    {"exe": "pso_cuda_bench.exe",      "type": "pso",    "alpha": False, "rneigh": False, "speed": "fast"},
    {"exe": "boids_pso_cpu.exe",       "type": "boids",  "alpha": True,  "rneigh": True,  "speed": "very slow"},
    {"exe": "boids_pso_cuda.exe",      "type": "boids",  "alpha": True,  "rneigh": True,  "speed": "very slow"},
    {"exe": "boids_pso_cpu_global.exe","type": "global", "alpha": False, "rneigh": False, "speed": "slow"},
    {"exe": "boids_pso_cuda_global.exe","type": "global","alpha": False, "rneigh": False, "speed": "fast"},
]

# Параметры для разных типов
PARTICLES_FAST = [100, 500, 1000, 5000, 50000]
ITER_FAST = [300]

PARTICLES_SLOW = [50, 100, 500, 1000]   # чтобы не ждать часами
ITER_SLOW = [200]

PARTICLES_VERY_SLOW = [50, 100]
ITER_VERY_SLOW = [200]

# Для boids-версий, которые принимают alpha/beta/gamma
ALPHA_VALUES = [0.02, 0.07, 0.2]
BETA_VALUES  = [0.01, 0.05, 0.1]
GAMMA_VALUES = [0.01, 0.05, 0.1]
#RNEIGH_VALUES = [2.0, 5.0]    # если нужен (для медленных)

# Для global-версий, которые не принимают alpha, но имеют beta/gamma
BETA_GLOBAL_VALUES = [0.01, 0.02, 0.05, 0.1]
GAMMA_GLOBAL_VALUES = [0.01, 0.02, 0.05, 0.1]

# ----------------------------------------------------------------------
#  Функция запуска и извлечения результатов
# ----------------------------------------------------------------------
def run_and_parse(cmd):
    """Запускает программу, возвращает (время_сек, лучшее_значение)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        stdout = result.stdout
        # Ищем строки вида "CPU_TIME: 12.34" или "CUDA_TIME: 12.34"
        time_match = re.search(r"(?:CPU|CUDA)_TIME:\s+([\d\.]+)", stdout)
        val_match = re.search(r"BEST_VALUE:\s+([\deE\.\-]+)", stdout)
        if time_match and val_match:
            return float(time_match.group(1)), float(val_match.group(1))
        else:
            print(f"Не удалось разобрать вывод команды: {cmd}")
            print(stdout)
            return None, None
    except subprocess.TimeoutExpired:
        print(f"Таймаут (>300 c): {cmd}")
        return None, None
    except Exception as e:
        print(f"Ошибка запуска {cmd}: {e}")
        return None, None

# ----------------------------------------------------------------------
#  Генерация комбинаций параметров для каждой программы
# ----------------------------------------------------------------------
all_runs = []

for prog in PROGRAMS:
    exe = prog["exe"]
    ptype = prog["type"]
    has_alpha = prog["alpha"]
    has_rneigh = prog["rneigh"]
    speed = prog["speed"]

    if speed == "fast":
        particles_list = PARTICLES_FAST
        iter_list = ITER_FAST
    elif speed == "slow":
        particles_list = PARTICLES_SLOW
        iter_list = ITER_SLOW
    elif speed == "very slow":
        particles_list = PARTICLES_VERY_SLOW
        iter_list = ITER_VERY_SLOW

    # Базовый список параметров (всегда присутствуют)
    base_params = {
        "dim": DIM_VALUES,
        "func": FUNC_VALUES,
        "particles": particles_list,
        "iter": iter_list,
        "seed": SEEDS,
    }

    # Дополнительные параметры в зависимости от типа
    if ptype == "pso":
        # нет никаких дополнительных
        extra_params = {}
    elif ptype == "boids":
        extra_params = {
            "alpha": ALPHA_VALUES,
            "beta": BETA_VALUES,
            "gamma": GAMMA_VALUES,
            #"r_neigh": RNEIGH_VALUES if has_rneigh else [None],
        }
    elif ptype == "global":
        extra_params = {
            "beta": BETA_GLOBAL_VALUES,
            "gamma": GAMMA_GLOBAL_VALUES,
        }

    # Объединяем все ключи
    all_keys = list(base_params.keys()) + list(extra_params.keys())
    # Генерируем декартово произведение всех значений
    product_lists = [base_params[k] for k in base_params] + [extra_params[k] for k in extra_params]
    combinations = list(itertools.product(*product_lists))

    for combo in combinations:
        # Строим словарь параметров
        params_dict = dict(zip(all_keys, combo))
        # Формируем аргументы командной строки
        args = [exe]
        args += ["-dim", str(params_dict["dim"])]
        args += ["-func", params_dict["func"]]
        args += ["-particles", str(params_dict["particles"])]
        args += ["-iter", str(params_dict["iter"])]
        args += ["-seed", str(params_dict["seed"])]

        # Границы для разных функций (можно задать автоматически, но для простоты используем -lb/-ub,
        # хотя программы могут иметь значения по умолчанию, но лучше явно указать)
        # Для rosenbrock лучше использовать -2.048 2.048, для ackley -32.768 32.768, для rastrigin -5.12 5.12.
        # Будем задавать явно, чтобы избежать сюрпризов.
        func = params_dict["func"]
        if func == "rosenbrock":
            args += ["-lb", "-2.048", "-ub", "2.048"]
        elif func == "ackley":
            args += ["-lb", "-32.768", "-ub", "32.768"]
        else:
            args += ["-lb", "-5.12", "-ub", "5.12"]

        if "alpha" in params_dict and params_dict["alpha"] is not None:
            args += ["-alpha", str(params_dict["alpha"])]
        if "beta" in params_dict and params_dict["beta"] is not None:
            args += ["-beta", str(params_dict["beta"])]
        if "gamma" in params_dict and params_dict["gamma"] is not None:
            args += ["-gamma", str(params_dict["gamma"])]
        if "r_neigh" in params_dict and params_dict["r_neigh"] is not None and has_rneigh:
            args += ["-r_neigh", str(params_dict["r_neigh"])]

        # Добавляем информацию о запуске
        info = {
            "exe": exe,
            "type": ptype,
            "params": params_dict,
            "cmd": args
        }
        all_runs.append(info)

# ----------------------------------------------------------------------
#  Формирование единого списка полей для CSV
# ----------------------------------------------------------------------
csv_fields = ["exe", "type", "dim", "func", "particles", "iter", "seed",
              "alpha", "beta", "gamma", "r_neigh", "time_sec", "best_value", "wall_time"]

# ----------------------------------------------------------------------
#  Выполнение запусков и сбор данных
# ----------------------------------------------------------------------
results = []

print(f"Всего запланировано запусков: {len(all_runs)}")
for i, run_info in enumerate(all_runs):
    cmd = run_info["cmd"]
    print(f"[{i+1}/{len(all_runs)}] Запуск: {' '.join(cmd)}")
    start_time = time.time()
    time_sec, best_val = run_and_parse(cmd)
    elapsed = time.time() - start_time
    if time_sec is not None and best_val is not None:
        # Создаём словарь только с полями из csv_fields
        entry = {field: None for field in csv_fields}
        entry["exe"] = run_info["exe"]
        entry["type"] = run_info["type"]
        # Параметры из run_info["params"] (dim, func, particles, iter, seed и др.)
        for key, value in run_info["params"].items():
            if key in entry:
                entry[key] = value
        entry["time_sec"] = time_sec
        entry["best_value"] = best_val
        entry["wall_time"] = elapsed
        results.append(entry)
        print(f"  -> time={time_sec:.3f}s, value={best_val:.6e}")
    else:
        print("  -> Ошибка/таймаут, результат не сохранён")

# ----------------------------------------------------------------------
#  Сохранение в CSV
# ----------------------------------------------------------------------
if not results:
    print("Нет успешных результатов.")
    exit()

csv_file = "benchmark_results.csv"
with open(csv_file, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(results)
print(f"\nВсе результаты сохранены в {csv_file}")

# Формируем сводную таблицу: группируем по программе, функции, размерности, числу частиц, итерациям
# и выводим среднее время и лучшее значение по всем seed и дополнительным параметрам.
from collections import defaultdict

aggregated = defaultdict(lambda: {"times": [], "values": []})

for r in results:
    # Ключ агрегации: программа, функция, размерность, частицы, итерации (игнорируем alpha,beta,gamma,seed)
    key = (r["exe"], r["func"], r["dim"], r["particles"], r["iter"])
    aggregated[key]["times"].append(r["time_sec"])
    aggregated[key]["values"].append(r["best_value"])

print("\nСводная таблица средних результатов:\n")
header = f"{'Программа':<30} {'Функция':<12} {'D':>3} {'P':>6} {'Iter':>5} {'Ср.время,с':>12} {'Ср.значение':>15}"
print(header)
print("-" * len(header))

for (exe, func, dim, parts, iters), data in sorted(aggregated.items()):
    avg_time = sum(data["times"]) / len(data["times"])
    avg_val = sum(data["values"]) / len(data["values"])
    print(f"{exe:<30} {func:<12} {dim:>3} {parts:>6} {iters:>5} {avg_time:>12.3f} {avg_val:>15.6e}")

print("\nГотово.")