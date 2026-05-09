// boids_pso_cuda_optimized.cu
// Оптимизированная версия Boids‑PSO с глобальными средними (O(N·dim)).
// Главные оптимизации:
// 1. Память переведена в раскладку SoA (dim × particles) — доступ к глобальной памяти
//    становится коалесцированным во всех основных циклах.
// 2. Средние по рою вычисляются полностью на GPU за один проход на каждое измерение,
//    без копирования частичных сумм на хост.
// 3. Убран расход разделяемой памяти O(blockDim * dim) в редукции средних;
//    теперь используется стандартная редукция с O(blockDim) shared-памяти.
// 4. Ядра обновления частиц максимально легковесны и не содержат больших локальных массивов.
// 5. Добавлены проверки ошибок CUDA.

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <unordered_map>
#include <corecrt_math_defines.h>

// ----------------------------------------------------------------------
// Макрос для проверки ошибок CUDA
// ----------------------------------------------------------------------
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":"       \
                      << __LINE__ << " - " << cudaGetErrorString(err) \
                      << " (" << #call << ")" << std::endl;       \
            exit(EXIT_FAILURE);                                    \
        }                                                           \
    } while (0)

// ----------------------------------------------------------------------
// Целевые функции (double)
// ----------------------------------------------------------------------
__device__ double rastrigin(const double* x, int dim, int particles, int idx) {
    double s = 10.0 * dim;
    for (int j = 0; j < dim; ++j) {
        double val = x[j * particles + idx];
        s += val * val - 10.0 * cos(2.0 * M_PI * val);
    }
    return s;
}

__device__ double rosenbrock(const double* x, int dim, int particles, int idx) {
    double s = 0.0;
    for (int j = 0; j < dim - 1; ++j) {
        double xj = x[j * particles + idx];
        double xj1 = x[(j + 1) * particles + idx];
        double t1 = xj1 - xj * xj;
        double t2 = 1.0 - xj;
        s += 100.0 * t1 * t1 + t2 * t2;
    }
    return s;
}

__device__ double ackley(const double* x, int dim, int particles, int idx) {
    double sum1 = 0.0, sum2 = 0.0;
    for (int j = 0; j < dim; ++j) {
        double val = x[j * particles + idx];
        sum1 += val * val;
        sum2 += cos(2.0 * M_PI * val);
    }
    double inv = 1.0 / dim;
    return -20.0 * exp(-0.2 * sqrt(inv * sum1))
           - exp(inv * sum2) + 20.0 + M_E;
}

enum class FuncID { Rastrigin = 0, Rosenbrock, Ackley };
FuncID get_func_id(const std::string& name) {
    if (name == "rosenbrock") return FuncID::Rosenbrock;
    if (name == "ackley") return FuncID::Ackley;
    return FuncID::Rastrigin;
}

__device__ double compute_fitness(const double* x, int dim, int particles, int idx, int func_id) {
    switch (func_id) {
        case 0: return rastrigin(x, dim, particles, idx);
        case 1: return rosenbrock(x, dim, particles, idx);
        case 2: return ackley(x, dim, particles, idx);
        default: return rastrigin(x, dim, particles, idx);
    }
}

// ----------------------------------------------------------------------
// Границы по умолчанию
// ----------------------------------------------------------------------
struct Bounds { double lb, ub; };
std::unordered_map<std::string, Bounds> func_bounds = {
    {"rastrigin", {-5.12, 5.12}},
    {"rosenbrock",{-2.048, 2.048}},
    {"ackley",    {-32.768, 32.768}}
};

// ----------------------------------------------------------------------
// Инициализация частиц (SoA‑раскладка)
// ----------------------------------------------------------------------
__global__ void init_kernel(double* X, double* V, double* pbest_pos, double* pbest_val,
                            int dim, int particles, double lower, double upper,
                            unsigned int seed, int func_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particles) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);
    double range = upper - lower;

    // Каждая нить инициализирует одну частицу, записывая все её измерения
    for (int j = 0; j < dim; ++j) {
        double x_val = lower + curand_uniform_double(&state) * range;
        double v_val = curand_uniform_double(&state) * 2.0 - 1.0;
        X[j * particles + i] = x_val;
        V[j * particles + i] = v_val;
        pbest_pos[j * particles + i] = x_val;
    }
    pbest_val[i] = compute_fitness(X, dim, particles, i, func_id);
}

// ----------------------------------------------------------------------
// Параллельная редукция сумм для всех частиц вдоль одного измерения.
// Запускается с dim блоками, каждый обрабатывает своё измерение.
// Вычисляет одновременно сумму X и сумму V и записывает средние в mean_X, mean_V.
// ----------------------------------------------------------------------
__global__ void reduce_mean_per_dim(const double* X, const double* V,
                                    int dim, int particles,
                                    double* mean_X, double* mean_V)
{
    int j = blockIdx.x;               // номер измерения
    if (j >= dim) return;

    const double* x_ptr = X + j * particles;
    const double* v_ptr = V + j * particles;
    double sum_x = 0.0, sum_v = 0.0;

    // Каждая нить суммирует свою часть элементов
    for (int i = threadIdx.x; i < particles; i += blockDim.x) {
        sum_x += x_ptr[i];
        sum_v += v_ptr[i];
    }

    // Редукция внутри блока
    extern __shared__ double shared[];
    double* s_x = shared;
    double* s_v = shared + blockDim.x;
    s_x[threadIdx.x] = sum_x;
    s_v[threadIdx.x] = sum_v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_x[threadIdx.x] += s_x[threadIdx.x + s];
            s_v[threadIdx.x] += s_v[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        mean_X[j] = s_x[0] / particles;
        mean_V[j] = s_v[0] / particles;
    }
}

// ----------------------------------------------------------------------
// Обновление частиц с использованием глобальных средних (Boids + PSO)
// ----------------------------------------------------------------------
__global__ void update_particles_global(double* X, double* V,
                                        double* pbest_pos, double* pbest_val,
                                        int dim, int particles,
                                        const double* __restrict__ gbest_pos,
                                        const double* __restrict__ mean_X,
                                        const double* __restrict__ mean_V,
                                        double lower, double upper,
                                        double w, double c1, double c2,
                                        double beta, double gamma,
                                        unsigned int seed, int func_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particles) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);

    double range = upper - lower;
    double max_vel_coeff = 0.2;

    for (int j = 0; j < dim; ++j) {
        // Загрузка координат (коалесцированный доступ: соседние нити читают соседние элементы)
        double x_ij = X[j * particles + i];
        double v_ij = V[j * particles + i];
        double p_ij = pbest_pos[j * particles + i];

        // Глобальные средние и лучший
        double mean_X_j = mean_X[j];
        double mean_V_j = mean_V[j];
        double gbest_j  = gbest_pos[j];

        // Boids‑компоненты (разделение (separation) отключено, оставлены alignment и cohesion)
        double alignment = mean_V_j - v_ij;
        double cohesion  = mean_X_j - x_ij;

        double r1 = curand_uniform_double(&state);
        double r2 = curand_uniform_double(&state);

        v_ij = w * v_ij
             + c1 * r1 * (p_ij - x_ij)
             + c2 * r2 * (gbest_j - x_ij)
             + beta  * alignment
             + gamma * cohesion;

        // Ограничение скорости
        double max_vel = max_vel_coeff * range;
        v_ij = fmin(fmax(v_ij, -max_vel), max_vel);

        // Обновление позиции с отражением от границ
        double new_x = x_ij + v_ij;
        if (new_x < lower) { new_x = lower; v_ij = 0.0; }
        if (new_x > upper) { new_x = upper; v_ij = 0.0; }

        X[j * particles + i] = new_x;
        V[j * particles + i] = v_ij;
    }

    // Обновление персонального лучшего
    double new_val = compute_fitness(X, dim, particles, i, func_id);
    if (new_val < pbest_val[i]) {
        pbest_val[i] = new_val;
        for (int j = 0; j < dim; ++j)
            pbest_pos[j * particles + i] = X[j * particles + i];
    }
}

// ----------------------------------------------------------------------
// Редукция минимума в блоке (для поиска глобального лучшего)
// ----------------------------------------------------------------------
__global__ void block_reduce_min(const double* pbest_val, int particles,
                                 double* d_block_vals, int* d_block_idxs)
{
    extern __shared__ double s_vals[];
    int* s_idxs = (int*)&s_vals[blockDim.x];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double val = (i < particles) ? pbest_val[i] : INFINITY;
    int idx = i;
    s_vals[tid] = val;
    s_idxs[tid] = idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_vals[tid + s] < s_vals[tid] ||
                (s_vals[tid + s] == s_vals[tid] && s_idxs[tid + s] < s_idxs[tid])) {
                s_vals[tid] = s_vals[tid + s];
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_block_vals[blockIdx.x] = s_vals[0];
        d_block_idxs[blockIdx.x] = s_idxs[0];
    }
}

// ----------------------------------------------------------------------
// Поиск и обновление глобального лучшего (частично на хосте)
// ----------------------------------------------------------------------
void find_and_update_global_best(const double* d_pbest_val, const double* d_X,
                                 double* d_gbest_val, double* d_gbest_pos,
                                 int dim, int particles,
                                 double* d_block_vals, int* d_block_idxs,
                                 int grid_size, int block_size,
                                 double& host_gbest_val, std::vector<double>& host_gbest_pos)
{
    // Редукция по блокам
    block_reduce_min<<<grid_size, block_size, 2 * block_size * sizeof(double)>>>(
        d_pbest_val, particles, d_block_vals, d_block_idxs);

    std::vector<double> block_vals(grid_size);
    std::vector<int> block_idxs(grid_size);
    CUDA_CHECK(cudaMemcpy(block_vals.data(), d_block_vals, grid_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(block_idxs.data(), d_block_idxs, grid_size * sizeof(int), cudaMemcpyDeviceToHost));

    double min_val = block_vals[0];
    int best_idx = block_idxs[0];
    for (int b = 1; b < grid_size; ++b) {
        if (block_vals[b] < min_val) {
            min_val = block_vals[b];
            best_idx = block_idxs[b];
        }
    }

    if (min_val < host_gbest_val) {
        host_gbest_val = min_val;
        // В SoA‑раскладке позиция лучшей частицы разбросана по измерениям:
        //   координата j хранится в d_X[j * particles + best_idx]
        // Копируем её на хост (можно было бы оставить на устройстве, но для единообразия копируем)
        std::vector<double> temp_pos(dim);
        for (int j = 0; j < dim; ++j) {
            CUDA_CHECK(cudaMemcpy(&temp_pos[j],
                                  d_X + j * particles + best_idx,
                                  sizeof(double), cudaMemcpyDeviceToHost));
        }
        host_gbest_pos = temp_pos;
    }

    CUDA_CHECK(cudaMemcpy(d_gbest_val, &host_gbest_val, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gbest_pos, host_gbest_pos.data(), dim * sizeof(double), cudaMemcpyHostToDevice));
}

// ----------------------------------------------------------------------
// Основная процедура Boids‑PSO (глобальные средние, SoA‑раскладка)
// ----------------------------------------------------------------------
void boids_pso_optimized(int dim, int particles, int iterations,
                         double w, double c1, double c2,
                         double beta, double gamma,
                         double lower, double upper,
                         std::vector<double>& best_pos, double& best_val,
                         std::vector<double>& history, unsigned int seed,
                         const std::string& func_name)
{
    // ---------- Выделение памяти ----------
    // Все векторы частиц хранятся как [dim][particles]
    double *d_X, *d_V, *d_pbest_pos, *d_pbest_val;
    CUDA_CHECK(cudaMalloc(&d_X, dim * particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, dim * particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pbest_pos, dim * particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pbest_val, particles * sizeof(double)));

    double *d_gbest_pos, *d_gbest_val;
    CUDA_CHECK(cudaMalloc(&d_gbest_pos, dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gbest_val, sizeof(double)));

    // Средние на устройстве
    double *d_mean_X, *d_mean_V;
    CUDA_CHECK(cudaMalloc(&d_mean_X, dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mean_V, dim * sizeof(double)));

    // Временные массивы для редукции минимума
    int block_size_min = 256;
    int grid_size_min = (particles + block_size_min - 1) / block_size_min;
    double *d_block_vals;
    int *d_block_idxs;
    CUDA_CHECK(cudaMalloc(&d_block_vals, grid_size_min * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_idxs, grid_size_min * sizeof(int)));

    int func_id = static_cast<int>(get_func_id(func_name));

    // ---------- Инициализация ----------
    init_kernel<<<grid_size_min, block_size_min>>>(
        d_X, d_V, d_pbest_pos, d_pbest_val,
        dim, particles, lower, upper, seed, func_id);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Первичный выбор глобального лучшего
    double host_gbest_val = INFINITY;
    std::vector<double> host_gbest_pos(dim);
    find_and_update_global_best(d_pbest_val, d_X,
                                d_gbest_val, d_gbest_pos,
                                dim, particles,
                                d_block_vals, d_block_idxs,
                                grid_size_min, block_size_min,
                                host_gbest_val, host_gbest_pos);

    // Параметры запуска для редукции средних (один блок на измерение)
    int block_size_mean = 256;   // можно увеличить до 512/1024 при наличии ресурсов
    // shared memory: 2 * block_size_mean * sizeof(double)
    int shared_mem_mean = 2 * block_size_mean * sizeof(double);

    // ---------- Главный цикл ----------
    for (int t = 0; t < iterations; ++t) {
        // 1. Вычисление глобальных средних X и V полностью на GPU
        reduce_mean_per_dim<<<dim, block_size_mean, shared_mem_mean>>>(
            d_X, d_V, dim, particles, d_mean_X, d_mean_V);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2. Обновление частиц
        update_particles_global<<<grid_size_min, block_size_min>>>(
            d_X, d_V, d_pbest_pos, d_pbest_val,
            dim, particles, d_gbest_pos,
            d_mean_X, d_mean_V,
            lower, upper, w, c1, c2,
            beta, gamma, seed + t + 1, func_id);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. Поиск нового глобального лучшего
        find_and_update_global_best(d_pbest_val, d_X,
                                    d_gbest_val, d_gbest_pos,
                                    dim, particles,
                                    d_block_vals, d_block_idxs,
                                    grid_size_min, block_size_min,
                                    host_gbest_val, host_gbest_pos);

        history[t] = host_gbest_val;
    }

    best_val = host_gbest_val;
    best_pos = host_gbest_pos;

    // ---------- Очистка ----------
    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_pbest_pos)); CUDA_CHECK(cudaFree(d_pbest_val));
    CUDA_CHECK(cudaFree(d_gbest_pos)); CUDA_CHECK(cudaFree(d_gbest_val));
    CUDA_CHECK(cudaFree(d_mean_X)); CUDA_CHECK(cudaFree(d_mean_V));
    CUDA_CHECK(cudaFree(d_block_vals)); CUDA_CHECK(cudaFree(d_block_idxs));
}

// ----------------------------------------------------------------------
// main
// ----------------------------------------------------------------------
int main(int argc, char** argv) {
    int dim = 30, particles = 500, iterations = 500, file = 0;
    double w = 0.7, c1 = 1.5, c2 = 1.5;
    double beta = 0.01, gamma = 0.01;
    unsigned int seed = 42;
    std::string func = "rastrigin";
    double lb = INFINITY, ub = INFINITY;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-dim") == 0 && i+1 < argc) dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "-particles") == 0 && i+1 < argc) particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "-iter") == 0 && i+1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-w") == 0 && i+1 < argc) w = atof(argv[++i]);
        else if (strcmp(argv[i], "-c1") == 0 && i+1 < argc) c1 = atof(argv[++i]);
        else if (strcmp(argv[i], "-c2") == 0 && i+1 < argc) c2 = atof(argv[++i]);
        else if (strcmp(argv[i], "-beta") == 0 && i+1 < argc) beta = atof(argv[++i]);
        else if (strcmp(argv[i], "-gamma") == 0 && i+1 < argc) gamma = atof(argv[++i]);
        else if (strcmp(argv[i], "-seed") == 0 && i+1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "-func") == 0 && i+1 < argc) func = argv[++i];
        else if (strcmp(argv[i], "-lb") == 0 && i+1 < argc) lb = atof(argv[++i]);
        else if (strcmp(argv[i], "-ub") == 0 && i+1 < argc) ub = atof(argv[++i]);
        else if (strcmp(argv[i], "-file") == 0 && i+1 < argc) file = atoi(argv[++i]);
    }
    if (lb == INFINITY) lb = func_bounds[func].lb;
    if (ub == INFINITY) ub = func_bounds[func].ub;

    std::cout << "Optimized Boids-PSO (SoA, dim=" << dim
              << ", particles=" << particles << ", iter=" << iterations << ")\n";

    std::vector<double> best_pos(dim);
    double best_val;
    std::vector<double> history(iterations);

    auto start = std::chrono::high_resolution_clock::now();
    boids_pso_optimized(dim, particles, iterations, w, c1, c2,
                        beta, gamma, lb, ub,
                        best_pos, best_val, history, seed, func);
    auto end = std::chrono::high_resolution_clock::now();

    double time_sec = std::chrono::duration<double>(end - start).count();
    std::cout << "CUDA_TIME: " << time_sec << std::endl;
    std::cout << "BEST_VALUE: " << best_val << std::endl;

    if (file != 0) {
        char fname[256];
        snprintf(fname, sizeof(fname),
                 "boids_pso_optimized_%s_d%d_p%d_i%d_w%.2f_b%.4f_g%.4f_seed%u.txt",
                 func.c_str(), dim, particles, iterations, w, beta, gamma, seed);
        std::ofstream f(fname);
        f << "iteration,best_value\n";
        for (int i = 0; i < iterations; ++i)
            f << i << "," << history[i] << "\n";
        f.close();
    }

    return 0;
}

// Компиляция:
// nvcc -O2 boids_pso_cuda_global.cu -o boids_pso_cuda_global.exe -lcurand