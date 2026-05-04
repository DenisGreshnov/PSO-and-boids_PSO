// pso_cuda_bench.cu
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
#include <functional>
#include <corecrt_math_defines.h>

// Функция Растригина (double precision)
__device__ double rastrigin(const double* x, int dim) {
    double s = 10.0 * dim;
    for (int i = 0; i < dim; ++i)
        s += x[i] * x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
    return s;
}

__device__ double rosenbrock(const double* x, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim - 1; ++i) {
        double t1 = x[i+1] - x[i]*x[i];
        double t2 = 1.0 - x[i];
        s += 100.0 * t1*t1 + t2*t2;
    }
    return s;
}

__device__ double ackley(const double* x, int dim) {
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < dim; ++i) {
        sum1 += x[i] * x[i];
        sum2 += cos(2.0 * M_PI * x[i]);
    }
    double inv = 1.0 / dim;
    return -20.0 * exp(-0.2 * sqrt(inv * sum1))
           - exp(inv * sum2) + 20.0 + M_E;
}

enum class FuncID { Rastrigin = 0, Rosenbrock, Ackley };
FuncID get_func_id(const std::string& name) {
    if (name == "rosenbrock") return FuncID::Rosenbrock;
    if (name == "ackley") return FuncID::Ackley;
    return FuncID::Rastrigin; // по умолчанию
}

__device__ double compute_fitness(const double* x, int dim, int func_id) {
    switch (func_id) {
        case 0: return rastrigin(x, dim);
        case 1: return rosenbrock(x, dim);
        case 2: return ackley(x, dim);
        default: return rastrigin(x, dim);
    }
}

struct Bounds { double lb, ub; };
std::unordered_map<std::string, Bounds> func_bounds = {
    {"rastrigin", {-5.12, 5.12}},
    {"rosenbrock",{-2.048, 2.048}},
    {"ackley",    {-32.768, 32.768}}
};

// Ядро инициализации частиц
__global__ void init_particles(double* X, double* V, double* pbest_pos, double* pbest_val,
                               int dim, int particles, double lower, double upper, unsigned int seed, int func_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    double* x = X + idx * dim;
    double* v = V + idx * dim;
    double* p = pbest_pos + idx * dim;
    double range = upper - lower;

    for (int j = 0; j < dim; ++j) {
        x[j] = lower + curand_uniform_double(&state) * range;
        v[j] = curand_uniform_double(&state) * 2.0 - 1.0; // [-1, 1]
        p[j] = x[j];
    }
    pbest_val[idx] = compute_fitness(x, dim, func_id);
}

// Ядро обновления частиц и персональных лучших
__global__ void update_particles(double* X, double* V, double* pbest_pos, double* pbest_val,
                                 int dim, int particles,
                                 const double* __restrict__ gbest_pos, double lower, double upper,
                                 double w, double c1, double c2, unsigned int seed, int func_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particles) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    double* x = X + idx * dim;
    double* v = V + idx * dim;
    double* p = pbest_pos + idx * dim;
    double max_vel_coeff = 0.2;

    for (int j = 0; j < dim; ++j) {
        double r1 = curand_uniform_double(&state);
        double r2 = curand_uniform_double(&state);
        v[j] = w * v[j] + c1 * r1 * (p[j] - x[j]) + c2 * r2 * (gbest_pos[j] - x[j]);

        double max_vel = max_vel_coeff * (upper - lower);
        v[j] = fmin(fmax(v[j], -max_vel), max_vel);

        x[j] += v[j];
        // Отражающие границы с занулением скорости
        if (x[j] < lower) { x[j] = lower; v[j] = 0.0; }
        if (x[j] > upper) { x[j] = upper; v[j] = 0.0; }
    }

    double new_val = compute_fitness(x, dim, func_id);
    if (new_val < pbest_val[idx]) {
        pbest_val[idx] = new_val;
        for (int j = 0; j < dim; ++j)
            p[j] = x[j];
    }
}

// Ядро частичной редукции: находит минимум в своём блоке и записывает результат в d_block_vals / d_block_idxs
__global__ void block_reduce(const double* pbest_val, int particles,
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

// Вспомогательная функция поиска глобального минимума: запускает редукцию, доделывает на хосте, обновляет gbest_val и копирует позицию
void find_and_update_global_best(const double* d_pbest_val, const double* d_X,
                                 double* d_gbest_val, double* d_gbest_pos,
                                 int dim, int particles,
                                 double* d_block_vals, int* d_block_idxs,
                                 int grid_size, int block_size,
                                 double& host_gbest_val, std::vector<double>& host_gbest_pos)
{
    // Редукция по блокам
    block_reduce<<<grid_size, block_size, 2 * block_size * sizeof(double)>>>(d_pbest_val, particles,
                                                                             d_block_vals, d_block_idxs);

    // Копируем результаты блоков на хост
    std::vector<double> block_vals(grid_size);
    std::vector<int> block_idxs(grid_size);
    cudaMemcpy(block_vals.data(), d_block_vals, grid_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_idxs.data(), d_block_idxs, grid_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Поиск минимума на хосте
    double min_val = block_vals[0];
    int best_idx = block_idxs[0];
    for (int i = 1; i < grid_size; ++i) {
        if (block_vals[i] < min_val) {
            min_val = block_vals[i];
            best_idx = block_idxs[i];
        }
    }

    // Обновляем глобальный лучший, если он улучшился (или при инициализации)
    if (min_val < host_gbest_val) {
        host_gbest_val = min_val;
        // Копируем позицию лучшей частицы с устройства
        cudaMemcpy(host_gbest_pos.data(), d_X + best_idx * dim, dim * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Записываем текущее значение на устройство (для использования в ядре update)
    cudaMemcpy(d_gbest_val, &host_gbest_val, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gbest_pos, host_gbest_pos.data(), dim * sizeof(double), cudaMemcpyHostToDevice);
}

// Основная процедура PSO на GPU
void pso_cuda(int dim, int particles, int iterations,
              double w, double c1, double c2,
              double lower, double upper,
              std::vector<double>& best_pos, double& best_val,
              std::vector<double>& history, unsigned int seed,
              const std::string& func_name)
{
    // Выделение памяти на устройстве
    double *d_X, *d_V, *d_pbest_pos, *d_pbest_val;
    double *d_gbest_pos, *d_gbest_val;
    cudaMalloc(&d_X, particles * dim * sizeof(double));
    cudaMalloc(&d_V, particles * dim * sizeof(double));
    cudaMalloc(&d_pbest_pos, particles * dim * sizeof(double));
    cudaMalloc(&d_pbest_val, particles * sizeof(double));
    cudaMalloc(&d_gbest_pos, dim * sizeof(double));
    cudaMalloc(&d_gbest_val, sizeof(double));

    // Временные массивы для редукции
    int block_size = 256;
    int grid_size = (particles + block_size - 1) / block_size;
    double *d_block_vals; int *d_block_idxs;
    cudaMalloc(&d_block_vals, grid_size * sizeof(double));
    cudaMalloc(&d_block_idxs, grid_size * sizeof(int));

    int func_id = static_cast<int>(get_func_id(func_name));

    // Инициализация частиц
    init_particles<<<grid_size, block_size>>>(d_X, d_V, d_pbest_pos, d_pbest_val,
                                              dim, particles, lower, upper, seed, func_id);
    cudaDeviceSynchronize();

    // Глобальный лучший (хост-копия)
    double gbest_val_host = INFINITY;
    std::vector<double> gbest_pos_host(dim);

    // Первичный поиск глобального лучшего
    find_and_update_global_best(d_pbest_val, d_X, d_gbest_val, d_gbest_pos,
                                dim, particles, d_block_vals, d_block_idxs,
                                grid_size, block_size, gbest_val_host, gbest_pos_host);

    // Главный цикл
    for (int t = 0; t < iterations; ++t) {
        // Обновление скоростей и позиций
        update_particles<<<grid_size, block_size>>>(d_X, d_V, d_pbest_pos, d_pbest_val,
                                                    dim, particles, d_gbest_pos, lower, upper,
                                                    w, c1, c2, seed + t + 1, func_id);
        cudaDeviceSynchronize();

        // Поиск нового глобального лучшего среди всех персональных лучших
        find_and_update_global_best(d_pbest_val, d_X, d_gbest_val, d_gbest_pos,
                                    dim, particles, d_block_vals, d_block_idxs,
                                    grid_size, block_size, gbest_val_host, gbest_pos_host);

        // Запись в историю
        history[t] = gbest_val_host;
    }

    // Финальный результат
    best_val = gbest_val_host;
    best_pos = gbest_pos_host;

    // Очистка памяти
    cudaFree(d_X); cudaFree(d_V); cudaFree(d_pbest_pos); cudaFree(d_pbest_val);
    cudaFree(d_gbest_pos); cudaFree(d_gbest_val);
    cudaFree(d_block_vals); cudaFree(d_block_idxs);
}

int main(int argc, char** argv) {
    int dim = 30;
    int particles = 500;
    int iterations = 500;
    int file = 0;
    double w = 0.7, c1 = 1.5, c2 = 1.5;
    unsigned int seed = 42;
    std::string func = "rastrigin";
    double lb = INFINITY, ub = INFINITY;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-dim") == 0 && i+1 < argc) dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "-particles") == 0 && i+1 < argc) particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "-iter") == 0 && i+1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-w") == 0 && i+1 < argc) w = atof(argv[++i]);
        else if (strcmp(argv[i], "-c1") == 0 && i+1 < argc) c1 = atof(argv[++i]);
        else if (strcmp(argv[i], "-c2") == 0 && i+1 < argc) c2 = atof(argv[++i]);
        else if (strcmp(argv[i], "-seed") == 0 && i+1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "-func") == 0 && i+1 < argc) func = argv[++i];
        else if (strcmp(argv[i], "-lb") == 0 && i+1 < argc) lb = atof(argv[++i]);
        else if (strcmp(argv[i], "-ub") == 0 && i+1 < argc) ub = atof(argv[++i]);
        else if (strcmp(argv[i], "-file") == 0 && i+1 < argc) file = atoi(argv[++i]);
    }
    if (lb == INFINITY) lb = func_bounds[func].lb;
    if (ub == INFINITY) ub = func_bounds[func].ub;


    std::vector<double> best_pos;
    double best_val;
    std::vector<double> history(iterations);

    auto start = std::chrono::high_resolution_clock::now();
    pso_cuda(dim, particles, iterations, w, c1, c2, lb, ub, best_pos, best_val, history, seed, func);
    auto end = std::chrono::high_resolution_clock::now();

    double time_sec = std::chrono::duration<double>(end - start).count();
    std::cout << "CUDA_TIME: " << time_sec << std::endl;
    std::cout << "BEST_VALUE: " << best_val << std::endl;

    if (file != 0){
        char fname[256];
        snprintf(fname, sizeof(fname), "pso_cuda_%s_d%d_p%d_i%d_w%.2f_c1%.2f_c2%.2f_seed%u.txt",
                 func.c_str(), dim, particles, iterations, w, c1, c2, seed);
        std::ofstream f(fname);
        f << "iteration,best_value\n";
        for (int i = 0; i < iterations; ++i)
            f << i << "," << history[i] << "\n";
        f.close();
    }

    return 0;
}
//компиляция nvcc -O2 pso_cuda_bench.cu -o pso_cuda_bench.exe -lcurand