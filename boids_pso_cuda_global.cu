// boids_pso_global.cu
// Исправленная версия Boids-PSO: вместо дорогого поиска соседей по радиусу
// используются глобальные средние положения и скорости роя.
// Сложность O(N·dim) на итерацию вместо O(N²·dim).

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

// ----------------------------------------------------------------------
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

// ----------------------------------------------------------------------
// Инициализация частиц (аналогично оригиналу)
// ----------------------------------------------------------------------
__global__ void init_kernel(double* X, double* V, double* pbest_pos, double* pbest_val,
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
        v[j] = curand_uniform_double(&state) * 2.0 - 1.0;
        p[j] = x[j];
    }
    pbest_val[idx] = compute_fitness(x, dim, func_id);
}

// ----------------------------------------------------------------------
// Редукция для поиска глобального минимума (блок → массив)
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
// Редукция векторов X и V для вычисления средних по рою
// Каждый блок суммирует свои частицы, результат записывается в два блочных массива
// Размер shared memory: 2 * blockDim.x * dim * sizeof(double)
// ----------------------------------------------------------------------
__global__ void reduce_vectors(const double* X, const double* V,
                               int dim, int particles,
                               double* d_block_Xsums, double* d_block_Vsums)
{
    extern __shared__ double s[];
    // Раскладка: первые blockDim.x * dim ячеек — для X, затем blockDim.x * dim — для V
    int tid = threadIdx.x;
    int block_stride = blockDim.x * dim;
    double* s_X = s;
    double* s_V = s + block_stride;

    // Загрузка частиц блока в shared память
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < particles) {
        for (int d = 0; d < dim; ++d) {
            s_X[tid * dim + d] = X[idx * dim + d];
            s_V[tid * dim + d] = V[idx * dim + d];
        }
    } else {
        // Выходящие за границы нити зануляют свои элементы
        for (int d = 0; d < dim; ++d) {
            s_X[tid * dim + d] = 0.0;
            s_V[tid * dim + d] = 0.0;
        }
    }
    __syncthreads();

    // Парная редукция внутри блока
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int d = 0; d < dim; ++d) {
                s_X[tid * dim + d] += s_X[(tid + s) * dim + d];
                s_V[tid * dim + d] += s_V[(tid + s) * dim + d];
            }
        }
        __syncthreads();
    }

    // Запись результатов блока
    if (tid == 0) {
        for (int d = 0; d < dim; ++d) {
            d_block_Xsums[blockIdx.x * dim + d] = s_X[d];
            d_block_Vsums[blockIdx.x * dim + d] = s_V[d];
        }
    }
}

// ----------------------------------------------------------------------
// Ядро обновления частиц с глобальными средними (вместо локальных соседей)
// ----------------------------------------------------------------------
__global__ void update_particles_global(double* X, double* V,
                                        double* pbest_pos, double* pbest_val,
                                        int dim, int particles,
                                        const double* __restrict__ gbest_pos,
                                        const double* __restrict__ mean_X,
                                        const double* __restrict__ mean_V,
                                        double lower, double upper,
                                        double w, double c1, double c2,
                                        double alpha, double beta, double gamma,
                                        unsigned int seed, int func_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particles) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);
    double* x_i = X + i * dim;
    double* v_i = V + i * dim;
    double* p_i = pbest_pos + i * dim;

    double max_vel_coeff = 0.2;
    double range = upper - lower;

    for (int j = 0; j < dim; ++j) {
        // Boids-компоненты от глобальных средних
        //double separation   = x_i[j] - mean_X[j];   // от центра масс
        double alignment    = mean_V[j] - v_i[j];   // к средней скорости
        double cohesion     = mean_X[j] - x_i[j];   // к центру масс

        double r1 = curand_uniform_double(&state);
        double r2 = curand_uniform_double(&state);
        v_i[j] = w * v_i[j]
               + c1 * r1 * (p_i[j] - x_i[j])
               + c2 * r2 * (gbest_pos[j] - x_i[j])
               //+ alpha * separation
               + beta  * alignment
               + gamma * cohesion;

        double max_vel = max_vel_coeff * range;
        v_i[j] = fmin(fmax(v_i[j], -max_vel), max_vel);
        x_i[j] += v_i[j];
        if (x_i[j] < lower) { x_i[j] = lower; v_i[j] = 0.0; }
        if (x_i[j] > upper) { x_i[j] = upper; v_i[j] = 0.0; }
    }

    double new_val = compute_fitness(x_i, dim, func_id);
    if (new_val < pbest_val[i]) {
        pbest_val[i] = new_val;
        for (int j = 0; j < dim; ++j)
            p_i[j] = x_i[j];
    }
}

// ----------------------------------------------------------------------
// Вспомогательная функция поиска и обновления глобального лучшего
// ----------------------------------------------------------------------
void find_and_update_global_best(const double* d_pbest_val, const double* d_X,
                                 double* d_gbest_val, double* d_gbest_pos,
                                 int dim, int particles,
                                 double* d_block_vals, int* d_block_idxs,
                                 int grid_size, int block_size,
                                 double& host_gbest_val, std::vector<double>& host_gbest_pos)
{
    block_reduce_min<<<grid_size, block_size, 2 * block_size * sizeof(double)>>>(
        d_pbest_val, particles, d_block_vals, d_block_idxs);

    std::vector<double> block_vals(grid_size);
    std::vector<int> block_idxs(grid_size);
    cudaMemcpy(block_vals.data(), d_block_vals, grid_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_idxs.data(), d_block_idxs, grid_size * sizeof(int), cudaMemcpyDeviceToHost);

    double min_val = block_vals[0];
    int best_idx = block_idxs[0];
    for (int i = 1; i < grid_size; ++i) {
        if (block_vals[i] < min_val) {
            min_val = block_vals[i];
            best_idx = block_idxs[i];
        }
    }

    if (min_val < host_gbest_val) {
        host_gbest_val = min_val;
        cudaMemcpy(host_gbest_pos.data(), d_X + best_idx * dim, dim * sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(d_gbest_val, &host_gbest_val, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gbest_pos, host_gbest_pos.data(), dim * sizeof(double), cudaMemcpyHostToDevice);
}

// ----------------------------------------------------------------------
// Основная процедура Boids-PSO с глобальными средними
// ----------------------------------------------------------------------
void boids_pso_global(int dim, int particles, int iterations,
                      double w, double c1, double c2,
                      double alpha, double beta, double gamma,
                      double lower, double upper,
                      std::vector<double>& best_pos, double& best_val,
                      std::vector<double>& history, unsigned int seed,
                      const std::string& func_name)
{
    // Основные массивы
    double *d_X, *d_V, *d_pbest_pos, *d_pbest_val;
    double *d_gbest_pos, *d_gbest_val;
    cudaMalloc(&d_X, particles * dim * sizeof(double));
    cudaMalloc(&d_V, particles * dim * sizeof(double));
    cudaMalloc(&d_pbest_pos, particles * dim * sizeof(double));
    cudaMalloc(&d_pbest_val, particles * sizeof(double));
    cudaMalloc(&d_gbest_pos, dim * sizeof(double));
    cudaMalloc(&d_gbest_val, sizeof(double));

    // Блоковые массивы для редукции (минимум)
    int block_size_min = 256;
    int grid_size_min = (particles + block_size_min - 1) / block_size_min;
    double *d_block_vals; int *d_block_idxs;
    cudaMalloc(&d_block_vals, grid_size_min * sizeof(double));
    cudaMalloc(&d_block_idxs, grid_size_min * sizeof(int));

    // Для вычисления средних: меньший размер блока, чтобы уместиться в shared memory
    int block_size_mean = 128;
    int grid_size_mean = (particles + block_size_mean - 1) / block_size_mean;
    // Память для блочных сумм (X и V)
    double *d_block_Xsums, *d_block_Vsums;
    cudaMalloc(&d_block_Xsums, grid_size_mean * dim * sizeof(double));
    cudaMalloc(&d_block_Vsums, grid_size_mean * dim * sizeof(double));

    // Буферы для глобальных средних на устройстве
    double *d_mean_X, *d_mean_V;
    cudaMalloc(&d_mean_X, dim * sizeof(double));
    cudaMalloc(&d_mean_V, dim * sizeof(double));

    int func_id = static_cast<int>(get_func_id(func_name));

    // Инициализация
    init_kernel<<<grid_size_min, block_size_min>>>(
        d_X, d_V, d_pbest_pos, d_pbest_val,
        dim, particles, lower, upper, seed, func_id);
    cudaDeviceSynchronize();

    // Первичный глобальный лучший
    double host_gbest_val = INFINITY;
    std::vector<double> host_gbest_pos(dim);
    find_and_update_global_best(d_pbest_val, d_X,
                                d_gbest_val, d_gbest_pos,
                                dim, particles,
                                d_block_vals, d_block_idxs,
                                grid_size_min, block_size_min,
                                host_gbest_val, host_gbest_pos);

    // Главный цикл
    for (int t = 0; t < iterations; ++t) {
        // 1. Вычисление глобальных средних X и V
        reduce_vectors<<<grid_size_mean, block_size_mean,
                         2 * block_size_mean * dim * sizeof(double)>>>(
            d_X, d_V, dim, particles, d_block_Xsums, d_block_Vsums);
        // Копируем блочные суммы на хост и вычисляем средние
        std::vector<double> h_block_Xsums(grid_size_mean * dim);
        std::vector<double> h_block_Vsums(grid_size_mean * dim);
        cudaMemcpy(h_block_Xsums.data(), d_block_Xsums,
                   grid_size_mean * dim * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_block_Vsums.data(), d_block_Vsums,
                   grid_size_mean * dim * sizeof(double), cudaMemcpyDeviceToHost);

        std::vector<double> mean_X(dim, 0.0), mean_V(dim, 0.0);
        for (int b = 0; b < grid_size_mean; ++b) {
            for (int d = 0; d < dim; ++d) {
                mean_X[d] += h_block_Xsums[b * dim + d];
                mean_V[d] += h_block_Vsums[b * dim + d];
            }
        }
        for (int d = 0; d < dim; ++d) {
            mean_X[d] /= particles;
            mean_V[d] /= particles;
        }
        // Копируем средние на устройство
        cudaMemcpy(d_mean_X, mean_X.data(), dim * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mean_V, mean_V.data(), dim * sizeof(double), cudaMemcpyHostToDevice);

        // 2. Обновление частиц (с использованием глобальных средних)
        update_particles_global<<<grid_size_min, block_size_min>>>(
            d_X, d_V, d_pbest_pos, d_pbest_val,
            dim, particles, d_gbest_pos,
            d_mean_X, d_mean_V,
            lower, upper, w, c1, c2,
            alpha, beta, gamma, seed + t + 1, func_id);
        cudaDeviceSynchronize();

        // 3. Поиск и обновление глобального лучшего
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

    // Очистка памяти
    cudaFree(d_X); cudaFree(d_V); cudaFree(d_pbest_pos); cudaFree(d_pbest_val);
    cudaFree(d_gbest_pos); cudaFree(d_gbest_val);
    cudaFree(d_block_vals); cudaFree(d_block_idxs);
    cudaFree(d_block_Xsums); cudaFree(d_block_Vsums);
    cudaFree(d_mean_X); cudaFree(d_mean_V);
}

// ----------------------------------------------------------------------
int main(int argc, char** argv) {
    int dim = 30, particles = 500, iterations = 500, file = 0;
    double w = 0.7, c1 = 1.5, c2 = 1.5;
    double alpha = 0.02, beta = 0.01, gamma = 0.01;
    unsigned int seed = 42;
    std::string func = "rastrigin";
    double lb = INFINITY, ub = INFINITY;

    // Параметр r_neigh больше не нужен, но оставим для совместимости
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-dim") == 0 && i+1 < argc) dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "-particles") == 0 && i+1 < argc) particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "-iter") == 0 && i+1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-w") == 0 && i+1 < argc) w = atof(argv[++i]);
        else if (strcmp(argv[i], "-c1") == 0 && i+1 < argc) c1 = atof(argv[++i]);
        else if (strcmp(argv[i], "-c2") == 0 && i+1 < argc) c2 = atof(argv[++i]);
        //else if (strcmp(argv[i], "-alpha") == 0 && i+1 < argc) alpha = atof(argv[++i]);
        else if (strcmp(argv[i], "-beta") == 0 && i+1 < argc) beta = atof(argv[++i]);
        else if (strcmp(argv[i], "-gamma") == 0 && i+1 < argc) gamma = atof(argv[++i]);
        else if (strcmp(argv[i], "-seed") == 0 && i+1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "-func") == 0 && i+1 < argc) func = argv[++i];
        else if (strcmp(argv[i], "-lb") == 0 && i+1 < argc) lb = atof(argv[++i]);
        else if (strcmp(argv[i], "-ub") == 0 && i+1 < argc) ub = atof(argv[++i]);
        else if (strcmp(argv[i], "-file") == 0 && i+1 < argc) file = atoi(argv[++i]);
        // -r_neigh игнорируется
    }
    if (lb == INFINITY) lb = func_bounds[func].lb;
    if (ub == INFINITY) ub = func_bounds[func].ub;

    std::cout << "Boids-PSO with global means (dim=" << dim
              << ", particles=" << particles << ", iter=" << iterations << ")\n";

    std::vector<double> best_pos(dim);
    double best_val;
    std::vector<double> history(iterations);

    auto start = std::chrono::high_resolution_clock::now();
    boids_pso_global(dim, particles, iterations, w, c1, c2,
                     alpha, beta, gamma, lb, ub,
                     best_pos, best_val, history, seed, func);
    auto end = std::chrono::high_resolution_clock::now();

    double time_sec = std::chrono::duration<double>(end - start).count();
    std::cout << "CUDA_TIME: " << time_sec << std::endl;
    std::cout << "BEST_VALUE: " << best_val << std::endl;

    if (file != 0){
        char fname[256];
        snprintf(fname, sizeof(fname), "boids_pso_global_%s_d%d_p%d_i%d_w%.2f_b%.4f_g%.4f_seed%u.txt",
                 func.c_str(), dim, particles, iterations, w, beta, gamma, seed);
        std::ofstream f(fname);
        f << "iteration,best_value\n";
        for (int i = 0; i < iterations; ++i)
            f << i << "," << history[i] << "\n";
        f.close();
    }

    return 0;
}

//компиляция nvcc -O2 boids_pso_cuda_global.cu -o boids_pso_cuda_global.exe -lcurand