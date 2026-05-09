// boids_pso_cuda_optimized.cu
// Версия с поиском соседей (Boids - O(N²·dim)).
// Оптимизации:
// - SoA‑раскладка (dim × particles) для X, V, pbest_pos.
// - Динамическое выделение локальных массивов вместо фиксированных [2000].
// - Коалесцированные/широковещательные обращения к глобальной памяти во всех циклах.

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

struct Bounds { double lb, ub; };
std::unordered_map<std::string, Bounds> func_bounds = {
    {"rastrigin", {-5.12, 5.12}},
    {"rosenbrock",{-2.048, 2.048}},
    {"ackley",    {-32.768, 32.768}}
};

double compute_default_radius(int dim, int particles, double lower, double upper) {
    double range = upper - lower;
    if (particles <= 1) return range;
    int target_neighbors = std::min(20, std::max(5, particles / 50));
    double volume_ratio = static_cast<double>(target_neighbors) / particles;
    if (dim <= 20) {
        double C_d = pow(M_PI, dim / 2.0) / tgamma(dim / 2.0 + 1.0);
        double r = range * pow(volume_ratio / C_d, 1.0 / dim);
        double max_dist = range * sqrt(static_cast<double>(dim));
        if (r > max_dist * 0.5) r = max_dist * 0.5;
        return r;
    } else {
        return range * pow(volume_ratio, 1.0 / dim) * 3.0;
    }
}

// ----------------------------------------------------------------------
// Инициализация (SoA)
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
    for (int j = 0; j < dim; ++j) {
        double xv = lower + curand_uniform_double(&state) * range;
        double vv = curand_uniform_double(&state) * 2.0 - 1.0;
        X[j * particles + i] = xv;
        V[j * particles + i] = vv;
        pbest_pos[j * particles + i] = xv;
    }
    pbest_val[i] = compute_fitness(X, dim, particles, i, func_id);
}

// ----------------------------------------------------------------------
// Обновление с локальными соседями (SoA + динамические массивы)
// ----------------------------------------------------------------------
__global__ void update_boids_kernel(double* X, double* V,
                                    double* pbest_pos, double* pbest_val,
                                    int dim, int particles,
                                    const double* __restrict__ gbest_pos,
                                    double lower, double upper,
                                    double w, double c1, double c2,
                                    double alpha, double beta, double gamma,
                                    double r_neigh,
                                    unsigned int seed, int func_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particles) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);

    // Выделяем локальные массивы для накопления Boids-сил
    double* sep = new double[dim];
    double* alg = new double[dim];
    double* coh = new double[dim];
    for (int j = 0; j < dim; ++j) {
        sep[j] = 0.0;
        alg[j] = 0.0;
        coh[j] = 0.0;
    }

    int neigh_count = 0;
    double r_neigh_sq = r_neigh * r_neigh;
    double max_vel_coeff = 0.2;
    double range = upper - lower;

    // Поиск соседей (полный перебор)
    for (int k = 0; k < particles; ++k) {
        if (k == i) continue;

        // Расстояние между i и k
        double dist_sq = 0.0;
        for (int j = 0; j < dim; ++j) {
            double diff = X[j * particles + i] - X[j * particles + k];
            dist_sq += diff * diff;
        }

        if (dist_sq < r_neigh_sq && dist_sq > 1e-12) {
            neigh_count++;
            double inv_dist = rsqrt(dist_sq);
            for (int j = 0; j < dim; ++j) {
                double diff_x = X[j * particles + i] - X[j * particles + k];
                sep[j] += diff_x * inv_dist;
                alg[j] += V[j * particles + k] - V[j * particles + i];
                coh[j] += X[j * particles + k] - X[j * particles + i];
            }
        }
    }

    // Нормировка
    if (neigh_count > 0) {
        double inv_n = 1.0 / neigh_count;
        for (int j = 0; j < dim; ++j) {
            sep[j] *= inv_n;
            alg[j] *= inv_n;
            coh[j] *= inv_n;
        }
    }

    // Обновление скорости и позиции
    for (int j = 0; j < dim; ++j) {
        double x_ij = X[j * particles + i];
        double v_ij = V[j * particles + i];
        double p_ij = pbest_pos[j * particles + i];
        double r1 = curand_uniform_double(&state);
        double r2 = curand_uniform_double(&state);

        v_ij = w * v_ij
             + c1 * r1 * (p_ij - x_ij)
             + c2 * r2 * (gbest_pos[j] - x_ij)
             + alpha * sep[j]
             + beta  * alg[j]
             + gamma * coh[j];

        double max_vel = max_vel_coeff * range;
        v_ij = fmin(fmax(v_ij, -max_vel), max_vel);

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

    delete[] sep;
    delete[] alg;
    delete[] coh;
}

// ----------------------------------------------------------------------
// Редукция минимума (как в исходной)
// ----------------------------------------------------------------------
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

// ----------------------------------------------------------------------
// Вспомогательная функция выбора глобального лучшего
// ----------------------------------------------------------------------
void find_and_update_global_best(const double* d_pbest_val, const double* d_X,
                                 double* d_gbest_val, double* d_gbest_pos,
                                 int dim, int particles,
                                 double* d_block_vals, int* d_block_idxs,
                                 int grid_size, int block_size,
                                 double& host_gbest_val, std::vector<double>& host_gbest_pos)
{
    block_reduce<<<grid_size, block_size, 2 * block_size * sizeof(double)>>>(
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
        for (int j = 0; j < dim; ++j)
            CUDA_CHECK(cudaMemcpy(&host_gbest_pos[j], d_X + j * particles + best_idx,
                                  sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(d_gbest_val, &host_gbest_val, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gbest_pos, host_gbest_pos.data(), dim * sizeof(double), cudaMemcpyHostToDevice));
}

// ----------------------------------------------------------------------
// Основная процедура Boids‑PSO (локальные соседи)
// ----------------------------------------------------------------------
void boids_pso_cuda(int dim, int particles, int iterations,
                    double w, double c1, double c2,
                    double alpha, double beta, double gamma, double r_neigh,
                    double lower, double upper,
                    std::vector<double>& best_pos, double& best_val,
                    std::vector<double>& history, unsigned int seed,
                    const std::string& func_name)
{
    if (r_neigh <= 0.0)
        r_neigh = compute_default_radius(dim, particles, lower, upper);

    // Выделение памяти (SoA)
    double *d_X, *d_V, *d_pbest_pos, *d_pbest_val;
    CUDA_CHECK(cudaMalloc(&d_X, dim * particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_V, dim * particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pbest_pos, dim * particles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pbest_val, particles * sizeof(double)));

    double *d_gbest_pos, *d_gbest_val;
    CUDA_CHECK(cudaMalloc(&d_gbest_pos, dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_gbest_val, sizeof(double)));

    int block_size = 256;
    int grid_size = (particles + block_size - 1) / block_size;
    double *d_block_vals; int *d_block_idxs;
    CUDA_CHECK(cudaMalloc(&d_block_vals, grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_idxs, grid_size * sizeof(int)));

    int func_id = static_cast<int>(get_func_id(func_name));

    // Инициализация
    init_kernel<<<grid_size, block_size>>>(d_X, d_V, d_pbest_pos, d_pbest_val,
                                           dim, particles, lower, upper, seed, func_id);
    CUDA_CHECK(cudaDeviceSynchronize());

    double host_gbest_val = INFINITY;
    std::vector<double> host_gbest_pos(dim);
    find_and_update_global_best(d_pbest_val, d_X, d_gbest_val, d_gbest_pos,
                                dim, particles, d_block_vals, d_block_idxs,
                                grid_size, block_size, host_gbest_val, host_gbest_pos);

    // Главный цикл
    for (int t = 0; t < iterations; ++t) {
        update_boids_kernel<<<grid_size, block_size>>>(d_X, d_V, d_pbest_pos, d_pbest_val,
                                                       dim, particles, d_gbest_pos,
                                                       lower, upper,
                                                       w, c1, c2, alpha, beta, gamma, r_neigh,
                                                       seed + t + 1, func_id);
        CUDA_CHECK(cudaDeviceSynchronize());

        find_and_update_global_best(d_pbest_val, d_X, d_gbest_val, d_gbest_pos,
                                    dim, particles, d_block_vals, d_block_idxs,
                                    grid_size, block_size, host_gbest_val, host_gbest_pos);
        history[t] = host_gbest_val;
    }

    best_val = host_gbest_val;
    best_pos = host_gbest_pos;

    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_pbest_pos)); CUDA_CHECK(cudaFree(d_pbest_val));
    CUDA_CHECK(cudaFree(d_gbest_pos)); CUDA_CHECK(cudaFree(d_gbest_val));
    CUDA_CHECK(cudaFree(d_block_vals)); CUDA_CHECK(cudaFree(d_block_idxs));
}

// ----------------------------------------------------------------------
int main(int argc, char** argv) {
    int dim = 30, particles = 500, iterations = 500, file = 0;
    double w = 0.7, c1 = 1.5, c2 = 1.5;
    double alpha = 0.02, beta = 0.01, gamma = 0.01, r_neigh = -1.0;
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
        else if (strcmp(argv[i], "-alpha") == 0 && i+1 < argc) alpha = atof(argv[++i]);
        else if (strcmp(argv[i], "-beta") == 0 && i+1 < argc) beta = atof(argv[++i]);
        else if (strcmp(argv[i], "-gamma") == 0 && i+1 < argc) gamma = atof(argv[++i]);
        else if (strcmp(argv[i], "-r_neigh") == 0 && i+1 < argc) r_neigh = atof(argv[++i]);
        else if (strcmp(argv[i], "-seed") == 0 && i+1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "-func") == 0 && i+1 < argc) func = argv[++i];
        else if (strcmp(argv[i], "-lb") == 0 && i+1 < argc) lb = atof(argv[++i]);
        else if (strcmp(argv[i], "-ub") == 0 && i+1 < argc) ub = atof(argv[++i]);
        else if (strcmp(argv[i], "-file") == 0 && i+1 < argc) file = atoi(argv[++i]);
    }
    if (lb == INFINITY) lb = func_bounds[func].lb;
    if (ub == INFINITY) ub = func_bounds[func].ub;

    std::vector<double> best_pos(dim);
    double best_val;
    std::vector<double> history(iterations);
    auto start = std::chrono::high_resolution_clock::now();
    boids_pso_cuda(dim, particles, iterations, w, c1, c2, alpha, beta, gamma, r_neigh,
                   lb, ub, best_pos, best_val, history, seed, func);
    auto end = std::chrono::high_resolution_clock::now();
    double time_sec = std::chrono::duration<double>(end - start).count();
    std::cout << "CUDA_TIME: " << time_sec << std::endl;
    std::cout << "BEST_VALUE: " << best_val << std::endl;

    if(file != 0){
        char fname[256];
        snprintf(fname, sizeof(fname), "boids_pso_cuda_optimized_%s_d%d_p%d_i%d_w%.2f_a%.4f_b%.4f_g%.4f_r%.2f_seed%u.txt",
                 func.c_str(), dim, particles, iterations, w, alpha, beta, gamma, r_neigh, seed);
        std::ofstream f(fname);
        f << "iteration,best_value\n";
        for (int i = 0; i < iterations; ++i) f << i << "," << history[i] << "\n";
        f.close();
    }
    return 0;
}

// Компиляция:
// nvcc -O2 boids_pso_cuda.cu -o boids_pso_cuda.exe -lcurand