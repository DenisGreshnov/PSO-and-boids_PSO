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

__device__ double rastrigin(const double* x, int dim) {
    double s = 10.0 * dim;
    for (int i = 0; i < dim; ++i)
        s += x[i]*x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
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

// Автоматический подбор радиуса соседства
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

// Инициализация частиц
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

// Ядро обновления с силами boids (полный перебор соседей)
__global__ void update_boids_kernel(double* X, double* V, double* pbest_pos, double* pbest_val,
                                    int dim, int particles,
                                    const double* __restrict__ gbest_pos,
                                    double lower, double upper,
                                    double w, double c1, double c2,
                                    double alpha, double beta, double gamma, double r_neigh,
                                    unsigned int seed, int func_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particles) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);

    double* x_i = X + i * dim;
    double* v_i = V + i * dim;
    double* p_i = pbest_pos + i * dim;

    // Сначала накопить силы boids
    double separation[200] = {0};   // предполагаем dim <= 200, для гибкости можно выделить shared
    double alignment[200] = {0};
    double cohesion[200] = {0};
    int neigh_count = 0;

    // Поиск соседей (полный перебор)
    for (int k = 0; k < particles; ++k) {
        if (k == i) continue;
        double dist_sq = 0.0;
        for (int j = 0; j < dim; ++j) {
            double diff = x_i[j] - X[k * dim + j];
            dist_sq += diff * diff;
        }
        if (dist_sq < r_neigh * r_neigh && dist_sq > 1e-12) {
            neigh_count++;
            double dist = sqrt(dist_sq);
            for (int j = 0; j < dim; ++j) {
                separation[j] += (x_i[j] - X[k * dim + j]) / dist;
                alignment[j] += V[k * dim + j] - v_i[j];
                cohesion[j] += X[k * dim + j] - x_i[j];
            }
        }
    }

    if (neigh_count > 0) {
        double inv = 1.0 / neigh_count;
        for (int j = 0; j < dim; ++j) {
            separation[j] *= inv;
            alignment[j] *= inv;
            cohesion[j] *= inv;
        }
    }

    double max_vel_coeff = 0.2;
    double range = upper - lower;

    for (int j = 0; j < dim; ++j) {
        double r1 = curand_uniform_double(&state);
        double r2 = curand_uniform_double(&state);
        v_i[j] = w * v_i[j]
                + c1 * r1 * (p_i[j] - x_i[j])
                + c2 * r2 * (gbest_pos[j] - x_i[j])
                + alpha * separation[j]
                + beta  * alignment[j]
                + gamma * cohesion[j];

        double max_vel = max_vel_coeff * range;
        v_i[j] = fmin(fmax(v_i[j], -max_vel), max_vel);

        x_i[j] += v_i[j];
        if (x_i[j] < lower) { x_i[j] = lower; v_i[j] = 0; }
        if (x_i[j] > upper) { x_i[j] = upper; v_i[j] = 0; }
    }

    double new_val = compute_fitness(x_i, dim, func_id);
    if (new_val < pbest_val[i]) {
        pbest_val[i] = new_val;
        for (int j = 0; j < dim; ++j)
            p_i[j] = x_i[j];
    }
}

// Редукция минимума по блокам (как в исправленном PSO)
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

// Функция поиска и обновления глобального лучшего (перенос на хост)
void find_and_update_global_best(const double* d_pbest_val, const double* d_X,
                                 double* d_gbest_val, double* d_gbest_pos,
                                 int dim, int particles,
                                 double* d_block_vals, int* d_block_idxs,
                                 int grid_size, int block_size,
                                 double& host_gbest_val, std::vector<double>& host_gbest_pos)
{
    block_reduce<<<grid_size, block_size, 2 * block_size * sizeof(double)>>>(d_pbest_val, particles,
                                                                             d_block_vals, d_block_idxs);
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

void boids_pso_cuda(int dim, int particles, int iterations,
                    double w, double c1, double c2,
                    double alpha, double beta, double gamma, double r_neigh,
                    double lower, double upper,
                    std::vector<double>& best_pos, double& best_val,
                    std::vector<double>& history, unsigned int seed,
                    const std::string& func_name)
{
    if (r_neigh <= 0.0) {
        r_neigh = compute_default_radius(dim, particles, lower, upper);
    }

    double *d_X, *d_V, *d_pbest_pos, *d_pbest_val;
    double *d_gbest_pos, *d_gbest_val;
    cudaMalloc(&d_X, particles * dim * sizeof(double));
    cudaMalloc(&d_V, particles * dim * sizeof(double));
    cudaMalloc(&d_pbest_pos, particles * dim * sizeof(double));
    cudaMalloc(&d_pbest_val, particles * sizeof(double));
    cudaMalloc(&d_gbest_pos, dim * sizeof(double));
    cudaMalloc(&d_gbest_val, sizeof(double));

    int block_size = 256;
    int grid_size = (particles + block_size - 1) / block_size;
    double *d_block_vals; int *d_block_idxs;
    cudaMalloc(&d_block_vals, grid_size * sizeof(double));
    cudaMalloc(&d_block_idxs, grid_size * sizeof(int));

    int func_id = static_cast<int>(get_func_id(func_name));

    // Инициализация
    init_kernel<<<grid_size, block_size>>>(d_X, d_V, d_pbest_pos, d_pbest_val,
                                           dim, particles, lower, upper, seed, func_id);
    cudaDeviceSynchronize();

    double host_gbest_val = INFINITY;
    std::vector<double> host_gbest_pos(dim);
    find_and_update_global_best(d_pbest_val, d_X, d_gbest_val, d_gbest_pos,
                                dim, particles, d_block_vals, d_block_idxs,
                                grid_size, block_size, host_gbest_val, host_gbest_pos);

    for (int t = 0; t < iterations; ++t) {
        update_boids_kernel<<<grid_size, block_size>>>(d_X, d_V, d_pbest_pos, d_pbest_val,
                                                       dim, particles, d_gbest_pos,
                                                       lower, upper,
                                                       w, c1, c2, alpha, beta, gamma, r_neigh,
                                                       seed + t + 1, func_id);
        cudaDeviceSynchronize();

        find_and_update_global_best(d_pbest_val, d_X, d_gbest_val, d_gbest_pos,
                                    dim, particles, d_block_vals, d_block_idxs,
                                    grid_size, block_size, host_gbest_val, host_gbest_pos);
        history[t] = host_gbest_val;
    }

    best_val = host_gbest_val;
    best_pos = host_gbest_pos;

    cudaFree(d_X); cudaFree(d_V); cudaFree(d_pbest_pos); cudaFree(d_pbest_val);
    cudaFree(d_gbest_pos); cudaFree(d_gbest_val);
    cudaFree(d_block_vals); cudaFree(d_block_idxs);
}

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
        snprintf(fname, sizeof(fname), "boids_pso_cuda_%s_d%d_p%d_i%d_w%.2f_a%.4f_b%.4f_g%.4f_r%.2f_seed%u.txt",
                 func.c_str(), dim, particles, iterations, w, alpha, beta, gamma, r_neigh, seed);
        std::ofstream f(fname);
        f << "iteration,best_value\n";
        for (int i = 0; i < iterations; ++i) f << i << "," << history[i] << "\n";
        f.close();
    }

    return 0;
}

//Компиляция: nvcc -O2 boids_pso_cuda.cu -o boids_pso_cuda.exe -lcurand