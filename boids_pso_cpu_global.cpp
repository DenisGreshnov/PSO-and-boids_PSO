#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <unordered_map>
#include <functional>

// ----------------------------------------------------------------------
// Функции
// ----------------------------------------------------------------------
double rastrigin(const std::vector<double>& x) {
    int n = x.size();
    double s = 10.0 * n;
    for (int i = 0; i < n; ++i)
        s += x[i]*x[i] - 10.0 * std::cos(2.0 * M_PI * x[i]);
    return s;
}

// Rosenbrock
double rosenbrock(const std::vector<double>& x) {
    int n = x.size();
    double s = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        double t1 = x[i+1] - x[i]*x[i];
        double t2 = 1.0 - x[i];
        s += 100.0 * t1*t1 + t2*t2;
    }
    return s;
}

// Ackley
double ackley(const std::vector<double>& x) {
    int n = x.size();
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; ++i) {
        sum1 += x[i] * x[i];
        sum2 += cos(2.0 * M_PI * x[i]);
    }
    double n_inv = 1.0 / n;
    return -20.0 * exp(-0.2 * sqrt(n_inv * sum1))
           - exp(n_inv * sum2) + 20.0 + M_E;
}

using ObjectiveFunc = std::function<double(const std::vector<double>&)>;

std::unordered_map<std::string, ObjectiveFunc> cpu_func_map = {
    {"rastrigin", rastrigin},
    {"rosenbrock", rosenbrock},
    {"ackley",    ackley}
};

struct Bounds { double lb, ub; };
std::unordered_map<std::string, Bounds> func_bounds = {
    {"rastrigin", {-5.12, 5.12}},
    {"rosenbrock",{-2.048, 2.048}},
    {"ackley",    {-32.768, 32.768}}
};

// ----------------------------------------------------------------------
// Boids-PSO с глобальными средними (CPU)
// ----------------------------------------------------------------------
void boids_pso_cpu_global(int dim, int particles, int iterations,
                          double w, double c1, double c2,
                          double alpha, double beta, double gamma,
                          const std::vector<double>& lower, const std::vector<double>& upper,
                          std::vector<double>& best_pos, double& best_val,
                          std::vector<double>& history, unsigned int seed,
                          const std::string& func_name)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist_pos(0.0, 1.0);
    std::uniform_real_distribution<double> dist_vel(-1.0, 1.0);
    std::uniform_real_distribution<double> dist_r(0.0, 1.0);

    std::vector<std::vector<double>> X(particles, std::vector<double>(dim));
    std::vector<std::vector<double>> V(particles, std::vector<double>(dim));
    std::vector<std::vector<double>> pbest_pos(particles, std::vector<double>(dim));
    std::vector<double> pbest_val(particles);

    std::vector<double> gbest_pos(dim);
    double gbest_val = INFINITY;

    auto fitness = cpu_func_map[func_name];

    // Инициализация
    for (int i = 0; i < particles; ++i) {
        for (int j = 0; j < dim; ++j) {
            X[i][j] = lower[j] + dist_pos(rng) * (upper[j] - lower[j]);
            V[i][j] = dist_vel(rng);
        }
        pbest_val[i] = fitness(X[i]);
        pbest_pos[i] = X[i];
        if (pbest_val[i] < gbest_val) {
            gbest_val = pbest_val[i];
            gbest_pos = X[i];
        }
    }

    // Главный цикл
    for (int t = 0; t < iterations; ++t) {
        // --- Вычисление глобальных средних ---
        std::vector<double> mean_X(dim, 0.0), mean_V(dim, 0.0);
        for (int i = 0; i < particles; ++i) {
            for (int j = 0; j < dim; ++j) {
                mean_X[j] += X[i][j];
                mean_V[j] += V[i][j];
            }
        }
        for (int j = 0; j < dim; ++j) {
            mean_X[j] /= particles;
            mean_V[j] /= particles;
        }

        // Обновление каждой частицы
        for (int i = 0; i < particles; ++i) {
            // --- Boids-компоненты от глобальных средних ---
            for (int j = 0; j < dim; ++j) {
                //double separation = X[i][j] - mean_X[j];   // от центра масс
                double alignment  = mean_V[j] - V[i][j];   // к средней скорости
                double cohesion   = mean_X[j] - X[i][j];   // к центру масс

                double r1 = dist_r(rng), r2 = dist_r(rng);
                V[i][j] = w * V[i][j]
                          + c1 * r1 * (pbest_pos[i][j] - X[i][j])
                          + c2 * r2 * (gbest_pos[j] - X[i][j])
                          //+ alpha * separation
                          + beta  * alignment
                          + gamma * cohesion;

                double max_vel = 0.2 * (upper[j] - lower[j]);
                if (V[i][j] > max_vel) V[i][j] = max_vel;
                if (V[i][j] < -max_vel) V[i][j] = -max_vel;

                X[i][j] += V[i][j];
                if (X[i][j] < lower[j]) { X[i][j] = lower[j]; V[i][j] = 0.0; }
                if (X[i][j] > upper[j]) { X[i][j] = upper[j]; V[i][j] = 0.0; }
            }

            double val = fitness(X[i]);
            if (val < pbest_val[i]) {
                pbest_val[i] = val;
                pbest_pos[i] = X[i];
                if (val < gbest_val) {
                    gbest_val = val;
                    gbest_pos = X[i];
                }
            }
        }
        history[t] = gbest_val;
    }

    best_pos = gbest_pos;
    best_val = gbest_val;
}

// ----------------------------------------------------------------------
int main(int argc, char** argv) {
    int dim = 30, particles = 500, iterations = 500, file = 0;
    double w = 0.7, c1 = 1.5, c2 = 1.5;
    double alpha = 0.02, beta = 0.01, gamma = 0.01;
    unsigned int seed = 42;
    std::string func = "rastrigin";
    double lb = INFINITY, ub = INFINITY;

    // r_neigh больше не используется, оставим для обратной совместимости
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

    std::cout << "Boids-PSO with global means (CPU, dim=" << dim
              << ", particles=" << particles << ", iter=" << iterations << ")\n";

    std::vector<double> lower(dim, lb), upper(dim, ub);
    std::vector<double> best_pos(dim);
    double best_val;
    std::vector<double> history(iterations);

    auto start = std::chrono::high_resolution_clock::now();
    boids_pso_cpu_global(dim, particles, iterations, w, c1, c2, alpha, beta, gamma,
                         lower, upper, best_pos, best_val, history, seed, func);
    auto end = std::chrono::high_resolution_clock::now();
    double time_sec = std::chrono::duration<double>(end - start).count();
    std::cout << "CPU_TIME: " << time_sec << std::endl;
    std::cout << "BEST_VALUE: " << best_val << std::endl;

    if (file != 0){
        char fname[256];
        snprintf(fname, sizeof(fname), "boids_pso_cpu_global_%s_d%d_p%d_i%d_w%.2f_b%.4f_g%.4f_seed%u.txt",
                 func.c_str(), dim, particles, iterations, w, beta, gamma, seed);
        std::ofstream f(fname);
        f << "iteration,best_value\n";
        for (int i = 0; i < iterations; ++i) f << i << "," << history[i] << "\n";
        f.close();
    }

    return 0;
}

// Компиляция:
// g++ -O2 boids_pso_cpu_global.cpp -o boids_pso_cpu_global.exe