// pso_cpu_bench.cpp
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

double rastrigin(const std::vector<double>& x) {
    int n = x.size();
    double s = 10.0 * n;
    for (int i = 0; i < n; ++i)
        s += x[i]*x[i] - 10.0 * cos(2.0 * M_PI * x[i]);
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

void pso_cpu(int dim, int particles, int iterations,
             double w, double c1, double c2,
             const std::vector<double>& lower, const std::vector<double>& upper,
             std::vector<double>& best_pos, double& best_val,
             std::vector<double>& history, unsigned int seed,
             const std::string& func_name)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist_pos(0.0, 1.0);
    std::uniform_real_distribution<double> dist_vel(-1.0, 1.0);

    std::vector<std::vector<double>> X(particles, std::vector<double>(dim));
    std::vector<std::vector<double>> V(particles, std::vector<double>(dim));
    std::vector<std::vector<double>> pbest_pos(particles, std::vector<double>(dim));
    std::vector<double> pbest_val(particles);

    std::vector<double> gbest_pos(dim);
    double gbest_val = INFINITY;

    auto fitness = cpu_func_map[func_name];

    // init
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

    std::uniform_real_distribution<double> dist_r(0.0, 1.0);

    for (int t = 0; t < iterations; ++t) {
        for (int i = 0; i < particles; ++i) {
            for (int j = 0; j < dim; ++j) {
                double r1 = dist_r(rng);
                double r2 = dist_r(rng);
                V[i][j] = w * V[i][j]
                        + c1 * r1 * (pbest_pos[i][j] - X[i][j])
                        + c2 * r2 * (gbest_pos[j] - X[i][j]);

                double max_vel = 0.2 * (upper[j] - lower[j]);
                if (V[i][j] > max_vel) V[i][j] = max_vel;
                if (V[i][j] < -max_vel) V[i][j] = -max_vel;

                X[i][j] += V[i][j];
                // boundary: absorbing
                if (X[i][j] < lower[j]) { X[i][j] = lower[j]; V[i][j] = 0; }
                if (X[i][j] > upper[j]) { X[i][j] = upper[j]; V[i][j] = 0; }
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

int main(int argc, char** argv) {
    // Параметры по умолчанию
    int dim = 30;
    int particles = 500;
    int iterations = 500;
    int file = 0;
    double w = 0.7, c1 = 1.5, c2 = 1.5;
    unsigned int seed = 42;
    std::string func_name = "rastrigin";
    double lower_bound = INFINITY, upper_bound = INFINITY;

    // парсинг аргументов
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-dim") == 0 && i+1 < argc) dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "-particles") == 0 && i+1 < argc) particles = atoi(argv[++i]);
        else if (strcmp(argv[i], "-iter") == 0 && i+1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-w") == 0 && i+1 < argc) w = atof(argv[++i]);
        else if (strcmp(argv[i], "-c1") == 0 && i+1 < argc) c1 = atof(argv[++i]);
        else if (strcmp(argv[i], "-c2") == 0 && i+1 < argc) c2 = atof(argv[++i]);
        else if (strcmp(argv[i], "-seed") == 0 && i+1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "-func") == 0 && i+1 < argc) func_name = argv[++i];
        else if (strcmp(argv[i], "-lb") == 0 && i+1 < argc) lower_bound = atof(argv[++i]);
        else if (strcmp(argv[i], "-ub") == 0 && i+1 < argc) upper_bound = atof(argv[++i]);
        else if (strcmp(argv[i], "-file") == 0 && i+1 < argc) file = atoi(argv[++i]);
    }
    if (lower_bound == INFINITY) lower_bound = func_bounds[func_name].lb;
    if (upper_bound == INFINITY) upper_bound = func_bounds[func_name].ub;

    std::vector<double> lower(dim, lower_bound);
    std::vector<double> upper(dim, upper_bound);

    std::vector<double> best_pos(dim);
    double best_val;
    std::vector<double> history(iterations);

    auto start = std::chrono::high_resolution_clock::now();
    pso_cpu(dim, particles, iterations, w, c1, c2, lower, upper, best_pos, best_val, history, seed, func_name);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Вывод на экран
    std::cout << "CPU_TIME: " << elapsed.count() << std::endl;
    std::cout << "BEST_VALUE: " << best_val << std::endl;

    // Сохранение истории в файл с уникальным именем
    if (file != 0){
        char fname[256];
        snprintf(fname, sizeof(fname), "pso_cpu_%s_d%d_p%d_i%d_w%.2f_c1%.2f_c2%.2f_seed%u.txt",
                 func_name.c_str(), dim, particles, iterations, w, c1, c2, seed);
        std::ofstream f(fname);
        f << "iteration,best_value\n";
        for (int i = 0; i < iterations; ++i)
            f << i << "," << history[i] << "\n";
        f.close();
    }

    return 0;
}
//компиляция g++ -O2 pso_cpu_bench.cpp -o pso_cpu_bench.exe