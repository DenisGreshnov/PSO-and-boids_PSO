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

void boids_pso_cpu(int dim, int particles, int iterations,
                   double w, double c1, double c2,
                   double alpha, double beta, double gamma, double r_neigh,
                   const std::vector<double>& lower, const std::vector<double>& upper,
                   std::vector<double>& best_pos, double& best_val,
                   std::vector<double>& history, unsigned int seed,
                   const std::string& func_name)
{
    if (r_neigh <= 0.0) {
        r_neigh = compute_default_radius(dim, particles, lower[0], upper[0]);
    }

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
        for (int i = 0; i < particles; ++i) {
            // Поиск соседей и вычисление сил boids
            std::vector<double> separation(dim, 0.0), alignment(dim, 0.0), cohesion(dim, 0.0);
            int neigh_count = 0;
            for (int k = 0; k < particles; ++k) {
                if (k == i) continue;
                double dist_sq = 0.0;
                for (int j = 0; j < dim; ++j) {
                    double diff = X[i][j] - X[k][j];
                    dist_sq += diff * diff;
                }
                double dist = sqrt(dist_sq);
                if (dist < r_neigh && dist > 1e-12) {
                    neigh_count++;
                    for (int j = 0; j < dim; ++j) {
                        separation[j] += (X[i][j] - X[k][j]) / dist; // отталкивание
                        alignment[j] += V[k][j] - V[i][j];          // выравнивание скоростей
                        cohesion[j] += X[k][j] - X[i][j];           // к центру масс
                    }
                }
            }
            if (neigh_count > 0) {
                for (int j = 0; j < dim; ++j) {
                    separation[j] /= neigh_count;
                    alignment[j] /= neigh_count;
                    cohesion[j] /= neigh_count;
                }
            }

            // Обновление скорости
            for (int j = 0; j < dim; ++j) {
                double r1 = dist_r(rng), r2 = dist_r(rng);
                V[i][j] = w * V[i][j]
                          + c1 * r1 * (pbest_pos[i][j] - X[i][j])
                          + c2 * r2 * (gbest_pos[j] - X[i][j])
                          + alpha * separation[j]
                          + beta  * alignment[j]
                          + gamma * cohesion[j];
                double max_vel = 0.2 * (upper[j] - lower[j]);
                if (V[i][j] > max_vel) V[i][j] = max_vel;
                if (V[i][j] < -max_vel) V[i][j] = -max_vel;

                X[i][j] += V[i][j];
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

    std::vector<double> lower(dim, lb), upper(dim, ub);
    std::vector<double> best_pos(dim);
    double best_val;
    std::vector<double> history(iterations);

    auto start = std::chrono::high_resolution_clock::now();
    boids_pso_cpu(dim, particles, iterations, w, c1, c2, alpha, beta, gamma, r_neigh,
                  lower, upper, best_pos, best_val, history, seed, func);
    auto end = std::chrono::high_resolution_clock::now();
    double time_sec = std::chrono::duration<double>(end - start).count();
    std::cout << "CPU_TIME: " << time_sec << std::endl;
    std::cout << "BEST_VALUE: " << best_val << std::endl;

    if (file != 0){
        char fname[256];
        snprintf(fname, sizeof(fname), "boids_pso_cpu_%s_d%d_p%d_i%d_w%.2f_a%.4f_b%.4f_g%.4f_r%.2f_seed%u.txt",
                 func.c_str(), dim, particles, iterations, w, alpha, beta, gamma, r_neigh, seed);
        std::ofstream f(fname);
        f << "iteration,best_value\n";
        for (int i = 0; i < iterations; ++i) f << i << "," << history[i] << "\n";
        f.close();
    }

    return 0;
}

//Компиляция: g++ -O2 boids_pso_cpu.cpp -o boids_pso_cpu.exe