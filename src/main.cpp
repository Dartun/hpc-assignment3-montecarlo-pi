#include <iostream>
#include <random>
#include <cstdlib>   // for atoll
#include <cmath>     // for std::abs, M_PI (maybe)
#include <chrono>    // for timing
#include <iomanip>   // for precision

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " N\n";
        return 1;
    }

    const long long n = std::atoll(argv[1]);
    if (n <= 0) {
        std::cerr << "Error: N must be a positive integer\n";
        return 1;
    }

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    long long h = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (long long i = 0; i < n; ++i) {
        const double x = dist(gen);
        const double y = dist(gen);

        if (x*x + y*y <= 1.0) {
            ++h;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    const double pi_est = 4.0 * static_cast<double>(h) / static_cast<double>(n);
    const double pi_true = 3.14159265358979323846;
    const double elapsed = dt.count();
    const double samples_per_sec = static_cast<double>(n) / elapsed;
    const double time_per_sample = elapsed / static_cast<double>(n);

    std::cout << "n=" << n << "\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "pi_est=" << pi_est << "\n";
    std::cout << "abs_error=" << std::abs(pi_est - pi_true) << "\n";
    std::cout << std::setprecision(6);
    std::cout << "time_seconds=" << elapsed << "\n";
    std::cout << std::setprecision(2);
    std::cout << "samples_per_second=" << samples_per_sec << "\n";
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "time_per_sample=" << time_per_sample << "\n";

    return 0;
}
