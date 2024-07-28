#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>

// Function to estimate Pi using the Monte Carlo method in a single-threaded manner
double estimatePiSingleThread(int totalPoints) {
    int pointsInsideCircle = 0;
    std::random_device randDevice;
    std::mt19937 generator(randDevice());
    std::uniform_real_distribution<> distribution(0.0, 1.0);

    for (int i = 0; i < totalPoints; ++i) {
        double x = distribution(generator);
        double y = distribution(generator);
        if (x * x + y * y <= 1.0) {
            ++pointsInsideCircle;
        }
    }
    return 4.0 * pointsInsideCircle / totalPoints;
}

// Function to estimate Pi using the Monte Carlo method in a multi-threaded manner
double estimatePiMultiThread(int totalPoints, int threadCount) {
    int pointsInsideCircle = 0;

    #pragma omp parallel num_threads(threadCount)
    {
        std::random_device randDevice;
        std::mt19937 generator(randDevice());
        std::uniform_real_distribution<> distribution(0.0, 1.0);
        int localCount = 0;

        #pragma omp for
        for (int i = 0; i < totalPoints; ++i) {
            double x = distribution(generator);
            double y = distribution(generator);
            if (x * x + y * y <= 1.0) {
                ++localCount;
            }
        }

        #pragma omp atomic
        pointsInsideCircle += localCount;
    }

    return 4.0 * pointsInsideCircle / totalPoints;
}

int main() {
    while (true) {
        int points, threads;

        std::cout << "Enter the number of points (0 to exit): ";
        std::cin >> points;
        if (points == 0) {
            std::cout << "Exiting." << std::endl;
            break;
        }

        std::cout << "Enter the number of threads: ";
        std::cin >> threads;

        auto startSingleThread = std::chrono::high_resolution_clock::now();
        double piSingleThread = estimatePiSingleThread(points);
        auto endSingleThread = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationSingleThread = endSingleThread - startSingleThread;

        auto startMultiThread = std::chrono::high_resolution_clock::now();
        double piMultiThread = estimatePiMultiThread(points, threads);
        auto endMultiThread = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationMultiThread = endMultiThread - startMultiThread;

        std::cout << "Single-threaded Pi estimation: " << piSingleThread << std::endl;
        std::cout << "Single-threaded computation time: " << durationSingleThread.count() << " seconds." << std::endl;
        std::cout << "Multi-threaded Pi estimation: " << piMultiThread << std::endl;
        std::cout << "Multi-threaded computation time: " << durationMultiThread.count() << " seconds." << std::endl;
    }

    return 0;
}