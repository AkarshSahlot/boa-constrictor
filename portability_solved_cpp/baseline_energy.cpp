// Measure baseline (idle) energy consumption over a configurable duration.
// Build the same way as boa_constrictor with -DENABLE_ENERGY:
//
// Linux:
//   nvcc -o baseline_energy baseline_energy.cpp CPPJoules/src/cppJoules.cpp \
//     CPPJoules/src/energy_state.cpp CPPJoules/src/nvidia_devices.cpp \
//     CPPJoules/src/rapl_devices.cpp -O3 -std=c++17 -ICPPJoules/include \
//     -ICPPJoules/src -ldl
//
// Windows:
//   nvcc -o baseline_energy.exe baseline_energy.cpp CPPJoules/src/cppJoules.cpp ^
//     CPPJoules/src/energy_state.cpp CPPJoules/src/nvidia_devices.cpp ^
//     CPPJoules/src/rapl_devices.cpp -O3 -std=c++17 -ICPPJoules/include ^
//     -ICPPJoules/src
//
// Usage: ./baseline_energy [seconds]   (default: 60)

#include "cppJoules.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    int duration_sec = 60;
    if (argc > 1) {
        duration_sec = std::atoi(argv[1]);
        if (duration_sec <= 0) {
            std::cerr << "Usage: " << argv[0] << " [seconds]  (default: 60)\n";
            return 1;
        }
    }

    std::cout << "Measuring baseline energy for " << duration_sec << " seconds...\n";
    std::cout << "Keep the system idle during this measurement.\n\n";

    EnergyTracker tracker;
    tracker.start();

    // Sleep in 1-second intervals so we can show progress
    for (int i = 1; i <= duration_sec; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (i % 10 == 0 || i == duration_sec) {
            std::cout << "\r  " << i << " / " << duration_sec << " s" << std::flush;
        }
    }
    std::cout << "\n\n";

    tracker.stop();
    tracker.calculate_energy();

    std::cout << "=== Baseline Energy Results ===\n";
    tracker.print_energy();

    std::string csv_path = "baseline_energy_" + std::to_string(duration_sec) + "s.csv";
    tracker.save_csv(csv_path);
    std::cout << "\nSaved to " << csv_path << "\n";

    return 0;
}
