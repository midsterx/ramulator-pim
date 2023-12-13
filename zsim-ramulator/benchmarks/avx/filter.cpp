#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdint>
#include <sys/time.h>

#include "/users/mthevend/mnt/CloudStorage/midhush/ramulator-pim/zsim-ramulator/misc/hooks/zsim_hooks.h"

using namespace std;

const int ARRAY_SIZE = 32768;

void filterRangeAVX2(float* input, float* output, int size, float lower_bound, float upper_bound) {
    zsim_roi_begin(); 

    __m256 lower_bound_vec = _mm256_set1_ps(lower_bound);
    __m256 upper_bound_vec = _mm256_set1_ps(upper_bound);

    for (int i = 0; i < size; i += 8) {
        zsim_PIM_function_begin();
        __m256 data = _mm256_loadu_ps(&input[i]);

        __m256 mask = _mm256_and_ps(_mm256_cmp_ps(data, lower_bound_vec, _CMP_GE_OQ),
                                     _mm256_cmp_ps(data, upper_bound_vec, _CMP_LE_OQ));

        _mm256_maskstore_ps(&output[i], _mm256_castps_si256(mask), data);
        zsim_PIM_function_end();
    }

    zsim_roi_end();
}

void filterRange(float* input, float* output, int size, float lower_bound, float upper_bound) {
    zsim_roi_begin(); 

    for(int i = 0 ; i < size ; i++) {
        zsim_PIM_function_begin();
        if(input[i]>=lower_bound && input[i]<=upper_bound) {
            output[i] = input[i];
        }
        else{
            output[i] = 0;
        }
        zsim_PIM_function_end();
    }

    zsim_roi_end();
}

int main() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;


    float input[ARRAY_SIZE];  
    float output[ARRAY_SIZE]; 
    float lower_bound = 10.0f;
    float upper_bound = 2000.0f;

    for(int i = 0 ; i  < ARRAY_SIZE ; i++) {
        input[i] = (float)i;
    }

    auto t1 = high_resolution_clock::now();
    filterRangeAVX2(input, output, ARRAY_SIZE, lower_bound, upper_bound);
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";

    // auto t1 = high_resolution_clock::now();
    // filterRange(input, output, ARRAY_SIZE, lower_bound, upper_bound);
    // auto t2 = high_resolution_clock::now();
    // auto ms_int = duration_cast<milliseconds>(t2 - t1);
    // duration<double, std::milli> ms_double = t2 - t1;
    // std::cout << ms_int.count() << "ms\n";
    // std::cout << ms_double.count() << "ms\n";

    return 0;
}
