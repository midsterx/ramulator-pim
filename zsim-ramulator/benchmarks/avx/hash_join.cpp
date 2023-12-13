#include <immintrin.h>
#include <cstdint>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <random>
#include <chrono>
#include <thread>
#include <cstdint>

#include "/users/mthevend/mnt/CloudStorage/midhush/ramulator-pim/zsim-ramulator/misc/hooks/zsim_hooks.h"

#define ARRAY_SIZE 1000000
#define NUM_BUCKETS 1024

void hashJoinAVX(const int* keys, const int* values, const int* keys_tb2, const int* values_tb2, int* hashTable, int* result) {
    zsim_roi_begin(); 
    zsim_PIM_function_begin();

    for (int i = 0; i < NUM_BUCKETS; ++i) {
        hashTable[i] = -1;
    }

    for (int i = 0; i < ARRAY_SIZE; i += 8) {
        __m256i keyVector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&keys[i]));
        __m256i hash = _mm256_add_epi32(keyVector, _mm256_set1_epi32(NUM_BUCKETS));
        hash = _mm256_and_si256(hash, _mm256_set1_epi32(NUM_BUCKETS - 1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&hashTable[i]), hash);
    }

    for (int i = 0; i < ARRAY_SIZE; i += 8) {
        __m256i keyVector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&keys_tb2[i]));
        __m256i hash = _mm256_add_epi32(keyVector, _mm256_set1_epi32(NUM_BUCKETS));
        hash = _mm256_and_si256(hash, _mm256_set1_epi32(NUM_BUCKETS - 1));

        __m256i resultVector = _mm256_i32gather_epi32(&hashTable[0], hash, 4);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i]), resultVector);
    }

    zsim_PIM_function_end();
    zsim_roi_end();
}

void hashJoin(const int* keys, const int* values, const int* keys_tb2, const int* values_tb2, int* hashTable, int* result) {
    zsim_roi_begin(); 
    zsim_PIM_function_begin();

    for (int i = 0; i < NUM_BUCKETS; ++i) {
        hashTable[i] = -1;
    }

    for (int i = 0; i < ARRAY_SIZE; ++i) {
        int hash = (keys[i] + NUM_BUCKETS) & (NUM_BUCKETS - 1);
        hashTable[hash] = values[i];
    }

    for (int i = 0; i < ARRAY_SIZE; ++i) {
        int hash = (keys_tb2[i] + NUM_BUCKETS) & (NUM_BUCKETS - 1);
        result[i] = (hashTable[hash] != -1) ? hashTable[hash] : 0;  // Assuming 0 as a default value for not found
    }

    zsim_PIM_function_end();
    zsim_roi_end();
}

int main() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;


    int keys[ARRAY_SIZE];     
    int values[ARRAY_SIZE];

    int keys_tb2[ARRAY_SIZE];
    int values_tb2[ARRAY_SIZE];

    int hashTable[NUM_BUCKETS];    
    int result[ARRAY_SIZE];

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> distribution(1, 100);

    for(int i = 0 ; i < ARRAY_SIZE ; i++) {
        keys[i] = distribution(mt);
        values[i] = distribution(mt);

        keys_tb2[i] = distribution(mt);
        values_tb2[i] = distribution(mt);
    }

    // Call the AVX hash join function
    auto t1 = high_resolution_clock::now();
    hashJoinAVX(keys, values, keys_tb2, values_tb2, hashTable, result);
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";

    // t1 = high_resolution_clock::now();
    // hashJoin(keys, values, keys_tb2, values_tb2, hashTable, result);
    // t2 = high_resolution_clock::now();
    // ms_int = duration_cast<milliseconds>(t2 - t1);
    // ms_double = t2 - t1;
    // std::cout << ms_int.count() << "ms\n";
    // std::cout << ms_double.count() << "ms\n";

    return 0;
}
