// Compile-only diagnostic. This is never a model/benchmark kernel or GPU load.
extern "C" __global__ void embodied_eval_toolchain_probe(float *value) {
    value[threadIdx.x] = 1.0f;
}
