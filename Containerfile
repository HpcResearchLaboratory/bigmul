FROM nvcr.io/nvidia/nvhpc:24.7-devel-cuda_multi-ubuntu22.04 AS build

WORKDIR /bigmul
COPY Makefile .
COPY include/ include/
COPY src/ src/

ARG CUDA_ARCH=sm_89
RUN echo "NVCCFLAGS = -O2 -std=c++20 -arch=${CUDA_ARCH} -Xcompiler -static-libstdc++" > config.mk \
    && make all \
    && make bench

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /bigmul
COPY --from=build /bigmul/build/bin/ build/bin/
COPY --from=build /bigmul/build/lib/ build/lib/
COPY script/ script/
