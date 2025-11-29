cmake_minimum_required(VERSION 3.18)
project(NeuralNetCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Găsim pachetele CUDA necesare
find_package(CUDAToolkit REQUIRED)

# Setăm arhitectura GPU (poți ajusta în funcție de placa ta, ex: 75, 86)
set(CMAKE_CUDA_ARCHITECTURES native)

# Include directories
include_directories(include)

# Colectăm sursele
file(GLOB_RECURSE SOURCES "source/*.cpp" "source/*.cu")

# Executabilul
add_executable(neural_net ${SOURCES} ${CMAKE_SOURCE_DIR}/main.cpp)

# Linkare cu biblioteci CUDA (runtime și math library dacă e cazul)
target_link_libraries(neural_net PRIVATE CUDA::cudart)