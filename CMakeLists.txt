cmake_minimum_required(VERSION 2.8)
project( vDNNNeuralNet )
include_directories( include /usr/include /usr/local/cuda/include /usr/include/x86_64-linux-gnu /usr/local/cuda/samples/common/inc )

link_directories( /usr/local/cuda/lib64/ )

find_package(CUDA)



cuda_add_executable( mnist_test.out src/mnist_test.cu src/solver.cu src/neural_net.cu src/neural_net_time.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp)
target_link_libraries( mnist_test.out -lcudnn -lcublas -lcurand )




