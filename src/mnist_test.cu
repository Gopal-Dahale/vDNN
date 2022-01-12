#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "alexnet.h"
#include "cxxopts.hpp"
#include "vgg16.h"
#include "vgg19.h"

using namespace std;

typedef unsigned char uchar;

int num_train = 1024, num_test = 512;

int reverseInt(int n) {
  int bytes = 4;
  unsigned char ch[bytes];
  for (int i = 0; i < bytes; i++) {
    ch[i] = (n >> i * 8) & 255;
  }
  int p = 0;
  for (int i = 0; i < bytes; i++) {
    p += (int)ch[i] << (bytes - i - 1) * 8;
  }
  return p;
}

void readMNIST(vector<vector<uchar>> &train_images,
               vector<vector<uchar>> &test_images, vector<uchar> &train_labels,
               vector<uchar> &test_labels) {
  string filename_train_images = "data/train-images.idx3-ubyte";
  string filename_train_labels = "data/train-labels.idx1-ubyte";

  string filename_test_images = "data/t10k-images.idx3-ubyte";
  string filename_test_labels = "data/t10k-labels.idx1-ubyte";

  // read train/test images
  for (int i = 0; i < 2; i++) {
    string filename;
    if (i == 0)
      filename = filename_train_images;
    else
      filename = filename_test_images;

    ifstream f(filename.c_str(), ios::binary);
    if (!f.is_open()) printf("Cannot read MNIST from %s\n", filename.c_str());

    // read metadata
    int magic_number = 0, n_images = 0, n_rows = 0, n_cols = 0;
    f.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    f.read((char *)&n_images, sizeof(n_images));
    n_images = reverseInt(n_images);
    f.read((char *)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    f.read((char *)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    for (int k = 0; k < n_images; k++) {
      vector<uchar> temp;
      temp.reserve(n_rows * n_cols);
      for (int j = 0; j < n_rows * n_cols; j++) {
        uchar t = 0;
        f.read((char *)&t, sizeof(t));
        temp.push_back(t);
      }
      if (i == 0)
        train_images.push_back(temp);
      else
        test_images.push_back(temp);
    }
    f.close();
  }

  // read train/test labels
  for (int i = 0; i < 2; i++) {
    string filename;
    if (i == 0)
      filename = filename_train_labels;
    else
      filename = filename_test_labels;

    ifstream f(filename.c_str(), ios::binary);
    if (!f.is_open()) printf("Cannot read MNIST from %s\n", filename.c_str());

    // read metadata
    int magic_number = 0, n_labels = 0;
    f.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    f.read((char *)&n_labels, sizeof(n_labels));
    n_labels = reverseInt(n_labels);

    for (int k = 0; k < n_labels; k++) {
      uchar t = 0;
      f.read((char *)&t, sizeof(t));
      if (i == 0)
        train_labels.push_back(t);
      else
        test_labels.push_back(t);
    }

    f.close();
  }
}

void readMNIST224(vector<vector<uchar>> &train_images,
                  vector<vector<uchar>> &test_images,
                  vector<uchar> &train_labels, vector<uchar> &test_labels,
                  int num_train, int num_test) {
  string filename_train_images =
      "/kaggle/input/mnist224by224testdataset/train-images-224by224-";
  string filename_train_labels = "data/train-labels.idx1-ubyte";

  string filename_test_images =
      "/kaggle/input/mnist224by224testdataset/test-images-224by224-";
  string filename_test_labels = "data/t10k-labels.idx1-ubyte";

  // read train/test images
  int images_per_file = 2000;
  int num_train_files =
      min((int)(ceil(num_train / float(images_per_file))), 30);
  int num_test_files = min((int)(ceil(num_test / float(images_per_file))), 5);

  for (int i = 0; i < 2; i++) {
    int num_files = (i == 0 ? num_train_files : num_test_files);
    for (int j = 0; j < num_files; j++) {
      string filename;
      if (i == 0)
        filename = filename_train_images;
      else
        filename = filename_test_images;
      filename = filename + to_string(j) + ".idx3-ubyte";

      ifstream f(filename.c_str(), ios::binary);
      if (!f.is_open()) printf("Cannot read MNIST from %s\n", filename.c_str());

      // read metadata
      int magic_number = 0, n_images = 0, n_rows = 0, n_cols = 0;
      f.read((char *)&magic_number, sizeof(magic_number));
      magic_number = reverseInt(magic_number);
      f.read((char *)&n_images, sizeof(n_images));
      n_images = reverseInt(n_images);
      f.read((char *)&n_rows, sizeof(n_rows));
      n_rows = reverseInt(n_rows);
      f.read((char *)&n_cols, sizeof(n_cols));
      n_cols = reverseInt(n_cols);

      for (int k = 0; k < n_images; k++) {
        vector<uchar> temp;
        temp.reserve(n_rows * n_cols);
        for (int j = 0; j < n_rows * n_cols; j++) {
          uchar t = 0;
          f.read((char *)&t, sizeof(t));
          temp.push_back(t);
        }
        if (i == 0) {
          train_images.push_back(temp);
          if ((j * n_images + k + 1) >= num_train) break;
        } else {
          test_images.push_back(temp);
          if ((j * n_images + k + 1) >= num_test) break;
        }
      }
      f.close();
    }
  }

  // read train/test labels
  for (int i = 0; i < 2; i++) {
    string filename;
    if (i == 0)
      filename = filename_train_labels;
    else
      filename = filename_test_labels;

    ifstream f(filename.c_str(), ios::binary);
    if (!f.is_open()) printf("Cannot read MNIST from %s\n", filename.c_str());

    // read metadata
    int magic_number = 0, n_labels = 0;
    f.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    f.read((char *)&n_labels, sizeof(n_labels));
    n_labels = reverseInt(n_labels);

    if (i == 0)
      n_labels = min(n_labels, num_train);
    else
      n_labels = min(n_labels, num_test);

    for (int k = 0; k < n_labels; k++) {
      uchar t = 0;
      f.read((char *)&t, sizeof(t));
      if (i == 0)
        train_labels.push_back(t);
      else
        test_labels.push_back(t);
    }
    f.close();
  }

  assert(train_images.size() == train_labels.size());
  assert(test_images.size() == test_labels.size());
}

int main(int argc, char *argv[]) {
  /******************* Parse command line arguments ********************/
  cxxopts::Options options("VDNN", "VDNN");

  options.add_options()(
      "batch-size", "Batch Size",
      cxxopts::value<int>()->default_value("64"))  // a bool parameter
      ("softmax-eps", "softmax eps",
       cxxopts::value<float>()->default_value("1e-8"))(
          "init-std-dev", "initial standard deviation",
          cxxopts::value<float>()->default_value("0.01"))(
          "epochs", "Number of epochs",
          cxxopts::value<int>()->default_value("5"))(
          "learning-rate", "Learning Rate",
          cxxopts::value<double>()->default_value("0.01"))(
          "learning-rate-decay", "Learning Rate Decay",
          cxxopts::value<double>()->default_value("1"))(
          "num-train", "Number of training examples to use",
          cxxopts::value<int>()->default_value("1024"))(
          "num-test", "Number of testing examples to use",
          cxxopts::value<int>()->default_value("512"))(
          "type", "VDNN type", cxxopts::value<int>()->default_value("0"))(
          "algo", "VDNN algo", cxxopts::value<int>()->default_value("0"))(
          "net", "network",
          cxxopts::value<std::string>()->default_value("vgg16"))("help",
                                                                 "Print Usage");

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  num_train = result["num-train"].as<int>();
  num_test = result["num-test"].as<int>();

  auto vDNN_type = vDNN_ALL;
  int type = result["type"].as<int>();
  switch (type) {
    case 0:
      vDNN_type = vDNN_ALL;
      std::cout << "vDNN_ALL" << std::endl;
      break;
    case 1:
      vDNN_type = vDNN_CONV;
      std::cout << "vDNN_CONV" << std::endl;
      break;
    case 2:
      vDNN_type = vDNN_NONE;
      std::cout << "vDNN_NONE" << std::endl;
      break;
    case 3:
      vDNN_type = vDNN_DYN;
      std::cout << "vDNN_DYN" << std::endl;
      break;
    case 4:
      vDNN_type = vDNN_ALTERNATE_CONV;
      std::cout << "vDNN_ALTERNATE_CONV" << std::endl;
    default:
      break;
  }

  auto vDNN_algo = vDNN_PERFORMANCE_OPTIMAL;
  int algo = result["algo"].as<int>();
  switch (algo) {
    case 0:
      vDNN_algo = vDNN_PERFORMANCE_OPTIMAL;
      std::cout << "vDNN_PERFORMANCE_OPTIMAL" << std::endl;
      break;
    case 1:
      vDNN_algo = vDNN_MEMORY_OPTIMAL;
      std::cout << "vDNN_MEMORY_OPTIMAL" << std::endl;
      break;
    default:
      break;
  }

  int rows = 224, cols = 224, channels = 1;
  float *f_train_images, *f_test_images;
  int *f_train_labels, *f_test_labels;
  // int rows = 28, cols = 28, channels = 1;
  int input_size = rows * cols * channels;
  f_train_images = (float *)malloc(num_train * input_size * sizeof(float));
  f_train_labels = (int *)malloc(num_train * sizeof(int));
  f_test_images = (float *)malloc(num_test * input_size * sizeof(float));
  f_test_labels = (int *)malloc(num_test * sizeof(int));

  {
    vector<vector<uchar>> train_images, test_images;
    vector<uchar> train_labels, test_labels;
    readMNIST224(train_images, test_images, train_labels, test_labels,
                 num_train, num_test);

    for (int k = 0; k < num_train; k++) {
      for (int j = 0; j < rows * cols; j++) {
        f_train_images[k * input_size + j] = (float)train_images[k][j];
      }
      f_train_labels[k] = (int)train_labels[k];
    }

    for (int k = 0; k < num_test; k++) {
      for (int j = 0; j < rows * cols; j++) {
        f_test_images[k * input_size + j] = (float)test_images[k][j];
      }
      f_test_labels[k] = (int)test_labels[k];
    }
  }

  float *mean_image;
  mean_image = (float *)malloc(input_size * sizeof(float));

  for (int i = 0; i < input_size; i++) {
    mean_image[i] = 0;
    for (int k = 0; k < num_train; k++) {
      mean_image[i] += f_train_images[k * input_size + i];
    }
    mean_image[i] /= num_train;
  }

  for (int i = 0; i < num_train; i++) {
    for (int j = 0; j < input_size; j++) {
      f_train_images[i * input_size + j] -= mean_image[j];
    }
  }

  for (int i = 0; i < num_test; i++) {
    for (int j = 0; j < input_size; j++) {
      f_test_images[i * input_size + j] -= mean_image[j];
    }
  }

  VGG16 vgg16;
  auto layer_specifier = vgg16.layer_specifier;
  auto neural_net = result["net"].as<std::string>();
  if (neural_net == "vgg19") {
    VGG19 vgg19;
    layer_specifier = vgg19.layer_specifier;
    std::cout << "Network: VGG19" << std::endl;
  } else if (neural_net == "alexnet") {
    Alexnet alexnet;
    layer_specifier = alexnet.layer_specifier;
    std::cout << "Network: Alexnet" << std::endl;
  } else {
    std::cout << "Network: VGG16" << std::endl;
  }

  /**************************** Configuration ****************************/
  int batch_size = result["batch-size"].as<int>();
  float softmax_eps = result["softmax-eps"].as<float>();
  float init_std_dev = result["init-std-dev"].as<float>();
  int num_epoch = result["epochs"].as<int>();
  double learning_rate = result["learning-rate"].as<double>();
  double learning_rate_decay = result["learning-rate-decay"].as<double>();
  long long dropout_seed = 1;
  /************************ Display configuration *************************/
  unordered_map<string, double> configs = {
      {"batch_size", batch_size},
      {"softmax_eps", softmax_eps},
      {"init_std_dev", init_std_dev},
      {"num_epoch", num_epoch},
      {"learning_rate", learning_rate},
      {"learning_rate_decay", learning_rate_decay},
      {"num_train", num_train},
      {"num_test", num_test}};

  for (auto &config : configs) {
    cout << config.first << ": " << config.second << endl;
  }

  /************** Write network and batch size to file *******************/
  std::fstream f;
  f.open("./res/config.txt", std::fstream::out);
  f << "vgg16" << std::endl;
  f << batch_size << std::endl;
  f.close();

  NeuralNet net(layer_specifier, DATA_FLOAT, batch_size, TENSOR_NCHW,
                dropout_seed, softmax_eps, init_std_dev, vDNN_type, vDNN_algo,
                SGD);

  cudaEvent_t train_start, train_stop;
  cudaEventCreate(&train_start);
  cudaEventCreate(&train_stop);
  float train_time = 0;

  Solver solver(&net, (void *)f_train_images, f_train_labels,
                (void *)f_train_images, f_train_labels, num_epoch, SGD,
                learning_rate, learning_rate_decay, num_train, num_train);

  vector<float> loss;
  vector<int> val_acc;

  cudaEventRecord(train_start);

  solver.train(loss, val_acc);

  cudaEventRecord(train_stop);
  cudaEventSynchronize(train_stop);
  cudaEventElapsedTime(&train_time, train_start, train_stop);

  // Write the total execution time (train_time) to file
  f.open("./res/config.txt", std::ios::in);
  std::string net_name;
  f >> net_name;
  int bs;
  f >> bs;
  f.close();
  std::string file_name =
      "./res/total_time" + net_name + "_" + std::to_string(bs) + ".txt";
  f.open(file_name, std::ios::out);
  f << train_time << std::endl;
  f.close();

  int num_correct;
  solver.checkAccuracy(f_train_images, f_train_labels, num_train, &num_correct);
  cout << num_correct << endl;
}
