#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include "onnxruntime_cxx_api.h"

namespace {
constexpr std::size_t kObservationSize = 2049;
constexpr std::size_t kActionSize = 29;

int parse_port(const char* value) {
  const long port = std::stol(value);
  if (port < 1 || port > 65535) throw std::invalid_argument("port must be in [1, 65535]");
  return static_cast<int>(port);
}

class Actor {
 public:
  explicit Actor(const std::string& model_path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "transition_1502"),
        options_(make_session_options()),
        session_(env_, model_path.c_str(), options_) {
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
      throw std::runtime_error("Expected one ONNX input and one output");
    }
    const auto input_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const auto output_shape = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (input_shape.size() != 2 || input_shape[1] != static_cast<int64_t>(kObservationSize) ||
        output_shape.size() != 2 || output_shape[1] != static_cast<int64_t>(kActionSize)) {
      throw std::runtime_error("ONNX dimensions do not match [batch,2049] -> [batch,29]");
    }
    input_name_ = session_.GetInputNameAllocated(0, allocator_).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator_).get();
  }

  std::array<float, kActionSize> infer(const float* observation) {
    std::array<int64_t, 2> shape{1, static_cast<int64_t>(kObservationSize)};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(observation),
        kObservationSize, shape.data(), shape.size());
    const char* inputs[] = {input_name_.c_str()};
    const char* outputs[] = {output_name_.c_str()};
    auto result = session_.Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
    const float* values = result.front().GetTensorData<float>();
    std::array<float, kActionSize> action{};
    std::copy(values, values + kActionSize, action.begin());
    return action;
  }

 private:
  static Ort::SessionOptions make_session_options() {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return options;
  }

  Ort::Env env_;
  Ort::SessionOptions options_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::string input_name_;
  std::string output_name_;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3) {
      std::cerr << "Usage: " << argv[0] << " <actor.onnx> [udp_port]\n";
      return 2;
    }
    const int port = argc == 3 ? parse_port(argv[2]) : 5555;
    Actor actor(argv[1]);

    const int socket_fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd < 0) throw std::runtime_error("socket(): " + std::string(std::strerror(errno)));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
      const std::string error = std::strerror(errno);
      ::close(socket_fd);
      throw std::runtime_error("bind(): " + error);
    }

    std::cout << "C++ ONNX policy ready on UDP " << port
              << "; waiting for 2049-float observations" << std::endl;
    std::vector<float> observation(kObservationSize);
    std::array<float, kActionSize> action{};
    std::array<char, kObservationSize * sizeof(float) + 1> packet{};
    std::uint64_t count = 0;
    while (true) {
      sockaddr_in peer{};
      socklen_t peer_len = sizeof(peer);
      const ssize_t size = ::recvfrom(socket_fd, packet.data(), packet.size(), 0,
          reinterpret_cast<sockaddr*>(&peer), &peer_len);
      if (size < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error("recvfrom(): " + std::string(std::strerror(errno)));
      }
      if (size != static_cast<ssize_t>(kObservationSize * sizeof(float))) continue;
      std::memcpy(observation.data(), packet.data(), static_cast<std::size_t>(size));
      action = actor.infer(observation.data());
      const ssize_t sent = ::sendto(socket_fd, action.data(), sizeof(action), 0,
          reinterpret_cast<sockaddr*>(&peer), peer_len);
      if (sent != static_cast<ssize_t>(sizeof(action))) {
        throw std::runtime_error("sendto() failed while returning action");
      }
      if (++count % 100 == 0) {
        std::cout << "inferences=" << count << " action[0]=" << action[0] << std::endl;
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "policy server error: " << error.what() << std::endl;
    return 1;
  }
}
