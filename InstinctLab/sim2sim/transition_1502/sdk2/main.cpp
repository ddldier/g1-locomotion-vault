#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include "onnxruntime_cxx_api.h"
#include "unitree/dds_wrapper/robots/g1/g1.h"

namespace {
using Clock = std::chrono::steady_clock;
constexpr int kScan = 527;
constexpr int kJoints = 29;
constexpr int kObservation = 2049;
constexpr int kPeriodMs = 20;
constexpr int kScanMaxAgeMs = 100;

// Frozen actor order, mapped to Unitree SDK2's official 29-DoF motor IDs.
constexpr std::array<int, kJoints> kMotorIds = {
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};
constexpr std::array<float, kJoints> kDefault = {
    -0.312f, -0.312f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    0.669f, 0.669f, 0.2f, -0.2f, -0.363f, -0.363f,
    0.2f, -0.2f, 0.f, 0.f, 0.f, 0.f, 0.6f, 0.6f,
    0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

template <std::size_t N>
class History {
 public:
  explicit History(std::size_t length) : length_(length) {}
  void push(const std::array<float, N>& sample) {
    if (values_.empty()) values_.assign(length_, sample);
    else { values_.pop_front(); values_.push_back(sample); }
  }
  void append(std::vector<float>& out) const {
    for (const auto& sample : values_) out.insert(out.end(), sample.begin(), sample.end());
  }
  void clear() { values_.clear(); }
 private:
  std::size_t length_;
  std::deque<std::array<float, N>> values_;
};

struct State {
  std::array<float, 4> quaternion{};
  std::array<float, 3> gyro{};
  std::array<float, kJoints> q{};
  std::array<float, kJoints> dq{};
};

std::array<float, 3> projected_gravity(const std::array<float, 4>& q) {
  const auto [w, x, y, z] = q;
  const float norm = w*w + x*x + y*y + z*z;
  if (!std::isfinite(norm) || std::abs(norm - 1.f) > 0.1f)
    throw std::runtime_error("Invalid IMU quaternion");
  return {2.f*(w*y-x*z), -2.f*(w*x+y*z), 2.f*(x*x+y*y)-1.f};
}

class Observation {
 public:
  void reset() {
    height_.clear(); gravity_.clear(); gyro_.clear(); pos_.clear();
    vel_.clear(); action_.clear(); last_action_.fill(0.f);
  }
  std::vector<float> step(const State& state, const std::array<float, kScan>& scan,
                          const std::array<float, 3>& command) {
    std::array<float, kScan> clipped = scan;
    for (auto& h : clipped) h = std::clamp(h, -1.f, 1.f);
    std::array<float, kJoints> pos{};
    for (int i=0; i<kJoints; ++i) pos[i] = state.q[i] - kDefault[i];
    height_.push(clipped);
    gravity_.push(projected_gravity(state.quaternion));
    gyro_.push(state.gyro);
    pos_.push(pos);
    vel_.push(state.dq);
    action_.push(last_action_);
    std::vector<float> out;
    out.reserve(kObservation);
    height_.append(out);
    out.insert(out.end(), command.begin(), command.end());
    gravity_.append(out);
    gyro_.append(out);
    pos_.append(out);
    vel_.append(out);
    action_.append(out);
    if (out.size() != kObservation) throw std::runtime_error("Observation length mismatch");
    for (float v : out) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite observation");
    return out;
  }
  void set_last_action(const std::array<float, kJoints>& action) { last_action_ = action; }
 private:
  History<kScan> height_{3};
  History<3> gravity_{5}, gyro_{5};
  History<kJoints> pos_{5}, vel_{5}, action_{5};
  std::array<float, kJoints> last_action_{};
};

class Actor {
 public:
  explicit Actor(const std::string& path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "g1_sdk2_probe"), session_(env_, path.c_str(), options_) {
    const auto in = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const auto out = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (in.size()!=2 || in[1]!=kObservation || out.size()!=2 || out[1]!=kJoints)
      throw std::runtime_error("ONNX input/output must be [*,2049] -> [*,29]");
    input_ = session_.GetInputNameAllocated(0, allocator_).get();
    output_ = session_.GetOutputNameAllocated(0, allocator_).get();
  }
  std::array<float, kJoints> infer(std::vector<float>& observation) {
    const std::array<int64_t, 2> shape{1, kObservation};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory, observation.data(), observation.size(), shape.data(), shape.size());
    const char* input_names[] = {input_.c_str()};
    const char* output_names[] = {output_.c_str()};
    auto result = session_.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
    std::array<float, kJoints> action{};
    std::memcpy(action.data(), result.front().GetTensorData<float>(), sizeof(action));
    for (float v : action) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite actor action");
    return action;
  }
 private:
  Ort::Env env_;
  Ort::SessionOptions options_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::string input_, output_;
};

class ScanSocket {
 public:
  explicit ScanSocket(int port) {
    fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_<0) throw std::runtime_error("Cannot open scan socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
      throw std::runtime_error("Cannot bind scan socket");
    fcntl(fd_, F_SETFL, fcntl(fd_, F_GETFL, 0) | O_NONBLOCK);
  }
  ~ScanSocket() { close(fd_); }
  bool latest(std::array<float, kScan>& scan) {
    // Leave one byte of headroom so oversized datagrams are rejected rather
    // than silently truncated to a packet that looks valid.
    std::array<std::uint8_t, sizeof(float) * kScan + 1> packet{};
    while (true) {
      const auto n = recv(fd_, packet.data(), packet.size(), 0);
      if (n < 0) break;
      if (n == static_cast<ssize_t>(sizeof(float) * kScan)) {
        std::memcpy(scan_.data(), packet.data(), sizeof(scan_));
        received_ = Clock::now();
      }
    }
    if (Clock::now()-received_ > std::chrono::milliseconds(kScanMaxAgeMs)) return false;
    scan = scan_;
    for (float v : scan) if (!std::isfinite(v)) return false;
    return true;
  }
 private:
  int fd_ = -1;
  std::array<float, kScan> scan_{};
  Clock::time_point received_{};
};

void self_test(const std::string& model_path) {
  std::array<bool, kJoints> used{};
  for (int id : kMotorIds) {
    if (id < 0 || id >= kJoints || used[id]) throw std::runtime_error("Invalid motor map");
    used[id] = true;
  }
  Actor actor(model_path);
  Observation obs;
  State state{};
  state.quaternion = {1.f,0.f,0.f,0.f};
  state.q = kDefault;
  std::array<float, kScan> scan{};
  std::array<float, 3> command{1.f,0.f,0.f};
  auto input = obs.step(state, scan, command);
  auto action = actor.infer(input);
  obs.set_last_action(action);
  input = obs.step(state, scan, command);
  if (input.size()!=kObservation || input[kScan*3]!=1.f)
    throw std::runtime_error("Observation contract self-test failed");
  std::cout << "SDK2 probe self-test passed: 29 unique motor IDs, 2049 inputs, 29 actions.\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 5) {
      std::cerr << "Usage: " << argv[0] << " <actor.onnx> --self-test | <actor.onnx> --network <interface> [scan_port]\n";
      return 2;
    }
    const std::string model_path = argv[1];
    if (std::string(argv[2])=="--self-test") { self_test(model_path); return 0; }
    if (std::string(argv[2])!="--network" || argc<4) throw std::runtime_error("Expected --network <interface>");
    const int port = argc==5 ? std::stoi(argv[4]) : 5556;
    if (port<1 || port>65535) throw std::runtime_error("Invalid scan port");
    Actor actor(model_path);
    ScanSocket scans(port);
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[3]);
    auto lowstate = std::make_shared<unitree::robot::g1::subscription::LowState>();
    std::cout << "Waiting for SDK2 rt/lowstate; this probe NEVER publishes rt/lowcmd.\n";
    lowstate->wait_for_connection();
    Observation obs;
    std::size_t count = 0;
    bool had_scan = false;
    bool had_lowstate = false;
    std::uint32_t last_tick = 0;
    bool have_tick = false;
    auto next = Clock::now();
    while (true) {
      next += std::chrono::milliseconds(kPeriodMs);
      std::array<float, kScan> scan{};
      const bool scan_ready = scans.latest(scan);
      if (lowstate->isTimeout() || !scan_ready) {
        if (had_scan || had_lowstate || count==0)
          std::cout << "Waiting for fresh LowState and 527-float scan on 127.0.0.1:" << port << "\n";
        had_scan = false;
        had_lowstate = false;
        have_tick = false;
        obs.reset();
        std::this_thread::sleep_until(next);
        continue;
      }
      had_scan = true;
      State state;
      std::uint32_t tick = 0;
      {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        const auto& msg = lowstate->msg_;
        tick = msg.tick();
        if (msg.mode_machine()!=0 && msg.mode_machine()!=5) throw std::runtime_error("Expected G1 29-DoF mode_machine=5");
        for (int i=0; i<4; ++i) state.quaternion[i] = msg.imu_state().quaternion()[i];
        for (int i=0; i<3; ++i) state.gyro[i] = msg.imu_state().gyroscope()[i];
        for (int i=0; i<kJoints; ++i) {
          state.q[i] = msg.motor_state()[kMotorIds[i]].q();
          state.dq[i] = msg.motor_state()[kMotorIds[i]].dq();
        }
      }
      // Do not run the policy repeatedly on a frozen DDS sample.
      if (have_tick && tick == last_tick) {
        std::this_thread::sleep_until(next);
        continue;
      }
      have_tick = true;
      last_tick = tick;
      had_lowstate = true;
      auto input = obs.step(state, scan, {1.f,0.f,0.f});
      auto action = actor.infer(input);
      obs.set_last_action(action);
      if (++count % 50 == 0) std::cout << "inferences=" << count << " action[0]=" << action[0] << " (dry run)\n";
      std::this_thread::sleep_until(next);
    }
  } catch (const std::exception& e) {
    std::cerr << "SDK2 probe error: " << e.what() << '\n';
    return 1;
  }
}
