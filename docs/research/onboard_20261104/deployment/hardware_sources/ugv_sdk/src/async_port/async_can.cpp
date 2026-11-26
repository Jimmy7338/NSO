/*
 * async_can.cpp
 * Copyright (c) 2020 Weston Robot Pte. Ltd.
 * Local NSO commissioning patch; source baseline recorded in source-manifest.
 */
#include "ugv_sdk/details/async_port/async_can.hpp"

#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <linux/can/raw.h>
#include <atomic>
#include <cstring>
#include <deque>
#include <iostream>
#include <utility>

#include "asio.hpp"
#include "asio/posix/basic_stream_descriptor.hpp"

namespace westonrobot {
namespace {
#if ASIO_VERSION < 101200L
using IoContext = asio::io_service;
#else
using IoContext = asio::io_context;
#endif
template <typename Handler>
void Post(IoContext &io, Handler handler) {
#if ASIO_VERSION < 101200L
  io.post(std::move(handler));
#else
  asio::post(io, std::move(handler));
#endif
}
}  // namespace

// Handlers own State rather than AsyncCAN. The external AsyncCAN destructor
// can therefore cancel and join without being kept alive by its own handlers.
struct AsyncCAN::State : public std::enable_shared_from_this<State> {
  IoContext io;
  asio::posix::basic_stream_descriptor<> socket;
  std::unique_ptr<IoContext::work> keep_alive;
  std::atomic<bool> accepting{false};
  std::atomic<bool> closed{false};
  bool closing = false;  // IO thread only, except before it starts/after join.
  bool write_active = false;
  can_frame receive_frame{};
  ReceiveCallback receive_callback;
  std::deque<std::shared_ptr<const can_frame>> pending;
  static thread_local State *current;
  static const std::size_t kMaximumPendingFrames = 32;

  State() : socket(io), keep_alive(new IoContext::work(io)) {}

  void CloseOnIo() {
    if (closing) return;
    accepting.store(false);
    closing = true;
    pending.clear();
    // The active handler independently owns its frame until cancellation has
    // completed. Only the descriptor closes its assigned fd: no raw ::close.
    asio::error_code ignored;
    socket.cancel(ignored);
    socket.close(ignored);
    closed.store(true);
    keep_alive.reset();
  }

  void Read() {
    if (closing) return;
    auto self = shared_from_this();
    socket.async_read_some(
        asio::buffer(&receive_frame, sizeof(receive_frame)),
        [self](asio::error_code error, std::size_t bytes) {
          if (self->closing) return;
          if (error || bytes != sizeof(can_frame)) {
            self->CloseOnIo();
            return;
          }
          try {
            if (self->receive_callback) self->receive_callback(&self->receive_frame);
          } catch (...) {
            self->CloseOnIo();
            return;
          }
          self->Read();
        });
  }

  void Enqueue(std::shared_ptr<const can_frame> frame) {
    if (closing) return;
    if (pending.size() >= kMaximumPendingFrames) {
      std::cerr << "CAN send queue overflow; closing transport" << std::endl;
      CloseOnIo();
      return;
    }
    pending.push_back(std::move(frame));
    WriteNext();
  }

  void WriteNext() {
    if (closing || write_active || pending.empty()) return;
    auto frame = pending.front();
    pending.pop_front();
    write_active = true;
    auto self = shared_from_this();
    // CAN frames are indivisible records. A short write is a transport failure;
    // do not use async_write to retry a fragment as another CAN frame.
    socket.async_write_some(asio::buffer(frame.get(), sizeof(can_frame)),
        [self, frame](asio::error_code error, std::size_t bytes) {
          self->write_active = false;
          if (self->closing) return;
          if (error || bytes != sizeof(can_frame)) {
            std::cerr << "CAN frame write failed; closing transport" << std::endl;
            self->CloseOnIo();
            return;
          }
          self->WriteNext();
        });
  }
};
thread_local AsyncCAN::State *AsyncCAN::State::current = nullptr;

AsyncCAN::AsyncCAN(std::string can_port)
    : port_(std::move(can_port)), state_(std::make_shared<State>()) {}

AsyncCAN::~AsyncCAN() {
  Close();
  // If ownership is released inside a receive callback, State (not this) owns
  // every remaining handler and the worker lambda. It may finish independently.
  if (io_thread_.joinable() && State::current == state_.get()) io_thread_.detach();
}

bool AsyncCAN::Open() {
  std::lock_guard<std::mutex> lifecycle(lifecycle_mutex_);
  std::lock_guard<std::mutex> admission(submission_mutex_);
  if (io_thread_.joinable() || state_->accepting.load()) return false;
  state_ = std::make_shared<State>();
  state_->receive_callback = receive_callback_;
  int fd = -1;
  try {
    if (port_.empty() || port_.size() >= IFNAMSIZ) return false;
    fd = ::socket(PF_CAN, SOCK_RAW | SOCK_NONBLOCK, CAN_RAW);
    if (fd < 0) return false;
    struct ifreq ifr{};
    std::memcpy(ifr.ifr_name, port_.c_str(), port_.size() + 1);
    if (::ioctl(fd, SIOCGIFINDEX, &ifr) < 0) {
      ::close(fd);
      return false;
    }
    struct sockaddr_can address{};
    address.can_family = AF_CAN;
    address.can_ifindex = ifr.ifr_ifindex;
    if (::bind(fd, reinterpret_cast<struct sockaddr *>(&address), sizeof(address)) < 0) {
      ::close(fd);
      return false;
    }
    asio::error_code error;
    state_->socket.assign(fd, error);
    if (error) {
      ::close(fd);
      return false;
    }
    fd = -1;  // Descriptor exclusively owns the fd from this point.
    auto state = state_;
    Post(state->io, [state]() { state->Read(); });
    state->accepting.store(true);
    io_thread_ = std::thread([state]() {
      State::current = state.get();
      try {
        state->io.run();
      } catch (...) {
        state->CloseOnIo();
        // Drain cancellation callbacks before releasing their buffers.
        state->io.reset();
        state->io.run();
      }
      State::current = nullptr;
    });
    return true;
  } catch (const std::exception &error) {
    if (fd >= 0) ::close(fd);
    // Thread construction failed, so no IO worker can race this cleanup.
    state_->CloseOnIo();
    state_->io.reset();
    state_->io.poll();
    std::cerr << error.what() << std::endl;
    return false;
  }
}

void AsyncCAN::Close() {
  {
    std::lock_guard<std::mutex> admission(submission_mutex_);
    if (State::current == state_.get()) {
      state_->CloseOnIo();
      return;
    }
  }
  std::lock_guard<std::mutex> lifecycle(lifecycle_mutex_);
  std::shared_ptr<State> state;
  {
    std::lock_guard<std::mutex> admission(submission_mutex_);
    state = state_;
    state->accepting.store(false);
    if (io_thread_.joinable()) {
      if (!state->closed.load()) Post(state->io, [state]() { state->CloseOnIo(); });
    } else {
      state->CloseOnIo();
    }
  }
  if (io_thread_.joinable()) io_thread_.join();
  // A producer admitted just before an IO error may have posted after run()
  // returned. No thread remains; discard those closed-state handlers safely.
  state->io.reset();
  state->io.poll();
}

bool AsyncCAN::IsOpened() const {
  std::lock_guard<std::mutex> admission(submission_mutex_);
  return state_->accepting.load();
}

void AsyncCAN::SetReceiveCallback(ReceiveCallback callback) {
  std::lock_guard<std::mutex> admission(submission_mutex_);
  receive_callback_ = std::move(callback);
  auto state = state_;
  auto copied_callback = receive_callback_;
  if (state->accepting.load()) {
    Post(state->io, [state, copied_callback]() {
      if (!state->closing) state->receive_callback = copied_callback;
    });
  }
}

void AsyncCAN::SendFrame(const struct can_frame &frame) {
  std::lock_guard<std::mutex> admission(submission_mutex_);
  auto state = state_;
  if (!state->accepting.load()) return;
  // Copy while the caller's stack object is still alive, before posting.
  std::shared_ptr<const can_frame> owned = std::make_shared<can_frame>(frame);
  Post(state->io, [state, owned]() { state->Enqueue(owned); });
}
}  // namespace westonrobot
