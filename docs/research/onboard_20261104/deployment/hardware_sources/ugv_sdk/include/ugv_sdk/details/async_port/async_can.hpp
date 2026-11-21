/*
 * async_can.hpp
 * Copyright (c) 2020 Weston Robot Pte. Ltd.
 * Local NSO commissioning patch: owned, serialized CAN writes and shutdown.
 */
#ifndef ASYNC_CAN_HPP
#define ASYNC_CAN_HPP

#include <linux/can.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace westonrobot {
class AsyncCAN : public std::enable_shared_from_this<AsyncCAN> {
 public:
  using ReceiveCallback = std::function<void(can_frame *rx_frame)>;
  explicit AsyncCAN(std::string can_port = "can0");
  ~AsyncCAN();
  AsyncCAN(const AsyncCAN &) = delete;
  AsyncCAN &operator=(const AsyncCAN &) = delete;

  bool Open();
  // External callers wait for all handlers; an IO-thread caller only requests
  // shutdown. The external owner must later Close/destroy to join that thread.
  void Close();
  bool IsOpened() const;
  void SetReceiveCallback(ReceiveCallback cb);
  void SendFrame(const struct can_frame &frame);

 private:
  struct State;
  std::string port_;
  std::shared_ptr<State> state_;
  ReceiveCallback receive_callback_;
  std::thread io_thread_;
  // Never held by the IO thread. Serializes external Open/Close and joining.
  std::mutex lifecycle_mutex_;
  // Serializes producer admission and posting; never held while joining.
  mutable std::mutex submission_mutex_;
};
}  // namespace westonrobot
#endif
