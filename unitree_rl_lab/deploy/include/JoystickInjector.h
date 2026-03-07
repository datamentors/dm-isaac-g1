#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

// A tiny UDP -> joystick state injector.
// Listens on 127.0.0.1:15001
// Message examples:
//   "RB=1 X=1"         (hold)
//   "RB=0 X=0"
//   "TAP X"            (one-frame tap)
//   "TAP UP"
//   "HOLD LT 1.0"      (trigger axis: 0..1 -> mapped to L2 bit in packet)
//   "SET lx 0.7 ly -0.3 rx 0 ry 0"

class JoystickInjector {
public:
  static JoystickInjector &Instance() {
    static JoystickInjector inst;
    return inst;
  }

  void start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true))
      return;
    thread_ = std::thread(&JoystickInjector::loop, this);
    thread_.detach();
  }

  // Call once per control loop AFTER lowstate->update()
  //
  // IMPORTANT:
  // We must drive UnitreeJoystick via joy.extract(REMOTE_DATA_RX),
  // because that is where LT/RT are interpreted from L2/R2 *button bits*,
  // and where on_pressed/pressed_time are computed correctly.
  void apply(unitree::common::UnitreeJoystick &joy) {
    Desired d;
    {
      std::lock_guard<std::mutex> lk(mu_);
      d = desired_;
      // taps should only persist for one apply()
      desired_.tap_mask = 0;
    }

    // Optional light debug (prints ~every 500 calls)
    static int _cnt = 0;
    if ((_cnt++ % 500) == 0) {
      std::cout << "[JoystickInjector] apply() running. joy_ptr="
                << (void *)&joy << " desired.lt=" << d.lt
                << " desired.rt=" << d.rt << " hold_mask=" << d.hold_mask
                << " tap_mask=" << d.tap_mask << "\n"
                << std::flush;
    }

    // Build a Unitree REMOTE_DATA_RX packet and let UnitreeJoystick::extract()
    // update all KeyBase/Axes states (pressed, on_pressed, pressed_time,
    // smoothing, etc.)
    unitree::common::REMOTE_DATA_RX key{};
    key.RF_RX.btn.value = 0;

    auto is_on = [&](uint32_t bit) -> uint8_t {
      return ((d.hold_mask & bit) || (d.tap_mask & bit)) ? 1 : 0;
    };

    // Buttons
    key.RF_RX.btn.components.R1 = is_on(B_RB);
    key.RF_RX.btn.components.L1 = is_on(B_LB);
    key.RF_RX.btn.components.A = is_on(B_A);
    key.RF_RX.btn.components.B = is_on(B_B);
    key.RF_RX.btn.components.X = is_on(B_X);
    key.RF_RX.btn.components.Y = is_on(B_Y);
    key.RF_RX.btn.components.up = is_on(B_UP);
    key.RF_RX.btn.components.down = is_on(B_DOWN);
    key.RF_RX.btn.components.left = is_on(B_LEFT);
    key.RF_RX.btn.components.right = is_on(B_RIGHT);
    key.RF_RX.btn.components.Start = is_on(B_START);
    key.RF_RX.btn.components.Select = is_on(B_BACK);
    key.RF_RX.btn.components.f1 = is_on(B_F1);
    key.RF_RX.btn.components.f2 = is_on(B_F2);

    // IMPORTANT: triggers in UnitreeJoystick::extract() come from button bits:
    //   LT(key.RF_RX.btn.components.L2);
    //   RT(key.RF_RX.btn.components.R2);
    // So map our float triggers to those bits.
    key.RF_RX.btn.components.L2 = (d.lt > 0.5f) ? 1 : 0;
    key.RF_RX.btn.components.R2 = (d.rt > 0.5f) ? 1 : 0;

    // Axes
    key.RF_RX.lx = d.lx;
    key.RF_RX.ly = d.ly;
    key.RF_RX.rx = d.rx;
    key.RF_RX.ry = d.ry;

    // Apply to joystick (this updates pressed/on_pressed/pressed_time
    // correctly)
    joy.extract(key);

    // Optional debug to prove LT is actually becoming pressed
    static int _k = 0;
    if ((_k++ % 500) == 0) {
      std::cout << "[JoystickInjector] after apply: LT.pressed="
                << joy.LT.pressed << " LT.on_pressed=" << joy.LT.on_pressed
                << " LT.pressed_time=" << joy.LT.pressed_time << "\n"
                << std::flush;
    }
  }

private:
  JoystickInjector() = default;

  // Bitmask mapping for tap/hold buttons
  static constexpr uint32_t B_RB = 1u << 0;
  static constexpr uint32_t B_LB = 1u << 1;
  static constexpr uint32_t B_A = 1u << 2;
  static constexpr uint32_t B_B = 1u << 3;
  static constexpr uint32_t B_X = 1u << 4;
  static constexpr uint32_t B_Y = 1u << 5;
  static constexpr uint32_t B_UP = 1u << 6;
  static constexpr uint32_t B_DOWN = 1u << 7;
  static constexpr uint32_t B_LEFT = 1u << 8;
  static constexpr uint32_t B_RIGHT = 1u << 9;
  static constexpr uint32_t B_START = 1u << 10;
  static constexpr uint32_t B_BACK = 1u << 11;
  static constexpr uint32_t B_F1 = 1u << 12;
  static constexpr uint32_t B_F2 = 1u << 13;

  struct Desired {
    uint32_t hold_mask{0};
    uint32_t tap_mask{0};
    float lx{0}, ly{0}, rx{0}, ry{0};
    float lt{0}, rt{0}; // 0..1 triggers
  };

  std::atomic<bool> started_{false};
  std::mutex mu_;
  Desired desired_{};
  std::thread thread_;

  static uint32_t key_to_bit(const std::string &k) {
    std::string s = k;
    for (auto &c : s)
      c = (char)tolower((unsigned char)c);

    if (s == "rb" || s == "r1")
      return B_RB;
    if (s == "lb" || s == "l1")
      return B_LB;
    if (s == "a")
      return B_A;
    if (s == "b")
      return B_B;
    if (s == "x")
      return B_X;
    if (s == "y")
      return B_Y;
    if (s == "up")
      return B_UP;
    if (s == "down")
      return B_DOWN;
    if (s == "left")
      return B_LEFT;
    if (s == "right")
      return B_RIGHT;
    if (s == "start")
      return B_START;
    if (s == "back" || s == "select")
      return B_BACK;
    if (s == "f1")
      return B_F1;
    if (s == "f2")
      return B_F2;

    return 0;
  }

  void loop() {
    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0)
      return;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(15001);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (bind(sock, (sockaddr *)&addr, sizeof(addr)) < 0) {
      perror("[JoystickInjector] bind");
      close(sock);
      return;
    }

    std::cout << "[JoystickInjector] UDP listening on 127.0.0.1:15001\n"
              << std::flush;

    char buf[1024];
    while (true) {
      int n = recv(sock, buf, sizeof(buf) - 1, 0);
      if (n <= 0)
        continue;
      buf[n] = '\0';
      handle_line(std::string(buf));
    }
  }

  void handle_line(const std::string &line) {
    // Very small parser:
    // - "TAP X"
    // - "HOLD LT 1.0"
    // - "SET lx 0.1 ly -0.2 ..."
    // - "RB=1 X=0 up=1"
    std::cout << "[JoystickInjector] rx: " << line << "\n" << std::flush;

    auto tok = [&](const std::string &s) {
      std::vector<std::string> out;
      std::string cur;
      for (char c : s) {
        if (isspace((unsigned char)c)) {
          if (!cur.empty()) {
            out.push_back(cur);
            cur.clear();
          }
        } else {
          cur.push_back(c);
        }
      }
      if (!cur.empty())
        out.push_back(cur);
      return out;
    };

    auto t = tok(line);
    if (t.empty())
      return;

    std::lock_guard<std::mutex> lk(mu_);

    auto cmd = t[0];
    for (auto &c : cmd)
      c = (char)tolower((unsigned char)c);

    if (cmd == "tap" && t.size() >= 2) {
      uint32_t b = key_to_bit(t[1]);
      if (b)
        desired_.tap_mask |= b;
      return;
    }

    if (cmd == "hold" && t.size() >= 3) {
      std::string key = t[1];
      for (auto &c : key)
        c = (char)tolower((unsigned char)c);

      float v = std::stof(t[2]);
      if (key == "lt")
        desired_.lt = v;
      else if (key == "rt")
        desired_.rt = v;
      return;
    }

    if (cmd == "set") {
      // pairs: name value
      for (size_t i = 1; i + 1 < t.size(); i += 2) {
        std::string k = t[i];
        for (auto &c : k)
          c = (char)tolower((unsigned char)c);
        float v = std::stof(t[i + 1]);

        if (k == "lx")
          desired_.lx = v;
        else if (k == "ly")
          desired_.ly = v;
        else if (k == "rx")
          desired_.rx = v;
        else if (k == "ry")
          desired_.ry = v;
        else if (k == "lt")
          desired_.lt = v;
        else if (k == "rt")
          desired_.rt = v;
      }
      return;
    }

    // Default: parse key=value tokens
    for (const auto &s : t) {
      auto eq = s.find('=');
      if (eq == std::string::npos)
        continue;

      std::string k = s.substr(0, eq);
      std::string v = s.substr(eq + 1);
      for (auto &c : k)
        c = (char)tolower((unsigned char)c);

      if (k == "lx")
        desired_.lx = std::stof(v);
      else if (k == "ly")
        desired_.ly = std::stof(v);
      else if (k == "rx")
        desired_.rx = std::stof(v);
      else if (k == "ry")
        desired_.ry = std::stof(v);
      else if (k == "lt")
        desired_.lt = std::stof(v);
      else if (k == "rt")
        desired_.rt = std::stof(v);
      else {
        uint32_t b = key_to_bit(k);
        if (!b)
          continue;

        int iv = std::stoi(v);
        if (iv)
          desired_.hold_mask |= b;
        else
          desired_.hold_mask &= ~b;
      }
    }
  }
};
