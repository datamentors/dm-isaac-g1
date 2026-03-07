// JoystickInjector - UDP-based virtual joystick for sim2sim
// Listens on 127.0.0.1:15001 for plain-text commands from Python.
//
// Protocol:
//   Button commands:  "lt=1", "rb=0", "x=1", "up=0"
//   Multi-button:     "rb=1 x=0"
//   Axis commands:    "set lx 0.5 ly 0.3 rx 0.1 ry 0.0 lt 0.0 rt 0.0"
//   Hold trigger:     "hold lt 0.8"
//
// Usage in FSMState::pre_run():
//   JoystickInjector::Instance().start();
//   JoystickInjector::Instance().apply(lowstate->joystick);

#pragma once

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include <spdlog/spdlog.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <sstream>
#include <vector>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

class JoystickInjector
{
public:
    static JoystickInjector& Instance()
    {
        static JoystickInjector instance;
        return instance;
    }

    void start()
    {
        if (started_) return;
        started_ = true;

        sock_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_fd_ < 0) {
            spdlog::error("[JoystickInjector] Failed to create UDP socket");
            return;
        }

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        addr.sin_addr.s_addr = inet_addr(host_);

        if (bind(sock_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            spdlog::error("[JoystickInjector] Failed to bind to {}:{}", host_, port_);
            close(sock_fd_);
            sock_fd_ = -1;
            return;
        }

        // Non-blocking
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 1000; // 1ms timeout
        setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        running_ = true;
        recv_thread_ = std::thread([this] {
            spdlog::info("[JoystickInjector] UDP listening on {}:{}", host_, port_);
            char buf[512];
            while (running_) {
                ssize_t n = recvfrom(sock_fd_, buf, sizeof(buf) - 1, 0, nullptr, nullptr);
                if (n > 0) {
                    buf[n] = '\0';
                    std::lock_guard<std::mutex> lock(mtx_);
                    parse_message(std::string(buf));
                }
            }
        });
        recv_thread_.detach();
    }

    void apply(unitree::common::UnitreeJoystick& joy)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        // Build REMOTE_DATA_RX from desired state
        unitree::common::REMOTE_DATA_RX key{};
        std::memset(&key, 0, sizeof(key));

        key.RF_RX.btn.components.L2    = desired_.lt > 0.5f ? 1 : 0;
        key.RF_RX.btn.components.R2    = desired_.rt > 0.5f ? 1 : 0;
        key.RF_RX.btn.components.L1    = desired_.lb;
        key.RF_RX.btn.components.R1    = desired_.rb;
        key.RF_RX.btn.components.A     = desired_.a;
        key.RF_RX.btn.components.B     = desired_.b;
        key.RF_RX.btn.components.X     = desired_.x;
        key.RF_RX.btn.components.Y     = desired_.y;
        key.RF_RX.btn.components.up    = desired_.up;
        key.RF_RX.btn.components.down  = desired_.down;
        key.RF_RX.btn.components.left  = desired_.left;
        key.RF_RX.btn.components.right = desired_.right;
        key.RF_RX.btn.components.Start = desired_.start;
        key.RF_RX.btn.components.Select = desired_.back;
        key.RF_RX.btn.components.f1    = desired_.f1;
        key.RF_RX.btn.components.f2    = desired_.f2;

        key.RF_RX.lx = desired_.lx;
        key.RF_RX.ly = desired_.ly;
        key.RF_RX.rx = desired_.rx;
        key.RF_RX.ry = desired_.ry;
        key.RF_RX.L2 = desired_.lt;

        joy.extract(key);
    }

private:
    JoystickInjector() = default;
    ~JoystickInjector()
    {
        running_ = false;
        if (sock_fd_ >= 0) close(sock_fd_);
    }

    JoystickInjector(const JoystickInjector&) = delete;
    JoystickInjector& operator=(const JoystickInjector&) = delete;

    void parse_message(const std::string& msg)
    {
        auto tokens = split(msg);
        if (tokens.empty()) return;

        // "set lx 0.5 ly 0.3 rx 0.1 ry 0.0 lt 0.0 rt 0.0"
        if (tokens[0] == "set") {
            for (size_t i = 1; i + 1 < tokens.size(); i += 2) {
                const auto& k = tokens[i];
                float v = std::stof(tokens[i + 1]);
                if      (k == "lx") desired_.lx = v;
                else if (k == "ly") desired_.ly = v;
                else if (k == "rx") desired_.rx = v;
                else if (k == "ry") desired_.ry = v;
                else if (k == "lt") desired_.lt = v;
                else if (k == "rt") desired_.rt = v;
            }
            return;
        }

        // "hold lt 0.8"
        if (tokens[0] == "hold" && tokens.size() >= 3) {
            const auto& k = tokens[1];
            float v = std::stof(tokens[2]);
            if      (k == "lt") desired_.lt = v;
            else if (k == "rt") desired_.rt = v;
            return;
        }

        // Button commands: "rb=1", "lt=1 up=0", etc.
        for (const auto& tok : tokens) {
            auto eq = tok.find('=');
            if (eq == std::string::npos) continue;
            std::string name = tok.substr(0, eq);
            int val = std::stoi(tok.substr(eq + 1));
            set_button(name, val);
        }
    }

    void set_button(const std::string& name, int val)
    {
        if      (name == "a")     desired_.a     = val;
        else if (name == "b")     desired_.b     = val;
        else if (name == "x")     desired_.x     = val;
        else if (name == "y")     desired_.y     = val;
        else if (name == "lb")    desired_.lb    = val;
        else if (name == "rb")    desired_.rb    = val;
        else if (name == "lt")    desired_.lt    = static_cast<float>(val);
        else if (name == "rt")    desired_.rt    = static_cast<float>(val);
        else if (name == "up")    desired_.up    = val;
        else if (name == "down")  desired_.down  = val;
        else if (name == "left")  desired_.left  = val;
        else if (name == "right") desired_.right = val;
        else if (name == "start") desired_.start = val;
        else if (name == "back")  desired_.back  = val;
        else if (name == "f1")    desired_.f1    = val;
        else if (name == "f2")    desired_.f2    = val;
    }

    static std::vector<std::string> split(const std::string& s)
    {
        std::vector<std::string> tokens;
        std::istringstream iss(s);
        std::string tok;
        while (iss >> tok) tokens.push_back(tok);
        return tokens;
    }

    struct Desired {
        // Buttons
        int a = 0, b = 0, x = 0, y = 0;
        int lb = 0, rb = 0;
        int up = 0, down = 0, left = 0, right = 0;
        int start = 0, back = 0;
        int f1 = 0, f2 = 0;
        // Axes
        float lx = 0, ly = 0, rx = 0, ry = 0;
        float lt = 0, rt = 0;
    };

    Desired desired_;
    std::mutex mtx_;
    std::atomic<bool> running_{false};
    std::atomic<bool> started_{false};
    int sock_fd_ = -1;
    std::thread recv_thread_;

    static constexpr const char* host_ = "127.0.0.1";
    static constexpr int port_ = 15001;
};
