// Linux IPC framing only. JSON is opaque here; Python validates the probe schema.
#pragma once
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace autoocr {
using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;
constexpr std::uint32_t max_meta = 4096;
constexpr std::uint32_t max_payload = 64 * 1024 * 1024;
using Header = std::array<unsigned char, 24>;

struct ProtocolError : std::runtime_error { using std::runtime_error::runtime_error; };
struct PeerClosed : std::runtime_error { using std::runtime_error::runtime_error; };
struct Timeout : std::runtime_error { using std::runtime_error::runtime_error; };

class Fd {
    int value_;
public:
    explicit Fd(int value) : value_(value) { if (value < 0) throw std::runtime_error("socket failed"); }
    ~Fd() { if (value_ >= 0) ::close(value_); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    int get() const { return value_; }
};

inline void bounds(std::uint64_t id, std::uint32_t meta, std::uint32_t payload) {
    if (!id || id > std::uint64_t(std::numeric_limits<std::int64_t>::max()) ||
        !meta || meta > max_meta || payload > max_payload)
        throw ProtocolError("invalid frame bounds");
}

inline void put(Header& h, unsigned offset, std::uint64_t value, unsigned count) {
    for (unsigned i = 0; i < count; ++i) { h[offset + count - i - 1] = value & 255; value >>= 8; }
}
inline std::uint64_t get(const Header& h, unsigned offset, unsigned count) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < count; ++i) value = (value << 8) | h[offset + i];
    return value;
}
inline Header make_header(std::uint64_t id, std::uint32_t meta, std::uint32_t payload) {
    bounds(id, meta, payload);
    Header h{};
    std::memcpy(h.data(), "AOCR", 4);
    put(h, 4, 1, 2); put(h, 8, meta, 4); put(h, 12, payload, 4); put(h, 16, id, 8);
    return h;
}
struct Sizes { std::uint64_t id; std::uint32_t meta, payload; };
inline Sizes parse_header(const Header& h) {
    if (std::memcmp(h.data(), "AOCR", 4) || get(h, 4, 2) != 1 || get(h, 6, 2))
        throw ProtocolError("unknown protocol/version/flags");
    Sizes s{get(h, 16, 8), std::uint32_t(get(h, 8, 4)), std::uint32_t(get(h, 12, 4))};
    bounds(s.id, s.meta, s.payload);
    return s;
}

inline void wait_ready(int fd, short events, Deadline deadline) {
    for (;;) {
        auto left = deadline - Clock::now();
        if (left <= Clock::duration::zero()) throw Timeout("IPC deadline exceeded");
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(left).count() + 1;
        int timeout = int(std::min<long long>(ms, std::numeric_limits<int>::max()));
        pollfd p{fd, events, 0};
        int result = ::poll(&p, 1, timeout);
        if (result < 0 && errno == EINTR) continue;
        if (result < 0 || (p.revents & (POLLERR | POLLNVAL))) throw PeerClosed("IPC poll failed");
        if (!result) continue;
        if (p.revents & (events | POLLHUP)) {
            if (Clock::now() >= deadline) throw Timeout("IPC deadline exceeded");
            return;
        }
    }
}

inline void transfer(int fd, unsigned char* bytes, std::size_t count, bool sending, Deadline deadline) {
    std::size_t offset = 0;
    while (offset < count) {
        wait_ready(fd, sending ? POLLOUT : POLLIN, deadline);
        ssize_t n = sending ? ::send(fd, bytes + offset, count - offset, MSG_DONTWAIT | MSG_NOSIGNAL)
                            : ::recv(fd, bytes + offset, count - offset, MSG_DONTWAIT);
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (n <= 0) throw PeerClosed("IPC peer closed/error");
        offset += std::size_t(n);
    }
}

struct Frame { std::uint64_t id; std::string metadata; std::vector<unsigned char> payload; };
inline void send_frame(int fd, const Frame& frame, Deadline deadline) {
    if (frame.metadata.size() > max_meta || frame.payload.size() > max_payload)
        throw ProtocolError("oversized outbound frame");
    Header h = make_header(frame.id, std::uint32_t(frame.metadata.size()), std::uint32_t(frame.payload.size()));
    transfer(fd, h.data(), h.size(), true, deadline);
    transfer(fd, reinterpret_cast<unsigned char*>(const_cast<char*>(frame.metadata.data())), frame.metadata.size(), true, deadline);
    transfer(fd, const_cast<unsigned char*>(frame.payload.data()), frame.payload.size(), true, deadline);
    if (Clock::now()>=deadline) throw Timeout("IPC deadline exceeded");
}
template<class Validator> Frame receive_frame(int fd, Deadline deadline, Validator validate) {
    Header h{}; transfer(fd, h.data(), h.size(), false, deadline);
    auto s = parse_header(h);
    Frame frame{s.id, std::string(s.meta, '\0'), {}};
    transfer(fd, reinterpret_cast<unsigned char*>(&frame.metadata[0]), s.meta, false, deadline);
    validate(frame.metadata, s.payload);
    if (Clock::now()>=deadline) throw Timeout("IPC deadline exceeded");
    frame.payload.resize(s.payload);
    transfer(fd, frame.payload.data(), frame.payload.size(), false, deadline);
    if (Clock::now()>=deadline) throw Timeout("IPC deadline exceeded");
    return frame;
}
inline Frame receive_frame(int fd, Deadline deadline) {
    // shortcut: legacy echo probe keeps metadata opaque; business callers must pass a validator.
    return receive_frame(fd, deadline, [](const std::string&, std::uint32_t) {});
}

inline void check_peer(int fd) {
    ucred peer{}; socklen_t count = sizeof(peer);
    if (::getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &count) || count != sizeof(peer) || peer.uid != ::geteuid())
        throw ProtocolError("IPC peer UID differs/unavailable");
}
inline void connect_local(int fd, const std::string& path, Deadline deadline) {
    sockaddr_un address{}; address.sun_family = AF_UNIX;
    if (path.empty() || path[0] != '/' || path.size() >= sizeof(address.sun_path))
        throw ProtocolError("invalid socket path");
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    int result = ::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
    if (result < 0 && errno != EINPROGRESS) throw PeerClosed("IPC connect failed/busy");
    if (result < 0) {
        wait_ready(fd, POLLOUT, deadline);
        int error = 0; socklen_t size = sizeof(error);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) || error) throw PeerClosed("IPC connect error");
    }
    check_peer(fd);
}
} // namespace autoocr
