#include "ipc_wire.hpp"
#include <iostream>
#include <thread>

using namespace autoocr;

Frame synthetic(std::uint64_t id) {
    Frame frame{id, R"({"dtype":"uint8","image_id":"synthetic_1","layout":"HWC","op":"probe","shape":[2,4,3],"stride":12})", {}};
    for (unsigned i = 0; i < 24; ++i) frame.payload.push_back((i + id) & 255);
    return frame;
}

void require(bool value) { if (!value) throw ProtocolError("probe assertion failed"); }
template<class E, class F> void rejects(F function) {
    bool caught = false;
    try { function(); } catch (const E&) { caught = true; }
    require(caught);
}

void self_test() {
    auto h = make_header(0x0102030405060708ULL, 17, 31);
    require(h[16] == 1 && h[23] == 8 && parse_header(h).payload == 31);
    rejects<ProtocolError>([] { make_header(0, 1, 1); });
    rejects<ProtocolError>([] { make_header(1, max_meta + 1, 1); });
    rejects<ProtocolError>([] { make_header(1, 1, max_payload + 1); });
    h[6] = 1; rejects<ProtocolError>([&] { parse_header(h); });
    int pair[2]; require(!::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair));
    Fd a(pair[0]), b(pair[1]); check_peer(a.get()); check_peer(b.get());
    std::exception_ptr sender_error;
    std::thread sender([&] {
        try {
            auto deadline = Clock::now() + std::chrono::seconds(2);
            send_frame(b.get(), synthetic(101), deadline); send_frame(b.get(), synthetic(102), deadline);
        } catch (...) { sender_error = std::current_exception(); }
    });
    try {
        auto deadline = Clock::now() + std::chrono::seconds(2);
        for (auto id : {101, 102}) {
            auto frame = receive_frame(a.get(), deadline);
            require(frame.id == unsigned(id) && frame.payload == synthetic(id).payload);
        }
    } catch (...) { sender.join(); throw; }
    sender.join(); if (sender_error) std::rethrow_exception(sender_error);
    rejects<Timeout>([&] { receive_frame(a.get(), Clock::now() + std::chrono::milliseconds(20)); });
    ::shutdown(b.get(), SHUT_RDWR);
    rejects<PeerClosed>([&] { receive_frame(a.get(), Clock::now() + std::chrono::seconds(1)); });
    rejects<PeerClosed>([&] { send_frame(a.get(), synthetic(1), Clock::now() + std::chrono::seconds(1)); });
    std::cout << "{\"status\":\"cpp_transport_self_test_passed\"}\n";
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
        if (argc != 2) throw ProtocolError("socket argument required");
        Fd fd(::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0));
        auto deadline = Clock::now() + std::chrono::seconds(2);
        connect_local(fd.get(), argv[1], deadline);
        for (auto id : {101, 102}) send_frame(fd.get(), synthetic(id), deadline);
        for (auto id : {101, 102}) {
            auto got = receive_frame(fd.get(), deadline), expected = synthetic(id);
            require(got.id == expected.id && got.metadata == expected.metadata && got.payload == expected.payload);
        }
        std::cout << "{\"status\":\"cpp_python_roundtrip_passed\",\"messages\":2,\"payload_bytes\":48}\n";
        return 0;
    } catch (const Timeout&) {
        std::cerr << "{\"status\":\"error\",\"kind\":\"timeout\"}\n";
    } catch (const PeerClosed&) {
        std::cerr << "{\"status\":\"error\",\"kind\":\"peer_closed\"}\n";
    } catch (const std::exception&) {
        std::cerr << "{\"status\":\"error\",\"kind\":\"protocol_or_system\"}\n";
    }
    return 1;
}
