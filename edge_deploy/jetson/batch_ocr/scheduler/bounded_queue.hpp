// Queue occupancy and live-buffer reservations are different resources.
#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace autoocr {
using QueueClock = std::chrono::steady_clock;
using QueueDeadline = QueueClock::time_point;

class ByteBudget {
    struct State {
        explicit State(std::size_t n) : limit(n) {}
        std::mutex mutex;
        std::condition_variable ready;
        const std::size_t limit;
        std::size_t used = 0, high = 0;
        bool closed = false;
    };
    std::shared_ptr<State> state_;
public:
    class Lease {
        friend class ByteBudget;
        std::shared_ptr<State> state_;
        std::size_t bytes_ = 0;
        Lease(std::shared_ptr<State> state, std::size_t bytes) noexcept
            : state_(std::move(state)), bytes_(bytes) {}
    public:
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept : state_(std::move(other.state_)), bytes_(std::exchange(other.bytes_, 0)) {}
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) { reset(); state_ = std::move(other.state_); bytes_ = std::exchange(other.bytes_, 0); }
            return *this;
        }
        ~Lease() { reset(); }
        explicit operator bool() const { return bool(state_); }
        std::size_t bytes() const { return bytes_; }
        void reset() noexcept {
            auto state = std::move(state_);
            if (!state) return;
            { std::lock_guard<std::mutex> lock(state->mutex); state->used -= bytes_; }
            bytes_ = 0;
            state->ready.notify_all();
        }
    };
    struct Stats { std::size_t used, high, limit; bool closed; };
    explicit ByteBudget(std::size_t limit) : state_(std::make_shared<State>(limit)) {
        if (!limit) throw std::invalid_argument("zero byte budget");
    }
    ByteBudget(const ByteBudget&) = delete;
    ByteBudget& operator=(const ByteBudget&) = delete;
    ~ByteBudget() { close(); }
    Lease acquire(std::size_t bytes, QueueDeadline deadline) {
        auto state = state_;
        if (!bytes || bytes > state->limit) throw std::invalid_argument("invalid reservation size");
        std::unique_lock<std::mutex> lock(state->mutex);
        if (QueueClock::now() >= deadline || !state->ready.wait_until(lock, deadline, [&] {
                return state->closed || bytes <= state->limit - state->used;
            }) || state->closed || QueueClock::now() >= deadline) return {};
        state->used += bytes; state->high = std::max(state->high, state->used);
        return Lease(state, bytes);
    }
    void close() noexcept {
        auto state = state_;
        { std::lock_guard<std::mutex> lock(state->mutex); state->closed = true; }
        state->ready.notify_all();
    }
    Stats stats() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return {state_->used, state_->high, state_->limit, state_->closed};
    }
};

template<class T> class BoundedQueue {
    struct Entry {
        std::unique_ptr<T> value;
        std::size_t bytes;
        Entry(std::unique_ptr<T>&& v, std::size_t n) noexcept : value(std::move(v)), bytes(n) {}
    };
    const std::size_t max_items_, max_bytes_;
    std::deque<Entry> items_;
    std::size_t bytes_ = 0, high_items_ = 0, high_bytes_ = 0;
    bool closed_ = false;
    mutable std::mutex mutex_;
    std::condition_variable readable_, writable_;
public:
    struct Stats { std::size_t items, bytes, high_items, high_bytes; bool closed; };
    BoundedQueue(std::size_t count, std::size_t bytes) : max_items_(count), max_bytes_(bytes) {
        if (!count || !bytes) throw std::invalid_argument("zero queue capacity");
    }
    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;
    bool push(std::unique_ptr<T>&& task, std::size_t bytes, QueueDeadline deadline) {
        if (!task || !bytes || bytes > max_bytes_) throw std::invalid_argument("invalid queue task/weight");
        std::unique_lock<std::mutex> lock(mutex_);
        if (QueueClock::now() >= deadline || !writable_.wait_until(lock, deadline, [&] {
                return closed_ || (items_.size() < max_items_ && bytes <= max_bytes_ - bytes_);
            }) || closed_ || QueueClock::now() >= deadline) return false;
        items_.emplace_back(std::move(task), bytes);
        bytes_ += bytes; high_items_ = std::max(high_items_, items_.size()); high_bytes_ = std::max(high_bytes_, bytes_);
        lock.unlock(); readable_.notify_one();
        return true;
    }
    bool pop(std::unique_ptr<T>& task, QueueDeadline deadline) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (QueueClock::now() >= deadline || !readable_.wait_until(lock, deadline, [&] {
                return closed_ || !items_.empty();
            }) || QueueClock::now() >= deadline || items_.empty()) return false;
        auto received = std::move(items_.front().value);
        bytes_ -= items_.front().bytes; items_.pop_front();
        lock.unlock(); writable_.notify_one();
        task = std::move(received); // Release the caller's old task outside the queue lock.
        return true;
    }
    void close(bool drain = true) {
        std::deque<Entry> discarded;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
            if (!drain) { discarded.swap(items_); bytes_ = 0; }
        }
        readable_.notify_all(); writable_.notify_all();
        // Discarded tasks release leases after unlocking, not inside the queue mutex.
    }
    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {items_.size(), bytes_, high_items_, high_bytes_, closed_};
    }
};
} // namespace autoocr
