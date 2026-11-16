#include "bounded_queue.hpp"
#include <atomic>
#include <dirent.h>
#include <exception>
#include <future>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace autoocr;
using Milliseconds = std::chrono::milliseconds;
QueueDeadline later(int ms = 2000) { return QueueClock::now() + Milliseconds(ms); }
void check_at(bool value, int line) {
    if (!value) throw std::runtime_error("scheduler assertion line " + std::to_string(line));
}
#define check(value) check_at((value), __LINE__)
template<class F> void invalid(F work) {
    bool caught = false;
    try { work(); } catch (const std::invalid_argument&) { caught = true; }
    check(caught);
}
struct Task {
    unsigned id;
    ByteBudget::Lease lease;
    std::vector<unsigned char> data;
    Task(unsigned id, ByteBudget::Lease lease) : id(id), lease(std::move(lease)), data(16, id & 255) {}
};
std::unique_ptr<Task> task(ByteBudget& budget, unsigned id = 0) {
    auto lease = budget.acquire(16, later()); check(bool(lease));
    return std::unique_ptr<Task>(new Task(id, std::move(lease)));
}

void bounds_and_deadlines() {
    invalid([] { ByteBudget b(0); });
    invalid([] { BoundedQueue<Task> q(0,16); });
    ByteBudget b(32); BoundedQueue<Task> q(2,32);
    invalid([&] { b.acquire(0,later()); });
    invalid([&] { b.acquire(std::numeric_limits<std::size_t>::max(),later()); });
    auto a = task(b);
    invalid([&] { q.push(std::move(a),0,later()); }); check(bool(a));
    invalid([&] { q.push(std::move(a),std::numeric_limits<std::size_t>::max(),later()); });
    check(!q.push(std::move(a),16,later(-1)) && bool(a));
    auto none = b.acquire(16,later(-1)); check(!none);
    a.reset(); check(b.stats().used == 0);
}

void byte_and_count_backpressure() {
    ByteBudget b(128); BoundedQueue<Task> q(4,24);
    auto a=task(b,1), c=task(b,2); check(q.push(std::move(a),16,later()));
    check(!q.push(std::move(c),16,later(20)) && bool(c));
    std::unique_ptr<Task> out; check(q.pop(out,later()));
    check(q.push(std::move(c),16,later()));
    check(q.stats().high_items==1 && q.stats().high_bytes==16);
    q.close(false); out.reset(); check(b.stats().used==0);
    BoundedQueue<Task> one(1,128); a=task(b); c=task(b);
    check(one.push(std::move(a),16,later()));
    check(!one.push(std::move(c),16,later(20)) && bool(c));
    one.close(false); c.reset(); check(b.stats().used==0);
}

void dequeue_keeps_live_reservation() {
    ByteBudget b(16); BoundedQueue<Task> q(1,16);
    auto a=task(b); check(q.push(std::move(a),16,later()));
    std::unique_ptr<Task> active; check(q.pop(active,later()));
    check(q.stats().bytes==0 && b.stats().used==16);
    check(!b.acquire(1,later(20)));
    active.reset(); auto next=b.acquire(16,later()); check(bool(next));
    next.reset(); next.reset(); check(b.stats().used==0);
}

void close_wakes_push_pop_and_credit() {
    ByteBudget b(64); BoundedQueue<Task> full(1,16), empty(1,16);
    auto a=task(b); check(full.push(std::move(a),16,later()));
    auto producer=std::async(std::launch::async,[&] {
        auto waiting=task(b); bool ok=full.push(std::move(waiting),16,later());
        return !ok && bool(waiting);
    });
    auto consumer=std::async(std::launch::async,[&] { std::unique_ptr<Task> out; return empty.pop(out,later()); });
    check(producer.wait_for(Milliseconds(20))==std::future_status::timeout);
    check(consumer.wait_for(Milliseconds(20))==std::future_status::timeout);
    full.close(false); empty.close();
    check(producer.get()); check(!consumer.get()); check(b.stats().used==0);
    auto held=b.acquire(64,later());
    auto credit=std::async(std::launch::async,[&] { return bool(b.acquire(1,later())); });
    check(credit.wait_for(Milliseconds(20))==std::future_status::timeout);
    b.close(); check(!credit.get()); held.reset(); check(b.stats().used==0);
}

void drain_cancel_and_lease_lifetime() {
    ByteBudget b(64); BoundedQueue<Task> q(4,64);
    auto a=task(b,1), c=task(b,2); check(q.push(std::move(a),16,later()) && q.push(std::move(c),16,later()));
    q.close(); std::unique_ptr<Task> out;
    check(q.pop(out,later()) && out->id==1); check(q.pop(out,later()) && out->id==2);
    check(!q.pop(out,later())); out.reset(); check(b.stats().used==0);
    BoundedQueue<Task> cancelled(4,64); a=task(b); c=task(b);
    check(cancelled.push(std::move(a),16,later())); check(cancelled.push(std::move(c),16,later()));
    check(cancelled.pop(out,later())); cancelled.close(false); check(b.stats().used==16);
    out.reset(); check(b.stats().used==0);
    ByteBudget::Lease late;
    { ByteBudget local(32); auto first=local.acquire(16,later()); late=std::move(first); check(!first && late.bytes()==16); }
    late.reset(); check(!late);
}

void page_admission_reservations() {
    constexpr std::size_t mib=1024*1024;
    ByteBudget pages(512*mib);
    auto a=pages.acquire(256*mib,later()), b=pages.acquire(256*mib,later());
    check(bool(a) && bool(b) && pages.stats().used==512*mib);
    check(!pages.acquire(256*mib,later(20)));
    a.reset(); auto c=pages.acquire(256*mib,later()); check(bool(c));
    b.reset(); c.reset(); check(pages.stats().used==0 && pages.stats().high==512*mib);
    // Reservations are counters, not 512MiB of test buffers.
}

void synthetic_two_one_two_pipeline() {
    constexpr unsigned count=80;
    ByteBudget budget(128); BoundedQueue<Task> ready(4,64), done(4,64);
    std::atomic<unsigned> next{0}; std::atomic<bool> failed{false};
    std::atomic<unsigned> live_workers{0};
    std::mutex error_mutex, result_mutex;
    std::exception_ptr error;
    std::vector<unsigned> ordered(count,0), completions;
    std::promise<void> first_result;
    auto first_done=first_result.get_future().share();
    auto guarded=[&](auto work) {
        struct Running {
            std::atomic<unsigned>& count;
            explicit Running(std::atomic<unsigned>& n) : count(n) { ++count; }
            ~Running() { --count; }
        } running(live_workers);
        try { work(); } catch (...) {
            { std::lock_guard<std::mutex> lock(error_mutex); if (!error) error=std::current_exception(); }
            failed=true; budget.close(); ready.close(false); done.close(false);
        }
    };
    auto produce=[&] { guarded([&] {
        for (;;) {
            auto id=next.fetch_add(1); if (id>=count || failed) return;
            auto lease=budget.acquire(16,later()); if (!lease && failed) return;
            check(bool(lease)); auto t=std::unique_ptr<Task>(new Task(id,std::move(lease)));
            if (!ready.push(std::move(t),16,later()) && !failed) check(false);
        }
    }); };
    auto execute=[&] { guarded([&] {
        std::unique_ptr<Task> t;
        while (ready.pop(t,later())) {
            for (auto& byte:t->data) byte ^= 0x5a; // Simulated stage; no GPU API or OCR.
            if (!done.push(std::move(t),16,later()) && !failed) check(false);
        }
        done.close();
    }); };
    auto finish=[&] { guarded([&] {
        std::unique_ptr<Task> t;
        while (done.pop(t,later())) {
            for (auto byte:t->data) check(byte==((t->id & 255)^0x5a));
            if (t->id==0) check(first_done.wait_for(Milliseconds(1000))==std::future_status::ready);
            if (t->id%3==0) std::this_thread::sleep_for(Milliseconds(1));
            { std::lock_guard<std::mutex> lock(result_mutex); check(!ordered[t->id]); ordered[t->id]=t->id+1; completions.push_back(t->id); }
            if (t->id==1) first_result.set_value();
            t.reset();
        }
    }); };
    std::thread in1(produce), in2(produce), simulated(execute), out1(finish), out2(finish);
    in1.join(); in2.join(); ready.close(); simulated.join(); out1.join(); out2.join();
    if (error) std::rethrow_exception(error);
    check(!failed && completions.size()==count && budget.stats().used==0 && live_workers==0);
    check(std::find(completions.begin(),completions.end(),1)<std::find(completions.begin(),completions.end(),0));
    for (unsigned i=0;i<count;++i) check(ordered[i]==i+1);
    auto a=ready.stats(), b=done.stats();
    check(a.high_items<=4 && b.high_items<=4 && a.high_bytes<=64 && b.high_bytes<=64 && budget.stats().high<=128);
    std::cout << "{\"synthetic_tasks\":80,\"input_threads\":2,\"simulated_executor\":1,\"result_threads\":2,\"live_bytes_end\":0,\"live_workers_end\":0}\n";
}

unsigned entries(const char* path) {
    DIR* dir=::opendir(path); if (!dir) throw std::runtime_error("proc inventory failed");
    unsigned n=0; while (auto* entry=::readdir(dir)) if (entry->d_name[0]!='.') ++n;
    ::closedir(dir); return n;
}
int main() {
    const char* phase="start";
    try {
        auto fds=entries("/proc/self/fd"), threads=entries("/proc/self/task");
        phase="bounds"; bounds_and_deadlines();
        phase="backpressure"; byte_and_count_backpressure();
        phase="live_reservation"; dequeue_keeps_live_reservation();
        phase="close_wakeup"; close_wakes_push_pop_and_credit();
        phase="drain_cancel"; drain_cancel_and_lease_lifetime();
        phase="page_reservation"; page_admission_reservations();
        phase="synthetic_pipeline"; synthetic_two_one_two_pipeline();
        phase="proc_inventory";
        auto final_fds=entries("/proc/self/fd"), final_threads=entries("/proc/self/task");
        const auto initial_threads=final_threads;
        const auto settle_start=QueueClock::now(), settle_deadline=settle_start+Milliseconds(1000);
        // /proc is sampled after join; keep an exact bounded end-state check, not an immediate snapshot assertion.
        while (final_threads!=threads && QueueClock::now()<settle_deadline) {
            std::this_thread::sleep_for(Milliseconds(1)); final_threads=entries("/proc/self/task");
        }
        auto settle_us=std::chrono::duration_cast<std::chrono::microseconds>(QueueClock::now()-settle_start).count();
        if (final_fds!=fds || final_threads!=threads)
            std::cerr << "inventory fds=" << final_fds << "/" << fds << " threads=" << final_threads << "/" << threads << "\n";
        check(final_fds==fds && final_threads==threads);
        std::cout << "{\"status\":\"scheduler_primitives_passed_not_ocr\",\"test_groups\":7,\"fds\":" << fds << ",\"threads\":" << threads
                  << ",\"initial_proc_threads\":" << initial_threads << ",\"proc_settle_us\":" << settle_us << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "scheduler test phase=" << phase << " " << error.what() << "\n"; return 1;
    }
}
