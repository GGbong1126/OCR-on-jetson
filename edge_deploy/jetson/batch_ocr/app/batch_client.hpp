#pragma once
#include "../ipc/stage_json.hpp"
#include "../scheduler/bounded_queue.hpp"
#include <atomic>
#include <csignal>
#include <functional>
#include <thread>
#ifdef AUTOOCR_JPEG
#include "../jpeg_pixels.hpp"
#endif

namespace autoocr { namespace batch {
using stage::Json;
constexpr std::size_t mib=1024*1024, page_bytes=256*mib;

inline Json validate(const std::string& raw, std::size_t bytes) {
    auto m=stage::object(raw,max_meta,16);
    stage::need(m.at("schema")=="ocr.batch/1");
    stage::number(m.at("remaining_ms"),1,600000);
    if (m.at("op")=="batch.health") { stage::need(m.size()==3 && bytes==0); return m; }
    stage::need(m.at("op")=="batch.submit" && m.size()==4);
    const auto& rows=m.at("images"); stage::need(rows.is_array() && !rows.empty() && rows.size()<=4);
    std::set<std::string> seen; std::uint64_t total=0;
    for (const auto& row:rows) {
        stage::need(row.is_object() && row.size()==3 && stage::token(row.at("image_id")));
        stage::need(seen.insert(stage::string(row.at("image_id"))).second);
        stage::number(row.at("image_type"),1,3);
        total+=stage::number(row.at("encoded_bytes"),1,16*mib);
    }
    stage::need(total==bytes && total<=64*mib); return m;
}

class Channel {
    Fd fd_{::socket(AF_UNIX,SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC,0)};
    std::mutex mutex_;
    std::uint64_t id_=1;
public:
    std::string session,schema="ocr.stage/1";
    Channel(const std::string& path,const std::string& role,const std::string& expected="",unsigned rec_slots=0) {
        if (rec_slots) schema="ocr.stage/2";
        auto until=Clock::now()+std::chrono::seconds(5); connect_local(fd_.get(),path,until);
        auto hello=stage::receive(fd_.get(),until); stage::need(hello.meta.at("op")=="session.ready");
        stage::need(hello.meta.at("schema")==schema);
        if (rec_slots) stage::need(hello.meta.at("rec_slots")==rec_slots);
        session=stage::string(hello.meta.at("session")); stage::need(expected.empty() || expected==session);
        auto m=meta("channel.bind",300000); m["channel"]=role;
        stage::send(fd_.get(),Frame{1,m.dump(),{}},until);
        auto ack=stage::receive(fd_.get(),until);
        stage::need(ack.frame.id==1 && ack.meta.at("op")=="channel.bound" && ack.meta.at("channel")==role && ack.meta.at("session")==session);
    }
    int fd() const { return fd_.get(); }
    Json meta(const std::string& op,std::uint64_t ms) const {
        return Json{{"schema",schema},{"op",op},{"session",session},{"remaining_ms",ms}};
    }
    void shutdown() { ::shutdown(fd(),SHUT_RDWR); }
    stage::Message rpc(const std::string& op,Json extra,std::vector<unsigned char> body,Deadline page_deadline) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto left=std::chrono::duration_cast<std::chrono::milliseconds>(page_deadline-Clock::now()).count();
        if (left<=0) throw Timeout("page expired");
        auto m=meta(op,std::min<std::uint64_t>(left,300000));
        for (auto it=extra.begin();it!=extra.end();++it) m[it.key()]=it.value();
        auto until=std::min(page_deadline,Clock::now()+std::chrono::seconds(5));
        auto id=++id_; stage::send(fd(),Frame{id,m.dump(),std::move(body)},until);
        auto reply=stage::receive(fd(),until);
        stage::need(reply.frame.id==id && reply.meta.at("session")==session && reply.meta.at("schema")==schema);
        stage::need(reply.meta.at("op")== (op=="image.collect" ? "image.result":"rpc.reply"));
        if (op=="image.collect") stage::need(reply.meta.at("image_id")==extra.at("image_id"));
        else stage::need(reply.meta.at("for_op")==op);
        return reply;
    }
};

struct Job {
    std::string id;
    int kind;
    Deadline deadline;
    ByteBudget::Lease active, result_credit;
    Json result;
    bool admitted=false, submitted=false, final_event=false, done=false,det_seen=false;
    Deadline created=Clock::now();
    double admission_wait_ms=0,wall_ms=0;
};
struct Input { std::shared_ptr<Job> job; std::shared_ptr<std::vector<unsigned char>> bytes; std::size_t offset,size; };
struct Event { Json meta; };

inline bool pre_det_dependency(const Json& det,const Json& other,bool seen) {
    return det.at("model")=="det" && det.at("image_type")==other.at("image_type") && !seen &&
        std::set<std::string>{"layout","cell","yolo","enhance"}.count(stage::string(other.at("model")));
}

class Client {
    Channel input_,gpu_,result_,events_;
    BoundedQueue<Input> input_q_{4,64*mib};
    BoundedQueue<Event> ready_q_{4,64*mib}, result_q_{4,32*mib};
    std::deque<Json> rec_ready_;
    std::size_t rec_bytes_=0,rec_high_items_=0,rec_high_bytes_=0;
    ByteBudget active_, ingress_{64*mib}, results_{64*mib}, transport_{64*mib};
    std::vector<std::thread> threads_;
    std::atomic<bool> stopping_{false};
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::map<std::string,std::shared_ptr<Job>> jobs_;
    std::map<std::uint64_t,std::shared_ptr<Job>> calls_;
    std::set<std::uint64_t> running_;
    std::map<unsigned,std::uint64_t> rec_running_;
    unsigned rec_slots_=0;
    bool legacy_running_=false;
    std::uint64_t rec_submitted_=0,rec_high_slots_=0;
    std::uint64_t event_id_=1, model_runs_=0, batches_=0, det_pairs_=0,det_predecessors_=0;
    bool det_batch_,hardware_jpeg_;
    std::uint64_t jpeg_used_=0,jpeg_software_=0,jpeg_failed_=0;
    double jpeg_decode_ms_=0,jpeg_color_ms_=0,jpeg_total_ms_=0;
    std::set<std::thread::id> input_threads_,result_threads_;
    std::string failure_;
    std::vector<ByteBudget::Lease> delivered_results_;
    std::atomic<bool>* interrupted_;

    template<class F> void guarded(F function) {
        try { function(); }
        catch (const Timeout&) { stop("TIMEOUT"); }
        catch (...) { stop("DISCONNECTED"); }
    }
    static void ok(const stage::Message& reply) { stage::need(reply.meta.at("status")=="ok" && reply.meta.at("code")=="OK"); }
    static std::size_t weight(const Json& m) {
        std::size_t n=0; for (const auto& v:m.at("inputs")) n+=stage::number(v.at("bytes"),1,max_payload); return n;
    }
    void push_result(Json meta,std::size_t bytes,Deadline until) {
        auto task=std::unique_ptr<Event>(new Event{std::move(meta)});
        if (!result_q_.push(std::move(task),bytes,until)) throw Timeout("result queue full/closed");
    }
    void inputs() {
#ifdef AUTOOCR_JPEG
        JpegDecoder decoder;
#endif
        { std::lock_guard<std::mutex> lock(mutex_); input_threads_.insert(std::this_thread::get_id()); }
        while (!stopping_) {
            std::unique_ptr<Input> task;
            if (!input_q_.pop(task,Clock::now()+std::chrono::milliseconds(200))) continue;
            auto job=task->job; auto credit=active_.acquire(rec_slots_ ? 128*mib:page_bytes,job->deadline);
            if (!credit) throw Timeout("page credit unavailable");
            { std::lock_guard<std::mutex> lock(mutex_); job->active=std::move(credit); job->admitted=true;
              job->admission_wait_ms=std::chrono::duration<double,std::milli>(Clock::now()-job->created).count(); }
            auto transfer=transport_.acquire(hardware_jpeg_ ? 32*mib:task->size,job->deadline);
            if (!transfer) throw Timeout("input transfer credit");
            Json fields{{"image_id",job->id},{"image_type",job->kind},{"format","encoded"}};
            std::vector<unsigned char> body;
#ifdef AUTOOCR_JPEG
            if (hardware_jpeg_) {
                auto start=Clock::now();
                auto decoded=decoder.decode(task->bytes->data()+task->offset,task->size);
                { std::lock_guard<std::mutex> lock(mutex_);
                  jpeg_total_ms_+=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                  jpeg_decode_ms_+=decoded.decode_ms; jpeg_color_ms_+=decoded.color_ms;
                  if (!decoded.status) ++jpeg_used_; else if (decoded.status==3) ++jpeg_software_; else ++jpeg_failed_; }
                if (decoded.status==2) {
                    auto result_credit=results_.acquire(stage::result_limit,job->deadline);
                    if (!result_credit) throw Timeout("invalid image result credit");
                    transfer.reset(); task.reset();
                    std::lock_guard<std::mutex> lock(mutex_);
                    job->result=Json{{"image_id",job->id},{"status","error"},{"code","STATE"}};
                    job->result_credit=std::move(result_credit); job->active.reset();
                    job->done=job->submitted=job->final_event=true;
                    job->wall_ms=std::chrono::duration<double,std::milli>(Clock::now()-job->created).count();
                    changed_.notify_all(); continue;
                }
                if (!decoded.status) {
                    fields["format"]="bgr8"; fields["shape"]=Json::array({decoded.height,decoded.width,3});
                    fields["stride"]=decoded.width*3; body=std::move(decoded.bgr);
                }
            }
#endif
            if (body.empty()) body.assign(task->bytes->begin()+task->offset,task->bytes->begin()+task->offset+task->size);
            ok(input_.rpc("image.submit",fields,std::move(body),job->deadline));
            transfer.reset(); task.reset();
            { std::lock_guard<std::mutex> lock(mutex_); job->submitted=true; changed_.notify_all(); }
        }
    }
    void dispatch() {
        if (rec_slots_) { dispatch_multi(); return; }
        std::unique_ptr<Event> pending;
        while (!stopping_) {
            std::unique_ptr<Event> task=std::move(pending);
            if (!task && !ready_q_.pop(task,Clock::now()+std::chrono::milliseconds(200))) continue;
            auto id=stage::number(task->meta.at("call_id")); std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                changed_.wait(lock,[&] { return stopping_ || running_.empty(); });
                if (stopping_) return;
                auto found=calls_.find(id); if (found==calls_.end()) continue;
                job=found->second; if (job->final_event || Clock::now()>=job->deadline) continue;
            }
            auto ids=Json::array({id}); auto until=job->deadline;
            if (det_batch_ && task->meta.at("model")=="det") {
                std::unique_ptr<Event> other;
                // Two IPC/resume turns plus CPU preparation can exceed 10ms even with a ready peer page.
                if (ready_q_.pop(other,std::min(until,Clock::now()+std::chrono::milliseconds(100)))) {
                    auto a=task->meta.at("inputs"),b=other->meta.at("inputs");
                    for (auto& value:a) value.erase("handle");
                    for (auto& value:b) value.erase("handle");
                    const bool compatible=other->meta.at("model")=="det" && other->meta.at("image_type")==task->meta.at("image_type") && a==b;
                    std::lock_guard<std::mutex> lock(mutex_);
                    auto other_id=stage::number(other->meta.at("call_id")); auto it=calls_.find(other_id);
                    if (compatible && it!=calls_.end() && !it->second->final_event && Clock::now()<it->second->deadline) {
                        ids.push_back(other_id); until=std::min(until,it->second->deadline);
                    } else if (it!=calls_.end() && !it->second->final_event && pre_det_dependency(task->meta,other->meta,it->second->det_seen)) {
                        // Advance the other page's pre-det dependency, not idle-wait on a call it cannot yet publish.
                        pending=std::move(task); task=std::move(other); id=other_id; job=it->second;
                        ids=Json::array({id}); until=job->deadline; ++det_predecessors_;
                    } else pending=std::move(other);
                }
            }
            { std::lock_guard<std::mutex> lock(mutex_);
              if (job->final_event || Clock::now()>=until) continue;
              for (const auto& value:ids) running_.insert(stage::number(value));
              ++model_runs_; if (ids.size()==2) ++det_pairs_; }
            ok(gpu_.rpc("model.run",{{"call_ids",ids}},{},until));
        }
    }
    void dispatch_multi() {
        std::unique_ptr<Event> next;
        while (!stopping_) {
            if (!next && ready_q_.stats().items) ready_q_.pop(next,Clock::now()+std::chrono::milliseconds(20));
            bool run=false;
            { std::lock_guard<std::mutex> lock(mutex_); run=next && running_.empty(); }
            if (run) {
                auto id=stage::number(next->meta.at("call_id")); std::shared_ptr<Job> job;
                { std::lock_guard<std::mutex> lock(mutex_);
                  auto it=calls_.find(id); if (it==calls_.end() || it->second->final_event) { next.reset(); continue; }
                  job=it->second; }
                auto ids=Json::array({id}); auto until=job->deadline; std::unique_ptr<Event> later;
                if (det_batch_ && next->meta.at("model")=="det") {
                    std::unique_ptr<Event> other;
                    if (ready_q_.pop(other,std::min(until,Clock::now()+std::chrono::milliseconds(100)))) {
                        auto a=next->meta.at("inputs"),b=other->meta.at("inputs");
                        for (auto& v:a) v.erase("handle");
                        for (auto& v:b) v.erase("handle");
                        auto oid=stage::number(other->meta.at("call_id"));
                        std::lock_guard<std::mutex> lock(mutex_); auto it=calls_.find(oid);
                        if (it!=calls_.end() && !it->second->final_event && Clock::now()<it->second->deadline) {
                            if (other->meta.at("model")=="det" && next->meta.at("image_type")==other->meta.at("image_type") && a==b) {
                                ids.push_back(oid); until=std::min(until,it->second->deadline);
                            } else if (pre_det_dependency(next->meta,other->meta,it->second->det_seen)) {
                                later=std::move(next); next=std::move(other); id=oid; job=it->second;
                                ids=Json::array({id}); until=job->deadline; ++det_predecessors_;
                            } else later=std::move(other);
                        }
                    }
                }
                if (Clock::now()<until) {
                    { std::lock_guard<std::mutex> lock(mutex_);
                      for (const auto& value:ids) running_.insert(stage::number(value));
                      legacy_running_=next->meta.at("model")!="det";
                      ++model_runs_; if (ids.size()==2) ++det_pairs_; }
                    ok(gpu_.rpc("model.run",{{"call_ids",ids}},{},until));
                }
                next=std::move(later);
            }
            bool barrier;
            { std::lock_guard<std::mutex> lock(mutex_); barrier=legacy_running_ || (next && next->meta.at("model")!="det"); }
            if (!barrier) {
                unsigned slot=0; std::uint64_t id=0; std::shared_ptr<Job> job;
                { std::lock_guard<std::mutex> lock(mutex_);
                  for (auto it=rec_ready_.begin();it!=rec_ready_.end();) {
                      auto candidate=stage::number(it->at("call_id")); auto found=calls_.find(candidate);
                      if (found==calls_.end() || found->second->final_event || Clock::now()>=found->second->deadline) {
                          rec_bytes_-=weight(*it); it=rec_ready_.erase(it); continue;
                      }
                      if (it->at("route")=="cpu") { if (!rec_running_.count(5)) slot=5; }
                      else for (unsigned i=1;i<=rec_slots_;++i) if (!rec_running_.count(i)) { slot=i; break; }
                      if (!slot) { ++it; continue; }
                      id=candidate; job=found->second; rec_bytes_-=weight(*it); rec_ready_.erase(it);
                      rec_running_[slot]=id; ++rec_submitted_; ++model_runs_;
                      rec_high_slots_=std::max<std::uint64_t>(rec_high_slots_,rec_running_.size()); break;
                  } }
                if (slot) ok(gpu_.rpc("model.run",{{"call_ids",Json::array({id})},{"slot",slot}},{},job->deadline));
            }
            std::unique_lock<std::mutex> lock(mutex_); changed_.wait_for(lock,std::chrono::milliseconds(1));
        }
    }
    void results() {
        { std::lock_guard<std::mutex> lock(mutex_); result_threads_.insert(std::this_thread::get_id()); }
        while (!stopping_) {
            std::unique_ptr<Event> task;
            if (!result_q_.pop(task,Clock::now()+std::chrono::milliseconds(200))) continue;
            if (task->meta.at("op")=="model.resume") {
                auto id=stage::number(task->meta.at("call_id")); std::shared_ptr<Job> job;
                { std::lock_guard<std::mutex> lock(mutex_); auto it=calls_.find(id); if (it==calls_.end()) continue; job=it->second; }
                // Timeout/cancellation may retire a completed call before its resume reaches I/O.
                result_.rpc("model.resume",{{"call_id",id}},{},Clock::now()+std::chrono::seconds(5));
                { std::lock_guard<std::mutex> lock(mutex_);
                  if (std::none_of(rec_running_.begin(),rec_running_.end(),[&](const std::pair<const unsigned,std::uint64_t>& x) { return x.second==id; })) calls_.erase(id); }
            } else {
                std::shared_ptr<Job> job;
                { std::lock_guard<std::mutex> lock(mutex_); job=jobs_.at(stage::string(task->meta.at("image_id"))); }
                auto cleanup=Clock::now()+std::chrono::seconds(5);
                auto credit=results_.acquire(stage::result_limit,cleanup);
                if (!credit) throw Timeout("result credit unavailable");
                Json record;
                {
                    auto transfer=transport_.acquire(2*stage::result_limit,cleanup);
                    if (!transfer) throw Timeout("result transfer credit");
                    auto reply=result_.rpc("image.collect",{{"image_id",job->id}},{},cleanup);
                    record=Json{{"image_id",job->id},{"status",reply.meta.at("status")}};
                    if (reply.meta.at("status")=="ok") record["data"]=stage::object(
                        std::string(reply.frame.payload.begin(),reply.frame.payload.end()),stage::result_limit,64);
                    else record["code"]=reply.meta.at("code");
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_); stage::need(!job->done);
                    job->result=std::move(record); job->result_credit=std::move(credit); job->done=true;
                    job->wall_ms=std::chrono::duration<double,std::milli>(Clock::now()-job->created).count();
                    job->active.reset(); changed_.notify_all();
                }
            }
        }
    }
    void events() {
        while (!stopping_) {
            pollfd p{events_.fd(),POLLIN,0}; int n=::poll(&p,1,200);
            if (n<0 && errno==EINTR) continue;
            if (n<0 || p.revents&(POLLERR|POLLHUP|POLLNVAL)) throw PeerClosed("events closed");
            if (!n) continue;
            auto message=stage::receive(events_.fd(),Clock::now()+std::chrono::seconds(5)); const auto& m=message.meta;
            stage::need(m.at("session")==input_.session && m.at("schema")==input_.schema && message.frame.id>event_id_); event_id_=message.frame.id;
            auto op=stage::string(m.at("op"));
            if (op=="model.failed") { stop("GPU_FAILED"); return; }
            if (op=="model.ready") {
                std::shared_ptr<Job> job;
                { std::lock_guard<std::mutex> lock(mutex_);
                  job=jobs_.at(stage::string(m.at("image_id")));
                  stage::need(job->admitted && !job->final_event && m.at("image_type")==job->kind);
                  stage::need(calls_.emplace(stage::number(m.at("call_id")),job).second);
                  if (m.at("model")=="det") job->det_seen=true; }
                auto task=std::unique_ptr<Event>(new Event{m});
                if (rec_slots_ && m.at("model")=="rec") {
                    std::lock_guard<std::mutex> lock(mutex_);
                    stage::need(rec_ready_.size()<8 && rec_bytes_+weight(m)<=16*mib);
                    rec_ready_.push_back(m); rec_bytes_+=weight(m);
                    rec_high_items_=std::max(rec_high_items_,rec_ready_.size()); rec_high_bytes_=std::max(rec_high_bytes_,rec_bytes_);
                } else if (Clock::now()<job->deadline && !ready_q_.push(std::move(task),weight(m),job->deadline)) throw Timeout("GPU queue full");
                changed_.notify_all();
            } else if (op=="model.done") {
                for (const auto& value:m.at("call_ids")) {
                    auto id=stage::number(value);
                    { std::lock_guard<std::mutex> lock(mutex_);
                      auto rec=std::find_if(rec_running_.begin(),rec_running_.end(),[&](const std::pair<const unsigned,std::uint64_t>& x) { return x.second==id; });
                      stage::need(rec!=rec_running_.end() || running_.erase(id)==1);
                      if (running_.empty()) legacy_running_=false;
                      stage::need(calls_.count(id)==1); changed_.notify_all(); }
                    push_result(Json{{"op","model.resume"},{"call_id",id}},4096,Clock::now()+std::chrono::seconds(5));
                }
            } else if (op=="rec.consumed") {
                auto slot=unsigned(stage::number(m.at("slot"),1,5)); auto id=stage::number(m.at("call_id"));
                std::lock_guard<std::mutex> lock(mutex_);
                stage::need(rec_slots_ && rec_running_.count(slot) && rec_running_.at(slot)==id);
                rec_running_.erase(slot); calls_.erase(id); changed_.notify_all();
            } else if (op=="image.done") {
                std::shared_ptr<Job> job;
                { std::lock_guard<std::mutex> lock(mutex_); job=jobs_.at(stage::string(m.at("image_id")));
                  stage::need(job->admitted && !job->final_event); job->final_event=true;
                  for (auto it=rec_ready_.begin();it!=rec_ready_.end();) {
                      if (it->at("image_id")==job->id) { rec_bytes_-=weight(*it); it=rec_ready_.erase(it); }
                      else ++it;
                  }
                  // image.done follows CPU retirement and safe GPU completion, including cancelled calls.
                  for (auto it=calls_.begin();it!=calls_.end();) {
                      if (it->second==job) {
                          running_.erase(it->first);
                          for (auto r=rec_running_.begin();r!=rec_running_.end();) {
                              if (r->second==it->first) r=rec_running_.erase(r); else ++r;
                          }
                          it=calls_.erase(it);
                      }
                      else ++it;
                  }
                  changed_.notify_all(); }
                push_result(Json{{"op","image.collect"},{"image_id",job->id}},stage::result_limit,Clock::now()+std::chrono::seconds(5));
            } else throw ProtocolError("unexpected backend event");
        }
    }
public:
    explicit Client(const std::string& path,std::atomic<bool>* interrupted=nullptr,bool det_batch=false,bool hardware_jpeg=false,unsigned rec_slots=0)
      : input_(path,"input","",rec_slots),gpu_(path,"gpu",input_.session,rec_slots),result_(path,"result",input_.session,rec_slots),events_(path,"events",input_.session,rec_slots),active_(rec_slots ? 256*mib:2*page_bytes),rec_slots_(rec_slots),det_batch_(det_batch),hardware_jpeg_(hardware_jpeg),interrupted_(interrupted) {
        stage::need(rec_slots==0 || rec_slots==1 || rec_slots==2 || rec_slots==4);
#ifndef AUTOOCR_JPEG
        stage::need(!hardware_jpeg_);
#endif
        try {
            for (unsigned i=0;i<2;++i) threads_.emplace_back([this] { guarded([this] { inputs(); }); });
            threads_.emplace_back([this] { guarded([this] { dispatch(); }); });
            for (unsigned i=0;i<2;++i) threads_.emplace_back([this] { guarded([this] { results(); }); });
            threads_.emplace_back([this] { guarded([this] { events(); }); });
        } catch (...) { stop("STATE"); join(); throw; }
    }
    ~Client() { stop(); join(); }
    void stop(const std::string& reason="STOPPED") {
        if (stopping_.exchange(true)) return;
        { std::lock_guard<std::mutex> lock(mutex_); failure_=reason; rec_ready_.clear(); rec_bytes_=0; }
        input_.shutdown(); gpu_.shutdown(); result_.shutdown(); events_.shutdown();
        input_q_.close(false); ready_q_.close(false); result_q_.close(false);
        active_.close(); ingress_.close(); results_.close(); transport_.close(); changed_.notify_all();
    }
    void join() { for (auto& thread:threads_) if (thread.joinable()) thread.join(); }
    bool stopped() const { return stopping_; }
    std::string failure() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::set<std::string>{"TIMEOUT","DISCONNECTED","GPU_FAILED","LIMIT","CANCELLED"}.count(failure_) ? failure_:"STATE";
    }
    ByteBudget::Lease serialization_credit() {
        auto credit=results_.acquire(32*mib,Clock::now()+std::chrono::seconds(5));
        if (!credit) throw Timeout("serialization credit");
        return credit;
    }
    void sent() { delivered_results_.clear(); }
    Json stats() const {
        std::lock_guard<std::mutex> lock(mutex_); auto a=active_.stats(); auto i=ingress_.stats(); auto r=results_.stats();
        auto iq=input_q_.stats(); auto gq=ready_q_.stats(); auto rq=result_q_.stats(); auto tx=transport_.stats();
        return Json{{"rec_slots",rec_slots_},{"rec_submitted",rec_submitted_},{"rec_high_slots",rec_high_slots_},
          {"rec_queue_high",rec_high_items_},{"rec_queue_bytes_high",rec_high_bytes_},
          {"active_reserved",a.used},{"active_high",a.high},{"ingress_reserved",i.used},{"result_reserved",r.used},
          {"result_high",r.high},{"transport_reserved",tx.used},{"transport_high",tx.high},{"transport_limit",tx.limit},
          {"input_queue_high",iq.high_items},{"ready_queue_high",gq.high_items},{"result_queue_high",rq.high_items},
          {"input_workers_observed",input_threads_.size()},{"result_workers_observed",result_threads_.size()},
          {"gpu_dispatchers",1},{"event_receivers",1},{"model_runs",model_runs_},{"det_batch_enabled",det_batch_},{"det_pairs",det_pairs_},
          {"det_predecessors",det_predecessors_},
          {"hardware_jpeg_enabled",hardware_jpeg_},{"jpeg_used",jpeg_used_},{"jpeg_software",jpeg_software_},{"jpeg_failed",jpeg_failed_},
          {"jpeg_decode_ms",jpeg_decode_ms_},{"jpeg_color_ms",jpeg_color_ms_},{"jpeg_total_ms",jpeg_total_ms_},
          {"batches",batches_},{"stopped",bool(stopping_)}};
    }
    Json run(const Json& metadata,std::vector<unsigned char> payload,int front_fd=-1) {
        validate(metadata.dump(),payload.size()); stage::need(metadata.at("op")=="batch.submit");
        auto until=Clock::now()+std::chrono::milliseconds(stage::number(metadata.at("remaining_ms"),1,600000));
        auto bytes=ingress_.acquire(payload.size(),until); if (!bytes) throw Timeout("ingress closed/full");
        auto body=std::make_shared<std::vector<unsigned char>>(std::move(payload));
        std::vector<std::shared_ptr<Job>> ordered;
        { std::lock_guard<std::mutex> lock(mutex_); stage::need(jobs_.empty() && calls_.empty() && running_.empty() && rec_running_.empty() && rec_ready_.empty() && !stopping_ && delivered_results_.empty()); }
        std::size_t offset=0;
        for (const auto& row:metadata.at("images")) {
            auto job=std::make_shared<Job>(); job->id=stage::string(row.at("image_id")); job->kind=int(stage::number(row.at("image_type"),1,3));
            job->deadline=std::min(until,Clock::now()+std::chrono::seconds(300));
            ordered.push_back(job);
            { std::lock_guard<std::mutex> lock(mutex_); jobs_.emplace(job->id,job); }
            auto n=stage::number(row.at("encoded_bytes"),1,16*mib);
            auto task=std::unique_ptr<Input>(new Input{job,body,offset,std::size_t(n)}); offset+=n;
            if (!input_q_.push(std::move(task),n,until)) throw Timeout("input queue closed/full");
        }
        body.reset();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!stopping_ && !std::all_of(ordered.begin(),ordered.end(),[](const std::shared_ptr<Job>& j) { return j->done && j->submitted; })) {
                if (Clock::now()>=until || (interrupted_ && *interrupted_)) { lock.unlock(); stop("TIMEOUT"); lock.lock(); break; }
                if (front_fd>=0) {
                    pollfd p{front_fd,POLLIN,0}; int n=::poll(&p,1,0);
                    if (n>0) { lock.unlock(); stop("DISCONNECTED"); lock.lock(); break; }
                }
                changed_.wait_for(lock,std::chrono::milliseconds(100));
            }
            if (stopping_) throw PeerClosed(failure_);
            stage::need(calls_.empty() && running_.empty() && rec_running_.empty() && rec_ready_.empty()); jobs_.clear(); ++batches_;
        }
        Json result=Json::array(); for (auto& job:ordered) result.push_back(std::move(job->result));
        for (auto& job:ordered) delivered_results_.push_back(std::move(job->result_credit));
        auto execution=stats(); execution["pages"]=Json::array();
        for (const auto& job:ordered) execution["pages"].push_back(Json{{"image_id",job->id},
            {"admission_wait_ms",job->admission_wait_ms},{"wall_ms",job->wall_ms}});
        bytes.reset(); return Json{{"results",std::move(result)},{"execution",execution}};
    }
};
}} // namespace autoocr::batch
