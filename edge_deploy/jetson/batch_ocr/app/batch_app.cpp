#include "batch_client.hpp"
#include <cstdlib>
#include <iostream>
#include <sys/stat.h>

using namespace autoocr;
std::atomic<bool> interrupted{false};
void signal_stop(int) { interrupted.store(true,std::memory_order_relaxed); }

class Listener {
    Fd fd_{::socket(AF_UNIX,SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC,0)};
    std::string path_;
    struct stat owned_{};
    bool bound_=false;
public:
    explicit Listener(const std::string& path):path_(path) {
        auto slash=path.find_last_of('/'); stage::need(slash!=std::string::npos && slash>0 && path.size()<108);
        auto parent=path.substr(0,slash); struct stat info{};
        char* resolved=::realpath(parent.c_str(),nullptr); stage::need(resolved!=nullptr);
        std::string real(resolved); std::free(resolved);
        stage::need(real==parent && !::lstat(parent.c_str(),&info) && S_ISDIR(info.st_mode) && info.st_uid==::geteuid() && !(info.st_mode&0077));
        stage::need(::lstat(path.c_str(),&info)!=0 && errno==ENOENT);
        sockaddr_un address{}; address.sun_family=AF_UNIX; std::memcpy(address.sun_path,path.c_str(),path.size()+1);
        if (::bind(fd_.get(),reinterpret_cast<sockaddr*>(&address),sizeof(address))) throw std::runtime_error("front bind");
        if (::lstat(path.c_str(),&owned_)) throw std::runtime_error("front inode");
        bound_=true;
        if (::chmod(path.c_str(),0600) || ::listen(fd_.get(),1)) { cleanup(); throw std::runtime_error("front permissions/listen"); }
    }
    ~Listener() { cleanup(); }
    void cleanup() {
        struct stat now{};
        if (bound_ && !::lstat(path_.c_str(),&now) && S_ISSOCK(now.st_mode) && now.st_dev==owned_.st_dev && now.st_ino==owned_.st_ino) ::unlink(path_.c_str());
        bound_=false;
    }
    int accept(const batch::Client& client) {
        while (!interrupted && !client.stopped()) {
            pollfd p{fd_.get(),POLLIN,0}; int n=::poll(&p,1,200);
            if (n<0 && errno==EINTR) continue;
            if (n<0) throw std::runtime_error("front poll");
            if (n>0) {
                int fd=::accept4(fd_.get(),nullptr,nullptr,SOCK_NONBLOCK | SOCK_CLOEXEC);
                if (fd>=0) return fd;
                if (errno!=EAGAIN && errno!=EINTR) throw std::runtime_error("front accept");
            }
        }
        return -1;
    }
};

void reply(int fd,const std::string& status,std::size_t count,const std::string& body,Deadline until) {
    stage::need(body.size()<=32*batch::mib);
    auto meta=batch::Json{{"schema","ocr.batch/1"},{"op","batch.reply"},{"status",status},{"count",count}}.dump();
    auto h=make_header(1,meta.size(),body.size());
    transfer(fd,h.data(),h.size(),true,until);
    transfer(fd,reinterpret_cast<unsigned char*>(&meta[0]),meta.size(),true,until);
    transfer(fd,reinterpret_cast<unsigned char*>(const_cast<char*>(body.data())),body.size(),true,until);
}

int main(int argc,char** argv) {
    try {
        if (argc==2 && std::string(argv[1])=="--self-test") {
            batch::Json det{{"model","det"},{"image_type",3}};
            for (const auto& role:{"layout","cell","yolo","enhance"}) {
                batch::Json other{{"model",role},{"image_type",3}};
                stage::need(batch::pre_det_dependency(det,other,false));
                stage::need(!batch::pre_det_dependency(det,other,true));
                other["image_type"]=1; stage::need(!batch::pre_det_dependency(det,other,false));
            }
            for (const auto& role:{"det","rec"}) stage::need(!batch::pre_det_dependency(det,
                batch::Json{{"model",role},{"image_type",3}},false));
            std::cout<<"batch_policy_self_test_passed\n"; return 0;
        }
        if (argc<3 || argc>6) throw ProtocolError("backend socket and front socket required");
        bool batch=false,jpeg=false; unsigned rec=0;
        for (int i=3;i<argc;++i) {
            if (std::string(argv[i])=="--det-batch" && !batch) batch=true;
            else if (std::string(argv[i])=="--jpeg" && !jpeg) jpeg=true;
            else if (!rec && (std::string(argv[i])=="--rec-slots=1" || std::string(argv[i])=="--rec-slots=2" || std::string(argv[i])=="--rec-slots=4")) rec=unsigned(argv[i][12]-'0');
            else throw ProtocolError("unsupported option");
        }
        if (!interrupted.is_lock_free()) throw ProtocolError("Signal flag must be lock-free");
        std::signal(SIGTERM,signal_stop); std::signal(SIGINT,signal_stop);
        batch::Client client(argv[1],&interrupted,batch,jpeg,rec); Listener front(argv[2]);
        std::cerr << "{\"status\":\"cpp_batch_app_ready\"}\n";
        while (!interrupted && !client.stopped()) {
            int accepted=front.accept(client); if (accepted<0) break;
            Fd peer(accepted);
            bool submitted=false;
            try {
                check_peer(peer.get()); batch::Json metadata;
                auto frame=receive_frame(peer.get(),Clock::now()+std::chrono::seconds(5),[&](const std::string& raw,std::uint32_t n) { metadata=batch::validate(raw,n); });
                stage::need(frame.id==1);
                submitted=metadata.at("op")=="batch.submit";
                auto data=!submitted ? batch::Json{{"execution",client.stats()}}:client.run(metadata,std::move(frame.payload),peer.get());
                auto credit=client.serialization_credit(); auto body=data.dump();
                reply(peer.get(),"ok",data.count("results") ? data.at("results").size():0,body,Clock::now()+std::chrono::seconds(5));
                client.sent();
            } catch (...) {
                if (submitted) client.stop("STATE");
                try { reply(peer.get(),"error",0,batch::Json{{"code",client.failure()}}.dump(),Clock::now()+std::chrono::seconds(1)); } catch (...) {}
                if (client.stopped()) break;
            }
        }
        bool failed=client.stopped() && !interrupted;
        client.stop(); client.join();
        std::cerr << "{\"status\":\"cpp_batch_app_stopped\"}\n";
        return failed ? 1:0;
    } catch (...) { std::cerr << "{\"status\":\"cpp_batch_app_failed\"}\n"; return 1; }
}
