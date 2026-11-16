#include "stage_json.hpp"
#include <iostream>
#include <memory>

using namespace autoocr;
using stage::Json;

std::string input(std::size_t max) {
    std::string raw(max+1,'\0'); std::cin.read(&raw[0],raw.size()); raw.resize(std::cin.gcount());
    stage::need(raw.size()<=max); return raw;
}
void self_test() {
    auto m=stage::Json{{"schema","ocr.stage/1"},{"op","session.ready"},{"session",std::string(32,'a')},{"remaining_ms",300000}};
    stage::validate(m.dump(),0);
    bool rejected=false;
    try { stage::validate(m.dump().substr(0,m.dump().size()-1)+",\"op\":\"session.ready\"}",0); }
    catch (const ProtocolError&) { rejected=true; }
    stage::need(rejected);
    std::cout << "{\"status\":\"cpp_stage_self_test_passed\"}\n";
}
void exchange(const std::string& path, bool grouped=false, bool workers=false) {
    Fd fd(::socket(AF_UNIX,SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC,0));
    auto deadline=Clock::now()+std::chrono::seconds(workers ? 10:2);
    connect_local(fd.get(),path,deadline);
    auto hello=stage::receive(fd.get(),deadline);
    stage::need(hello.meta.at("op")=="session.ready");
    const auto session=stage::string(hello.meta.at("session"));
    std::map<std::string,std::unique_ptr<Fd>> extra;
    auto handle=[&](const std::string& role) { return !grouped || role=="input" ? fd.get() : extra.at(role)->get(); };
    if (grouped) for (const char* role:{"input","gpu","result","events"}) {
        if (std::string(role)!="input") {
            extra.emplace(role,std::unique_ptr<Fd>(new Fd(::socket(AF_UNIX,SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC,0))));
            connect_local(handle(role),path,deadline);
            auto other=stage::receive(handle(role),deadline);
            stage::need(other.meta.at("op")=="session.ready" && other.meta.at("session")==session);
        }
        Json bind{{"schema","ocr.stage/1"},{"op","channel.bind"},{"session",session},{"remaining_ms",2000},{"channel",role}};
        stage::send(handle(role),Frame{1,bind.dump(),{}},deadline);
        auto ack=stage::receive(handle(role),deadline);
        stage::need(ack.frame.id==1 && ack.meta.at("op")=="channel.bound" && ack.meta.at("channel")==role && ack.meta.at("session")==session);
    }
    std::uint64_t requests=0,event_id=1;
    std::map<std::string,std::uint64_t> ids{{"input",1},{"gpu",1},{"result",1}};
    std::set<std::string> finished;
    auto receive=[&](const char* op,const std::string& role) {
        auto m=stage::receive(handle(role),deadline);
        while (workers && role=="events" && m.meta.at("op")=="image.done") {
            auto image=stage::string(m.meta.at("image_id"));
            stage::need(m.meta.at("session")==session && m.frame.id>event_id && m.meta.at("status")=="ok");
            stage::need((image=="a" || image=="b") && finished.insert(image).second);
            event_id=m.frame.id;
            if (std::string(op)=="image.done") return m;
            m=stage::receive(handle(role),deadline);
        }
        stage::need(m.meta.at("session")==session && m.meta.at("op")==op); return m;
    };
    auto rpc=[&](const char* op,Json extra,std::vector<unsigned char> body,const char* expected) {
        Json meta{{"schema","ocr.stage/1"},{"op",op},{"session",session},{"remaining_ms",2000}};
        for (auto it=extra.begin();it!=extra.end();++it) meta[it.key()]=it.value();
        std::string role=std::string(op)=="image.submit" ? "input" : (std::string(op)=="model.run" ? "gpu":"result");
        ++requests; auto id=grouped ? ++ids.at(role) : requests;
        stage::send(handle(role),Frame{id,meta.dump(),std::move(body)},deadline);
        auto m=receive(std::string(op)=="image.collect" && std::string(expected)=="RESULT" ? "image.result":"rpc.reply",role);
        stage::need(m.frame.id==id);
        if (std::string(expected)=="RESULT") {
            stage::need(m.meta.at("image_id")==extra.at("image_id") && m.meta.at("status")=="ok");
            auto value=stage::object(std::string(m.frame.payload.begin(),m.frame.payload.end()),stage::result_limit,64);
            stage::need(value==Json({{"synthetic",true},{"image_id",extra.at("image_id")}}));
        } else {
            stage::need(m.meta.at("for_op")==op && m.meta.at("code")==expected);
            stage::need(m.meta.at("status")== (std::string(expected)=="OK" ? "ok":"error"));
        }
    };
    for (const char* image:{"a","b"})
        rpc("image.submit",{{"image_id",image},{"image_type",3},{"format","bgr8"},{"shape",{1,2,3}},{"stride",6}},{1,2,3,4,5,6},"OK");
    std::map<std::string,Json> ready;
    for (unsigned i=0;i<2;++i) {
        auto event=receive("model.ready","events"); auto image=stage::string(event.meta.at("image_id"));
        if (grouped) { stage::need(event.frame.id>event_id); event_id=event.frame.id; }
        stage::need((image=="a" || image=="b") && !ready.count(image) && event.meta.at("model")=="det" && event.meta.at("image_type")==3);
        ready.emplace(image,std::move(event.meta));
    }
    stage::need(ready.at("a").at("call_id")!=ready.at("b").at("call_id"));
    rpc("image.collect",{{"image_id","a"}},{},"STATE");
    for (const char* image:{"a","b"}) {
        auto call=ready.at(image).at("call_id");
        rpc("model.run",{{"call_ids",Json::array({call})}},{},"OK");
        auto done=receive("model.done","events");
        if (grouped) { stage::need(done.frame.id>event_id); event_id=done.frame.id; }
        stage::need(done.meta.at("call_ids")==Json::array({call}) && done.meta.at("output_bytes")==Json::array({4}));
        rpc("model.resume",{{"call_id",call}},{},"OK");
    }
    while (workers && finished.size()<2) receive("image.done","events");
    for (const char* image:{"b","a"}) rpc("image.collect",{{"image_id",image}},{},"RESULT");
    std::cout << "{\"status\":\"" << (workers ? "cpp_python_workers_passed_not_ocr" : (grouped ? "cpp_python_channels_passed_not_ocr":"cpp_python_stages_passed_not_ocr"))
              << "\",\"images\":2,\"requests\":" << requests << ",\"reverse_collection\":true,\"channels\":" << (grouped ? 4:1) << "}\n";
}
int main(int argc,char** argv) {
    try {
        if (argc==2 && std::string(argv[1])=="--self-test") { self_test(); return 0; }
        if (argc==3 && std::string(argv[1])=="--channels") { exchange(argv[2],true); return 0; }
        if (argc==3 && std::string(argv[1])=="--workers") { exchange(argv[2],true,true); return 0; }
        if (argc==3 && std::string(argv[1])=="--metadata") {
            std::size_t end=0; auto n=std::stoull(argv[2],&end); stage::need(end==std::strlen(argv[2]));
            stage::validate(input(max_meta),n);
        } else if (argc==2 && std::string(argv[1])=="--result") {
            stage::object(input(stage::result_limit),stage::result_limit,64);
        } else if (argc==2) { exchange(argv[1]); return 0; }
        else throw ProtocolError("arguments required");
        std::cout << "{\"accepted\":true}\n"; return 0;
    } catch (const Timeout&) { std::cerr << "{\"accepted\":false,\"kind\":\"timeout\"}\n"; }
      catch (const PeerClosed&) { std::cerr << "{\"accepted\":false,\"kind\":\"peer_closed\"}\n"; }
      catch (const std::exception&) { std::cerr << "{\"accepted\":false,\"kind\":\"protocol_or_system\"}\n"; }
    return 1;
}
