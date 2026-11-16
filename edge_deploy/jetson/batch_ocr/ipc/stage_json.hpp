// Strict OCR metadata consumer; transport and lifecycle ownership remain separate.
#pragma once
#include "ipc_wire.hpp"
#include "../third_party/nlohmann_json/json.hpp"
#include <map>
#include <set>

namespace autoocr { namespace stage {
using Json = nlohmann::json;
constexpr std::uint64_t max_id = 0x7fffffffffffffffULL;
constexpr std::uint64_t result_limit = 8*1024*1024;
inline void need(bool value) { if (!value) throw ProtocolError("invalid OCR stage metadata"); }

inline Json object(const std::string& raw, std::size_t limit, int max_depth) {
    need(!raw.empty() && raw.size()<=limit && raw.compare(0,3,"\xef\xbb\xbf")!=0 && raw.find('\0')==std::string::npos);
    std::map<int,std::set<std::string>> keys;
    try {
        auto callback=[&](int depth, Json::parse_event_t event, Json& value) {
            need(depth<=max_depth);
            if (event==Json::parse_event_t::object_start) keys[depth+1].clear();
            if (event==Json::parse_event_t::key) need(keys[depth].insert(value.get<std::string>()).second);
            if (event==Json::parse_event_t::object_end) keys.erase(depth+1);
            return true;
        };
        auto result=Json::parse(raw,callback);
        need(result.is_object()); return result;
    } catch (const Json::exception&) { throw ProtocolError("invalid JSON object"); }
}
inline std::string string(const Json& value) { need(value.is_string()); return value.get<std::string>(); }
inline bool one(const Json& value, std::initializer_list<const char*> choices) {
    auto text=string(value);
    return std::any_of(choices.begin(),choices.end(),[&](const char* choice) { return text==choice; });
}
inline std::uint64_t number(const Json& value, std::uint64_t low=1, std::uint64_t high=max_id) {
    need(value.is_number_integer());
    if (!value.is_number_unsigned()) need(value.get<std::int64_t>()>=0);
    auto n=value.get<std::uint64_t>(); need(low<=n && n<=high); return n;
}
inline bool token(const Json& value, bool session=false) {
    auto s=string(value);
    if (session ? s.size()!=32 : (s.empty() || s.size()>64)) return false;
    for (unsigned char c:s) {
        bool hex=(c>='0' && c<='9') || (c>='a' && c<='f');
        bool ascii=(c>='0' && c<='9') || (c>='a' && c<='z') || (c>='A' && c<='Z') || c=='_' || c=='-' || c=='.';
        if (!(session ? hex : ascii)) return false;
    }
    return true;
}
inline void fields(const Json& value, std::initializer_list<const char*> extra) {
    std::set<std::string> expected{"schema","op","session","remaining_ms"};
    expected.insert(extra.begin(),extra.end()); need(value.size()==expected.size());
    for (const auto& name:expected) need(value.count(name)==1);
}
inline void ids(const Json& value) {
    need(value.is_array() && value.size()>=1 && value.size()<=2);
    std::set<std::uint64_t> seen;
    for (const auto& item:value) need(seen.insert(number(item)).second);
}
inline bool error_code(const Json& value) {
    return one(value,{"BUSY","LIMIT","STATE","SESSION","TIMEOUT","CANCELLED","GPU_FAILED","DISCONNECTED"});
}
inline void ready(const Json& m) {
    need(token(m.at("image_id"))); number(m.at("image_type"),1,3); number(m.at("call_id"));
    auto model=string(m.at("model"));
    need(one(m.at("model"),{"layout","cell","yolo","enhance","det","rec"}));
    const auto& inputs=m.at("inputs"); need(inputs.is_array() && !inputs.empty() && inputs.size()<=3);
    std::set<std::string> expected=(model=="layout" || model=="cell")
        ? std::set<std::string>{"image","im_shape","scale_factor"}
        : std::set<std::string>{(model=="det" || model=="rec") ? "x" : "images"};
    std::set<std::string> names; std::set<std::uint64_t> handles; std::uint64_t total=0;
    for (const auto& input:inputs) {
        need(input.is_object() && input.size()==5);
        for (const char* key:{"name","handle","dtype","shape","bytes"}) need(input.count(key)==1);
        auto name=string(input.at("name")); need(expected.count(name) && names.insert(name).second);
        need(handles.insert(number(input.at("handle"))).second && input.at("dtype")=="float32");
        const auto& shape=input.at("shape"); need(shape.is_array() && !shape.empty() && shape.size()<=4);
        std::vector<std::uint64_t> dims; std::uint64_t bytes=4;
        for (const auto& item:shape) {
            auto n=number(item,1,8192); need(bytes<=max_payload/n); bytes*=n; dims.push_back(n);
        }
        need(number(input.at("bytes"),1,max_payload)==bytes); total+=bytes; need(total<=max_payload);
        if (name=="im_shape" || name=="scale_factor") need(dims==std::vector<std::uint64_t>({1,2}));
        else {
            need(dims.size()==4 && dims[0]==1 && dims[1]==3);
            if (model=="layout") need(dims[2]==800 && dims[3]==800);
            else if (model=="cell") need(dims[2]==640 && dims[3]==640);
            else if (model=="enhance") need(dims[2]==512 && dims[3]==512);
            else if (model=="rec") need(dims[2]==48 && dims[3]>=32 && dims[3]<=3200);
            else for (unsigned i:{2,3}) need(dims[i]>=(model=="det" ? 32u:64u) && dims[i]<=(model=="det" ? 960u:640u) && dims[i]%32==0);
        }
    }
    need(names==expected);
}
inline Json validate(const std::string& raw, std::uint64_t payload) {
    need(payload<=max_payload);
    try {
        auto m=object(raw,max_meta,16);
        if (m.at("schema")=="ocr.stage/2") {
            auto base=m; base["schema"]="ocr.stage/1";
            if (m.at("op")=="rec.consumed") {
                fields(m,{"call_id","slot"}); number(m.at("slot"),1,5); base.erase("slot"); base["op"]="model.resume";
            } else if (m.at("op")=="session.ready") {
                fields(m,{"rec_slots","rec_queue_items","rec_queue_bytes"});
                auto slots=number(m.at("rec_slots"),1,4); need(slots==1 || slots==2 || slots==4);
                number(m.at("rec_queue_items"),8,8); number(m.at("rec_queue_bytes"),16*1024*1024,16*1024*1024);
                for (auto k:{"rec_slots","rec_queue_items","rec_queue_bytes"}) base.erase(k);
            } else if (m.at("op")=="model.ready" && m.at("model")=="rec") {
                fields(m,{"image_id","image_type","call_id","model","inputs","group_id","crop_id","box_index","route"});
                ready(m); number(m.at("group_id")); number(m.at("crop_id"),0); number(m.at("box_index"),0);
                auto w=number(m.at("inputs").at(0).at("shape").at(3),32,3200);
                need(m.at("route")== (w<=1536 ? "narrow":w<=2304 ? "wide":"cpu"));
                for (auto k:{"group_id","crop_id","box_index","route"}) base.erase(k);
            } else if (m.at("op")=="model.run" && m.count("slot")) {
                number(m.at("slot"),1,5); need(m.at("call_ids").size()==1); base.erase("slot");
            }
            validate(base.dump(),payload); return m;
        }
        need(m.at("schema")=="ocr.stage/1" && token(m.at("session"),true));
        number(m.at("remaining_ms"),1,300000); auto op=string(m.at("op"));
        if (op=="session.ready") { fields(m,{}); need(!payload); }
        else if (op=="channel.bind" || op=="channel.bound") {
            fields(m,{"channel"}); need(!payload && one(m.at("channel"),{"input","gpu","result","events"}));
        }
        else if (op=="rpc.reply") {
            fields(m,{"for_op","status","code"}); need(!payload);
            need(one(m.at("for_op"),{"image.submit","model.run","model.resume","image.collect","image.cancel"}));
            need(one(m.at("status"),{"ok","error"}));
            need(m.at("status")=="ok" ? m.at("code")=="OK" : error_code(m.at("code")));
        } else if (op=="image.submit") {
            need(token(m.at("image_id"))); number(m.at("image_type"),1,3);
            if (m.at("format")=="encoded") {
                fields(m,{"image_id","image_type","format"}); need(payload>=1 && payload<=16*1024*1024);
            } else {
                fields(m,{"image_id","image_type","format","shape","stride"}); need(m.at("format")=="bgr8");
                const auto& s=m.at("shape"); need(s.is_array() && s.size()==3);
                auto h=number(s.at(0),1,8192), w=number(s.at(1),1,8192); number(s.at(2),3,3);
                need(h*w<=16000000 && number(m.at("stride"),1,max_payload)==w*3 && payload==h*w*3);
            }
        } else if (op=="model.ready") {
            fields(m,{"image_id","image_type","call_id","model","inputs"}); need(!payload); ready(m);
        } else if (op=="model.run") { fields(m,{"call_ids"}); need(!payload); ids(m.at("call_ids")); }
        else if (op=="model.resume") { fields(m,{"call_id"}); need(!payload); number(m.at("call_id")); }
        else if (op=="image.collect" || op=="image.cancel") {
            fields(m,{"image_id"}); need(!payload && token(m.at("image_id")));
        } else if (op=="model.done") {
            fields(m,{"call_ids","output_bytes"}); need(!payload); ids(m.at("call_ids"));
            const auto& sizes=m.at("output_bytes"); need(sizes.is_array() && sizes.size()==m.at("call_ids").size());
            for (const auto& size:sizes) number(size,0,max_payload);
        } else if (op=="model.failed") {
            fields(m,{"call_ids","code"}); need(!payload && m.at("code")=="GPU_FAILED"); ids(m.at("call_ids"));
        } else if (op=="image.result" || op=="image.done") {
            need(one(m.at("status"),{"ok","error","cancelled"}));
            if (m.at("status")=="ok") {
                fields(m,{"image_id","status"}); need(op=="image.done" ? !payload : payload>=1 && payload<=result_limit);
            } else {
                fields(m,{"image_id","status","code"}); need(!payload && error_code(m.at("code")));
                need((m.at("status")=="cancelled")== (m.at("code")=="CANCELLED"));
            }
            need(token(m.at("image_id")));
        } else throw ProtocolError("unknown OCR stage operation");
        return m;
    } catch (const Json::exception&) { throw ProtocolError("invalid OCR stage fields"); }
}

struct Message { Frame frame; Json meta; };
inline Message receive(int fd, Deadline deadline) {
    Message m;
    m.frame=receive_frame(fd,deadline,[&](const std::string& raw,std::uint32_t n) { m.meta=validate(raw,n); });
    if (m.meta.at("op")=="image.result" && m.meta.at("status")=="ok")
        object(std::string(m.frame.payload.begin(),m.frame.payload.end()),result_limit,64);
    return m;
}
inline void send(int fd, const Frame& frame, Deadline deadline) {
    auto m=validate(frame.metadata,frame.payload.size());
    if (m.at("op")=="image.result" && m.at("status")=="ok")
        object(std::string(frame.payload.begin(),frame.payload.end()),result_limit,64);
    send_frame(fd,frame,deadline);
}
}} // namespace autoocr::stage
