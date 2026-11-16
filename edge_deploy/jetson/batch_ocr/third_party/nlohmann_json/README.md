# nlohmann/json（项目内固定依赖）

- 上游：[JSON for Modern C++ v3.12.0](https://github.com/nlohmann/json/releases/tag/v3.12.0)。
- json.hpp来自官方tag的single_include/nlohmann/json.hpp，未修改。
- SHA256：`aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63`，与官方发布页相符。
- MIT许可证原样保留在LICENSE.MIT，SHA256：`46a65cffd1ea955132d95a8dd921640714a8d6b537d2e4e482d31145ae95b603`。
- E157板端没有可用JSON开发头，严格解析不是适合自写的几行业务代码，因此引入单头；不装系统包、不在CMake配置时联网拉取。
- 构建须使用此固定副本；更新版本／SHA时重新跑跨语言schema与IPC测试，不自动跟踪latest。
