#include <fstream>
#include <iterator>

#include "video_renderer.cpp"

int main(void)
{
    std::string base = "/home/akram/ucalgary/research/istream-player/dataset";
    std::vector<std::string> files = {
        "/videos/av1-1sec/Aspen/init-stream4.m4s",
        "/videos/av1-1sec/Aspen/chunk-stream4-00001.m4s",
        "/videos/av1-1sec/Aspen/chunk-stream4-00002.m4s",
        "/videos/av1-1sec/Aspen/chunk-stream4-00003.m4s",
        "/videos/av1-1sec/Aspen/chunk-stream4-00004.m4s",
        "/videos/av1-1sec/Aspen/chunk-stream4-00005.m4s",
        "/videos/av1-1sec/Aspen/chunk-stream4-00006.m4s",
        "/videos/av1-1sec/Aspen/chunk-stream4-00007.m4s",
        "/videos/av1-1sec/Aspen/chunk-stream4-00008.m4s",
        "/videos/av1-1sec/Aspen/chunk-stream4-00009.m4s",
        "/videos/av1-1sec/Aspen/chunk-stream4-00010.m4s",
        // "/videos-raw/aspen_open.mp4"
    };

    IStreamGUI gui;

    gui.schedule_play_frames(299);
    for (auto path: files) {
        printf("'%s'\n", (base + path).c_str());
        std::ifstream input(base + path, std::ios::binary);
        std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
        gui.decode(&buffer[0], buffer.size());
    }
    gui.start_player();
    gui.close();

    gui.gui_thread.join();
    return 0;
}

