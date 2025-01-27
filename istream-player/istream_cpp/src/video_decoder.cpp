
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/pixdesc.h>
#include <libavutil/imgutils.h>
#include <inttypes.h>
}
#include <boost/circular_buffer.hpp>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <atomic>
#include <memory>

#define DBG_INT(var_name) printf("DBG: " #var_name " = %d\n", var_name)
static const int kBufferSize = 40 * 1024;

// struct SwsContext
// {
//     int srcW;
//     int srcH;
// };

template <typename T>
using cbuff_t = boost::circular_buffer<T>;

template <typename T>
class RingBuffer : public cbuff_t<T>
{
private:
    std::mutex m_mutex;
    std::condition_variable m_cond_notfull;
    std::condition_variable m_cond_notempty;

    // uint64_t write_sum;
    // uint64_t read_sum;

public:
    bool closed = false;
    
    RingBuffer(int size)
        : cbuff_t<T>(size) {}

    /// @brief Writes size  number of items from items.
    /// @param items Pointer to the array of items
    /// @param size Number of items to write
    /// @return The number of bytes written
    inline int write(T *buffer, int buf_size)
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (this->closed)
            return 0;
        while (this->full())
        {
            printf("Queue full\n");
            m_cond_notfull.wait(lock);
        }
        if (this->closed)
            return 0;

        int num_items = std::min((int)(this->capacity() - this->size()), buf_size);

        for (int i = 0; i < num_items; i++)
        {
            this->push_back(buffer[i]);
            // write_sum += buffer[i];
        }
        // printf("Write sum : %llu\n", write_sum);

        this->m_cond_notempty.notify_one();

        return num_items;
    }

    /// @brief Readed size items into buffer
    /// @param buffer Pointer to buffer
    /// @param size Size of buffer
    /// @return The number of bytes read
    inline int read(T *buffer, int buf_size)
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        while (this->empty())
        {
            if (this->closed)
                return 0;
            m_cond_notempty.wait(lock);
        }

        int num_items = std::min((int)this->size(), buf_size);

        for (int i = 0; i < num_items; i++)
        {
            buffer[i] = this->front();
            this->pop_front();
            // read_sum += buffer[i];
        }
        // printf("Read sum : %llu\n", read_sum);

        this->m_cond_notfull.notify_one();
        return num_items;
    }

    inline T pop_one()
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        while (this->empty())
        {
            if (this->closed)
                return 0;
            m_cond_notempty.wait(lock);
        }

        T ret = this->front();
        this->pop_front();

        this->m_cond_notfull.notify_one();
        return ret;
    }

    inline int push_one(T value)
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (this->closed)
            return 0;
        while (this->full())
        {
            m_cond_notfull.wait(lock);
        }
        if (this->closed)
            return 0;

        this->push_back(value);
        this->m_cond_notempty.notify_one();

        return 1;
    }

    inline void close()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        closed = true;
        this->m_cond_notempty.notify_all();
        this->m_cond_notfull.notify_all();
    }

    // inline bool empty()
    // {
    //     std::unique_lock<std::mutex> lock(m_mutex);
    //     return cbuff_t<T>::empty();
    // }

    // inline bool full()
    // {
    //     std::unique_lock<std::mutex> lock(m_mutex);
    //     return cbuff_t<T>::full();
    // }
};

// av_err2str returns a temporary array. This doesn't work in gcc.
// This function can be used as a replacement for av_err2str.
static const char *av_make_error(int errnum)
{
    static char str[AV_ERROR_MAX_STRING_SIZE];
    memset(str, 0, sizeof(str));
    return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}

static AVPixelFormat correct_for_deprecated_pixel_format(AVPixelFormat pix_fmt)
{
    // Fix swscaler deprecated pixel format warning
    // (YUVJ has been deprecated, change pixel format to regular YUV)
    switch (pix_fmt)
    {
    case AV_PIX_FMT_YUVJ420P:
        return AV_PIX_FMT_YUV420P;
    case AV_PIX_FMT_YUVJ422P:
        return AV_PIX_FMT_YUV422P;
    case AV_PIX_FMT_YUVJ444P:
        return AV_PIX_FMT_YUV444P;
    case AV_PIX_FMT_YUVJ440P:
        return AV_PIX_FMT_YUV440P;
    default:
        return pix_fmt;
    }
}


class TileDecoder
{
private:

};


class VideoDecoder
{
private:
    int video_stream_index;
    AVFormatContext *av_format_ctx;
    AVCodecContext *av_codec_ctx;
    AVPacket *av_packet;
    SwsContext *sws_scaler_ctx;
    AVIOContext *avio_ctx;
    AVFrame *av_frame;

    int last_in_width, last_in_height;
    AVPixelFormat last_in_pix_fmt;

    bool eof = false;

public:
    RingBuffer<AVFrame *> free_frames;
    RingBuffer<AVFrame *> decoded_buffer;
    RingBuffer<unsigned char> encoded_buffer;

    std::atomic<int> skip_frames;

    // Public things for other parts of the program to read from
    int out_width, out_height;
    AVRational time_base;

    VideoDecoder(int w, int h)
        : out_width(w), out_height(h),
          free_frames(150),
          decoded_buffer(100),
          encoded_buffer(40000 * 1024),
          sws_scaler_ctx(NULL),
          last_in_width(-1), last_in_height(-1),
          skip_frames(0)
    {
        int i = 0;
        // printf("Allocating frames : \n");
        while (!free_frames.full())
        {
            // printf("Allocating frame %d \n", ++i);
            AVFrame *dest_frame = av_frame_alloc();
            if (!dest_frame)
                throw "Couldn't allocate AVFrame2";

            int num_bytes = avpicture_get_size(AV_PIX_FMT_RGB24, out_width, out_height);
            uint8_t *frame2_buffer = (uint8_t *)av_malloc(num_bytes * sizeof(uint8_t));
            avpicture_fill((AVPicture *)dest_frame, frame2_buffer, AV_PIX_FMT_RGB24, out_width, out_height);

            free_frames.push_one(dest_frame);
        }

        printf("Allocated %d frames\n", free_frames.size());
    }

    ~VideoDecoder()
    {
        sws_freeContext(sws_scaler_ctx);
        avformat_close_input(&av_format_ctx);
        avformat_free_context(av_format_ctx);
        av_frame_free(&av_frame);
        av_packet_free(&av_packet);
        avcodec_free_context(&av_codec_ctx);
        av_free(&avio_ctx);

        while (!free_frames.empty())
        {
            AVFrame *frame = free_frames.pop_one();
            av_frame_free(&frame);
        }

        while (!decoded_buffer.empty())
        {
            AVFrame *frame = decoded_buffer.pop_one();
            av_frame_free(&frame);
        }
    }

    void decoder_loop()
    {
        if (!video_reader_open())
        {
            printf("Couldn't open video file (make sure you set a video file that exists)\n");
            return;
        }
        printf("Video stream found\n");
        while (true)
        {
            AVFrame *dest_frame = free_frames.pop_one();
            int64_t pts;
            if (!video_reader_read_frame(dest_frame))
            {

                printf("Failed to decode frame\n");
                break;
            } else if (eof) {
                printf("EOF reached \n");
                decoded_buffer.close();
                break;
            } else {
                decoded_buffer.push_one(dest_frame);
            }
        }
        printf("Decoder loop eneded \n");
    }

    static int read_packet(void *opaque, unsigned char *buf, int buf_size)
    {
        // printf("Trying to read\n");
        VideoDecoder *state = (VideoDecoder *)opaque;
        int ret = state->encoded_buffer.read(buf, buf_size);
        // printf("Read %d bytes\n", ret);
        if (ret == 0)
        {
            state->eof = true;
        }

        return ret;
    }

    bool video_reader_open()
    {
        auto avio_buffer = (unsigned char *)av_malloc(kBufferSize);

        avio_ctx = avio_alloc_context(avio_buffer, kBufferSize, 0, this,
                                      &VideoDecoder::read_packet, NULL, NULL);

        if (!avio_ctx)
        {
            printf("Couldn't created AVIOContext\n");
            return false;
        }

        // Open the file using libavformat
        av_format_ctx = avformat_alloc_context();
        if (!av_format_ctx)
        {
            printf("Couldn't created AVFormatContext\n");
            return false;
        }
        av_format_ctx->pb = avio_ctx;

        if (avformat_open_input(&av_format_ctx, "", NULL, NULL) != 0)
        {
            printf("Couldn't open video file\n");
            return false;
        }
        printf("-----------------------------\n");

        // Find the first valid video stream inside the file
        video_stream_index = -1;
        AVCodecParameters *av_codec_params;
        AVCodec *av_codec;
        for (int i = 0; i < av_format_ctx->nb_streams; ++i)
        {
            av_codec_params = av_format_ctx->streams[i]->codecpar;
            av_codec = avcodec_find_decoder(av_codec_params->codec_id);
            if (!av_codec)
            {
                continue;
            }
            if (av_codec_params->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                video_stream_index = i;
                time_base = av_format_ctx->streams[i]->time_base;
                break;
            }
        }
        if (video_stream_index == -1)
        {
            printf("Couldn't find valid video stream inside file\n");
            return false;
        }

        // Set up a codec context for the decoder
        av_codec_ctx = avcodec_alloc_context3(av_codec);
        if (!av_codec_ctx)
        {
            printf("Couldn't create AVCodecContext\n");
            return false;
        }
        if (avcodec_parameters_to_context(av_codec_ctx, av_codec_params) < 0)
        {
            printf("Couldn't initialize AVCodecContext\n");
            return false;
        }
        if (avcodec_open2(av_codec_ctx, av_codec, NULL) < 0)
        {
            printf("Couldn't open codec\n");
            return false;
        }

        av_frame = av_frame_alloc();
        if (!av_frame)
        {
            printf("Couldn't allocate AVFrame\n");
            return false;
        }

        av_packet = av_packet_alloc();
        if (!av_packet)
        {
            printf("Couldn't allocate AVPacket\n");
            return false;
        }

        return true;
    }

    bool video_reader_read_frame(AVFrame *dest_frame)
    {
        // Decode one frame
        int response;

        while (!eof)
        {
            response = av_read_frame(av_format_ctx, av_packet);
            if (response < 0)
            {
                if ((response == AVERROR_EOF || avio_feof(av_format_ctx->pb)) && !eof)
                {
                    eof = 1;
                }
                if (av_format_ctx->pb && av_format_ctx->pb->error)
                {
                    return false;
                }
            }
            if (skip_frames.load() > 0) {
                skip_frames--;
                av_packet_unref(av_packet);
                continue;
            }
            if (av_packet->stream_index != video_stream_index)
            {
                av_packet_unref(av_packet);
                continue;
            }

            response = avcodec_send_packet(av_codec_ctx, av_packet);
            if (response < 0)
            {
                printf("Failed to decode packet: %s\n", av_make_error(response));
                return false;
            }

            response = avcodec_receive_frame(av_codec_ctx, av_frame);
            if (response == AVERROR(EAGAIN) || response == AVERROR_EOF)
            {
                av_packet_unref(av_packet);
                continue;
            }
            else if (response < 0)
            {
                printf("Failed to decode packet: %s\n", av_make_error(response));
                return false;
            }

            av_packet_unref(av_packet);
            break;
        }

        if (eof) {
            // printf("Coded stream eof reacjhed \n");
            return true;
        }

        dest_frame->pts = av_frame->pts;
        dest_frame->width = out_width;
        dest_frame->height = out_height;

        // Tiling properties
        int dest_offset_x = 0;
        int dest_offset_y = 0;
        int dest_width = out_width;
        int dest_height = out_height;

        AVPixelFormat source_pix_fmt = correct_for_deprecated_pixel_format(av_codec_ctx->pix_fmt);
        // Check if sws_scalar_ctx is compatible
        if (!sws_scaler_ctx || last_in_width != av_frame->width || last_in_height != av_frame->height || last_in_pix_fmt != source_pix_fmt)
        {
            if (sws_scaler_ctx)
            {
                sws_freeContext(sws_scaler_ctx);
                printf("Input frames format changed to (%dx%d, %s)\n", av_frame->width, av_frame->height, av_get_pix_fmt_name(source_pix_fmt));
            }
            sws_scaler_ctx = sws_getContext(av_frame->width, av_frame->height, source_pix_fmt,
                                            dest_width, dest_height, AV_PIX_FMT_RGB24,
                                            SWS_BILINEAR, NULL, NULL, NULL);
            if (!sws_scaler_ctx)
            {
                printf("Couldn't initialize sw scaler\n");
                return false;
            }

            last_in_width = av_frame->width;
            last_in_height = av_frame->height;
            last_in_pix_fmt = source_pix_fmt;
        }

        int dest_linesize[] = {dest_frame->width * 3, 0, 0, 0, 0, 0, 0, 0};
        uint8_t *dest_data[] = {
            dest_frame->data[0] + dest_offset_y * dest_linesize[0] + (dest_offset_x * 3),
            NULL, NULL, NULL, NULL, NULL, NULL, NULL};

        sws_scale(sws_scaler_ctx, av_frame->data, av_frame->linesize, 0, av_frame->height, dest_data, dest_linesize);

        return true;
    }

    bool video_reader_seek_frame(VideoDecoder *state, int64_t ts)
    {

        // Unpack members of state
        auto &av_format_ctx = state->av_format_ctx;
        auto &av_codec_ctx = state->av_codec_ctx;
        auto &video_stream_index = state->video_stream_index;
        auto &av_packet = state->av_packet;
        auto &av_frame = state->av_frame;

        av_seek_frame(av_format_ctx, video_stream_index, ts, AVSEEK_FLAG_BACKWARD);

        // av_seek_frame takes effect after one frame, so I'm decoding one here
        // so that the next call to video_reader_read_frame() will give the correct
        // frame
        int response;
        while (av_read_frame(av_format_ctx, av_packet) >= 0)
        {
            if (av_packet->stream_index != video_stream_index)
            {
                av_packet_unref(av_packet);
                continue;
            }

            response = avcodec_send_packet(av_codec_ctx, av_packet);
            if (response < 0)
            {
                printf("Failed to decode packet: %s\n", av_make_error(response));
                return false;
            }

            response = avcodec_receive_frame(av_codec_ctx, av_frame);
            if (response == AVERROR(EAGAIN) || response == AVERROR_EOF)
            {
                av_packet_unref(av_packet);
                continue;
            }
            else if (response < 0)
            {
                printf("Failed to decode packet: %s\n", av_make_error(response));
                return false;
            }

            av_packet_unref(av_packet);
            break;
        }

        return true;
    }
};
