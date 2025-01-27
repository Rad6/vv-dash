#include <stdio.h>
#include <stdlib.h>
#include <GLFW/glfw3.h>
#include "video_decoder.cpp"
#include <thread>
#include <chrono>
#include <unistd.h>

void gui_loop(VideoDecoder *vr_state, GLFWwindow *window);

class IStreamGUI
{
private:
public:
    GLFWwindow *window;
    VideoDecoder video_decoder;
    std::thread gui_thread;
    std::thread decoder_thread;
    // bool tex_initialized;

    int rem_play_frames;
    std::mutex rem_play_frames_mutex;
    std::condition_variable rem_play_frames_cond;
    bool closed = false;

    int frame_drops = 0;

    IStreamGUI() : video_decoder(1920, 1080)
    {
        rem_play_frames = 0;
        // tex_initialized = false;
    }
    ~IStreamGUI() {}

    void decode(unsigned char *buffer, int buffer_size)
    {
        int written = 0;
        // printf("Writing %d bytes \n", buffer_size);
        while (written < buffer_size)
        {
            written += video_decoder.encoded_buffer.write(buffer + written, buffer_size - written);
            // printf("Written %d bytes \n", written);
        }
        return;
    }

    int start_player()
    {
        gui_thread = std::thread(&IStreamGUI::gui_loop, this);
        decoder_thread = std::thread(&VideoDecoder::decoder_loop, std::ref(video_decoder));
        return 0;
    }

    void schedule_play_frames(int count)
    {
        std::unique_lock<std::mutex> lock(rem_play_frames_mutex);
        rem_play_frames += count;
        rem_play_frames_cond.notify_one();
    }

    int wait_for_play_frames()
    {
        std::unique_lock<std::mutex> rem_play_frames_lock(rem_play_frames_mutex);
        while (rem_play_frames == 0)
        {
            if (closed)
                return 0;
            printf("Remaining frames is zero\n", rem_play_frames);
            rem_play_frames_cond.wait(rem_play_frames_lock);
        }
        rem_play_frames--;
        return 1;
        // printf("Remaining %d frames\n", rem_play_frames);
    }

    void gui_loop()
    {
        // VideoDecoder vr_state;

        /* Initialize the library */
        if (!glfwInit())
            return;

        /* Create a windowed mode window and its OpenGL context */
        window = glfwCreateWindow(1920, 1080, "IStream Player", NULL, NULL);
        if (!window)
        {
            printf("Couldn't open window\n");
            glfwTerminate();
            return;
        }

        /* Make the window's context current */
        glfwSetWindowAttrib(window, GLFW_RESIZABLE, true);
        glfwMakeContextCurrent(window);

        // Generate textures
        GLuint tex_handle;
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &tex_handle);
        glBindTexture(GL_TEXTURE_2D, tex_handle);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

        // Set up orthogonal projection
        int win_width, win_height;
        glfwGetFramebufferSize(window, &win_width, &win_height);

        // Initialize 2D texture with blank data
        int num_bytes = video_decoder.out_width * video_decoder.out_height * 3;
        uint8_t *dummy_data = (uint8_t *)malloc(num_bytes);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, video_decoder.out_width, video_decoder.out_height, 0, GL_RGB, GL_UNSIGNED_BYTE, dummy_data);
        free(dummy_data);

        const double fr = atof(getenv("FR") == NULL ? "30.0" : getenv("FR"));
        const double pts_duration = 1/fr;
        double pt_in_seconds = 0;

        int play_frame = wait_for_play_frames();
        int decoded_frames = 0;

        while (!glfwWindowShouldClose(window))
        {
            // auto start = std::chrono::steady_clock::now();
            // glfwGetFramebufferSize(window, &win_width, &win_height);
            // glViewport(0, 0, win_width, win_height);


            // int play_frame = wait_for_play_frames();
            // if (!play_frame)
            // {
            //     printf("Ending playback. Remaining decoded frames: %d\n", video_decoder.decoded_buffer.size());
            //     glfwSetWindowShouldClose(window, GLFW_TRUE);
            //     continue;
            // }


            static bool first_frame = true;
            if (first_frame)
            {
                glfwSetTime(0.0);
                first_frame = false;
            }

            // double pt_in_seconds = (frame->pts * (double)video_decoder.time_base.num / (double)video_decoder.time_base.den)/speedup;
            pt_in_seconds += pts_duration;
            double curr_time = glfwGetTime();


            if ((pt_in_seconds + pts_duration + 0.3) < curr_time)
            {
                // auto end = std::chrono::steady_clock::now();
                // printf("elapsed = %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
                printf("CAPTURE_DROPPED=%d\n", ++frame_drops);
                AVFrame *frame = video_decoder.decoded_buffer.pop_one();
                video_decoder.free_frames.push_one(frame);

                // glfwSwapBuffers(window);
                // glfwPollEvents();
                continue;
            } else {
                decoded_frames += 1;
            }

            while (pt_in_seconds > glfwGetTime())
            {
                glfwWaitEventsTimeout(pt_in_seconds - glfwGetTime());
            }




            if (video_decoder.decoded_buffer.empty())
            {
                printf("No more decoded frames \n", frame_drops);
                video_decoder.skip_frames++;
            }

            AVFrame *frame = video_decoder.decoded_buffer.pop_one();

            if (!frame)
            {
                printf("No more decoded frames \n");
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                continue;
            }

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame->width, frame->height, GL_RGB, GL_UNSIGNED_BYTE, frame->data[0]);

            // Recycle frame
            video_decoder.free_frames.push_one(frame);

            glBegin(GL_QUADS);
            glTexCoord2f(0, 1);
            glVertex3f(-1, -1, 0);

            glTexCoord2f(1, 1);
            glVertex3f(1, -1, 0);

            glTexCoord2f(1, 0);
            glVertex3f(1, 1, 0);

            glTexCoord2f(0, 0);
            glVertex3f(-1, 1, 0);
            glEnd();

            glfwSwapBuffers(window);
            // glfwPollEvents();
        }

        printf("GUI loop ended\n");
        printf("CAPTURE_RENDERED=%d\n", decoded_frames);
        printf("CAPTURE_DROPPED=%d\n", frame_drops);
        fflush(stdout);

        glfwTerminate();
        // exit(0);
    }

    void close()
    {
        printf("Closing encoded buffer \n");
        this->video_decoder.encoded_buffer.close();
        // glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    void end()
    {
        std::unique_lock<std::mutex> rem_play_frames_lock(rem_play_frames_mutex);
        closed = true;
        this->rem_play_frames_cond.notify_all();
    }
};
