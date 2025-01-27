#include "../extern/pybind11/include/pybind11/pybind11.h"
#include "../extern/pybind11/include/pybind11/stl.h"
#include "./video_renderer.cpp"
#include <chrono>
#include <thread>

namespace py = pybind11;

class AwaitThread
{
private:
    py::object asyncio_fut_iter;
    py::object asyncio_fut;
    py::object asyncio_loop;

    bool waiting = false;
    std::thread *inner_thread;
    std::thread *waiter_thread;

public:
    // AwaitThread() {
    //     printf("Making default awaiter \n");
    // }
    
    AwaitThread(std::thread *t) : inner_thread(t)
    {
        // printf("Making awaiter \n");
    }

    ~AwaitThread() {
        // printf("AwaitThread delete called\n");
    }

    void join_inner()
    {
        inner_thread->join();
        PyGILState_STATE state = PyGILState_Ensure();
        asyncio_loop.attr("call_soon_threadsafe")(asyncio_fut.attr("set_result"), py::none());
        // printf("Releasing GIL\n");
        PyGILState_Release(state);
        // printf("GIL released\n");
    }

    AwaitThread *__iter__()
    {
        // printf("__iter__ called \n");
        return this;
    }

    AwaitThread *__await__()
    {
        // printf("__await__ called \n");
        return this;
    }

    py::object __next__()
    {
        // printf("__next__ called \n");
        if (!waiting)
        {
            asyncio_loop = py::module_::import("asyncio").attr("get_running_loop")();
            asyncio_fut = asyncio_loop.attr("create_future")();
            waiter_thread = new std::thread(&AwaitThread::join_inner, this);
            // waiter_thread.detach();
            asyncio_fut_iter = asyncio_fut.attr("__await__")();
            waiting = true;
        }
        // printf("Calling __next__ iteration\n");

        // throw py::stop_iteration();
        try
        {
            return asyncio_fut_iter.attr("__next__")();
        }
        catch (py::error_already_set &e)
        {
            // printf("Exception raised by fut.__next__\n");
            if (e.matches(PyExc_StopIteration))
            {
                // printf("Exception is StopIteration\n");
                // ex()
                // auto exc = py::stop_iteration();
                // PyErr_SetString(PyExc_StopIteration, "Inner thread finished");
                throw py::stop_iteration();
            }
            else
            {

                printf("Exception is not stop Iteration\n");
            }
        }
        return py::none();
    }
};

class ThreadAwait : public std::thread
{
private:
    py::object async_iter;
    py::object asyncio_loop;

public:
    py::object ret_val = py::none();

    ThreadAwait(py::object awaitable) : std::thread(&ThreadAwait::run, this)
    {
        async_iter = awaitable().attr("__await__")();
        asyncio_loop = py::module_::import("asyncio").attr("get_running_loop")();
    }

    void run()
    {
        // printf("Starting thread \n");
        PyGILState_STATE state = PyGILState_Ensure();
        try
        {
            while (true)
            {
                asyncio_loop.attr("call_soon_threadsafe")(async_iter.attr("__next__"));
            }
        }
        catch (py::error_already_set &e)
        {
            if (e.matches(PyExc_StopIteration))
            {
                ret_val = e.value();
            }
            else
            {
                PyGILState_Release(state);
                // throw e;
            }
        }
        PyGILState_Release(state);
    }
};

struct Image
{
    int width, height;
    std::string
};

struct IStreamDecoder
{
    VideoDecoder video_decoder;
    std::thread decoder_thread;

    IStreamDecoder(): video_decoder(1920, 1080)
    {
        printf("Hello from module init function (constructor).\n");
    }

    void setup(py::object config)
    {
        decoder_thread = std::thread(&VideoDecoder::decoder_loop, std::ref(video_decoder));
    }

    void close()
    {
    }

    void decode(std::string content)
    {
        // printf("Inside decode\n");
        // gui->decode((unsigned char *)content.c_str(), content.length());
        video_decoder.encoded_buffer.write((unsigned char *)content.c_str(), content.length());
    }

    std::string read()
    {
        AVFrame *frame = video_decoder.decoded_buffer.pop_one();
        return "";
    }

    // std::thread _t;

    // AwaitThread await_func(py::object obj)
    // {
    //     _t = ThreadAwait(obj);
    //     return AwaitThread(&_t);
    // }
};





PYBIND11_MODULE(istream_decoder, m)
{
    m.doc() = "pybind11 IStream C++ plugin"; // optional module docstring


    // py::class_<AwaitThread>(m, "AwaitableRun")
    //     .def("__iter__", &AwaitThread::__iter__)
    //     .def("__await__", &AwaitThread::__await__)
    //     .def("__next__", &AwaitThread::__next__);

    py::class_<IStreamDecoder>(m, "IStreamDecoder")
        .def(py::init())

        .def("setup", &IStreamDecoder::setup)
        .def("close", &IStreamDecoder::close)
        .def("decode", &IStreamDecoder::decode)
        .def("read", &IStreamDecoder::read);
}
