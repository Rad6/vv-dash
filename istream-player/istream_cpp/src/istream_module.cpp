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

struct IStreamRenderer
{
    IStreamGUI *gui;

    IStreamRenderer()
    {
        gui = new IStreamGUI();
        // printf("Hello from module init function (constructor).\n");
    }

    void setup(py::object config)
    {
    }

    AwaitThread *run()
    {
        gui->start_player();
        // printf("Started player\n");
        return new AwaitThread(&gui->gui_thread);
    }

    void close()
    {
        gui->close();
    }

    void end()
    {
        gui->end();
    }

    void decode(std::string content, int length)
    {
        // printf("Inside decode\n");
        gui->decode((unsigned char *)content.c_str(), content.length());
    }

    void play_frames(int count)
    {
        gui->schedule_play_frames(count);
    }

    std::thread _t;

    AwaitThread await_func(py::object obj)
    {
        _t = ThreadAwait(obj);
        return AwaitThread(&_t);
    }
};





PYBIND11_MODULE(istream_renderer, m)
{
    m.doc() = "pybind11 IStream C++ plugin"; // optional module docstring


    py::class_<AwaitThread>(m, "AwaitableRun")
        .def("__iter__", &AwaitThread::__iter__)
        .def("__await__", &AwaitThread::__await__)
        .def("__next__", &AwaitThread::__next__);

    py::class_<IStreamRenderer>(m, "IStreamRenderer")
        .def(py::init())

        .def("setup", &IStreamRenderer::setup)
        .def("run", &IStreamRenderer::run)
        .def("close", &IStreamRenderer::close)
        .def("decode", &IStreamRenderer::decode)
        .def("play_frames", &IStreamRenderer::play_frames)
        .def("end", &IStreamRenderer::end)
        .def("await_func", &IStreamRenderer::await_func);
}
