
// For python callback in threads

static int pygillock()
{
    return PyGILState_Ensure();
}

static void pygilrelease(int gstate)
{
    PyGILState_STATE gst = (PyGILState_STATE)gstate;
    PyGILState_Release(gst);
}

static void* pysavethread()
{
    return PyEval_SaveThread();
}

static void pyrestorethread(void* state)
{
    PyThreadState* tstate = (PyThreadState*)(state);
    PyEval_RestoreThread(tstate);
}