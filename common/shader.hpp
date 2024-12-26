#include <Orochi/Orochi.h>
#include <hiprt/hiprt.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#define SH_ASSERT(ExpectTrue) \
    if ((ExpectTrue) == 0) { __debugbreak(); }
#elif defined(__GNUC__)
#define SH_ASSERT(ExpectTrue) \
    if ((ExpectTrue) == 0) { raise(SIGTRAP); }
#endif

#define NV_ARG_LINE_INFO "--generate-line-info"
#define AMD_ARG_LINE_INFO "-g"

inline void loadAsVector(std::vector<char>* buffer, const char* fllePath)
{
    FILE* fp = fopen(fllePath, "rb");
    if (fp == nullptr) { return; }

    fseek(fp, 0, SEEK_END);

    buffer->resize(ftell(fp));

    fseek(fp, 0, SEEK_SET);

    size_t s = fread(buffer->data(), 1, buffer->size(), fp);
    if (s != buffer->size())
    {
        buffer->clear();
        return;
    }
    fclose(fp);
    fp = nullptr;
}

struct ShaderArgument
{
    template <class T>
    ShaderArgument& value(const T& v)
    {
        m_args.emplace_back(std::unique_ptr<Arg>(new ArgByValue<T>(v)));
        return *this;
    }
    template <class T>
    ShaderArgument& ptr(T* p)
    {
        m_args.emplace_back(std::unique_ptr<Arg>(new ArgByPointer<T>(p)));
        return *this;
    }
    std::vector<void*> kernelParams() const
    {
        std::vector<void*> ps;
        for (int i = 0; i < m_args.size(); i++)
        {
            ps.push_back(m_args[i]->ptr());
        }
        return ps;
    }

   private:
    struct Arg
    {
        virtual ~Arg() {}
        virtual void* ptr() = 0;
    };
    template <class T>
    struct ArgByValue : public Arg
    {
        ArgByValue(const T& value) : m_value(value) {}
        virtual void* ptr() override { return static_cast<void*>(&m_value); }
        T m_value;
    };
    template <class T>
    struct ArgByPointer : public Arg
    {
        ArgByPointer(T* p) : m_ptr(p) {}
        virtual void* ptr() override { return static_cast<void*>(m_ptr); }
        T* m_ptr;
    };
    std::vector<std::unique_ptr<Arg>> m_args;
};

// You must declare all functions to use as parameters for Shader
struct UseHiprt
{
    UseHiprt(hiprtContext context) : m_context(context) {}
    UseHiprt& func(const char* functionName)
    {
        m_functions.push_back(functionName);
        return *this;
    }
    operator bool() { return m_context != nullptr; }

    hiprtContext m_context = nullptr;
    std::vector<const char*> m_functions;
};

class Shader
{
   public:
    Shader(const char* filename, const char* kernelLabel,
           const std::vector<std::string>& options,
           UseHiprt useHiprt = UseHiprt(nullptr))
    {
        // load shader source code
        std::vector<char> src;
        loadAsVector(&src, filename);
        SH_ASSERT(0 < src.size());
        src.push_back('\0');

        // create shader program
        orortcProgram program = 0;
        orortcCreateProgram(&program, src.data(), kernelLabel, 0, 0, 0);

        // add shader compile options
        std::vector<const char*> optionChars;
        for (int i = 0; i < options.size(); ++i)
        {
            optionChars.push_back(options[i].c_str());
        }

        if (useHiprt)
        {
            std::vector<oroFunction> compiledFuncs(useHiprt.m_functions.size());
            hiprtError compileResult = hiprtBuildTraceKernels(
                useHiprt.m_context, useHiprt.m_functions.size(),
                useHiprt.m_functions.data(), src.data(), kernelLabel, 0,
                nullptr, nullptr, optionChars.size(), optionChars.data(),
                0 /*numGeomTypes*/, 1 /*numRayTypes*/, nullptr,
                (hiprtApiFunction*)compiledFuncs.data(),
                (hiprtApiModule*)&m_module, false /*use cache*/);
            SH_ASSERT(compileResult == hiprtSuccess);

            for (int i = 0; i < useHiprt.m_functions.size(); i++)
            {
                m_functions[useHiprt.m_functions[i]] = compiledFuncs[i];
            }
        }
        else
        {
            // compile shader
            orortcResult compileResult = orortcCompileProgram(
                program, optionChars.size(), optionChars.data());

            // print compilation log
            size_t logSize = 0;
            orortcGetProgramLogSize(program, &logSize);
            if (1 < logSize)
            {
                std::vector<char> compileLog(logSize);
                orortcGetProgramLog(program, compileLog.data());
                printf("%s", compileLog.data());
            }
            SH_ASSERT(compileResult == ORORTC_SUCCESS);

            // get compiled code
            size_t codeSize = 0;
            orortcGetCodeSize(program, &codeSize);

            std::vector<char> codec(codeSize);
            orortcGetCode(program, codec.data());

            orortcDestroyProgram(&program);

            orortcResult re;
            oroError e = oroModuleLoadData(&m_module, codec.data());
            SH_ASSERT(e == oroSuccess);
        }
    }

    ~Shader() { oroModuleUnload(m_module); }

    void launch(const char* name, const ShaderArgument& arguments,
                unsigned int gridDimX, unsigned int gridDimY,
                unsigned int gridDimZ, unsigned int blockDimX,
                unsigned int blockDimY, unsigned int blockDimZ,
                oroStream hStream)
    {
        if (m_functions.count(name) == 0)
        {
            oroFunction f = 0;
            oroError e = oroModuleGetFunction(&f, m_module, name);
            SH_ASSERT(e == oroSuccess);
            m_functions[name] = f;
        }

        auto params = arguments.kernelParams();
        oroFunction f = m_functions[name];
        oroError e = oroModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                           blockDimX, blockDimY, blockDimZ, 0,
                                           hStream, params.data(), 0);
        SH_ASSERT(e == oroSuccess);
    }

   private:
    oroModule m_module = 0;
    std::map<std::string, oroFunction> m_functions;
};