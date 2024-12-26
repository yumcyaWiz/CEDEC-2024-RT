newoption {
    trigger = "nvidia",
    description = "use nvidia gpu"
}

function setDebugAndOptimizationFlags()
    -- Debug symbols
    filter { "configurations:Debug or RelWithDebInfo or DebugGpu" }
        symbols "Full"
        runtime "Debug"
    
    -- Debug macro
    filter { "configurations:Debug or DebugGpu or RelWithDebInfo" }
        defines { "_DEBUG", "DEBUG" }
    filter { "configurations:DebugGpu" }
        defines { "DEBUG_GPU" }

    -- Optimization level
    filter { "configurations:Debug or DebugGpu" }
        optimize "Debug"
    filter { "configurations:RelWithDebInfo" }
        optimize "On"

    filter { "toolset:msc-v*" }
        buildoptions { "/favor:AMD64" }

    filter { "configurations:RelWithDebInfo", "toolset:clang" }
        --buildoptions { "-flto" } -- todo. This option causes linker error on Ubuntu
    filter{}
end

workspace "CEDEC_2024_RT"
    configurations {"Debug", "RelWithDebInfo", "DebugGpu"}
    language "C++"
    platforms "x64"
    architecture "amd64"

    cppdialect "C++20"

    if os.istarget("windows") then
        systemversion "latest"
        buildoptions { "/diagnostics:caret", "/nologo" }
        links { "version" }
    end

    if os.istarget("linux") then
        runpathdirs { "bin/" }
    end

    -- TODO: set compiler warnings

    filter { "platforms:x64", "configurations:Debug or configurations:DebugGpu"}
        targetsuffix "64D"
    filter { "platforms:x64", "configurations:RelWithDebInfo"}
        targetsuffix "64"
    filter {}

    flags { "MultiProcessorCompile" }

    -- set debug symbols and optimization settings
    setDebugAndOptimizationFlags()

    -- Orochi
    if _OPTIONS["nvidia"] then
        defines {"OROCHI_ENABLE_CUEW"}
        includedirs {"$(CUDA_PATH)/include"}
    end
    includedirs { "libs/orochi" }
    files { "libs/orochi/Orochi/Orochi.h" }
    files { "libs/orochi/Orochi/Orochi.cpp" }
    includedirs { "libs/orochi/contrib/hipew/include" }
    files { "libs/orochi/contrib/hipew/src/hipew.cpp" }
    includedirs { "libs/orochi/contrib/cuew/include" }
    files { "libs/orochi/contrib/cuew/src/cuew.cpp" }

    -- postbuildcommands { 
    --     -- "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin",
    --     -- "{COPYFILE} ../libs/hiprt/hiprt/win/hiprt0200264.dll ../bin",
    --     "{COPYFILE} ../libs/hiprt/hiprt/win/*.dll ../bin",
    --     "{COPYFILE} ../libs/hiprt/hiprt/win/*.fatbin ../bin",
    --     "{COPYFILE} ../libs/hiprt/hiprt/win/*.hipfb ../bin",
    --     "{COPYFILE} ../libs/hiprt/hiprt/win/*.zip ../bin",
    --     "tar -xf ../bin/hiprt02002_5.7_amd.zip -C ../bin",
    -- }

    -- glfw
    externalincludedirs { "libs/glfw/include"}
    if os.istarget("windows") then
        libdirs { "libs/glfw/lib-vc2022" }
        links { "glfw3", "opengl32" }
    end
    if os.istarget("linux") then
        links { "glfw", "GL", "X11", "Xrandr", "Xi", "Xxf86vm", "Xinerama", "Xcursor", "dl", "pthread" }
    end

    -- Stbs
    includedirs { "libs/stb" }
    files { "libs/stb/*" }

    -- objloader
    includedirs { "libs/tiny_obj_loader" }
    files { "libs/tiny_obj_loader/*" }

    -- font
    files { "libs/freeglut/bitmapString.cpp" }
    includedirs { "libs/freeglut" }

    -- hiprt
    includedirs { "libs/hiprt"}
    if os.istarget("windows") then
        libdirs { "libs/hiprt/hiprt/win" }
        links { "hiprt0200464" }
    end
    if os.istarget("linux") then
        libdirs { "libs/hiprt/hiprt/linux64" }
        links { "hiprt0200464" }
    end

    -- Other common settings
    includedirs { "./"}
    files { "./common/*.hpp", "./common/*.natvis" }

    -- Build configurations
    location "build"
    startproject "01_helloworld"

    project "copy_dependences"
        kind "ConsoleApp"
        targetdir "../bin"
        files{ "examples/dummy.cpp" }

        if os.istarget("windows") then
            postbuildcommands { 
                -- "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin",
                -- "{COPYFILE} ../libs/hiprt/hiprt/win/hiprt0200264.dll ../bin",
                "{COPYFILE} ../libs/hiprt/hiprt/win/*.dll ../bin",
                "{COPYFILE} ../libs/hiprt/hiprt/win/*.fatbin ../bin",
                "{COPYFILE} ../libs/hiprt/hiprt/win/*.hipfb ../bin",
                "{COPYFILE} ../libs/hiprt/hiprt/win/*.zip ../bin",
                "tar -xf ../bin/amd_comgr0600.zip -C ../bin",
            }
        end
        if os.istarget("linux") then
            postbuildcommands { 
                "mkdir -p ../bin",
                "{COPYFILE} ../libs/hiprt/hiprt/linux64/*.so ../bin",
                "{COPYFILE} ../libs/hiprt/hiprt/linux64/*.hipfb ../bin",
            }
        end

    include "./examples/"
