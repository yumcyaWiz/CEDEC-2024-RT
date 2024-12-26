project "01_helloworld"
    kind "ConsoleApp"
    targetdir "../bin"

    files { project().name .. "/*.cpp", project().name .. "/*.cu" }
    dependson {"copy_dependences"}

project "02_triangle"
    kind "ConsoleApp"
    targetdir "../bin"
    
    files { project().name .. "/*.cpp", project().name .. "/*.cu" }
    dependson {"copy_dependences"}

project "03_cornelbox"
    kind "ConsoleApp"
    targetdir "../bin"
    
    files { project().name .. "/*.cpp", project().name .. "/*.cu" }
    dependson {"copy_dependences"}

project "04_ao"
    kind "ConsoleApp"
    targetdir "../bin"
    
    files { project().name .. "/*.cpp", project().name .. "/*.cu" }
    dependson {"copy_dependences"}

project "05_ao_boundingbox"
    kind "ConsoleApp"
    targetdir "../bin"
    
    files { project().name .. "/*.cpp", project().name .. "/*.cu" }
    dependson {"copy_dependences"}

project "06_ao_hiprt"
    kind "ConsoleApp"
    targetdir "../bin"
    
    files { project().name .. "/*.cpp", project().name .. "/*.cu" }
    dependson {"copy_dependences"}

project "07_pt"
    kind "ConsoleApp"
    targetdir "../bin"
    
    files { project().name .. "/*.cpp", project().name .. "/*.cu" }
    dependson {"copy_dependences"}

project "08_nee"
    kind "ConsoleApp"
    targetdir "../bin"
    
    files { project().name .. "/*.cpp", project().name .. "/*.cu" }
    dependson {"copy_dependences"}

project "09_ris"
    kind "ConsoleApp"
    targetdir "../bin"
    
    files { project().name .. "/*.cpp", project().name .. "/*.cu" }
    dependson {"copy_dependences"}

project "10_restir_di"
    kind "ConsoleApp"
    targetdir "../bin"
    
    files { project().name .. "/*.cpp", project().name .. "/*.cu" }
    dependson {"copy_dependences"}