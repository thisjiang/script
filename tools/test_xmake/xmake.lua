add_defines("LEARN")

set_languages("c++17")
add_rules("mode.release", "mode.debug")
if is_mode("release") then
    set_optimize("fastest")
end

add_includedirs("src")

includes("third")
includes("src")
includes("test")
