for _, filepath in ipairs(os.files("*.cc")) do
    local filename = path.basename(filepath)
    target(filename)
        set_kind("binary")
        add_deps("hello")
        add_packages("gtest")
        add_files(filepath)
    target_end()
end
