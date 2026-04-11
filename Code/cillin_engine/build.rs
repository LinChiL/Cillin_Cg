fn main() {
    let mut build = cxx_build::bridge("src/lib.rs");
    build.file("cpp/sdf_logic.cpp")
        .include("cpp")
        .std("c++20"); // 这样写更标准，cc-rs 会自动处理不同平台的 flag

    build.compile("cillin_math");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cpp/sdf_logic.cpp");
    println!("cargo:rerun-if-changed=cpp/sdf_logic.hpp");
}