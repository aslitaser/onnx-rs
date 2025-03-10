fn main() {
    println!("cargo:rerun-if-changed=proto/onnx.proto");
    prost_build::compile_protos(&["proto/onnx.proto"], &["proto"]).unwrap();
}