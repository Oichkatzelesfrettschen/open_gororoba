use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only build shader on explicit request or in specific CI profiles 
    // to avoid nightly-only build dependencies in standard dev loops.
    if std::env::var("BUILD_VULKAN_SHADERS").is_ok() {
        SpirvBuilder::new("shader", "spirv-unknown-spv1.5")
            .print_metadata(MetadataPrintout::Full)
            .build()?;
    }
    Ok(())
}
