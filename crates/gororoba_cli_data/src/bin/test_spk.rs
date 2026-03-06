use data_core::spice::daf::DafReader;
use data_core::spice::spk::SpkReader;

fn main() -> anyhow::Result<()> {
    println!("Loading DE440s.bsp...");
    let daf = DafReader::open("data/external/de440s.bsp")?;
    let _spk = SpkReader::new(daf)?;

    println!("Loading galileo_cruise.bsp...");
    let daf_gll = DafReader::open("data/external/galileo_cruise.bsp")?;
    let spk_gll = SpkReader::new(daf_gll)?;
    
    println!("Galileo Cruise Segments:");
    for (i, seg) in spk_gll.segments().iter().enumerate() {
        println!("  {}: Target {}, Center {}, Type {}, JED {:.2} to {:.2}", 
                 i, seg.target, seg.center, seg.spk_type, seg.start_jed, seg.end_jed);
    }

    Ok(())
}
