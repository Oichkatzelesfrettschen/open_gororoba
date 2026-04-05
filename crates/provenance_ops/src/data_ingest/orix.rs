use anyhow::{Context, Result};
use std::{
    fs::{File, create_dir_all},
    io::{Cursor, Read, copy},
    path::Path,
};
use zip::ZipArchive;

const ORIX_DATA_URL: &str = "https://github.com/pyxem/orix-data/archive/refs/heads/master.zip";

/// Download the orix-data example dataset from GitHub.
/// This includes .ang, .ctf, and .h5ebsd files for misorientation verification.
pub fn download_orix_dataset(target_dir: &Path) -> Result<()> {
    if !target_dir.exists() {
        create_dir_all(target_dir).context("Failed to create orix-data directory")?;
    }

    println!(
        "<EMOJI+1F4E5> Downloading orix-data from: {}",
        ORIX_DATA_URL
    );
    let agent = ureq::Agent::new_with_defaults();
    let response = agent.get(ORIX_DATA_URL).call()?;

    if response.status() != 200 {
        return Err(anyhow::anyhow!(
            "Failed to download orix-data: HTTP {}",
            response.status()
        ));
    }

    let mut buffer = Vec::new();
    response
        .into_body()
        .into_reader()
        .read_to_end(&mut buffer)?;

    let reader = Cursor::new(buffer);
    let mut archive = ZipArchive::new(reader).context("Failed to open orix-data ZIP")?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.enclosed_name().context("Invalid file name in ZIP")?;

        // Skip the top-level directory (e.g., "orix-data-master/")
        let mut components = name.components();
        components.next(); // Skip "orix-data-master"
        let relative_path = components.as_path();

        if relative_path.as_os_str().is_empty() {
            continue;
        }

        let outpath = target_dir.join(relative_path);

        if (*file.name()).ends_with('/') {
            create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent()
                && !p.exists()
            {
                create_dir_all(p)?;
            }
            let mut outfile = File::create(&outpath)?;
            copy(&mut file, &mut outfile)?;
        }
    }

    println!("<EMOJI+2705> orix-data extracted to: {:?}", target_dir);
    Ok(())
}
