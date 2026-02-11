use thiserror::Error;

#[derive(Debug, Error)]
pub enum SniaError {
    #[error("density must be positive, got {0}")]
    NonPositiveDensity(f64),
    #[error("pressure must be non-negative, got {0}")]
    NegativePressure(f64),
    #[error("temperature must be positive, got {0}")]
    NonPositiveTemperature(f64),
    #[error("grid must contain at least one cell")]
    EmptyGrid,
    #[error("invalid time step {0}")]
    InvalidTimeStep(f64),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML serialize error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),
    #[error("TOML parse error: {0}")]
    TomlDeserialize(#[from] toml::de::Error),
    #[cfg(feature = "hdf5-export")]
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),
}
