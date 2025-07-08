use std::fmt;

#[derive(Debug)]
pub enum VectorError {
    DimensionsMismatch { expected: usize, found: usize },
    KeysAndVectorsMismatch,
    KeyNotFound,
    KeyAlreadyExists(String),
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            VectorError::DimensionsMismatch { expected, found } => {
                write!(f, "Dimension mismatch: expected {expected}, found {found}")
            }
            VectorError::KeysAndVectorsMismatch => write!(f, "Number of vectors must match number of keys"),
            VectorError::KeyNotFound => write!(f, "Key not found in the index"),
            VectorError::KeyAlreadyExists(ref key) => write!(f, "Key '{key}' already exists"),
        }
    }
}

impl std::error::Error for VectorError {} 