pub mod vector;
pub mod distance;

pub use vector::Vector;
pub use distance::{DistanceType, DistanceFactory, EuclideanDistance, ManhattanDistance, CosineSimilarity};