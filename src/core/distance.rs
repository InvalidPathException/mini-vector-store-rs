use crate::core::vector::Vector;

pub trait DistanceType {
    fn distance(&self, v1: &Vector, v2: &Vector) -> f32;
    fn name(&self) -> &'static str;
}

#[derive(Debug, Clone)]
pub struct EuclideanDistance;

impl DistanceType for EuclideanDistance {
    fn distance(&self, v1: &Vector, v2: &Vector) -> f32 {
        assert_eq!(v1.size(), v2.size(), "Vectors must have the same size");
        v1.data()
            .iter()
            .zip(v2.data())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn name(&self) -> &'static str {
        "euclidean"
    }
}

#[derive(Debug, Clone)]
pub struct ManhattanDistance;

impl DistanceType for ManhattanDistance {
    fn distance(&self, v1: &Vector, v2: &Vector) -> f32 {
        assert_eq!(v1.size(), v2.size(), "Vectors must have the same size");
        v1.data()
            .iter()
            .zip(v2.data())
            .map(|(a, b)| (a - b).abs())
            .sum()
    }

    fn name(&self) -> &'static str {
        "manhattan"
    }
}

#[derive(Debug, Clone)]
pub struct CosineSimilarity;

impl DistanceType for CosineSimilarity {
    fn distance(&self, v1: &Vector, v2: &Vector) -> f32 {
        assert_eq!(v1.size(), v2.size(), "Vectors must have the same size");

        let dot = v1.dot_product(v2);
        let norm1 = v1.norm();
        let norm2 = v2.norm();

        if norm1 == 0.0 || norm2 == 0.0 {
            return 1.0;
        }

        let cosine_similarity = (dot / (norm1 * norm2)).clamp(-1.0, 1.0);
        1.0 - cosine_similarity
    }

    fn name(&self) -> &'static str {
        "cosine"
    }
}

#[derive(Debug, Clone)]
pub struct ChebyshevDistance;

impl DistanceType for ChebyshevDistance {
    fn distance(&self, v1: &Vector, v2: &Vector) -> f32 {
        assert_eq!(v1.size(), v2.size(), "Vectors must have the same size");
        v1.data()
            .iter()
            .zip(v2.data())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    fn name(&self) -> &'static str {
        "chebyshev"
    }
}

pub struct DistanceFactory;

impl DistanceFactory {
    pub fn create(name: &str) -> Box<dyn DistanceType> {
        match name.to_lowercase().as_str() {
            "euclidean" | "e" => Box::new(EuclideanDistance),
            "manhattan" | "m" => Box::new(ManhattanDistance),
            "cosine"    | "c" => Box::new(CosineSimilarity),
            "chebyshev" | "t" => Box::new(ChebyshevDistance),
            _ => panic!("Unknown distance metric: {}", name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let d = EuclideanDistance;
        let v1 = Vector::from_slice(&[0.0, 0.0]);
        let v2 = Vector::from_slice(&[3.0, 4.0]);
        assert_eq!(d.distance(&v1, &v2), 5.0);
    }

    #[test]
    fn test_manhattan_distance() {
        let d = ManhattanDistance;
        let v1 = Vector::from_slice(&[0.0, 0.0]);
        let v2 = Vector::from_slice(&[3.0, 4.0]);
        assert_eq!(d.distance(&v1, &v2), 7.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let d = CosineSimilarity;
        let v1 = Vector::from_slice(&[1.0, 0.0]);
        let v2 = Vector::from_slice(&[1.0, 0.0]);
        // Same vectors should have distance 0 (similarity 1)
        assert!((d.distance(&v1, &v2) - 0.0).abs() < 1e-6);

        let v3 = Vector::from_slice(&[-1.0, 0.0]);
        // Opposite vectors should have distance 2 (similarity -1)
        assert!((d.distance(&v1, &v3) - 2.0).abs() < 1e-6);

        let v4 = Vector::from_slice(&[0.0, 1.0]);
        // Perpendicular vectors should have distance 1 (similarity 0)
        assert!((d.distance(&v1, &v4) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_chebyshev() {
        let v1 = Vector::from_slice(&[1.0, 5.0]);
        let v2 = Vector::from_slice(&[4.0, 1.0]);
        let d = ChebyshevDistance;
        assert_eq!(d.distance(&v1, &v2), 4.0);
    }

    #[test]
    fn test_distance_metric_factory() {
        let euclidean = DistanceFactory::create("e");
        let manhattan = DistanceFactory::create("m");
        let cosine    = DistanceFactory::create("c");
        let chebyshev = DistanceFactory::create("t");

        let v1 = Vector::from_slice(&[0.0, 0.0]);
        let v2 = Vector::from_slice(&[3.0, 4.0]);

        assert_eq!(euclidean.distance(&v1, &v2), 5.0);
        assert_eq!(manhattan.distance(&v1, &v2), 7.0);
        assert!((cosine.distance(&v1, &v2) - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Unknown distance metric")]
    fn test_unknown_metric() {
        DistanceFactory::create("unknown");
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same size")]
    fn test_size_mismatch() {
        let d = EuclideanDistance;
        let v1 = Vector::from_slice(&[1.0, 2.0]);
        let v2 = Vector::from_slice(&[1.0, 2.0, 3.0]);
        d.distance(&v1, &v2);
    }
} 