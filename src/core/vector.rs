use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vector {
    data: Vec<f32>,
}

impl Vector {
    /// Create a new vector with the specified size, initialized to zeros
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
        }
    }

    /// Create a new vector from a slice of f32 values
    pub fn from_slice(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }

    /// Create a new vector from a Vec<f32>
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Get the size (number of dimensions) of the vector
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Compute the dot product of two vectors
    pub fn dot_product(&self, other: &Vector) -> f32 {
        assert_eq!(self.size(), other.size(), "Vectors must have the same size for dot product");

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Compute the L2 norm (magnitude) of the vector
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize the vector to unit length, mutates the vector
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 0.0 {
            for x in &mut self.data {
                *x /= norm;
            }
        }
    }

    /// Create a normalized copy of the vector
    pub fn normalized(&self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }

    /// Returns a new vector equal to self + other
    pub fn add(&self, other: &Vector) -> Vector {
        assert_eq!(self.size(), other.size(), "Vectors must have the same size for addition");

        Vector {
            data: self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Returns a new vector equal to self - other
    pub fn subtract(&self, other: &Vector) -> Vector {
        assert_eq!(self.size(), other.size(), "Vectors must have the same size for subtraction");

        Vector {
            data: self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Returns a new vector equals to this vector multiplied by a scalar
    pub fn scale(&self, scalar: f32) -> Vector {
        Vector {
            data: self.data.iter().map(|x| x * scalar).collect(),
        }
    }
}

impl Index<usize> for Vector {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Hash for Vector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the bytes of the vector data
        for &x in &self.data {
            x.to_bits().hash(state);
        }
    }
}

impl Eq for Vector {}

impl std::fmt::Display for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, &x) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", x)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let v = Vector::new(3);
        assert_eq!(v.size(), 3);
        assert_eq!(v.data(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_vector_from_slice() {
        let data = [1.0, 2.0, 3.0];
        let v = Vector::from_slice(&data);
        assert_eq!(v.size(), 3);
        assert_eq!(v.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_indexing() {
        let mut v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);

        v[1] = 5.0;
        assert_eq!(v[1], 5.0);
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let v2 = Vector::from_slice(&[4.0, 5.0, 6.0]);
        assert_eq!(v1.dot_product(&v2), 32.0);
    }

    #[test]
    fn test_norm() {
        let v = Vector::from_slice(&[3.0, 4.0]);
        assert_eq!(v.norm(), 5.0);
    }

    #[test]
    fn test_normalize() {
        let mut v = Vector::from_slice(&[3.0, 4.0]);
        v.normalize();
        assert!((v.norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_operations() {
        let v1 = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let v2 = Vector::from_slice(&[4.0, 5.0, 6.0]);

        let sum = v1.add(&v2);
        assert_eq!(sum.data(), &[5.0, 7.0, 9.0]);

        let diff = v1.subtract(&v2);
        assert_eq!(diff.data(), &[-3.0, -3.0, -3.0]);

        let scaled = v1.scale(2.0);
        assert_eq!(scaled.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same size for dot product")]
    fn test_dot_product_size_mismatch() {
        let v1 = Vector::from_slice(&[1.0, 2.0]);
        let v2 = Vector::from_slice(&[1.0, 2.0, 3.0]);
        v1.dot_product(&v2);
    }
}