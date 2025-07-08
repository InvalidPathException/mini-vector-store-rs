use crate::core::{Vector, Distance};
use crate::error::VectorError;
use rand::Rng;
use rand_distr::{Normal, Cauchy, StandardNormal};
use std::collections::HashMap;

/// Enum for different hash function types
#[derive(Debug, Clone)]
enum HashFunction {
    Euclidean(EuclideanHashFunction),
    Manhattan(ManhattanHashFunction),
    CosineSim(CosineSimHashFunction),
}

impl HashFunction {
    fn hash(&self, v: &Vector) -> Result<isize, VectorError> {
        match self {
            HashFunction::Euclidean(hf) => hf.hash(v),
            HashFunction::Manhattan(hf) => hf.hash(v),
            HashFunction::CosineSim(hf) => hf.hash(v),
        }
    }
}

impl Distance {
    fn create_hash_function(&self, dims: usize, width: f32) -> HashFunction {
        match self {
            Distance::Euclidean => HashFunction::Euclidean(EuclideanHashFunction::new(dims, width)),
            Distance::Manhattan => HashFunction::Manhattan(ManhattanHashFunction::new(dims, width)),
            Distance::CosineSim => HashFunction::CosineSim(CosineSimHashFunction::new(dims)),
        }
    }
}

/// E2LSH hash function for Euclidean distance
#[derive(Debug, Clone)]
struct EuclideanHashFunction {
    random_vector: Vector,
    bias: f32,
    width: f32,
}

impl EuclideanHashFunction {
    fn new(dims: usize, width: f32) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let mut random_vector = Vector::new(dims);
        for i in 0..dims {
            random_vector[i] = rng.sample(normal);
        }
        
        let bias = rng.gen_range(0.0..width);
        
        Self {
            random_vector,
            bias,
            width,
        }
    }
}

impl EuclideanHashFunction {
    fn hash(&self, v: &Vector) -> Result<isize, VectorError> {
        let projection = self.random_vector.dot_product(v)? + self.bias;
        
        Ok((projection / self.width).floor() as isize)
    }
}

/// L1LSH hash function for Manhattan distance
#[derive(Debug, Clone)]
struct ManhattanHashFunction {
    random_vector: Vector,
    bias: f32,
    width: f32,
}

impl ManhattanHashFunction {
    fn new(dims: usize, width: f32) -> Self {
        let mut rng = rand::thread_rng();
        let cauchy = Cauchy::new(0.0, 1.0).unwrap();
        
        let mut random_vector = Vector::new(dims);
        for i in 0..dims {
            random_vector[i] = rng.sample(cauchy);
        }
        
        let bias = rng.gen_range(0.0..width);
        
        Self {
            random_vector,
            bias,
            width,
        }
    }
}

impl ManhattanHashFunction {
    fn hash(&self, v: &Vector) -> Result<isize, VectorError> {
        let projection = self.random_vector.dot_product(v)? + self.bias;
        
        Ok((projection / self.width).floor() as isize)
    }
}

/// SimHash function for cosine similarity
#[derive(Debug, Clone)]
struct CosineSimHashFunction {
    random_vector: Vector,
}

impl CosineSimHashFunction {
    fn new(dims: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        
        let mut random_vector = Vector::new(dims);
        for i in 0..dims {
            random_vector[i] = rng.sample(normal);
        }
        
        Self { random_vector }
    }
}

impl CosineSimHashFunction {
    fn hash(&self, v: &Vector) -> Result<isize, VectorError> {
        if self.random_vector.dot_product(v)? >= 0.0 {
            Ok(1)
        } else {
            Ok(0)
        }
    }
}

/// LSH index for approximate nearest neighbor search
pub struct LSHIndex {
    hash_functions: Vec<Vec<HashFunction>>,
    hash_tables: Vec<HashMap<isize, HashMap<String, Vector>>>,
    vector_map: HashMap<String, Vector>,
    dimensions: usize,
    distance_metric: Distance
}

impl LSHIndex {
    /// Create a new LSH index
    pub fn new(
        dimensions: usize,
        num_tables: usize,
        num_hash_functions: usize,
        distance_metric: Distance,
        width: f32,
    ) -> Self {
        let mut hash_functions = Vec::with_capacity(num_tables);
        let mut hash_tables = Vec::with_capacity(num_tables);
        for _ in 0..num_tables {
            let mut table_functions = Vec::with_capacity(num_hash_functions);
            for _ in 0..num_hash_functions {
                let hash_function = distance_metric.create_hash_function(dimensions, width);
                table_functions.push(hash_function);
            }
            hash_functions.push(table_functions);
            hash_tables.push(HashMap::new());
        }
        Self {
            hash_functions,
            hash_tables,
            vector_map: HashMap::new(),
            dimensions,
            distance_metric
        }
    }
    
    /// Insert a vector into the LSH index
    pub fn insert(&mut self, vector: Vector, key: String) -> Result<(), VectorError> {
        if vector.size() != self.dimensions {
            return Err(VectorError::DimensionsMismatch { expected: self.dimensions, found: vector.size() });
        }
        if self.vector_map.contains_key(&key) {
            return Err(VectorError::KeyAlreadyExists(key));
        }
        self.vector_map.insert(key.clone(), vector.clone());
        for (table_idx, table_functions) in self.hash_functions.iter().enumerate() {
            let mut hash_values = Vec::with_capacity(table_functions.len());
            for hash_function in table_functions {
                hash_values.push(hash_function.hash(&vector)?);
            }
            let bucket_key = self.combine_hashes(&hash_values);
            self.hash_tables[table_idx]
                .entry(bucket_key)
                .or_default()
                .insert(key.clone(), vector.clone());
        }
        Ok(())
    }
    
    /// Find the nearest neighbor to a query vector
    pub fn nearest_neighbor(&self, query: &Vector) -> Result<Option<String>, VectorError> {
        self.nearest_neighbors(query, 1).map(|mut results| results.pop().map(|(key, _)| key))
    }
    
    /// Remove a vector from the LSH index by key
    pub fn remove(&mut self, key: &str) {
        self.vector_map.remove(key);
        for table in &mut self.hash_tables {
            for bucket in table.values_mut() {
                bucket.remove(key);
            }
        }
    }

    /// Search for approximate nearest neighbors, no guarantee you will get all k nearest neighbors
    pub fn nearest_neighbors(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>, VectorError> {
        if query.size() != self.dimensions {
            return Err(VectorError::DimensionsMismatch { expected: self.dimensions, found: query.size() });
        }
        let mut candidates = HashMap::new();
        for (table_idx, table_functions) in self.hash_functions.iter().enumerate() {
            let mut hash_values = Vec::with_capacity(table_functions.len());
            for hash_function in table_functions {
                hash_values.push(hash_function.hash(query)?);
            }
            let bucket_key = self.combine_hashes(&hash_values);
            if let Some(bucket) = self.hash_tables[table_idx].get(&bucket_key) {
                for (key, vector) in bucket {
                    candidates.insert(key.clone(), vector.clone());
                }
            }
        }
        let mut results: Vec<(String, f32)> = candidates
            .iter()
            .map(|(key, vector)| {
                let distance = self.distance_metric.distance(query, vector).unwrap();
                (key.clone(), distance)
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        Ok(results)
    }
    
    /// Combine multiple hash values into a single bucket key
    fn combine_hashes(&self, hash_values: &[isize]) -> isize {
        let mut combined: isize = 0;
        for &hash_value in hash_values {
            combined = combined.wrapping_mul(31).wrapping_add(hash_value);
        }
        combined
    }
    
    /// Get the hash type name
    pub fn hash_type_name(&self) -> &'static str {
        match self.distance_metric {
            Distance::Euclidean => "euclidean",
            Distance::Manhattan => "manhattan",
            Distance::CosineSim => "cosinesim",
        }
    }

    /// Get a vector by key
    pub fn get_vector(&self, key: &str) -> Option<&Vector> {
        self.vector_map.get(key)
    }

    /// Get the number of vectors in the index
    pub fn size(&self) -> usize {
        self.vector_map.len()
    }

    /// Get all vectors in the index
    pub fn get_all_vectors(&self) -> &HashMap<String, Vector> {
        &self.vector_map
    }

    /// Get the distance metric used by this LSH index
    pub fn distance_metric(&self) -> Distance {
        self.distance_metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Vector;

    #[test]
    fn test_lsh_comprehensive_search_with_negative_buckets() {
        fn fixed_lsh_with_custom_hash(distance: Distance) -> LSHIndex {
            let dims = 2;
            let width = 1.0;
            let vector = Vector::from_slice(&[1.0, 1.0]); // Fixed vector for dot product
    
            let hash_function = match distance {
                Distance::Euclidean => HashFunction::Euclidean(EuclideanHashFunction {
                    random_vector: vector.clone(),
                    bias: 0.0,
                    width,
                }),
                Distance::Manhattan => HashFunction::Manhattan(ManhattanHashFunction {
                    random_vector: vector.clone(),
                    bias: 0.0,
                    width,
                }),
                Distance::CosineSim => HashFunction::CosineSim(CosineSimHashFunction {
                    random_vector: vector.clone(),
                }),
            };
    
            let hash_functions = vec![vec![hash_function; 5]; 10];
            let hash_tables = vec![HashMap::new(); 10];
    
            LSHIndex {
                hash_functions,
                hash_tables,
                vector_map: HashMap::new(),
                dimensions: dims,
                distance_metric: distance,
            }
        }
    
        fn run_test(distance: Distance) {
            let mut lsh = fixed_lsh_with_custom_hash(distance);
        
            // Positive projection case
            let v1 = Vector::from_slice(&[1.0, 2.0]);
            let v2 = Vector::from_slice(&[3.0, 4.0]);
            lsh.insert(v1.clone(), "key1".to_string()).unwrap();
            lsh.insert(v2.clone(), "key2".to_string()).unwrap();
        
            let query_pos = Vector::from_slice(&[1.0, 2.0]);
            let result_pos = lsh.nearest_neighbors(&query_pos, 1).unwrap();
            assert_eq!(result_pos[0].0, "key1", "Expected key1 for {:?} (positive projection)", distance);
        
            // Negative projection case
            let mut lsh_neg = fixed_lsh_with_custom_hash(distance);
        
            let v3 = Vector::from_slice(&[-1.0, -2.0]);
            let v4 = Vector::from_slice(&[-3.0, -4.0]);
            lsh_neg.insert(v3.clone(), "neg1".to_string()).unwrap();
            lsh_neg.insert(v4.clone(), "neg2".to_string()).unwrap();
        
            let query_neg = Vector::from_slice(&[-1.0, -2.0]);
            let result_neg = lsh_neg.nearest_neighbors(&query_neg, 1).unwrap();
            assert_eq!(result_neg[0].0, "neg1", "Expected neg1 for {:?} (negative projection)", distance);
        }
    
        run_test(Distance::Euclidean);
        run_test(Distance::Manhattan);
        run_test(Distance::CosineSim);
    }
    
    #[test]
    fn test_lsh_dimension_mismatch_errors() {
        let mut lsh = LSHIndex::new(3, 10, 5, Distance::Euclidean, 1.0);
        
        let vector_2d = Vector::from_slice(&[1.0, 2.0]);
        let result = lsh.insert(vector_2d.clone(), "key1".to_string());
        assert!(result.is_err());
        
        let result = lsh.nearest_neighbor(&vector_2d);
        assert!(result.is_err());
    }

    #[test]
    fn test_lsh_duplicate_key_prevention() {
        let mut lsh = LSHIndex::new(2, 10, 5, Distance::Euclidean, 1.0);
        let vector = Vector::from_slice(&[1.0, 1.0]);
        lsh.insert(vector.clone(), "key1".to_string()).unwrap();
        let result = lsh.insert(vector, "key1".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_lsh_remove_and_cleanup() {
        let mut lsh = LSHIndex::new(2, 10, 5, Distance::Euclidean, 1.0);
        let vector = Vector::from_slice(&[1.0, 1.0]);
        lsh.insert(vector, "key1".to_string()).unwrap();
        assert!(lsh.vector_map.contains_key("key1"));

        lsh.remove("key1");
        assert!(!lsh.vector_map.contains_key("key1"));

        let results = lsh.nearest_neighbors(&Vector::from_slice(&[1.0, 1.0]), 1).unwrap();
        assert!(results.is_empty());
    }
} 