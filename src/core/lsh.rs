use crate::core::{Vector, Distance};
use rand::Rng;
use rand_distr::{Normal, Cauchy, StandardNormal};
use std::collections::HashMap;

/// Enum for different hash function types
#[derive(Debug)]
enum HashFunction {
    Euclidean(EuclideanHashFunction),
    Manhattan(ManhattanHashFunction),
    CosineSim(CosineSimHashFunction),
}

impl HashFunction {
    fn hash(&self, v: &Vector) -> usize {
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
#[derive(Debug)]
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
    fn hash(&self, v: &Vector) -> usize {
        let projection = v.dot_product(&self.random_vector) + self.bias;
        (projection / self.width).floor() as usize
    }
}

/// L1LSH hash function for Manhattan distance
#[derive(Debug)]
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
    fn hash(&self, v: &Vector) -> usize {
        let projection = v.dot_product(&self.random_vector) + self.bias;
        (projection / self.width).floor() as usize
    }
}

/// SimHash function for cosine similarity
#[derive(Debug)]
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
    fn hash(&self, v: &Vector) -> usize {
        if v.dot_product(&self.random_vector) >= 0.0 {
            1
        } else {
            0
        }
    }
}

/// LSH index for approximate nearest neighbor search
pub struct LSHIndex {
    hash_functions: Vec<Vec<HashFunction>>,
    hash_tables: Vec<HashMap<usize, HashMap<String, Vector>>>,
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
    pub fn insert(&mut self, vector: Vector, key: String) {
        assert_eq!(vector.size(), self.dimensions, "Vector dimensions must match LSH index dimensions");
        assert!(!self.vector_map.contains_key(&key), "Key '{}' already exists in LSH index", key);
        self.vector_map.insert(key.clone(), vector.clone());
        for (table_idx, table_functions) in self.hash_functions.iter().enumerate() {
            let mut hash_values = Vec::with_capacity(table_functions.len());
            for hash_function in table_functions {
                hash_values.push(hash_function.hash(&vector));
            }
            let bucket_key = self.combine_hashes(&hash_values);
            self.hash_tables[table_idx]
                .entry(bucket_key)
                .or_insert_with(HashMap::new)
                .insert(key.clone(), vector.clone());
        }
    }
    
    /// Find the nearest neighbor to a query vector
    pub fn nearest_neighbor(&self, query: &Vector) -> Option<String> {
        self.nearest_neighbors(query, 1).into_iter().next().map(|(key, _)| key)
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
    pub fn nearest_neighbors(&self, query: &Vector, k: usize) -> Vec<(String, f32)> {
        assert_eq!(query.size(), self.dimensions, "Query vector dimensions must match LSH index dimensions");
        let mut candidates = HashMap::new();
        for (table_idx, table_functions) in self.hash_functions.iter().enumerate() {
            let mut hash_values = Vec::with_capacity(table_functions.len());
            for hash_function in table_functions {
                hash_values.push(hash_function.hash(query));
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
                let distance = self.distance_metric.distance(query, vector);
                (key.clone(), distance)
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }
    
    /// Combine multiple hash values into a single bucket key
    fn combine_hashes(&self, hash_values: &[usize]) -> usize {
        let mut combined: usize = 0;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_insertion_and_search_all_types() {
        // Test all distance types
        let distance_types = [Distance::Euclidean, Distance::Manhattan, Distance::CosineSim];
        
        for distance_type in distance_types {
            let mut lsh = LSHIndex::new(3, 5, 3, distance_type, 4.0);
            
            // Insert test vectors
            let vector1 = Vector::from_slice(&[1.0, 2.0, 3.0]);
            let vector2 = Vector::from_slice(&[4.0, 5.0, 6.0]);
            let vector3 = Vector::from_slice(&[7.0, 8.0, 9.0]);
            
            lsh.insert(vector1.clone(), "point1".to_string());
            lsh.insert(vector2.clone(), "point2".to_string());
            lsh.insert(vector3.clone(), "point3".to_string());
            
            // Test exact match search
            let results = lsh.nearest_neighbors(&vector1, 2);
            
            // Should find at least the exact match
            assert!(!results.is_empty(), "No results found for {:?}", distance_type);
            
            // The first result should be the exact match with distance 0 (or very close for floating point)
            if let Some((key, distance)) = results.first() {
                assert_eq!(key, "point1", "Wrong key for {:?}", distance_type);
                assert!(
                    distance.abs() < 1e-6, 
                    "Wrong distance for {:?}: expected ~0.0, got {}", 
                    distance_type, 
                    distance
                );
            }
            
            // Test nearest neighbor method
            let nearest = lsh.nearest_neighbor(&vector1);
            assert_eq!(nearest, Some("point1".to_string()), "Wrong nearest neighbor for {:?}", distance_type);
        }
    }

    #[test]
    #[should_panic(expected = "Vector dimensions must match LSH index dimensions")]
    fn test_lsh_dimension_mismatch_insert() {
        let mut lsh = LSHIndex::new(3, 2, 2, Distance::Euclidean, 4.0);
        lsh.insert(Vector::from_slice(&[1.0, 2.0]), "key1".to_string());
    }

    #[test]
    #[should_panic(expected = "Query vector dimensions must match LSH index dimensions")]
    fn test_lsh_dimension_mismatch_search() {
        let lsh = LSHIndex::new(3, 2, 2, Distance::Euclidean, 4.0);
        lsh.nearest_neighbors(&Vector::from_slice(&[1.0, 2.0]), 1);
    }

    #[test]
    fn test_distance_hash_function_creation() {
        // Test that we can create hash functions for each distance type
        let euclidean_hash = Distance::Euclidean.create_hash_function(3, 4.0);
        let manhattan_hash = Distance::Manhattan.create_hash_function(3, 4.0);
        let cosine_hash = Distance::CosineSim.create_hash_function(3, 4.0);
        
        let test_vector = Vector::from_slice(&[1.0, 2.0, 3.0]);
        euclidean_hash.hash(&test_vector);
        manhattan_hash.hash(&test_vector);
        cosine_hash.hash(&test_vector);
        // Should not panic
    }

    #[test]
    #[should_panic(expected = "Key 'key1' already exists in LSH index")]
    fn test_lsh_duplicate_key_prevention() {
        let mut lsh = LSHIndex::new(3, 2, 2, Distance::Euclidean, 4.0);
        
        lsh.insert(Vector::from_slice(&[1.0, 2.0, 3.0]), "key1".to_string());
        lsh.insert(Vector::from_slice(&[4.0, 5.0, 6.0]), "key1".to_string());
    }

    #[test]
    fn test_lsh_remove_functionality() {
        let mut lsh = LSHIndex::new(3, 2, 2, Distance::Euclidean, 4.0);
        
        let vector = Vector::from_slice(&[1.0, 2.0, 3.0]);
        lsh.insert(vector.clone(), "key1".to_string());
        
        // Should contain the key
        assert!(lsh.vector_map.contains_key("key1"));
        
        // Remove the key
        lsh.remove("key1");
        assert!(!lsh.vector_map.contains_key("key1"));
        
        // Search should return empty results
        let results = lsh.nearest_neighbors(&vector, 1);
        assert!(results.is_empty());
    }
} 