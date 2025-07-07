pub mod core;

use crate::core::{Distance, KDTree, LSHIndex};
use std::collections::HashMap;

/// Search result with metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub key: String,
    pub distance: f32,
    pub metadata: String,
}

/// Backing storage options for the vector database
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackingStorage {
    /// Use only KD-tree for exact search
    KDTreeOnly,
    /// Use only LSH for approximate search
    LSHOnly,
    /// Use both KD-tree and LSH for hybrid search
    Hybrid,
}

/// Query performance preference
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryPerformance {
    /// Prioritize speed over accuracy (uses LSH when available)
    Fast,
    /// Prioritize accuracy over speed (uses KD-tree when available)
    Accurate,
}

/// Main vector database implementation
pub struct VectorDatabase {
    kd_tree: Option<KDTree>,
    lsh_index: Option<LSHIndex>,
    backing_storage: BackingStorage,
    dimensions: usize,
    metadata_map: HashMap<String, String>,
}

impl VectorDatabase {
    /// Create a new vector database
    pub fn new(
        dimensions: usize,
        backing_storage: BackingStorage,
        lsh_params: Option<LSHParams>,
    ) -> Self {
        let distance_metric = Distance::Euclidean;
        
        let kd_tree = match backing_storage {
            BackingStorage::KDTreeOnly | BackingStorage::Hybrid => {
                Some(KDTree::new(dimensions, distance_metric))
            }
            BackingStorage::LSHOnly => None,
        };
        
        let lsh_index = match backing_storage {
            BackingStorage::LSHOnly | BackingStorage::Hybrid => {
                let params = lsh_params.unwrap_or_default();
                Some(LSHIndex::new(
                    dimensions,
                    params.num_tables,
                    params.num_hash_functions,
                    distance_metric,
                    params.width,
                ))
            }
            BackingStorage::KDTreeOnly => None,
        };
        
        Self {
            kd_tree,
            lsh_index,
            backing_storage,
            dimensions,
            metadata_map: HashMap::new(),
        }
    }
    
    /// Insert a vector with a key
    pub fn insert(&mut self, vector: Vector, key: String) {
        if vector.size() != self.dimensions {
            panic!("Vector dimensions must match database dimensions");
        }
        
        if let Some(ref mut kd_tree) = self.kd_tree {
            kd_tree.insert(vector.clone(), key.clone());
        }
        
        if let Some(ref mut lsh_index) = self.lsh_index {
            lsh_index.insert(vector, key);
        }
    }
    
    /// Insert a vector with a key and metadata
    pub fn insert_with_metadata(&mut self, vector: Vector, key: String, metadata: String) {
        self.metadata_map.insert(key.clone(), metadata);
        self.insert(vector, key);
    }
    
    /// Insert multiple vectors in batch
    pub fn batch_insert(&mut self, vectors: Vec<Vector>, keys: Vec<String>) {
        if vectors.len() != keys.len() {
            panic!("Number of vectors must match number of keys");
        }
        
        for (vector, key) in vectors.into_iter().zip(keys) {
            self.insert(vector, key);
        }
    }
    
    /// Perform similarity search
    pub fn similarity_search(&self, query: &Vector, k: usize, performance: QueryPerformance) -> Vec<(String, f32)> {
        if query.size() != self.dimensions {
            panic!("Query vector dimensions must match database dimensions");
        }
        
        match (self.backing_storage, performance) {
            // KD-tree only scenarios
            (BackingStorage::KDTreeOnly, _) => {
                if let Some(ref kd_tree) = self.kd_tree {
                    kd_tree.nearest_neighbors(query, k)
                } else {
                    Vec::new()
                }
            }
            
            // LSH only scenarios
            (BackingStorage::LSHOnly, _) => {
                if let Some(ref lsh_index) = self.lsh_index {
                    lsh_index.nearest_neighbors(query, k)
                } else {
                    Vec::new()
                }
            }
            
            // Hybrid scenarios
            (BackingStorage::Hybrid, QueryPerformance::Fast) => {
                // Use LSH for speed
                if let Some(ref lsh_index) = self.lsh_index {
                    lsh_index.nearest_neighbors(query, k)
                } else {
                    Vec::new()
                }
            }
            
            (BackingStorage::Hybrid, QueryPerformance::Accurate) => {
                // Use KD-tree for accuracy
                if let Some(ref kd_tree) = self.kd_tree {
                    kd_tree.nearest_neighbors(query, k)
                } else {
                    Vec::new()
                }
            }
        }
    }
    
    /// Perform similarity search with metadata
    pub fn similarity_search_with_metadata(&self, query: &Vector, k: usize, performance: QueryPerformance) -> Vec<SearchResult> {
        let results = self.similarity_search(query, k, performance);
        
        results
            .into_iter()
            .map(|(key, distance)| {
                let metadata = self.metadata_map.get(&key).cloned().unwrap_or_default();
                SearchResult {
                    key,
                    distance,
                    metadata,
                }
            })
            .collect()
    }
    
    /// Perform batch similarity search
    pub fn batch_similarity_search(&self, queries: Vec<Vector>, k: usize, performance: QueryPerformance) -> Vec<Vec<(String, f32)>> {
        queries
            .iter()
            .map(|query| self.similarity_search(query, k, performance))
            .collect()
    }
    
    /// Get metadata for a key
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata_map.get(key)
    }
    
    /// Get a vector by key
    pub fn get_vector(&self, key: &str) -> Option<&Vector> {
        if let Some(ref kd_tree) = self.kd_tree {
            if let Some(vector) = kd_tree.get_vector(key) {
                return Some(vector);
            }
        }
        
        if let Some(ref lsh_index) = self.lsh_index {
            return lsh_index.get_vector(key);
        }
        
        None
    }
    
    /// Get the number of dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    
    /// Get the current backing storage
    pub fn backing_storage(&self) -> BackingStorage {
        self.backing_storage
    }
    
    /// Get all vectors (from either KD-tree or LSH index)
    pub fn get_all_vectors(&self) -> &HashMap<String, Vector> {
        if let Some(ref kd_tree) = self.kd_tree {
            return kd_tree.get_all_vectors();
        } else if let Some(ref lsh_index) = self.lsh_index {
            return lsh_index.get_all_vectors();
        }
        
        // Return empty HashMap reference - we need to store this somewhere
        // For now, we'll use a static empty HashMap
        static EMPTY: std::sync::OnceLock<HashMap<String, Vector>> = std::sync::OnceLock::new();
        EMPTY.get_or_init(HashMap::new)
    }
    
    /// Get the number of vectors in the database
    pub fn size(&self) -> usize {
        if let Some(ref kd_tree) = self.kd_tree {
            kd_tree.size()
        } else if let Some(ref lsh_index) = self.lsh_index {
            lsh_index.size()
        } else {
            0
        }
    }
    
    /// Check if the database is empty
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }
    
    /// Remove a vector by key
    pub fn remove(&mut self, key: &str) {
        if let Some(ref mut kd_tree) = self.kd_tree {
            kd_tree.remove(key);
        }
        
        if let Some(ref mut lsh_index) = self.lsh_index {
            lsh_index.remove(key);
        }
        
        self.metadata_map.remove(key);
    }
}

/// Parameters for LSH index configuration
#[derive(Debug, Clone)]
pub struct LSHParams {
    pub num_tables: usize,
    pub num_hash_functions: usize,
    pub width: f32,
}

impl Default for LSHParams {
    fn default() -> Self {
        Self {
            num_tables: 20,
            num_hash_functions: 4,
            width: 4.0,
        }
    }
}

// Re-export core types for public use
pub use crate::core::Vector;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_database_insertion() {
        let mut db = VectorDatabase::new(3, BackingStorage::KDTreeOnly, None);
        
        let vector = Vector::from_slice(&[1.0, 2.0, 3.0]);
        db.insert(vector, "key1".to_string());
        
        assert_eq!(db.size(), 1);
        assert!(!db.is_empty());
    }

    #[test]
    fn test_vector_database_insertion_with_metadata() {
        let mut db = VectorDatabase::new(3, BackingStorage::KDTreeOnly, None);
        
        let vector = Vector::from_slice(&[1.0, 2.0, 3.0]);
        db.insert_with_metadata(vector, "key1".to_string(), "metadata1".to_string());
        
        assert_eq!(db.get_metadata("key1"), Some(&"metadata1".to_string()));
    }

    #[test]
    fn test_vector_database_similarity_search() {
        let mut db = VectorDatabase::new(3, BackingStorage::KDTreeOnly, None);
        
        db.insert(Vector::from_slice(&[1.0, 1.0, 1.0]), "point1".to_string());
        db.insert(Vector::from_slice(&[2.0, 2.0, 2.0]), "point2".to_string());
        db.insert(Vector::from_slice(&[3.0, 3.0, 3.0]), "point3".to_string());
        
        let query = Vector::from_slice(&[1.5, 1.5, 1.5]);
        let results = db.similarity_search(&query, 2, QueryPerformance::Accurate);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "point1");
        assert_eq!(results[1].0, "point2");
    }

    #[test]
    fn test_batch_insert() {
        let mut db = VectorDatabase::new(3, BackingStorage::KDTreeOnly, None);
        
        let vectors = vec![
            Vector::from_slice(&[1.0, 2.0, 3.0]),
            Vector::from_slice(&[4.0, 5.0, 6.0]),
        ];
        let keys = vec!["key1".to_string(), "key2".to_string()];
        
        db.batch_insert(vectors, keys);
        assert_eq!(db.size(), 2);
    }

    #[test]
    #[should_panic(expected = "Vector dimensions must match database dimensions")]
    fn test_dimension_mismatch() {
        let mut db = VectorDatabase::new(3, BackingStorage::KDTreeOnly, None);
        db.insert(Vector::from_slice(&[1.0, 2.0]), "key1".to_string());
    }
}
