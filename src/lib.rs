pub mod core;
pub mod error;
pub mod feature;

use crate::core::{Distance, KDTree, LSHIndex};
use crate::error::VectorError;
use std::collections::HashMap;
use std::cell::RefCell;

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

pub struct VectorDatabase {
    kd_tree: Option<KDTree>,
    lsh_index: Option<LSHIndex>,
    backing_storage: BackingStorage,
    dimensions: usize,
    metadata_map: HashMap<String, String>,
    cache: Option<RefCell<crate::feature::QueryCache>>,
}

impl VectorDatabase {
    /// Create a new vector database
    /// 
    /// # Arguments
    /// 
    /// * `dimensions` - The number of dimensions for vectors
    /// * `backing_storage` - The storage backend to use (KDTree, LSH, or Hybrid)
    /// * `lsh_params` - Optional LSH parameters (only used for LSH and Hybrid storage)
    /// * `enable_cache` - Whether to enable query result caching
    /// 
    /// # Examples
    /// 
    /// ```
    /// use mini_vector_store_rs::{VectorDatabase, BackingStorage};
    /// 
    /// // Create without caching
    /// let db = VectorDatabase::new(128, BackingStorage::KDTreeOnly, None, false);
    /// 
    /// // Create with caching enabled
    /// let db_cached = VectorDatabase::new(128, BackingStorage::KDTreeOnly, None, true);
    /// ```
    pub fn new(
        dimensions: usize,
        backing_storage: BackingStorage,
        lsh_params: Option<LSHParams>,
        enable_cache: bool,
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
        
        let cache = if enable_cache {
            Some(RefCell::new(crate::feature::QueryCache::new()))
        } else {
            None
        };
        
        Self {
            kd_tree,
            lsh_index,
            backing_storage,
            dimensions,
            metadata_map: HashMap::new(),
            cache,
        }
    }
    
    /// Insert a vector with a key
    pub fn insert(&mut self, vector: Vector, key: String) -> Result<(), VectorError> {
        if vector.size() != self.dimensions {
            return Err(VectorError::DimensionsMismatch { expected: self.dimensions, found: vector.size() });
        }
        
        if let Some(ref mut kd_tree) = self.kd_tree {
            kd_tree.insert(vector.clone(), key.clone())?;
        }
        
        if let Some(ref mut lsh_index) = self.lsh_index {
            lsh_index.insert(vector, key)?;
        }
        
        // Invalidate cause result/error instead of panicche when data is modified
        if let Some(ref cache) = self.cache {
            cache.borrow_mut().invalidate_all();
        }
        
        Ok(())
    }
    
    /// Insert a vector with a key and metadata
    pub fn insert_with_metadata(&mut self, vector: Vector, key: String, metadata: String) -> Result<(), VectorError> {
        self.metadata_map.insert(key.clone(), metadata);
        self.insert(vector, key)
    }
    
    /// Insert multiple vectors in batch
    pub fn batch_insert(&mut self, vectors: Vec<Vector>, keys: Vec<String>) -> Result<(), VectorError> {
        if vectors.len() != keys.len() {
            return Err(VectorError::KeysAndVectorsMismatch);
        }
        
        for (vector, key) in vectors.into_iter().zip(keys) {
            self.insert(vector, key)?;
        }
        Ok(())
    }
    
    /// Perform similarity search
    pub fn similarity_search(&self, query: &Vector, k: usize, performance: QueryPerformance) -> Result<Vec<(String, f32)>, VectorError> {
        if query.size() != self.dimensions {
            return Err(VectorError::DimensionsMismatch { expected: self.dimensions, found: query.size() });
        }
        
        // Check cache first if enabled
        if let Some(ref cache) = self.cache {
            if let Some(cached_results) = cache.borrow().get(query, k, performance) {
                return Ok(cached_results);
            }
        }
        
        let (results, actual_performance) = match (self.backing_storage, performance) {
            // KD-tree only scenarios
            (BackingStorage::KDTreeOnly, _) => {
                if let Some(ref kd_tree) = self.kd_tree {
                    (kd_tree.nearest_neighbors(query, k)?, QueryPerformance::Accurate)
                } else {
                    (Vec::new(), QueryPerformance::Accurate)
                }
            }
            
            // LSH only scenarios
            (BackingStorage::LSHOnly, _) => {
                if let Some(ref lsh_index) = self.lsh_index {
                    (lsh_index.nearest_neighbors(query, k)?, QueryPerformance::Fast)
                } else {
                    (Vec::new(), QueryPerformance::Fast)
                }
            }
            
            // Hybrid scenarios
            (BackingStorage::Hybrid, QueryPerformance::Fast) => {
                // Use LSH for speed
                if let Some(ref lsh_index) = self.lsh_index {
                    (lsh_index.nearest_neighbors(query, k)?, QueryPerformance::Fast)
                } else {
                    (Vec::new(), QueryPerformance::Fast)
                }
            }
            
            (BackingStorage::Hybrid, QueryPerformance::Accurate) => {
                // Use KD-tree for accuracy
                if let Some(ref kd_tree) = self.kd_tree {
                    (kd_tree.nearest_neighbors(query, k)?, QueryPerformance::Accurate)
                } else {
                    (Vec::new(), QueryPerformance::Accurate)
                }
            }
        };
        
        // Store results in cache if enabled
        if let Some(ref cache) = self.cache {
            cache.borrow_mut().put(query, k, actual_performance, results.clone());
        }
        
        Ok(results)
    }
    
    /// Perform similarity search with metadata
    pub fn similarity_search_with_metadata(&self, query: &Vector, k: usize, performance: QueryPerformance) -> Result<Vec<SearchResult>, VectorError> {
        let results = self.similarity_search(query, k, performance)?;
        
        Ok(results
            .into_iter()
            .map(|(key, distance)| {
                let metadata = self.metadata_map.get(&key).cloned().unwrap_or_default();
                SearchResult {
                    key,
                    distance,
                    metadata,
                }
            })
            .collect())
    }
    
    /// Perform batch similarity search
    pub fn batch_similarity_search(&self, queries: Vec<Vector>, k: usize, performance: QueryPerformance) -> Result<Vec<Vec<(String, f32)>>, VectorError> {
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
        
        // Invalidate cache when data is modified
        if let Some(ref cache) = self.cache {
            cache.borrow_mut().invalidate_all();
        }
    }
    
    /// Check if caching is enabled
    pub fn is_cache_enabled(&self) -> bool {
        self.cache.is_some()
    }
    
    /// Get cache statistics (returns None if caching is disabled)
    pub fn cache_stats(&self) -> Option<(usize, usize, usize)> {
        self.cache.as_ref().map(|cache| {
            let cache_ref = cache.borrow();
            (cache_ref.len(), cache_ref.fast_cache_len(), cache_ref.accurate_cache_len())
        })
    }
    
    /// Manually clear the cache (returns false if caching is disabled)
    pub fn clear_cache(&mut self) -> bool {
        if let Some(ref cache) = self.cache {
            cache.borrow_mut().invalidate_all();
            true
        } else {
            false
        }
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
            num_tables: 10,
            num_hash_functions: 5,
            width: 4.0,
        }
    }
}

// Re-export core types for public use
pub use crate::core::Vector;
pub use crate::feature::QueryCache;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_database_basic_insertion() {
        let mut db = VectorDatabase::new(2, BackingStorage::KDTreeOnly, None, false);
        let vector = Vector::from_slice(&[1.0, 2.0]);
        db.insert(vector, "test".to_string()).unwrap();
        assert!(!db.is_empty());
    }

    #[test]
    fn test_vector_database_insertion_with_metadata() {
        let mut db = VectorDatabase::new(2, BackingStorage::KDTreeOnly, None, false);
        let vector = Vector::from_slice(&[1.0, 2.0]);
        db.insert_with_metadata(vector, "test".to_string(), "metadata".to_string()).unwrap();
        assert_eq!(db.get_metadata("test"), Some(&"metadata".to_string()));
    }

    #[test]
    fn test_vector_database_similarity_search() {
        let mut db = VectorDatabase::new(2, BackingStorage::KDTreeOnly, None, false);
        let vector1 = Vector::from_slice(&[1.0, 2.0]);
        let vector2 = Vector::from_slice(&[3.0, 4.0]);
        db.insert(vector1, "test1".to_string()).unwrap();
        db.insert(vector2, "test2".to_string()).unwrap();

        let query = Vector::from_slice(&[1.0, 1.0]);
        let results = db.similarity_search(&query, 1, QueryPerformance::Accurate).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "test1");
    }

    #[test]
    fn test_vector_database_batch_operations() {
        let mut db = VectorDatabase::new(2, BackingStorage::KDTreeOnly, None, false);
        let vectors = vec![
            Vector::from_slice(&[1.0, 2.0]),
            Vector::from_slice(&[3.0, 4.0]),
        ];
        let keys = vec!["test1".to_string(), "test2".to_string()];
        db.batch_insert(vectors, keys).unwrap();
        assert_eq!(db.size(), 2);
    }

    #[test]
    fn test_vector_database_dimension_mismatch_errors() {
        let mut db = VectorDatabase::new(3, BackingStorage::KDTreeOnly, None, false);
        let vector = Vector::from_slice(&[1.0, 2.0]);
        let result = db.insert(vector, "test".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_database_with_caching() {
        let mut db = VectorDatabase::new(2, BackingStorage::KDTreeOnly, None, true);
        assert!(db.is_cache_enabled());
        
        // Insert vectors
        let vector1 = Vector::from_slice(&[1.0, 2.0]);
        let vector2 = Vector::from_slice(&[3.0, 4.0]);
        db.insert(vector1, "test1".to_string()).unwrap();
        db.insert(vector2, "test2".to_string()).unwrap();

        // First search should miss cache
        let query1 = Vector::from_slice(&[1.0, 1.0]);
        let results1 = db.similarity_search(&query1, 1, QueryPerformance::Accurate).unwrap();
        assert_eq!(results1.len(), 1);
        
        // Check cache stats
        let stats = db.cache_stats().unwrap();
        assert_eq!(stats.0, 1); // total entries
        assert_eq!(stats.2, 1); // accurate entries
        
        // Insert should invalidate cache
        let vector3 = Vector::from_slice(&[5.0, 6.0]);
        db.insert(vector3, "test3".to_string()).unwrap();
        let stats = db.cache_stats().unwrap();
        assert_eq!(stats.0, 0); // total entries

        let query2 = Vector::from_slice(&[1.0, 2.0]);
        let results2 = db.similarity_search(&query2, 1, QueryPerformance::Accurate).unwrap();
        assert_eq!(results2.len(), 1);
        
        // Clear cache manually
        assert!(db.clear_cache());
        let stats = db.cache_stats().unwrap();
        assert_eq!(stats.0, 0);
    }

    #[test]
    fn test_vector_database_cache_performance_preference() {
        let mut db = VectorDatabase::new(2, BackingStorage::Hybrid, None, true);
        
        let vector1 = Vector::from_slice(&[1.0, 2.0]);
        db.insert(vector1, "test1".to_string()).unwrap();
        
        let query = Vector::from_slice(&[1.0, 1.0]);
        
        // First search with Fast performance (will use LSH)
        let _results_fast = db.similarity_search(&query, 1, QueryPerformance::Fast).unwrap();
        
        // Second search with Accurate performance (will use KD-tree)
        let _results_accurate = db.similarity_search(&query, 1, QueryPerformance::Accurate).unwrap();
        
        // Both searches should be cached
        let stats = db.cache_stats().unwrap();
        assert_eq!(stats.0, 2); // total entries (fast + accurate)
        
        // If we request Fast again, should return Accurate if available
        let _results_fast2 = db.similarity_search(&query, 1, QueryPerformance::Fast).unwrap();
        
        // Cache should still have the same entries
        let stats = db.cache_stats().unwrap();
        assert_eq!(stats.0, 2); // should still be 2
    }
}
