use crate::core::Vector;
use crate::QueryPerformance;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Cache key combining vector hash and k value
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    vector_hash: u64,
    k: usize,
}

impl CacheKey {
    fn new(vector: &Vector, k: usize) -> Self {
        let mut hasher = DefaultHasher::new();
        for &value in vector.data() {
            value.to_bits().hash(&mut hasher);
        }
        
        Self {
            vector_hash: hasher.finish(),
            k,
        }
    }
}

#[derive(Debug, Clone)]
struct CacheEntry {
    results: Vec<(String, f32)>,
}

pub struct QueryCache {
    fast_cache: HashMap<CacheKey, CacheEntry>,
    accurate_cache: HashMap<CacheKey, CacheEntry>,
}

impl QueryCache {
    /// Create a new query cache
    pub fn new() -> Self {
        Self {
            fast_cache: HashMap::new(),
            accurate_cache: HashMap::new(),
        }
    }
    
    /// Get cached results for a query vector
    /// If requesting fast but accurate is available, return accurate
    /// If requesting accurate but only fast is available, return None
    pub fn get(&self, query: &Vector, k: usize, performance: QueryPerformance) -> Option<Vec<(String, f32)>> {
        let key = CacheKey::new(query, k);
        
        match performance {
            QueryPerformance::Fast => {
                if let Some(entry) = self.accurate_cache.get(&key) {
                    Some(entry.results.clone())
                } else {
                    self.fast_cache.get(&key).map(|entry| entry.results.clone())
                }
            }
            QueryPerformance::Accurate => {
                self.accurate_cache.get(&key).map(|entry| entry.results.clone())
            }
        }
    }
    
    /// Store results for a query vector
    pub fn put(&mut self, query: &Vector, k: usize, performance: QueryPerformance, results: Vec<(String, f32)>) {
        let key = CacheKey::new(query, k);
        let entry = CacheEntry { results };
        
        match performance {
            QueryPerformance::Fast => {
                self.fast_cache.insert(key, entry);
            }
            QueryPerformance::Accurate => {
                self.accurate_cache.insert(key, entry);
            }
        }
    }
    
    /// Check if a query is cached for the given performance level
    pub fn contains(&self, query: &Vector, k: usize, performance: QueryPerformance) -> bool {
        let key = CacheKey::new(query, k);
        
        match performance {
            QueryPerformance::Fast => {
                self.accurate_cache.contains_key(&key) || self.fast_cache.contains_key(&key)
            }
            QueryPerformance::Accurate => {
                self.accurate_cache.contains_key(&key)
            }
        }
    }
    
    /// Clear all cached entries (called when database is modified)
    pub fn invalidate_all(&mut self) {
        self.fast_cache.clear();
        self.accurate_cache.clear();
    }
    
    /// Get the total number of cached entries
    pub fn len(&self) -> usize {
        self.fast_cache.len() + self.accurate_cache.len()
    }
    
    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.fast_cache.is_empty() && self.accurate_cache.is_empty()
    }
    
    /// Get the number of fast cached entries
    pub fn fast_cache_len(&self) -> usize {
        self.fast_cache.len()
    }
    
    /// Get the number of accurate cached entries
    pub fn accurate_cache_len(&self) -> usize {
        self.accurate_cache.len()
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_cache_put_and_get() {
        let mut cache = QueryCache::new();
        let query = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let results = vec![("key1".to_string(), 0.5), ("key2".to_string(), 1.0)];
        
        assert!(!cache.contains(&query, 5, QueryPerformance::Fast));
        assert!(cache.get(&query, 5, QueryPerformance::Fast).is_none());
        
        cache.put(&query, 5, QueryPerformance::Fast, results.clone());
        
        assert!(cache.contains(&query, 5, QueryPerformance::Fast));
        assert_eq!(cache.get(&query, 5, QueryPerformance::Fast), Some(results));
        assert_eq!(cache.fast_cache_len(), 1);
        assert_eq!(cache.accurate_cache_len(), 0);


        cache = QueryCache::new();
        let query = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let results = vec![("key1".to_string(), 0.5), ("key2".to_string(), 1.0)];
        
        cache.put(&query, 5, QueryPerformance::Accurate, results.clone());
        
        assert!(cache.contains(&query, 5, QueryPerformance::Accurate));
        assert_eq!(cache.get(&query, 5, QueryPerformance::Accurate), Some(results));
        assert_eq!(cache.fast_cache_len(), 0);
        assert_eq!(cache.accurate_cache_len(), 1);
    }

    #[test]
    fn test_query_cache_performance_preference() {
        let mut cache = QueryCache::new();
        let query = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let fast_results = vec![("fast_key".to_string(), 0.5)];
        let accurate_results = vec![("accurate_key".to_string(), 0.3)];
        
        // Put both fast and accurate results
        cache.put(&query, 5, QueryPerformance::Fast, fast_results.clone());
        cache.put(&query, 5, QueryPerformance::Accurate, accurate_results.clone());
        
        // When requesting fast, should return accurate (better quality)
        assert_eq!(cache.get(&query, 5, QueryPerformance::Fast), Some(accurate_results.clone()));
        
        // When requesting accurate, should return accurate
        assert_eq!(cache.get(&query, 5, QueryPerformance::Accurate), Some(accurate_results));
        
        // Fast should be available for fast requests
        assert!(cache.contains(&query, 5, QueryPerformance::Fast));
        // Accurate should be available for accurate requests
        assert!(cache.contains(&query, 5, QueryPerformance::Accurate));

        // Reset
        cache.invalidate_all();
        
        // Put only fast results
        cache.put(&query, 5, QueryPerformance::Fast, fast_results.clone());
        
        // Fast request should work
        assert_eq!(cache.get(&query, 5, QueryPerformance::Fast), Some(fast_results));
        
        // Accurate request should return None (no accurate results available)
        assert!(cache.get(&query, 5, QueryPerformance::Accurate).is_none());
        
        // Contains should reflect this
        assert!(cache.contains(&query, 5, QueryPerformance::Fast));
        assert!(!cache.contains(&query, 5, QueryPerformance::Accurate));
    }
} 