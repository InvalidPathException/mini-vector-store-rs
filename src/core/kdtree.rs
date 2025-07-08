use crate::core::{Vector, Distance};
use crate::error::VectorError;
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;

struct KDTreeNode {
    vector: Vector,
    key: String,
    left: Option<Box<KDTreeNode>>,
    right: Option<Box<KDTreeNode>>,
}

impl KDTreeNode {
    fn new(vector: Vector, key: String) -> Self {
        Self {
            vector,
            key,
            left: None,
            right: None,
        }
    }
}

pub struct KDTree {
    root: Option<Box<KDTreeNode>>,
    dimensions: usize,
    distance_metric: Distance,
    vector_map: HashMap<String, Vector>,
    tombstones: HashMap<String, Vector>,
}

impl KDTree {
    pub fn new(dimensions: usize, distance_metric: Distance) -> Self {
        Self {
            root: None,
            dimensions,
            distance_metric,
            vector_map: HashMap::new(),
            tombstones: HashMap::new(),
        }
    }

    /// Insert a vector into the KD-tree
    pub fn insert(&mut self, vector: Vector, key: String) -> Result<(), VectorError> {
        if vector.size() != self.dimensions {
            return Err(VectorError::DimensionsMismatch { expected: self.dimensions, found: vector.size() });
        }
        if self.vector_map.contains_key(&key) {
            return Err(VectorError::KeyAlreadyExists(key));
        }
        
        // If key is tombstoned, rebuild once and proceed
        if self.tombstones.remove(&key).is_some() {
            self.rebuild_tree();  // clears all tombstones
        }
        
        self.vector_map.insert(key.clone(), vector.clone());
        
        if let Some(ref mut root) = self.root {
            Self::insert_recursive_static(root, vector, key, 0, self.dimensions);
        } else {
            self.root = Some(Box::new(KDTreeNode::new(vector, key)));
        }
        Ok(())
    }

    /// Recursive insertion helper
    fn insert_recursive_static(node: &mut Box<KDTreeNode>, vector: Vector, key: String, depth: usize, dimensions: usize) {
        let split_dim = depth % dimensions;
        
        let comparison = vector[split_dim].partial_cmp(&node.vector[split_dim]).unwrap();
        
        match comparison {
            std::cmp::Ordering::Less => {
                if let Some(ref mut left) = node.left {
                    Self::insert_recursive_static(left, vector, key, depth + 1, dimensions);
                } else {
                    node.left = Some(Box::new(KDTreeNode::new(vector, key)));
                }
            }
            _ => {
                if let Some(ref mut right) = node.right {
                    Self::insert_recursive_static(right, vector, key, depth + 1, dimensions);
                } else {
                    node.right = Some(Box::new(KDTreeNode::new(vector, key)));
                }
            }
        }
    }

    /// Remove a vector from the KD-tree (adds to tombstones)
    pub fn remove(&mut self, key: &str) {
        if let Some((owned_key, vector)) = self.vector_map.remove_entry(key) {
            self.tombstones.insert(owned_key, vector);
            
            // Check if we need to rebuild due to tombstone balance
            if self.tombstones.len() > self.vector_map.len() {
                self.rebuild_tree();
            }
        }
    }

    /// Rebuild the entire tree from the current vector_map
    fn rebuild_tree(&mut self) {
        // Filter out tombstoned entries and collect non-tombstoned vectors
        let mut vectors: Vec<(String, Vector)> = Vec::new();
        
        for (key, vector) in self.vector_map.iter() {
            if !self.tombstones.contains_key(key) {
                vectors.push((key.clone(), vector.clone()));
            }
        }

        // Clear tombstones when rebuilding
        self.tombstones.clear();
        
        if vectors.is_empty() {
            self.root = None;
            return;
        }
        
        self.root = Some(Self::build_tree_recursive(&mut vectors, 0, self.dimensions));
    }

    /// Recursively build a balanced KD-tree
    fn build_tree_recursive(vectors: &mut [(String, Vector)], depth: usize, dimensions: usize) -> Box<KDTreeNode> {
        if vectors.len() == 1 {
            let (key, vector) = vectors[0].clone();
            return Box::new(KDTreeNode::new(vector, key));
        }
        
        let split_dim = depth % dimensions;
        
        // Sort by the split dimension
        vectors.sort_by(|a, b| a.1[split_dim].partial_cmp(&b.1[split_dim]).unwrap());
        
        let median_idx = vectors.len() / 2;
        let (key, vector) = vectors[median_idx].clone();
        
        let mut node = Box::new(KDTreeNode::new(vector, key));
        
        // Build left subtree
        if median_idx > 0 {
            let left_vectors = &mut vectors[..median_idx];
            node.left = Some(Self::build_tree_recursive(left_vectors, depth + 1, dimensions));
        }
        
        // Build right subtree
        if median_idx + 1 < vectors.len() {
            let right_vectors = &mut vectors[median_idx + 1..];
            node.right = Some(Self::build_tree_recursive(right_vectors, depth + 1, dimensions));
        }
        
        node
    }

    /// Find the nearest neighbor to a query vector
    pub fn nearest_neighbor(&self, query: &Vector) -> Result<Option<String>, VectorError> {
        self.nearest_neighbors(query, 1).map(|mut results| results.pop().map(|(key, _)| key))
    }

    /// Find k nearest neighbors to a query vector
    pub fn nearest_neighbors(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>, VectorError> {
        if query.size() != self.dimensions {
            return Err(VectorError::DimensionsMismatch { expected: self.dimensions, found: query.size() });
        }
        
        if self.root.is_none() || k == 0 {
            return Ok(Vec::new());
        }
        
        let mut heap = BinaryHeap::new();
        self.nearest_neighbors_recursive(self.root.as_ref().unwrap(), query, k, &mut heap, 0)?;
        
        Ok(heap.into_sorted_vec()
            .into_iter()
            .rev()
            .map(|Reverse((distance, key))| (key, distance.0))
            .collect())
    }

    /// Recursive k-nearest neighbors search helper
    fn nearest_neighbors_recursive(
        &self,
        node: &KDTreeNode,
        query: &Vector,
        k: usize,
        heap: &mut BinaryHeap<Reverse<(FloatOrd, String)>>,
        depth: usize,
    ) -> Result<(), VectorError> {
        let current_distance = self.distance_metric.distance(&node.vector, query)?;
        
        if !self.tombstones.contains_key(&node.key) {
            let entry = Reverse((FloatOrd(current_distance), node.key.clone()));
            if heap.len() < k {
                heap.push(entry);
            } else if FloatOrd(current_distance) < heap.peek().unwrap().0.0 {
                heap.pop();
                heap.push(entry);
            }
        }
        
        let split_dim = depth % self.dimensions;
        let query_val = query[split_dim];
        let node_val = node.vector[split_dim];
        
        let (first_child, second_child) = if query_val < node_val {
            (node.left.as_ref(), node.right.as_ref())
        } else {
            (node.right.as_ref(), node.left.as_ref())
        };
        
        if let Some(child) = first_child {
            self.nearest_neighbors_recursive(child, query, k, heap, depth + 1)?;
        }
        
        let split_distance = (query_val - node_val).abs();
        let worst_distance = heap.peek().map(|r| r.0.0 .0).unwrap_or(f32::INFINITY);
        
        if split_distance < worst_distance {
            if let Some(child) = second_child {
                self.nearest_neighbors_recursive(child, query, k, heap, depth + 1)?;
            }
        }
        Ok(())
    }

    /// Get a vector by key
    pub fn get_vector(&self, key: &str) -> Option<&Vector> {
        self.vector_map.get(key)
    }

    /// Get the number of vectors in the tree
    pub fn size(&self) -> usize {
        self.vector_map.len()
    }

    /// Get all vectors in the tree 
    pub fn get_all_vectors(&self) -> &HashMap<String, Vector> {
        &self.vector_map
    }

    /// Get the distance metric used by this KD-tree
    pub fn distance_metric(&self) -> Distance {
        self.distance_metric
    }

    /// Get the number of tombstones in the tree
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    /// Check if a key is marked as a tombstone
    pub fn is_tombstoned(&self, key: &str) -> bool {
        self.tombstones.contains_key(key)
    }
}

#[derive(PartialEq)]
struct FloatOrd(pub f32);

impl Eq for FloatOrd {}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdtree_basic_insertion() {
        let mut tree = KDTree::new(3, Distance::Euclidean);
        let vector = Vector::from_slice(&[1.0, 2.0, 3.0]);
        tree.insert(vector, "point1".to_string()).unwrap();
        assert_eq!(tree.size(), 1);
    }

    #[test]
    fn test_kdtree_nearest_neighbor_search() {
        let mut tree = KDTree::new(3, Distance::Euclidean);
        tree.insert(Vector::from_slice(&[1.0, 2.0, 3.0]), "point1".to_string()).unwrap();
        tree.insert(Vector::from_slice(&[4.0, 5.0, 6.0]), "point2".to_string()).unwrap();

        let query = Vector::from_slice(&[1.1, 2.1, 3.1]);
        let nearest = tree.nearest_neighbor(&query).unwrap().unwrap();
        assert_eq!(nearest, "point1");
    }

    #[test]
    fn test_kdtree_k_nearest_neighbors_search() {
        let mut tree = KDTree::new(3, Distance::Euclidean);
        tree.insert(Vector::from_slice(&[1.0, 2.0, 3.0]), "point1".to_string()).unwrap();
        tree.insert(Vector::from_slice(&[4.0, 5.0, 6.0]), "point2".to_string()).unwrap();
        tree.insert(Vector::from_slice(&[8.0, 9.0, 10.0]), "point3".to_string()).unwrap();
        
        let query = Vector::from_slice(&[1.1, 2.1, 3.1]);
        let nearest_2 = tree.nearest_neighbors(&query, 2).unwrap();
        assert_eq!(nearest_2.len(), 2);
        assert_eq!(nearest_2[0].0, "point1");
        assert_eq!(nearest_2[1].0, "point2");

        let query_far = Vector::from_slice(&[100.0, 100.0, 100.0]);
        let nearest_3 = tree.nearest_neighbors(&query_far, 3).unwrap();
        assert_eq!(nearest_3.len(), 3);
        
        for i in 1..nearest_3.len() {
            assert!(nearest_3[i-1].1 <= nearest_3[i].1, "Distances not in ascending order");
        }
    }

    #[test]
    fn test_kdtree_tombstone_and_rebuild_mechanism() {
        let mut tree = KDTree::new(3, Distance::Euclidean);
        tree.insert(Vector::from_slice(&[1.0, 2.0, 3.0]), "point1".to_string()).unwrap();
        tree.insert(Vector::from_slice(&[4.0, 5.0, 6.0]), "point2".to_string()).unwrap();

        tree.remove("point1");
        assert_eq!(tree.size(), 1);
        assert!(tree.is_tombstoned("point1"));

        let query = Vector::from_slice(&[1.1, 2.1, 3.1]);
        let nearest = tree.nearest_neighbor(&query).unwrap().unwrap();
        assert_eq!(nearest, "point2");

        tree.rebuild_tree();
        assert_eq!(tree.size(), 1);
        assert!(!tree.is_tombstoned("point1"));
    }

    #[test]
    fn test_kdtree_tombstone_reinsert_behavior() {
        let mut tree = KDTree::new(2, Distance::Euclidean);
        tree.insert(Vector::from_slice(&[1.0, 1.0]), "point1".to_string()).unwrap();
        tree.insert(Vector::from_slice(&[2.0, 2.0]), "point2".to_string()).unwrap();
        tree.remove("point1");
        assert!(tree.is_tombstoned("point1"));
        
        tree.insert(Vector::from_slice(&[2.0, 2.0]), "point1".to_string()).unwrap();
        
        assert!(!tree.is_tombstoned("point1"));
        assert_eq!(tree.size(), 2);
        assert_eq!(tree.get_vector("point1").unwrap().data(), &[2.0, 2.0]);
    }

    #[test]
    fn test_kdtree_automatic_rebuild_trigger() {
        let mut tree = KDTree::new(2, Distance::Euclidean);
        
        // Add 3 points
        tree.insert(Vector::from_slice(&[1.0, 1.0]), "point1".to_string()).unwrap();
        tree.insert(Vector::from_slice(&[2.0, 2.0]), "point2".to_string()).unwrap();
        tree.insert(Vector::from_slice(&[3.0, 3.0]), "point3".to_string()).unwrap();
        
        // Remove 2 points (tombstone count > vector count)
        tree.remove("point1");
        tree.remove("point2");
        
        // Should trigger rebuild and clear tombstones
        assert_eq!(tree.size(), 1);
        assert_eq!(tree.tombstone_count(), 0);
    }

    #[test]
    fn test_kdtree_dimension_mismatch_errors() {
        let mut tree = KDTree::new(3, Distance::Euclidean);
        
        let vector_2d = Vector::from_slice(&[1.0, 2.0]);
        let result = tree.insert(vector_2d.clone(), "key1".to_string());
        assert!(result.is_err());
        
        let result = tree.nearest_neighbor(&vector_2d);
        assert!(result.is_err());
    }

    #[test]
    fn test_kdtree_duplicate_key_prevention() {
        let mut tree = KDTree::new(3, Distance::Euclidean);
        tree.insert(Vector::from_slice(&[1.0, 2.0, 3.0]), "point1".to_string()).unwrap();
        let result = tree.insert(Vector::from_slice(&[4.0, 5.0, 6.0]), "point1".to_string());
        assert!(result.is_err());
    }
} 