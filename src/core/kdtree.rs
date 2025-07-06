use crate::core::{Vector, Distance};
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use std::cmp::Reverse;

struct KDTreeNode {
    vector: Vector,
    key: String,
    left: Option<Box<KDTreeNode>>,
    right: Option<Box<KDTreeNode>>,
    split_dimension: usize,
}

impl KDTreeNode {
    fn new(vector: Vector, key: String) -> Self {
        Self {
            vector,
            key,
            left: None,
            right: None,
            split_dimension: 0,
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
    pub fn insert(&mut self, vector: Vector, key: String) {
        assert_eq!(vector.size(), self.dimensions, "Vector dimensions must match KD-tree dimensions");
        assert!(!self.vector_map.contains_key(&key), "Key '{}' already exists in tree", key);
        
        // Check if this key was previously tombstoned
        if self.tombstones.contains_key(&key) {
            // Remove from tombstones and rebuild tree (which clears all tombstones)
            self.tombstones.remove(&key);
            self.rebuild_tree();
            // Now insert the new node as a fresh insert
            self.insert(vector, key);
            return;
        }
        
        self.vector_map.insert(key.clone(), vector.clone());
        
        if let Some(ref mut root) = self.root {
            Self::insert_recursive_static(root, vector, key, 0, self.dimensions);
        } else {
            let mut node = Box::new(KDTreeNode::new(vector, key));
            node.split_dimension = 0;
            self.root = Some(node);
        }
    }

    /// Recursive insertion helper
    fn insert_recursive_static(node: &mut Box<KDTreeNode>, vector: Vector, key: String, depth: usize, dimensions: usize) {
        let split_dim = depth % dimensions;
        node.split_dimension = split_dim;
        
        let comparison = vector[split_dim].partial_cmp(&node.vector[split_dim]).unwrap();
        
        match comparison {
            std::cmp::Ordering::Less => {
                if let Some(ref mut left) = node.left {
                    Self::insert_recursive_static(left, vector, key, depth + 1, dimensions);
                } else {
                    let mut new_node = Box::new(KDTreeNode::new(vector, key));
                    new_node.split_dimension = (depth + 1) % dimensions;
                    node.left = Some(new_node);
                }
            }
            _ => {
                if let Some(ref mut right) = node.right {
                    Self::insert_recursive_static(right, vector, key, depth + 1, dimensions);
                } else {
                    let mut new_node = Box::new(KDTreeNode::new(vector, key));
                    new_node.split_dimension = (depth + 1) % dimensions;
                    node.right = Some(new_node);
                }
            }
        }
    }

    /// Remove a vector from the KD-tree (adds to tombstones)
    pub fn remove(&mut self, key: &str) {
        if let Some(vector) = self.vector_map.remove(key) {
            self.tombstones.insert(key.to_string(), vector);
            
            // Check if we need to rebuild due to tombstone balance
            if self.tombstones.len() >= self.vector_map.len() {
                self.rebuild_tree();
            }
        }
    }

    /// Rebuild the entire tree from the current vector_map
    fn rebuild_tree(&mut self) {
        // Only include non-tombstoned entries in the rebuilt tree
        let mut vectors: Vec<(String, Vector)> = self.vector_map
            .iter()
            .filter(|(k, _)| !self.tombstones.contains_key(*k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

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
            let mut node = Box::new(KDTreeNode::new(vector, key));
            node.split_dimension = depth % dimensions;
            return node;
        }
        
        let split_dim = depth % dimensions;
        
        // Sort by the split dimension
        vectors.sort_by(|a, b| a.1[split_dim].partial_cmp(&b.1[split_dim]).unwrap());
        
        let median_idx = vectors.len() / 2;
        let (key, vector) = vectors[median_idx].clone();
        
        let mut node = Box::new(KDTreeNode::new(vector, key));
        node.split_dimension = split_dim;
        
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
    pub fn nearest_neighbor(&self, query: &Vector) -> Option<String> {
        self.nearest_neighbors(query, 1).into_iter().next().map(|(key, _)| key)
    }

    /// Find k nearest neighbors to a query vector
    pub fn nearest_neighbors(&self, query: &Vector, k: usize) -> Vec<(String, f32)> {
        assert_eq!(query.size(), self.dimensions, "Query vector dimensions must match KD-tree dimensions");
        
        if self.root.is_none() || k == 0 {
            return Vec::new();
        }
        
        let mut heap = BinaryHeap::new();
        self.nearest_neighbors_recursive(self.root.as_ref().unwrap(), query, k, &mut heap, 0);
        
        heap.into_sorted_vec()
            .into_iter()
            .rev()
            .map(|Reverse((distance, key))| (key, distance.0))
            .collect()
    }

    /// Recursive k-nearest neighbors search helper
    fn nearest_neighbors_recursive(
        &self,
        node: &KDTreeNode,
        query: &Vector,
        k: usize,
        heap: &mut BinaryHeap<Reverse<(FloatOrd, String)>>,
        depth: usize,
    ) {
        let current_distance = self.distance_metric.distance(&node.vector, query);
        
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
            self.nearest_neighbors_recursive(child, query, k, heap, depth + 1);
        }
        
        let split_distance = (query_val - node_val).abs();
        let worst_distance = heap.peek().map(|r| r.0.0 .0).unwrap_or(f32::INFINITY);
        
        if split_distance < worst_distance {
            if let Some(child) = second_child {
                self.nearest_neighbors_recursive(child, query, k, heap, depth + 1);
            }
        }
    }

    /// Get a vector by key
    pub fn get_vector(&self, key: &str) -> Option<&Vector> {
        self.vector_map.get(key)
    }

    /// Get the number of vectors in the tree
    pub fn size(&self) -> usize {
        self.vector_map.len()
    }

    /// Get the number of tombstoned vectors
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    /// Check if a key is tombstoned
    pub fn is_tombstoned(&self, key: &str) -> bool {
        self.tombstones.contains_key(key)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
struct FloatOrd(pub f32);

impl Eq for FloatOrd {}
impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kd_tree_insertion() {
        let mut tree = KDTree::new(2, Distance::Euclidean);
        
        tree.insert(Vector::from_slice(&[1.0, 2.0]), "point1".to_string());
        tree.insert(Vector::from_slice(&[3.0, 4.0]), "point2".to_string());
        tree.insert(Vector::from_slice(&[5.0, 6.0]), "point3".to_string());
        
        assert_eq!(tree.size(), 3);
    }

    #[test]
    fn test_kd_tree_nearest_neighbor() {
        let mut tree = KDTree::new(2, Distance::Euclidean);
        
        // Add points in a grid pattern
        tree.insert(Vector::from_slice(&[0.0, 0.0]), "origin".to_string());
        tree.insert(Vector::from_slice(&[1.0, 0.0]), "right".to_string());
        tree.insert(Vector::from_slice(&[0.0, 1.0]), "up".to_string());
        tree.insert(Vector::from_slice(&[1.0, 1.0]), "diagonal".to_string());
        tree.insert(Vector::from_slice(&[2.0, 0.0]), "far_right".to_string());
        tree.insert(Vector::from_slice(&[0.0, 2.0]), "far_up".to_string());
        
        let query = Vector::from_slice(&[0.5, 0.5]);
        
        // Test k=1 (should be origin)
        let nearest = tree.nearest_neighbors(&query, 1);
        assert_eq!(nearest.len(), 1);
        assert_eq!(nearest[0].0, "origin");
        
        // Test k=3 (should be origin, right, up in some order)
        let nearest_3 = tree.nearest_neighbors(&query, 3);
        assert_eq!(nearest_3.len(), 3);
        
        // Verify all returned points are among the expected closest ones
        let expected_keys = ["origin", "right", "up"];
        for (key, _) in &nearest_3 {
            assert!(expected_keys.contains(&key.as_str()), "Unexpected key: {}", key);
        }
        
        // Verify distances are in ascending order
        for i in 1..nearest_3.len() {
            assert!(nearest_3[i-1].1 <= nearest_3[i].1, "Distances not in ascending order");
        }
    }

    #[test]
    fn test_kd_tree_k_nearest_neighbors() {
        let mut tree = KDTree::new(2, Distance::Euclidean);
        
        tree.insert(Vector::from_slice(&[1.0, 1.0]), "point1".to_string());
        tree.insert(Vector::from_slice(&[3.0, 3.0]), "point2".to_string());
        tree.insert(Vector::from_slice(&[5.0, 5.0]), "point3".to_string());
        
        let query = Vector::from_slice(&[2.0, 2.0]);
        let nearest = tree.nearest_neighbors(&query, 2);
        
        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0].0, "point1");
        assert_eq!(nearest[1].0, "point2");
    }

    #[test]
    fn test_kd_tree_removal_and_rebuild() {
        let mut tree = KDTree::new(2, Distance::Euclidean);
        
        tree.insert(Vector::from_slice(&[1.0, 1.0]), "point1".to_string());
        tree.insert(Vector::from_slice(&[3.0, 3.0]), "point2".to_string());
        tree.insert(Vector::from_slice(&[5.0, 5.0]), "point3".to_string());
        
        assert_eq!(tree.size(), 3);
        assert_eq!(tree.tombstone_count(), 0);
        
        // Remove a point
        tree.remove("point1");
        assert_eq!(tree.size(), 2);
        assert_eq!(tree.tombstone_count(), 1);
        assert!(tree.is_tombstoned("point1"));
        
        // Query should not return the removed point
        let query = Vector::from_slice(&[1.5, 1.5]);
        let nearest = tree.nearest_neighbor(&query);
        assert_eq!(nearest, Some("point2".to_string()));
    }

    #[test]
    fn test_kd_tree_reinsert_tombstoned() {
        let mut tree = KDTree::new(2, Distance::Euclidean);
        
        tree.insert(Vector::from_slice(&[1.0, 1.0]), "point1".to_string());
        tree.insert(Vector::from_slice(&[3.0, 3.0]), "point2".to_string());
        
        // Remove point1 (this triggers a rebuild and clears tombstones)
        tree.remove("point1");
        println!("After remove: tombstone_count = {}", tree.tombstone_count());
        assert_eq!(tree.tombstone_count(), 0);
        
        // Reinsert the same key - should work as a fresh insert
        tree.insert(Vector::from_slice(&[1.0, 1.0]), "point1".to_string());
        println!("After reinsert: tombstone_count = {}", tree.tombstone_count());
        assert_eq!(tree.tombstone_count(), 0);
        assert_eq!(tree.size(), 2);
        
        // Should be able to find the point again
        let query = Vector::from_slice(&[1.5, 1.5]);
        let nearest = tree.nearest_neighbor(&query);
        assert_eq!(nearest, Some("point1".to_string()));
    }

    #[test]
    fn test_kd_tree_reinsert_tombstoned_with_three_points() {
        let mut tree = KDTree::new(2, Distance::Euclidean);
        
        tree.insert(Vector::from_slice(&[1.0, 1.0]), "point1".to_string());
        tree.insert(Vector::from_slice(&[3.0, 3.0]), "point2".to_string());
        tree.insert(Vector::from_slice(&[5.0, 5.0]), "point3".to_string());
        
        // Remove point1 (should NOT trigger a rebuild yet)
        tree.remove("point1");
        println!("After remove (3 points): tombstone_count = {}", tree.tombstone_count());
        assert_eq!(tree.tombstone_count(), 1);
        assert!(tree.is_tombstoned("point1"));
        
        // Reinsert the same key - should trigger a rebuild and clear tombstones
        tree.insert(Vector::from_slice(&[1.0, 1.0]), "point1".to_string());
        println!("After reinsert (3 points): tombstone_count = {}", tree.tombstone_count());
        assert_eq!(tree.tombstone_count(), 0);
        assert_eq!(tree.size(), 3);
        
        // Should be able to find the point again
        let query = Vector::from_slice(&[1.5, 1.5]);
        let nearest = tree.nearest_neighbor(&query);
        assert_eq!(nearest, Some("point1".to_string()));
    }

    #[test]
    fn test_kd_tree_automatic_rebuild() {
        let mut tree = KDTree::new(2, Distance::Euclidean);
        
        // Add 3 points
        tree.insert(Vector::from_slice(&[1.0, 1.0]), "point1".to_string());
        tree.insert(Vector::from_slice(&[2.0, 2.0]), "point2".to_string());
        tree.insert(Vector::from_slice(&[3.0, 3.0]), "point3".to_string());
        
        // Remove 3 points (tombstone count >= vector count)
        tree.remove("point1");
        tree.remove("point2");
        
        // Should trigger rebuild and clear tombstones
        assert_eq!(tree.size(), 1);
        assert_eq!(tree.tombstone_count(), 0);
    }

    #[test]
    #[should_panic(expected = "Vector dimensions must match KD-tree dimensions")]
    fn test_kd_tree_dimension_mismatch() {
        let mut tree = KDTree::new(2, Distance::Euclidean);
        tree.insert(Vector::from_slice(&[1.0, 2.0, 3.0]), "point1".to_string());
    }

    #[test]
    fn test_different_distance_metrics() {
        let mut tree_euclidean = KDTree::new(2, Distance::Euclidean);
        let mut tree_manhattan = KDTree::new(2, Distance::Manhattan);
        
        tree_euclidean.insert(Vector::from_slice(&[0.0, 0.0]), "origin".to_string());
        tree_manhattan.insert(Vector::from_slice(&[0.0, 0.0]), "origin".to_string());
        
        let query = Vector::from_slice(&[3.0, 4.0]);
        
        // Both should find the origin as nearest neighbor
        assert_eq!(tree_euclidean.nearest_neighbor(&query), Some("origin".to_string()));
        assert_eq!(tree_manhattan.nearest_neighbor(&query), Some("origin".to_string()));
    }

    #[test]
    #[should_panic(expected = "Key 'point1' already exists in tree")]
    fn test_duplicate_key_insertion() {
        let mut tree = KDTree::new(2, Distance::Euclidean);
        tree.insert(Vector::from_slice(&[1.0, 1.0]), "point1".to_string());
        tree.insert(Vector::from_slice(&[1.0, 1.0]), "point1".to_string());
    }
} 