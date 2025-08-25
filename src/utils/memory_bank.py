import torch
import numpy as np
from sklearn.cluster import KMeans

class MemoryBank:
    """
    Memory Bank implementation for storing and retrieving features
    """
    def __init__(self, capacity=1000, feature_dim=512, device='cpu'):
        """
        Initialize memory bank
        
        Args:
            capacity (int): Memory bank capacity
            feature_dim (int): Feature dimension
            device (str): Device to store features
        """
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.device = device
        self.features = torch.zeros(capacity, feature_dim).to(device)
        self.labels = []
        self.ptr = 0
        self.size = 0
        
    def add(self, features, labels):
        """
        Add features and labels to memory bank
        
        Args:
            features (torch.Tensor): Features to add [batch_size, feature_dim]
            labels (list): Corresponding labels
        """
        # Ensure features are on the correct device
        features = features.to(self.device)
        
        batch_size = features.size(0)
        if self.ptr + batch_size > self.capacity:
            remaining = self.capacity - self.ptr
            self.features[self.ptr:] = features[:remaining]
            self.labels.extend(labels[:remaining])
            self.features[:batch_size - remaining] = features[remaining:]
            self.labels.extend(labels[remaining:])
            self.ptr = batch_size - remaining
        else:
            self.features[self.ptr:self.ptr + batch_size] = features
            self.labels.extend(labels)
            self.ptr = (self.ptr + batch_size) % self.capacity
            
        self.size = min(self.size + batch_size, self.capacity)
        
    def get_nearest_neighbors(self, query_features, k=5):
        """
        Get nearest neighbor features
        
        Args:
            query_features (torch.Tensor): Query features [query_num, feature_dim]
            k (int): Number of nearest neighbors to return
            
        Returns:
            tuple: (similarity scores, corresponding labels)
        """
        if self.size == 0:
            return None, None
            
        # Ensure query features are on the correct device
        query_features = query_features.to(self.device)
            
        memory_features = self.features[:self.size]
        
        # Normalize features
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)
        memory_features = memory_features / memory_features.norm(dim=-1, keepdim=True)
        
        # Calculate cosine similarity
        similarities = torch.mm(query_features, memory_features.t())
        topk_similarities, topk_indices = torch.topk(similarities, min(k, self.size), dim=1)
        
        # Get corresponding labels
        neighbor_labels = []
        for i in range(query_features.size(0)):
            labels_for_query = [self.labels[idx.item()] for idx in topk_indices[i]]
            neighbor_labels.append(labels_for_query)
        
        return topk_similarities, neighbor_labels
        
    def cluster_features(self, n_clusters=10):
        """
        Cluster features in memory bank
        
        Args:
            n_clusters (int): Number of clusters
            
        Returns:
            tuple: (cluster centers, cluster labels) or None
        """
        if self.size < n_clusters:
            return None
            
        # Move features to CPU for clustering
        memory_features = self.features[:self.size].cpu().numpy()
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(memory_features)
        cluster_labels = kmeans.labels_
        
        # Calculate cluster centers and representative labels
        cluster_centers = []
        cluster_label_names = []
        
        for i in range(n_clusters):
            indices = np.where(cluster_labels == i)[0]
            if len(indices) > 0:
                center = np.mean(memory_features[indices], axis=0)
                
                # Get most common label in cluster
                cluster_labels_batch = [self.labels[idx] for idx in indices]
                unique_labels, counts = np.unique(cluster_labels_batch, return_counts=True)
                most_common_label = unique_labels[np.argmax(counts)]
                
                cluster_centers.append(center)
                cluster_label_names.append(most_common_label)
                
        return np.array(cluster_centers), cluster_label_names
        
    def get_size(self):
        """
        Get current memory bank size
        """
        return self.size
        
    def clear(self):
        """
        Clear memory bank
        """
        self.features = torch.zeros(self.capacity, self.feature_dim).to(self.device)
        self.labels = []
        self.ptr = 0
        self.size = 0
        
    def get_feature_statistics(self):
        """
        Get feature statistics in memory bank
        """
        if self.size == 0:
            return {}
            
        label_counts = {}
        for label in self.labels[:self.size]:
            label_counts[label] = label_counts.get(label, 0) + 1
            
        return {
            'total_features': self.size,
            'label_distribution': label_counts
        }