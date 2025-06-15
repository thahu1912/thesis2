import torch
from .custom_miners import BaseTupleMiner

class CacheMiner(BaseTupleMiner):
    def __init__(self, cache_size=5000, update_freq=1000, distance="euclidean", **kwargs):
        super().__init__(**kwargs)
        self.cache_size = cache_size
        self.update_freq = update_freq
        self.distance = distance
        self.cache = None
        self.cache_labels = None
        self.iter_count = 0
        
    def update_cache(self, embeddings, labels):
        """Update the cache with new embeddings"""
        if not isinstance(embeddings, torch.Tensor) or not isinstance(labels, torch.Tensor):
            return
            
        if self.cache is None:
            self.cache = embeddings.detach()
            self.cache_labels = labels.detach()
        else:
            # Keep only the most recent cache_size embeddings
            assert isinstance(self.cache, torch.Tensor) and isinstance(self.cache_labels, torch.Tensor)
            new_cache = torch.cat((self.cache, embeddings.detach()))
            new_labels = torch.cat((self.cache_labels, labels.detach()))
            self.cache = new_cache[-self.cache_size:]
            self.cache_labels = new_labels[-self.cache_size:]
    
    def mine(self, embeddings, labels, ref_emb=None, ref_labels=None):
        """
        Mine hard triplets using the cache
        Args:
            embeddings: Current batch embeddings
            labels: Current batch labels
            ref_emb: Reference embeddings (cache)
            ref_labels: Reference labels (cache)
        """
        self.iter_count += 1
        
        # Update cache if needed
        if self.iter_count % self.update_freq == 0:
            self.update_cache(embeddings, labels)
        
        # Use cache as reference if not provided
        if ref_emb is None:
            ref_emb = self.cache
            ref_labels = self.cache_labels
        
        if ref_emb is None or ref_labels is None:
            return None, None, None
        
        # Calculate distances between current batch and cache
        if self.distance == "euclidean":
            dist_mat = torch.cdist(embeddings, ref_emb)
        else:
            raise NotImplementedError(f"Distance {self.distance} not implemented")
        
        # Find hard negatives (closest negative samples)
        anchor_idx = []
        positive_idx = []
        negative_idx = []
        
        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            # Get distances to all samples in cache
            dists = dist_mat[i]
            
            # Find closest negative
            neg_mask = ref_labels != label
            if neg_mask.any():
                hard_neg_idx = torch.argmin(dists[neg_mask])
                neg_idx = torch.where(neg_mask)[0][hard_neg_idx]
                
                # Find random positive
                pos_mask = ref_labels == label
                if pos_mask.any():
                    pos_idx = torch.where(pos_mask)[0][torch.randint(0, pos_mask.sum(), (1,))]
                    anchor_idx.append(i)
                    positive_idx.append(pos_idx.item())
                    negative_idx.append(neg_idx.item())
        
        if not anchor_idx:
            return None, None, None
            
        return (
            torch.tensor(anchor_idx),
            torch.tensor(positive_idx),
            torch.tensor(negative_idx)
        ) 