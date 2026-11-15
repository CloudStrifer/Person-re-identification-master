"""
Adaptive Attention-Weighted Dynamic Feature Pooling (A²DFP)

Novel Contribution: Integrates CLIP backbone's attention mechanism
into feature selection and weighting for improved discriminative pooling.

Author: Research Implementation
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class AdaptiveAttentionWeighting(nn.Module):
    """
    Adaptive Attention Weighting Module

    Learns to weight features based on:
    1. Attention scores from backbone
    2. Feature importance (learned)
    3. Context information

    Innovation: First to explicitly model attention-based importance in DFP
    """

    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Attention importance learning
        self.attention_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        # Context-aware gating
        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

        # Temperature for attention softening
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, features, attention_scores=None):
        """
        Args:
            features: [batch_size, seq_len, embed_dim]
            attention_scores: [batch_size, num_heads, seq_len, seq_len] or None

        Returns:
            weighted_features: [batch_size, seq_len, embed_dim]
            attention_weights: [batch_size, seq_len]
        """
        batch_size, seq_len, embed_dim = features.shape

        # Learn feature importance
        importance = self.attention_fc(features)  # [batch, seq_len, 1]
        importance = importance.squeeze(-1)  # [batch, seq_len]

        # Aggregate attention scores if provided
        if attention_scores is not None:
            # Average across heads: [batch, seq_len, seq_len]
            attention_avg = attention_scores.mean(dim=1)
            # Sum attention received: [batch, seq_len]
            attention_weight = attention_avg.sum(dim=1) / (seq_len + 1e-8)
            # Normalize
            attention_weight = F.softmax(attention_weight / self.temperature, dim=-1)
        else:
            attention_weight = torch.ones(batch_size, seq_len, device=features.device) / seq_len

        # Context-aware gating
        context = self.context_gate(features)  # [batch, seq_len, 1]
        context = context.squeeze(-1)  # [batch, seq_len]

        # Combine: importance * attention_weight * context_gate
        combined_weight = importance * attention_weight * context
        combined_weight = F.softmax(combined_weight, dim=-1)

        # Apply weights
        weighted_features = features * combined_weight.unsqueeze(-1)

        return weighted_features, combined_weight


class ContrastiveFeaturePooling(nn.Module):
    """
    Contrastive Feature Pooling with Hard Mining

    Novel Contribution: Integrates hard-positive and hard-negative mining
    into DFP for improved discriminative learning

    Key Innovation:
    1. Hard-positive mining: Select most uncertain positive features
    2. Hard-negative mining: Select most confusing negative features
    3. Contrastive projection: Learn features that maximize margin
    """

    def __init__(self, input_dim=512, output_dim=2048, margin=0.1, temperature=0.07):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.margin = margin
        self.temperature = temperature

        # Main projection
        self.projection = nn.Linear(input_dim, output_dim)

        # Hard mining attention
        self.hard_mining_scorer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )

        # Contrastive loss temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def hard_positive_mining(self, features, pids):
        """
        Mine hard positive samples within same PID cluster

        Hard positives are: similar PIDs but different images/texts
        (most uncertain positives)

        Returns: hard_positive_mask [batch_size, batch_size]
        """
        batch_size = features.shape[0]

        # Cosine similarity
        features_norm = F.normalize(features, dim=-1)
        sim_matrix = torch.matmul(features_norm, features_norm.t())  # [B, B]

        # PID matching
        pid_matrix = (pids.unsqueeze(0) == pids.unsqueeze(1)).float()

        # Remove diagonal (same sample)
        pid_matrix.fill_diagonal_(0)

        # Find hardest positives: high similarity but still positive
        # We want samples with lowest similarity among positives
        sim_matrix = sim_matrix * pid_matrix
        hard_positive_mask = torch.zeros_like(sim_matrix)

        for i in range(batch_size):
            positives = torch.where(pid_matrix[i] > 0)[0]
            if len(positives) > 0:
                # Get hardest (most similar to negative distribution)
                sims = sim_matrix[i, positives]
                hardest_idx = positives[sims.argmin()]
                hard_positive_mask[i, hardest_idx] = 1.0

        return hard_positive_mask

    def hard_negative_mining(self, features, pids, k=2):
        """
        Mine hard negative samples across PIDs

        Hard negatives are: different PIDs but high similarity
        (most confusing negatives)

        Returns: hard_negative_mask [batch_size, batch_size]
        """
        batch_size = features.shape[0]

        # Cosine similarity
        features_norm = F.normalize(features, dim=-1)
        sim_matrix = torch.matmul(features_norm, features_norm.t())  # [B, B]

        # PID matching
        pid_matrix = (pids.unsqueeze(0) == pids.unsqueeze(1)).float()

        # Get negatives: different PIDs
        neg_mask = 1 - pid_matrix
        neg_mask.fill_diagonal_(0)

        # Find hardest negatives: highest similarity among negatives
        hard_negative_mask = torch.zeros_like(sim_matrix)

        for i in range(batch_size):
            negatives = torch.where(neg_mask[i] > 0)[0]
            if len(negatives) >= k:
                sims = sim_matrix[i, negatives]
                # Get top-k hardest negatives
                _, topk_indices = torch.topk(sims, min(k, len(negatives)))
                hard_negative_mask[i, negatives[topk_indices]] = 1.0
            elif len(negatives) > 0:
                hard_negative_mask[i, negatives] = 1.0

        return hard_negative_mask

    def contrastive_loss(self, features, pids):
        """
        Compute contrastive loss with hard mining

        Loss encourages:
        1. Hard positives to be close together
        2. Hard negatives to be far apart
        3. Margin maintained between positive and negative clusters
        """
        batch_size = features.shape[0]

        # Get hard samples
        hard_pos_mask = self.hard_positive_mining(features, pids)
        hard_neg_mask = self.hard_negative_mining(features, pids)

        # Normalize features
        features_norm = F.normalize(features, dim=-1)

        # Compute similarities
        sim_matrix = torch.matmul(features_norm, features_norm.t())
        sim_matrix = sim_matrix / self.temperature

        # Mask out self-similarity
        sim_matrix.fill_diagonal_(-1e9)

        # Contrastive loss computation
        pos_sim = (sim_matrix * hard_pos_mask).sum(dim=1) / (hard_pos_mask.sum(dim=1) + 1e-8)
        neg_sim = (sim_matrix * hard_neg_mask).sum(dim=1) / (hard_neg_mask.sum(dim=1) + 1e-8)

        # Loss: push negatives below positives with margin
        loss = F.relu(neg_sim - pos_sim + self.margin).mean()

        return loss

    def forward(self, features, pids=None, return_loss=True):
        """
        Args:
            features: [batch_size, seq_len, input_dim]
            pids: [batch_size] person IDs
            return_loss: whether to compute contrastive loss

        Returns:
            projected_features: [batch_size, output_dim]
            loss: contrastive loss (if return_loss=True)
        """
        batch_size = features.shape[0]

        # Max pooling over sequence dimension
        pooled = features.max(dim=1)[0]  # [batch_size, input_dim]

        # Project to output dimension
        projected = self.projection(pooled)  # [batch_size, output_dim]

        # Compute contrastive loss if needed
        loss = None
        if return_loss and pids is not None:
            loss = self.contrastive_loss(features, pids)

        return projected, loss


class A2DFP_VisualEmbedding(nn.Module):
    """
    A²DFP Visual Embedding Layer

    Combines:
    1. Adaptive Attention Weighting (A²)
    2. Contrastive Feature Pooling (C)

    For enhanced visual feature representation
    """

    def __init__(self, input_dim=512, embed_dim=2048, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Component 1: Adaptive Attention
        self.attention_weighting = AdaptiveAttentionWeighting(input_dim, num_heads)

        # Component 2: Contrastive Pooling
        self.contrastive_pooling = ContrastiveFeaturePooling(input_dim, embed_dim)

        # MLP for final transformation
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # Loss weighting
        self.contrastive_loss_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, features, pids, attention_scores=None):
        """
        Args:
            features: [batch_size, seq_len, input_dim] image patches
            pids: [batch_size] person IDs
            attention_scores: optional attention from backbone

        Returns:
            enhanced_features: [batch_size, embed_dim]
            losses: dict with loss components
        """
        losses = {}

        # Step 1: Adaptive Attention Weighting
        weighted_features, attn_weights = self.attention_weighting(features, attention_scores)

        # Step 2: Contrastive Feature Pooling
        projected_features, contrastive_loss = self.contrastive_pooling(
            weighted_features, pids, return_loss=True
        )

        if contrastive_loss is not None:
            losses['contrastive_loss'] = contrastive_loss * self.contrastive_loss_weight

        # Step 3: MLP transformation
        enhanced_features = self.mlp(projected_features)
        enhanced_features = F.normalize(enhanced_features, dim=-1)

        return enhanced_features, losses


class A2DFP_TextualEmbedding(nn.Module):
    """
    A²DFP Textual Embedding Layer

    Similar to visual but optimized for text tokens
    """

    def __init__(self, input_dim=512, embed_dim=2048, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Component 1: Adaptive Attention
        self.attention_weighting = AdaptiveAttentionWeighting(input_dim, num_heads)

        # Component 2: Contrastive Pooling
        self.contrastive_pooling = ContrastiveFeaturePooling(input_dim, embed_dim)

        # MLP for final transformation
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # Loss weighting
        self.contrastive_loss_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, features, pids, attention_scores=None):
        """
        Args:
            features: [batch_size, seq_len, input_dim] text tokens
            pids: [batch_size] person IDs
            attention_scores: optional attention from backbone

        Returns:
            enhanced_features: [batch_size, embed_dim]
            losses: dict with loss components
        """
        losses = {}

        # Step 1: Adaptive Attention Weighting
        weighted_features, attn_weights = self.attention_weighting(features, attention_scores)

        # Step 2: Contrastive Feature Pooling
        projected_features, contrastive_loss = self.contrastive_pooling(
            weighted_features, pids, return_loss=True
        )

        if contrastive_loss is not None:
            losses['contrastive_loss'] = contrastive_loss * self.contrastive_loss_weight

        # Step 3: MLP transformation
        enhanced_features = self.mlp(projected_features)
        enhanced_features = F.normalize(enhanced_features, dim=-1)

        return enhanced_features, losses


if __name__ == "__main__":
    # Test A²DFP
    batch_size = 32
    seq_len = 50
    embed_dim = 512

    # Create dummy data
    features = torch.randn(batch_size, seq_len, embed_dim).cuda()
    pids = torch.randint(0, 10, (batch_size,)).cuda()

    # Create model
    model = A2DFP_VisualEmbedding(input_dim=embed_dim, embed_dim=2048).cuda()

    # Forward pass
    enhanced_features, losses = model(features, pids)

    print(f"Input shape: {features.shape}")
    print(f"Output shape: {enhanced_features.shape}")
    print(f"Losses: {losses}")
    print("A²DFP module test successful!")