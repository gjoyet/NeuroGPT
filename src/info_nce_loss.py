import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from einops import rearrange
import random
from collections import deque


class InfoNCELoss(nn.Module):
    """
    Compute InfoNCE loss.

    anchor: Tensor of shape (batch_size, embed_dim)
    label: Tensor of shape (batch_size,)
    """
    def __init__(self, num_subjects, embedding_dim, queue_size):
        super(InfoNCELoss, self).__init__()
        self.memory = self.SubjectMemoryBank(num_subjects=num_subjects,
                                             embedding_dim=embedding_dim,
                                             queue_size=queue_size)

    def forward(self, anchor, label, temperature=0.07):
        batch_size = anchor.shape[0]

        positive = torch.stack([self.memory.sample_positive(sid) for sid in label], dim=0)  # (batch_size, embed_dim)
        negatives = torch.stack([self.memory.sample_negatives(sid, 10) for sid in label],
                                dim=0)  # (batch_size, num_negatives, embed_dim)

        # Ensure tensors are contiguous
        anchor = anchor.contiguous()
        positive = positive.contiguous()
        negatives = negatives.contiguous()

        # Compute similarity scores
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        pos_sim = torch.matmul(anchor, positive.T).diag()  # (batch_size, batch_size)
        neg_sim = torch.matmul(anchor.unsqueeze(1), negatives.permute(0, 2, 1))  # (batch_size, 1, num_negatives)

        # Reshape if necessary
        neg_sim = neg_sim.contiguous().view(batch_size, -1)  # Ensure correct shape

        # Concatenate positive with negative similarity scores. Cross-entropy loss with label set to 0
        # (i.e. the position of the positive similarity score) is equivalent to the InfoNCE loss.
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temperature  # (batch_size, 1 + num_negatives)

        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    class SubjectMemoryBank:
        # For now, num_subjects and embedding_dim are hard-coded. Not nice, but did not find a good way to pass the
        # corresponding arguments down from classes calling this one.
        def __init__(self, num_subjects, embedding_dim, queue_size):
            """
            Memory bank storing past embeddings for each subject.

            Args:
            - num_subjects (int): Total number of subjects.
            - embedding_dim (int): Dimensionality of embedding vectors.
            - queue_size (int): Maximum number of embeddings per subject.
            """
            self.num_subjects = num_subjects
            self.embedding_dim = embedding_dim
            self.queue_size = queue_size
            self.queues = {i: deque(maxlen=queue_size) for i in range(num_subjects)}

        def add_embeddings(self, embeddings, subject_labels):
            """
            Adds new embeddings to the correct subject queues.

            Args:
            - embeddings (Tensor): Shape (batch_size, embedding_dim), new embeddings.
            - subject_labels (Tensor): Shape (batch_size,), corresponding subject IDs.
            """
            for emb, subject in zip(embeddings, subject_labels):
                self.queues[subject.item()].append(emb.detach())  # Store detached embeddings to prevent graph retention

        def sample_positive(self, subject):
            """
            Samples a positive example from the same subject queue.

            Args:
            - subject (int): Subject ID.

            Returns:
            - Tensor: A random positive embedding, or a zero tensor if the queue is empty.
            """
            queue = self.queues[subject.item()]
            if len(queue) == 0:
                return torch.zeros(self.embedding_dim)  # Return zero tensor if empty
            return random.choice(queue)  # Random positive sample

        def sample_negatives(self, subject, num_negatives):
            """
            Samples negative examples from different subject queues.

            Args:
            - subject (int): Subject ID to exclude.
            - num_negatives (int): Number of negative samples to return.

            Returns:
            - Tensor: Shape (num_negatives, embedding_dim), sampled negatives.
            """
            negative_samples = []
            available_subjects = [s for s in self.queues.keys() if s != subject.item() and len(self.queues[s]) > 0]

            while len(negative_samples) < num_negatives:
                if not available_subjects:
                    negative_samples.append(torch.zeros(self.embedding_dim))  # Return zeros if no negatives available
                else:
                    neg_subject = random.choice(available_subjects)
                    negative_samples.append(random.choice(self.queues[neg_subject]))

            return torch.stack(negative_samples)  # Convert list to tensor