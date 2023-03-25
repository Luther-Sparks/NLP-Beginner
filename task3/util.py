import torch
import torch.nn as nn
import torch.nn.functional as F

def sort_by_seq_len(sequence_batch : torch.Tensor, sequences_lengths : torch.Tensor, descending=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort a batch of padded variable length sequences by their length.

    Args:
        sequence_batch (torch.Tensor(batch, max_sequence_length))
        sequences_lengths (torch.Tensor(batch))
        descending (bool, optional): Defaults to True.

    Returns:
        sorted_batch: A tensor containing the input batch reordered by
            sequences lengths.
        sorted_seq_lens: A tensor containing the sorted lengths of the
            sequences in the input batch.
        restoration_idx: A tensor containing the indices that can be used to
            restore the order of the sequences in 'sorted_batch' so that it
            matches the input batch.
    """
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = sequence_batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths))
    _, reversed_mapping_idx = sorting_index.sort(0, descending=False)
    restoration_idx = idx_range.index_select(0, reversed_mapping_idx)
    
    return sorted_batch, sorted_seq_lens, restoration_idx

def masked_softmax(tensor : torch.Tensor, mask : torch.Tensor):
    """
    Apply a masked softmax on the last dimension of a tensor

    Args:
        torch (torch.Tensor(batch, *, sequence_length)): The tensor on which the softmax function will be applied along the last dimension.
        mask (torch.Tensor(batch, *, sequence_length)): A mask of the same size as the tensor with 0 in the postions of the values which will be masked and 1 elsewhere.
    """
    reshaped_tensor = tensor.view(-1, tensor.shape[-1])
    while mask.dim() < tensor.dim():
        mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.shape[-1])
    
    result = F.softmax(reshaped_tensor * reshaped_mask, -1)
    result = result * reshaped_mask
    # epsilon := 1e-13, add epsilon to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(tensor.shape)

def weighted_sum(tensor : torch.Tensor, weight : torch.Tensor, mask : torch.Tensor):
    """calculate the weighted sum of tensor and mask the result with `mask`
    """
    weighted = weight.matmul(tensor)
    while mask.dim() < weighted.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted).contiguous().float()
    
    return weighted * mask

def get_mask(sequences_batch : torch.Tensor, sequence_length : torch.Tensor):
    """get the mask for a batch of padded variable length sequences.

    Args:
        sequences_batch (torch.Tensor(batch, sequence)): A batch of padded variable length sequences containing word indices. 
        sequence_length (torch.Tensor(batch)): A tensor containing the lengths of the sequences in `sequence_batch`.
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequence_length)
    mask = torch.ones(batch_size, max_length, dtype=float)
    mask[sequences_batch[:,:max_length] == 0] = 0.0
    return mask
