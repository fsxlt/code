import torch


def seqcls_collocate(batched, device=None):
    if device is None:
        device = torch.cuda.current_device()
    uids, input_ids, golds, attention_mask, token_type_ids = batched
    batched = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "token_type_ids": token_type_ids.to(device),
    }
    return (batched, golds.to(device), uids, None)


def tagging_batch_to_device(batched, device=None):
    if device is None:
        device = torch.cuda.current_device()
    uides, input_idses, attention_maskes, tags_ides, if_tgtes = batched
    batched = {
        "input_ids": input_idses.to(device),
        "attention_mask": attention_maskes.to(device),
        "if_tgts": if_tgtes.to(device),
    }
    golds = tags_ides[if_tgtes].to(device)
    return (
        batched,
        golds,
        uides,
        tags_ides,
    )


task2collocate_fn = {
    "marc": seqcls_collocate,
    "mldoc": seqcls_collocate,
    "conll2003": tagging_batch_to_device,
    "argustan": seqcls_collocate,
    "pawsx": seqcls_collocate,
    "xnli": seqcls_collocate,
    "panx": tagging_batch_to_device,
    "udpos": tagging_batch_to_device,
}
