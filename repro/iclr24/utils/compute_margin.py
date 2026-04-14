import torch


def margins_batch(model, device, data, target):
    """
    Compute the margins for a batch of data.

    Args:
        model (torch.nn.Module): The neural network model.
        device (torch.device): The device on which the computation is performed.
        data (torch.Tensor): Input data for the batch.
        target (torch.Tensor): Target labels for the batch.

    Returns:
        torch.Tensor: Margins for each sample in the batch.
    """
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = model(data)
        # get the output for the correct class
        output_correct_class = output[range(len(target)), target]
        # set the correct class to -inf so that it is not considered
        output[range(len(target)), target] = float("-inf")
        # get the max of the rest of the classes
        output_max = output.max(1)[0]
        margins = output_correct_class - output_max
    return margins


def get_all_margins(model, dataloader, device):
    """
    Compute margins for all samples in a dataloader.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): The device on which the computation is performed.

    Returns:
        torch.Tensor: Margins for all samples in the dataloader.
    """
    margins_chunks = []
    model.eval()
    for i, (data, target) in enumerate(dataloader):
        if i % 100 == 0 or i == 1:
            if i == 1:
                print(
                    f"batch {i}/{len(dataloader)}, will now print every 100 batches"
                )
            else:
                print(f"batch: {i}/{len(dataloader)}")
        # compute the margin
        margins_chunks.append(margins_batch(model, device, data, target).cpu())
    if not margins_chunks:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(margins_chunks, dim=0)


def get_quantile_margins(model, dataloader, device):
    """
    Get p-quantiles of the margin distribution over the dataloader
    for p=(1-t)error_train_top1 + t for t=0, 1/3, 1/2, 2/3, 1
    where error_train_top1 = 1 - train_top1.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): The device on which the computation is performed.

    Returns:
        list: p-quantiles of the margin distribution.
    """
    # first get all the margins
    margins = get_all_margins(model, dataloader, device)
    # then get the quantiles
    quantile_margins = []
    number_negative_margins = (margins < 0).sum()
    sorted_margins = torch.sort(margins).values

    # compute the quantiles by taking the margin
    # at the index corresponding to the quantiles
    # (1-t) * number_negative_margins + t * len(margins)
    for i in [
        number_negative_margins,
        int((2 / 3) * number_negative_margins + (len(margins) - 1) / 3),
        int(0.5 * number_negative_margins + 0.5 * (len(margins) - 1)),
        int((1 / 3) * number_negative_margins + 2 * (len(margins) - 1) / 3),
        len(margins) - 1,
    ]:
        quantile_margins.append(sorted_margins[i].item())

    return quantile_margins
