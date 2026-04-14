import torch
import torch.nn as nn
from torch.nn.utils import prune


def resume_for_imp(
    model,
    optimizer,
    scheduler,
    imp_iter,
    exp_dir,
    gpu,
    best=False,
    from_epoch_5=False,
):
    if best and from_epoch_5:
        raise ValueError(
            "You got to choose between resuming at best val top 1 (best=True) OR at epoch 5 (from_epoch_5=True)"
        )
    if best:
        imp_iter_to_load = imp_iter - 1
        if imp_iter_to_load > 0:
            resume_path = (
                exp_dir / f"imp_iter{imp_iter_to_load}/model_best.pth.tar"
            )
        else:
            resume_path = exp_dir / "model_best.pth.tar"
    elif from_epoch_5:
        resume_path = exp_dir / "epoch_5.pth.tar"
    else:
        raise ValueError("resume_for_imp called without best or from_epoch_5")
    # print("=> Loading checkpoint  '{}'".format(resume_path))
    if gpu is None:
        checkpoint = torch.load(resume_path)
    elif torch.cuda.is_available():
        # Map models to be loaded to specified single gpu.
        loc = "cuda:{}".format(gpu)
        checkpoint = torch.load(resume_path, map_location=loc)

    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if best:
        # print(model.state_dict().keys())
        # print(checkpoint["state_dict"].keys())
        model.load_state_dict(checkpoint["state_dict"])
    elif from_epoch_5:
        # in this case, the input model is pruned while epoch 5 is not
        # so we must put the .weight attributes of epoch 5 into
        # the .weight_orig attributes of the pruned layers
        # Indeed, the conv2d and linear layers are pruned so they are
        # re-parameterized with weight_orig (and weight_mask)
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                m.weight_orig.data = checkpoint["state_dict"][n + ".weight"]
                if m.bias is not None:
                    m.bias.data = checkpoint["state_dict"][n + ".bias"]
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = checkpoint["state_dict"][n + ".weight"]
                m.bias.data = checkpoint["state_dict"][n + ".bias"]
                m.running_mean.data = checkpoint["state_dict"][
                    n + ".running_mean"
                ]
                m.running_var.data = checkpoint["state_dict"][
                    n + ".running_var"
                ]
            elif isinstance(m, nn.Linear):
                m.weight_orig.data = checkpoint["state_dict"][n + ".weight"]
                if m.bias is not None:
                    m.bias.data = checkpoint["state_dict"][n + ".bias"]

    print(
        "checkpoint '{}' loaded (epoch {})".format(
            resume_path, checkpoint["epoch"]
        )
    )

    corresponding_epoch = checkpoint["epoch"]

    return model, corresponding_epoch


def do_rewind(
    model, optimizer, scheduler, imp_iter, exp_dir, gpu, percentage_pruning
):
    print("=> Reloading best checkpoint to compute mask")
    model, corresponding_epoch = resume_for_imp(
        model,
        optimizer,
        scheduler,
        imp_iter,
        exp_dir,
        gpu,
        best=True,
        from_epoch_5=False,
    )

    print(f"=> Computing mask ({percentage_pruning} percent pruned)")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(
                module, name="weight", amount=percentage_pruning
            )
        elif isinstance(module, torch.nn.Linear):
            # prune only half for the last layer
            prune.l1_unstructured(
                module,
                name="weight",
                amount=percentage_pruning / 2,
            )

    print("=> Rewinding to epoch 5 with new mask")
    model, corresponding_epoch = resume_for_imp(
        model,
        optimizer,
        scheduler,
        imp_iter,
        exp_dir,
        gpu,
        best=False,
        from_epoch_5=True,
    )
    return model, corresponding_epoch
