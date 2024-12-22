import torch
from torch.nn.utils import prune

import copy
import torch
import json


def get_list_models() -> list:
    """Return list of models that satisfy path-norm toolkit conditions.
    """
    try:
        with open("ok_models.json", 'r') as in_file:
            ok_models = json.load(in_file)
        return ok_models.keys()
    except IOError:
        raise Exception("Did not find 'ok_models.json' file. " +
                        "Please run check_models.py script.")


def is_model_ok(name) -> bool:
    """Return True if the model satisfies path-norm toolkit conditions.

    Args:
        name: str
        Name of the model

    Return:
        bool
    """
    ok = False
    try:
        with open("ok_models.json", 'r') as in_file:
            ok_models = json.load(in_file)
            if name in ok_models.keys():
                ok = True
        # with open("ok_models.out", 'r') as in_file:
        #     lines = in_file.readlines()
        #     for l in lines:
        #         if l == name:
        #             ok = True
        #             break
    except IOError:
        raise Exception("Did not find 'ok_models.json' file. " +
                        "Please run check_models.py script.")
    return ok


def get_in_out_channels(model) -> tuple:
    """Return in and out channels of the model.
    Raise an error if model does not satisfy path-norm toolkit conditions.

    Args:
        model: str
        Name of the model

    Return:
        tuple (list of in channels, list of out channels, names)

    Raises:
        Exception
            model does not satisfy path-norm toolkit conditions.
    """
    try:
        with open("ok_models.json", 'r') as in_file:
            ok_models = json.load(in_file)
            if model not in ok_models.keys():
                raise Exception("model does not satisfy" +
                                " path-norm toolkit conditions.")
            in_channels, out_channels = [], []
            names = []
            if "MaxPool2d" in ok_models[model].keys():
                for k in ok_models[model]["MaxPool2d"].keys():
                    in_channels.append(
                        ok_models[model]["MaxPool2d"][k]["in_channels"])
                    out_channels.append(
                        ok_models[model]["MaxPool2d"][k]["out_channels"])
                    names.append(k)
            return in_channels, out_channels, names
    except IOError:
        raise Exception("Did not find 'ok_models.json' file. " +
                        "Please run check_models.py script.")


def get_kernel_shapes(name) -> list:
    """Return list of kernel shapes
    (built from AdaptiveAvgPool2d) of the model.
    Raise an error if model does not satisfy path-norm toolkit conditions.

    Args:
        name: str
        Name of the model.

    Return:
        list

    Raises:
        Exception
            model does not satisfy path-norm toolkit conditions.
    """
    try:
        with open("ok_models.json", 'r') as in_file:
            ok_models = json.load(in_file)
            if name not in ok_models.keys():
                raise Exception("model does not satisfy" +
                                " path-norm toolkit conditions.")
            kernel_shape = []
            if "AdaptiveAvgPool2d" in ok_models[name].keys():
                for k in ok_models[name]["AdaptiveAvgPool2d"].keys():
                    kernel_shape.append(
                        ok_models[name][
                            "AdaptiveAvgPool2d"][k]["kernel_shape"])
            return kernel_shape
    except IOError:
        raise Exception("Did not find 'ok_models.json' file. " +
                        "Please run check_models.py script.")


def replace_maxpool2d_with_conv2d(model, name, device, in_place=False):
    """
    Replace max-pooling layers with convolutional layers with weights constant
    equal to one. Works with the usual ResNet PyTorch class, and it is expected
    to be easily adaptable to work with other architectures.

    Args:
        model (torch.nn.Module): Input model.
        name (str): Name of the model.
        device (torch.device): Device on which the model should be placed.
        in_place (bool, optional): If True, modify the input model in place.
        If False, create a deep copy of the model and modify the copy.
        Defaults to False.

    Returns:
        torch.nn.Module: Modified model.
    """
    if in_place is False:
        try:
            new_model = copy.deepcopy(model)
        except Exception as e:
            print(e)
            raise RuntimeError(
                "Deep copy failed, set 'in_place=True' to run the" +
                " function with in place modification of the model"
            )
    else:
        new_model = model
    # Get a list of in/out channels
    # List of tuples representing input and
    # output channels of each max-pooling layer being replaced.
    in_channels, out_channels, names = get_in_out_channels(name)
    i = 0
    idx = []
    layer_names = []
    params = []
    # Save MaxPool2d info
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.MaxPool2d):
            idx.append(i)
            layer_names.append(n)
            params.append([m.kernel_size, m.stride,
                           m.padding, m.dilation])
            # print(i, n, m, names[i])
            assert n == names[i]
            i += 1
    # Replace each MaxPool2d by Conv2d
    for i, n, p in zip(idx, layer_names, params):
        new_model._modules[n] = torch.nn.Conv2d(
            in_channels=in_channels[i],
            out_channels=out_channels[i],
            kernel_size=p[0],
            stride=p[1],
            padding=p[2],
            padding_mode="zeros",
            dilation=p[3],
            groups=in_channels[i],
            bias=False,
            device=device,
        )
        # setattr(new_model, n, torch.nn.Conv2d(
        #     in_channels=in_channels[i],
        #     out_channels=out_channels[i],
        #     kernel_size=p[0],
        #     stride=p[1],
        #     padding=p[2],
        #     padding_mode="zeros",
        #     dilation=p[3],
        #     groups=in_channels[i],
        #     bias=False,
        #     device=device,
        # ))
        new_model._modules[n].weight.data.fill_(1)
    # from torchinfo import summary
    # print(summary(new_model, (1, 3, 224, 224)))
    return new_model


def set_weights_for_path_norm(
    model, exponent=1, provide_original_weights=True
):
    """
    Applies $w\\mapsto |w|^{exponent}$ to all weights $w$ of the model
    for path-norm computation. It handles cases where the weights of the
    convolutional and linear layers are pruned with the torch.nn.utils.prune
    library (but not when their biases are pruned).

    Args:
        model (torch.nn.Module): Input model.
        exponent (float, optional): Exponent for weight transformation.
        Defaults to 1.
        provide_original_weights (bool, optional): If True, provide the
        original weights for resetting later. Defaults to True.

    Returns:
        dict: Original weights of the model if
        `provide_original_weights` is True; otherwise, empty dict.
    """
    # If a module is pruned, its original weights
    # are in weight_orig instead of weight.
    orig_weights = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            if prune.is_pruned(m):
                if provide_original_weights:
                    orig_weights[n + ".weight"] = (
                        m.weight_orig.detach().clone()
                    )
                m.weight_orig.data = torch.abs(m.weight_orig.detach())
                if exponent != 1:
                    m.weight_orig.data = torch.pow(
                        m.weight_orig.detach(), exponent
                    )
            else:
                if provide_original_weights:
                    orig_weights[n + ".weight"] = m.weight.detach().clone()
                m.weight.data = torch.abs(m.weight.detach())
                if exponent != 1:
                    m.weight.data = torch.pow(m.weight.detach(), exponent)
            if m.bias is not None:
                if provide_original_weights:
                    orig_weights[n + ".bias"] = m.bias.detach().clone()
                m.bias.data = torch.abs(m.bias.detach())
                if exponent != 1:
                    m.bias.data = torch.pow(m.bias.detach(), exponent)
        elif isinstance(m, torch.nn.BatchNorm2d):
            if provide_original_weights:
                orig_weights[n + ".weight"] = m.weight.detach().clone()
                orig_weights[n + ".bias"] = m.bias.detach().clone()
                orig_weights[n + ".running_mean"] = (
                    m.running_mean.detach().clone()
                )
                orig_weights[n + ".running_var"] = (
                    m.running_var.detach().clone()
                )
            m.weight.data = torch.abs(m.weight.detach())
            m.bias.data = torch.abs(m.bias.detach())
            m.running_mean.data = torch.abs(m.running_mean.detach())
            # Running_var already non-negative,
            # no need to put it in absolute value

            if exponent != 1:
                m.weight.data = torch.pow(m.weight.detach(), exponent)
                m.bias.data = torch.pow(m.bias.detach(), exponent)
                m.running_mean.data = torch.pow(
                    m.running_mean.detach(), exponent
                )
                m.running_var.data = torch.pow(
                    m.running_var.detach(), exponent
                )
    return orig_weights


def reset_model(name, model, orig_weights):
    """
    Reset weights and maxpool layer of a model.

    Args:
        name (str): Name of the model.
        model (torch.nn.Module): Input model.
        orig_weights (dict): Original weights of the model.
    """
    for n, m in model.named_modules():
        if (
                isinstance(m, torch.nn.Conv2d) or
                isinstance(m, torch.nn.Linear)
        ):
            if prune.is_pruned(m):
                m.weight_orig.data = orig_weights[n + ".weight"]
            else:
                m.weight.data = orig_weights[n + ".weight"]
            if m.bias is not None:
                m.bias.data = orig_weights[n + ".bias"]
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data = orig_weights[n + ".weight"]
            m.bias.data = orig_weights[n + ".bias"]
            m.running_mean.data = orig_weights[n + ".running_mean"]
            m.running_var.data = orig_weights[n + ".running_var"]

    if is_model_ok(name):
        try:
            with open("ok_models.json", 'r') as in_file:
                ok_models = json.load(in_file)
                if name not in ok_models.keys():
                    raise Exception("model does not satisfy" +
                                    " path-norm toolkit conditions.")
                if "MaxPool2d" in ok_models[name].keys():
                    for k, v in ok_models[name]["MaxPool2d"].items():
                        setattr(model, k,
                                torch.nn.MaxPool2d(
                                    kernel_size=v["reset"]["kernel_size"],
                                    stride=v["reset"]["stride"],
                                    padding=v["reset"]["padding"],
                                    dilation=v["reset"]["dilation"],
                                    ceil_mode=False,
                                ))
        except IOError:
            raise Exception("Did not find 'ok_models.json' file. " +
                            "Please run check_models.py script.")
    else:
        raise NotImplementedError(
            "model has to be in:\n" + get_list_models()
        )


def get_path_norm(
    model,
    name,
    device,
    exponent=1,
    constant_input=1,
    in_place=True,
):
    """
    Get the $L^{\textrm{exponent}}$ path-norm of a ResNet model. It does so by
    applying the following recipe given by Theorem A.1 of
    https://arxiv.org/abs/2310.01225.
    1. Modify the model (in_place or not): replace max-pooling-neurons with
    identity ones.
    2. Set all the weights $w$ to $|w|^{\textrm{exponent}}$.
    3. Compute the path norm of the model with a forward-pass with constant
    input equal to one.
    4. Reset the model to its original state.

    Args:
        model (torch.nn.Module): Input model.
        name (str): Name of the model.
        device (torch.device): Device on which the model should be placed.
        exponent (float, optional): Exponent for weight transformation.
        Defaults to 1.
        constant_input (float, optional): Constant input value. Defaults to 1.
        in_place (bool, optional): If True, modify the input model in place.
        Defaults to True.

    Returns:
        float: Path-norm of the model.
    """

    # NOTA BENE: if pruned, we do not remove the pruned framework, and then
    # reapply it afterwards by pruning amount=0 # since this would lose the
    # masks

    if is_model_ok(name):
        kernel_shapes = get_kernel_shapes(name)
        weights_average_pool = 1.0
        for k in kernel_shapes:
            number_antecedents_average_pool = k[0] * k[1]
            weights_average_pool *= 1.0 / number_antecedents_average_pool
    else:
        raise NotImplementedError(
            "model has to be in:\n" + str(get_list_models())
        )

    # 1. Modify the model (in_place or not): replace max-pooling-neurons
    # with identity ones as prescribed in Theorem A.1
    # of https://arxiv.org/abs/2310.01225.
    new_model = replace_maxpool2d_with_conv2d(
        model, name, device, in_place=in_place
    )

    # 2. Set all the weights w to |w|^exponent.
    orig_weights = set_weights_for_path_norm(
        new_model, exponent=exponent, provide_original_weights=in_place
    )

    # 3. Compute the path norm of the model with constant input equal to one.
    new_model.eval()
    batch_size = 1
    input_size = (batch_size, 3, 224, 224)
    x = constant_input * torch.ones(input_size)
    x = x.to(device)
    with torch.no_grad():
        try:
            if exponent != 0:
                path_norm = torch.pow(new_model(x).sum(), 1 / exponent).item()
                path_norm *= weights_average_pool ** (exponent - 1)
            else:
                path_norm = new_model(x).sum().item()
        except AttributeError:
            path_norm = torch.pow(
                new_model(x)['out'].sum(), 1 / exponent).item()
            path_norm *= weights_average_pool ** (exponent - 1)

    # 4. Reset the model to its original state.
    if in_place:
        reset_model(name, model, orig_weights)
    return path_norm


def compute_path_norms(model, name, exponents, device, data_parallel):
    """
    Compute path-norms for different exponents of a ResNet model.

    Args:
        model (torch.nn.Module): The ResNet model.
        name (str): Name of the model.
        exponents (list): List of exponents for computing path-norms.
        device (torch.device): Device on which the model should be placed.
        data_parallel (bool): If True, the model is parallelized using
        DataParallel. Currently not supported for path-norm computation.

    Returns:
        list: List of path-norms corresponding to different exponents.
    Raises:
        ValueError: If data_parallel is set to True. Data parallelism is not
        supported yet for the computation of path-norm. A special treatment
        would be needed to replace the maxpool layer with a conv layer.
    """

    path_norms = []

    if data_parallel:
        raise ValueError(
            "Data parallelism is not supported yet for the computation of" +
            " path-norm. A special treatment would be needed to replace" +
            " the maxpool layer with a conv layer."
        )

    for exponent in exponents:
        path_norm = get_path_norm(
            model,
            name,
            device,
            exponent=exponent,
        )
        path_norms.append(path_norm)

    return path_norms
