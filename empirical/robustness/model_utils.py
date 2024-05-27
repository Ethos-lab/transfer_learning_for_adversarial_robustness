import torch as ch
from torch import nn
import dill
import os
from .tools import helpers, constants
from .attacker import AttackerModel
import pdb


class FeatureExtractor(ch.nn.Module):
    '''
    Tool for extracting layers from models.

    Args:
        submod (torch.nn.Module): model to extract activations from
        layers (list of functions): list of functions where each function,
            when applied to submod, returns a desired layer. For example, one
            function could be `lambda model: model.layer1`.

    Returns:
        A model whose forward function returns the activations from the layers
            corresponding to the functions in `layers` (in the order that the
            functions were passed in the list).
    '''
    def __init__(self, submod, layers):
        # layers must be in order
        super(FeatureExtractor, self).__init__()
        self.submod = submod
        self.layers = layers
        self.n = 0

        for layer_func in layers:
            layer = layer_func(self.submod)
            def hook(module, _, output):
                module.register_buffer('activations', output)

            layer.register_forward_hook(hook)

    def forward(self, *args, **kwargs):
        """
        """
        # self.layer_outputs = {}
        out = self.submod(*args, **kwargs)
        activs = [layer_fn(self.submod).activations for layer_fn in self.layers]
        return [out] + activs

class DummyModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        return self.model(x)


class NormalizerModel(nn.Module):
    def __init__(self, model, dataset):
        super().__init__()
        self.model = model
        self.normalize = helpers.InputNormalize(dataset.mean, dataset.std)

    def forward(self, x, *args, **kwargs):
        x = self.normalize(x)
        return self.model(x)


def make_and_restore_model(*_, arch, dataset, resume_path=None,
         parallel=False, pytorch_pretrained=False, add_custom_forward=False, attack_model=True):
    """
    Makes a model and (optionally) restores it from a checkpoint.

    Args:
        arch (str|nn.Module): Model architecture identifier or otherwise a
            torch.nn.Module instance with the classifier
        dataset (Dataset class [see datasets.py])
        resume_path (str): optional path to checkpoint saved with the
            robustness library (ignored if ``arch`` is not a string)
        not a string
        parallel (bool): if True, wrap the model in a DataParallel
            (defaults to False)
        pytorch_pretrained (bool): if True, try to load a standard-trained
            checkpoint from the torchvision library (throw error if failed)
        add_custom_forward (bool): ignored unless arch is an instance of
            nn.Module (and not a string). Normally, architectures should have a
            forward() function which accepts arguments ``with_latent``,
            ``fake_relu``, and ``no_relu`` to allow for adversarial manipulation
            (see `here`<https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-with-custom-architectures>
            for more info). If this argument is True, then these options will
            not be passed to forward(). (Useful if you just want to train a
            model and don't care about these arguments, and are passing in an
            arch that you don't want to edit forward() for, e.g.  a pretrained model)
    Returns:
        A tuple consisting of the model (possibly loaded with checkpoint), and the checkpoint itself
    """
    if (not isinstance(arch, str)) and add_custom_forward:
        arch = DummyModel(arch)

    classifier_model = dataset.get_model(arch, pytorch_pretrained) if \
                            isinstance(arch, str) else arch

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = ch.load(resume_path, pickle_module=dill)
        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        try:
            sd = checkpoint[state_dict_path]
            load_epoch = checkpoint['epoch']
        except KeyError:
            print('checkpoint only has state dict !!!')
            sd = checkpoint
            load_epoch = -1

        strict_loading = True
        if 'fc.weight' not in sd.keys() and 'fc.bias' not in sd.keys():
            strict_loading = False
        sd = {k.replace('1.module.', ''):v for k,v in sd.items()}
        sd = {k.replace('module.model.', ''):v for k,v in sd.items()}
        sd = {k:v for k,v in sd.items() if 'attacker' not in k}

        classifier_model.load_state_dict(sd, strict=strict_loading)


        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, load_epoch))

    elif resume_path:
        error_msg = "=> no checkpoint found at '{}'".format(resume_path)
        raise ValueError(error_msg)

    if attack_model:
        model = AttackerModel(classifier_model, dataset)
    else:
        model = NormalizerModel(classifier_model, dataset)

    if parallel:
        model = ch.nn.DataParallel(model)
    model = model.cuda()

    return model, checkpoint

def model_dataset_from_store(s, overwrite_params={}, which='last'):
    '''
    Given a store directory corresponding to a trained model, return the
    original model, dataset object, and args corresponding to the arguments.
    '''
    # which options: {'best', 'last', integer}
    if type(s) is tuple:
        s, e = s
        s = cox.store.Store(s, e, mode='r')

    m = s['metadata']
    df = s['metadata'].df

    args = df.to_dict()
    args = {k:v[0] for k,v in args.items()}
    fns = [lambda x: m.get_object(x), lambda x: m.get_pickle(x)]
    conds = [lambda x: m.schema[x] == s.OBJECT, lambda x: m.schema[x] == s.PICKLE]
    for fn, cond in zip(fns, conds):
        args = {k:(fn(v) if cond(k) else v) for k,v in args.items()}

    args.update(overwrite_params)
    args = Parameters(args)

    data_path = os.path.expandvars(args.data)
    if not data_path:
        data_path = '/tmp/'

    dataset = DATASETS[args.dataset](data_path)

    if which == 'last':
        resume = os.path.join(s.path, constants.CKPT_NAME)
    elif which == 'best':
        resume = os.path.join(s.path, constants.CKPT_NAME_BEST)
    else:
        assert isinstance(which, int), "'which' must be one of {'best', 'last', int}"
        resume = os.path.join(s.path, ckpt_at_epoch(which))

    model, _ = make_and_restore_model(arch=args.arch, dataset=dataset,
                                      resume_path=resume, parallel=False)
    return model, dataset, args
