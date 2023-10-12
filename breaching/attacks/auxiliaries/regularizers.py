"""Various regularizers that can be re-used for multiple attacks."""

import torch

from .deepinversion import DeepInversionFeatureHook
import torchvision


class _LinearFeatureHook:
    """Hook to retrieve input to given module."""

    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        input_features = input[0]
        self.features = input_features

    def close(self):
        self.hook.remove()



class FeatureRegularization(torch.nn.Module):
    """Feature regularization implemented for the last linear layer at the end."""

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, shared_data, labels, *args, **kwargs):
        self.measured_features = []
        for user_data in shared_data:
            # Assume last two gradient vector entries are weight and bias:
            weights = user_data["gradients"][-2]
            bias = user_data["gradients"][-1]
            grads_fc_debiased = weights / bias[:, None]
            features_per_label = []
            for label in labels:
                if bias[label] != 0:
                    features_per_label.append(grads_fc_debiased[label])
                else:
                    features_per_label.append(torch.zeros_like(grads_fc_debiased[0]))
            self.measured_features.append(torch.stack(features_per_label))

        self.refs = [None for model in models]
        for idx, model in enumerate(models):
            for module in model.modules():
                # Keep only the last linear layer here:
                if isinstance(module, torch.nn.Linear):
                    self.refs[idx] = _LinearFeatureHook(module)

    def forward(self, tensor, *args, **kwargs):
        regularization_value = 0
        for ref, measured_val in zip(self.refs, self.measured_features):
            regularization_value += (ref.features - measured_val).pow(2).mean()
        return regularization_value * self.scale

    def __repr__(self):
        return f"Feature space regularization, scale={self.scale}"


class LinearLayerRegularization(torch.nn.Module):
    """Linear layer regularization implemented for arbitrary linear layers. WIP Example."""

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, shared_data, *args, **kwargs):
        self.measured_features = []
        self.refs = [list() for model in models]

        for idx, (model, user_data) in enumerate(zip(models, shared_data)):
            # 1) Find linear layers:
            linear_layers = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linear_layers.append(name)
                    self.refs[idx].append(_LinearFeatureHook(module))
            named_grads = {name: g for (g, (name, param)) in zip(user_data["gradients"], model.named_parameters())}

            # 2) Check features
            features = []
            for linear_layer in linear_layers:
                weights = named_grads[linear_layer + ".weight"]
                bias = named_grads[linear_layer + ".bias"]
                grads_fc_debiased = (weights / bias[:, None]).mean(dim=0)  # At some point todo: Make this smarter
                features.append(grads_fc_debiased)
            self.measured_features.append(features)

    def forward(self, tensor, *args, **kwargs):
        regularization_value = 0
        for model_ref, data_ref in zip(self.refs, self.measured_features):
            for linear_layer, data in zip(model_ref, data_ref):
                regularization_value += (linear_layer.features.mean(dim=0) - data).pow(2).sum()

    def __repr__(self):
        return f"Feature space regularization, scale={self.scale}"


class TotalVariation(torch.nn.Module):
    """Computes the total variation value of an (image) tensor, based on its last two dimensions.
    Optionally also Color TV based on its last three dimensions.

    The value of this regularization is scaled by 1/sqrt(M*N) times the given scale."""

    def __init__(self, setup, scale=0.1, inner_exp=1, outer_exp=1, double_opponents=False, eps=1e-8):
        """scale is the overall scaling. inner_exp and outer_exp control isotropy vs anisotropy.
        Optionally also includes proper color TV via double opponents."""
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.inner_exp = inner_exp
        self.outer_exp = outer_exp
        self.eps = eps
        self.double_opponents = double_opponents

        grad_weight = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], **setup).unsqueeze(0).unsqueeze(1)
        grad_weight = torch.cat((torch.transpose(grad_weight, 2, 3), grad_weight), 0)
        self.groups = 6 if self.double_opponents else 3
        grad_weight = torch.cat([grad_weight] * self.groups, 0)
        
        self.sclae_factor = 1

        self.register_buffer("weight", grad_weight)

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        """Use a convolution-based approach."""

        if (scale_factor:=kwargs.get("tv_scale_factor")) is not None:
            if self.sclae_factor != scale_factor:
                self.sclae_factor = scale_factor
                self.scale *= scale_factor
        if self.double_opponents:
            tensor = torch.cat(
                [
                    tensor,
                    tensor[:, 0:1, :, :] - tensor[:, 1:2, :, :],
                    tensor[:, 0:1, :, :] - tensor[:, 2:3, :, :],
                    tensor[:, 1:2, :, :] - tensor[:, 2:3, :, :],
                ],
                dim=1,
            )
        diffs = torch.nn.functional.conv2d(
            tensor, self.weight, None, stride=1, padding=1, dilation=1, groups=self.groups
        )
        squares = (diffs.abs() + self.eps).pow(self.inner_exp)
        squared_sums = (squares[:, 0::2] + squares[:, 1::2]).pow(self.outer_exp)
        return squared_sums.mean() * self.scale

    def __repr__(self):
        return (
            f"Total Variation, scale={self.scale}. p={self.inner_exp} q={self.outer_exp}. "
            f"{'Color TV: double oppponents' if self.double_opponents else ''}"
        )


class OrthogonalityRegularization(torch.nn.Module):
    """This is the orthogonality regularizer described Qian et al.,

    "MINIMAL CONDITIONS ANALYSIS OF GRADIENT-BASED RECONSTRUCTION IN FEDERATED LEARNING"
    """

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        if tensor.shape[0] == 1:
            return 0
        else:
            B = tensor.shape[0]
            full_products = (tensor.unsqueeze(0) * tensor.unsqueeze(1)).pow(2).view(B, B, -1).mean(dim=2)
            idx = torch.arange(0, B, out=torch.LongTensor())
            full_products[idx, idx] = 0
            return full_products.sum()

    def __repr__(self):
        return f"Input Orthogonality, scale={self.scale}"


class NormRegularization(torch.nn.Module):
    """Implement basic norm-based regularization, e.g. an L2 penalty."""

    def __init__(self, setup, scale=0.1, pnorm=2.0):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.pnorm = pnorm

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        return 1 / self.pnorm * tensor.pow(self.pnorm).mean() * self.scale

    def __repr__(self):
        return f"Input L^p norm regularization, scale={self.scale}, p={self.pnorm}"


class DeepInversion(torch.nn.Module):
    """Implement a DeepInversion based regularization as proposed in DeepInversion as used for reconstruction in
    Yin et al, "See through Gradients: Image Batch Recovery via GradInversion".
    """

    def __init__(self, setup, scale=0.1, first_bn_multiplier=10):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.first_bn_multiplier = first_bn_multiplier

    def initialize(self, models, *args, **kwargs):
        """Initialize forward hooks."""
        self.losses = [list() for model in models]
        for idx, model in enumerate(models):
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    self.losses[idx].append(DeepInversionFeatureHook(module))

    def forward(self, tensor, *args, **kwargs):
        rescale = [self.first_bn_multiplier] + [1.0 for _ in range(len(self.losses[0]) - 1)]
        feature_reg = 0
        for loss in self.losses:
            feature_reg += sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss)])
        return self.scale * feature_reg

    def __repr__(self):
        return f"Deep Inversion Regularization (matching batch norms), scale={self.scale}, first-bn-mult={self.first_bn_multiplier}"

class Activation(torch.nn.Module):
    """compute the activaton difference before fc layer"""
    def __init__(self, setup, scale=0.1, loss_fn='MSE'):
        """scale is the overall scaling. inner_exp and outer_exp control isotropy vs anisotropy.
        Optionally also includes proper color TV via double opponents."""
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.loss_fn = loss_fn
        self.constraints = None

    def initialize(self, models, share_ddata, labels, constraints=None, *args, **kwargs):
        self.constraints = constraints
        self.refs = [None for model in models]
        for idx, model in enumerate(models):
            for module in model.modules():
                # Keep only the last linear layer here:
                if isinstance(module, torch.nn.Linear):
                    self.refs[idx] = _LinearFeatureHook(module)
    def forward(self, tensor, *args, **kwargs):

        regularization_value = 0
        for ref, activation in zip(self.refs, self.constraints):
            # regularization_value += (ref.features - measured_val).pow(2).mean()
            if self.loss_fn == 'MSE':
                regularization_value += torch.nn.MSELoss()(ref.features, activation)
            elif self.loss_fn == 'Cosin':
                regularization_value += torch.nn.CosineSimilarity()(ref.features, activation)
                # loss = torch.nn.CosineSimilarity(self.constraints, tensor)
            else:
                raise NotImplementedError("Not Implement the Loss Function of Activation")
        
        return regularization_value * self.scale


    def __repr__(self):
        return f"Activation, scale={self.scale}. loss_function={self.loss_fn}. "



class Channelgrad(torch.nn.Module):
    """compute the channel gradient difference"""
    def __init__(self, setup, scale=0.3):
        """scale is the overall scaling. inner_exp and outer_exp control isotropy vs anisotropy.
        Optionally also includes proper color TV via double opponents."""
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, *args, **kwargs):
        pass
    def forward(self, tensor, *args, **kwargs):

        if len(tensor.shape) != 4:
            raise Exception(f'Error candiate shape {tensor.shape}, excpet 4')
        cos_0_1 = self.cal_cosine(tensor.grad[:, 0, :, :],tensor.grad[:, 1, :, :] )
        cos_0_2 = self.cal_cosine(tensor.grad[:, 0, :, :],tensor.grad[:, 2, :, :] )
        cos_1_2 = self.cal_cosine(tensor.grad[:, 1, :, :],tensor.grad[:, 2, :, :] )

    # print(cos_0_1, cos_0_2, cos_1_2)

    
        return self.scale * (1 - (cos_0_1 + cos_0_2 + cos_1_2)/3)
        # return regularization_value * self.scale

    def cal_cosine(self, a, b):
        scalar_product = (a * b).sum() 
        rec_norm = a.pow(2).sum()
        data_norm = b.pow(2).sum()

        objective = 1 - scalar_product / ((rec_norm.sqrt() * data_norm.sqrt()) + 1e-6)
        
        return objective

    def __repr__(self):
        return f"channel grad, scale={self.scale}."



class Clip(torch.nn.Module):
    """Computes the total variation value of an (image) tensor, based on its last two dimensions.
    Optionally also Color TV based on its last three dimensions.

    The value of this regularization is scaled by 1/sqrt(M*N) times the given scale."""

    def __init__(self, setup, scale=0.1, range_exp=0.1, percentage_exp=0.1, dm=0, ds=1, channel=3, eps=1e-8):
        """scale is the overall scaling. inner_exp and outer_exp control isotropy vs anisotropy.
        Optionally also includes proper color TV via double opponents."""
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.range_exp = range_exp
        self.percentage_exp = percentage_exp
        self.dm = dm
        self.ds = ds
        self.eps = eps

        self.dm = torch.as_tensor(self.dm)[None, None, None, None].expand(-1, channel, -1, -1)
        self.ds = torch.as_tensor(self.ds)[None, None, None, None].expand(-1, channel, -1, -1)

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        
        self.dm = self.dm.to(device=tensor.device)
        self.ds = self.ds.to(device=tensor.device)

        tensor_clip = torch.max(torch.min(tensor, (1 - self.dm) / self.ds), -self.dm / self.ds) 
        min_val, max_val = torch.min(tensor), torch.max(tensor)
        tensor_scale = (tensor - min_val) * max_val / (max_val - min_val + self.eps)
        
        # reg_range =  torch.sum((tensor - tensor_clip)**2)
        # reg_scale =  torch.sum((tensor - tensor_scale)**2)

        reg_range =  torch.mean((tensor - tensor_clip)**2)
        reg_scale =  torch.mean((tensor - tensor_scale)**2)
        
        # return self.range_exp*reg_range + self.percentage_exp*reg_scale
        return self.scale*(reg_range + reg_scale)

class GroupRegularization(torch.nn.Module):
    def __init__(self, setup, scale=0.1, **kwargs) -> None:
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        candidate_mean = kwargs["avg_candidates"]

        # candidate_mean = torch.mean(torch.stack(candidates, dim=0)) 
        return self.scale * torch.mean((tensor - candidate_mean)**2)

class PixRegularization(torch.nn.Module):
    def __init__(self, setup, scale=0.1, loss_type="l2") -> None:
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.loss_type = loss_type
        self.scale_factor = 1

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        if (scale_factor:=kwargs.get("pix_scale_factor")) is not None:
            if self.scale_factor != scale_factor:
                self.scale_factor = scale_factor
                self.scale *= scale_factor
        
        
        if kwargs.get("ref_candidate") is None:
            return 0
        if self.loss_type == "cosim":
            ref_candidate = kwargs["ref_candidate"]
            scalar_product = (ref_candidate * tensor).sum()
            ref_norm = ref_candidate.pow(2).sum()
            tensor_norm = tensor.pow(2).sum()
            objective = 1 - scalar_product / (ref_norm.sqrt() * tensor_norm.sqrt() + 1e-8)
        elif self.loss_type == "l2":
            objective = torch.mean((tensor - ref_candidate)**2)
        else:
            raise AssertionError("loss_type must be either 'cosim' or 'l2'")
    
        if not torch.isfinite(self.scale * objective):
            print("1234")
            exit(0)
        return self.scale * objective

    def __repr__(self):
        return (
            f"PixRegularization, scale={self.scale}."
            f"Loss type: {self.loss_type}."
        )
class PerpRegularization(torch.nn.Module):
    def __init__(self, setup, scale=0.1, percpt_weight=0.1, style_weight=0.1,  loss_type="l2", model_type="alexnet", ignore_layers=6) -> None:
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.scale_factor = 1
        self.perceptual_weight = percpt_weight
        self.style_weight = style_weight
        
        self.model_type = model_type
        self.loss_type = loss_type
        
        self.ignore_layers = ignore_layers

        
        self.perp_model = getattr(torchvision.models, self.model_type)(pretrained=True)  
        self.perp_model = self.perp_model.features.to(**self.setup) # only use the feature layer for feature extraction
        self.perp_model.eval()
        for param in self.perp_model.parameters():
            param.requires_grad = False
            
        if self.ignore_layers >= len(self.perp_model):
            raise AssertionError("The ignore layers must less than the total length of perp model")

        # self.layer_weights = [1.0 / len(self.perp_model)] * len(self.perp_model)
        # #increasing layer weights for deeper layers
        # self.layer_weights = [self.layer_weights[i] * (i+1) for i in range(len(self.layer_weights))]
        
        self.layer_weights = [0 for i in range(self.ignore_layers)]
        
        self.layer_weights += torch.arange(1, len(self.perp_model) - self.ignore_layers +1, 1, dtype=torch.float32, device=self.setup["device"]) / (len(self.perp_model) - self.ignore_layers)

        if self.loss_type == 'l1':
            self.loss = torch.nn.L1Loss()
        elif self.loss_type == 'l2':
            self.loss = torch.nn.MSELoss()
        elif self.loss_type == 'fro':
            self.loss = None
        else:
            raise NotImplementedError(f'{loss_type} loss has not been supported.')

    def initialize(self, models, *args, **kwargs):
        pass
        
    
    def featureExtractor(self, x):
        output = []
        for key, layer in self.perp_model._modules.items():
            x = layer(x)
            output.append(x.clone())
        return output
            
    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def forward(self, tensor, *args, **kwargs):
        if (scale_factor:=kwargs.get("perp_scale_factor")) is not None:
            if self.scale_factor != scale_factor:
                self.scale_factor = scale_factor
                self.scale *= scale_factor
        if kwargs.get("ref_candidate") is None:
            return 0
        # breakpoint() 
        ref_feature = self.featureExtractor(kwargs["ref_candidate"]) 
        tensor_feature = self.featureExtractor(tensor)
        
        ##TODO add perception loss 

        ## calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            
            for k, (ref, t) in enumerate(zip(ref_feature, tensor_feature)):
                if self.loss_type == 'fro':
                    percep_loss += torch.norm(t - ref, p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.loss(t, ref) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        ## calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k, (ref, t) in enumerate(zip(ref_feature, tensor_feature)):
                if self.loss_type == 'fro':
                    style_loss += torch.norm(self._gram_mat(t) - self._gram_mat(ref), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.loss(self._gram_mat(t), self._gram_mat(ref)) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None
        
        objective = percep_loss + style_loss      
        return self.scale * objective

    def __repr__(self):
        return (
            f"PerpRegularization, scale={self.scale}. percpt_weight={self.perceptual_weight} style_weights={self.style_weight}. "
            f"Loss type: {self.loss_type}. Model type: {self.model_type}."
            f"Layer weights: {self.layer_weights}."
        )
regularizer_lookup = dict(
    total_variation=TotalVariation,
    orthogonality=OrthogonalityRegularization,
    norm=NormRegularization,
    deep_inversion=DeepInversion,
    features=FeatureRegularization,
    #add by me
    activation=Activation,
    channelgrad=Channelgrad,
    clip=Clip,
    group=GroupRegularization,
    pix=PixRegularization,
    perp=PerpRegularization,
)
