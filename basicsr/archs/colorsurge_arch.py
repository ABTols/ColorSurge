import torch
import torch.nn as nn
from basicsr.archs.colorsurge_arch_utils.unet3d import UnetBlock3D
from basicsr.archs.colorsurge_arch_utils.unet import Hook, CustomPixelShuffle_ICNR,  UnetBlockWide, NormType, custom_conv_layer
from basicsr.archs.colorsurge_arch_utils.convnextv2 import ConvNeXtV2
from basicsr.archs.colorsurge_arch_utils.transformer_utils import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from basicsr.archs.colorsurge_arch_utils.position_encoding import PositionEmbeddingSine
from basicsr.archs.colorsurge_arch_utils.transformer import Transformer
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.spynet_arch import SPyNet
from basicsr.utils.video_frame_util import tensor_video_lab2rgb, tensor_lab2rgb
from basicsr.archs.colorspread import ColorSpread
from einops import rearrange, repeat

@ARCH_REGISTRY.register()
class ColorSurge(nn.Module):
    def __init__(self,
                 encoder_name='convnext-l',
                 decoder_name='TinyColorDecoder',
                 num_input_channels=3,
                 input_size=(256, 256),
                 temporal_dim=16,
                 nf=512,
                 num_output_channels=2,
                 last_norm='Weight',
                 do_normalize=False,
                 num_queries=256,
                 num_scales=3,
                 dec_layers=9,
                 encoder_from_pretrain=False,
                 use_temp_query=True):
        super().__init__()

        self.encoder = Encoder(encoder_name, ['norm0', 'norm1', 'norm2', 'norm3'], from_pretrain=encoder_from_pretrain)
        self.encoder.eval()
        self.encoder.cuda()
        test_input = torch.randn(1, num_input_channels, *input_size).cuda()

        self.encoder(test_input)

        self.decoder = Decoder(
            self.encoder.hooks,
            nf=nf,
            temporal_dim=temporal_dim,
            last_norm=last_norm,
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
            decoder_name=decoder_name,
            use_temp_query=use_temp_query,
        )
        self.mapper_layers = nn.Sequential(custom_conv_layer(num_queries + 3, num_output_channels, ks=1, use_activ=False, norm_type=NormType.Spectral))
    
        self.do_normalize = do_normalize
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # optical flow network for feature alignment
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth'
        self.spynet = SPyNet(pretrained=spynet_pretrained)
    
    
    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True
    
    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, x, frame_len):
        _,C,H,W = x.shape
        x = x.view(-1,frame_len,C,H,W)
        self.check_if_mirror_extended(x)

        flows_forward, flows_backward = self.compute_flow(x)
        
        if len(x.shape)==5:
            x = x.view(-1,C,H,W)
            
        if x.shape[1] == 3:
            x = self.normalize(x)

        self.encoder(x)
        out_feat = self.decoder(flows_forward, flows_backward, frame_len)
        
        concat_feat = torch.cat([out_feat, x], dim=1)
        out = self.mapper_layers(concat_feat)

        if self.do_normalize:
            out = self.denormalize(out)
        return out


class Decoder(nn.Module):

    def __init__(self,
                 hooks,
                 nf=512,
                 temporal_dim=16,
                 blur=True,
                 last_norm='Weight',
                 num_queries=256,
                 num_scales=3,
                 dec_layers=9,
                 decoder_name='TinyColorDecoder',
                 use_temp_query=False):
        super().__init__()
        self.hooks = hooks
        self.nf = nf
        self.blur = blur
        self.last_norm = getattr(NormType, last_norm)
        self.decoder_name = decoder_name

        if self.decoder_name == 'TinyColorDecoder':
            self.layers = self.make_layers()
        else:
            self.layers = self.make_layers_3d(frame_len=temporal_dim)

        embed_dim = nf // 2

        self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, blur=self.blur, norm_type=self.last_norm, scale=4)
        
        if self.decoder_name == 'TinyColorDecoder':
            self.color_decoder = TinyColorDecoder(
                in_channels=[512, 512, 256],
                num_queries=num_queries,
                temporal_dim=temporal_dim,
                num_scales=num_scales,
                dec_layers=dec_layers,
                use_temp_query=use_temp_query
            )
        else:
            self.color_decoder = LargeColorDecoder(
                in_channels=[512, 512, 256],
                num_queries=num_queries,
                temporal_dim=temporal_dim,
                num_scales=num_scales,
                dec_layers=dec_layers,
                use_temp_query=use_temp_query
            )

        self.color_spread = ColorSpread(input_channels=256, 
                                        mid_channels=64, 
                                        num_blocks=2, 
                                        )

    def forward(self, flows_forward, flows_backward, frame_len):
        encode_feat = self.hooks[-1].feature 

        out0 = self.layers[0](encode_feat) 
        out1 = self.layers[1](out0)  
        out2 = self.layers[2](out1)  
        out3 = self.last_shuf(out2)
        
        f_b_color = self.color_spread(out3, flows_forward, flows_backward, frame_len)
        out_fusion = out3 + f_b_color
        out = self.color_decoder([out0, out1, out2], out_fusion, frame_len)     
        return out

    def make_layers_3d(self, frame_len):
        decoder_layers = []
        e_in_c = self.hooks[-1].feature.shape[1]
        in_c = e_in_c
        out_c = self.nf
        setup_hooks = self.hooks[-2::-1]

        for layer_index, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[1]
            if layer_index == len(setup_hooks) - 1:
                out_c = out_c // 2
            decoder_layers.append(
                UnetBlock3D(
                    in_c, feature_c, out_c, hook, frame_len=frame_len, final_div=False, blur=self.blur, self_attention=False, norm_type=NormType.Spectral))
            in_c = out_c
        
        return nn.Sequential(*decoder_layers)

    def make_layers(self):
        decoder_layers = []

        e_in_c = self.hooks[-1].feature.shape[1]
        in_c = e_in_c
        out_c = self.nf
        setup_hooks = self.hooks[-2::-1]
        for layer_index, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[1]
            if layer_index == len(setup_hooks) - 1:
                out_c = out_c // 2
            decoder_layers.append(
                UnetBlockWide(
                    in_c, feature_c, out_c, hook, blur=self.blur, self_attention=False, norm_type=NormType.Spectral))
            in_c = out_c
        return nn.Sequential(*decoder_layers)


class Encoder(nn.Module):
    def __init__(self, encoder_name, hook_names, from_pretrain, **kwargs):
        super().__init__()
        if encoder_name == 'convnextv2-l':
            self.arch = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        else:
            raise NotImplementedError

        self.encoder_name = encoder_name
        self.hook_names = hook_names
        self.hooks = self.setup_hooks()

        if from_pretrain:
            self.load_pretrain_model()

    def setup_hooks(self):
        hooks = [Hook(self.arch._modules[name]) for name in self.hook_names]
        return hooks

    def forward(self, x):
        return self.arch(x)
    
    def load_pretrain_model(self):
        if self.encoder_name == 'convnextv2-l':
            self.load('pretrain_models/convnextv2_large_22k_384_ema.pt')
        else:
            raise NotImplementedError
        print('Loaded pretrained convnext model.')

    def load(self, path):
        from basicsr.utils import get_root_logger
        logger = get_root_logger()
        if not path:
            logger.info("No checkpoint found. Initializing model from scratch")
            return
        logger.info("[Encoder] Loading from {} ...".format(path))
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        checkpoint_state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        incompatible = self.arch.load_state_dict(checkpoint_state_dict, strict=False)

        if incompatible.missing_keys:
            msg = "Some model parameters or buffers are not found in the checkpoint:\n"
            msg += str(incompatible.missing_keys)
            logger.warning(msg)
        if incompatible.unexpected_keys:
            msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
            msg += str(incompatible.unexpected_keys)
            logger.warning(msg)


class TinyColorDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_dim=256,
        temporal_dim = 16,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=9,
        pre_norm=False,
        color_embed_dim=256,
        enforce_input_project=True,
        num_scales=3,
        use_temp_query=False
    ):
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)


        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_temp_attention_layers = nn.ModuleList()

        self.transformer_ffn_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_temp_attention_layers.append(
                CrossAttentionLayer(
                    d_model=temporal_dim,
                    nhead=2,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries

        self.use_temp_query = use_temp_query
        if use_temp_query:
            self.query_feat = nn.Embedding(num_queries*temporal_dim, hidden_dim)
            self.query_embed = nn.Embedding(num_queries*temporal_dim, hidden_dim)
        else:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding
        self.num_feature_levels = num_scales
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # input projections
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels):
            if in_channels[i] != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1))
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

    def forward(self, x, img_features, frame_len):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []

        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)    

        _, bs, _ = src[0].shape

        # QxNxC
        if self.use_temp_query:
            bs = bs//frame_len
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            query_embed = rearrange(query_embed, '(c f) b d -> c (b f) d', f=frame_len)
            output = rearrange(output, '(c f) b d -> c (b f) d', f=frame_len)
        else:
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            # Temporal-Attention
            if self.transformer_temp_attention_layers[i] is not None:
                d = output.shape[2]
                output = rearrange(
                    output, 'c (b f) d -> c (b d) f', f=frame_len)
                query_embed = rearrange(
                    query_embed, 'c (b f) d -> c (b d) f', f=frame_len)
                src[level_index] = rearrange(
                    src[level_index], 'c (b f) d -> c (b d) f', f=frame_len)
                pos[level_index] = rearrange(
                    pos[level_index], 'c (b f) d -> c (b d) f', f=frame_len)
                output = self.transformer_temp_attention_layers[i](
                                    output, src[level_index],
                                    memory_mask=None,
                                    memory_key_padding_mask=None,
                                    pos=pos[level_index], query_pos=query_embed
                                    )
                output = rearrange(
                    output, 'c (b d) f-> c (b f) d', d=d)
                query_embed = rearrange(
                    query_embed, 'c (b d) f-> c (b f) d', d=d)
                src[level_index] = rearrange(
                    src[level_index], 'c (b d) f-> c (b f) d', d=d)
                pos[level_index] = rearrange(
                    pos[level_index], 'c (b d) f-> c (b f) d', d=d)
                

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [N, bs, C]  -> [bs, N, C]
        color_embed = self.color_embed(decoder_output)
        out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features)

        return out


class LargeColorDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_dim=256,
        temporal_dim = 16,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=9,
        pre_norm=False,
        color_embed_dim=256,
        enforce_input_project=True,
        num_scales=3,
        use_temp_query=False
    ):
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.transformer_temp_self_attention_layers = nn.ModuleList()
        self.transformer_temp_cross_attention_layers = nn.ModuleList()
        self.transformer_temp_ffn_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            # temp_att
            self.transformer_temp_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=temporal_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_temp_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=temporal_dim,
                    nhead=2,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_temp_ffn_layers.append(
                FFNLayer(
                    d_model=temporal_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries

        self.use_temp_query = use_temp_query
        if use_temp_query:
            self.query_feat = nn.Embedding(num_queries*temporal_dim, hidden_dim)
            self.query_embed = nn.Embedding(num_queries*temporal_dim, hidden_dim)
        else:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding
        self.num_feature_levels = num_scales
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # input projections
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels):
            if in_channels[i] != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1))
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

    def forward(self, x, img_features, frame_len):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []

        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)    

        _, bs, _ = src[0].shape

        # QxNxC
        if self.use_temp_query:
            bs = bs//frame_len
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            query_embed = rearrange(query_embed, '(c f) b d -> c (b f) d', f=frame_len)
            output = rearrange(output, '(c f) b d -> c (b f) d', f=frame_len)
        else:
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            

            if self.transformer_temp_self_attention_layers[i] is not None:
                d = output.shape[2]
                output = rearrange(
                    output, 'c (b f) d -> c (b d) f', f=frame_len)
                query_embed = rearrange(
                    query_embed, 'c (b f) d -> c (b d) f', f=frame_len)
                src[level_index] = rearrange(
                    src[level_index], 'c (b f) d -> c (b d) f', f=frame_len)
                pos[level_index] = rearrange(
                    pos[level_index], 'c (b f) d -> c (b d) f', f=frame_len)

                output = self.transformer_temp_cross_attention_layers[i](
                                    output, src[level_index],
                                    memory_mask=None,
                                    memory_key_padding_mask=None,
                                    pos=pos[level_index], query_pos=query_embed
                                    )
                output = self.transformer_temp_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                output = self.transformer_temp_ffn_layers[i](
                    output
                )

                output = rearrange(
                    output, 'c (b d) f-> c (b f) d', d=d)
                query_embed = rearrange(
                    query_embed, 'c (b d) f-> c (b f) d', d=d)
                src[level_index] = rearrange(
                    src[level_index], 'c (b d) f-> c (b f) d', d=d)
                pos[level_index] = rearrange(
                    pos[level_index], 'c (b d) f-> c (b f) d', d=d)     

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [N, bs, C]  -> [bs, N, C]
        color_embed = self.color_embed(decoder_output)
        out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features)

        return out
