
# hide
import sys
sys.path.append("..")
from nbdev.showdoc import *
# default_exp models.unet
# export
from fastai.basics import *
# from faimed3d.basics import *
# from fastai.vision.all import create_body, hook_outputs
from einops import rearrange, repeat

# export
import faimed3d
from faimed3d.layers import *
## Dynamic Unet 3D
# Fastai's `DynamicUnet` allows construction of a UNet using any pretrained CNN as backbone/encoder. A key module is `nn.PixelShuffle` which allows subpixel convolutions for upscaling in the UNet Blocks. However, `nn.PixelShuffle` is only for 2D images, so in faimed3d `nn.ConvTranspose3d` is used instead. 
# export
class ConvTranspose3D(nn.Sequential):
    "Upsample by 2` from `ni` filters to `nf` (default `ni`), using `nn.ConvTranspose3D`."
    def __init__(self, ni, nf=None, scale=2, blur=False, act_cls=None, norm_type=None, **kwargs):
        super().__init__()
        nf = ifnone(nf, ni)
        layers = [ConvLayer(ni, nf, ndim=3, act_cls=act_cls, norm_type=norm_type, transpose=True, **kwargs)]
      #  layers[0].weight.data.copy_(icnr_init(layers[0].weight.data))
        if blur: layers += [nn.ReplicationPad3d((1,0,1,0,1,0)), nn.AvgPool3d(2, stride=1)]
        super().__init__(*layers)
        



# # Fastai's `PixelShuffle_ICNR` first performes a convolution to increase the layer size, then applies `PixelShuffle` to resize the image. A special initialization technique is applied to `PixelShuffle`, which can reduce checkerboard artifacts (see https://arxiv.org/pdf/1707.02937.pdf). It is probably not needed for `nn.ConvTranspose3d`
# ConvTranspose3D(256, 128)(torch.randn((1, 256, 3, 13, 13))).size()
# ConvTranspose3D(256, 128, blur = True)(torch.randn((1, 256, 3, 13, 13))).size()



# To work with 3D data, the `UnetBlock` from fastai is adapted, replacing `PixelShuffle_ICNR` with the above created `ConvTranspose3D` and also adapting all conv-layers and norm-layers to the 3rd dimension. As small differences in size may appear, `forward`-func contains a interpolation step, which is also adapted to work with 5D input instead of 4D. `UnetBlock3D` receives the lower level features as hooks.
# export
class UnetBlock3D(Module):
    "A quasi-UNet block, using `ConvTranspose3d` for upsampling`."
    @delegates(ConvLayer.__init__)
    def __init__(self, up_in_c, x_in_c, out_c, hook, frame_len=16, final_div=True, blur=False, act_cls=defaults.activation,
                 self_attention=False, init=nn.init.kaiming_normal_, norm_type=None, **kwargs):
        self.hook = hook
        self.frame_len = frame_len
        self.up = ConvTranspose3D(up_in_c, up_in_c//2, blur=blur, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.bn = BatchNorm(x_in_c, ndim=3)
        ni = up_in_c//2 + x_in_c
        nf = ni if final_div else ni//2
        self.conv1 = ConvLayer(ni, nf, ndim=3, act_cls=act_cls, norm_type=norm_type, **kwargs)
        
        self.conv2 = ConvLayer(nf, out_c, ndim=3, act_cls=act_cls, norm_type=norm_type,
                               xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.relu = act_cls()
        apply_init(nn.Sequential(self.conv1, self.conv2), init)

    def forward(self, up_in):
        # import pdb
        # pdb.set_trace()
        # s = self.hook.stored
        # s = self.hook.feature
        s = rearrange(self.hook.feature, '(b f) c h w -> b c f h w',f=self.frame_len)
        up_in_feature = rearrange(up_in, '(b f) c h w -> b c f h w',f=self.frame_len)
        up_out = self.up(up_in_feature)
        ssh = s.shape[-3:]
        if ssh != up_out.shape[-3:]:
            up_out = F.interpolate(up_out, s.shape[-3:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        result = self.conv2(self.conv1(cat_x))
        return rearrange(result, 'b c f h w -> (b f) c h w')

if __name__=="__main__":
    CT3d1 = ConvTranspose3D(256, 128)
    input_tensor = torch.randn((1, 256, 16, 8, 8))
    output1 = CT3d1(input_tensor)
    print(output1.shape)

    CT3d2 = ConvTranspose3D(256, 128, blur = True)
    output2 = CT3d1(input_tensor)
    print(output2.shape)



    unet_input = torch.randn([1, 1536, 10, 8, 8])
    unet_re = torch.randn([1, 786, 10, 16, 16])

    unet_model = UnetBlock3D(up_in_c=1536, x_in_c=786, hook=unet_re, final_div=False, blur=True, self_attention=False,).eval()
    unet_output = unet_model(unet_input)

    import pdb
    pdb.set_trace()

