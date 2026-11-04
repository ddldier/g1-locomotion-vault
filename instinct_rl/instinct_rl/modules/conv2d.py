# 导入 PyTorch 主库。
import torch

# 从 instinct_rl.modules.mlp 导入 MlpModel，用于构建全连接网络。
from instinct_rl.modules.mlp import MlpModel
# 从 instinct_rl.modules.utils 导入 conv2d_output_shape，用于计算卷积/池化层的输出尺寸。
from instinct_rl.modules.utils import conv2d_output_shape


# Conv2dModel 类：一个二维卷积模型组件，支持可选的 max-pooling 或通过步长下采样。
# 需要提供输入通道数，但不需要输入的空间尺寸。内部使用 torch.nn.Conv2d。
class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    # 初始化 Conv2dModel。
    # 参数：
    #   in_channels: 输入通道数。
    #   channels: 每个卷积层的输出通道数列表。
    #   kernel_sizes: 每个卷积层的卷积核大小列表。
    #   strides: 每个卷积层的步长列表。
    #   paddings: 每个卷积层的填充列表，若为 None 则全部填充 0。
    #   nonlinearity: 非线性激活函数模块类（不是函数），默认为 torch.nn.ReLU。
    #   use_maxpool: 若为 True，则卷积层步长强制为 1，下采样由 MaxPool2d 完成；否则使用卷积步长下采样。
    #   head_sizes: 可选参数，在卷积层顶部添加 MLP 头（当前代码中未使用，可能是遗留参数）。
    #   normlayer: 归一化层，若为 None 则不使用；若为字符串则从 torch.nn 中获取对应类。
    def __init__(
        self,
        in_channels,
        channels,
        kernel_sizes,
        strides,
        paddings=None,
        nonlinearity=torch.nn.ReLU,  # Module, not Functional.
        use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
        head_sizes=None,  # Put an MLP head on top.
        normlayer=None,  # If None, will not be used
    ):
        # 调用父类 nn.Module 的初始化。
        super().__init__()
        # 如果 paddings 为 None，则生成与 channels 等长的全 0 列表。
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        # 如果 normlayer 是字符串，则从 torch.nn 中获取对应的类。
        if isinstance(normlayer, str):
            normlayer = getattr(torch.nn, normlayer)
        # 断言 channels、kernel_sizes、strides、paddings 的长度一致。
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        # 构建每个卷积层的输入通道列表：第一个为 in_channels，后续为前一层输出通道。
        in_channels = [in_channels] + channels[:-1]
        # 创建一个全 1 列表，长度与 strides 相同。
        ones = [1 for _ in range(len(strides))]
        # 如果使用 maxpool，则卷积步长全部设为 1，maxpool 步长使用原始 strides。
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        # 否则卷积步长使用原始 strides，maxpool 步长全部设为 1。
        else:
            maxp_strides = ones
        # 根据配置创建卷积层列表。
        conv_layers = [
            torch.nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=k, stride=s, padding=p)
            for (ic, oc, k, s, p) in zip(in_channels, channels, kernel_sizes, strides, paddings)
        ]
        # 用于构建顺序容器的列表。
        sequence = list()
        # 遍历每个卷积层、输出通道数、对应的 maxpool 步长。
        for conv_layer, oc, maxp_stride in zip(conv_layers, channels, maxp_strides):
            # 如果指定了归一化层，则依次添加卷积层、归一化层、非线性激活。
            if normlayer is not None:
                sequence.extend([conv_layer, normlayer(oc), nonlinearity()])
            # 否则只添加卷积层和非线性激活。
            else:
                sequence.extend([conv_layer, nonlinearity()])
            # 如果 maxpool 步长大于 1，则添加 MaxPool2d 层（无填充）。
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        # 将所有层组合成 nn.Sequential 模块。
        self.conv = torch.nn.Sequential(*sequence)

    # 前向传播：输入形状应为 [B, C, H, W]，直接通过卷积序列。
    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)

    # 辅助函数：给定输入的高度 h、宽度 w 和可选的通道数 c，计算经过卷积序列后的输出展平尺寸。
    # 不实际执行前向传播，而是遍历子层，使用 conv2d_output_shape 更新尺寸。
    def conv_out_size(self, h, w, c=None):
        """Helper function to return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        # 遍历卷积序列中的每个子层。
        for child in self.conv.children():
            try:
                # 如果是卷积层或池化层，使用 conv2d_output_shape 更新 h, w。
                h, w = conv2d_output_shape(h, w, child.kernel_size, child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                # 如果是卷积层，更新输出通道数 c。
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        # 返回展平后的总元素数：h * w * c。
        return h * w * c

    # 辅助函数：给定输入高度 h 和宽度 w，返回经过卷积序列后的输出分辨率 (h, w)。
    def conv_out_resolution(self, h, w):
        """Helper function that return the resolution (H, W) for a giben input resolution"""
        # 遍历卷积序列中的每个子层。
        for child in self.conv.children():
            try:
                # 如果是卷积层或池化层，更新 h, w。
                h, w = conv2d_output_shape(h, w, child.kernel_size, child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                # 如果是卷积层，更新输出通道数 c（此处未使用，但保留以匹配原代码逻辑）。
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        # 返回最终的高度和宽度。
        return h, w


# Conv2dHeadModel 类：由 Conv2dModel 和一个全连接 MlpModel 头组成。
# 需要提供完整的输入图像形状来实例化 MLP 头。
class Conv2dHeadModel(torch.nn.Module):
    """Model component composed of a ``Conv2dModel`` component followed by
    a fully-connected ``MlpModel`` head.  Requires full input image shape to
    instantiate the MLP head.
    """

    # 初始化 Conv2dHeadModel。
    # 参数：
    #   image_shape: 输入图像形状，格式为 (C, H, W)。
    #   channels: 卷积层输出通道数列表。
    #   kernel_sizes: 卷积核大小列表。
    #   strides: 步长列表。
    #   hidden_sizes: MLP 头的隐藏层大小列表或单个整数。
    #   output_size: 输出维度，若为 None 则最后一层不加非线性激活。
    #   paddings: 卷积填充列表。
    #   nonlinearity: 非线性激活函数类或名称字符串。
    #   use_maxpool: 是否使用 maxpool 下采样。
    #   normlayer: 归一化层，若为 None 则不使用。
    def __init__(
        self,
        image_shape,
        channels,
        kernel_sizes,
        strides,
        hidden_sizes,
        output_size=None,  # if None: nonlinearity applied to output.
        paddings=None,
        nonlinearity=torch.nn.ReLU,
        use_maxpool=False,
        normlayer=None,  # if None, will not be used
    ):
        # 调用父类 nn.Module 的初始化。
        super().__init__()
        # 如果 nonlinearity 是字符串，则从 torch.nn 中获取对应的类。
        if isinstance(nonlinearity, str):
            nonlinearity = getattr(torch.nn, nonlinearity)
        # 解析输入图像形状。
        c, h, w = image_shape
        # 创建卷积模块，注意这里 normlayer 被硬编码为 None（原代码如此）。
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            nonlinearity=nonlinearity,
            use_maxpool=use_maxpool,
            normlayer=None,  # if None, will not be used
        )
        # 计算卷积后的展平尺寸。
        conv_out_size = self.conv.conv_out_size(h, w)
        # 如果指定了 hidden_sizes 或 output_size，则创建 MLP 头。
        if hidden_sizes or output_size:
            self.head = MlpModel(conv_out_size, hidden_sizes, output_size=output_size, nonlinearity=nonlinearity)
            # 记录最终输出维度。
            if output_size is not None:
                self._output_size = output_size
            else:
                # 如果 hidden_sizes 是整数，则输出维度为 hidden_sizes；否则为最后一个隐藏层大小。
                self._output_size = hidden_sizes if isinstance(hidden_sizes, int) else hidden_sizes[-1]
        # 否则不添加 MLP 头，使用恒等映射。
        else:
            self.head = lambda x: x
            self._output_size = conv_out_size

    # 前向传播：输入形状 [B, C, H, W]。
    # 先通过卷积模块，然后将输出展平为 [B, -1]，再通过 MLP 头。
    def forward(self, input):
        """Compute the convolution and fully connected head on the input;
        assumes correct input shape: [B,C,H,W]."""
        return self.head(self.conv(input).view(input.shape[0], -1))

    # 属性：返回最终输出大小。
    @property
    def output_size(self):
        """Returns the final output size after MLP head."""
        return self._output_size