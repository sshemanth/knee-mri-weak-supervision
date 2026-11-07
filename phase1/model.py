import torch
import torch.nn as nn


class SliceEncoder(nn.Module):
    """Pretrained backbone (first conv adapted for MRI slices) when torchvision is
    available; falls back to a small from-scratch CNN otherwise if no weights are found.
    backbone_type: None (auto: resnet18 if pretrained else scratch) or 'radimagenet_resnet50'
    to use the RadImageNet-pretrained ResNet50 (domain-appropriate radiology pretraining).
    in_channels: 1 (averages the pretrained 3-channel conv1 down to 1 channel,
    for single-slice-per-channel input) or 3 (2.5D -- uses RadImageNet's original
    3-channel conv1 unmodified, for 3-adjacent-slices-as-channels input)."""

    def __init__(self, out_dim=256, pretrained=True, backbone_type=None, in_channels=1):
        super().__init__()
        self.backbone_type = 'scratch'
        self.in_channels = in_channels

        if backbone_type == 'radimagenet_resnet50':
            try:
                import torchvision.models as tvm
                from huggingface_hub import hf_hub_download
                ckpt_path = hf_hub_download(repo_id='Lab-Rasool/RadImageNet', filename='ResNet50.pt',
                                             local_files_only=True)
                state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                m = tvm.resnet50(weights=None)
                backbone_seq = nn.Sequential(*list(m.children())[:-1])  # conv1,bn1,relu,maxpool,layer1-4,avgpool
                wrapper = nn.Module()
                wrapper.backbone = backbone_seq
                wrapper.load_state_dict(state, strict=True)
                backbone_seq = wrapper.backbone

                if in_channels == 3:
                    # RadImageNet's pretrained conv1 is already 3-channel -- use it as-is,
                    # more faithful to the pretraining than averaging down to 1 channel.
                    pass
                elif in_channels == 1:
                    old_conv = backbone_seq[0]
                    new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                          stride=old_conv.stride, padding=old_conv.padding, bias=False)
                    with torch.no_grad():
                        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                    backbone_seq[0] = new_conv
                else:
                    raise ValueError(f"unsupported in_channels={in_channels} for radimagenet_resnet50")

                self.backbone = backbone_seq
                self.proj = nn.Linear(2048, out_dim)
                self.backbone_type = 'radimagenet_resnet50'
            except Exception as e:
                print(f"[SliceEncoder] radimagenet_resnet50 unavailable ({e}); falling back")

        if self.backbone_type == 'scratch' and pretrained:
            try:
                import torchvision.models as tvm
                backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
                old_conv = backbone.conv1
                backbone.conv1 = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                            stride=old_conv.stride, padding=old_conv.padding, bias=False)
                with torch.no_grad():
                    backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                backbone.fc = nn.Identity()
                self.backbone = backbone
                self.proj = nn.Linear(512, out_dim)
                self.backbone_type = 'resnet18'
            except Exception as e:
                print(f"[SliceEncoder] pretrained backbone unavailable ({e}); falling back to scratch CNN")

        if self.backbone_type == 'scratch':
            def block(cin, cout, stride=2):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                )

            self.net = nn.Sequential(
                block(1, 32), block(32, 64), block(64, 128), block(128, 256), block(256, 256),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(256, out_dim)

    def forward(self, x):  # x: [N,C,H,W]
        if self.backbone_type in ('resnet18', 'radimagenet_resnet50'):
            feat = self.backbone(x)
            feat = feat.flatten(1)
        else:
            feat = self.net(x).flatten(1)
        return self.proj(feat)  # [N,out_dim]


class AttnPool(nn.Module):
    """Attention pooling: learns a weight per item and returns the weighted mean."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x):  # x: [N,dim]
        w = torch.softmax(self.attn(x), dim=0)
        return (w * x).sum(dim=0)  # [dim]


class SeriesMetaEncoder(nn.Module):
    """Encodes a series' (Anatomical_Plane, Fluid_Sensitive, Fat_Suppression) metadata
    from train_series.csv into a small embedding
    fused with that series' pooled image features before study-level pooling. Plane is
    an index (0=Sagittal,1=Coronal,2=Axial,3=Unknown); fluid/fat are 0/1 (0.5=unknown)."""

    def __init__(self, out_dim=32, n_planes=4, plane_embed_dim=16):
        super().__init__()
        self.plane_embed = nn.Embedding(n_planes, plane_embed_dim)
        self.mlp = nn.Sequential(nn.Linear(plane_embed_dim + 2, out_dim), nn.ReLU())

    def forward(self, meta_vec):  # meta_vec: [3] = (plane_idx, fluid, fat)
        plane_idx = meta_vec[0].long().clamp(min=0, max=self.plane_embed.num_embeddings - 1)
        plane_e = self.plane_embed(plane_idx)  # [plane_embed_dim]
        x = torch.cat([plane_e, meta_vec[1:]], dim=0)
        return self.mlp(x)  # [out_dim]


class KneeMRIModel(nn.Module):
    """slice CNN -> attention pool over slices (per series) [+ optional series-metadata
    fusion] -> attention pool over series (per study) -> MLP head -> 12 finding logits."""

    def __init__(self, n_findings=12, embed_dim=256, pretrained=True, backbone_type=None,
                 in_channels=1, use_series_meta=False, meta_dim=32):
        super().__init__()
        self.slice_encoder = SliceEncoder(embed_dim, pretrained=pretrained, backbone_type=backbone_type,
                                           in_channels=in_channels)
        self.slice_pool = AttnPool(embed_dim)
        self.series_pool = AttnPool(embed_dim)
        self.embed_dim = embed_dim
        self.use_series_meta = use_series_meta
        if use_series_meta:
            self.series_meta_encoder = SeriesMetaEncoder(out_dim=meta_dim)
            # Residual fusion with zero initialisation: the layer starts as a no-op,
            # so the model initially matches the version without metadata.
            self.fuse = nn.Linear(embed_dim + meta_dim, embed_dim)
            nn.init.zeros_(self.fuse.weight)
            nn.init.zeros_(self.fuse.bias)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(embed_dim, n_findings),
        )

    def forward_study(self, series_list, meta_list=None):
        series_embeds = []
        for i, series in enumerate(series_list):
            if series.numel() == 0:
                continue
            x = series.unsqueeze(1) if series.dim() == 3 else series  # [S,1,H,W] 1-channel or [S,C,H,W] 2.5D
            slice_feats = self.slice_encoder(x)  # [S,embed_dim]
            series_feat = self.slice_pool(slice_feats)  # [embed_dim]
            if self.use_series_meta:
                assert meta_list is not None, "use_series_meta=True but no meta_list passed"
                meta_embed = self.series_meta_encoder(meta_list[i])  # [meta_dim]
                series_feat = series_feat + self.fuse(torch.cat([series_feat, meta_embed], dim=0))
            series_embeds.append(series_feat)
        if not series_embeds:
            return torch.zeros(self.embed_dim, device=next(self.parameters()).device)
        stacked = torch.stack(series_embeds)  # [n_series, embed_dim]
        return self.series_pool(stacked)

    def forward(self, batch_series_list, batch_meta_list=None):
        if batch_meta_list is None:
            embeds = torch.stack([self.forward_study(s) for s in batch_series_list])
        else:
            embeds = torch.stack([self.forward_study(s, m) for s, m in zip(batch_series_list, batch_meta_list)])
        return self.head(embeds)  # [batch, n_findings]
