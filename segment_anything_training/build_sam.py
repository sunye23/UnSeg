# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam,  TwoWayTransformer, NoiseSam, MaskDecoderNoise, \
    ScratchSam, MaskDecoderScratch

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[0.485, 0.456, 0.406],
        # pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[0.229, 0.224, 0.225],
        # pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


def _build_noise_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    epsilon=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = NoiseSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderNoise(model_type='vit_b', epsilon=epsilon),
        pixel_mean=[0.485, 0.456, 0.406],
        # pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[0.229, 0.224, 0.225],
        # pixel_std=[58.395, 57.12, 57.375],
    )
    backbone_path = './pretrained_checkpoint/sam_vit_b_backbone.pth'
    prompt_path = './pretrained_checkpoint/sam_vit_b_prompt_decoder.pth'
    if checkpoint is None:
        with open(backbone_path, "rb") as f:
            state_dict = torch.load(f)
        sam.image_encoder.load_state_dict(state_dict)
        for n, p in sam.image_encoder.named_parameters():
            p.requires_grad = False

        with open(prompt_path, "rb") as f:
            state_dict = torch.load(f)
        sam.prompt_encoder.load_state_dict(state_dict)
        for n, p in sam.prompt_encoder.named_parameters():
            p.requires_grad = False
        print("Noisy backbone and prompt Encoder is Frozen ......")
    else:
        print("Init noise generator weight from checkpoint: {}".format(checkpoint))
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
        print("noise sam weight has been loaded successfully...")
    return sam


def _build_scratch_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    epsilon=None,
    epsilon_bg=None,
):
    prompt_embed_dim = 256
    image_size = 512
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = ScratchSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderScratch(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[0.485, 0.456, 0.406],
        # pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[0.229, 0.224, 0.225],
        # pixel_std=[58.395, 57.12, 57.375],
        epsilon=epsilon,
        epsilon_bg=epsilon_bg,
    )
    checkpoint = './pretrained_checkpoint/modified_mae_pretrain_vit_base.pth'
    with open(checkpoint, "rb") as f:
        state_dict = torch.load(f)
    sam.image_encoder.load_state_dict(state_dict, strict=False)
    print("Scratch sam image encoder weight init from MAE pretrained ViT ......")
    return sam


def build_noise_sam_vit_b(checkpoint=None, epsilon=0.0627450980392157):
    return _build_noise_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        epsilon=epsilon,
    )



def build_sam_vit_b_scratch(checkpoint=None, epsilon=None, epsilon_bg=None):
    return _build_scratch_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        epsilon=epsilon,
        epsilon_bg=epsilon_bg,
    )


sam_model_registry = {
    "default": build_sam,
    "vit_h": build_sam,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

noise_sam_model_registry = {
    # "default": build_sam,
    # "vit_h": build_sam,
    # "vit_l": build_sam_vit_l,
    "vit_b": build_noise_sam_vit_b,
}

scratch_sam_model_registry = {
    # "default": build_sam,
    # "vit_h": build_sam,
    # "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b_scratch,
}