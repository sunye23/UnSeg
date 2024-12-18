# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision.transforms.transforms
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder, MaskDecoderNoise, MaskDecoderScratch
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        noise = None,
        multimask_output=False,
    ):
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"], noise=noise[i]) for i, x in enumerate(batched_input)], dim=0)
        
        image_embeddings, interm_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )
            
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold

            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "encoder_embedding": curr_embedding.unsqueeze(0),
                    "image_pe": self.prompt_encoder.get_dense_pe(),
                    "sparse_embeddings":sparse_embeddings,
                    "dense_embeddings":dense_embeddings,
                }
            )

        return outputs, interm_embeddings

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor, noise=None) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        if noise is not None:
            x = torch.clamp(x + noise, min=0, max=255)
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        return x


class ScratchSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoderScratch,
            pixel_mean: List[float] = [0.485, 0.456, 0.406],
            pixel_std: List[float] = [0.229, 0.224, 0.225],
            epsilon: float = 0.0,
            epsilon_bg: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.generator_epoch = 4
        self.epsilon = epsilon
        self.epsilon_bg = epsilon_bg
        print("Epsilon in ScratchSam: {}".format(self.epsilon))
        print("Epsilon_bg in ScratchSam: {}".format(self.epsilon_bg))
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
            self,
            batched_input: List[Dict[str, Any]],
            noise=None,
            epoch=None,
            noise_type='full',
            multimask_output=False,
    ):
        if noise is not None:
            input_images = torch.stack([self.preprocess(x["resize_image"], noise=noise[i], epoch=epoch, label=x['original_label'], noise_type=noise_type
                                                        ) for i, x in enumerate(batched_input)], dim=0)
        else:
            input_images = torch.stack([self.preprocess(x["resize_image"], noise=None, epoch=epoch) for i, x in enumerate(batched_input)], dim=0)

        image_embeddings, interm_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "resize_point_coords" in image_record:
                points = (image_record["resize_point_coords"], image_record["resize_point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("resize_boxes", None),
                masks=image_record.get("resize_mask_inputs", None),
            )
            low_res_masks = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output
            )
            outputs.append(low_res_masks)
        outputs = torch.cat(outputs, dim=0)
        return outputs
            # masks = self.postprocess_masks(
            #     low_res_masks,
            #     input_size=image_record["image"].shape[-2:],
            #     original_size=image_record["original_size"],
            # )
            # masks_norm = masks > self.mask_threshold
            #
            # outputs.append(
            #     {
            #         "masks": masks,
            #         "masks_norm": masks_norm,
            #         "low_res_logits": low_res_masks,
            #     }
            # )

        # return outputs, interm_embeddings

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, input: torch.Tensor, noise=None, epoch=None, label=None, noise_type=None) -> torch.Tensor:

        h, w = input.shape[-2:]
        if noise is not None:
            noise_h, noise_w = noise.shape[-2:]
            label_resize = F.interpolate(label, size=(self.image_encoder.img_size, self.image_encoder.img_size), mode='bilinear').squeeze(0) / 255.0
            label_resize = torch.round(label_resize)
            noise = F.interpolate(noise.unsqueeze(0), scale_factor=(self.image_encoder.img_size / noise_h,
                                                                    self.image_encoder.img_size / noise_w), mode='bilinear').squeeze(0)
            label_noise = label_resize.clone().detach()
            label_noise = label_noise * self.epsilon + (1.0 - label_noise) * self.epsilon_bg
            noise = noise * label_noise
            if epoch % self.generator_epoch != 0:
                x = input.clone().detach().requires_grad_(True)
                x = torch.clamp(x + noise, min=0.0, max=1.0)
            else:
                x = torch.clamp(input + noise.detach(), min=0.0, max=1.0)
            x = (x - self.pixel_mean) / self.pixel_std
            return x
        else:
            x = (input - self.pixel_mean) / self.pixel_std
            return x

class NoiseSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoderNoise,
            # pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_mean: List[float] = [0.485, 0.456, 0.406],
            # pixel_std: List[float] = [58.395, 57.12, 57.375],
            pixel_std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:

        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
            self,
            batched_input: List[Dict[str, Any]], post_process=False,
    ):
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings, interm_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            outputs.append(
                {
                    "encoder_embedding": curr_embedding.unsqueeze(0),
                    "image_pe": self.prompt_encoder.get_dense_pe(),
                    "sparse_embeddings": sparse_embeddings,
                    "dense_embeddings": dense_embeddings,
                }
            )

        batch_len = len(outputs)
        encoder_embedding = torch.cat([outputs[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
        image_pe = [outputs[i_l]['image_pe'] for i_l in range(batch_len)]
        sparse_embeddings = [outputs[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
        dense_embeddings = [outputs[i_l]['dense_embeddings'] for i_l in range(batch_len)]

        mask_noises = self.mask_decoder(
            image_embeddings=encoder_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            hq_token_only=True,
            interm_embeddings=interm_embeddings,
        )
        return mask_noises

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize color
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        # h, w = x.shape[-2:]
        # padh = self.image_encoder.img_size - h
        # padw = self.image_encoder.img_size - w
        # x = F.pad(x, (0, padw, 0, padh))
        return x