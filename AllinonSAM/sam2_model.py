import torch
import torch.nn as nn
import clip
from prompt_adapted_segment_anything.modeling import TwoWayTransformer
from prompt_adapted_SAM2.modeling.backbones.image_encoder import ImageEncoder
from prompt_adapted_SAM2.modeling.backbones.hieradet import Hiera
from prompt_adapted_SAM2.modeling.backbones.image_encoder import FpnNeck
from prompt_adapted_SAM2.modeling.position_encoding import PositionEmbeddingSine
from prompt_adapted_SAM2.modeling.sam.prompt_encoder import PromptEncoder
from prompt_adapted_SAM2.modeling.sam.mask_decoder import MaskDecoder
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

class Prompt_Adapted_SAM2(nn.Module):
    def __init__(
        self,
        config,
        label_text_dict = {},
        training_strategy='biastuning'
    ):
        super().__init__()
        self.img_size = config['sam']['img_size']
        self.num_classes = config['sam']['num_classes']
        self.label_dict = label_text_dict
        self.prompt_config = config['prompts']
        self.im_type = config['img_type']
        self.use_fdn = config['use_fdn']
        self.training_strategy = training_strategy
        self.encoder_embed_dim= 144 # large SAM2
        self.encoder_num_heads= 2
        self.encoder_global_attn_indexes=[7, 15, 23, 31] if config['sam']['sam_type']=='huge' else [2, 5, 8, 11]
        # Hiera Specific
        self.stages = config["Hiera"]["stages"]
        self.window_pos_embed_bkg_spatial_size = config["Hiera"]["window_pos_embed_bkg_spatial_size"]
        self.window_spec = config["Hiera"]["window_spec"]
        self.backbone_channel_list = config["Neck"]["backbone_channel_list"]
        self.fpn_top_down_levels = config["Neck"]["fpn_top_down_levels"]
        prompt_embed_dim=256 # should match the the output dim of the neck embedding (it is )
        image_embedding_size=16 # need to check it should be self.img_size // backbone_stride
        mask_in_chans=16
        
        print(self.prompt_config)
        # THE BIG DIFFERENCE
        # 1. Image encoder with Hiera trunk and FPN neck (Need to be revised)
        self.image_encoder = ImageEncoder(
            trunk=Hiera(
                embed_dim=self.encoder_embed_dim,
                num_heads=self.encoder_num_heads,
                stages=self.stages,
                global_att_blocks=self.encoder_global_attn_indexes,
                window_pos_embed_bkg_spatial_size=self.window_pos_embed_bkg_spatial_size,
                window_spec=self.window_spec,
            ),
            neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(
                    num_pos_feats=256,
                    normalize=True,
                    temperature=10000
                ),
                d_model=256,# should match the prompt encoder embedding 
                backbone_channel_list=self.backbone_channel_list,
                fpn_top_down_levels=self.fpn_top_down_levels,
                fpn_interp_model='nearest'
            ),
            scalp=1
        )

        # 2. Text prompt components (optional) (Similar to SAM)
        self.clip_model, _ = clip.load("ViT-B/32")
        self.Text_Embedding_Affine = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

        # 3. Prompt encoder (Similar to SAM) 
        # same config as https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py
        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        # They do use the same encoder and Decoder !!!
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size), # should we take the output from the neck?
            input_image_size=(self.img_size, self.img_size),
            mask_in_chans=mask_in_chans
        )

        # 4. Mask decoder (Similar to SAM)
        self.mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        )
        sam_ckpt = '/home/abdelrahman.elsayed/sam2/checkpoints/sam2.1_hiera_base_plus.pt'
        sam_state_dict = torch.load(sam_ckpt)

        for k in list(sam_state_dict.keys()):
            if self.img_size!=1024:
                #pos embed can be loaded only when image size is 1024
                if "pos_embed" in k:
                    full_matrix = sam_state_dict.pop(k)

                    adapted_matrix = nn.functional.adaptive_avg_pool2d(full_matrix.permute(0,3,1,2), (self.sam_encoder.pos_embed.shape[1], self.sam_encoder.pos_embed.shape[2]))
                    adapted_matrix = adapted_matrix.permute(0,2,3,1)
                    sam_state_dict[k] = adapted_matrix

            if "image_encoder." in k:
                if "qkv" in k:
                    print(k)
                if 'image_encoder.neck' in k:
                    if '0' in k:
                        # print(k)
                        new_key = k.replace('0','conv1')
                    if '1' in k:
                        new_key = k.replace('1','ln1')
                    if '2' in k:
                        new_key = k.replace('2','conv2')
                    if '3' in k:
                        new_key = k.replace('3','ln2')
                    new_key = new_key[14:]
                    sam_state_dict[new_key] = sam_state_dict[k]
                    _ = sam_state_dict.pop(k)
                
                else:
                    sam_state_dict[k[14:]] = sam_state_dict.pop(k)


            if "prompt_encoder." in k:
                sam_state_dict[k[15:]] = sam_state_dict.pop(k)

            if "mask_decoder." in k:
                sam_state_dict[k[13:]] = sam_state_dict.pop(k)


        missing_keys, unexpected_keys = self.sam_encoder.load_state_dict(sam_state_dict,strict=False)
        print("For self.sam_encoder:")
        print("  Missing keys:   ", missing_keys)
        print("  Unexpected keys:", unexpected_keys)
        print()
        self.prompt_encoder.load_state_dict(sam_state_dict, strict=False)
        print("For self.sam_prompt_encoder:")
        print("  Missing keys:   ", missing_keys)
        print("  Unexpected keys:", unexpected_keys)
        print()
        self.mask_decoder.load_state_dict(sam_state_dict,strict=False)
        print("For self.mask_decoder:")
        print("  Missing keys:   ", missing_keys)
        print("  Unexpected keys:", unexpected_keys)
        print()


    def forward(self, x_img, x_text):
        B,C,H,W = x_img.shape
        x_text = list(x_text)
        
        # Image encoding
        encoder_output, reg_loss = self.image_encoder(x_img)
        image_embeddings = encoder_output["vision_features"]

        # Text encoding
        text_inputs = (clip.tokenize(x_text)).to(self.device)
        # with torch.no_grad():
        text_features = self.clip_model.encode_text(text_inputs)
        # Prompt encoding
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        
        try:
            text_features_affine= self.Text_Embedding_Affine(text_features.float())
        except:
            print(text_features.shape)
            1/0
        text_features_affine = text_features_affine.repeat(1,self.prompt_config['NUM_TEXT_REPEAT'],1)
        sparse_embeddings = sparse_embeddings.to(self.device).repeat(B,1,1)
        #Assuming NO USE_SLICE_NUM:
        sparse_embeddings = torch.cat(
                [sparse_embeddings, text_features_affine], dim=1)    
        # Mask decoding
        masks, ious, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return self._postprocess_masks(masks) , reg_loss

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
            (self.sam_encoder.img_size, self.sam_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        masks = torch.sigmoid(masks)
        return masks.squeeze(1)
