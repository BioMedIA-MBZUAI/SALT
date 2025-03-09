from prompt_adapted_segment_anything.modeling.image_encoder import ImageEncoderViT
from prompt_adapted_segment_anything.modeling.mask_decoder import MaskDecoder
from prompt_adapted_segment_anything.modeling.prompt_encoder import PromptEncoder
from prompt_adapted_segment_anything.modeling import TwoWayTransformer
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple
import clip
from functools import partial, reduce
from operator import mul
import math
from typing import Union, List
from prompt_adapted_segment_anything.modeling import TwoWayTransformer
from prompt_adapted_SAM2.modeling.backbones.image_encoder import ImageEncoder
from prompt_adapted_SAM2.modeling.backbones.hieradet import Hiera
from prompt_adapted_SAM2.modeling.backbones.image_encoder import FpnNeck
from prompt_adapted_SAM2.modeling.position_encoding import PositionEmbeddingSine
from prompt_adapted_SAM2.modeling.sam.prompt_encoder import PromptEncoder
from prompt_adapted_SAM2.modeling.sam.mask_decoder import MaskDecoder

class Prompt_Adapted_SAM(nn.Module):
    def __init__(
        self, 
        config, 
        label_text_dict = {},
        device = 'cuda:0',
        training_strategy='biastuning'
        ):
        super().__init__()
        self.device = device
        self.img_size = config['sam']['img_size']
        self.num_classes = config['sam']['num_classes']
        self.label_dict = label_text_dict
        self.prompt_config = config['prompts']
        self.im_type = config['img_type']
        self.use_fdn = config['use_fdn']
        self.training_strategy = training_strategy
        self.encoder_embed_dim= 1280 if config['sam']['sam_type']=='huge' else 768
        self.encoder_depth=32 if config['sam']['sam_type']=='huge' else 12
        self.encoder_num_heads=16 if config['sam']['sam_type']=='huge' else 12
        self.encoder_global_attn_indexes=[7, 15, 23, 31] if config['sam']['sam_type']=='huge' else [2, 5, 8, 11]

        #define hyperparameters, can be taken to a config later
        prompt_embed_dim=256
        image_embedding_size=16
        mask_in_chans=16

        print(self.prompt_config)
        #define pretrained clip and sam models
        self.sam_encoder = ImageEncoderViT(img_size=self.img_size,prompt_config=self.prompt_config, mlp_transform=config['mlp_transform'], use_lora=config['use_lora'], embed_dim=self.encoder_embed_dim, depth=self.encoder_depth, num_heads=self.encoder_num_heads, global_attn_indexes=self.encoder_global_attn_indexes)
        self.clip_model, _  = clip.load("ViT-B/32", device=device)

        #define the components of sam
        self.prompt_encoder=PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(self.img_size, self.img_size),
        mask_in_chans=mask_in_chans,
        )

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

        
        #define text prompt layers if they are to be used
        if self.prompt_config['USE_TEXT_PROMPT']:
            if self.prompt_config['USE_SLICE_NUM']:
                self.Text_Embedding_Affine = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128)
                )
            else:
                self.Text_Embedding_Affine = nn.Sequential(
                    nn.Linear(512, 256), 
                    nn.ReLU(),
                    nn.BatchNorm1d(256)
                )
            if self.training_strategy=='prompttuning':
                self.text_prompt_dropout = nn.Dropout(self.prompt_config['DROPOUT'])
                self.text_prompt_embeddings = nn.Parameter(torch.zeros(self.num_classes+1, prompt_embed_dim))
                nn.init.xavier_uniform_(self.text_prompt_embeddings.data)

                self.label_dict = self.label_dict.update({
                                        'other': self.num_classes
                                    })

        #define the slice number embedding
        if self.prompt_config['USE_SLICE_NUM']:
            self.slice_embedding = nn.Embedding(1024,128)

        #initialize sam with pretrained weights
        sam_ckpt = '/home/abdelrahman.elsayed/med-cvpr/AllinonSAM/weights.pth'
        # sam_ckpt = '/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/segment-anything/checkpoints/sam_vit_h_4b8939.pth'
        # sam_ckpt = '/mnt/store/jparanj1/sam_vit_b_01ec64.pth'
        sam_state_dict = torch.load(sam_ckpt)

        #for medsam analysis
        # sam_ckpt = '/media/ubuntu/New Volume/jay/medsam_vit_b.pth'
        # sam_state_dict = torch.load(sam_ckpt)


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

        self.mask_decoder.load_state_dict(sam_state_dict,strict=False)

    def forward(self, x_img, x_text, slice_num=0):
        B, C, H, W = x_img.shape
        x_text = list(x_text)
        
        if self.prompt_config['USE_TEXT_PROMPT']:
            if self.training_strategy=='prompttuning':
                prompt_text = []
                for t in x_text:
                    try:
                        prompt_text.append(self.text_prompt_embeddings[self.label_dict[t]])
                    except:
                        prompt_text.append(self.text_prompt_embeddings[-1])
                prompt_text = torch.stack(prompt_text)
        
        image_embeddings, reg_loss = self.sam_encoder(x_img)
        if self.use_fdn:
            image_embeddings = self.FDN_branch(image_embeddings, x_img)

        text_inputs = (clip.tokenize(x_text)).to(self.device)
        # with torch.no_grad():
        text_features = self.clip_model.encode_text(text_inputs)
            # text_features = text_features.unsqueeze(1)
        # print(text_features.shape)


        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

        # print(sparse_embeddings.shape)
        try:
            if self.prompt_config['USE_TEXT_PROMPT']:
                text_features_affine = self.Text_Embedding_Affine(text_features.float())
            else:
                text_features_affine = text_features[:,:256]
        except:
            print(text_features.shape)
            1/0

        if self.prompt_config['USE_SLICE_NUM']:
            # print("slice num: ", slice_num)
            slice_features = self.slice_embedding(torch.LongTensor(slice_num).to(self.device))
            slice_features = slice_features.unsqueeze(1)
        if self.prompt_config['USE_TEXT_PROMPT'] and self.training_strategy=='prompttuning':
            text_features_affine = text_features_affine + prompt_text
        text_features_affine = text_features_affine.unsqueeze(1)
        text_features_affine = text_features_affine.repeat(1,self.prompt_config['NUM_TEXT_REPEAT'],1)
        sparse_embeddings = sparse_embeddings.to(self.device).repeat(B,1,1)
        if self.prompt_config['USE_SLICE_NUM']:
            # print(sparse_embeddings.shape)
            # print(text_features_affine.shape)
            # print(slice_features.shape)
            sparse_embeddings = torch.cat(
                [sparse_embeddings, torch.cat([text_features_affine, slice_features], dim=-1)], dim=1)
        else:
            sparse_embeddings = torch.cat(
                [sparse_embeddings, text_features_affine], dim=1)    
        

        # print("sparse embedding shape: ", sparse_embeddings.shape)
        # sparse_embeddings = sparse_embeddings.squeeze()
        # sparse_embeddings = sparse_embeddings.unsqueeze(1)

        low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                use_gsam = False
            )
        high_res_masks = self.postprocess_masks(low_res_masks, (self.img_size,self.img_size), (self.img_size,self.img_size))
        return high_res_masks, reg_loss

    def get_image_embeddings(self, x_img):
        with torch.no_grad():
            B, C, H, W = x_img.shape
            image_embeddings,_ = self.sam_encoder(x_img)
            if self.use_fdn:
                image_embeddings = self.FDN_branch(image_embeddings, x_img)
            return image_embeddings

    def get_masks_with_manual_prompts(self, img_embeds, points=None, boxes=None, masks=None):
        B = img_embeds.shape[0]
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=masks,
            )
        # print("sparse embeddings shape: ", sparse_embeddings.shape)
        low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=img_embeds,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    use_gsam = False
                )
        high_res_masks = self.postprocess_masks(low_res_masks, (self.img_size,self.img_size), (self.img_size,self.img_size))
        return high_res_masks
        



    def get_masks_for_multiple_labels(self, img_embeds, x_text):
        '''
        img_embeds - image embeddings obtained from get_imgae_embeddings function
        xtext - text prompts. image encoder wont be run and only the decoder will be run for each of these
        '''
        B = img_embeds.shape[0]
        with torch.no_grad():
            x_text = list(x_text)
            if self.prompt_config['USE_TEXT_PROMPT']:
                if self.training_strategy=='prompttuning':
                    prompt_text = []
                    for t in x_text:
                        try:
                            prompt_text.append(self.text_prompt_embeddings[self.label_dict[t]])
                        except:
                            prompt_text.append(self.text_prompt_embeddings[-1])
                    prompt_text = torch.stack(prompt_text)

            text_inputs = (clip.tokenize(x_text)).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )

            if self.prompt_config['USE_TEXT_PROMPT']:
                text_features_affine = self.Text_Embedding_Affine(text_features.float())
            else:
                text_features_affine = text_features[:,:256]

            if self.prompt_config['USE_TEXT_PROMPT'] and self.training_strategy=='prompttuning':
                text_features_affine = text_features_affine + prompt_text
            
            text_features_affine = text_features_affine.unsqueeze(1)
            sparse_embeddings = sparse_embeddings.to(self.device).repeat(B,1,1)
            sparse_embeddings = torch.cat(
                [sparse_embeddings,text_features_affine], dim=1)

            low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=img_embeds,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    use_gsam = False
                )
            high_res_masks = self.postprocess_masks(low_res_masks, (self.img_size,self.img_size), (self.img_size,self.img_size))
            return high_res_masks


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



class Prompt_Adapted_SAM2(nn.Module):
    def __init__(
        self,
        config,
        label_text_dict = {},
        device = 'cuda:0',
        training_strategy='biastuning'
    ):
        super().__init__()
        self.device = device
        self.img_size = config['sam']['img_size']
        self.num_classes = config['sam']['num_classes']
        self.label_dict = label_text_dict
        self.prompt_config = config['prompts']
        self.im_type = config['img_type']
        self.use_fdn = config['use_fdn']
        self.training_strategy = training_strategy
        # Hiera Specific
        self.type = config["sam"]["sam_type"]
        if self.type == "large":
            self.encoder_embed_dim= 144 # large SAM2
            self.encoder_num_heads= 2
            self.encoder_global_attn_indexes= config["Hiera"]["global_att_blocks"]
            self.stages = config["Hiera"]["stages"]
            self.window_pos_embed_bkg_spatial_size = config["Hiera"]["window_pos_embed_bkg_spatial_size"]
            self.window_spec = config["Hiera"]["window_spec"]
            self.backbone_channel_list = config["Hiera"]["backbone_channel_list"]
            self.fpn_top_down_levels = config["Hiera"]["fpn_top_down_levels"]
        else:
            self.encoder_embed_dim= 112 # Base SAM2
            self.encoder_num_heads= 2
            self.backbone_channel_list = config["Hiera"]["backbone_channel_list"]
            self.fpn_top_down_levels = config["Hiera"]["fpn_top_down_levels"]
        prompt_embed_dim=256 # should match the the output dim of the neck embedding (it is )
        if self.img_size == 1024:
            image_embedding_size=64# need to check it should be self.img_size // backbone_stride (FOR 1024)
        else:
            image_embedding_size=32
        mask_in_chans=16
        
        print(self.prompt_config)
        # THE BIG DIFFERENCE
        # 1. Image encoder with Hiera trunk and FPN neck (Need to be revised)
        if self.type == "large":
            trunk =Hiera(
                embed_dim=self.encoder_embed_dim,
                num_heads=self.encoder_num_heads,
                stages=self.stages,
                global_att_blocks=self.encoder_global_attn_indexes,
                window_pos_embed_bkg_spatial_size=self.window_pos_embed_bkg_spatial_size,
                window_spec=self.window_spec,
            )
        else:
            trunk =Hiera(
                embed_dim=self.encoder_embed_dim,
                num_heads=self.encoder_num_heads,
            )
        self.image_encoder = ImageEncoder(
            trunk=trunk,
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
            nn.Linear(512, 256), # imgsize
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
        if self.type == "large":
            sam_ckpt = '/home/abdelrahman.elsayed/sam2/checkpoints/sam2.1_hiera_large.pt'
        else:
            sam_ckpt = '/home/abdelrahman.elsayed/sam2/checkpoints/sam2.1_hiera_base_plus.pt'
        pretrained_dict = torch.load(sam_ckpt)["model"]
        model_dict = self.state_dict()
    
        # 1. Filter out unnecessary memory-related keys
        filtered_pretrained = {k: v for k, v in pretrained_dict.items() 
                            if not k.startswith(('memory_attention', 'memory_encoder', 'maskmem'))}

        
        aligned_dict = {}
        for pretrained_key, tensor in filtered_pretrained.items():
            # handle non-standard image size
            # Inside __init__() when loading SAM2 weights:
            if self.img_size != 1024:  # Only adapt if using non-default size
                # Adapt global pos_embed
                hiera_pos_embed = pretrained_dict['image_encoder.trunk.pos_embed']
                target_shape = self.image_encoder.trunk.pos_embed.shape[1:3]  # (H, W)
                adapted_pos_embed = F.adaptive_avg_pool2d(
                    hiera_pos_embed.permute(0, 3, 1, 2),  # (1, H, W, C) → (1, C, H, W)
                    target_shape
                ).permute(0, 2, 3, 1)  # (1, C, H, W) → (1, H, W, C)
                pretrained_dict['image_encoder.trunk.pos_embed'] = adapted_pos_embed

                # Adapt window pos_embed
                window_pos_embed = pretrained_dict['image_encoder.trunk.pos_embed_window']
                target_window_shape = self.image_encoder.trunk.pos_embed_window.shape[1:3]
                adapted_window_embed = F.adaptive_avg_pool2d(
                    window_pos_embed.permute(0, 3, 1, 2), 
                    target_window_shape
                ).permute(0, 2, 3, 1)
                pretrained_dict['image_encoder.trunk.pos_embed_window'] = adapted_window_embed
            # Handle key prefixes
            if 'image_encoder.trunk' in pretrained_key:
                new_key = pretrained_key.replace('model.', '')
            elif 'image_encoder.neck' in pretrained_key:
                new_key = pretrained_key.replace('model.', '')
                # print(new_key)
            elif 'sam_prompt_encoder' in pretrained_key:
                new_key = pretrained_key.replace('sam_prompt_encoder', 'prompt_encoder')
            elif 'sam_mask_decoder' in pretrained_key:
                new_key = pretrained_key.replace('sam_mask_decoder', 'mask_decoder')
                if 'transformer.layers.1.mlp.layers.0' in new_key:
                    new_key = new_key.replace("transformer.layers.1.mlp.layers.0", "transformer.layers.1.mlp.lin1")
                if 'transformer.layers.1.mlp.layers.1' in new_key:
                    new_key = new_key.replace("transformer.layers.1.mlp.layers.1", "transformer.layers.1.mlp.lin2")
                if 'transformer.layers.0.mlp.layers.0' in new_key:
                    new_key = new_key.replace("transformer.layers.0.mlp.layers.0", "transformer.layers.0.mlp.lin1")
                if 'transformer.layers.0.mlp.layers.1' in new_key:
                    new_key = new_key.replace("transformer.layers.0.mlp.layers.1", "transformer.layers.0.mlp.lin2")     
            else:
                continue
                
            if new_key in model_dict:
                aligned_dict[new_key] = tensor
                
        matched_dict = {k: v for k, v in aligned_dict.items() if k in model_dict}
        
        model_dict.update(matched_dict)
        self.load_state_dict(model_dict, strict=False)
        
        print(f"Loaded {len(matched_dict)}/{len(filtered_pretrained)} pretrained parameters")
        print(f"Missing keys: {[k for k in model_dict if k not in matched_dict]}")

       
    def forward(self, x_img, x_text,slice_num=0):
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
            if self.prompt_config['USE_TEXT_PROMPT']:
                text_features_affine = self.Text_Embedding_Affine(text_features.float())
            else:
                text_features_affine = text_features[:,:256]
        except:
            print(text_features.shape)
            1/0

        if self.prompt_config['USE_SLICE_NUM']:
            # print("slice num: ", slice_num)
            slice_features = self.slice_embedding(torch.LongTensor(slice_num).to(self.device))
            slice_features = slice_features.unsqueeze(1)
        if self.prompt_config['USE_TEXT_PROMPT'] and self.training_strategy=='prompttuning':
            text_features_affine = text_features_affine + prompt_text
        text_features_affine = text_features_affine.unsqueeze(1)
        text_features_affine = text_features_affine.repeat(1,self.prompt_config['NUM_TEXT_REPEAT'],1)
        sparse_embeddings = sparse_embeddings.to(self.device).repeat(B,1,1)
        if self.prompt_config['USE_SLICE_NUM']:
            # print(sparse_embeddings.shape)
            # print(text_features_affine.shape)
            # print(slice_features.shape)
            sparse_embeddings = torch.cat(
                [sparse_embeddings, torch.cat([text_features_affine, slice_features], dim=-1)], dim=1)
        else:
            sparse_embeddings = torch.cat(
                [sparse_embeddings, text_features_affine], dim=1)    
        
        # print("text_features_affine shape:", text_features_affine.shape)
        # print("sparse_embeddings shape:", sparse_embeddings.shape)
        sparse_embeddings = torch.cat(
                [sparse_embeddings, text_features_affine], dim=1)    
        # Mask decoding
        masks, ious, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
        )

        return self.postprocess_masks(masks , (self.img_size,self.img_size), (self.img_size,self.img_size)) , reg_loss

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
            (self.img_size, self.img_size), # keep for now
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        masks = torch.sigmoid(masks)
        return masks.squeeze(1)
