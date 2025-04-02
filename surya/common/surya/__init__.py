import warnings
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast

from surya.common.s3 import S3DownloaderMixin
from surya.common.surya.config import SuryaModelConfig
from surya.common.surya.decoder.__init__ import SuryaDecoderModel
from surya.common.surya.embedder.__init__ import SimpleTokenEmbedder
from surya.common.surya.encoder.__init__ import SuryaEncoderModel


@dataclass
class SuryaModelOutput(CausalLMOutputWithPast):
    bbox_logits: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None


class KwargsForCausalLM(FlashAttentionKwargs): ...


class SuryaModel(S3DownloaderMixin, PreTrainedModel):
    config_class = SuryaModelConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True
    main_input_name = "input_ids"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: SuryaModelConfig,
        embedder: SimpleTokenEmbedder = None,
        vision_encoder: SuryaEncoderModel = None,
        decoder: SuryaDecoderModel = None,
    ):
        super().__init__(config)

        if vision_encoder is None:
            vision_encoder = SuryaEncoderModel(
                config.vision_encoder
            )

        if decoder is None:
            decoder = SuryaDecoderModel(config.decoder)

        if embedder is None:
            embedder = SimpleTokenEmbedder(config)

        self.vision_encoder = vision_encoder
        self.decoder = decoder
        self.embedder = embedder

        # Tying configs
        self.vision_encoder.config = self.config.vision_encoder
        self.decoder.config = self.config.decoder

        if self.config.num_register_tokens > 0:
            self.register_token_embeds = nn.Parameter(torch.randn(1, self.config.num_register_tokens, self.decoder.hidden_size))
        else:
            self.register_token_embeds = None

        self.vision_projector = nn.Sequential(
            nn.Linear(self.vision_encoder.config.hidden_size, self.decoder.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size),
        )

        self.bbox_head = nn.Linear(config.hidden_size, 6)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def tie_weights(self):
        self._tie_weights()

    def _tie_weights(self):
        # Tie weights of lm head and token embedder
        self._tie_or_clone_weights(self.lm_head, self.embedder.token_embed)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.embedder.token_embed

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.embedder.token_embed = new_embeddings

    def get_image_embeddings(self, image_tiles, batch_size: int):
        # embed all images with the vision encoder after they have already been tiled and flattened into a single batch
        all_image_features = None
        for i in range(0, len(image_tiles), batch_size):
            image_batch = image_tiles[i:i + batch_size]
            image_features = self.vision_projector(
                self.vision_encoder.embed_images(image_batch=image_batch)
            )
            if i == 0:
                all_image_features = image_features
            else:
                all_image_features = torch.cat([all_image_features, image_features], dim=0)
        return all_image_features

    def embed_ids_boxes_images(self, input_ids, input_boxes, image_tiles):
        """
        Insert embedded image tiles into the corresponding positions into the full input sequence

        Positions to insert new tokens are indicated by the special image token index
        """
        inputs_embeds = self.embedder.embed(
            input_tokens=input_ids, input_bboxes=input_boxes
        )
        if image_tiles is not None:
            image_features = self.get_image_embeddings(image_tiles=image_tiles, batch_size=len(input_ids))

            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_tokens = torch.sum((input_ids == self.config.image_token_id))
                n_image_features = image_features.shape[0] * image_features.shape[1]
                warnings.warn(
                    f"Image features and image tokens do not match: tokens {n_image_tokens}, features {n_image_features}. This may lead to unexpected results"
                )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )
        else:
            assert (input_ids == self.config.image_token_id).sum() == 0, (
                "Image tokens were present in the input but no input images were provided"
            )

        if self.config.num_register_tokens > 0:
            expanded_register_tokens = self.register_token_embeds.expand(inputs_embeds.shape[0], 1, 1)
            special_register_mask = (input_ids == self.config.register_token_id).unsqueeze(-1)
            special_register_mask = special_register_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )

            total_mismatch = (
                inputs_embeds[special_register_mask].numel() != expanded_register_tokens.numel()
            )
            if total_mismatch:
                n_register_tokens = torch.sum((input_ids == self.config.register_token_id))
                n_register_features = expanded_register_tokens.shape[0] * expanded_register_tokens.shape[1]
                print(
                    f"register features and register tokens do not match: tokens {n_register_tokens}, features {n_register_features}. This may lead to unexpected results"
                )
                print(
                    f"Skipping register embedding for {n_register_tokens} tokens due to mismatch."
                )
                print(
                    f"Input embeds shape is {inputs_embeds.shape}, register features shape is {expanded_register_tokens.shape}"
                )
                register_tokens_by_row = torch.sum(
                    (input_ids == self.config.register_token_id), dim=1
                )
                print(f"Register tokens by row: {register_tokens_by_row}")

                skipped = True

                # Insert pad tokens instead of register tokens
                pad_embeds = self.embedder.embed(
                    input_tokens=torch.full_like(input_ids, self.config.pad_token_id),
                    input_bboxes=input_boxes,
                )
                # Only replace at the register token positions
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_register_mask, pad_embeds[special_register_mask]
                )
            else:
                expanded_register_tokens = expanded_register_tokens.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_register_mask, expanded_register_tokens
                )

        return inputs_embeds

    def forward(
        self,
        input_ids=None,
        input_boxes=None,
        image_tiles=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_hidden_states=False,
        output_attentions=False,
        use_cache=False,
        logits_to_keep=None,
        cache_position=None,
        **kwargs: KwargsForCausalLM,
    ):
        # Process the mixed batch if provided
        if inputs_embeds is None:
            inputs_embeds = self.embed_ids_boxes_images(
                input_ids, input_boxes, image_tiles
            )

        # Special image attention mask during prefill, ignored during decoding
        if self.config.unmask_image and inputs_embeds.shape[1] != 1:
            if cache_position is None:
                past_seen_tokens = (
                    past_key_values.get_seq_length() if past_key_values is not None else 0
                )
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
            causal_mask = self.decoder._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                output_attentions
            )
            expanded_image_token_mask = (input_ids == self.config.image_token_id)[:, None, None, :]
            expanded_register_token_mask = (input_ids == self.config.register_token_id)[:, None, None, :]
            unmasked_position_mask = torch.logical_or(expanded_image_token_mask, expanded_register_token_mask)
            
            # Causal mask has 0s for unmasked positions, and -inf for masked positions
            # Image positions are causally masked by default - We unmask by setting these positions to 0 (from -inf)
            causal_mask.masked_fill_(unmasked_position_mask, 0)
            attention_mask = causal_mask

        outputs = self.decoder(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only keep the last `logits_to_keep` logits, should bring down memory usage during inference
        if logits_to_keep is not None:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        bbox_logits = F.sigmoid(self.bbox_head(hidden_states))
        lm_logits = self.lm_head(hidden_states)

        return SuryaModelOutput(
            bbox_logits=bbox_logits,
            lm_logits=lm_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            past_key_values=outputs.past_key_values,
        )