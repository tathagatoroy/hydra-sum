"""Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause"""

import torch
from transformers import PreTrainedModel, BartModel, BartConfig, BartPretrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutput
from torch import nn
import train_seq2seq_utils
from train_seq2seq_utils import skld_loss, cosine_similarity, cosine_similarity_on_features
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder
import torch.nn.functional as F
import copy
from generation_utils_multi_heads import GenerationMixinCustom



class BartModelMultHeads(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        self.decoder1 = BartDecoder(config, self.shared)

        self.num_decoder_layers_shared = None

        self.head_selector = nn.Linear(config.d_model, 2, bias=False) # this should be num of decoders
        self.feature_embedding = nn.Embedding(config.num_bins, config.d_model) # this scales the feature score to the same dimension as the decoder output
        self.feature_score = nn.Linear(config.d_model, 2, bias=True) # this should be num of decoders


        self.init_weights()

    # unchanged
    def get_input_embeddings(self):
        return self.shared

    # unchanged
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # unchanged
    def get_encoder(self):
        return self.encoder

    # unchanged
    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            use_mixed=True,
            use_head=0,
            overlap_bin = None,  # this is int with range(0,9) which denotes the bin the abstractivity
            
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        # but decoder id is ideally provid
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = train_seq2seq_utils.shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        decoder_args = {'input_ids': decoder_input_ids,
                        'attention_mask': decoder_attention_mask,
                        'encoder_hidden_states': encoder_outputs[0],
                        'encoder_attention_mask': attention_mask,
                        'head_mask': decoder_head_mask,
                        'cross_attn_head_mask': cross_attn_head_mask,
                        'past_key_values': past_key_values,
                        'inputs_embeds': decoder_inputs_embeds,
                        'use_cache': use_cache,
                        'output_attentions': output_attentions,
                        'output_hidden_states': True,
                        'return_dict': return_dict}

        if use_mixed:
            # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
            decoder_outputs = self.decoder(**decoder_args)
            decoder_outputs1 = self.decoder1(**decoder_args)




            decoder_layer_common_output = decoder_outputs.hidden_states[self.num_decoder_layers_shared] #B, L, D
            logits = self.head_selector(decoder_layer_common_output) # this (B, L, 2)
            # if overlap_bin is not None:
            #     overlap_embedding = self.feature_embedding(overlap_bin) #(Batch_Size, L)
            #     overlap_score = self.feature_score(overlap_embedding)   #(Batch_Size, 2)
            #     #unsqueeze to (Batch_Size, 1, 2)
            #     overlap_score = overlap_score.unsqueeze(1)
            #     logits = overlap_score * logits # (Batch_Size, L, 2)
            prob_head_selector = nn.functional.softmax(logits, dim=-1)

            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            ), Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs1.last_hidden_state,
                past_key_values=decoder_outputs1.past_key_values,
                decoder_hidden_states=decoder_outputs1.hidden_states,
                decoder_attentions=decoder_outputs1.attentions,
                cross_attentions=decoder_outputs1.cross_attentions,
                encoder_last_hidden_state=None,
                encoder_hidden_states=None,
                encoder_attentions=None,
            ), prob_head_selector

        else:
            if use_head == 0:
                # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
                decoder_outputs = self.decoder(**decoder_args)
            else:
                decoder_outputs = self.decoder1(**decoder_args)

            if not return_dict:
                print('NEEDS TO BE IMPLEMENTED: Generation_mutlhead_utils. Use return_dict')
                # raise not implemented error
                raise NotImplementedError
                exit()

            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )


class ConditionalGenerationCustomBartMultHeads(GenerationMixinCustom, BartPretrainedModel):
    base_model_prefix = "model"
    authorized_missing_keys = [r"final_logits_bias", r"encoder\.version", r"decoder\.version"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartModelMultHeads(config)
        self.model = base_model
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    def initialize_correct_weights(self, config: BartConfig, num_decoder_layers_shared=6):
        num_layers = config.decoder_layers
        if num_decoder_layers_shared > num_layers:
            print(f'setting common decoder layers to max layers = {num_layers}')

        self.model.decoder1 = copy.deepcopy(self.model.decoder)

        for k in range(num_decoder_layers_shared):
            _tie_decoder_weights(self.model.decoder.layers[k],
                                 self.model.decoder1.layers[k], f'decoder_layer{k}')

        self.model.num_decoder_layers_shared = num_decoder_layers_shared

    def freeze_weights(self):
        print("freezing weights")
        self.model.encoder.requires_grad_(False)
        for k in range(self.model.num_decoder_layers_shared):
            self.model.decoder.layers[k].requires_grad_(False)
            self.model.decoder1.layers[k].requires_grad_(False)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            lm_labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            generate=True,
            use_mixed=True,
            use_head=None,
            gate=None,
            use_gate_supervision=False,
            gate_prob=None,
            use_sentence_gate_supervision=False,
            sent_gate=None,
            overlap = None,
            overlap_bin = None, # this is int with range(0,9)
            divergence_loss=None,
            divergence_weight=None,
            use_overlap_supervision = False,
            use_distance_loss_pre_lm_layer = False,
            use_distance_loss_post_lm_layer = False,
            use_one_head_distance_loss = False,
            use_two_head_distance_loss = False,
            use_feature_level_gating = False,
            use_last_layer_gating = True,
            **unused,
    ):
        if "lm_labels" in unused:
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_args = {'input_ids': input_ids,
                      'attention_mask': attention_mask,
                      'decoder_input_ids': decoder_input_ids,
                      'encoder_outputs': encoder_outputs,
                      'decoder_attention_mask': decoder_attention_mask,
                      'past_key_values': past_key_values,
                      'use_cache': use_cache,
                      'output_attentions': output_attentions,
                      'output_hidden_states': output_hidden_states,
                      'return_dict': return_dict,
                      'use_mixed': use_mixed,
                      'use_head': use_head,
                      'overlap_bin': overlap_bin}

        if use_mixed:
            outputs, outputs1, prob_head_selector = self.model.forward(**input_args)
            #this F.Linear computes the logits from the decoder output using a linear layer which is the same as the nn.Embedding weight
            #print("shape of outputs : {0} ".format(outputs[0].shape))
            #output shape is (B, L, D)

            lm_logits0 = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
            lm_logits1 = F.linear(outputs1[0], self.model.shared.weight, bias=self.final_logits_bias)

            softmax_0 = F.softmax(lm_logits0, dim=-1)
            softmax_1 = F.softmax(lm_logits1, dim=-1)

            #apply kl divergence loss here 
            # unlikely to be well conditioned as maximizing is not clear, maybe something like max(0,cos(softmax_0, softmax_1)) is a better divergence loss





            if gate_prob is not None:
                softmax_0 = softmax_0 * gate_prob
                softmax_1 = softmax_1 * (1 - gate_prob)
            elif use_overlap_supervision:
                #assert that if overlap is none , give error 
                assert overlap is not None, 'overlap is none'
                #print("using overlap supervision")
                #convert shape (Batch) ot (Batch,1,1)
                overlap_gate = overlap.unsqueeze(1).unsqueeze(2)
                softmax_0 = softmax_0 * (1 - overlap_gate)
                softmax_1 = softmax_1 * overlap_gate
            elif use_gate_supervision:
                #assert that if overlap is none , give error 
                assert overlap is not None, 'overlap is none'
                #print("using gate supervision")
                prob_0 = 1 - overlap
                prob_1 = overlap
                softmax_0 = softmax_0 * prob_0
                softmax_1 = softmax_1 * prob_1

            elif use_sentence_gate_supervision:
                # softmax_0 = torch.mul(softmax_0, (1 - sent_gate).unsqueeze(1).unsqueeze(2))
                # softmax_1 = torch.mul(softmax_1, sent_gate.unsqueeze(1).unsqueeze(2))
                softmax_0 = torch.mul(softmax_0, (1 - sent_gate).unsqueeze(2))
                softmax_1 = torch.mul(softmax_1, sent_gate.unsqueeze(2))
                #print(sent_gate)
                #print(softmax_0)
                #print(softmax_1)
            else:
                #prob0 is (B, L, 2)
                prob0 = prob_head_selector[:, :, 0].unsqueeze(2) # (B, L, 1)
                prob1 = prob_head_selector[:, :, 1].unsqueeze(2) # (B, L, 1)
                softmax_0 = torch.mul(softmax_0, prob0)  # (B, L, V) as softmax is (B, L, V)
                softmax_1 = torch.mul(softmax_1, prob1) # (B, L, V) as softmax is (B, L, V)

            lm_logits = torch.log(F.relu(softmax_0 + softmax_1) + 1e-6)  # TODO: This is not logits, rename
        else:
            outputs = self.model.forward(**input_args)
            lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
            lm_logits = F.log_softmax(lm_logits, dim=-1)  # TODO: This is not logits, rename

        masked_lm_loss = None
        gate_loss = None
        if not generate:
                        
            if divergence_loss is not None:
                if use_distance_loss_post_lm_layer:
                    if divergence_loss == 'kl':
                        if use_one_head_distance_loss:
                            softmax_0_detached = softmax_0.detach()
                            divergence_loss = skld_loss(softmax_0_detached, softmax_1)
                        elif use_two_head_distance_loss:
                            divergence_loss = skld_loss(softmax_0, softmax_1)
                        #divergence_loss = skld_loss(softmax_0, softmax_1)
                    elif divergence_loss == 'cosine':
                        if use_one_head_distance_loss:
                            softmax_0_detached = softmax_0.detach()
                            divergence_loss = cosine_similarity(softmax_0_detached, softmax_1)
                        elif use_two_head_distance_loss:
                            divergence_loss = cosine_similarity(softmax_0, softmax_1)
                        #divergence_loss = cosine_similarity(softmax_0, softmax_1)
                    else:
                        print(f'loss {divergence_loss} not implemented')
                        raise NotImplementedError
                        exit()
                elif use_distance_loss_pre_lm_layer:
                    if divergence_loss == 'kl':
                        if use_one_head_distance_loss:
                            feature_1 = outputs[0]
                            feature_2 = outputs1[0]
                            feature_1_detached = feature_1.detach()
                            divergence_loss = skld_loss(feature_1_detached, feature_2)
                        elif use_two_head_distance_loss:
                            feature_1 = outputs[0]
                            feature_2 = outputs1[0]
                            divergence_loss = skld_loss(feature_1, feature_2)

                        #divergence_loss = skld_loss(softmax_0, softmax_1)
                    elif divergence_loss == 'cosine':
                        if use_one_head_distance_loss:
                            feature_1 = outputs[0]
                            feature_2 = outputs1[0]
                            feature_1_detached = feature_1.detach()
                            divergence_loss = cosine_similarity_on_features(feature_1_detached, feature_2)
                        elif use_two_head_distance_loss:
                            feature_1 = outputs[0]
                            feature_2 = outputs1[0]
                            divergence_loss = cosine_similarity_on_features(feature_1, feature_2)

                        #divergence_loss = cosine_similarity(softmax_0, softmax_1)
                    else:
                        print(f'loss {divergence_loss} not implemented')
                        raise NotImplementedError
                        exit()
            lm_labels = train_seq2seq_utils.shift_tokens_left(decoder_input_ids, 1)

            loss_fct = nn.NLLLoss(ignore_index=1)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            if use_mixed and divergence_loss is not None:
                total_loss = masked_lm_loss + divergence_weight * divergence_loss
            # I am not using gate supervision here
            if use_mixed and use_gate_supervision:
                #negative log likelyhood loss

                loss_fct_gate = nn.NLLLoss(ignore_index=-1)
                gate_loss = loss_fct_gate(torch.log(prob_head_selector.view(-1, 2)), gate.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        if use_mixed and divergence_loss is None:
            return_output = Seq2SeqLMOutput(
                            loss=masked_lm_loss,
                            logits=lm_logits,
                            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                            encoder_hidden_states=outputs.encoder_hidden_states,
                            encoder_attentions=outputs.encoder_attentions
                            )
        elif use_mixed and divergence_loss is not None:
            return_output = Seq2SeqLMOutput(
                            loss=total_loss,
                            logits=lm_logits,
                            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                            encoder_hidden_states=outputs.encoder_hidden_states,
                            encoder_attentions=outputs.encoder_attentions
                            )
                
            return return_output, divergence_loss, masked_lm_loss
        
        else:
            return_output = Seq2SeqLMOutput(
                            loss=masked_lm_loss,
                            logits=lm_logits,
                            past_key_values=outputs.past_key_values,
                            decoder_hidden_states=outputs.decoder_hidden_states,
                            decoder_attentions=outputs.decoder_attentions,
                            cross_attentions=outputs.cross_attentions,
                            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                            encoder_hidden_states=outputs.encoder_hidden_states,
                            encoder_attentions=outputs.encoder_attentions,
                            )
 
        #will not reach if use gate supervision and use mixed is true
        if use_gate_supervision:
            return return_output, gate_loss, prob_head_selector
        else:
            return return_output

    # unchanged
    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    # unchanged
    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    # unchanged
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    # unchanged
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    # unchanged
    def get_decoder(self):
        return self.model.get_decoder()

    # unchanged
    def get_encoder(self):
        return self.model.get_encoder()

    # unchanged
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return train_seq2seq_utils.shift_tokens_right(labels, self.config.pad_token_id,
                                                      self.config.decoder_start_token_id)


def _tie_decoder_weights(decoder1: nn.Module, decoder2: nn.Module, module_name: str):
    def tie_decoder_recursively(
            decoder1_pointer: nn.Module,
            decoder2_pointer: nn.Module,
            module_name: str,
            depth=0,
    ):
        assert isinstance(decoder1_pointer, nn.Module) and isinstance(
            decoder2_pointer, nn.Module
        ), f"{decoder1_pointer} and {decoder2_pointer} have to be of type nn.Module"
        if hasattr(decoder1_pointer, "weight"):
            assert hasattr(decoder2_pointer, "weight")
            decoder1_pointer.weight = decoder2_pointer.weight
            if hasattr(decoder1_pointer, "bias"):
                assert hasattr(decoder2_pointer, "bias")
                decoder1_pointer.bias = decoder2_pointer.bias
            return

        decoder1_modules = decoder1_pointer._modules
        decoder2_modules = decoder2_pointer._modules
        if len(decoder2_modules) > 0:
            assert (
                    len(decoder1_modules) > 0
            ), f"Decoder modules do not match"

            all_decoder_weights = set([module_name + "/" + sub_name for sub_name in decoder1_modules.keys()])
            for name, module in decoder2_modules.items():
                tie_decoder_recursively(
                    decoder1_modules[name],
                    decoder2_modules[name],
                    module_name + "/" + name,
                    depth=depth + 1,
                )
                all_decoder_weights.remove(module_name + "/" + name)

            assert len(all_decoder_weights) == 0, 'There are some extra parameters in one of the decoders'

    # tie weights recursively
    tie_decoder_recursively(decoder1, decoder2, module_name)
