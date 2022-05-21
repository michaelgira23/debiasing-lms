import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2DoubleHeadsModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from types import MethodType

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def get_model(device='cpu', gpt2_name='gpt2', in_net=False, in_net_init_identity=True, out_net=False, out_net_init_identity=True, freeze_ln=False, freeze_pos=True,
              freeze_wte=True, freeze_ff=True, freeze_attn=True, dup_lm_head=False, dup_lm_head_bias=False):

    # ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    model = GPT2LMHeadModel.from_pretrained(gpt2_name).to(device)
    # model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

    """
    Initialize linear input layer
    """

    in_layer_sizes = []
    out_layer_sizes = []
    input_dim = model.config.n_embd
    dropout = 0.1
    orth_gain = 1.41
    # orth_gain = None
    in_net_init_identity = True

    #Model - in_net
    if in_net:
        in_layers = []
        last_output_size = input_dim

        for size in in_layer_sizes:
            layer = nn.Linear(last_output_size, size)
            if orth_gain is not None:
                torch.nn.init.orthogonal_(layer.weight, gain=orth_gain)
            layer.bias.data.zero_()

            in_layers.append(layer)
            in_layers.append(nn.ReLU())
            in_layers.append(nn.Dropout(dropout))
            last_output_size = size

        in_final_linear = nn.Linear(last_output_size, model.config.n_embd)
        # if orth_gain is not None:
        #     torch.nn.init.orthogonal_(in_final_linear.weight, gain=orth_gain)
        # in_final_linear.bias.data.zero_()

        # Initialize final_linear layer to identity transformation
        if in_net_init_identity:
            nn.init.eye_(in_final_linear.weight)
            in_final_linear.bias.data.zero_()

        in_layers.append(in_final_linear)
        in_layers.append(nn.Dropout(dropout))

        model.in_net = nn.Sequential(*in_layers)

        model.in_net.requires_grad = True

    """
    Initialize linear output layer
    """
    if out_net:
        output_dim = model.config.n_embd
        out_layers = []
        last_output_size = model.config.n_embd
        for size in out_layer_sizes:
            out_layers.append(nn.Linear(last_output_size, size))
            out_layers.append(nn.ReLU())
            out_layers.append(nn.Dropout(dropout))
            last_output_size = size

        out_final_linear = nn.Linear(last_output_size, output_dim)

        if out_net_init_identity:
            nn.init.eye_(out_final_linear.weight)
            out_final_linear.bias.data.zero_()

        out_layers.append(out_final_linear)
        model.out_net = nn.Sequential(*out_layers)

        model.out_net.requires_grad = True

    """
    out layer on top of lm_head
    """
    # # TODO
    # out_net_top = nn.Linear(model.config.vocab_size, model.config.vocab_size)
    # nn.init.eye_(out_net_top.weight)
    # model.out_net_top = out_net_top
    # model.out_net_top.requires_grad = True

    # TODO: duplicated lm_head
    if dup_lm_head:
        lm_head_new = nn.Linear(model.config.n_embd,
                                model.config.vocab_size, bias=dup_lm_head_bias)
        lm_head_new.weight = torch.nn.Parameter(
            model.lm_head.weight.data.detach().clone(), requires_grad=True)
        # lm_head_new.bias.data.zero_()
        model.lm_head_new = lm_head_new
        model.lm_head_new.requires_grad = True

    """
    Freeze transformer layers
    """

    total_parameters = 0
    target_parameters = 0

    for name, p in model.transformer.named_parameters():
        name = name.lower()

        size = p.size()
        param_count = 1
        for dimension in size:
            param_count *= dimension

        total_parameters += param_count

        if 'ln' in name or 'norm' in name:
            p.requires_grad = not freeze_ln
        elif 'wpe' in name or 'position_embeddings' in name or 'pos_drop' in name:
            p.requires_grad = not freeze_pos
            target_parameters += param_count
        elif 'mlp' in name:
            p.requires_grad = not freeze_ff
        elif 'attn' in name:
            p.requires_grad = not freeze_attn
        elif 'wte' in name:
            p.requires_grad = not freeze_wte
        else:
            p.requires_grad = False

    # print(f'Total params: {total_parameters}')
    # print(
    #     f'Target params: {target_parameters} ({target_parameters / total_parameters * 100:.2f}%)')

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Convert from input ids to word embeddings so that we can apply a linear layer
        x = self.transformer.wte(input_ids)

        try:
            x = self.in_net(x)
        except AttributeError:
            pass

        transformer_outputs = self.transformer(
            inputs_embeds=x,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        try:
            hidden_states = self.out_net(hidden_states)
        except AttributeError:
            pass

        try:
            lm_logits = self.lm_head_new(hidden_states)
        except AttributeError:
            lm_logits = self.lm_head(hidden_states)

        # # TODO
        # lm_logits = self.out_net_top(lm_logits)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    model.forward = MethodType(forward, model)

    return model


# model = get_model()
'''
    only for testing purpose
'''
if __name__ == "__main__":
    model = get_model(gpt2_name='gpt2', in_net=False, in_net_init_identity=True, out_net=False, out_net_init_identity=False, freeze_ln=True, freeze_pos=True,
                      freeze_wte=True, freeze_ff=True, freeze_attn=True)
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.requires_grad)

    for p in model.lm_head_new.parameters():
        print('lm_head_new', p)

    # for p in model.out_net.parameters():
    #     print('out_net',p)
