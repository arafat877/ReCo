import torch
from .wan_video_dit import DiTBlock, DiTBlock_w_attnscore


class VaceWanAttentionBlock_w_attnscore(DiTBlock_w_attnscore):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs, obj_mask=None):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        if obj_mask is not None:
            c, attn_score_list = super().forward(c, context, t_mod, freqs, obj_mask)
            c_skip = self.after_proj(c)
            all_c += [c_skip, c]
            c = torch.stack(all_c)
            return c, attn_score_list
        else:
            c = super().forward(c, context, t_mod, freqs)
            c_skip = self.after_proj(c)
            all_c += [c_skip, c]
            c = torch.stack(all_c)
            return c


# Direct change model structure..
class VaceWanModel_w_attnscore(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,                             # NOTE: change..[noise, cond_latent1, cond_latent2, mask]
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # vace blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock_w_attnscore(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)


    def forward(self, x, vace_context, context, t_mod, freqs, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False, obj_mask=None):
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        all_attnscore = []
        for block in self.vace_blocks:
            if obj_mask is not None:  
                if self.training:
                    if use_gradient_checkpointing:
                        c, attn_score = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            c, x, context, t_mod, freqs, obj_mask,
                            use_reentrant=False, 
                        )
                    elif use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            c, attn_score = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                c, x, context, t_mod, freqs, obj_mask,
                                use_reentrant=False,
                            )
                    else:
                        NotImplementedError
                    
                else:
                    c, attn_score = block(c, x, context, t_mod, freqs, obj_mask)
                
                all_attnscore.extend(attn_score)        # merge each list for each attn layers
            else:
                if self.training:
                    if use_gradient_checkpointing:
                        c = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            c, x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                    elif use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            c = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                c, x, context, t_mod, freqs,
                                use_reentrant=False,
                            )
                    else:
                        NotImplementedError
                    
                else:
                    c = block(c, x, context, t_mod, freqs)

        hints = torch.unbind(c)[:-1]
        if obj_mask is not None:
            return hints, all_attnscore
        else:
            return hints
    

    @staticmethod
    def state_dict_converter():
        return VaceWanModelDictConverter()
    

class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


# Direct change model structure..
class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,                             # NOTE: change..[noise, cond_latent1, cond_latent2, mask]
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # vace blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, vace_context, context, t_mod, freqs, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False):
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.vace_blocks:
            if self.training:
                if use_gradient_checkpointing:
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c, x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
                elif use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        c = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            c, x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    NotImplementedError
                
            else:
                c = block(c, x, context, t_mod, freqs)
        hints = torch.unbind(c)[:-1]
        return hints
    
    @staticmethod
    def state_dict_converter():
        return VaceWanModelDictConverter()
    
    
class VaceWanModelDictConverter:
    def __init__(self):
        pass
    
    def from_civitai(self, state_dict):
        state_dict_ = {name: param for name, param in state_dict.items() if name.startswith("vace")}
        return state_dict_
