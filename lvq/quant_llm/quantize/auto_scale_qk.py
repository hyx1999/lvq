import torch

__all__ = ["auto_scale_block", "apply_scale"]

@torch.no_grad()
def get_act_scale(q: torch.Tensor):
    B, H, L, D = q.shape
    return q.transpose(1, 2).abs().reshape(B * L, H, D).mean(0)

@torch.no_grad()
def compress_q(q: torch.Tensor, num_groups: int):
    B, H, L, D = q.shape
    return q.view(B, H // num_groups, num_groups, L, D).mean(dim=2)    

@torch.no_grad()
def auto_scale_block(attn_module, attn_kwargs, input_feat):

    # find the best scale ratio
    def _search_module_scale(block, qk_scaler, x, q, kwargs={}):
        # w: co, ci
        # x: n, ci
        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        q_max = get_act_scale(compress_q(q, attn_module.num_key_value_groups))

        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        attn_module.config.enable_kv_quant = True
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = q_max.pow(ratio).clamp(min=1e-4)
            scales = scales / (scales.amax(dim=-1) * scales.amin(dim=-1)).sqrt().unsqueeze(-1)  # [H, D]
            qk_scaler.scales.copy_(scales)
            
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (
                (org_out - out).float().pow(2).mean().item()
            )  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)
        attn_module.config.enable_kv_quant = False
        if best_ratio == -1:
            print(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    scales = _search_module_scale(
        attn_module, 
        attn_module.qk_scaler, 
        input_feat["self_attn.q_proj"], 
        input_feat["self_attn.qk_scaler"],
        attn_kwargs
    )
    return scales


def apply_scale(attn_module: torch.nn.Module, scales):
    device = next(attn_module.parameters()).device
    attn_module.qk_scaler.scales.copy_(scales.to(device))
