"""Patches to the Miles checkout in the gemma4-miles image for LoRA GRPO on Gemma 4 31B (run at container start).

Each patch asserts that its target text occurs exactly once, so a changed Miles fails loudly instead of silently.
  1. hf_weight_iterator_bridge: with --lora-train-only only base weights are synced to SGLang (SGLang has no Gemma 4
     LoRA). Export them with the adapters merged in, so the engines serve the current policy (W + BA) every step.
  2. hf_lora_targets: Miles defines no HF LoRA layout for Gemma 4 (a VL config; text layers under
     model.language_model.layers.*). Dense q/k/v/o + gate/up/down.
  3. target_modules: the global layers share K and V and have no v_proj; the bridge's _Gemma4QKVMapping tolerates
     that, but Miles required LoRA on the fused linear_qkv to cover q, k and v of every layer. Ignore projections absent
     from the checkpoint.
  4. lora/bridge: Gemma4VLModel.forward returns (logits, loss_mask) when given a loss_mask. The non-LoRA provider keeps
     the logits (model_provider.py, _logits_only_forward); the LoRA path lacked that wrapper. Pre-wrap hooks are chained
     (model = hook(model)) and _install_model_companions returns None, so the wrapper is registered before it.
  5. modeling_gemma4 (Megatron-Bridge): the Gemma 4 dense layer spec uses Megatron's local DotProductAttention for
     every layer, which builds the full [s, s] score matrix and masks the 1024-token window of the 50 sliding layers
     afterwards. With G4RL_SLIDING_FLASH=1 (default) the sliding layers run FlashAttention-3 instead (causal, the same
     window and softmax_scale); the 10 global layers (head_dim 512, beyond FlashAttention) keep the original code,
     or with G4RL_GLOBAL_D512=1 run kernels/attn_d512.py for sequences of at least G4RL_D512_MIN_SEQ tokens
     (O(s) memory; slower than the fp32 score matrix below ~16k tokens, so the threshold keeps short segments fast).
  6. fused_softmax (Megatron-LM): the apex causal kernel ignores the sliding window for sk <= 4096 (upstream fix).
  7. actor sleep/wake_up: with --colocate --offload-train, Miles keeps the LoRA param buffers resident (they have no
     host backup) and reads the frozen base from the memory-saver backup during update_weights. Both only when
     lora_rollout_enabled(); with --lora-train-only, sleep() paused everything (pause(tag=None)), so the adapters were
     discarded and update_weights read unmapped memory (the actor died with "cpu_backup does not exist"). Gate the
     LoRA branches on is_lora_enabled(), which is what _weight_sync_reads_tms_backup already uses.
  8. actor update_weights: with --lora-train-only the merged export still reads some tensors through the live model
     (Gemma 4's buffer-like layer_scalar/scale params are not in the exported weight dict, and the bridge mappings read
     module state); while asleep those pages are unmapped and every trainer rank hit an Xid 31 MMU fault, hanging the
     colocated engines. Map the "default" region back (~16 GB/rank; SGLang holds only its weights then) for the export
     and pause it again afterwards, so colocate exports from live tensors exactly like the resident 4+4 layout.
  9. lora/bridge + actor sleep/wake_up: after a training step the trainer keeps ~40 GB/rank of live allocations besides
     the 16 GB of frozen weights, so mapping the whole "default" region back for the export reached 133 GB with SGLang's
     weights resident. Build the model inside its own memory-saver region ("g4rl_model", CPU backup) so that patch 8
     maps back only the frozen base and buffers; sleep()/wake_up() pause/resume that region alongside "default".
 10. actor sleep(): with G4RL_DUMP_LIVE=1, list the largest live CUDA storages before offloading (diagnostic for 9).
 11. sglang_api_client.release_memory_occupation: abort all requests before flush_cache. A harness that kills a
     trajectory at its session budget leaves that generation running on the engine (the TITO proxy does not cancel
     it); flush_cache then fails for 60 s ('pending requests') and the whole job dies (dapo_colo12s_e15, step 5).
     Every rollout has returned when Miles offloads the engines, so anything still running is an orphan.
 12. rollout_executor.load: resume reads the rollout-side state (data-source position) from `--load`; in bridge
     mode Miles rewrites args.load to the HF base when the dir holds no Megatron tracker file, which is always the
     case for LoRA checkpoints (iter_*/adapter only), so a resume looked under /root/models/base/rollout/<id>. Use
     the originally requested --load (args.requested_load). Resume = --lora-adapter-path <iter_X>/adapter
     (restore_lora_adapter returns the saved iteration; start_rollout_id = iteration + 1); RESUME=1 in run_experiment.sh.
 13. megatron_config.resolve_args_checkpoint_load: in bridge mode Miles forces args.start_rollout_id = 0 whenever --load
     has no Megatron tracker (always, for LoRA checkpoints), and create_training_model treats that 0 as an explicit
     request, so a LoRA resume restored the adapter but restarted the rollout loop at 0 (even with --start-rollout-id 18).
     Leave the value alone when a LoRA adapter is being restored or when the user passed --start-rollout-id.
"""
M = '/root/miles/miles'


def patch(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, f'patch target not found in {path}: {old[:80]!r}'
    open(path, 'w').write(s.replace(old, new))
    print('patched', path)


patch(f'{M}/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py',
      'merge_adapter_weights=False,',
      "merge_adapter_weights=bool(getattr(self.args, 'lora_train_only', False)),")

patch(f'{M}/utils/lora/hf_lora_targets.py',
      '_HF_LORA_MODELS = {',
      '_HF_LORA_MODELS = {\n'
      '    "gemma4": _HfLoraModelSpec(_dense_targets, layer_prefix="model.language_model.layers.*", '
      'unwrap_text_config=True),\n'
      '    "gemma4_text": _HfLoraModelSpec(_dense_targets),')

old = '            checkpoint_parameters = _checkpoint_parameters(mapping.resolve(match.groups()))\n'
patch(f'{M}/backends/megatron_utils/lora/target_modules.py', old,
      old + '            if hf_mapping.parameter_names:\n'
            '                checkpoint_parameters = {p for p in checkpoint_parameters\n'
            '                                         if hf_mapping.model_parameter(p) in hf_mapping.parameter_names}'
            ' or checkpoint_parameters\n')

old = '    provider.register_pre_wrap_hook(_install_model_companions)\n'
patch(f'{M}/backends/megatron_utils/lora/bridge.py', old, '''
    def _logits_only_hook(model_chunks):
        for chunk in model_chunks:
            bridge_forward = chunk.forward

            def logits_only_forward(*a, _f=bridge_forward, **k):
                out = _f(*a, **k)
                return out[0] if isinstance(out, tuple) else out

            chunk.forward = logits_only_forward
        return model_chunks

    provider.register_pre_wrap_hook(_logits_only_hook)
''' + old)

BRIDGE = '/opt/sglang/lib/python3.12/site-packages/megatron/bridge/models/gemma/modeling_gemma4.py'
patch(BRIDGE, '                core_attention=backend.core_attention(),',
      '                core_attention=_gemma4_core_attention(backend),')
anchor = '\ndef get_gemma4_layer_spec('
patch(BRIDGE, anchor, '''


# ---- G4RL patch (gemma4-rl/miles_patches.py): FlashAttention-3 for the sliding-window layers ----
def _gemma4_core_attention(backend):
    import os

    base = backend.core_attention()
    if os.environ.get("G4RL_SLIDING_FLASH", "1") != "1":
        return base

    use_d512 = os.environ.get("G4RL_GLOBAL_D512", "0") == "1"

    class Gemma4SlidingFlashAttention(base):
        """Megatron's DotProductAttention with faster kernels per layer type.

        Sliding layers (head_dim 256): FlashAttention-3 with the parent's causal mask, window (config.window_size, e.g.
        (1023, 0): the token and the 1023 before it) and softmax_scale. Global layers (head_dim 512, K=V), with
        G4RL_GLOBAL_D512=1: the Triton kernel in gemma4-rl/kernels/attn_d512.py (no shipped FlashAttention takes
        head_dim 512). Any unsupported case runs the parent unchanged.
        """

        def __init__(self, config, layer_number, *args, **kwargs):
            super().__init__(config, layer_number, *args, **kwargs)
            ws = config.window_size
            common = (config.context_parallel_size == 1 and not config.apply_query_key_layer_scaling
                      and getattr(config, "softmax_type", "vanilla") == "vanilla")
            sliding = bool(ws) and is_layer_window_attention(ws, config.window_attn_skip_freq, layer_number)
            self.g4_flash = common and sliding and ws[1] == 0
            self.g4_window = (ws[0], 0) if self.g4_flash else None
            self.g4_d512 = common and not sliding and use_d512 and self.hidden_size_per_attention_head == 512
            # the d512 kernel is slower than cuBLAS + fp32 softmax on short sequences (8 heads per rank underfill the
            # GPU) but needs O(s) memory instead of an [8, s, s] fp32 score matrix: use it from this length on
            import os as _os
            self.g4_d512_min = int(_os.environ.get("G4RL_D512_MIN_SEQ", "0"))

        def forward(self, query, key, value, attention_mask, attn_mask_type=None, attention_bias=None,
                    packed_seq_params=None):
            use_d512 = self.g4_d512 and query.shape[0] >= self.g4_d512_min
            if (not (self.g4_flash or use_d512) or attention_bias is not None or packed_seq_params is not None
                    or (self.training and self.config.attention_dropout > 0)):
                return super().forward(query, key, value, attention_mask, attn_mask_type=attn_mask_type,
                                       attention_bias=attention_bias, packed_seq_params=packed_seq_params)
            sq, b, nh, hn = query.shape
            q, k, v = (t.transpose(0, 1).contiguous() for t in (query, key, value))  # [b, s, h, d]
            if self.g4_flash:
                from flash_attn_interface import flash_attn_func
                out = flash_attn_func(q, k, v, softmax_scale=self.softmax_scale, causal=True, window_size=self.g4_window)
                out = out[0] if isinstance(out, tuple) else out
            else:
                import sys
                if "/root/g4rl/kernels" not in sys.path:
                    sys.path.insert(0, "/root/g4rl/kernels")
                from attn_d512 import attn_d512_causal
                out = attn_d512_causal(q, k, v, self.softmax_scale)
            return out.transpose(0, 1).reshape(sq, b, nh * hn)

    return Gemma4SlidingFlashAttention

''' + anchor)

# 6. Megatron fused softmax: refuse the apex causal kernel when a sliding window is set, and OR the window mask into
#    a caller-provided mask instead of replacing it (upstream Megatron-LM fix).
F = '/root/Megatron-LM/megatron/core/fusions/fused_softmax.py'
patch(F,
      "            self.scaled_masked_softmax_fusion  # user want to fuse\n",
      "            self.scaled_masked_softmax_fusion  # user want to fuse\n"
      "            and self.window_size is None  # apex kernels have no sliding-window support (upstream fix)\n")
patch(F,
      "        if self.window_size is not None:\n"
      "            mask = get_sliding_window_causal_mask(sq, sk, self.window_size)\n",
      "        if self.window_size is not None:\n"
      "            sliding_window_mask = get_sliding_window_causal_mask(sq, sk, self.window_size)\n"
      "            mask = sliding_window_mask if mask is None else (mask | sliding_window_mask)\n")

# 7. Miles actor: in colocate+offload mode keep the (no-backup) LoRA param buffers resident for every LoRA mode, not only
#    lora_rollout_enabled(); update_weights (_weight_sync_reads_tms_backup, gated on is_lora_enabled) reads them live and
#    the frozen base from the memory-saver host backup.
F = f'{M}/backends/megatron_utils/actor.py'
patch(F,
      '        elif lora_rollout_enabled(self.args):\n'
      '            # adapter params keep their host backup in "default"; the grad buffers have none\n',
      '        elif is_lora_enabled(self.args):  # g4rl: also --lora-train-only\n'
      '            # adapter param buffers have no host backup and stay resident for update_weights; the grad buffers\n'
      '            # have none either and are discarded (zeroed before the next step)\n')
patch(F,
      '        if lora_rollout_enabled(self.args):\n'
      '            torch_memory_saver.resume(tag="default")\n',
      '        if is_lora_enabled(self.args):  # g4rl: also --lora-train-only\n'
      '            torch_memory_saver.resume(tag="default")\n')

# 8. Miles actor: in colocate+offload with --lora-train-only, map the trainer's "default" region back for the merged
#    export (see docstring) and pause it again afterwards.
F = f'{M}/backends/megatron_utils/actor.py'
patch(F,
      '        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():\n'
      '            print_memory("before update_weights")\n',
      '        # g4rl: the merged export reads frozen base weights and Gemma 4 buffer-like params through the live model;\n'
      '        # asleep, those pages are unmapped (Xid 31). Map the default region back for the export.\n'
      '        g4rl_resume_default = self._asleep and is_lora_enabled(self.args) and not lora_rollout_enabled(self.args)\n'
      '        if g4rl_resume_default:\n'
      '            torch_memory_saver.resume(tag="g4rl_model")  # frozen base + buffers only, see lora/bridge patch\n'
      '            print_memory("after g4rl resume model region for export")\n'
      '        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():\n'
      '            print_memory("before update_weights")\n')
patch(F,
      '        if process_groups_are_temporary:\n'
      '            destroy_process_groups()\n'
      '\n'
      '        return WeightUpdateOutput(weight_version=weight_version, failed_cell_ids=failed_cell_ids)\n',
      '        if g4rl_resume_default:\n'
      '            clear_memory()\n'
      '            torch_memory_saver.pause(tag="g4rl_model")\n'
      '            print_memory("after g4rl pause model region after export")\n'
      '        if process_groups_are_temporary:\n'
      '            destroy_process_groups()\n'
      '\n'
      '        return WeightUpdateOutput(weight_version=weight_version, failed_cell_ids=failed_cell_ids)\n')

# 9. Model construction in its own memory-saver region; sleep/wake_up handle it next to "default".
F = f'{M}/backends/megatron_utils/lora/bridge.py'
patch(F,
      '    model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)\n'
      '    return model\n',
      '    # g4rl: frozen base weights and buffers get their own memory-saver region (CPU backup), so the colocated\n'
      '    # update_weights can map back just them for the merged export (actor.py patch 8).\n'
      '    from contextlib import nullcontext as _g4rl_nullcontext\n'
      '    from torch_memory_saver import torch_memory_saver as _g4rl_tms\n'
      '    _g4rl_region = (_g4rl_tms.region(tag="g4rl_model", enable_cpu_backup=True)\n'
      '                    if args.offload_train and getattr(args, "lora_train_only", False) else _g4rl_nullcontext())\n'
      '    with _g4rl_region:\n'
      '        model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)\n'
      '    return model\n')
F = f'{M}/backends/megatron_utils/actor.py'
patch(F,
      '                torch_memory_saver.pause(tag="grad_buffer")\n'
      '                self._grad_buffer_paused = True\n'
      '            torch_memory_saver.pause(tag="default")\n',
      '                torch_memory_saver.pause(tag="grad_buffer")\n'
      '                self._grad_buffer_paused = True\n'
      '            if os.environ.get("G4RL_DUMP_LIVE") == "1":\n'
      '                _g4rl_dump_live_cuda_storages()\n'
      '            torch_memory_saver.pause(tag="default")\n'
      '            if not lora_rollout_enabled(self.args):\n'
      '                torch_memory_saver.pause(tag="g4rl_model")  # g4rl: frozen base + buffers region\n')
patch(F,
      '            torch_memory_saver.resume(tag="default")\n'
      '            torch_memory_saver.resume(tag="grad_buffer")\n',
      '            torch_memory_saver.resume(tag="default")\n'
      '            if not lora_rollout_enabled(self.args):\n'
      '                torch_memory_saver.resume(tag="g4rl_model")  # g4rl: frozen base + buffers region\n'
      '            torch_memory_saver.resume(tag="grad_buffer")\n')
# 10. Diagnostic: largest live CUDA storages (deduplicated by storage) before offloading.
patch(F,
      'from torch_memory_saver import torch_memory_saver\n',
      'from torch_memory_saver import torch_memory_saver\n'
      '\n'
      '\n'
      'def _g4rl_dump_live_cuda_storages(top=15):\n'
      '    """g4rl diagnostic: what is still alive on the GPU before offload (deduplicated by storage)."""\n'
      '    import gc\n'
      '    seen = {}\n'
      '    for o in gc.get_objects():\n'
      '        try:\n'
      '            if torch.is_tensor(o) and o.is_cuda:\n'
      '                st = o.untyped_storage()\n'
      '                k = st.data_ptr()\n'
      '                if k not in seen or seen[k][0] < st.nbytes():\n'
      '                    seen[k] = (st.nbytes(), tuple(o.shape), str(o.dtype), type(o).__name__, bool(getattr(o, "requires_grad", False)))\n'
      '        except Exception:\n'
      '            continue\n'
      '    items = sorted(seen.values(), reverse=True)\n'
      '    total = sum(x[0] for x in items)\n'
      '    logger.info(f"[g4rl live cuda storages] {len(items)} storages, {total / 2**30:.2f} GB total; "\n'
      '                f"torch allocated {torch.cuda.memory_allocated() / 2**30:.2f} GB")\n'
      '    for nb, shape, dt, tn, rg in items[:top]:\n'
      '        logger.info(f"[g4rl live cuda storages]   {nb / 2**30:7.2f} GB  {shape} {dt} {tn} grad={rg}")\n'
      '    by_dtype = {}\n'
      '    for nb, shape, dt, tn, rg in items:\n'
      '        by_dtype[dt] = by_dtype.get(dt, 0) + nb\n'
      '    logger.info("[g4rl live cuda storages] by dtype: " + ", ".join(f"{k} {v / 2**30:.2f} GB" for k, v in sorted(by_dtype.items(), key=lambda kv: -kv[1])))\n')

# 11. Abort orphan generations before flushing the engines' cache (colocate offload path).
F = f'{M}/backends/sglang_utils/sglang_api_client.py'
patch(F,
      '        """Release memory occupation. Available tags: weights, kv_cache."""\n'
      '        await self.flush_cache()\n',
      '        """Release memory occupation. Available tags: weights, kv_cache."""\n'
      '        # g4rl: a trajectory killed at its harness budget leaves its generation running here; flush_cache would then\n'
      '        # fail for 60 s and kill the job. Every rollout has returned at this point, so what still runs is an orphan.\n'
      '        await self.abort_all_requests()\n'
      '        await self.flush_cache()\n')

# 12. Rollout-side resume state lives next to the LoRA checkpoints, under the requested --load, not the HF base.
F = f'{M}/ray/rollout/rollout_executor.py'
patch(F,
      '        if self.args.load is None:\n'
      '            logger.warning("no --load: the rollout side starts fresh")\n'
      '            return\n'
      '        assert rollout_id >= 0, f"rollout {rollout_id} is not a trained step"\n'
      '\n'
      '        directory = compute_rollout_checkpoint_dir(self.args.load, rollout_id=rollout_id)\n',
      '        # g4rl: bridge mode rewrites args.load to the HF base when --load holds no Megatron tracker (LoRA checkpoints\n'
      '        # never do); the rollout-side state was saved under the requested --load (= --save), so look there.\n'
      '        load_dir = getattr(self.args, "requested_load", None) or self.args.load\n'
      '        if load_dir is None:\n'
      '            logger.warning("no --load: the rollout side starts fresh")\n'
      '            return\n'
      '        assert rollout_id >= 0, f"rollout {rollout_id} is not a trained step"\n'
      '\n'
      '        directory = compute_rollout_checkpoint_dir(load_dir, rollout_id=rollout_id)\n')

# 13. Do not clobber the start rollout id of a LoRA resume (bridge mode, no Megatron tracker file).
F = f'{M}/backends/megatron_utils/megatron_config.py'
patch(F,
      '            args.load = args.ref_load or args.hf_checkpoint\n'
      '            args.start_rollout_id = 0\n',
      '            args.load = args.ref_load or args.hf_checkpoint\n'
      '            # g4rl: a LoRA resume (--lora-adapter-path) restores the trained iteration and create_training_model uses\n'
      '            # the restored rollout id when this stays None; an explicit --start-rollout-id must survive as well.\n'
      '            if args.start_rollout_id is None and getattr(args, "lora_adapter_path", None) is None:\n'
      '                args.start_rollout_id = 0\n')
