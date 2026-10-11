from dreamer._lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(__name__, {
    "Tier3PostProcessModV1": "dreamer.post_process.tier3_post_process_mod:Tier3PostProcessModV1",
})
