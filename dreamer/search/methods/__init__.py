from dreamer._lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(__name__, {
    "GeneticSearchMethod": "dreamer.search.methods.genetic:GeneticSearchMethod",
})
