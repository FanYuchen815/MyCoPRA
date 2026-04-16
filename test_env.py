import importlib
import sys

modules = [
    ("python", "sys"),
    ("torch", "torch"),
    ("pytorch_lightning", "pytorch_lightning"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "sklearn"),
    ("einops", "einops"),
    ("dm_tree", "tree"),
    ("diskcache", "diskcache"),
    ("fire", "fire"),
    ("gpustat", "gpustat"),
    ("wandb", "wandb"),
    ("easydict", "easydict"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("transformers", "transformers"),
    ("biopython", "Bio"),
    ("fair_esm", "esm"),
    ("torch_geometric", "torch_geometric"),
    ("torch_scatter", "torch_scatter"),
    ("torch_sparse", "torch_sparse"),
    ("torch_cluster", "torch_cluster"),
    # peft removed (LoRA support removed)
    ("biotite", "biotite"),
    ("cpdb_protein", "cpdb"),
    ("torchsummary", "torchsummary"),
    ("triton", "triton"),
]

def try_import(mod_name):
    try:
        m = importlib.import_module(mod_name)
        return True, getattr(m, "__version__", None)
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.splitlines()[0]}")
    results = []
    for label, mod in modules:
        ok, info = try_import(mod)
        if ok:
            ver = info or "(version unknown)"
            print(f"OK: imported {mod} — {ver}")
        else:
            print(f"FAIL: {mod} — {info}")

    # extra checks for torch
    try:
        import torch
        print(f"torch.__version__ = {torch.__version__}")
        try:
            cuda_avail = torch.cuda.is_available()
            print(f"torch.cuda.is_available() = {cuda_avail}")
            if cuda_avail:
                print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        except Exception as e:
            print(f"Could not query CUDA: {e}")
    except Exception:
        pass

    print("\nDone.")
