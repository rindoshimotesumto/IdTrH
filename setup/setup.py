import subprocess
import sys
import os
import re

# ------------------------
# helpers
# ------------------------
def run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)

def get_cuda_version():
    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8"
        )
        match = re.search(r"CUDA Version: (\d+\.\d+)", out)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None

def torch_index(cuda):
    if not cuda:
        return "https://download.pytorch.org/whl/cpu"

    major = int(cuda.split(".")[0])

    if major >= 12:
        return "https://download.pytorch.org/whl/cu121"
    elif major == 11:
        return "https://download.pytorch.org/whl/cu118"
    else:
        return "https://download.pytorch.org/whl/cpu"

# ------------------------
# main
# ------------------------
def main():
    print("=== SETUP ENVIRONMENT ===")

    # 1. create venv
    if not os.path.exists("venv"):
        run([sys.executable, "-m", "venv", "venv"])

    # paths
    if os.name == "nt":
        pip = "venv\\Scripts\\pip"
        python = "venv\\Scripts\\python"
    else:
        pip = "venv/bin/pip"
        python = "venv/bin/python"

    # 2. upgrade pip
    run([python, "-m", "pip", "install", "--upgrade", "pip"])

    # 3. install project dependencies
    if os.path.exists("requirements.txt"):
        print("Installing requirements.txt")
        run([pip, "install", "-r", "requirements.txt"])
    else:
        print("requirements.txt not found — skipping")

    # 4. detect CUDA
    cuda = get_cuda_version()
    index = torch_index(cuda)

    print(f"Detected CUDA: {cuda or 'None'}")
    print(f"Using PyTorch index: {index}")

    # 5. install PyTorch
    run([
        pip, "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", index
    ])

    print("✅ ENVIRONMENT READY")

if __name__ == "__main__":
    main()