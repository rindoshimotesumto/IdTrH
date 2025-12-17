import subprocess
import sys
import os
import re
import shutil

REQUIRED_PYTHON = (3, 10)

# ------------------------
# helpers
# ------------------------
def run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)

def find_python_310():
    candidates = [
        "python3.10",
        "python310",
        "py -3.10",
    ]

    for c in candidates:
        try:
            subprocess.check_output(
                c.split() + ["--version"],
                stderr=subprocess.DEVNULL
            )
            return c.split()
        except Exception:
            continue
    return None

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
    print("=== SETUP ENVIRONMENT (Python 3.10) ===")

    py310 = find_python_310()
    if not py310:
        print("❌ Python 3.10 not found.")
        print("Install Python 3.10 and ensure it is in PATH.")
        sys.exit(1)

    print("Using Python:", " ".join(py310))

    # 1. create venv with python 3.10
    if not os.path.exists("venv"):
        run(py310 + ["-m", "venv", "venv"])

    # paths
    if os.name == "nt":
        pip = "venv\\Scripts\\pip"
        python = "venv\\Scripts\\python"
    else:
        pip = "venv/bin/pip"
        python = "venv/bin/python"

    # 2. upgrade pip
    run([python, "-m", "pip", "install", "--upgrade", "pip"])

    # 3. install requirements
    if os.path.exists("requirements.txt"):
        run([pip, "install", "-r", "requirements.txt"])

    # 4. CUDA detect
    cuda = get_cuda_version()
    index = torch_index(cuda)

    print(f"Detected CUDA: {cuda or 'None'}")
    print(f"PyTorch index: {index}")

    # 5. install PyTorch
    run([
        pip, "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", index
    ])

    print("✅ ENV READY (Python 3.10)")

if __name__ == "__main__":
    main()