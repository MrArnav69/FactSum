import os
import subprocess
import sys

# --- CONFIGURATION ---
BASE_REPO_PATH = "/Users/mrarnav69/Models and Repos"
os.makedirs(BASE_REPO_PATH, exist_ok=True)
os.chdir(BASE_REPO_PATH)

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    subprocess.check_call(cmd, shell=True)

print(f"📂 Setting up repositories in: {BASE_REPO_PATH}")

# --- 1. BARTScore ---
if not os.path.exists("BARTScore"):
    print("⬇️ Cloning BARTScore...")
    run_cmd("git clone https://github.com/neulab/BARTScore.git")

# --- 2. UniEval ---
if not os.path.exists("UniEval"):
    print("⬇️ Cloning UniEval...")
    run_cmd("git clone https://github.com/maszhongming/UniEval.git")
    
    # Checkpoint Download
    ckpt_path = "UniEval/unieval_sum_v1.pth"
    if not os.path.exists(ckpt_path):
        print("⬇️ Downloading UniEval weights (1GB)...")
        # -L follows redirects, indispensable for HF downloads
        run_cmd(f"curl -L https://huggingface.co/zhmh/UniEval/resolve/main/unieval_sum_v1.pth -o {ckpt_path}")

# --- 3. AlignScore ---
if not os.path.exists("AlignScore"):
    print("⬇️ Cloning AlignScore...")
    run_cmd("git clone https://github.com/yuh-zha/AlignScore.git")
    
    # 3a. CRITICAL: Patch setup.py for Mac M3 (Torch > 2.0 support)
    setup_file = "AlignScore/setup.py"
    if os.path.exists(setup_file):
        with open(setup_file, 'r') as f:
            content = f.read()
        
        # Remove version caps that break on modern Macs
        if 'torch<2' in content:
            print("🔧 Patching AlignScore requirements...")
            content = content.replace('torch<2,>=1.12.1', 'torch')
            content = content.replace('pytorch-lightning<2,>=1.7.7', 'pytorch-lightning')
            
            with open(setup_file, 'w') as f:
                f.write(content)
            print("✅ Patched AlignScore/setup.py successfully.")

    # 3b. Install AlignScore in editable mode
    # We use --no-deps to prevent it from messing up your environment again
    run_cmd("pip install --no-deps -e AlignScore")
    
    # 3c. Checkpoint Download
    ckpt_path = "AlignScore/AlignScore-base.ckpt"
    if not os.path.exists(ckpt_path):
        print("⬇️ Downloading AlignScore Checkpoint (800MB)...")
        run_cmd(f"curl -L https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt -o {ckpt_path}")

print("\n✨ Setup Complete! You are ready to run the evaluation.")