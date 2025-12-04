import os
import sys
import subprocess
import time

class ZKProver:
    def __init__(self, project_root=".", build_dir="cmake-build-release"):
        self.project_root = os.path.abspath(project_root)
        self.build_dir = os.path.join(self.project_root, build_dir)
        # We will rename the executable to a generic 'zkcnn_cli'
        self.executable = os.path.join(self.build_dir, "src", "zkcnn_cli")
        self.script_dir = os.path.join(self.project_root, "script")
        
    def ensure_built(self):
        """Checks if C++ binary exists, builds if not."""
        if not os.path.exists(self.executable):
            print(f"[ZKProver] C++ binary not found at {self.executable}. Building project...")
            try:
                # We need to run build.sh from the script directory
                subprocess.check_call(["./build.sh"], cwd=self.script_dir, shell=True)
            except subprocess.CalledProcessError as e:
                print(f"[ZKProver] Build failed: {e}")
                sys.exit(1)
        else:
            print("[ZKProver] C++ binary found.")

    def prove(self, input_file, output_file, model_type="lenetCifar", num_classes=10):
        """
        Runs the C++ prover.
        
        Args:
            input_file (str): Path to the witness file (exported by python).
            output_file (str): Path where the prover should write results.
            model_type (str): The name of the model architecture (e.g., 'lenetCifar').
            num_classes (int): Number of output classes.
        """
        self.ensure_built()
        
        # Dummy config required by the legacy C++ argument parser
        dummy_config = os.path.join(self.project_root, "data", "dummy_config.csv")
        if not os.path.exists(os.path.dirname(dummy_config)):
            os.makedirs(os.path.dirname(dummy_config), exist_ok=True)
        if not os.path.exists(dummy_config):
             with open(dummy_config, 'w') as f: f.write("")

        # Arguments: input config output pic_cnt model_name num_classes
        # Note: We are updating the C++ CLI to accept model_name
        cmd = [
            self.executable,
            os.path.abspath(input_file),
            dummy_config,
            os.path.abspath(output_file),
            "1", # pic_cnt (batch size 1 for demo)
            model_type,
            str(num_classes)
        ]
        
        print(f"[ZKProver] Running {model_type} proof for {num_classes} classes...")
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check stderr for specific C++ success messages or failures
            print(result.stderr)
            
            if result.returncode != 0:
                print(f"[ZKProver] Error: {result.stderr}")
                return False
            
            elapsed = time.time() - start_time
            print(f"[ZKProver] Proof generated and verified in {elapsed:.2f} seconds.")
            return True
            
        except Exception as e:
            print(f"[ZKProver] Execution failed: {e}")
            return False
