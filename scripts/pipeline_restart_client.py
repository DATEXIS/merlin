import argparse
import subprocess

from src.kubernetes.yaml_spawner import load_config, render_template, apply_yaml, \
    calculate_hardware_requirements, convert_lists_to_str, wait_for_job

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    cfg = load_config('scripts/pipeline_config.yaml')
    cfg = convert_lists_to_str(cfg)
    cfg = calculate_hardware_requirements(cfg)

    client_name = f"vllm-client-{cfg["job_name"]}"
    client_yaml = render_template(cfg, template_type='client')
    print(f"🚀 Deploying client job with yaml \n{client_yaml}")
    result = subprocess.run(
                ["kubectl", "delete", 'job', client_name, "-n", cfg['namespace']],
                capture_output=True,
                text=True
            )

    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(result.stderr.strip())

    apply_yaml(client_yaml)
