import argparse

from src.kubernetes.yaml_spawner import load_config, render_template, apply_yaml, \
    calculate_hardware_requirements, convert_lists_to_str


if __name__ == "__main__":

    """Initialize working directory and logging for Jupyter notebooks."""


    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    cfg = load_config('scripts/pipeline_config.yaml')
    cfg = convert_lists_to_str(cfg)
    cfg = calculate_hardware_requirements(cfg)

    server_yaml = render_template(cfg, template_type='server')
    client_yaml = render_template(cfg, template_type='client')

    print(f"🚀 Deploying server with yaml: \n{server_yaml}")
    apply_yaml(server_yaml)
    print(f"🚀 Deploying client job with yaml \n{client_yaml}")
    apply_yaml(client_yaml)
