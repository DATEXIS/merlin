client_template = """
apiVersion: batch/v1
kind: Job
metadata:
  name: vllm-client-{{ cfg.job_name }}{{ cfg.get('client_suffix', '') }}
  namespace: {{ cfg.namespace }}
spec:
  template:
    spec:
      containers:
        - name: vllm-client
          image: {{ cfg.client_image }}
          command: ["python3", "/app/entrypoint.py",
                    "--file_name={{ cfg.Client_Job.file_name }}",
                    "--server_name=vllm-server-{{ cfg.job_name }}",
                    "--namespace={{ cfg.namespace }}",
                    "--chief_complaint={{ cfg.Client_Job.chief_complaint }}",
                    "--load_from_checkpoint={{ cfg.load_from_checkpoint }}",
                    "--budget={{ cfg.Client_Job.budget }}",
                    "--num_choices={{ cfg.Client_Job.num_choices }}",
                    "--seed={{ cfg.Client_Job.seed }}",
                    "--num_samples={{ cfg.Client_Job.num_samples }}",
                    "--concurrency={{ cfg.Client_Job.concurrency }}",
                    "--start_verifier={{ cfg.Client_Job.start_verifier }}",
                    "--temperatures={{ cfg.Client_Job.temperatures }}",
                    "--max_tokens={{ cfg.Client_Job.max_tokens }}",
                    "--thresholds={{ cfg.Client_Job.thresholds }}",
                    "--eval_mode={{ cfg.Client_Job.eval_mode }}",
                    "--split={{ cfg.Client_Job.get('split', '') }}",
                    "--merlin_mode={{ cfg.Client_Job.merlin_mode }}",
                    "--guided_decoding={{ cfg.Client_Job.guided_decoding }}",
                    "--think_about_labs={{ cfg.Client_Job.think_about_labs }}",
                    "--lora={{ cfg.Model.lora }}",
                    "--lora_name={{ cfg.Model.lora_modules }}",
                    "--config_string={{ cfg }}",
                    ]
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: HF_TOKEN
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wandb-secret
                  key: api-key
            {% if cfg.Client_Job.get('wandb_tags') %}
            {#- Comma-separated wandb run tags (e.g. "test_data"). Picked up by
                wandb.init() via the WANDB_TAGS env var, since init_wandb() in
                src/wandb/run.py passes tags=None -- template-side so no client
                image rebuild is needed. update_wandb_name_tags() appends its
                own tags with `run.tags +=`, which preserves these. #}
            - name: WANDB_TAGS
              value: "{{ cfg.Client_Job.wandb_tags }}"
            {% endif %}
          resources:
            limits:
              memory: "32Gi"
              {% if cfg.Hardware.client_gpu != 'None' %}
              nvidia.com/gpu: "1"
              {% endif %}
            requests:
              {% if cfg.Hardware.client_gpu != 'None' %}
              nvidia.com/gpu: "1"
              {% endif %}
          volumeMounts:
            - name: model-volume
              mountPath: /models
            - name: checkpoint-volume
              mountPath: /checkpoints
      volumes:
        - name: model-volume
          persistentVolumeClaim:
            claimName: model-volume
            {% if cfg.namespace == 'merlin'%}
        - name: checkpoint-volume
          persistentVolumeClaim:
            claimName: checkpoint-volume
            {% else %}
        - name: checkpoint-volume
          persistentVolumeClaim:
            claimName: checkpoints-volume
            {% endif %}
      imagePullSecrets:
        - name: private-registry-auth
      {% if cfg.Hardware.client_gpu is not string and cfg.Hardware.client_gpu is iterable %}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: gpu
                    operator: In
                    values: [{{ cfg.Hardware.client_gpu | join(', ') }}]
      {% elif cfg.Hardware.client_gpu != 'None' %}
      nodeSelector:
        gpu: {{ cfg.Hardware.client_gpu }}
      {% endif %}
      restartPolicy: Never
  backoffLimit: 0
"""