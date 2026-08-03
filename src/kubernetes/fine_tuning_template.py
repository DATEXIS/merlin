finetune_template = """
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ cfg.job_name }}
  namespace: {{ cfg.project.namespace }}
  labels:
    app: {{ cfg.job_name }}
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: {{ cfg.job_name }}
    spec:
      restartPolicy: Never
      nodeSelector:
        gpu: {{ cfg.Hardware.gpu }}

      imagePullSecrets:
        - name: private-registry-auth

      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 10Gi # Give the B200s plenty of room
        - name: model-volume
          persistentVolumeClaim:
            claimName: model-volume
        {% if cfg.project.namespace == 'merlin' %}
        - name: merlin-ddx-pvc
          persistentVolumeClaim:
            claimName: merlin-ddx-ckpts-pvc
        {% endif %}


      containers:
        - name: fine-tuning
          image: {{ cfg.project.image }}
          imagePullPolicy: Always
          {% if cfg.training.fsdp_full_finetuning|default(false) %}
          {#- Non-Unsloth full-FT path (the design notes): Unsloth is DDP-only
              (full model replicated per GPU), so it can't fit a model whose
              full optimizer state doesn't fit on a single GPU (32B+). This
              path bypasses Unsloth (src/fine_tuning/training_fsdp.py, chosen
              in entrypoint.py by --fsdp_full_finetuning) and shards params/
              grads/optimizer state across GPUs via Accelerate's FSDP
              integration instead of accelerate launch --multi_gpu's plain DDP. #}
          command: ["accelerate", "launch"]
          args: [
            "--use_fsdp",
            "--num_processes={{ cfg.Hardware.gpu_count }}",
            "--num_machines=1",
            "--mixed_precision=bf16",
            "--fsdp_sharding_strategy=FULL_SHARD",
            "--fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP",
            "--fsdp_transformer_layer_cls_to_wrap=Qwen3DecoderLayer",
            "--fsdp_state_dict_type=FULL_STATE_DICT",
            "--fsdp_backward_prefetch=BACKWARD_PRE",
            "--fsdp_cpu_ram_efficient_loading=true",
            "--fsdp_sync_module_states=true",
            "--fsdp_use_orig_params=true",
            "/app/entrypoint.py",
          {% elif cfg.Hardware.gpu_count > 1 %}
          command: ["accelerate", "launch"]
          args: [
            "--multi_gpu",
            "--num_processes={{ cfg.Hardware.gpu_count }}",
            "/app/entrypoint.py",
          {% else %}
          command: ["python", "/app/entrypoint.py"]
          args: [
          {% endif %}
            "--model_name={{ cfg.training.model_name }}",
            "--dataset_path={{ cfg.training.dataset_path }}",
            "--dataset_models={{ cfg.training.dataset_models}}",
            "--output_dir={{ cfg.models_root }}/{{ cfg.job_name }}",
            "--job_name={{ cfg.job_name }}",
            "--epochs={{ cfg.training.epochs }}",
            "--batch_size={{ cfg.training.batch_size }}",
            "--grad_accum={{ cfg.training.grad_accum }}",
            "--learning_rate={{ cfg.training.learning_rate }}",
            "--warmup_steps={{ cfg.training.warmup_steps }}",
            "--weight_decay={{ cfg.training.weight_decay }}",
            "--max_seq_length={{ cfg.training.max_seq_length }}",
            "--eval_batch_size={{ cfg.training.eval_batch_size }}",
            "--eval_samples={{ cfg.training.eval_samples }}",
            "--resume_from_checkpoint={{ cfg.training.resume_from_checkpoint }}",
            "--logging_steps={{ cfg.training.logging_steps }}",
            "--seed={{ cfg.training.seed }}",
            "--load_in_4bit={{ cfg.training.load_in_4bit }}",
            "--packing={{ cfg.training.packing }}",
            "--assistant_only_loss={{ cfg.training.assistant_only_loss }}",
            "--use_lora={{ cfg.training.use_lora }}",
            "--unsloth_full_finetuning={{ cfg.training.unsloth_full_finetuning }}",
            "--fsdp_full_finetuning={{ cfg.training.fsdp_full_finetuning|default(false) }}",
            "--lora_r={{ cfg.training.lora_r }}",
            "--lora_dropout={{ cfg.training.lora_dropout }}",
            "--wandb_project={{ cfg.project.wandb_project }}",
            "--wandb_entity={{ cfg.project.wandb_entity }}",
            "--hf_cache_dir={{ cfg.project.hf_cache_dir }}",
            "--wandb_cache_dir={{ cfg.project.wandb_cache_dir }}"
          ]

          resources:
            requests:
              memory: "32Gi" # Hardcoded or add to Hardware config
              nvidia.com/gpu: "{{ cfg.Hardware.gpu_count }}"
            limits:
              memory: "1000Gi"
              nvidia.com/gpu: "{{ cfg.Hardware.gpu_count }}"

          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
            - name: model-volume
              mountPath: /models
            {% if cfg.project.namespace == 'merlin' %}
            - name: merlin-ddx-pvc
              mountPath: /ft_models
            {% endif %}

          env:
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"
            - name: NCCL_DEBUG
              value: "WARN"
            - name: NCCL_SOCKET_FAMILY
              value: "AF_INET"

            - name: PYTORCH_CUDA_ALLOC_CONF
              value: expandable_segments:True
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
                  """