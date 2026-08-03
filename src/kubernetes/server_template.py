server_template = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server-{{ cfg.job_name }}
  namespace: {{ cfg.namespace }}
spec:
  replicas: {{ cfg.Hardware.replicas }}
  selector:
    matchLabels:
      app: vllm-server-{{ cfg.job_name }}
  template:
    metadata:
      labels:
        app: vllm-server-{{ cfg.job_name }}
    spec:
      restartPolicy: Always
      containers:
        - name: vllm-server
          image: {{ cfg.server_image }}
          command: ["python3", "-m", "vllm.entrypoints.openai.api_server",
                    {% if cfg.Model.full_finetuning %}
                    "--model={{ cfg.get('models_root', '/ft_models/merlin-ddx/output') }}/{{ cfg.Model.lora_modules }}",
                    {% else %}
                    "--model={{ cfg.Model.base_model }}",
                    {% endif %}
                    "--max-model-len={{ cfg.Model.max_model_len }}",
                    "--max-num-seqs={{ cfg.Model.max_num_seqs }}",
                    "--max-num-batched-tokens={{ cfg.Model.max_num_batched_tokens }}",
                    "--tensor-parallel-size={{ cfg.Hardware.gpu_count }}",
                    "--download-dir=/models",
                    "--trust-remote-code",
                    "--gpu-memory-utilization={{ cfg.Hardware.gpu_memory_utilization }}",
                    {% if cfg.Hardware.get('enforce_eager', False) %}
                    "--enforce-eager",
                    "--enable-chunked-prefill",
                    {% else %}
                    "--compilation-config", '{"cudagraph_specialize_lora": false}',
                    {% endif %}
                    {% if cfg.Model.lora %}
                    "--enable-lora",
                    "--lora-modules=lora_module={{ cfg.get('models_root', '/ft_models/merlin-ddx/output') }}/{{ cfg.Model.lora_modules }}",
                    "--max-lora-rank={{ cfg.Model.max_lora_rank }}",
                    {% endif %}
                    {% if cfg.Model.thinking and 'nightly' not in cfg.server_image %}
                    "--enable-reasoning",
                    {% endif %}
                    {% if cfg.Model.thinking %}
                    "--reasoning-parser=deepseek_r1",
                    {% endif %}
                    # "--guided-decoding-backend=outlines",
                    {# Logic for MedGemma / Multimodal-to-Text conversion #}
                    {% if 'medgemma-27b-it' in cfg.Model.base_model %}
                    "--limit-mm-per-prompt", '{"image": 0}',
                    "--skip-mm-profiling",
                    {% endif %}
                   ]

          ports:
            - containerPort: 8000

          resources:
            limits:
              memory: "{{ cfg.memory_limit_server }}Gi"
              nvidia.com/gpu: "{{ cfg.Hardware.gpu_count }}"
            requests:
              memory: "{{ cfg.memory_request_server }}Gi"
              nvidia.com/gpu: "{{ cfg.Hardware.gpu_count }}"

          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: HF_TOKEN
            - name: NCCL_DEBUG
              value: "INFO"
            - name: GLOO_SOCKET_IFNAME
              value: "eth0"
            - name: VLLM_DISABLE_TRITON_LORA
              value: "1"
            - name: VLLM_LORA_KERNEL_BACKEND
              value: "torch"
            - name: VLLM_USE_TRITON_GDC
              value: "0"
            - name: VLLM_ATTENTION_BACKEND
              value: "FLASH_ATTN"
            - name: VLLM_LORA_DISABLE_PDL
              value: "1"

          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
            - name: model-volume
              mountPath: /models
            {% if cfg.namespace == 'merlin' %}
            - name: merlin-ddx-pvc
              mountPath: /ft_models
            {% endif %}

      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
        - name: model-volume
          persistentVolumeClaim:
            claimName: model-volume
        {% if cfg.namespace == 'merlin' %}
        - name: merlin-ddx-pvc
          persistentVolumeClaim:
            claimName: merlin-ddx-ckpts-pvc
        {% endif %}

      imagePullSecrets:
        - name: private-registry-auth
      {% if cfg.Hardware.server_gpu is not string and cfg.Hardware.server_gpu is iterable %}
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: gpu
                    operator: In
                    values: [{{ cfg.Hardware.server_gpu | join(', ') }}]
          {#- List order = preference: when more than one listed type has a free
              node at scheduling time, the scheduler picks the earlier-listed one
              (descending weights). Still purely an eligible set otherwise -- if
              the preferred type is full, the pod lands on the next free one
              immediately instead of waiting, so "b200>h200>h100" never blocks. #}
          preferredDuringSchedulingIgnoredDuringExecution:
            {% for g in cfg.Hardware.server_gpu %}
            - weight: {{ 100 - loop.index0 * 30 }}
              preference:
                matchExpressions:
                  - key: gpu
                    operator: In
                    values: [{{ g }}]
            {% endfor %}
      {% else %}
      nodeSelector:
        gpu: {{ cfg.Hardware.server_gpu }}
        # kubernetes.io/hostname: cl-worker28
      {% endif %}
---


apiVersion: v1
kind: Service
metadata:
  name: vllm-server-{{ cfg.job_name }}
  namespace: {{ cfg.namespace }}
spec:
  selector:
    app: vllm-server-{{ cfg.job_name }}
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
"""
