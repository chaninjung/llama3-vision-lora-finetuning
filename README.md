# 📚 프로젝트 개요 / Project Overview

## 한글 설명
이 리포지토리는 Meta-Llama 3.2 11B Vision-Instruct 모델을 8-bit 양자화 후 LoRA(PEFT) 방식으로 파인튜닝하는 예제입니다.  
A100 80GB 한 장에서도 대용량 모델을 효율적으로 학습할 수 있도록 구성되어 있습니다.

## English Explanation
This repository demonstrates how to fine-tune the Meta-Llama 3.2 11B Vision-Instruct model using 8-bit quantization and LoRA (PEFT).  
It is designed to run efficiently even on a single A100 80GB GPU.

---

# 📂 파일 구조 / File Structure

## 한글 설명
```
llama3-vision-lora-finetuning/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── scripts/
│   └── run_lora_finetune.py
├── data/
│   └── placeholder.txt
└── README_IMAGES/
```
- `.gitignore`: 불필요한 캐시 및 체크포인트 제외  
- `LICENSE`: 프로젝트 라이선스 정보  
- `README.md`: 본 문서  
- `requirements.txt`: 필요한 패키지 목록  
- `scripts/`: 학습 스크립트 모음  
- `data/`: 실제 데이터는 포함되지 않으며, placeholder.txt 안내만 포함  
- `README_IMAGES/`: (선택) 다이어그램이나 스크린샷 등을 저장  

## English Explanation
```
llama3-vision-lora-finetuning/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── scripts/
│   └── run_lora_finetune.py
├── data/
│   └── placeholder.txt
└── README_IMAGES/
```
- `.gitignore`: Exclude unnecessary cache files and checkpoint folders  
- `LICENSE`: Project license information  
- `README.md`: This file  
- `requirements.txt`: List of required Python packages  
- `scripts/`: Fine-tuning scripts  
- `data/`: Placeholder only; actual JSONL data not included  
- `README_IMAGES/`: (Optional) Diagrams or screenshots  

---

# ⚙️ 사전 준비 / Prerequisites

## 한글 설명
1. Python 3.9 이상  
2. A100 80GB GPU (또는 충분한 VRAM을 갖춘 GPU)  
3. 리포지토리 클론 및 패키지 설치  
   ```bash
   git clone https://github.com/YourUser/llama3-vision-lora-finetuning.git
   cd llama3-vision-lora-finetuning
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```  
4. (선택) HuggingFace 로그인  
   ```bash
   huggingface-cli login
   ```  
   모델 다운로드용 토큰을 미리 설정해 두면 원활합니다.

## English Explanation
1. Python 3.9+  
2. A single A100 80GB GPU (or any GPU with enough VRAM for 8-bit + LoRA)  
3. Clone repo & install dependencies  
   ```bash
   git clone https://github.com/YourUser/llama3-vision-lora-finetuning.git
   cd llama3-vision-lora-finetuning
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```  
4. (Optional) HuggingFace login  
   ```bash
   huggingface-cli login
   ```  
   Pre-configure your token to avoid download issues.

---

# 🚀 사용 방법 / How to Run

## 한글 설명
1. `data/` 디렉터리에 JSONL 형태의 파인튜닝 데이터를 넣습니다.  
   - 예시 파일명: `construction_terms_full.jsonl`  
   - 각 줄이 `{ "instruction": "...", "input": "...", "output": "..." }` 형태여야 하며,  
     `output` 필드는 반드시 문자열(`""`)로 래핑되어 있어야 합니다.

2. 파인튜닝 실행 예시
   ```bash
   cd scripts
   python run_lora_finetune.py        --data_path ../data/construction_terms_full.jsonl        --output_dir ../lora-construction-terms-output        --per_device_train_batch_size 2        --gradient_accumulation_steps 2        --max_seq_length 512        --num_train_epochs 3
   ```
   - `--data_path`: JSONL 데이터 파일 경로  
   - `--output_dir`: 결과물을 저장할 디렉터리  
   - `--per_device_train_batch_size`: GPU당 배치 사이즈  
   - `--gradient_accumulation_steps`: 기울기 누적 스텝 수  
   - `--max_seq_length`: 입력 시퀀스 최대 길이 (예: 512)  
   - `--num_train_epochs`: 학습 epoch 수  

3. 학습이 완료되면 `lora-construction-terms-output/` 폴더에 다음 파일들이 생성됩니다:
   - `pytorch_model.bin`: 8-bit 양자화된 모델 가중치 + LoRA 어댑터  
   - `adapter_config.json`: LoRA 설정 정보  
   - `tokenizer.json`, `tokenizer_config.json`, `config.json`, `generation_config.json` 등

4. (선택) LoRA 어댑터 state_dict만 추출
   ```python
   from peft import get_peft_model_state_dict
   import torch

   peft_state_dict = get_peft_model_state_dict(model)
   torch.save(peft_state_dict, "lora_adapter_state_dict.pt")
   ```

## English Explanation
1. Place your fine-tuning data (JSONL) under the `data/` folder.  
   - Example filename: `construction_terms_full.jsonl`  
   - Each line must be of the form  
     `{ "instruction": "...", "input": "...", "output": "..." }`,  
     and `output` must always be wrapped as a string (`""`).

2. Run fine-tuning with, for example:
   ```bash
   cd scripts
   python run_lora_finetune.py        --data_path ../data/construction_terms_full.jsonl        --output_dir ../lora-construction-terms-output        --per_device_train_batch_size 2        --gradient_accumulation_steps 2        --max_seq_length 512        --num_train_epochs 3
   ```
   - `--data_path`: Path to the JSONL data file  
   - `--output_dir`: Directory to save outputs  
   - `--per_device_train_batch_size`: Batch size per GPU  
   - `--gradient_accumulation_steps`: Number of gradient accumulation steps  
   - `--max_seq_length`: Maximum input sequence length (e.g., 512)  
   - `--num_train_epochs`: Number of training epochs  

3. When training finishes, the `lora-construction-terms-output/` folder will contain:  
   - `pytorch_model.bin`: 8-bit quantized model weights + LoRA adapter  
   - `adapter_config.json`: LoRA configuration metadata  
   - `tokenizer.json`, `tokenizer_config.json`, `config.json`, `generation_config.json`, etc.

4. (Optional) Extract only the LoRA adapter state_dict:
   ```python
   from peft import get_peft_model_state_dict
   import torch

   peft_state_dict = get_peft_model_state_dict(model)
   torch.save(peft_state_dict, "lora_adapter_state_dict.pt")
   ```

---

# 🔧 파인튜닝 세부 옵션 설명 / Fine-Tuning Details

## 1. 8-bit 양자화 / 8-bit Quantization
- **한글**:  
  `load_in_8bit=True` 옵션으로 모델을 로드하면,  
  A100 80GB에서도 11B 모델을 메모리 한계 없이 올릴 수 있습니다.  
  FP16 혼합 정밀도와 호환되며, 내부적으로 자동 변환됩니다.

- **English**:  
  By setting `load_in_8bit=True`, you drastically reduce memory usage,  
  enabling you to load the 11B model even on a single A100 80GB.  
  It works seamlessly with FP16 mixed precision and is handled internally.

```python
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

---

## 2. LoRA(PEFT) 설정 / LoRA (PEFT) Configuration
- **한글**:  
  LoRA는 전체 모델을 학습하지 않고, 일부 저차원 파라미터만 학습하여 메모리·시간을 절약하는 기법입니다.  
  `r`(rank), `alpha`, `dropout`, `target_modules` 등을 설정합니다.

- **English**:  
  LoRA (PEFT) fine-tuning updates only a small subset of low-rank parameters rather than the full model,  
  saving memory and training time. You specify hyperparameters like `r`, `alpha`, `dropout`, and `target_modules`.

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=args.lora_rank,            # 기본: 8
    lora_alpha=args.lora_alpha,  # 기본: 32
    lora_dropout=args.lora_dropout,  # 기본: 0.05
    target_modules=args.target_modules.split(","),  # ["q_proj","k_proj","v_proj","o_proj"]
    bias="none",
)
```

- **r (Rank)**:  
  - 한글: LoRA의 저차원 임베딩 차원. 값이 크면 품질 개선 가능하지만, 메모리/속도 트레이드오프 발생.  
  - English: Low-rank embedding dimension for LoRA. Larger values can improve performance but cost more memory/compute.

- **α (Alpha)**:  
  - 한글: Scaling factor, 보통 16~32 범위가 무난.  
  - English: Scaling factor; 16–32 is a common choice.

- **Dropout**:  
  - 한글: 과적합 방지를 위해 0.05 정도 설정.  
  - English: Set around 0.05 to prevent overfitting.

---

## 3. 최대 시퀀스 길이 / Maximum Sequence Length
- **한글**:  
  `max_seq_length=512`로 설정하면,  
  attention 연산 비용(O(N²))을 1/4로 줄여 속도·메모리 효율을 크게 개선할 수 있습니다.  
  다만, 드물게 512 토큰을 넘는 예시가 있다면 뒤쪽이 잘릴 수도 있으니 주의가 필요합니다.

- **English**:  
  By halving from 1024→512, attention cost drops to (512/1024)² = 1/4.  
  This speeds up training and drastically reduces memory.  
  If some examples exceed 512 tokens, the tail may be truncated—check your data distribution first.

---

## 4. 배치 크기 & Gradient Accumulation / Batch Size & Gradient Accumulation
- **한글**:  
  `per_device_train_batch_size=2, gradient_accumulation_steps=2`로 설정하면,  
  한 스텝당 총 배치 크기가 4가 유지됩니다(2×2).  
  Overhead가 줄어들어 학습 속도가 약간 빨라지고, 메모리 사용 효율도 좋아집니다.

- **English**:  
  Setting `per_device_train_batch_size=2` and `gradient_accumulation_steps=2`  
  keeps the effective batch size at 4 (2×2) per step. Reduces overhead and slightly speeds up training.

```python
training_args = TrainingArguments(
    ...
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    ...
)
```

- **TIP**:  
  - 한글: VRAM이 넉넉하다면 `per_device_train_batch_size=4, gradient_accumulation_steps=1` 처럼 조정해도 좋습니다.  
  - English: If you have more VRAM, you can try `per_device_train_batch_size=4, gradient_accumulation_steps=1` to further reduce accumulation overhead.

---

## 5. 데이터 전처리 병렬화 / Data Preprocessing Parallelization
- **한글**:  
  HuggingFace `Dataset.map(..., num_proc=4)` 옵션을 주면,  
  CPU 4개를 동시에 사용해 **토크나이즈 및 레이블 생성**만 병렬 처리합니다.  
  학습 품질(모델 성능)에는 전혀 영향 없이 데이터 준비 시간을 단축할 수 있습니다.

- **English**:  
  By passing `num_proc=4` to `Dataset.map(...)`, you launch 4 CPU processes to parallelize tokenization & label creation only.  
  This does not affect model performance—only speeds up data loading.

```python
tokenized_dataset = raw_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,  # Adjust to your CPU core count
    remove_columns=raw_dataset.column_names,
)
```

---

# 📊 결과물 / Outputs

학습이 끝난 뒤 `lora-construction-terms-output/` 폴더에는:

```
lora-construction-terms-output/
├── adapter_config.json
├── config.json
├── generation_config.json
├── pytorch_model.bin       # 8bit 양자화된 모델 + LoRA 어댑터 가중치
├── tokenizer.json
├── tokenizer_config.json
└── training_args.bin
```

- **pytorch_model.bin**:  
  한글: 8-bit 양자화된 모델 가중치와 LoRA 어댑터 파라미터가 합쳐진 파일  
  English: Combined file of 8-bit quantized base weights and LoRA adapter parameters.

- **adapter_config.json**:  
  한글: LoRA 설정 상세 정보(모듈, rank, alpha, dropout 등)  
  English: Metadata about the LoRA adapter (modules, rank, alpha, dropout, etc.)

- **기타**: `tokenizer.json`, `tokenizer_config.json`, `config.json`, `generation_config.json`, `training_args.bin` 등  
  한글: 토크나이저 및 모델 설정 정보  
  English: Tokenizer and model configuration files.

---

# 📜 라이선스 / License

- **한글**:  
  본 프로젝트는 별도의 라이선스 없이 “None”으로 설정했습니다.  

- **English**:  
  This project currently has no license (“None”).  
