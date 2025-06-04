# Llama 3 Vision + LoRA Fine-Tuning Example

**Meta-Llama 3.2 11B Vision-Instruct** 모델을 8-bit 양자화한 뒤, LoRA(PEFT)로 파인튜닝하는 예제 스크립트 모음입니다.

---

## 📋 목차

1. [배경 및 목표](#-배경-및-목표)  
2. [파일 구조](#-파일-구조)  
3. [사전 준비](#-사전-준비)  
4. [사용 방법](#-사용-방법)  
5. [파인튜닝 세부 옵션 설명](#-파인튜닝-세부-옵션-설명)  
6. [결과물](#-결과물)  
7. [라이선스](#-라이선스)  
8. [참고 자료](#-참고-자료)

---

## 📌 배경 및 목표

- **목표**  
  - A100 80GB GPU 한 장에서 Meta-Llama/Llama-3.2-11B-Vision-Instruct 모델을 8-bit 양자화로 메모리 사용량을 줄이고,  
    LoRA(PEFT)로 극히 일부 파라미터만 학습하여 특정 도메인(예: 건설 용어) 예제에 맞춰 파인튜닝하는 방법을 시연합니다.  
  - 로컬(Colab, 사내 서버 등) 환경에서  
    1. FP16 혼합 정밀도,  
    2. GPU 메모리 절약(8-bit 양자화),  
    3. LoRA 파인튜닝 사용법을 한눈에 알 수 있도록 예제 스크립트와 설명을 제공합니다.

---

## 📂 파일 구조
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

- **`.gitignore`**  
  불필요한 캐시 파일이나 대용량 체크포인트 폴더를 제외합니다.

- **`LICENSE`**  
  MIT License (선택한 라이선스) 또는 필요 시 다른 오픈소스 라이선스를 명시합니다.

- **`README.md`**  
  이 파일 자체입니다. 프로젝트 개요, 설치 및 사용 방법, 옵션 설명 등을 한눈에 보여줍니다.

- **`requirements.txt`**  
  파인튜닝에 필요한 Python 패키지와 최소 버전을 나열해 둡니다.

- **`scripts/run_lora_finetune.py`**  
  LoRA 기반 파인튜닝 스크립트 전체 코드입니다.  
  - 데이터 로드 → 전처리 → 모델 로드(8-bit 양자화) → LoRA 설정 → Trainer로 학습 → LoRA 어댑터 저장 순으로 구성되어 있습니다.

- **`data/placeholder.txt`**  
  실제 JSONL 데이터는 포함하지 않고, “여기에 파인튜닝 데이터(construction_terms_full.jsonl 등)를 넣으세요”라는 안내문만 남겨둡니다.

- **`README_IMAGES/`**  
  (선택) 프로젝트 구조나 파이프라인 다이어그램을 보여주는 이미지 파일을 두는 폴더입니다.

---

## ⚙️ 사전 준비

1. **Python 3.9+**
2. **A100 80GB GPU** (또는 8-bit 양자화/LoRA를 사용할 수 있는 충분한 VRAM을 갖춘 GPU)
3. **로컬 저장소 클론 및 패키지 설치**  
   ```bash
   git clone https://github.com/YourUser/llama3-vision-lora-finetuning.git
   cd llama3-vision-lora-finetuning
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. **HuggingFace 로그인 (필요 시)**  
   ```bash
   huggingface-cli login
   ```
   허깅페이스 허브에서 모델을 다운받기 위해 사전에 토큰을 설정해 두면 원활하게 진행됩니다.

---

## 🚀 사용 방법

1. **데이터 준비**  
   - `data/` 폴더에 JSONL 형태의 파인튜닝 데이터를 넣습니다.  
   - 파일 이름 예시: `construction_terms_full.jsonl`  
   - 각 줄이 `{ "instruction": "...", "input": "...", "output": "..." }` 형태이어야 합니다.  
   - **주의**: `output` 필드는 반드시 문자열로 래핑해야 하며, 숫자나 null이 섞이지 않도록 합니다.

2. **파인튜닝 실행**  
   터미널에서 다음 예시 명령어를 참고해 적절히 옵션을 바꿔 실행하세요:

   ```bash
   cd scripts
   python run_lora_finetune.py        --data_path ../data/construction_terms_full.jsonl        --output_dir ../lora-construction-terms-output        --per_device_train_batch_size 2        --gradient_accumulation_steps 2        --max_seq_length 512        --num_train_epochs 3
   ```

   - `--data_path`: JSONL 데이터 파일 경로  
   - `--output_dir`: 결과물을 저장할 디렉터리  
   - `--per_device_train_batch_size`: GPU당 배치 사이즈  
   - `--gradient_accumulation_steps`: 기울기 누적 스텝 수  
   - `--max_seq_length`: 입력 시퀀스 최대 토큰 길이 (예: 512)  
   - `--num_train_epochs`: 학습 epoch 수

3. **학습 결과 확인**  
   학습이 완료되면 `lora-construction-terms-output/` 폴더에  
   - `pytorch_model.bin` (8-bit 양자화된 모델 + LoRA 어댑터 가중치)  
   - `adapter_config.json`, `config.json`, `tokenizer.json` 등 파일이 저장됩니다.

   필요 시 아래 코드로 LoRA 어댑터 `state_dict`만 추출할 수 있습니다:

   ```python
   from peft import get_peft_model_state_dict
   import torch

   peft_state_dict = get_peft_model_state_dict(model)
   torch.save(peft_state_dict, "lora_adapter_state_dict.pt")
   ```

---

## 🔧 파인튜닝 세부 옵션 설명

1. **8-bit 양자화 (`load_in_8bit=True`)**  
   - A100 80GB에서도 11B 모델을 올려놓을 수 있도록 메모리를 크게 줄여줍니다.  
   - FP16 혼합 정밀도와 호환되며, 내부적으로 자동 변환됩니다.  

   ```python
   model = AutoModelForCausalLM.from_pretrained(
       args.model_name_or_path,
       load_in_8bit=True,
       torch_dtype=torch.float16,
       device_map="auto",
   )
   ```

2. **LoRA(PEFT) 설정**  
   ```python
   lora_config = LoraConfig(
       task_type=TaskType.CAUSAL_LM,
       inference_mode=False,
       r=args.lora_rank,            # 기본값: 8
       lora_alpha=args.lora_alpha,  # 기본값: 32
       lora_dropout=args.lora_dropout,  # 기본값: 0.05
       target_modules=args.target_modules.split(","),  # ["q_proj","k_proj","v_proj","o_proj"]
       bias="none",
   )
   ```

   - **r (rank)**: LoRA 저차원 임베딩 차원. 값이 크면 성능 향상이 가능하지만, 메모리/속도 트레이드오프가 있습니다.  
   - **α (alpha)**: Scaling factor. 보통 16~32 사이가 무난합니다.  
   - **dropout**: 오버피팅 방지를 위해 0.05 정도 설정합니다.  

3. **최대 시퀀스 길이 (`max_seq_length=512`)**  
   - 기본 모델(1024) 대비 절반으로 줄여서 학습 속도를 약 2배, 메모리 사용량을 1/4 이상 절감할 수 있습니다.  
   - 데이터 예시 중 600~800 토큰을 사용하는 항목이 있다면, 뒷부분이 잘릴 수 있으니 사전에 데이터 분포를 확인해주세요.

4. **배치 크기 & Gradient Accumulation**  
   - `per_device_train_batch_size=2`, `gradient_accumulation_steps=2` →  
     한 스텝당 총 배치 사이즈 4를 유지하면서,  
     내부 Overhead(GPU 메모리 이터레이션)가 조금 줄어듭니다.

   VRAM 여유가 충분하다면:  
   - `per_device_train_batch_size=4`, `gradient_accumulation_steps=1`  
   - `per_device_train_batch_size=8`, `gradient_accumulation_steps=1`  
   등으로 조정할 수 있습니다.

5. **데이터 전처리 병렬화 (`num_proc=4`)**  
   ```python
   tokenized_dataset = raw_dataset.map(
       preprocess_function,
       batched=True,
       num_proc=4,  # CPU 코어 4개로 병렬 처리
       remove_columns=raw_dataset.column_names,
   )
   ```  
   - CPU 4개(또는 사용 가능한 코어 수)로 토크나이저 함수를 병렬 호출하여,  
     대량의 JSONL 데이터를 빠르게 전처리할 수 있습니다.  
   - 학습 품질(모델 성능)에는 전혀 영향을 주지 않고, 데이터 준비 시간만 단축됩니다.

---

## 📊 결과물

학습이 끝난 후 `lora-construction-terms-output/` 안에는 다음과 같은 주요 파일이 생성됩니다:

```
lora-construction-terms-output/
├── adapter_config.json
├── config.json
├── generation_config.json
├── pytorch_model.bin       # 8bit 양자화 + LoRA 어댑터 가중치 포함
├── tokenizer.json
├── tokenizer_config.json
└── training_args.bin
```

- `pytorch_model.bin`: 양자화된 모델 가중치와 LoRA 어댑터 파라미터가 합쳐져 있는 파일입니다.  
- `adapter_config.json`: LoRA 어댑터에 관한 메타정보(모듈, rank, alpha, dropout 등).  
- 필요하다면, 순수 LoRA 어댑터 `state_dict`만 추출해서 별도 저장할 수 있습니다:

  ```python
  from peft import get_peft_model_state_dict
  import torch

  peft_state_dict = get_peft_model_state_dict(model)
  torch.save(peft_state_dict, "lora_adapter_state_dict.pt")
  ```

---

## 📝 라이선스

이 저장소는 라이선스 없음 (None) 으로 설정되어 있습니다.

---

## 📖 참고 자료

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)  
- [PEFT LoRA Documentation](https://github.com/huggingface/peft)  
- [Meta-Llama GitHub](https://github.com/facebookresearch/llama)  
