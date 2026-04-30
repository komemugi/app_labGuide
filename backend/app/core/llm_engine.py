# backend/app/core/llm_engine.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import BitsAndBytesConfig

class LLMEngine:
  def __init__(self, model_id="LiquidAI/LFM2.5-1.2B-JP", quantization_8bit=False):
    """
    クラスの初期化時にモデルをロードする。
    """
    print(f"Loading model: {model_id}...")
    
    # トークナイザーのロード
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # パディングトークンの設定
    if self.tokenizer.pad_token_id is None:
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # モデルのロード
    if quantization_8bit:
      # 8bit量子化
      quantization_config = BitsAndBytesConfig(
                  load_in_8bit = True,
          )
      self.model = AutoModelForCausalLM.from_pretrained(
          model_id,
          device_map="auto",
          quantization_config=quantization_config, # 量子化設定反映
      )
    else:
      self.model = AutoModelForCausalLM.from_pretrained(
          model_id,
          device_map="auto",
          dtype=torch.bfloat16, # CPUでエラーが出たら torch.float32 に変更
      )
    
    # ストリーマーの準備（1文字ずつ表示するため）
    self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
    print("Model loaded successfully.")

  def chat(self, prompt, system_prompt="あなたは親切なAIキャラクターです。"):
    """
    return: self.tokenizer.decode(generated_tokens, skip_special_tokens=True), used_token_info
    """
    
    # 1. プロンプトの準備
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # 2. 一度プロンプトを「文字列」として取得する（デバッグでprintしやすいメリットも！）
    prompt_str = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False  # ここでまだ数字にしない
    )
    
    # 3. プロンプト文字列をトークナイズして Tensor に変換
    # これで inputs は確実に「input_ids」と「attention_mask」を持った辞書になる（エラー対策）
    inputs = self.tokenizer(
        prompt_str, 
        return_tensors="pt", 
        add_special_tokens=False 
    ).to(self.model.device)

    print("Generating...")
    
    # 4. 生成実行
    # inputs の中身（input_ids, attention_mask）を ** で展開して渡します
    outputs = self.model.generate(
        **inputs,  # input_ids と attention_mask が自動的に渡されます
        do_sample=True,
        temperature=0.8,
        max_new_tokens=200,
        streamer=self.streamer,
        pad_token_id=self.tokenizer.pad_token_id
    )

    # -----------------------------------------------------------------------------------------
    # トークン数の計算と表示：max_new_tokens を幾つに設定すべきかを確認するためのデバック用処理
    # -----------------------------------------------------------------------------------------
    input_len = len(inputs["input_ids"][0]) # 入力トークン数
    total_len = len(outputs[0])             # 総トークン数 (入力 + 出力)
    generated_len = total_len - input_len   # 生成されたトークン数

    used_token_info = f"{'='*30}\n [Token Usage Report]\n Input Tokens : {input_len}\n Output Tokens: {generated_len}\n Total Tokens : {total_len}\n{'='*30}"

    # 5. 結果のデコード
    # 入力の長さ分をカットして応答部分だけを取り出す
    generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
    return self.tokenizer.decode(generated_tokens, skip_special_tokens=True), used_token_info