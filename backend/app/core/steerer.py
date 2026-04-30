import torch
import torch.nn.functional as F

class Steerer:
  def __init__(self, model):
    """
    モデルの参照を受け取り、初期化。
    """
    self.model = model
    self._hook_handles = [] # 複数層のフックを同時に管理できるようにリスト

  def apply_hook(self, layer_idx: int, vectors: list, strengths: list):
    """
    指定した層（layer_idx）にステアリングベクトルを注入するフックをしかける。
    複数回呼び出すことで、異なる層に異なるベクトルを同時に適用可能です。
    """
    # LLM指定レイヤーを取得（LLMブロック出力を受け取る層添字を指定） 
    target_layer = self.model.model.layers[layer_idx]

    # フック関数を内部(クロージャ)で定義することで、層ごとの vectors と strengths を個別に記憶させる
    def hook_fn(module, inputs, outputs):
        """
        Forward 時に自動で呼ばれる関数。
        """
        # transformersのDecoderLayerの出力は通常タプルで、最初の要素がhidden_states
        if isinstance(outputs, tuple):
            hidden_states = outputs
            modified_hidden_states = hidden_states.clone()
            
            for vec, strength in zip(vectors, strengths):
                # マイナス（減算）も許可するため、> 0.0 ではなく != 0.0 
                if vec is not None and strength != 0.0:
                    v = vec.to(device=modified_hidden_states.device, dtype=modified_hidden_states.dtype)
                    modified_hidden_states = modified_hidden_states + (v * strength) 
            
            # タプルを再構築して返す
            return (modified_hidden_states,) + outputs[1:]
            
        else:
            # 万が一タプルでなかった場合のフォールバック
            hidden_states = outputs
            modified_hidden_states = hidden_states.clone()
            
            for vec, strength in zip(vectors, strengths):
                # マイナス（減算）も許可するため、> 0.0 ではなく != 0.0
                if vec is not None and strength != 0.0:
                    v = vec.to(device=modified_hidden_states.device, dtype=modified_hidden_states.dtype)
                    modified_hidden_states = modified_hidden_states + (v * strength)
                    
            return modified_hidden_states

    # フックの登録とハンドル保存
    handle = target_layer.register_forward_hook(hook_fn)
    self._hook_handles.append(handle) # リストに保存
    print(f"Hook applied to layer {layer_idx} with strengths {strengths}")

  def remove_hook(self):
    """
    仕掛けたすべてのフックを取り外す。通常の推論に戻す際に呼び出す。
    """
    # リスト内のすべてのフックを安全に解除
    if self._hook_handles:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        print("All hooks removed. Model returned to normal state.")