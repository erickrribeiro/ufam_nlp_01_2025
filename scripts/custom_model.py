import re
from typing import List
from deepeval.models.base_model import DeepEvalBaseLLM


class Mistral7B(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer,
        device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract_choice(self, output: str) -> str:
      match = re.search(r"\b([A-D])\b", output, re.IGNORECASE)
      if match:
        return match.group(1).upper()
      else:
        return "-"
      
    def load_model(self):
        self.model.to(self.device)
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids = model.generate(**model_inputs, 
                                       max_new_tokens=5,
                                       pad_token_id=self.tokenizer.eos_token_id)
        
        completion = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Remove o prompt da saída
        completion = completion[len(prompt):].strip()

        # Extrai a letra da resposta
        choice = self.extract_choice(completion)

        return choice

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        model = self.load_model()

        model_inputs = self.tokenizer(prompts, 
                                      return_tensors="pt", 
                                      padding=True, 
                                      truncation=True).to(self.device)
        generated_ids = model.generate(**model_inputs,
                                       max_new_tokens=5,
                                       pad_token_id=self.tokenizer.eos_token_id
                                       )
        
        predictions = []
        # Remove o prompt da saída e extrai a letra da resposta
        for i, output in enumerate(generated_ids):
            completion = self.tokenizer.decode(output, skip_special_tokens=True)
            completion = completion[len(prompts[i]):].strip()
            predictions.append(self.extract_choice(completion))
        return predictions

    def get_model_name(self) -> str:
        return self.model.name_or_path
