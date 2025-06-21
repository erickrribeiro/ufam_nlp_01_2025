import sys, os
import sqlite3
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

sys.path.append(os.path.abspath("."))
from utils import execute_and_fetch_with_timeout

class ExecutionAccuracyMetric(BaseMetric):
    def __init__(self):
      self.name = "Execution Accuracy"
      self.is_higher_better = True
      self.threshold = 1.0

    def measure(self, test_case: LLMTestCase) -> float:
      additional_metadata = test_case.additional_metadata
      self.error = None

      db_id = additional_metadata["db_id"]
      self.db_path = os.path.join(
        "spider/spider_data/database", db_id, f"{db_id}.sqlite")

      try:
        # Executa o SQL gold (esperado)
        pred_success, pred_result, pred_error = execute_and_fetch_with_timeout(self.db_path, test_case.actual_output)
      except Exception as e:
        self.error = f"[Error] Failed executing predicted SQL: {e}"

      try:
        gold_success, gold_result, gold_error = execute_and_fetch_with_timeout(self.db_path, test_case.expected_output)
      except Exception as e:
        self.error = f"[Error] Failed executing gold SQL: {e}"

      # Normaliza resultados (ordem não importa)
      if pred_success and gold_success:
        pred_set = set(pred_result)
        gold_set = set(gold_result)
        self.score = int(pred_set == gold_set)
      else:
        self.score = 0

      self.success = self.score >= self.threshold

      return self.score

    def is_successful(self):
      return self.success

    @property
    def __name__(self):
      return "AccuracyMetric (NLP-TP-04)"
