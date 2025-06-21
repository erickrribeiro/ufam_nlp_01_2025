import sys, os
import pytest
import pandas as pd
from deepeval import assert_test, evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase

sys.path.append(os.path.abspath("."))
from accuracy_metrics import ExecutionAccuracyMetric

df = pd.read_csv("mistral-sql-lora-v3-predictions.csv")
dataset = EvaluationDataset()
for _, row in df.iterrows():
  test_case = LLMTestCase(
    input=row.question,
    actual_output=row.predicted_sql,
    expected_output=row.gold_sql,
    additional_metadata={"db_id": row.db_id}
  )
  dataset.add_test_case(test_case)
metric = ExecutionAccuracyMetric()

result = evaluate(
  dataset,
  metrics=[metric],
  async_config=AsyncConfig(run_async=False)
)