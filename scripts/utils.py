import json
import os
import pandas as pd
import re
import random
import sqlite3
from enum import Enum
from multiprocessing import Process, Manager

import gdown
import numpy as np
import torch
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from tqdm.notebook import tqdm
from constants import CATEGORY_MAPPING

def download_spider_dataset():
    url = "https://drive.google.com/uc?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
    output = "spider_data.zip"
    gdown.download(url, output, quiet=False)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_tables(file_path):
  with open(file_path, "r") as f:
    return json.load(f)
  
def load_few_shot():
    return [
        {
            'db_id': 'department_management',
            'question': 'List the name, born state and age of the heads of departments ordered by age.',
            'query': 'SELECT name ,  born_state ,  age FROM head ORDER BY age',
        },
        {
            'db_id': 'farm',
            'question': 'What are the hosts of competitions whose theme is not "Aliens"?',
            'query': "SELECT Hosts FROM farm_competition WHERE Theme !=  'Aliens'",
        },
        {
            'db_id': 'student_assessment',
            'question': 'What are the ids of all students who have attended at least one course?',
            'query': 'SELECT student_id FROM student_course_attendance',
        },
    ]
  
def generate_schema_description(table):
    table_names = table["table_names_original"]
    columns = table["column_names_original"]
    foreign_keys = table["foreign_keys"]

    # Mapeia tabelas para suas colunas
    table_to_columns = {name: [] for name in table_names}

    for col in columns:
        table_idx, col_name = col
        if table_idx == -1:
            continue  # Ignora o '*'
        table_name = table_names[table_idx]
        table_to_columns[table_name].append(col_name)

    # Monta descrição das tabelas
    table_descriptions = []
    for table_name, cols in table_to_columns.items():
        col_list = " , ".join(cols)
        table_descriptions.append(f"| {table_name} : {col_list} |")

    # Monta descrição das foreign keys
    fk_descriptions = []
    for fk in foreign_keys:
        col1_idx, col2_idx = fk

        table1_idx, col1 = columns[col1_idx]
        table2_idx, col2 = columns[col2_idx]

        table1 = table_names[table1_idx]
        table2 = table_names[table2_idx]

        fk_descriptions.append(
            f"{table1}.{col1} = {table2}.{col2}"
        )

    # Juntar tudo
    description = "\n".join(table_descriptions)

    if fk_descriptions:
        description += "\nRelationships:\n" + " , ".join(fk_descriptions)

    return description

def extract_sql(response):
  # Procura dentro de blocos ```sql``` ... ```
  match = re.search(r"```sql(.*?)```", response, re.DOTALL | re.IGNORECASE)

  if match:
      sql = match.group(1).strip()
      return sql
  else:
      # Se não tiver bloco ```sql```, tenta pegar a primeira query SELECT na resposta
      fallback = re.search(r"(SELECT .*?;)", response, re.DOTALL | re.IGNORECASE)
      if fallback:
          return fallback.group(1).strip()
      else:
          return response
      
def _run_query(db_path, query, return_dict):
  try:
    conn = sqlite3.connect(db_path, timeout=5)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return_dict["success"] = True
    return_dict["results"] = results
    return_dict["error"] = None
  except Exception as e:
    return_dict["success"] = False
    return_dict["results"] = None
    return_dict["error"] = str(e)

def execute_and_fetch_with_timeout(db_path, query, timeout_sec=5):
  with Manager() as manager:
    return_dict = manager.dict()

    p = Process(target=_run_query, args=(db_path, query, return_dict))
    p.start()
    p.join(timeout_sec)

    if p.is_alive():
      p.terminate()
      p.join()
      return False, None, f"Timeout: consulta excedeu {timeout_sec} segundos"
    
    success = return_dict.get("success", False)
    results = return_dict.get("results", None)
    error = return_dict.get("error", "Unknown error" if not success else None)

    return success, results, error

def evaluate(results):
  execution_match = 0
  total_evaluated = 0

  for row in tqdm(results):
    db_id = row["db_id"]
    db_path = os.path.join("spider/spider_data/database", db_id, f"{db_id}.sqlite")

    if not os.path.exists(db_path):
      row["execution_accuracy"] = "DB_NOT_FOUND"
      raise Exception(f"DB not found: {db_path}")
      continue

    gold_query = row["gold_sql"]
    predicted_query = row["predicted_sql"]

    # Executa gold
    gold_success, gold_result, gold_error = execute_and_fetch_with_timeout(db_path, gold_query)

    # Executa predicted
    pred_success, pred_result, pred_error = execute_and_fetch_with_timeout(db_path, predicted_query)

    # Avaliação
    if gold_success and pred_success: # Todo: melhorar
      total_evaluated += 1

      pred_set = set(pred_result)
      gold_set = set(gold_result)

      if pred_set == gold_set:
        execution_match += 1
        row["execution_accuracy"] = "MATCH"
      else:
        row["execution_accuracy"] = "MISMATCH"
    else:
      error_msg = ""
      if not gold_success:
        error_msg += f"GOLD_FAIL: {gold_error} | "
      if not pred_success:
        error_msg += f"PRED_FAIL: {pred_error}"
      row["execution_accuracy"] = error_msg.strip()

  print(f"Total comparado (com sucesso em ambas): {total_evaluated}")
  print(f"✔️ Matches: {execution_match}")
  print(f"❌ Mismatches: {total_evaluated - execution_match}")


def map_task_to_category(task: MMLUTask) -> str:
  for category, tasks in CATEGORY_MAPPING.items():
    if task in tasks:
      return category
  return "Desconhecido"

def run_mmlu(model, mm_tasks, batch_size):
  benchmark = MMLU(
      tasks=mm_tasks, 
      n_shots=4
  )
  benchmark.evaluate(model=model, batch_size=batch_size)

  results = []
  for _, row in benchmark.task_scores.iterrows():  
    task = MMLUTask(row.Task)
    results.append({
        "task": task.value,
        "accuracy": row.Score,
        "category": map_task_to_category(task)
    })

  return pd.DataFrame(results)
