import json
import re
import random
import numpy as np
import torch


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