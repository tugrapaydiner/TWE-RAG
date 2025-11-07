# scripts/03_make_toy_temporal_qa.py
import json
from pathlib import Path

OUT = Path('data/toy_qa.jsonl')

# Each item: {question, answers: [{value, valid_from}], expect_latest: True}
EXAMPLES = [
  {
    'stem': 'Who is the CEO of ExampleCorp?',
    'answers': [
      {'value': 'Alice Newton', 'valid_from': '2019-01-01'},
      {'value': 'Bob Ortega',  'valid_from': '2022-06-01'},
      {'value': 'Cara Singh',  'valid_from': '2024-09-10'}
    ],
    'queries': [
      'Who is the current CEO of ExampleCorp?',
      'Who leads ExampleCorp as of now?',
      'Who is CEO of ExampleCorp?'
    ]
  },
]

if __name__ == '__main__':
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as w:
        for ex in EXAMPLES:
            for q in ex['queries']:
                w.write(json.dumps({'question': q, 'gold_latest': ex['answers'][-1]['value']}, ensure_ascii=False) + '\n')
    print(f'Wrote toy QA to {OUT}')
