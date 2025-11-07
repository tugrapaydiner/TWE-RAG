#!/usr/bin/env python3
"""Generate sample corpus for TWE-RAG testing."""
import json
from pathlib import Path
from datetime import datetime, timedelta
import random

OUT_DIR = Path('data/raw_texts')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample topics and content templates
TOPICS = {
    'product_launch': [
        'announced the release of {product} version {version}',
        'unveiled {product}, a new solution for {use_case}',
        'introduced {product} with enhanced {feature} capabilities',
    ],
    'financial': [
        'reported quarterly revenue of ${amount}M, {trend} {percent}%',
        'stock price {movement} to ${price} per share',
        'announced {type} investment of ${amount}M in {area}',
    ],
    'partnership': [
        'formed strategic partnership with {partner} to {goal}',
        'acquired {company} for ${amount}M to expand {area}',
        'announced collaboration with {partner} on {project}',
    ],
    'research': [
        'published research on {topic} showing {result}',
        'breakthrough in {technology} promises to {benefit}',
        'developed new {technology} that improves {metric} by {percent}%',
    ],
    'events': [
        'hosted {event} conference with over {count} attendees',
        'announced keynote speaker for {event} will be {person}',
        'registration opens for annual {event} summit',
    ]
}

PRODUCTS = ['CloudSync', 'DataVault', 'AIAssist', 'SecureNet', 'DevTools', 'AnalyticsPro']
FEATURES = ['security', 'performance', 'scalability', 'reliability', 'automation']
USE_CASES = ['enterprise workflows', 'data analytics', 'cloud migration', 'DevOps']
PARTNERS = ['TechCorp', 'DataSystems Inc', 'CloudFirst', 'InnovateLabs']
TECHNOLOGIES = ['machine learning', 'blockchain', 'edge computing', 'quantum computing']
EVENTS = ['DevCon', 'CloudSummit', 'AIForum', 'TechFest']

def generate_document(doc_id: int, date: datetime) -> dict:
    """Generate a single document."""
    topic = random.choice(list(TOPICS.keys()))
    template = random.choice(TOPICS[topic])

    # Fill template
    content = template.format(
        product=random.choice(PRODUCTS),
        version=f'{random.randint(1,5)}.{random.randint(0,9)}',
        use_case=random.choice(USE_CASES),
        feature=random.choice(FEATURES),
        amount=random.randint(10, 500),
        percent=random.randint(5, 50),
        trend='up' if random.random() > 0.5 else 'down',
        movement='rose' if random.random() > 0.5 else 'fell',
        price=random.randint(50, 300),
        type='strategic' if random.random() > 0.5 else 'capital',
        area=random.choice(['R&D', 'infrastructure', 'talent', 'expansion']),
        partner=random.choice(PARTNERS),
        goal='enhance customer experience',
        company=random.choice(PARTNERS),
        project='next-generation platform',
        topic=random.choice(TECHNOLOGIES),
        result='significant improvements',
        technology=random.choice(TECHNOLOGIES),
        benefit='transform the industry',
        metric='efficiency',
        event=random.choice(EVENTS),
        count=random.randint(500, 5000),
        person='Dr. Jane Smith'
    )

    title = f"ExampleCorp News - {date.strftime('%B %Y')}"

    text = f"""{title}

{date.strftime('%B %d, %Y')} - ExampleCorp {content}. This development represents a significant milestone for the company and reflects its ongoing commitment to innovation and excellence.

Industry analysts have responded positively to the announcement, noting ExampleCorp's strong position in the market. The company continues to invest in cutting-edge technologies and strategic initiatives that drive long-term value.

"This is an exciting time for ExampleCorp and our customers," said a company spokesperson. "We remain focused on delivering exceptional solutions that meet the evolving needs of the market."

ExampleCorp has been a leader in enterprise technology for over {random.randint(10, 25)} years, serving thousands of customers worldwide. The company's innovative approach and customer-centric philosophy have made it a trusted partner for organizations of all sizes.

Looking ahead, ExampleCorp plans to continue expanding its product portfolio and strengthening its market presence. Additional updates will be shared in the coming months.
"""

    return {
        'id': f'doc_{doc_id:04d}',
        'text': text,
        'timestamp': date.date().isoformat()
    }

if __name__ == '__main__':
    # Generate 50+ documents spanning 2019-2024
    start_date = datetime(2019, 1, 1)
    docs = []

    for i in range(60):
        # Spread documents across ~6 years
        days_offset = i * 36  # Roughly every 36 days
        doc_date = start_date + timedelta(days=days_offset)
        doc = generate_document(i, doc_date)
        docs.append(doc)

    # Write as JSONL
    jsonl_path = Path('data/sample_corpus.jsonl')
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open('w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f'Generated {len(docs)} sample documents')
    print(f'Saved to {jsonl_path}')
    print(f'\nTo use this corpus, run:')
    print(f'  python scripts/00_prepare_corpus.py --jsonl {jsonl_path}')
