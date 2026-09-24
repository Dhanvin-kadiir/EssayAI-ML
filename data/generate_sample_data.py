"""
generate_sample_data.py
Generates a realistic synthetic essay dataset (train.csv) that mirrors the
structure of the Kaggle AES 2.0 dataset (full_text + score 1-6).

Run:  python data/generate_sample_data.py
Output: data/train.csv  (~2,000 essays, stratified across scores 1-6)

Replace this with the real Kaggle data once downloaded:
  kaggle competitions download -c learning-agency-lab-automated-essay-scoring-2 -p data/
"""

import random
import csv
import os

random.seed(42)

# ── Essay templates by quality level ─────────────────────────────────────────
TOPICS = [
    "social media and teenagers",
    "climate change and individual responsibility",
    "the importance of education",
    "technology's impact on society",
    "immigration and cultural diversity",
    "the role of government in healthcare",
    "online learning vs traditional schooling",
    "the effects of video games on youth",
    "economic inequality in modern society",
    "space exploration funding priorities",
]

STRONG_INTROS = [
    "In contemporary society, {topic} has emerged as one of the most pressing and multifaceted issues demanding our collective attention.",
    "The debate surrounding {topic} has polarized communities, policymakers, and academics alike, necessitating a nuanced examination of the evidence.",
    "Few issues in modern discourse have generated as much controversy and scholarly interest as {topic}, a phenomenon with far-reaching implications.",
    "As our world becomes increasingly interconnected, the questions raised by {topic} grow ever more complex and consequential.",
]

WEAK_INTROS = [
    "{topic} is a very important thing that people talk about a lot.",
    "I think {topic} is interesting and I will write about it.",
    "Many people have different opinions about {topic}.",
    "In this essay I will discuss {topic} and why it matters.",
]

STRONG_BODY = [
    "Research conducted by {institution} demonstrates that {claim}. According to a {year} study published in the Journal of {field}, {stat}. This empirical evidence suggests that {inference}, a conclusion supported by {authority}.",
    "The economic implications of {topic} are profound. Data from the {org} indicates that {metric} has {change} by {pct}% over the past decade. Furthermore, {additional_point}, which underscores the urgency of addressing this issue through systematic policy reform.",
    "Critics argue that {counterargument}. However, this perspective overlooks the fundamental reality that {rebuttal}. When we examine the longitudinal data from {source}, it becomes clear that {conclusion}.",
]

WEAK_BODY = [
    "There are many good things and bad things about {topic}. Some people think it is good because it helps people. Other people think it is bad.",
    "I feel that {topic} is important. My friend told me that it can be good sometimes. But it can also be bad. So we need to think about it more.",
    "{topic} affects many people. It can change things a lot. People should care about it more and do something about it because it matters.",
]

STRONG_CONCLUSIONS = [
    "In conclusion, the evidence overwhelmingly suggests that a comprehensive, multi-stakeholder approach to {topic} is not merely advisable but essential. As we navigate the complexities of the twenty-first century, our collective response to this challenge will define the trajectory of future generations.",
    "Ultimately, addressing {topic} requires both individual commitment and systemic change. By synthesizing the empirical evidence, acknowledging diverse perspectives, and implementing evidence-based policies, societies can forge a more equitable and sustainable path forward.",
]

WEAK_CONCLUSIONS = [
    "So in conclusion, {topic} is important and we should think about it more.",
    "To sum up, I have written about {topic}. I hope people will learn more about it.",
    "In the end, {topic} is something that everyone should care about.",
]

FILLER_WORDS = ["very", "really", "quite", "so", "just", "basically", "stuff", "things", "a lot"]

def fill_template(template, topic):
    replacements = {
        "topic": topic,
        "institution": random.choice(["Harvard University", "MIT", "Stanford", "Oxford", "WHO", "UNICEF"]),
        "claim": "sustained engagement with this issue correlates with measurable improvements in societal wellbeing",
        "year": random.randint(2015, 2023),
        "field": random.choice(["Social Sciences", "Economics", "Psychology", "Public Health"]),
        "stat": f"approximately {random.randint(40, 85)}% of respondents reported significant impact",
        "inference": "policy intervention is both warranted and effective",
        "authority": random.choice(["leading economists", "behavioral scientists", "policy analysts"]),
        "org": random.choice(["World Bank", "OECD", "Pew Research Center", "Gallup Institute"]),
        "metric": random.choice(["public engagement", "economic output", "educational attainment", "social cohesion"]),
        "change": random.choice(["increased", "decreased", "shifted"]),
        "pct": random.randint(12, 47),
        "additional_point": "longitudinal cohort studies consistently replicate these findings",
        "source": random.choice(["peer-reviewed literature", "longitudinal surveys", "census data"]),
        "counterargument": "individual freedoms should supersede collective intervention",
        "rebuttal": "rights and responsibilities exist in dynamic tension, not isolation",
        "conclusion": "measured, evidence-informed policy yields superior outcomes",
    }
    try:
        return template.format(**replacements)
    except KeyError:
        return template


def generate_essay(score: int, topic: str) -> str:
    """Generate an essay with quality proportional to the score (1-6)."""
    paragraphs = []

    if score >= 5:
        intro = random.choice(STRONG_INTROS).format(topic=topic)
        body_count = random.randint(3, 4)
        bodies = [fill_template(random.choice(STRONG_BODY), topic) for _ in range(body_count)]
        conclusion = random.choice(STRONG_CONCLUSIONS).format(topic=topic)
    elif score == 4:
        intro = random.choice(STRONG_INTROS).format(topic=topic) if random.random() > 0.4 else random.choice(WEAK_INTROS).format(topic=topic)
        body_count = random.randint(2, 3)
        bodies = [fill_template(random.choice(STRONG_BODY if random.random() > 0.4 else WEAK_BODY), topic) for _ in range(body_count)]
        conclusion = random.choice(STRONG_CONCLUSIONS).format(topic=topic) if random.random() > 0.5 else random.choice(WEAK_CONCLUSIONS).format(topic=topic)
    elif score == 3:
        intro = random.choice(WEAK_INTROS).format(topic=topic) if random.random() > 0.3 else random.choice(STRONG_INTROS).format(topic=topic)
        body_count = random.randint(2, 3)
        bodies = [fill_template(random.choice(WEAK_BODY), topic) for _ in range(body_count)]
        conclusion = random.choice(WEAK_CONCLUSIONS).format(topic=topic)
    else:  # score 1-2
        intro = random.choice(WEAK_INTROS).format(topic=topic)
        body_count = random.randint(1, 2)
        bodies = [fill_template(random.choice(WEAK_BODY), topic) for _ in range(body_count)]
        # Add filler noise
        filler = " ".join(random.choices(FILLER_WORDS, k=random.randint(5, 15)))
        bodies.append(f"This is {filler} important.")
        conclusion = random.choice(WEAK_CONCLUSIONS).format(topic=topic)

    paragraphs = [intro] + bodies + [conclusion]
    return "\n\n".join(paragraphs)


def generate_dataset(n_per_score=350, output_path="train.csv"):
    """Generate balanced dataset with ~350 essays per score level (2,100 total)."""
    rows = []
    for score in range(1, 7):
        for _ in range(n_per_score):
            topic = random.choice(TOPICS)
            text  = generate_essay(score, topic)
            rows.append({"essay_id": len(rows), "full_text": text, "score": score})

    random.shuffle(rows)

    out = os.path.join(os.path.dirname(__file__), output_path)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["essay_id", "full_text", "score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DataGen] Saved {len(rows)} essays -> {out}")
    print(f"[DataGen] Score distribution: { {s: sum(1 for r in rows if r['score']==s) for s in range(1,7)} }")


if __name__ == "__main__":
    generate_dataset(n_per_score=350)
