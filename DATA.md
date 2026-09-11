# ThaiMouth Data Pipeline & Schema

## 1. Data Schema

ThaiMouth uses structured JSONL formats for both conversation and reasoning datasets.

### 1.1 Conversational Format
```json
{
  "type": "conversation",
  "messages": [
    {"role": "user", "content": "วันนี้เหนื่อยมากเลย"},
    {"role": "assistant", "content": "ไปเจออะไรมาหนักขนาดนั้น พักผ่อนบ้างนะ"}
  ]
}
```

### 1.2 Multi-Hop Reasoning Format
```json
{
  "type": "reasoning",
  "category": "comparison",
  "hops": 3,
  "question": "สมชาย สูงกว่า สมศักดิ์ และ สมศักดิ์ สูงกว่า วิชัย และ วิชัย สูงกว่า กานดา ถาม: ใครสูงที่สุด?",
  "reasoning": [
    "สมชาย สูงกว่า สมศักดิ์",
    "สมศักดิ์ สูงกว่า วิชัย",
    "วิชัย สูงกว่า กานดา",
    "ดังนั้น สมชาย สูงกว่าทุกคน"
  ],
  "answer": "สมชาย"
}
```

### 1.3 Plain Text Pretraining Format
```json
{
  "type": "text",
  "text": "ข้อความภาษาไทยทั่วไปสำหรับ pretraining..."
}
```

---

## 2. Synthetic Reasoning Tasks

The procedural generator in `thaimouth.data.synthetic_reasoning` creates:
- **Transitive Comparisons:** Multi-hop relation ordering ($A > B > C > D$).
- **Multi-step Arithmetic:** Sequential addition/subtraction word problems.
- **Logical Implication Chains:** If $A \to B$ and $B \to C$ then $A \implies C$.
- **Disjoint Splits:** Entity names for test sets (`THAI_NAMES_TEST`) are strictly disjoint from training names (`THAI_NAMES_TRAIN`) to prevent factual memorization.

---

## 3. Privacy, Licensing & Ethical Use

- **No Personally Identifiable Information (PII):** Do not ingest private chat logs or sensitive personal data into training corpora.
- **Licensing:** Ensure all external Thai corpora (e.g. Wikipedia, open Thai news datasets) adhere to permissive CC-BY, MIT, or Apache 2.0 licenses.
