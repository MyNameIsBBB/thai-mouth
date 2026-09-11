"""
Synthetic Thai reasoning and conversation dataset generator.
Generates multi-hop reasoning tasks (comparisons, arithmetic, logic, reachability)
and natural Thai conversational dialogues.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


# Thai names pools for generating disjoint entity sets
THAI_NAMES_TRAIN = [
    "สมชาย", "สมศักดิ์", "วิชัย", "กานดา", "สุชาติ", "นภา", "เอกชัย", "อนันต์",
    "ประเสริฐ", "มนัส", "สุนทร", "วิรัช", "ปิยะ", "ธนา", "ศิริพร", "วรรณา",
    "ชลธิชา", "พิมพา", "รัตนา", "จินดา", "กิตติ", "มานะ", "ชูใจ", "ปิติ",
    "กมล", "สมบัติ", "พรทิพย์", "ดนัย", "ธีระ", "นเรศ", "บุญรอด", "ประสิทธิ์"
]

THAI_NAMES_TEST = [
    "วีระ", "ดวงใจ", "อรุณ", "ภาณุ", "เกรียงไกร", "นวลจันทร์", "สมร", "บุญมา",
    "ชัยวัฒน์", "ศุภชัย", "ยุพา", "ทิพย์", "ไพโรจน์", "รุ่งเรือง", "เกษม", "วันดี",
    "พิชิต", "อำนาจ", "ดวงดาว", "ทิวา", "ศศิธร", "กานต์", "สินชัย", "บัวบาน",
    "จันทร", "ฉัตรชัย", "ณัฐวุฒิ", "ฐิติ", "ดาริกา", "ทรงพล", "ธิดา", "นที"
]

CASUAL_CONVERSATIONS = [
    ("วันนี้เหนื่อยมากเลย", "ไปเจออะไรมาหนักขนาดนั้น พักผ่อนบ้างนะ"),
    ("กินข้าวเที่ยงยัง", "กินแล้ว เพิ่งกินกะเพราไข่ดาวมา อิ่มมาก"),
    ("อากาศร้อนจังวันนี้", "จริง ร้อนจนแทบละลาย เปิดแอร์ฉ่ำๆ เลย"),
    ("เสาร์อาทิตย์นี้ไปเที่ยวไหนดี", "ไปทะเลไหม หรือชอบแบบคาเฟ่ชิลๆ"),
    ("งานเยอะจนปวดหัวไปหมดแล้ว", "สู้ๆ นะ ค่อยๆ เคลียร์ทีละอย่าง เดี๋ยวก็เสร็จ"),
    ("เข้าใจละ สรุปคือต้องรันสคริปต์นี้ก่อนใช่ไหม", "ใช่เลย รันตัวนี้เสร็จแล้วค่อยเทรนโมเดลต่อ"),
    ("555 ตลกมากเลยเรื่องเมื่อกี้", "555 ขำจริง ไม่คิดว่าจะพีคขนาดนี้"),
    ("งั้นเอางี้ เดี๋ยวเราช่วยดูส่วนนี้ให้", "ขอบคุณมากเลย ช่วยได้เยอะมาก"),
    ("มันประมาณว่าถ้าเราตั้งค่าตรงนี้ ผลลัพธ์จะเปลี่ยนใช่ไหม", "ถูกต้องเลย ตัวแปรนี้คุม learning rate ตรงๆ"),
    ("เดี๋ยวนะ ถ้าแบบนั้นมันจะไม่ error เหรอ", "อ๋อ ลืมบอก เราใส่ try-except ดักไว้เรียบร้อยแล้ว"),
    ("โอเค แบบนี้เห็นภาพชัดขึ้นเยอะเลย", "ยินดีครับ มีตรงไหนสงสัยถามเพิ่มได้ตลอดนะ"),
]


def generate_comparison_sample(names: List[str], hops: int = 2) -> Dict:
    """
    Generates transitive comparison reasoning of arbitrary hop depth.
    Example (hops=3): A สูงกว่า B, B สูงกว่า C, C สูงกว่า D -> ใครสูงที่สุด
    """
    assert len(names) >= hops + 1, f"Need at least {hops+1} names, got {len(names)}"
    selected_names = random.sample(names, hops + 1)
    attribute = random.choice([
        ("สูงกว่า", "เตี้ยกว่า", "ใครสูงที่สุด", selected_names[0]),
        ("มีเงินมากกว่า", "มีเงินน้อยกว่า", "ใครมีเงินมากที่สุด", selected_names[0]),
        ("วิ่งเร็วกว่า", "วิ่งช้ากว่า", "ใครวิ่งเร็วที่สุด", selected_names[0]),
        ("แก่กว่า", "เด็กกว่า", "ใครอายุมากที่สุด", selected_names[0]),
        ("เรียนเก่งกว่า", "เรียนอ่อนกว่า", "ใครเรียนเก่งที่สุด", selected_names[0]),
        ("แข็งแรงกว่า", "อ่อนแอกว่า", "ใครแข็งแรงที่สุด", selected_names[0]),
    ])
    
    comp_word, opp_word, q_text, answer = attribute
    premises = []
    reasoning_steps = []
    
    for i in range(hops):
        p = f"{selected_names[i]} {comp_word} {selected_names[i+1]}"
        premises.append(p)
        reasoning_steps.append(p)
        
    reasoning_steps.append(f"ดังนั้น {answer} {comp_word}ทุกคน")
    question = " และ ".join(premises) + f" ถาม: {q_text}?"

    return {
        "type": "reasoning",
        "category": "comparison",
        "hops": hops,
        "question": question,
        "reasoning": reasoning_steps,
        "answer": answer
    }


def generate_arithmetic_sample(names: List[str], steps: int = 2) -> Dict:
    """
    Generates multi-step arithmetic word problem of arbitrary step depth.
    """
    name = random.choice(names)
    current = random.randint(30, 100)
    premises = [f"{name}มีเงิน {current} บาท"]
    reasoning = [f"เริ่มต้นมี {current} บาท"]
    
    for step_idx in range(steps):
        is_add = (step_idx % 2 == 0) or (current < 20)
        if is_add:
            amount = random.randint(10, 40)
            current += amount
            premises.append(f"ได้รับเพิ่ม {amount} บาท")
            reasoning.append(f"ได้เพิ่ม {amount} บาท รวมเป็น {current} บาท")
        else:
            amount = random.randint(5, max(6, min(current - 5, 30)))
            current -= amount
            premises.append(f"ซื้อของไป {amount} บาท")
            reasoning.append(f"จ่ายไป {amount} บาท เหลือเงิน {current} บาท")
        
    question = " ".join(premises) + f" ถาม: ตอนนี้{name}เหลือเงินกี่บาท?"
    answer = f"{current} บาท"

    return {
        "type": "reasoning",
        "category": "arithmetic",
        "hops": steps,
        "question": question,
        "reasoning": reasoning,
        "answer": answer
    }


def generate_logic_sample(hops: int = 2) -> Dict:
    """
    Generates logical implication chain: E1 -> E2 -> ... -> E_{hops+1}
    """
    event_pool = [
        "ฝนตกหนัก", "ถนนเปียกน้ำ", "การจราจรติดขัด", "รถประจำทางมาช้า", "เดินทางถึงที่ทำงานสาย",
        "หัวหน้าเรียกพบ", "ต้องอยู่ทำงานล่วงเวลา", "กลับบ้านดึก", "นอนหลับพักผ่อนไม่เพียงพอ",
        "ตื่นนอนตอนเช้าไม่ไหว", "ลืมนาฬิกาปลุก", "พลาดมื้ออาหารเช้า", "รู้สึกหิวและอ่อนเพลีย",
        "ทำงานผิดพลาดเล็กน้อย", "ต้องแก้ไขงานใหม่", "เหนื่อยล้าสะสม", "ขอลาพักร้อนผ่อนคลาย"
    ]
    
    # Pick a contiguous slice of hops+1 events
    if len(event_pool) < hops + 1:
        start_idx = 0
    else:
        start_idx = random.randint(0, len(event_pool) - (hops + 1))
    
    events = event_pool[start_idx : start_idx + hops + 1]
    while len(events) < hops + 1:
        events.append(f"เหตุการณ์ที่_{len(events)+1}")

    premises = []
    for i in range(hops):
        premises.append(f"ถ้า{events[i]}แล้วจะเกิด{events[i+1]}")
        
    start_event = events[0]
    final_event = events[-1]
    
    reasoning = [f"จากสมมติฐาน: {p}" for p in premises]
    reasoning.append(f"เนื่องจาก {start_event} จึงนำไปสู่ผลลัพธ์คือ {final_event}")
    
    question = " และ ".join(premises) + f" หากเริ่มต้นเกิดเหตุการณ์ '{start_event}' จะส่งผลให้เกิดเหตุการณ์สุดท้ายคืออะไร?"
    answer = final_event

    return {
        "type": "reasoning",
        "category": "logic",
        "hops": hops,
        "question": question,
        "reasoning": reasoning,
        "answer": answer
    }


def generate_dataset(
    num_samples: int,
    names: List[str],
    is_train: bool = True
) -> List[Dict]:
    """Generates a balanced mixture of reasoning and conversational Thai data."""
    samples = []
    
    # 1. Add conversational samples
    num_conv = int(num_samples * 0.35)
    for _ in range(num_conv):
        u, a = random.choice(CASUAL_CONVERSATIONS)
        samples.append({
            "type": "conversation",
            "messages": [
                {"role": "user", "content": u},
                {"role": "assistant", "content": a}
            ]
        })

    # 2. Add reasoning samples with varied hops (1 to 4)
    num_reason = num_samples - len(samples)
    for _ in range(num_reason):
        hops = random.randint(1, 4)
        cat = random.choice(["comp", "arith", "logic"])
        if cat == "comp":
            samples.append(generate_comparison_sample(names, hops=hops))
        elif cat == "arith":
            samples.append(generate_arithmetic_sample(names, steps=min(hops, 2)))
        else:
            samples.append(generate_logic_sample(hops=min(hops, 3)))

    random.shuffle(samples)
    return samples


def save_jsonl(data: List[Dict], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[+] Saved {len(data)} samples to {filepath}")


def create_synthetic_splits(
    out_dir: Path,
    num_train: int = 1500,
    num_val: int = 200,
    num_test: int = 300,
    seed: int = 42
):
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_data = generate_dataset(num_train, THAI_NAMES_TRAIN, is_train=True)
    val_data = generate_dataset(num_val, THAI_NAMES_TRAIN, is_train=False)
    test_data = generate_dataset(num_test, THAI_NAMES_TEST, is_train=False)

    save_jsonl(train_data, out_dir / "train.jsonl")
    save_jsonl(val_data, out_dir / "val.jsonl")
    save_jsonl(test_data, out_dir / "test.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Thai reasoning & conversation dataset.")
    parser.add_argument("--out_dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--num_train", type=int, default=2000, help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=300, help="Number of validation samples")
    parser.add_argument("--num_test", type=int, default=500, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    create_synthetic_splits(
        out_dir=Path(args.out_dir),
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
