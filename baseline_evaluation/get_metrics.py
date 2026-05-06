import os
import glob
import json
from collections import defaultdict

# ======= 1. 路径配置 =======
RESULTS_DIR = r"path/to/llm/results"
SUMMARY_FILE = os.path.join(RESULTS_DIR, "leaderboard_pairwise_summary.json")

def get_pair_id(video_path):
    dir_name = os.path.dirname(video_path)
    base_name = os.path.basename(video_path)
    
    clean_dir = dir_name.replace("_negative_aligned", "").replace("_negative", "")
    clean_base = base_name.replace(".mp4", "").replace("_rev", "")
    
    return f"{clean_dir}/{clean_base}"

def main():
    evaluated_files = glob.glob(os.path.join(RESULTS_DIR, "*_evaluated*.json"))
    
    if not evaluated_files:
        print("❌ 没有找到任何评测结果文件，请检查路径！")
        return

    leaderboard = []

    for file_path in evaluated_files:
        model_name = os.path.basename(file_path).replace("_evaluated.json", "").replace("_evaluated_flash.json", "")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            continue
            
        pairs_dict = defaultdict(list)
        total_score = 0
        
        for sample in data:
            total_score += sample["evaluation"]["score"]
            vid_path = sample.get("video", "")
            pair_id = get_pair_id(vid_path)
            pairs_dict[pair_id].append(sample)
            
        avg_score = total_score / len(data)
        
        valid_pairs_count = 0
        correct_pairs_count = 0
        
        for pair_id, matched_samples in pairs_dict.items():
            positives = [s for s in matched_samples if "_negative" not in s.get("video", "")]
            negatives = [s for s in matched_samples if "_negative" in s.get("video", "")]
            pair_count = min(len(positives), len(negatives))
            
            for i in range(pair_count):
                valid_pairs_count += 1
                
                acc1 = positives[i]["evaluation"]["accuracy"]
                acc2 = negatives[i]["evaluation"]["accuracy"]
                
                if acc1 == 1 and acc2 == 1:
                    correct_pairs_count += 1

        if valid_pairs_count == 0:
            print(f"⚠️ {model_name} 没有找到任何有效的 Pair，请检查数据。")
            continue

        pairwise_acc = correct_pairs_count / valid_pairs_count
        
        leaderboard.append({
            "model": model_name,
            "pairwise_accuracy": round(pairwise_acc * 100, 2),
            "score": round(avg_score, 3),
            "valid_pairs": valid_pairs_count,
            "total_samples": len(data)
        })

    leaderboard = sorted(leaderboard, key=lambda x: (x["pairwise_accuracy"], x["score"]), reverse=True)

    print("\n" + "="*75)
    print(" 🏆 PACC Benchmark Leaderboard (Pairwise Consistency - Robust Hash) ")
    print("="*75)
    print(f"{'Model Name':<20} | {'Pairwise Acc (%)':<18} | {'Avg Score':<10} | {'Pairs Checked'}")
    print("-" * 75)
    for entry in leaderboard:
        print(f"{entry['model']:<20} | {entry['pairwise_accuracy']:<18} | {entry['score']:<10} | {entry['valid_pairs']}")
    print("="*75 + "\n")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 最终成对评测结果已覆盖保存至: {SUMMARY_FILE}")

if __name__ == "__main__":
    main()