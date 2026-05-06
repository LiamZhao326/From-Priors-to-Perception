import os
import glob
import subprocess
import shutil

# ==================================
POS_DIR = "path/to/your/positive/videos"
NEG_DIR = "path/to/your/negative/videos"
ALIGNED_NEG_DIR = "path/to/aligned_videos"
# ===========================================================

def get_duration(video_path):
    cmd_stream = [
        'ffprobe', '-v', 'error', 
        '-select_streams', 'v:0',
        '-show_entries', 'stream=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        video_path
    ]
    
    try:
        output = subprocess.check_output(cmd_stream).decode('utf-8').strip()
        if output and output.lower() != 'n/a':
            return float(output)
    except Exception:
        pass
    cmd_format = [
        'ffprobe', '-v', 'error', 
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        video_path
    ]
    return float(subprocess.check_output(cmd_format).decode('utf-8').strip())

def main():
    os.makedirs(ALIGNED_NEG_DIR, exist_ok=True)
    pos_files = glob.glob(os.path.join(POS_DIR, "*.mp4"))
    
    print(f"[*] 扫描到正样本文件夹中有 {len(pos_files)} 个视频文件准备处理。")
    
    for idx, pos_path in enumerate(pos_files, 1):
        filename = os.path.basename(pos_path)
        
        exact_match_path = os.path.join(NEG_DIR, filename)
        rev_match_path = os.path.join(NEG_DIR, filename.replace('.mp4', '_rev.mp4'))
        
        if os.path.exists(rev_match_path):
            out_path = os.path.join(ALIGNED_NEG_DIR, os.path.basename(rev_match_path))
            if os.path.exists(out_path):
                print(f"[{idx}/{len(pos_files)}] 跳过已复制: {os.path.basename(rev_match_path)}")
                continue
                
            print(f"[{idx}/{len(pos_files)}] ⚡ 极速复制 (时长一致): {os.path.basename(rev_match_path)}")
            shutil.copy2(rev_match_path, out_path)
            
        elif os.path.exists(exact_match_path):
            neg_path = exact_match_path
            out_path = os.path.join(ALIGNED_NEG_DIR, filename)
            
            if os.path.exists(out_path):
                print(f"[{idx}/{len(pos_files)}] 跳过已对齐: {filename}")
                continue
                
            pos_dur = get_duration(pos_path)
            neg_dur = get_duration(neg_path)
            ratio = pos_dur / neg_dur 
            
            print(f"[{idx}/{len(pos_files)}] ⚙️ 正在对齐 (FFmpeg): {filename} | 比例: {ratio:.3f}")
            
            cmd = [
                'ffmpeg', '-y', '-i', neg_path,
                '-filter:v', f'setpts={ratio}*PTS',
                '-r', '30', 
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                '-an', out_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        else:
            print(f"[{idx}/{len(pos_files)}] ❌ 警告: 找不到对应的负样本: {filename}")

if __name__ == "__main__":
    main()