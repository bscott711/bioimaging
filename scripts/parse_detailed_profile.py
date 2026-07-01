import os
import sys
import re
import glob

def parse_html_report(job_dir):
    file0_path = os.path.join(job_dir, 'file0.html')
    if not os.path.exists(file0_path):
        return None
        
    with open(file0_path, 'r', encoding='utf-8') as f:
        html = f.read()
        
    # Fix: Allow optional text like " (MEX-file)" between </a> and </td>
    pattern = re.compile(r'<td class="td-linebottomrt"><a href="(file\d+\.html)">(.*?)</a>(.*?)</td.*?<td class="td-linebottomrt">(\d+)</td>.*?<td class="td-linebottomrt">([\d\.]+) s</td>.*?<td class="td-linebottomrt">([\d\.]+) s</td>', re.IGNORECASE | re.DOTALL)
    
    matches = pattern.findall(html)
    functions = []
    for match in matches:
        href, func_name_part1, func_name_part2, calls, total_time, self_time = match
        func_name = (func_name_part1 + func_name_part2).replace('&gt;', '>').strip()
        functions.append({
            'href': href,
            'name': func_name,
            'calls': int(calls),
            'total_time': float(total_time),
            'self_time': float(self_time),
            'lines': []
        })
        
    functions.sort(key=lambda x: x['self_time'], reverse=True)
    
    for func in functions:
        if func['self_time'] < 0.01:
            continue
            
        file_path = os.path.join(job_dir, func['href'])
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            f_html = f.read()
            
        line_pattern = re.compile(r'<tr>(.*?)</tr>', re.DOTALL)
        rows = line_pattern.findall(f_html)
        
        lines = []
        for row in rows:
            if '<a name="Line' not in row:
                continue
                
            cols = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)
            if len(cols) < 4:
                continue
                
            time_col = cols[0]
            calls_col = cols[1]
            line_col = cols[2]
            code_col = cols[3]
            
            t_str = re.sub(r'<[^>]+>', '', time_col).strip()
            t_match = re.search(r'([\d\.]+)', t_str)
            time_val = float(t_match.group(1)) if t_match else 0.0
            
            if time_val == 0.0:
                continue
                
            c_str = re.sub(r'<[^>]+>', '', calls_col).strip()
            c_match = re.search(r'(\d+)', c_str)
            calls_val = int(c_match.group(1)) if c_match else 0
            
            l_match = re.search(r'Line(\d+)', row)
            line_num = int(l_match.group(1)) if l_match else 0
                
            code_val = re.sub(r'<[^>]+>', '', code_col).strip()
            code_val = code_val.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
            
            lines.append({
                'time': time_val,
                'calls': calls_val,
                'line': line_num,
                'code': code_val
            })
            
        lines.sort(key=lambda x: x['time'], reverse=True)
        func['lines'] = lines[:5]
        
    return functions

def main():
    if len(sys.argv) > 1:
        target_job = sys.argv[1]
    else:
        base_dir = "/dev/shm/petakit_jobs/profiling/"
        job_dirs = glob.glob(os.path.join(base_dir, '*_html'))
        if not job_dirs:
            print("No profiling directories found.")
            sys.exit(1)
        target_job = max(job_dirs, key=lambda d: len(os.listdir(d)))
        
    print(f"Parsing detailed report for {os.path.basename(target_job)}...")
    functions = parse_html_report(target_job)
    if not functions:
        print("Failed to parse.")
        sys.exit(1)
        
    md = ["# MATLAB Profiling Deep Dive (New Single-Timepoint Run)\n"]
    overall_total_time = max(f['total_time'] for f in functions) if functions else 0.0
    md.append(f"**Overall Pipeline Run Time:** {overall_total_time:.3f} s\n")
    
    for func in functions:
        if func['self_time'] < 0.01:
            continue
            
        md.append(f"## `{func['name']}`")
        md.append(f"- **Total Time:** {func['total_time']} s")
        md.append(f"- **Self Time:** {func['self_time']} s")
        md.append(f"- **Calls:** {func['calls']}\n")
        
        if func['lines']:
            md.append("### Most Expensive Lines:")
            md.append("| Line | Code | Calls | Time (s) | % Time |")
            md.append("|---|---|---|---|---|")
            for l in func['lines']:
                pct = (l['time'] / func['total_time']) * 100 if func['total_time'] > 0 else 0
                code_snippet = l['code'][:40] + "..." if len(l['code']) > 40 else l['code']
                md.append(f"| {l['line']} | `{code_snippet}` | {l['calls']} | {l['time']:.3f} | {pct:.1f}% |")
        else:
            md.append("*No specific line-level hotspots reported or time spent entirely in built-ins.*")
        md.append("\n---\n")
        
    output_path = "/home/SDSMT.LOCAL/bscott/projects/bioimaging/scripts/profiling_deep_dive.md"
    with open(output_path, 'w') as f:
        f.write("\n".join(md))
        
    print(f"Wrote report to {output_path}")

if __name__ == "__main__":
    main()
