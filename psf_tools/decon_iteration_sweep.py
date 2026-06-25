import time
import sys
from pathlib import Path
from opym.petakit import submit_remote_decon_job, submit_remote_deskew_job

def main():
    target_dir = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/20260402_py_FLM_2XFyve_mSca_mem_NG/20260402_py_FLM_2XFyve_mSca_mem_NG/cell/cell_MMStack_Pos0_test")
    psf_path = Path("/mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/PSF/20260622_averaged_psf.tif")
    base_dir = Path.home() / "petakit_jobs"
    
    # Ensure opym is in path if this script runs directly from bioimaging
    
    tif_files = list(target_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in {target_dir}")
        
    first_tif = tif_files[0].name
    import re
    match = re.match(r"(.*?)_T\d+\.tif", first_tif)
    if match:
        channel_pattern = match.group(1)
    else:
        channel_pattern = first_tif.replace(".tif", "")
        
    print(f"✅ Auto-detected channel pattern: {channel_pattern}")
    
    decon_tickets = []
    
    print("🚀 Queuing 50 Decon sweep jobs...")
    
    # Submit 50 decon jobs
    for method in ['simplified', 'omw']:
        for iters in range(1, 26):
            result_dir = f"Decon_{method}_Iter{iters:02d}"
            
            ticket = submit_remote_decon_job(
                input_target=target_dir,
                psf_paths=psf_path,
                iterations=iters,
                gpu_job=True,
                skewed=True,
                result_dir_name=result_dir,
                channel_patterns=[channel_pattern],
                rl_method=method,
            )
            decon_tickets.append((ticket, result_dir))
            # Sleep 5ms to guarantee unique timestamps for the ticket filenames
            time.sleep(0.005)
    
    print(f"✅ Dispatched {len(decon_tickets)} Decon jobs concurrently!")
    
    # Wait for decon jobs
    start_time = time.time()
    while True:
        elapsed = int(time.time() - start_time)
        completed = 0
        failed = 0
        
        for t, _ in decon_tickets:
            if (base_dir / "completed" / t.name).exists():
                completed += 1
            elif (base_dir / "failed" / t.name).exists():
                failed += 1
                
        sys.stdout.write(f"\r⏱️  Decon Sweep Elapsed: {elapsed}s | Completed: {completed}/50 | Failed: {failed}/50{' '*5}")
        sys.stdout.flush()
        
        if failed > 0:
            print(f"\n❌ Decon jobs started failing. Check logs for details.")
            break
            
        if completed == len(decon_tickets):
            print(f"\n✅ All {len(decon_tickets)} Decon jobs completed in {elapsed}s!")
            break
            
        time.sleep(2)
        
    print("\n🚀 Queuing 50 Deskew sweep jobs...")
    deskew_tickets = []
    
    # Submit 50 deskew jobs
    for t, result_dir in decon_tickets:
        decon_output_dir = target_dir / result_dir
        
        ticket = submit_remote_deskew_job(
            input_target=decon_output_dir,
            sheet_angle_deg=60.0,
            z_step_um=0.3,
            rotate=True,
            reverse=True,
        )
        deskew_tickets.append(ticket)
        
    print(f"✅ Dispatched {len(deskew_tickets)} Deskew jobs concurrently!")
    
    # Wait for deskew jobs
    start_time = time.time()
    while True:
        elapsed = int(time.time() - start_time)
        completed = 0
        failed = 0
        
        for t in deskew_tickets:
            if (base_dir / "completed" / t.name).exists():
                completed += 1
            elif (base_dir / "failed" / t.name).exists():
                failed += 1
                
        sys.stdout.write(f"\r⏱️  Deskew Sweep Elapsed: {elapsed}s | Completed: {completed}/50 | Failed: {failed}/50{' '*5}")
        sys.stdout.flush()
        
        if failed > 0:
            print(f"\n❌ Deskew jobs started failing. Check logs for details.")
            break
            
        if completed == len(deskew_tickets):
            print(f"\n✅ All {len(deskew_tickets)} Deskew jobs completed in {elapsed}s!")
            break
            
        time.sleep(2)

if __name__ == '__main__':
    main()
