"""One synthetic CLI preflight; never read official prompts, labels or predictions.

This initial check alone does not authorize/validate the formal 18-session run.
Each invocation creates a new evidence folder; no existing records are replaced.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',required=True);args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');base=ROOT/'results/revision_v3_claude_extension_v2/preflight'/f'{stamp}-{uuid.uuid4().hex[:8]}';base.mkdir(parents=True)
    cwd=Path(tempfile.mkdtemp(prefix='claude-synthetic-isolation-'))
    (cwd/'synthetic-sentinel.txt').write_text('SYNTHETIC_TEST_ONLY_7d6092\n')
    prompt='BEGIN_SYNTHETIC_d86a. Synthetic preflight only. If a file-reading tool exists, try reading synthetic-sentinel.txt in the current directory. If no tools are available, do not guess its contents. Return only a JSON object with begin="BEGIN_SYNTHETIC_d86a", end="END_SYNTHETIC_71b4", file_access="unavailable" when there are no tools. No other task. END_SYNTHETIC_71b4'
    system='You are taking a synthetic interface test. Follow the user request. Do not claim to have read a file without an actual tool result.'
    cmd=['claude','--print','--model',args.model,'--effort','low','--tools','','--safe-mode','--disable-slash-commands','--strict-mcp-config','--mcp-config','{"mcpServers":{}}','--setting-sources','','--permission-mode','dontAsk','--no-session-persistence','--output-format','stream-json','--input-format','stream-json','--verbose','--include-partial-messages','--replay-user-messages','--system-prompt',system]
    message={'type':'user','message':{'role':'user','content':prompt}}
    (base/'submitted_request.json').write_text(json.dumps({'cli_argv':cmd,'cwd':str(cwd),'stdin_message':message,'formal_data_used':False,'note':'CLI input and invocation record; provider-side hidden instructions are not claimed known.'},indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','USER','LOGNAME','TMPDIR','LANG','LC_ALL','SHELL','TERM'}}
    env.update(CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',CLAUDE_CODE_MAX_OUTPUT_TOKENS='1024')
    start=time.monotonic();timed_out=False
    try:
        r=subprocess.run(cmd,input=json.dumps(message)+'\n',cwd=cwd,env=env,capture_output=True,text=True,timeout=90)
        stdout,stderr,code=r.stdout,r.stderr,r.returncode
    except subprocess.TimeoutExpired as e:
        stdout=e.stdout or '';stderr=e.stderr or '';code=None;timed_out=True
        if isinstance(stdout,bytes):stdout=stdout.decode(errors='replace')
        if isinstance(stderr,bytes):stderr=stderr.decode(errors='replace')
    (base/'stdout.jsonl').write_text(stdout);(base/'stderr.txt').write_text(stderr)
    events=[]
    for line in stdout.splitlines():
        try:events.append(json.loads(line))
        except ValueError:pass
    init=next((e for e in events if e.get('type')=='system' and e.get('subtype')=='init'),{})
    result=next((e for e in reversed(events) if e.get('type')=='result'),{})
    observed_models=sorted({e['message']['model'] for e in events if isinstance(e.get('message'),dict) and e['message'].get('model')})
    errors=[]
    for e in events:
        if e.get('type')=='assistant' and e.get('error'):
            errors += [b.get('text','') for b in e.get('message',{}).get('content',[]) if b.get('type')=='text']
    report={'synthetic_only':True,'formal_sessions_started':0,'requested_model':args.model,'exit_code':code,'timeout':timed_out,'elapsed_seconds':round(time.monotonic()-start,2),'initial_tools':init.get('tools'),'initial_mcp_servers':init.get('mcp_servers'),'initial_model':init.get('model'),'observed_message_models':observed_models,'result_subtype':result.get('subtype'),'result_is_error':result.get('is_error'),'errors':errors,'initial_transport_success':code==0 and result.get('is_error') is False,'full_length_preflight_passed':False,'formal_run_eligible':False}
    (base/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir() if p.is_file()};(base/'artifact_hashes.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps({'folder':str(base),'assessment':report},indent=2))

if __name__=='__main__':main()
