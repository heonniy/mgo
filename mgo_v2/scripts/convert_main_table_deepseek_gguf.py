"""Convert the local DeepSeek checkpoint once to an unquantized BF16 GGUF."""
import hashlib
import json
import subprocess
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
TOOLS = Path('/home/hwlee/mgo-tools/headline-r4')
MODEL = Path('/home/hwlee/model/DeepSeek-V2-Lite-Chat')
OUTPUT = TOOLS / 'DeepSeek-V2-Lite-Chat-BF16.gguf'
RECEIPT = PKG / 'experiments/main_table_2x2_20261008/DEEPSEEK_GGUF_RECEIPT.json'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    assert not OUTPUT.exists(), 'preserve existing GGUF; audit it before conversion'
    command = [str(TOOLS / 'base-env/bin/python'), str(TOOLS / 'llama.cpp/convert_hf_to_gguf.py'),
               str(MODEL), '--outfile', str(OUTPUT), '--outtype', 'bf16']
    try:
        subprocess.run(command, check=True)
    except BaseException:
        # Leave the partial output for investigation rather than treating it
        # as a completed model; the receipt is written only on success.
        raise
    assert OUTPUT.stat().st_size > 20 * 2**30, 'BF16 output is unexpectedly small'
    receipt = dict(status='PASS', model_path=str(MODEL), output_path=str(OUTPUT),
                   bytes=OUTPUT.stat().st_size, sha256=digest(OUTPUT), command=command,
                   converter_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                            cwd=TOOLS / 'llama.cpp', text=True).strip())
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'PASS {OUTPUT}')


if __name__ == '__main__':
    main()
