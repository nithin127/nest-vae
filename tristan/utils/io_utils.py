import re
import os
import glob

def get_latest_checkpoint(folder, pattern='*.ckpt'):
    all_checkpoints = glob.glob(os.path.join(folder, pattern))
    if not all_checkpoints:
        raise ValueError('There is no checkpoint in '
            '`{0}`'.format(folder))

    number_re = re.compile(r'.+{0}'.format(
        pattern.replace('*', '_(\d+)').replace('.', '\.')))
    numbers = map(lambda x: number_re.match(x).group(1), all_checkpoints)
    numbers = sorted(map(int, numbers))

    last_pattern = os.path.join(folder, pattern.replace(
        '*', '*_{0}'.format(numbers[-1])))
    last_checkpoint = glob.glob(last_pattern)[0]

    return last_checkpoint
