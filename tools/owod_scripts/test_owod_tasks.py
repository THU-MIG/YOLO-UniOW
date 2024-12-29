import os.path as osp
import glob
from pathlib import Path
import subprocess


owod_settings = {
    "MOWODB": {
        "task_list": [0, 20, 40, 60, 80],
        "test_image_set": "all_task_test"
    },
    "SOWODB": {
        "task_list": [0, 19, 40, 60, 80],
        "test_image_set": "test",
    },
    "nuOWODB": {
        "task_list": [0, 10, 17, 23],
        "test_image_set": "test",
    }
}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate OWOD tasks')
    parser.add_argument('dataset', type=str, choices=["MOWODB", "SOWODB", "nuOWODB"])
    parser.add_argument('config', type=str)
    parser.add_argument('--threshold', type=float, default=0.05, help='Confidence score threshold for known class')
    parser.add_argument('--save', action='store_true', help='Save evaluation results to eval_output.txt')
    return parser.parse_args()


def run_command(command):
    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end="")
    for line in process.stderr:
        print(line, end="")
    return_code = process.wait()
    return return_code


def eval_dataset(dataset, config, ckpt, args):
    image_set = owod_settings[dataset]['test_image_set']
    task_num = len(owod_settings[dataset]['task_list'])

    for task in range(1, task_num):
        stem = Path(config).stem
        work_dir = f'work_dirs/eval_{dataset}_{image_set}_task{task}'
        ckpt_path = osp.join(ckpt.format(stem, dataset.lower(), task), "best*.pth")
        checkpoint = sorted(glob.glob(ckpt_path))[-1]

        command = (f"DATASET={dataset} TASK={task} THRESHOLD={args.threshold} SAVE={args.save} "
                   f"./tools/dist_test.sh {config} {checkpoint} 8 --work-dir {work_dir}")

        if args.save:
            with open('eval_outputs.txt', 'a') as f:
                f.write(f"Eval {dataset} [Task {task}]\n")

        print("<EVAL>:", command)
        return_code = run_command(command)
        if return_code != 0:
            print(f"Task {task} failed with return code {return_code}")
            break


if __name__ == '__main__':
    args = parse_args()
    ckpt = "work_dirs/{}_{}_train_task{}"
    eval_dataset(args.dataset, args.config, ckpt, args)