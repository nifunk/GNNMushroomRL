import subprocess
import os
import time

processes = []
processes.append(subprocess.Popen("python examples/boxes_2D_dqn__fill_multidim.py --save --name stack_s2v_ensemble_unlimited_neg_rew --model s2v --use-cuda", shell=True))
time.sleep(10)
processes.append(subprocess.Popen("python examples/boxes_2D_dqn__fill_multidim.py --save --name stack_mha_ensemble_unlimited_neg_rew --model mha --use-cuda", shell=True))
time.sleep(10)
processes.append(subprocess.Popen("python examples/boxes_2D_dqn__fill_multidim.py --save --name stack_mha_full_ensemble_unlimited_neg_rew --model mha_full --use-cuda", shell=True))
time.sleep(10)
processes.append(subprocess.Popen("python examples/boxes_2D_dqn__fill_multidim.py --save --name stack_s2v_ensemble_unlimited_neg_rew --model s2v --use-cuda", shell=True))
time.sleep(10)
processes.append(subprocess.Popen("python examples/boxes_2D_dqn__fill_multidim.py --save --name stack_mha_ensemble_unlimited_neg_rew --model mha --use-cuda", shell=True))
time.sleep(10)
processes.append(subprocess.Popen("python examples/boxes_2D_dqn__fill_multidim.py --save --name stack_mha_full_ensemble_unlimited_neg_rew --model mha_full --use-cuda", shell=True))

while (len(processes)>0):
    removal_list = []
    for i in range(len(processes)):
        poll = processes[i].poll()
        if poll is None:
            time.sleep(60)
        else:
            removal_list.append(i)
            time.sleep(60)

    if (len(removal_list)!=0):
        correcting_counter = 0
        for i in range(len(removal_list)):
            print ("PROCESS " + str(removal_list[i]) + " FINISHED")
            processes.pop(removal_list[i]-correcting_counter)
            correcting_counter += 1
